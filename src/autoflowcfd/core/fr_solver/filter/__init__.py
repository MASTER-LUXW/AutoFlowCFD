"""
AutoFlowCFD V2.0 - 求解主循环模态滤波接入 (Tier-0 修复)

背景：见 fr/modal_filter.py 模块文档——坍缩坐标节点配置法对高阶模态的
混叠噪声天然敏感，重复空间微分（无粘体积散度、粘性"先求梯度再求散度"）
会把机器精度量级的噪声逐步放大，真实网格（棱柱、四面体均复现）验证过
在显式时间推进的几步内从噪声放大到 NaN，且与局部 CFL 步长大小无关
（把步长人为缩小到 1/100 只能推迟 2~3 步发生，不能消除，证实是混叠
驱动的不稳定性，不是 CFL 稳定裕度问题）。

本模块把 fr/operators.py::FROperators.filter_tet/filter_prism 接入
FRSolver.step() 的主循环，产出 `build_filter_func` 回调。真实复现：
只在每个完整时间步（SSP-RK3 三个
Shu-Osher stage 全部组合完成后）滤波一次并不够——混叠噪声在*中间*
stage（Stage1/Stage2 各自重新求值残差时）就已经放大到 NaN，等不到
最终组合完成；因此 `build_filter_func` 产出的回调要传给
TimeIntegrator.step()/step_dual_time()，由它们在*每个* stage 的正定性
投影之后立即调用（见 core/time_integration.py::_ssp_rk_stage_step），
是谱/DG 方法处理这类混叠失稳的标准做法（Hesthaven & Warburton 2008
§5.3；Boyd 2001 Ch.11），不改变已解析到的低阶物理精度（滤波器对常数场
恒等，见 fr/modal_filter.py 单元验证）。

## 文件分工（2026-09-24 拆包，原 1008 行）

    apply.py         滤波矩阵的实际施加（numba kernel）+ 恒等阵短路判据
    mode.py          `AFCFD_FILTER_MODE` 档位解析与后端支持矩阵
    scalar.py        k/omega 标量场滤波与 `AFCFD_FILTER_TURB_GATE` 门控
    bounds_conn.py   BJ 型邻居极值判据所需的分布式面邻接构造
    sensor_gate.py   传感器门控滤波回调（数组版 + 对象版）
    __init__.py      顶层入口 `build_filter_func` + 全量 re-export

本 `__init__.py` re-export 全部既有公开名**以及测试在用的私有名**
（`_SENSOR_MODE_SUPPORTED_BACKENDS` 等），所以全仓库
`from autoflowcfd.core.fr_solver.filter import ...` 一个字都不用改。
"""

from typing import Callable

import numpy as np

from .apply import (  # noqa: F401
    _filter_flat_U,
    _filter_flat_U_by_cell_type,
    _filter_leading_vars_inplace_kernel,
    _filter_scalar_kernel,
    _matrices_are_identity,
    build_filter_func_by_cell_type,
)
from .mode import (  # noqa: F401
    _SENSOR_MODE_SUPPORTED_BACKENDS,
    resolve_filter_mode,
)
from .scalar import (  # noqa: F401
    compute_turb_troubled_mask,
    filter_scalar_field,
    filter_scalar_field_gated,
    resolve_turb_filter_gate,
)
from .bounds_conn import (  # noqa: F401
    _DIST_FACE_STENCIL_WARNED,
    _warn_distributed_face_stencil,
    build_distributed_bounds_conn,
)
from .sensor_gate import (  # noqa: F401
    build_sensor_gated_filter_func,
    build_sensor_gated_filter_func_arrays,
)


def build_filter_func(solver) -> Callable[[np.ndarray], np.ndarray]:
    """构造供 TimeIntegrator.step()/step_dual_time() 在每个 RK stage 后
    调用的滤波回调，操作对象是展平形状 (n_cells*n_sps, n_vars) 的数组
    （TimeIntegrator 内部约定，与 fr_solver.py::step 里 U_flat 的展平
    方式一致）。

    Returns:
        滤波回调，**或 None**——两个滤波矩阵都是单位阵时（`AFCFD_FILTER_
        MODE=off`，或 mild 档取 sigma_top=1.0）返回 None，让调用方整个
        跳过这次调用而不是白乘一遍单位阵。`TimeIntegrator.step`/
        `step_dual_time` 对 `filter_func=None` 有显式支持（分布式路径在
        n_sps==1 时本来就传 None）。
    """
    mesh = solver.mesh
    ops = solver.ops
    n_cells = mesh.n_cells
    n_sps = mesh.n_sps_per_cell
    n_prism = mesh.n_prism_cells
    filter_prism = ops.filter_prism
    # native 四面体（路径C，Part8 文档）：`filter_tet` 现在直接别名到
    # `filter_native_tet_padded`（见 fr/operators.py 模块文档"删除
    # collapsed 相关内容"一节，零填充行改用单位矩阵，见 native_tet_
    # padding.py::pad_native_filter_matrix_to_global 文档——滤波器
    # 直接作用在 U 本身，不是残差贡献，填充行必须原样通过而不是被
    # 重置为 0）。
    filter_tet = ops.filter_tet

    # `AFCFD_FILTER_MODE=off` 下两个矩阵都是单位阵（见 fr/modal_filter.py
    # 的 `FILTER_MODE == "off"` 短路），继续每个 RK stage 乘一遍是纯浪费：
    # 79 万单元 P1 下 U 是 (791492,8,5) ≈ 253MB，一步三个 stage 要多读写
    # 约 1.5GB，全是内存带宽。`TimeIntegrator.step`/`step_dual_time` 对
    # `filter_func=None` 有显式支持（分布式路径在 n_sps==1 时本来就传
    # None），所以直接返回 None 让调用方跳过整个调用。
    #
    # 判据不看环境变量而是**直接检查矩阵是否为单位阵（机器精度容差）**：
    # 那样连 `AFCFD_FILTER_SIGMA_TOP=1.0`（mild 档取 sigma_top=1，矩阵是
    # 数值算出的 V@I@inv(V)、不逐位等于 eye）这种等价配置也一并短路，
    # 而且不依赖"环境变量与算子构造保持同步"这个隐含假设。
    if _matrices_are_identity(filter_prism, filter_tet):
        return None

    def filter_func(U_flat: np.ndarray) -> np.ndarray:
        return _filter_flat_U(U_flat, n_cells, n_sps, n_prism, filter_prism, filter_tet)

    return filter_func


__all__ = [
    "build_distributed_bounds_conn",
    "build_filter_func",
    "build_filter_func_by_cell_type",
    "build_sensor_gated_filter_func",
    "build_sensor_gated_filter_func_arrays",
    "compute_turb_troubled_mask",
    "filter_scalar_field",
    "filter_scalar_field_gated",
    "resolve_filter_mode",
    "resolve_turb_filter_gate",
]
