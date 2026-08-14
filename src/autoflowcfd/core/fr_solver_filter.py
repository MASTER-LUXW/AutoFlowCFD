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
"""

from typing import Callable

import numpy as np


def _filter_flat_U(U_flat: np.ndarray, n_cells: int, n_sps: int, n_prism: int, filter_prism, filter_tet) -> np.ndarray:
    """对展平成 (n_cells*n_sps, n_vars) 的守恒变量数组施加模态滤波，
    只作用于前 5 个欧拉变量（质量/动量/能量）——湍流量（k,omega，若
    存在）不参与滤波，湍流输运方程走独立的单步显式更新（见
    fr_solver.py::step 文档"算子分裂"说明），本就不经过这条容易积累
    混叠噪声的多级 RK 残差重算路径。TimeIntegrator 内部统一用这种
    展平形状，见 fr_solver.py::step 的 U_flat 构造。
    """
    U = U_flat.reshape(n_cells, n_sps, -1)
    if n_prism > 0:
        U[:n_prism, :, :5] = np.einsum("sj,cjv->csv", filter_prism, U[:n_prism, :, :5])
    if n_cells > n_prism:
        U[n_prism:, :, :5] = np.einsum("sj,cjv->csv", filter_tet, U[n_prism:, :, :5])
    return U.reshape(U_flat.shape)


def build_filter_func(solver) -> Callable[[np.ndarray], np.ndarray]:
    """构造供 TimeIntegrator.step()/step_dual_time() 在每个 RK stage 后
    调用的滤波回调，操作对象是展平形状 (n_cells*n_sps, n_vars) 的数组
    （TimeIntegrator 内部约定，与 fr_solver.py::step 里 U_flat 的展平
    方式一致）。
    """
    mesh = solver.mesh
    ops = solver.ops
    n_cells = mesh.n_cells
    n_sps = mesh.n_sps_per_cell
    n_prism = mesh.n_prism_cells
    filter_prism = ops.filter_prism
    filter_tet = ops.filter_tet

    def filter_func(U_flat: np.ndarray) -> np.ndarray:
        return _filter_flat_U(U_flat, n_cells, n_sps, n_prism, filter_prism, filter_tet)

    return filter_func
