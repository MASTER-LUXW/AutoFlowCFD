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
    只作用于前 5 个欧拉变量（质量/动量/能量）。

    湍流量（k,omega）此前不参与这条滤波——原设计理由（"湍流输运方程走
    独立的单步显式更新，本就不经过这条容易积累混叠噪声的多级 RK 残差
    重算路径"）已被证明不完整，见 `filter_scalar_field` 文档：混叠是
    *空间*离散（坍缩坐标节点配置法对高阶模态的固有敏感性）导致的，跟
    "时间推进是单步显式还是多级 RK"无关——单步更新同样会在陡峭梯度区
    产生同类混叠振铃，只是不经过多级 RK residual 重算这条*额外*放大
    途径，不代表完全免疫。k/omega 现在改用 `filter_scalar_field` 单独
    滤波（见该函数完整推导），这里的说明同步更正。
    """
    U = U_flat.reshape(n_cells, n_sps, -1)
    if n_prism > 0:
        U[:n_prism, :, :5] = np.einsum("sj,cjv->csv", filter_prism, U[:n_prism, :, :5])
    if n_cells > n_prism:
        U[n_prism:, :, :5] = np.einsum("sj,cjv->csv", filter_tet, U[n_prism:, :, :5])
    return U.reshape(U_flat.shape)


def filter_scalar_field(phi: np.ndarray, n_prism: int, filter_prism, filter_tet) -> np.ndarray:
    """对湍流标量场（k 或 omega，形状 (n_cells, n_sps)）施加与平均流
    完全同一套模态滤波矩阵（真实 bug 修复，2026-09-12，cube_demo
    791,492 单元真实网格 P1 直连（无 Order Continuation）长程测试发现：
    全新、干净的 P1 启动在数十步内 omega 场大范围失控增长，8.6% 的单元
    omega 逼近 1e6 安全上限，对流残差量级达到 ~1e9~1e10（应为 O(1e2~
    1e4)），阻力系数 Cd 单调线性漂移到 4.9~16.6（正常方块应在 1~2.5），
    残差整体不收敛——决定性诊断定位到具体单元（如 153869）P1 多项式在
    单元内部相邻解点间出现数量级跳变（如同一单元 8 个解点里 2 个
    ~22 万、2 个恰好卡在 omega_inf 现实性下限 226.8、4 个恰好是某个
    壁面目标值 2252.6，物理上不可能这样分片跳跃），外插到面通量点后
    被上风格式放大成巨大的虚假对流残差，形成真实的复合增长——这正是
    `_filter_flat_U` 文档所述"坍缩坐标节点配置法对高阶模态混叠噪声
    天然敏感"的同一种病理，只是发生在 k/omega 而不是平均流守恒变量上。

    此前 k/omega 被排除在滤波之外的理由（"湍流方程走独立单步显式更新，
    不经过多级 RK 残差重算，因此不会积累混叠"）经真实数据证伪：混叠
    振铃是*空间*离散在陡峭梯度区的固有行为（Hesthaven & Warburton
    2008 §5.3；Boyd 2001 Ch.11 关于坍缩坐标/配置点法高阶混叠的标准
    结论，对任何标量场都成立，不区分它用几级龙格库塔推进），"单步
    显式"只是少了多级 RK residual 重新求值这一条*额外*的放大途径，
    从未意味着完全免疫——真实数据已经证明并非如此。

    与平均流滤波用的是完全相同的 `ops.filter_prism`/`ops.filter_tet`
    矩阵（对常数场恒等，P0 下 n_sps=1 时矩阵退化为 1x1 单位矩阵，
    天然是无操作，不需要额外的阶数判断分支）。

    Args:
        phi: (n_cells, n_sps) 标量场（k_field 或 omega_field）
        n_prism: 棱柱单元数（前 n_prism 个单元用 filter_prism）
        filter_prism, filter_tet: 与平均流共用的同一套模态滤波矩阵

    Returns:
        滤波后的标量场，形状不变
    """
    n_cells = phi.shape[0]
    out = phi.copy()
    if n_prism > 0:
        out[:n_prism] = np.einsum("sj,cj->cs", filter_prism, phi[:n_prism])
    if n_cells > n_prism:
        out[n_prism:] = np.einsum("sj,cj->cs", filter_tet, phi[n_prism:])
    return out


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
    # native 四面体（路径C，Part8 文档）：`filter_tet` 现在直接别名到
    # `filter_native_tet_padded`（见 fr/operators.py 模块文档"删除
    # collapsed 相关内容"一节，零填充行改用单位矩阵，见 native_tet_
    # padding.py::pad_native_tet_filter_matrix_to_global 文档——滤波器
    # 直接作用在 U 本身，不是残差贡献，填充行必须原样通过而不是被
    # 重置为 0）。
    filter_tet = ops.filter_tet

    def filter_func(U_flat: np.ndarray) -> np.ndarray:
        return _filter_flat_U(U_flat, n_cells, n_sps, n_prism, filter_prism, filter_tet)

    return filter_func
