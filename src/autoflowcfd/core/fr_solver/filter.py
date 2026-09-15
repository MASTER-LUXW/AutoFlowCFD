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
from numba import njit, prange


@njit(cache=True, parallel=True)
def _filter_leading_vars_inplace_kernel(U, F, n_var_filter) -> None:
    """就地对每个 cell 施加模态滤波矩阵，只作用于前 `n_var_filter` 个变量：
    `U[c,s,v] <- sum_j F[s,j] * U[c,j,v]`（v < n_var_filter）。

    性能优化（2026-09-13，用户反馈"每步耗时过长、对 CPU 核数不敏感"后的
    真实剖析）：原实现是
    `U[:n_prism,:,:5] = np.einsum("sj,cjv->csv", filter_prism, U[:n_prism,:,:5])`。
    这个 einsum **没有传 `optimize=True`**，走的是 numpy 的通用逐元素求和
    路径（不是 BLAS gemm），既单线程、又要为切片赋值物化一份完整中间
    数组；`U[...,:5]` 在 SST（n_vars=7）下还是跨步长切片，赋值两端各来
    一次跨步拷贝。79 万单元 P1 实测每次调用约 0.55s，而 SSP-RK3 每步要
    在 3 个 stage 各调一次（约 1.6s/步）。

    改成按 cell `prange` 的就地 kernel 后：无任何大中间数组、不受
    `:5` 跨步切片影响（直接按索引读写原数组）、完全并行。对 j 的求和
    顺序与非优化 einsum 的 j 递增顺序一致，结果逐位相同（等价性回归见
    tests/unit/test_perf_fusion_kernels.py）。
    """
    C, S, _ = U.shape
    for c in prange(C):
        tmp = np.empty((S, n_var_filter))
        for s in range(S):
            for v in range(n_var_filter):
                acc = 0.0
                for j in range(S):
                    acc += F[s, j] * U[c, j, v]
                tmp[s, v] = acc
        for s in range(S):
            for v in range(n_var_filter):
                U[c, s, v] = tmp[s, v]


@njit(cache=True, parallel=True)
def _filter_scalar_kernel(phi, F, out) -> None:
    """`out[c,s] = sum_j F[s,j] * phi[c,j]`（标量场版本，见
    `_filter_leading_vars_inplace_kernel` 同一处性能优化说明）。

    与原 `np.einsum("sj,cj->cs", ...)` 的等价性（实测）：n_sps=1 时逐位
    相同，n_sps=4/8/27 时为机器精度（1.4e-16~4.8e-16 相对误差）——2D
    einsum 的累加方式与这里的顺序循环略有不同。5 变量主流滤波
    （`_filter_leading_vars_inplace_kernel`）实测在全部形状下**逐位相同**。
    """
    C, S = phi.shape
    for c in prange(C):
        for s in range(S):
            acc = 0.0
            for j in range(S):
                acc += F[s, j] * phi[c, j]
            out[c, s] = acc


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
    # 就地并行 kernel（性能优化 2026-09-13，见
    # `_filter_leading_vars_inplace_kernel` 文档：原 einsum 未开 optimize、
    # 走单线程通用求和路径，且 `:5` 跨步切片赋值要额外两次大拷贝）。
    if n_prism > 0:
        _filter_leading_vars_inplace_kernel(U[:n_prism], np.ascontiguousarray(filter_prism), 5)
    if n_cells > n_prism:
        _filter_leading_vars_inplace_kernel(U[n_prism:], np.ascontiguousarray(filter_tet), 5)
    return U.reshape(U_flat.shape)


def _filter_flat_U_by_cell_type(U_flat: np.ndarray, n_cells: int, n_sps: int,
                                cell_is_prism: np.ndarray,
                                filter_prism, filter_tet) -> np.ndarray:
    """与 `_filter_flat_U` 完全等价，但按**单元类型掩码**分派而不是按
    "棱柱在前"的 `n_prism` 分界切分。

    为什么需要它（2026-09-14）：CPU 分布式路径的 local 单元按
    `partition.local_cells` 自身顺序排列，棱柱与四面体是**交错**的，
    没有"前 n_prism 个是棱柱"这个性质，`_filter_flat_U` 直接用不了。
    而模态滤波是纯逐单元操作（无邻居耦合），只要每个单元用对自己的
    滤波矩阵，结果就与单机逐位相同——这一点由
    `tests/unit/test_distributed_solver_main_init.py::
    TestDistributedStepMatchesSingleMachine` 的"推进一步后状态必须与
    单机一致"判据保证。

    分成两次按掩码取子集调用（而不是逐单元循环）：`U[mask]` 会产生副本，
    滤波后必须写回——这里显式写回，不依赖视图语义。
    """
    U = U_flat.reshape(n_cells, n_sps, -1)
    mask = np.asarray(cell_is_prism, dtype=bool)
    if mask.shape[0] != n_cells:
        raise ValueError(
            f"cell_is_prism 长度 {mask.shape[0]} 与 n_cells {n_cells} 不符")
    for sel, mat in ((mask, filter_prism), (~mask, filter_tet)):
        if not np.any(sel):
            continue
        sub = np.ascontiguousarray(U[sel])
        _filter_leading_vars_inplace_kernel(sub, np.ascontiguousarray(mat), 5)
        U[sel] = sub
    return U.reshape(U_flat.shape)


def build_filter_func_by_cell_type(ops, n_cells: int, n_sps: int,
                                   cell_is_prism: np.ndarray):
    """`build_filter_func` 的掩码版，供 CPU 分布式路径使用。

    分布式路径此前**完全没有**施加模态滤波（单机每个 RK stage 都施加，
    多 GPU 分布式也有 `filter_func_gpu`）——CPU 分布式是唯一的缺口，
    2026-09-14 补齐。模态滤波是 P>=1 的稳定性机制（坍缩坐标/配置点法
    对高阶模态混叠天然敏感，见 `_filter_flat_U` 与 `filter_scalar_field`
    的完整推导与真实复现记录），缺它意味着这条路径少了一层已经被真实
    算例证明必要的保护。
    """
    filter_prism = ops.filter_prism
    filter_tet = ops.filter_tet

    def filter_func(U_flat: np.ndarray) -> np.ndarray:
        return _filter_flat_U_by_cell_type(
            U_flat, n_cells, n_sps, cell_is_prism, filter_prism, filter_tet)

    return filter_func


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
    # 同上并行 kernel（机器精度等价，见 `_filter_scalar_kernel` 文档）
    if n_prism > 0:
        _filter_scalar_kernel(np.ascontiguousarray(phi[:n_prism]),
                              np.ascontiguousarray(filter_prism), out[:n_prism])
    if n_cells > n_prism:
        _filter_scalar_kernel(np.ascontiguousarray(phi[n_prism:]),
                              np.ascontiguousarray(filter_tet), out[n_prism:])
    return out


#: 目前实现了 `sensor` 档的后端。`legacy`/`off`/`mild` 三档**不需要**
#: 出现在这里——它们是在算子构造期改 `ops.filter_prism`/`filter_tet` 本身
#: （见 fr/modal_filter.py 的模块级常量），因此对全部后端自动生效；只有
#: `sensor` 需要在推进循环里逐单元门控，必须逐后端接线。
_SENSOR_MODE_SUPPORTED_BACKENDS = ("cpu-single",)


def resolve_filter_mode(backend: str) -> str:
    """读取 `AFCFD_FILTER_MODE` 并校验当前后端是否支持它。

    存在的理由：`sensor` 档只在单机 CPU 推进循环（`fr_solver/step.py`）里
    接线过。如果 CPU MPI 分布式 / 单 GPU / 多 GPU 分布式路径只是"读不到
    这个分支所以按 legacy 跑"，同一个环境变量在不同后端就意味着不同的
    数值方案，而且**没有任何提示**——那正是本项目一贯不接受的静默行为
    （同一原则见 gpu_time_integration.py 把"只用第一个 SP"改成显式校验）。
    所以这里直接报错，而不是 warning 后继续。

    Args:
        backend: 调用方后端标识，取 `_SENSOR_MODE_SUPPORTED_BACKENDS` 里的
            值或任意其它字符串（如 "cpu-mpi"/"gpu-single"/"gpu-mpi"）

    Returns:
        规范化（小写）后的模式名。

    Raises:
        NotImplementedError: 请求了 `sensor` 但该后端尚未接线。
    """
    import os
    mode = os.environ.get("AFCFD_FILTER_MODE", "legacy").lower()
    if mode == "sensor" and backend not in _SENSOR_MODE_SUPPORTED_BACKENDS:
        raise NotImplementedError(
            f"AFCFD_FILTER_MODE=sensor 尚未在后端 '{backend}' 上接线"
            f"（已接线：{', '.join(_SENSOR_MODE_SUPPORTED_BACKENDS)}）。"
            f"传感器门控需要在推进循环里逐单元求 Persson-Peraire 指示器，"
            f"不是算子构造期就能定下来的，必须逐后端实现。"
            f"legacy/off/mild 三档对全部后端都有效，可先用它们；"
            f"若确实需要在该后端用 sensor 档，请先补齐接线而不是忽略本错误。")
    return mode


def build_sensor_gated_filter_func(solver) -> Callable[[np.ndarray], np.ndarray]:
    """按 Persson-Peraire 传感器**逐单元门控**的模态滤波（2026-09-15）。

    为什么需要它：全局每-stage 施加的滤波器在本项目的工作阶数上会精确
    抹掉一整阶（order=1 -> 只剩常数，order=2 -> 只剩双线性，见
    `fr/modal_filter.py` 顶部的实测记录）。而且因为它每个 RK stage 都施加，
    任何 sigma<1 都会随步数复合累积，单纯调小 alpha 只是把清零推迟——
    正确的方向是**只在确实需要的单元上施加**。

    本项目已经有现成的欠分辨传感器：`fr_operators/artificial_viscosity.py::
    compute_persson_peraire_artificial_viscosity` 返回的 epsilon 场在解
    光滑（高阶模态能量占比低）的单元上恒为 0，只在真正欠分辨/振荡的单元
    上非零——这正是"哪些单元需要滤波"的判据，不需要另造一个传感器。

    施加方式：epsilon>0 的单元走完整的 legacy 滤波矩阵（该单元确实有
    需要压制的混叠内容），epsilon==0 的单元完全不动（保留全部已解析的
    多项式内容）。这样光滑区不再损失阶数，振荡区仍有与此前相同强度的
    保护。

    注意本函数**不改变滤波矩阵本身**（仍用 ops.filter_prism/filter_tet），
    只改变"对哪些单元施加"。`AFCFD_FILTER_MODE=off/mild` 那两档改的是
    矩阵，两个维度可以独立组合，便于受控 A/B。
    """
    mesh = solver.mesh
    ops = solver.ops
    n_sps = mesh.n_sps_per_cell
    n_prism = mesh.n_prism_cells
    filter_prism = ops.filter_prism
    filter_tet = ops.filter_tet

    def filter_func(U_flat: np.ndarray) -> np.ndarray:
        from autoflowcfd.core.fr_operators.artificial_viscosity import (
            compute_persson_peraire_artificial_viscosity,
        )
        n_cells = mesh.n_cells
        U = U_flat.reshape(n_cells, n_sps, -1)
        # 传感器要在**当前** stage 的解上求值（不是步首的解）——U_flat
        # 就是当前 stage 的解，临时装进 state 以复用现成的传感器入口。
        saved = solver.state.U
        solver.state.U = U
        try:
            eps = compute_persson_peraire_artificial_viscosity(
                solver, alpha_av=getattr(solver, "artificial_viscosity_alpha", 1.0))
        finally:
            solver.state.U = saved
        troubled = np.any(eps > 0.0, axis=1)
        if not np.any(troubled):
            return U_flat
        idx = np.flatnonzero(troubled)
        pr = idx[idx < n_prism]
        te = idx[idx >= n_prism]
        for sel, mat in ((pr, filter_prism), (te, filter_tet)):
            if sel.size == 0:
                continue
            sub = np.ascontiguousarray(U[sel])
            _filter_leading_vars_inplace_kernel(sub, np.ascontiguousarray(mat), 5)
            U[sel] = sub
        return U.reshape(U_flat.shape)

    return filter_func


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
