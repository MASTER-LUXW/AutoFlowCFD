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

import os
from typing import Callable

import numpy as np
from loguru import logger
from numba import njit, prange


def _matrices_are_identity(*mats) -> bool:
    """给定的滤波矩阵是否都是**机器精度意义上的**单位阵。

    用容差而不是 `array_equal`：`AFCFD_FILTER_MODE=off` 的矩阵是
    `np.eye` 直接返回、逐位相等，但 `mild` 档取 `sigma_top=1.0` 时矩阵是
    数值算出的 `V @ diag(1) @ inv(V)`——数学上是单位阵、浮点上偏差
    ~1e-16。那种配置同样是无操作，同样应当被短路。

    容差取 1e-12：矩阵元素是 O(1) 量级，Vandermonde 求逆的条件数在本
    项目工作阶数下只有个位数（order=3 也才 56），1e-12 远高于舍入噪声、
    又远低于任何有意义的滤波强度（legacy 档顶模态 sigma=2.2e-16，矩阵
    偏离单位阵是 O(1)）。

    `core/fr_solver/turbulence.py::_filter_matrices_are_identity` 是
    k/omega 那一侧的同一套判据（那里不经过 build_filter_func，所以
    独立实现），两者必须保持一致——有回归测试直接比对两个实现的结论。
    """
    for M in mats:
        if M is None:
            continue
        A = np.asarray(M)
        if not (A.ndim == 2 and A.shape[0] == A.shape[1]):
            return False
        if not np.allclose(A, np.eye(A.shape[0]), rtol=0.0, atol=1e-12):
            return False
    return True


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

    # 与 `build_filter_func` 同一条短路（两个矩阵都是机器精度意义上的
    # 单位阵时返回 None，让调用方整个跳过），理由见那边文档。分布式
    # 调用方（`core/mpi/distributed_solver.py`）把结果直接传给
    # `TimeIntegrator.step`，对 None 有显式支持。
    if _matrices_are_identity(filter_prism, filter_tet):
        return None

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


def filter_scalar_field_gated(
    phi: np.ndarray, filter_prism, filter_tet, troubled: np.ndarray,
    *, n_prism=None, cell_is_prism=None,
) -> np.ndarray:
    """`filter_scalar_field` 的逐单元门控版：只对 `troubled` 为真的单元
    施加滤波矩阵，其余单元逐位原样返回。

    矩阵与全局版完全相同，唯一区别是"对哪些单元施加"——`troubled` 全 True
    时结果与 `filter_scalar_field` 逐位一致（同一个 kernel、同一个矩阵、
    同样的分组顺序）。
    """
    if (n_prism is None) == (cell_is_prism is None):
        raise ValueError("n_prism 与 cell_is_prism 必须且只能给一个")
    n_cells = phi.shape[0]
    out = phi.copy()
    if not np.any(troubled):
        return out
    if cell_is_prism is not None:
        cip = np.asarray(cell_is_prism, dtype=bool)
        groups = ((np.flatnonzero(cip), filter_prism),
                  (np.flatnonzero(~cip), filter_tet))
    else:
        groups = ((np.arange(0, n_prism), filter_prism),
                  (np.arange(n_prism, n_cells), filter_tet))
    for sel_all, mat in groups:
        sel = sel_all[troubled[sel_all]]
        if sel.size == 0:
            continue
        sub_out = np.ascontiguousarray(phi[sel])
        _filter_scalar_kernel(np.ascontiguousarray(phi[sel]),
                              np.ascontiguousarray(mat), sub_out)
        out[sel] = sub_out
    return out


#: k/omega 滤波的门控方式，由 `AFCFD_FILTER_TURB_GATE` 选择：
#:   "all"（默认，与此前行为逐位一致）—— 所有单元都滤波
#:   "sensor"                         —— 只对传感器判定欠分辨的单元滤波
#:
#: **为什么这一维必须与 `AFCFD_FILTER_MODE` 独立**（2026-09-15 实测结论）：
#: `filter_scalar_field` 直接用 `ops.filter_prism`、完全不经过平均流那边
#: 的传感器门控，所以 `AFCFD_FILTER_MODE=sensor` 实际的语义一直是
#: "平均流门控 + k/omega 仍被完整清掉一整阶"。79 万单元真实网格 250 步
#: 对照决定性分离出了这一点：
#:
#:   档       平均流       k/omega     om_max 轨迹（起始 1.63e4）
#:   legacy   全局清零     全局清零    受控
#:   off      不滤波       **不滤波**  step100 达 1.65e5，增速持续加速
#:   sensor   门控(≈不滤波) 全局清零    step140 才 7.32e4，增速持续减速
#:
#: off 与 sensor 的**平均流**轨迹几乎逐位相同（step100 残差都是 2.211e9、
#: Cd 3.0831 vs 3.0830），说明传感器在平均流上几乎不触发；两者 om_max 的
#: 巨大差异**全部**来自 k/omega 那一维。也就是说 sensor 档的好处与"传感器
#: 在平均流上起作用"无关，而是来自 k/omega 仍被滤波——这正是
#: `filter_scalar_field` 文档记录的 2026-09-12 真实 P1 发散所需要的保护。
#:
#: 既然两维的效果可以完全分离，就不能再让一个环境变量同时决定它们。
def resolve_turb_filter_gate() -> str:
    """返回 k/omega 滤波的门控方式，并校验取值。

    每次调用都重读环境变量（不缓存模块级常量）：与
    `AFCFD_FILTER_MODE` 不同，这一维不影响算子构造，运行期读取是安全的，
    而且让测试可以用 monkeypatch 切换而不必 reload 模块。

    已接线的后端：**全部四条**——单机 CPU 与 CPU MPI 分布式
    （都经 `fr_solver/turbulence.py::compute_turbulence_source`，后者的
    `turb_view`/`mesh_adapter` 都在 compact"棱柱在前"索引空间，所以同一段
    `n_prism` 切片代码两条路径都正确）、单 GPU（`gpu_solver_io.py`）、
    多 GPU 分布式（`gpu_distributed_init.py`，后两条用
    `core/gpu/gpu_troubled_cell.py` 的 GPU 版同一套传感器，2026-09-15
    补齐）。所以这一维**不需要** `resolve_filter_mode` 那样的后端白名单
    ——平均流的 `sensor` 档要在 RK stage 内部逐 stage 求指标，k/omega 这
    一维只在每步湍流源项之后施加一次，补齐成本低得多。
    """
    gate = os.environ.get("AFCFD_FILTER_TURB_GATE", "all").lower()
    if gate not in ("all", "sensor"):
        raise ValueError(
            f"AFCFD_FILTER_TURB_GATE={gate!r} 不是合法取值（all | sensor）。"
            f"'all' 是既有行为（所有单元都滤波），'sensor' 只对传感器判定"
            f"欠分辨的单元滤波。注意 AFCFD_FILTER_MODE=off 会把滤波矩阵本身"
            f"变成单位阵，此时这一维无论取什么都是无操作。")
    return gate


def compute_turb_troubled_mask(k_field: np.ndarray, omega_field: np.ndarray,
                               order: int, *, n_prism=None,
                               cell_is_prism=None) -> np.ndarray:
    """k/omega 门控用的欠分辨掩码：对 k 与 omega **分别**求
    Persson-Peraire 指示器后取并集。

    为什么取并集而不是只看一个：2026-09-12 记录的真实发散发生在 omega
    （单元内部相邻解点间数量级跳变），而更早一轮攻关里失控的是 k（局部
    单元撞 k_max 上限，见 cube_demo omega realizability 那条记录）。两个
    场各自都会混叠，任一出问题都需要该单元被滤波，所以取并集而不是交集。

    为什么不复用平均流的掩码：一个单元完全可以密度光滑而 omega 有尖峰
    （实测 off/sensor 两档平均流轨迹几乎逐位相同、om_max 却差一个量级，
    就是这件事的直接证据）。
    """
    from autoflowcfd.core.fr_operators.artificial_viscosity import (
        compute_troubled_cell_mask,
    )
    kw = dict(n_prism=n_prism, cell_is_prism=cell_is_prism)
    mask_k = compute_troubled_cell_mask(np.ascontiguousarray(k_field), order, **kw)
    mask_om = compute_troubled_cell_mask(np.ascontiguousarray(omega_field), order, **kw)
    return mask_k | mask_om


#: 目前实现了 `sensor` 档的后端。`legacy`/`off`/`mild`/`project` 四档
#: **不需要**出现在这里——它们是在算子构造期改 `ops.filter_prism`/
#: `filter_tet` 本身（见 fr/modal_filter.py 的模块级常量），因此对全部
#: 后端自动生效；只有 `sensor` 需要在推进循环里逐单元门控。
#:
#: ## 另外三个后端缺的到底是什么（2026-09-17 查清，写成可执行的规格）
#:
#: 不是"接线没写"，是一处**真实的架构约束**：`bounds`（BJ 型）判据要的是
#: **面邻居的单元均值**，而分区边界上的邻居是 halo 单元；而 RK stage 的
#: 滤波回调 `filter_func(U_flat)` 只拿到**本地** U（`n_local` 个单元），
#: 拿不到 halo。所以补齐它需要把 halo 的单元均值也传进 stage 回调——那是
#: 时间积分器回调契约的改动，不是几行接线。
#:
#: 三个后端各自缺的：
#:   * `cpu-mpi`：面连接数组（`owner_cell_local`/`neighbor_cell_local`/
#:     `is_boundary`）与 `cell_is_prism` 都已就位（见
#:     `mpi/distributed_solver.py` 里 `build_filter_func_by_cell_type` 的
#:     调用处），**只缺 halo 单元均值**。注意索引空间：那些数组是"棱柱
#:     在前"的置换排列，而 `filter_func` 收到的是原生排列，映射是
#:     `native = perm[permuted]`（见 `distributed_flat_face` 的 inv_perm
#:     文档）——接线时必须转，否则会静默用错单元。
#:   * `gpu-single` / `gpu-mpi`：除上面那条外，还缺 `compute_bounds_
#:     violation_mask` 的 CuPy 版——它用 `np.add.at`/`np.maximum.at` 做
#:     scatter 归约，CuPy 没有直接对应（要用 `cupyx.scatter_add` 与
#:     手写的 scatter-max），不是把 np 换成 cp 就行。
#:
#: 在补齐之前，`resolve_filter_mode` 对**默认值**落到 sensor 的情形退到
#: `project` 并打一条量化了数值后果的警告；对**显式**请求仍然报错。
_SENSOR_MODE_SUPPORTED_BACKENDS = ("cpu-single",)

#: 已经为哪些后端打过"默认 sensor 退到 project"的警告（每后端只打一次，
#: 避免逐步刷屏；`resolve_filter_mode` 在每步都会被调用）。
_SENSOR_FALLBACK_WARNED = set()


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
    # **默认值必须与 `fr/modal_filter.py` 的同一个环境变量解析保持一致**
    # （2026-09-17 真实 bug）：那边定滤波**矩阵**的 sigma，这边定 `step.py`
    # 走不走**逐单元门控**分支。把默认值从 legacy 改成 sensor 时只改了
    # 那一处，于是默认路径变成"矩阵是精确投影、但全局逐 stage 施加"——
    # 功能上等于 legacy（实测 legacy 与 project 在 P1 上逐位相同），
    # 壁面剪应力照样被清零（实测 du/dy 恒为 0）。两处必须同源，所以这里
    # 直接复用那个模块的常量而不是再写一遍默认值。
    from autoflowcfd.fr.modal_filter import FILTER_MODE as _MATRIX_MODE

    _raw = os.environ.get("AFCFD_FILTER_MODE")
    explicit = _raw is not None and _raw.strip() != ""
    mode = (_raw.lower() if explicit else _MATRIX_MODE.lower())

    if mode == "sensor" and backend not in _SENSOR_MODE_SUPPORTED_BACKENDS:
        if explicit:
            # **显式请求**得不到满足时必须报错：给了明确指令却拿到别的
            # 数值方案，是本项目一贯不接受的静默行为。
            raise NotImplementedError(
                f"AFCFD_FILTER_MODE=sensor 尚未在后端 '{backend}' 上接线"
                f"（已接线：{', '.join(_SENSOR_MODE_SUPPORTED_BACKENDS)}）。"
                f"传感器门控需要在推进循环里逐单元求传感器指示器，不是算子"
                f"构造期就能定下来的，必须逐后端实现。"
                f"legacy/off/mild/project 四档对全部后端都有效，可先用它们；"
                f"若确实需要在该后端用 sensor 档，请先补齐接线而不是忽略"
                f"本错误。")
        # **默认值**落到 sensor 而该后端未接线：退到 `project`（同一个
        # 滤波矩阵、全局施加）并**大声**说明差异。为什么这一档可以退而
        # 显式请求不可以：默认值必须让每个后端都能跑起来，而一条警告
        # 使它不是"静默"——本项目禁止的是**无声**地把明确请求换掉。
        global _SENSOR_FALLBACK_WARNED
        if backend not in _SENSOR_FALLBACK_WARNED:
            _SENSOR_FALLBACK_WARNED.add(backend)
            logger.warning(
                f"[FILTER] 默认滤波档是 'sensor'（逐单元门控），但后端 "
                f"'{backend}' 尚未接线，本次退到 'project'（同一个精确投影"
                f"矩阵、**全局逐 RK stage 施加**）。数值后果是明确的："
                f"全局施加会精确抹掉最高一阶多项式内容，P1 因此退化成 P0，"
                f"实测壁面法向速度梯度被清零（平板边界层算例 du/dy 从 1734 "
                f"变成 0）。要在本后端拿到门控行为，需要补齐该后端的接线；"
                f"要显式选别的档请设 AFCFD_FILTER_MODE=off|mild|project|legacy。")
        return "project"
    return mode


def build_sensor_gated_filter_func_arrays(
    n_cells: int, n_sps: int, order: int, filter_prism, filter_tet,
    *, n_prism=None, cell_is_prism=None,
    sensor: str = "persson",
    owner_cell=None, neighbor_cell=None, is_boundary=None,
    freestream=None,
) -> Callable[[np.ndarray], np.ndarray]:
    """传感器门控模态滤波的**后端无关**实现（只吃数组，不吃 solver）。

    这是 `build_sensor_gated_filter_func` 的内核。拆出来的理由见
    `fr_operators/artificial_viscosity.py::compute_troubled_cell_mask`：
    第一版门控靠"把当前 stage 的解临时塞进 `solver.state.U` 再调用
    solver 版传感器"实现，那是个副作用 hack，而且让门控只能用在单机
    CPU 推进循环里。现在传感器是纯数组接口，CPU MPI 那条路径（local
    排列、棱柱/四面体交错，用 `cell_is_prism`）可以用同一个内核。

    门控判据：对**守恒密度**求 Persson-Peraire 指示器。守恒密度与原始
    密度只差一个整体缩放，而 S_e 是能量比值、对缩放不变，所以两者给出
    同一个判据——不需要先反算原始变量。

    施加方式：欠分辨的单元走完整的滤波矩阵（该单元确实有需要压制的
    混叠内容），其余单元**完全不动**（保留全部已解析的多项式内容）。
    本函数不改变滤波矩阵本身（仍用传入的 filter_prism/filter_tet），
    只改变"对哪些单元施加"——`AFCFD_FILTER_MODE=off/mild` 那两档改的是
    矩阵，两个维度可以独立组合，便于受控 A/B。

    Args:
        n_cells, n_sps: 单元数与每单元解点数
        order: 当前多项式阶数（order==0 时传感器恒不触发，等价于不滤波）
        filter_prism, filter_tet: 滤波矩阵
        n_prism / cell_is_prism: 单元类型划分，恰好给一个，语义同
            `compute_troubled_cell_mask`
        sensor: 门控判据（`AFCFD_TROUBLED_SENSOR`，见
            `fr_operators/bounds_sensor.py::resolve_troubled_sensor`）：

              persson  Persson-Peraire 模态能量指示器（默认，既有行为）
              bounds   邻居极值越界（BJ 型）
              both     两者取并集

            **为什么需要第二个判据**：Persson-Peraire 的 `s0 =
            -4*log10(order)` 在 order=1 时为 0，门限退化成"顶模态能量
            占比 >= 10%"，而 P1 的顶模态就是全部非常数模态——它在生产
            阶数 P1 上原理上不适用（实测 A/B 前 51 步逐字符相同）。
            BJ 型判据不依赖模态分解，没有这个退化。完整推导与真实网格
            实测见 `fr_operators/bounds_sensor.py` 模块文档。
        owner_cell / neighbor_cell / is_boundary: 面连接数组，`sensor`
            含 "bounds" 时**必须**给出（BJ 判据要邻居均值）。索引空间
            必须与 `n_cells` 一致——分布式 local 排列传 local 面数组。
        freestream: `solver.freestream` 字典，`sensor` 含 "bounds" 时
            **必须**给出——BJ 判据的绝对地板要用来流参考量级
            （见 `bounds_sensor.py` 模块文档"第一版用全场 RMS 做尺度
            为什么不行"一节：用全场 RMS 时实测标记了 11.08% 的单元）。
    """
    from autoflowcfd.core.fr_operators.artificial_viscosity import (
        compute_troubled_cell_mask,
    )
    from autoflowcfd.core.fr_operators.bounds_sensor import (
        compute_bounds_violation_mask,
    )
    from autoflowcfd.core.fr_solver.residual_diagnostics import (
        _reference_scales,
    )
    if (n_prism is None) == (cell_is_prism is None):
        raise ValueError("n_prism 与 cell_is_prism 必须且只能给一个")
    sensor = str(sensor).lower()
    if sensor not in ("persson", "bounds", "both"):
        raise ValueError(f"未知 sensor: {sensor!r}")
    need_conn = sensor in ("bounds", "both")
    if need_conn and freestream is None:
        raise ValueError(
            f"sensor={sensor!r} 需要 freestream（BJ 判据的绝对地板用来流"
            f"参考量级），不能静默退回用全场 RMS——实测那会标记 11% 的单元"
        )
    if need_conn and (owner_cell is None or neighbor_cell is None
                      or is_boundary is None):
        # 不静默退回 persson：那会让"我明明开了 bounds 档"与实际行为
        # 不一致，而这种不一致在日志里完全看不出来。
        raise ValueError(
            f"sensor={sensor!r} 需要 owner_cell/neighbor_cell/is_boundary "
            f"三个面连接数组，缺失的不能静默忽略"
        )
    if cell_is_prism is not None:
        cip = np.asarray(cell_is_prism, dtype=bool)
        prism_idx_all = np.flatnonzero(cip)
        tet_idx_all = np.flatnonzero(~cip)
    else:
        prism_idx_all = np.arange(0, n_prism)
        tet_idx_all = np.arange(n_prism, n_cells)

    def filter_func(U_flat: np.ndarray) -> np.ndarray:
        U = U_flat.reshape(n_cells, n_sps, -1)
        # Persson-Peraire 只探**守恒密度**（S_e 是能量比值、对整体缩放
        # 不变，所以守恒密度与原始密度给出同一个判据）。BJ 判据则探
        # 全部 5 个守恒变量并取并集：2026-09-16 的真实 checkpoint 实测
        # 里越界量最大的是横向动量与压力，只探密度会漏掉它们
        # （同一类问题：人工粘性的 DEFAULT_SENSOR_VAR_INDEX = 0 只探
        # 密度，而 P2 的失效模态在能量上）。
        troubled = np.zeros(n_cells, dtype=bool)
        if sensor in ("persson", "both"):
            troubled |= compute_troubled_cell_mask(
                np.ascontiguousarray(U[:, :, 0]), order,
                n_prism=n_prism, cell_is_prism=cell_is_prism)
        if sensor in ("bounds", "both"):
            troubled |= compute_bounds_violation_mask(
                np.ascontiguousarray(U[:, :, :5]),
                owner_cell, neighbor_cell, is_boundary,
                ref_scales=_reference_scales(freestream, 5))
        if not np.any(troubled):
            return U_flat
        for sel_all, mat in ((prism_idx_all, filter_prism),
                             (tet_idx_all, filter_tet)):
            sel = sel_all[troubled[sel_all]]
            if sel.size == 0:
                continue
            sub = np.ascontiguousarray(U[sel])
            _filter_leading_vars_inplace_kernel(sub, np.ascontiguousarray(mat), 5)
            U[sel] = sub
        return U.reshape(U_flat.shape)

    return filter_func


def build_sensor_gated_filter_func(solver) -> Callable[[np.ndarray], np.ndarray]:
    """按 Persson-Peraire 传感器**逐单元门控**的模态滤波（2026-09-15）。

    为什么需要它：全局每-stage 施加的滤波器在本项目的工作阶数上会精确
    抹掉一整阶（order=1 -> 只剩常数，order=2 -> 只剩双线性，见
    `fr/modal_filter.py` 顶部的实测记录）。而且因为它每个 RK stage 都施加，
    任何 sigma<1 都会随步数复合累积，单纯调小 alpha 只是把清零推迟——
    正确的方向是**只在确实需要的单元上施加**。

    本函数只是 `build_sensor_gated_filter_func_arrays` 的单机适配层
    （"棱柱在前"排列，order 取 `current_order`/`order`），全部说明见
    那边与 `fr_operators/artificial_viscosity.py::compute_troubled_cell_mask`。

    **重要范围说明**：本函数（以及 `AFCFD_FILTER_MODE=sensor`）只门控
    **平均流**滤波。k/omega 走的是 `filter_scalar_field`，由
    `fr_solver/turbulence.py` 单独调用、有**独立**的门控开关，理由见
    `filter_scalar_field` 文档"为什么 k/omega 的门控必须独立判定"一节。
    """
    from autoflowcfd.core.fr_operators.bounds_sensor import (
        resolve_troubled_sensor,
    )

    mesh = solver.mesh
    ops = solver.ops
    order = getattr(solver, "current_order", None)
    if order is None:
        order = solver.order
    sensor = resolve_troubled_sensor()
    conn = {}
    if sensor in ("bounds", "both"):
        fc = mesh.face_connectivity
        if fc is None:
            raise RuntimeError(
                "AFCFD_TROUBLED_SENSOR=bounds/both 需要 mesh.face_connectivity"
                "（BJ 判据要面邻居均值），当前网格没有构建面连接"
            )
        conn = dict(owner_cell=np.asarray(fc.owner_cell),
                    neighbor_cell=np.asarray(fc.neighbor_cell),
                    is_boundary=np.asarray(fc.is_boundary, dtype=bool),
                    freestream=solver.freestream)
    return build_sensor_gated_filter_func_arrays(
        mesh.n_cells, mesh.n_sps_per_cell, int(order),
        ops.filter_prism, ops.filter_tet, n_prism=mesh.n_prism_cells,
        sensor=sensor, **conn)


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
    # padding.py::pad_native_tet_filter_matrix_to_global 文档——滤波器
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
