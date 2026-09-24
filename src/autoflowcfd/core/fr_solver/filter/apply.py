"""AutoFlowCFD V2.0 - 滤波矩阵的实际施加（numba kernel 层）。

从 `core/fr_solver/filter.py`（原 1008 行）拆出（2026-09-24，项目"单文件
不超 500 行"规范）。**纯搬家，逻辑未改**。

只做一件事：给定滤波矩阵，按单元类型（棱柱在前 / 四面体在后）把它作用在
展平后的 `U` 上。恒等矩阵的短路判据（`_matrices_are_identity`）也在这里
—— `off` 档下滤波器是恒等阵，短路掉可以省掉整趟 kernel。
"""


import numpy as np
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
