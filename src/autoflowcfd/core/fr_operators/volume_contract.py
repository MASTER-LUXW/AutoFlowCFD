"""体积项 / 梯度里"共享算子 x 逐单元场"张量收缩与逆变通量的 numba 并行核。

`core/fr_residual/`（体积项过积分、粘性体积散度）与 `core/fr_operators/gradients.py`（物理梯度）大量出现::

    out[c, f, v] = sum_s   D[f, s]    X[c, s, v]       # contract_shared_operator_1axis
    out[c, f, v] = sum_j,m D[f, j, m] X[c, j, m, v]    # contract_shared_operator_2axis（(j, m) 合并成一个轴）

`D`（微分 / 插值 / 限制算子）对所有单元相同，只有 `X` 随单元变化。两种形状共用一个按单元 `prange` 的核
（`_contract_shared_kernel` -> `contract_local`，4 路累加器），逐单元融合核（`fused_volume/`、`gradients.py`）也
经过同一个 `contract_local`，求和顺序处处相同。逆变通量 `adj(J) F` 由 `contravariant_flux_from_metric` 融合求值，
`adj(J) = det(J) J^{-1}` 不落内存。

为什么不交给 BLAS（2026-09-13 实测）：逐单元的 gemm 只有 (8,24)@(24,5)（P1）~ (27,81)@(81,5)（P2），几百到几千
FLOPs，79 万单元的批量循环几乎全是调用开销，且 numpy 不在批量维度上并行（加核不提速）；`np.tensordot` 还会把
`X` 的单元轴转置出一份完整副本（364,555 单元 P2 实测 `Unable to allocate 1.10 GiB`）。按单元 `prange` 后 P1 形状
4.7 倍、P2 形状 3.5 倍，1 -> 8 线程 7.7 倍扩展。
"""

import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True)
def compute_adj_j(det_jacs: np.ndarray, inv_jacs: np.ndarray) -> np.ndarray:
    """等价于 `det_jacs[..., None, None] * inv_jacs`（构造 adj(J) = det(J) *
    J^{-1}，逆变通量/梯度链式法则要用的度量伴随矩阵）。

    Args:
        det_jacs: (n_cells, n_pts)
        inv_jacs: (n_cells, n_pts, 3, 3)

    Returns:
        (n_cells, n_pts, 3, 3)

    性能优化：这是一个纯 numpy 广播乘法，此前直接写成
    `det_jacs[...,None,None] * inv_jacs`——数学上没问题，但 numpy 的
    ufunc 广播默认单线程执行，在无粘残差的过积分
    fine 网格上（P2 阶数 n_pts=64）这一行单次调用要处理约 3.6GiB 数据，
    py-spy 采样过的真实 79万单元生产网格 profile 证实
    `compute_inviscid_residual_fr` 自身（不算任何子函数调用）的耗时里
    这类逐元素广播运算是主要成分之一。换成 numba `prange` 按 cell 并行
    的直接三重循环，消除 numpy 通用广播的单线程限制——与
    `contract_shared_operator_*axis`（同一模块文档）的"换计算路径、
    不改变数学公式"是同一类优化，已用随机张量在 P0-P3 各阶数对应的
    n_pts 上验证与原始广播乘法逐位一致（最大误差 0.0），79万单元×64点
    （P2 过积分 fine 网格）规模下实测 8.8 倍提速（1.32s -> 0.15s）。
    """
    n_cells, n_pts = det_jacs.shape
    out = np.empty((n_cells, n_pts, 3, 3))
    for c in prange(n_cells):
        for p in range(n_pts):
            d = det_jacs[c, p]
            for i in range(3):
                for j in range(3):
                    out[c, p, i, j] = d * inv_jacs[c, p, i, j]
    return out


@njit(cache=True, inline="always")
def contravariant_point(d, inv, F, out) -> None:
    """单点逆变通量 `out[i, v] = sum_j (d * inv[i, j]) * F[j, v]`（`inv`: (3, 3)，`F`/`out`: (3, V)）。

    批量核 `_contravariant_flux_kernel` 与逐单元融合核（`fused_volume/`）共用这一个定义，乘加顺序因此
    处处相同（融合前后逐位一致的前提）。
    """
    for i in range(3):
        a0 = d * inv[i, 0]
        a1 = d * inv[i, 1]
        a2 = d * inv[i, 2]
        for v in range(F.shape[1]):
            out[i, v] = a0 * F[0, v] + a1 * F[1, v] + a2 * F[2, v]


@njit(cache=True, parallel=True)
def _contravariant_flux_kernel(det_jacs, inv_jacs, F_phys, out) -> None:
    """`out[c,p,i,v] = det_jacs[c,p] * sum_j inv_jacs[c,p,i,j] * F_phys[c,p,j,v]`。"""
    C, P = det_jacs.shape
    for c in prange(C):
        for p in range(P):
            contravariant_point(det_jacs[c, p], inv_jacs[c, p], F_phys[c, p], out[c, p])


def contravariant_flux_from_metric(det_jacs: np.ndarray, inv_jacs: np.ndarray,
                                   F_phys: np.ndarray) -> np.ndarray:
    """等价于 `np.matmul(compute_adj_j(det_jacs, inv_jacs), F_phys)`——把
    "构造 adj(J) = det(J)*J^{-1}" 与 "逆变通量 F_tilde = adj(J) @ F_phys"
    融合成一个 numba 并行 kernel。

    Args:
        det_jacs: (C, P)
        inv_jacs: (C, P, 3, 3)
        F_phys: (C, P, 3, V) 物理通量

    Returns:
        F_tilde: (C, P, 3, V) 逆变通量

    性能优化（2026-09-13，与 `_contract_shared` 同一次真实剖析）：原路径
    先用 `compute_adj_j` 物化一份 (C,P,3,3) 的 adj(J)（P1 过积分细点
    P=27 时，79万单元下约 1.5GiB），再交给 `np.matmul` 做逐点
    3x3 @ 3xV 的批量微型 gemm——单次只有 45~135 FLOPs，纯调用开销主导，
    且同样**不随 CPU 核数并行**。融合后 adj(J) 只作为寄存器里的 3 个临时
    标量存在（完全不落内存），逐点工作集只有几十个 double、留在 L1 内，
    对 cell 轴 prange 并行。数学公式逐项对应原实现（`adj_j[c,p,i,j] =
    det*inv_jacs[c,p,i,j]`，再对 j 求和），乘加顺序与 `np.matmul` 的
    j=0,1,2 顺序一致，因此 V>=2 时结果与原路径**逐位相同**（随机张量实测
    最大误差 0.0；生产路径的平均流无粘/粘性通量都是 V=5）。V==1 时
    `np.matmul` 退化成矩阵-向量（gemv）路径、累加方式与 gemm 不同，只能到
    机器精度（实测 9.3e-17 相对误差，1 ULP）——生产路径里 V=1 只出现在
    标量对流的共享逆变质量通量（`core/turbulence/transport/::
    precompute_scalar_convection_geometry`）。等价性回归测试见
    tests/unit/test_perf_fusion_kernels.py。
    """
    # 度量量**不**转连续：核逐元素读取，接受任意步长——过积分四面体段传进来的
    # 是"逐单元常数广播到全部细点"的零拷贝视图（步长 0），此前每次残差求值都
    # 把它逐块物化一遍（plate_demo P1+SST 每个隐式步约 2 s 拷贝，2026-09-25 去掉）。
    F_c = np.ascontiguousarray(F_phys)
    out = np.empty_like(F_c)
    _contravariant_flux_kernel(det_jacs, inv_jacs, F_c, out)
    return out


@njit(cache=True, inline="always")
def grad_to_physical_point(g, inv, out) -> None:
    """单点链式法则 `out[v, n] = sum_m g[m, v] * inv[m, n]`（`g`: (3, V)，`inv`: (3, 3)，`out`: (V, 3)）。
    批量核与逐单元融合梯度（`gradients.py`）共用。"""
    for v in range(g.shape[1]):
        g0 = g[0, v]
        g1 = g[1, v]
        g2 = g[2, v]
        for n in range(3):
            out[v, n] = g0 * inv[0, n] + g1 * inv[1, n] + g2 * inv[2, n]


@njit(cache=True, parallel=True)
def _contract_shared_kernel(D_flat: np.ndarray, X_flat: np.ndarray, out: np.ndarray) -> None:
    """`out[c,f,v] = sum_k D_flat[f,k] * X_flat[c,k,v]`，按 cell 轴 prange 并行。

    1axis/2axis 两个公开入口收缩的都是"共享算子 × 逐 cell 场"这同一个
    形状（2axis 只是先把相邻的 (J,M) 两个收缩轴合并成一个 K 轴，见各自
    文档），因此共用这一个 kernel，不写两份。

    4 路累加器：K 轴按 4 拆成 4 个独立部分和再合并，而不是单个标量顺序
    累加。两个理由——(a) 4 条独立依赖链让 SIMD/流水线更容易填满；
    (b) 误差增长从 O(K·eps) 降到约 O(K/4·eps)，在本项目这种"长求和近乎
    精确抵消"的场景（离散 GCL / 自由流场保持性；P2 过积分细点下
    K = n_fine*3 = 375，P3 是 1029）更接近被替换掉的 BLAS gemm 的分块
    累加行为。

    **一次归因更正（2026-09-13，避免后人误读）**：这 4 路累加器最初是
    为了修复 `tests/unit/test_native_tet_inviscid_residual_wiring.py::
    test_native_mesh_free_stream_preservation` 棱柱 P2 判据的一次失败
    （8.07e-7 -> 3.35e-6，容差 1e-6）而加的，但随后的逐项回退实验**证伪
    了那个假设**：把本文件三个融合核全部换回原 numpy 实现后该判据仍是
    3.35e-6，单独把 BLAS 线程数从 1 改回 16 才恢复到 8.0e-7——真正的原因
    是当时把 BLAS 线程数在**进程级**压到了 1，改变了网格几何（inv_jacs
    由 LAPACK 求逆）与 FR 算子构造的最后一位，而保持性依赖这些度量量
    之间的精确抵消（完整记录见 `autoflowcfd/__init__.py` 顶部与
    `core/fr_solver/solver/threads.py::blas_threads_limited`）。4 路累加器本身对
    该判据几乎没有影响（3.3470e-6 -> 3.3394e-6），保留是因为上面 (a)(b)
    两条理由本身成立，**不是**那次失败的修复手段。
    """
    for c in prange(X_flat.shape[0]):
        contract_local(D_flat, X_flat[c], out[c])


@njit(cache=True, inline="always")
def contract_local(D, X, out) -> None:
    """`out[f, v] = sum_k D[f, k] * X[k, v]`（`X`: (K, V)），4 路累加器（见 `_contract_shared_kernel` 文档）。

    共享算子收缩、逐单元算子收缩与逐单元融合核（`fused_volume/`、`gradients.py`）都经过这一个定义，求和
    顺序处处相同。`X` 可以是非连续视图（逐元素读取）。
    """
    K = X.shape[0]
    V = X.shape[1]
    K4 = (K // 4) * 4
    for f in range(D.shape[0]):
        for v in range(V):
            s0 = 0.0
            s1 = 0.0
            s2 = 0.0
            s3 = 0.0
            for k in range(0, K4, 4):
                s0 += D[f, k] * X[k, v]
                s1 += D[f, k + 1] * X[k + 1, v]
                s2 += D[f, k + 2] * X[k + 2, v]
                s3 += D[f, k + 3] * X[k + 3, v]
            tail = 0.0
            for k in range(K4, K):
                tail += D[f, k] * X[k, v]
            out[f, v] = (s0 + s1) + (s2 + s3) + tail


def cell_blocks(n_cells: int) -> int:
    """逐单元融合核的并行块数：每块按单元区间顺序处理、块内只分配一次局部工作数组（不随单元数分配）。

    取线程数的 8 倍以平衡单元间工作量差异（混合面、不同组合编号的算子都在同一段里），不超过单元数。
    """
    import numba

    return max(1, min(int(n_cells), 8 * int(numba.get_num_threads())))


def _contract_shared(D_flat: np.ndarray, X_flat: np.ndarray) -> np.ndarray:
    """`_contract_shared_kernel` 的分配 + 连续性包装（为什么不用 BLAS 见模块文档）。"""
    D_c = np.ascontiguousarray(D_flat)
    X_c = np.ascontiguousarray(X_flat)
    out = np.empty((X_c.shape[0], D_c.shape[0], X_c.shape[2]))
    _contract_shared_kernel(D_c, X_c, out)
    return out


def contract_shared_operator_1axis(D: np.ndarray, X: np.ndarray) -> np.ndarray:
    """等价于 `np.einsum("fs,csv->cfv", D, X)`。

    Args:
        D: (F, S) —— 不依赖 cell 的共享算子（如 interp_c2f、restrict_f2c）
        X: (C, S, V) —— 逐 cell 的场

    Returns:
        (C, F, V)
    """
    return _contract_shared(D, X)


def contract_shared_operator_2axis(D: np.ndarray, X: np.ndarray) -> np.ndarray:
    """等价于 `np.einsum("fjm,cjmv->cfv", D, X)`。

    Args:
        D: (F, J, M) —— 不依赖 cell 的共享算子（如原生四面体 / 棱柱微分算子、细层微分算子）
        X: (C, J, M, V) —— 逐 cell 的场

    Returns:
        (C, F, V)
    """
    F, J, M = D.shape
    C, _, _, V = X.shape
    # J,M 相邻，合并成 K 轴是 no-copy view（小算子 D 同理）。
    return _contract_shared(D.reshape(F, J * M), X.reshape(C, J * M, V))


def fine_segments(oi):
    """过积分上下文逐段给出 `(lo, hi, c2f, det_f, inv_f, D2, f2c)`（空段跳过），供逐单元融合核使用：
    `D2` 是细点微分矩阵收缩轴合并后的 `(n_fine, 3 n_fine)`（行 `q`、列 `3 p + m`，与逆变通量按 `(p, m)`
    展平的顺序一致），算子都已转连续；细点度量保持原样（四面体段是步长 0 的广播视图，核逐元素读取）。"""
    for lo, hi, n_fine, det_f, inv_f, c2f, D_fine, f2c in oi["segs"]:
        if hi > lo:
            yield (lo, hi, np.ascontiguousarray(c2f), det_f, inv_f,
                   np.ascontiguousarray(D_fine).reshape(n_fine, 3 * n_fine), np.ascontiguousarray(f2c))


def lifted_segments(oi):
    """平均流体积算子 `K` 的逐段 `(lo, hi, c2f, det_f, inv_f, K2, combo)`：`K2` 是 `K_all` 收缩轴合并后的
    `(n_combo, n_sps, 3 n_fine)`，`combo` 为该段逐单元组合编号（`fr/face_flux_trace.py`）。"""
    for (lo, hi, _n_fine, det_f, inv_f, c2f, _D, _f2c), (K_all, combo) in zip(oi["segs"], oi["lifted_div"]):
        if hi > lo:
            n_combo, n_out, n_fine, three = K_all.shape
            yield (lo, hi, np.ascontiguousarray(c2f), det_f, inv_f,
                   np.ascontiguousarray(K_all).reshape(n_combo, n_out, n_fine * three),
                   np.ascontiguousarray(combo, dtype=np.int64))


def get_overintegration_context(mesh, ops):
    """取过积分所需的全部算子与细点度量；任一缺失则返回 None（调用方
    退回 coarse 路径）。

    三个消费方共用同一份：`fr_residual/inviscid.py`（无粘体积项，本项目
    唯一一直开着的）、`core/turbulence/transport/`（k/omega 对流与
    扩散体积项）、`fr_residual/viscous_flux/`（粘性体积项）。

    缺失的唯一正常情形是 `order == 0`：P0 是分片常数场、多项式导数恒为
    零，没有可去混叠的内容，`jacobians_fine` 与 overint 算子都不构造。

    Returns:
        dict 或 None。`segs`——
        `[(lo, hi, n_fine, det_fine, inv_fine, c2f, D_fine, f2c), ...]`
        两段（棱柱在前、四面体在后），与"棱柱在前"的单元存储顺序一致。
        每段自带**自己的** `n_fine` 与已按 `(seg_len, n_fine, ...)` 切好的
        细点度量；其中 `f2c` 是解点插值（湍流 k-omega 输运用）。`lifted_div`——与
        `segs` 逐段对应的 `(K_all, combo)`：平均流体积算子 `K` 按三角形面槽位组合各一份
        `(n_combo, n_sps, n_fine, 3)`（本侧通量迹已并入，`fr/face_flux_trace.py`）与该段
        逐单元组合编号，收缩在融合核里做（`lifted_segments`）。

    ## 为什么每段各自带 n_fine（2026-09-17 改动）

    此前返回的是**共享**的 `n_fine`/`det_fine`/`inv_fine`，因为棱柱与
    四面体的过积分算子都被填充到同一个 `(over_order+1)^3` 宽度。但 native
    四面体在 over_order 下只有 `(oo+1)(oo+2)(oo+3)/6` 个**真实**细点
    （P1: 10 vs 27，P2: 20 vs 64），填充槽位恒为零、对结果零贡献，却让
    整条过积分链在空点上白算——其中 `D_fine` 的收缩是 O(n_fine^2)，P2 上
    是 10.2 倍的无效 FLOPs。实测去掉细轴填充后 P1 加速 3.04x、P2 加速
    4.63x，最大相对差 1.4e-16 / 0.0（见 `fr/operators/` 那段说明）。

    **共享键被刻意移除**（不做向后兼容别名）：漏改的消费点会直接
    `KeyError`，而不是静默用棱柱的 `n_fine` 去切四面体段的度量——后者
    会产生形状不匹配或（更糟）安静的错误答案。

    ## 四面体段的细点度量从逐单元值广播

    直边四面体的 Jacobian **逐单元为常数**，`mesh.jacobians_fine` 的四面体段只存
    每单元一份（`tet_det (n_tet,)`、`tet_inv (n_tet,3,3)`，见
    `grid/high_order/order_jacobians.build_fine_metrics`），这里零拷贝广播到该段
    自己的 `n_fine_tet` 列。四面体的过积分阶数因此不受棱柱细点数约束（P3 可以
    取理想的 oo=6：去混叠误差在 oo=2*order 处断崖，oo=5 3.37e-3、oo=6 4.80e-6）。
    2026-09-30 之前四面体段按棱柱细点数展开存储、这里取第 0 列广播，数值相同，
    但 plate_demo P3 上那一段白占约 2.5 GB。
    """
    if getattr(mesh, "jacobians_fine", None) is None:
        return None
    for name in ("overint_interp_c2f_prism", "overint_D_fine_prism",
                 "overint_restrict_f2c_prism", "overint_interp_c2f_tet",
                 "overint_D_fine_tet", "overint_restrict_f2c_tet",
                 "overint_lifted_div_prism", "overint_lifted_div_tet"):
        if getattr(ops, name, None) is None:
            return None
    fine = mesh.jacobians_fine
    n_cells = mesh.n_cells
    n_prism = mesh.n_prism_cells
    det_prism, inv_prism = fine["prism_det"], fine["prism_inv"]

    # 两段的 n_fine 都从**矩阵自身的形状**推导（矩阵是单一事实来源），棱柱段再与
    # 网格几何的细点数对账：不一致即算子与网格几何用了不同的 over_order 或不同的
    # 棱柱基档（`fr/overintegration_order.py` 是唯一入口，这里是运行期第二道闸）。
    n_fine_prism = int(ops.overint_D_fine_prism.shape[0])
    if det_prism.shape != (n_prism, n_fine_prism):
        raise ValueError(
            f"棱柱过积分细点数不一致：算子 overint_D_fine_prism 是 "
            f"{ops.overint_D_fine_prism.shape}（n_fine={n_fine_prism}），而网格的棱柱"
            f"细点度量是 {det_prism.shape}——算子与网格几何用了不同的 over_order "
            f"或不同的棱柱基档"
        )
    n_fine_tet = int(ops.overint_D_fine_tet.shape[0])
    _declared = int(getattr(ops, "overint_n_fine_tet", 0) or 0)
    if _declared and _declared != n_fine_tet:
        raise ValueError(
            f"ops.overint_n_fine_tet={_declared} 与 overint_D_fine_tet 的"
            f"形状 {ops.overint_D_fine_tet.shape} 不一致——算子构造有 bug，"
            f"不静默采用其中一个"
        )
    # 四面体段：逐单元常数 -> 零拷贝广播到 n_fine_tet 宽；消费方分块时自己
    # `np.ascontiguousarray` 物化当前块（numba kernel 需要连续输入）。
    n_tet = n_cells - n_prism
    det_tet = np.broadcast_to(fine["tet_det"][:, None], (n_tet, n_fine_tet))
    inv_tet = np.broadcast_to(fine["tet_inv"][:, None], (n_tet, n_fine_tet, 3, 3))

    return dict(
        segs=(
            (0, n_prism, n_fine_prism,
             det_prism, inv_prism,
             ops.overint_interp_c2f_prism, ops.overint_D_fine_prism,
             ops.overint_restrict_f2c_prism),
            (n_prism, n_cells, n_fine_tet,
             det_tet, inv_tet,
             ops.overint_interp_c2f_tet, ops.overint_D_fine_tet,
             ops.overint_restrict_f2c_tet),
        ),
        lifted_div=((ops.overint_lifted_div_prism, fine["prism_k_combo"]),
                    (ops.overint_lifted_div_tet, fine["tet_k_combo"])),
    )
