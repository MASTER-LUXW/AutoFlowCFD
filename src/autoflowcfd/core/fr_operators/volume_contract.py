"""体积项/梯度里"共享算子 (不依赖 cell) 张量收缩"的高效实现 (性能优化配套)。

`fr_residual_inviscid.py`（体积项 over-integration 三件套 + 无 fine 几何回退
分支）、`fr_viscous_flux.py`（体积项散度）、`fr_gradients.py`（物理空间梯度）
里大量出现形如::

    np.einsum("fs,csv->cfv", D, X)      # D 不依赖 cell，只收缩 1 个轴
    np.einsum("fjm,cjmv->cfv", D, X)    # D 不依赖 cell，收缩 2 个轴 (j,m)

的调用——`D`（微分/插值/限制算子）对所有 cell 共享、完全相同，只有 `X`
随 cell 变化。这类收缩在数学上等价于"共享矩阵 @ 逐 cell 展平后的大矩阵"，
应当归约成一次 BLAS gemm；但 `np.einsum` 不传 `optimize=True` 时走的是
通用逐元素求和路径，不会自动识别并利用这个结构——在 545,597 cell 的
生产网格上实测是体积项的主要热点（py-spy 对 P2 阶数残差求值的采样，
`euler_physical_flux`/`einsum` 相关帧占据几乎全部采样）。

`np.tensordot` 对两个操作数的收缩，内部本身就是"reshape 成 2D + 调用
`np.dot`（BLAS gemm）+ reshape 回去"（numpy 自己的实现，非本项目新写的
数值算法）——用它替换这些 einsum 调用，是完全等价的同一个求和公式，
只是换一条计算路径，不改变数学结果（浮点重结合误差与此前 numba 化
界面项时处理的是同一类、已用相对容差验证过的现象，见
`fr_residual_inviscid_kernel.py` 模块文档）。

两个小函数只处理"D 在前、X 在后，D 的收缩轴固定是紧跟在输出轴之后的
1 或 2 个轴"这一种调用形状——这正是本代码库里所有出现该模式的调用点
的共同结构，不做成通用任意轴收缩工具。

实现选 `np.matmul` 广播、不用 `np.tensordot`：`tensordot(D, X, ...)` 把
不依赖 cell 的小算子 `D` 当作 `a`、逐 cell 的大数组 `X` 当作 `b`，内部
按 `newaxes_b = contract轴 + notcontract轴` 转置 `b`——X 的 cell 轴不在
被收缩的轴里，会被搬到中间，产生一份完整的转置副本（`.reshape` 无法
把这种转置合并成 no-copy view）。在 364,555 cell 的生产网格上 P2 阶数
实测触发过 `Unable to allocate 1.10 GiB`（cell 轴 153,950 个 prism 撞上
J×M×V 的转置副本）——`X` 的 cell 轴原本就在最前面（数组按 cell 存储，
下游按 cell 消费），不应该为了凑 tensordot 的轴序被搬到别处再搬回来。
`np.matmul(D_flat, X_flat)`（`D_flat`: (F,K)，`X_flat`: (C,K,V)）走
批量矩阵乘广播语义，把 (C,K,V) 当作 C 个 (K,V) 矩阵、循环与共享的
(F,K) 做 gemm，全程不触碰、不转置 X 的 cell 轴，不产生这份大副本；
数学上与 `tensordot`+`moveaxis` 完全等价（已用随机张量数值验证，
两者输出最大绝对误差在 float64 机器精度量级）。
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
    ufunc 广播默认单线程执行，在 `fr_residual_inviscid.py` 的过积分
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


@njit(cache=True, parallel=True)
def _contravariant_flux_kernel(det_jacs, inv_jacs, F_phys, out) -> None:
    """`out[c,p,i,v] = det_jacs[c,p] * sum_j inv_jacs[c,p,i,j] * F_phys[c,p,j,v]`。"""
    C, P = det_jacs.shape
    V = F_phys.shape[3]
    for c in prange(C):
        for p in range(P):
            d = det_jacs[c, p]
            for i in range(3):
                a0 = d * inv_jacs[c, p, i, 0]
                a1 = d * inv_jacs[c, p, i, 1]
                a2 = d * inv_jacs[c, p, i, 2]
                for v in range(V):
                    out[c, p, i, v] = (a0 * F_phys[c, p, 0, v]
                                       + a1 * F_phys[c, p, 1, v]
                                       + a2 * F_phys[c, p, 2, v])


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
    标量对流的共享逆变质量通量（`core/turbulence/transport.py::
    precompute_scalar_convection_geometry`）。等价性回归测试见
    tests/unit/test_perf_fusion_kernels.py。
    """
    det_c = np.ascontiguousarray(det_jacs)
    inv_c = np.ascontiguousarray(inv_jacs)
    F_c = np.ascontiguousarray(F_phys)
    out = np.empty_like(F_c)
    _contravariant_flux_kernel(det_c, inv_c, F_c, out)
    return out


@njit(cache=True, parallel=True)
def _grad_to_physical_kernel(grad_comp, inv_jacs, out) -> None:
    """`out[c,s,v,n] = sum_m grad_comp[c,s,m,v] * inv_jacs[c,s,m,n]`。"""
    C, S, _, V = grad_comp.shape
    for c in prange(C):
        for s in range(S):
            for v in range(V):
                g0 = grad_comp[c, s, 0, v]
                g1 = grad_comp[c, s, 1, v]
                g2 = grad_comp[c, s, 2, v]
                for n in range(3):
                    out[c, s, v, n] = (g0 * inv_jacs[c, s, 0, n]
                                       + g1 * inv_jacs[c, s, 1, n]
                                       + g2 * inv_jacs[c, s, 2, n])


def grad_computational_to_physical(grad_comp: np.ndarray, inv_jacs: np.ndarray) -> np.ndarray:
    """等价于 `np.matmul(np.swapaxes(grad_comp, -1, -2), inv_jacs)`——把
    计算空间梯度按链式法则转到物理空间。

    Args:
        grad_comp: (C, S, 3, V) 计算空间梯度（3 是计算坐标方向 m）
        inv_jacs: (C, S, 3, 3) 逆 Jacobian（inv_jacs[c,s,m,n]）

    Returns:
        grad_phys: (C, S, V, 3)

    性能优化（2026-09-13，与本模块其余两处同一次真实剖析）：原
    `np.matmul(np.swapaxes(grad_comp,-1,-2), inv_jacs)` 的左操作数是
    **非连续的转置视图**，numpy 为了做批量 gemm 必须先物化一份连续副本
    （79万单元 P1、V=5 时约 760MiB），随后又是逐 (cell,SP) 的
    (V,3)@(3,3) 微型 gemm——单次仅 45 FLOPs、630 万次，纯开销主导，同样
    不随 CPU 核数并行。融合成一个 prange kernel 后既不需要转置副本、也
    不再有 gemm 调用开销。对 m 的求和顺序（m=0,1,2）与 `np.matmul` 一致，
    结果逐位相同（随机张量实测最大误差 0.0）。
    """
    g_c = np.ascontiguousarray(grad_comp)
    inv_c = np.ascontiguousarray(inv_jacs)
    C, S, _, V = g_c.shape
    out = np.empty((C, S, V, 3))
    _grad_to_physical_kernel(g_c, inv_c, out)
    return out


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
    `core/fr_solver/solver.py::_limit_blas_threads`）。4 路累加器本身对
    该判据几乎没有影响（3.3470e-6 -> 3.3394e-6），保留是因为上面 (a)(b)
    两条理由本身成立，**不是**那次失败的修复手段。
    """
    C = X_flat.shape[0]
    K = X_flat.shape[1]
    V = X_flat.shape[2]
    F = D_flat.shape[0]
    K4 = (K // 4) * 4
    for c in prange(C):
        for f in range(F):
            for v in range(V):
                s0 = 0.0
                s1 = 0.0
                s2 = 0.0
                s3 = 0.0
                for k in range(0, K4, 4):
                    s0 += D_flat[f, k] * X_flat[c, k, v]
                    s1 += D_flat[f, k + 1] * X_flat[c, k + 1, v]
                    s2 += D_flat[f, k + 2] * X_flat[c, k + 2, v]
                    s3 += D_flat[f, k + 3] * X_flat[c, k + 3, v]
                tail = 0.0
                for k in range(K4, K):
                    tail += D_flat[f, k] * X_flat[c, k, v]
                out[c, f, v] = (s0 + s1) + (s2 + s3) + tail


def _contract_shared(D_flat: np.ndarray, X_flat: np.ndarray) -> np.ndarray:
    """`_contract_shared_kernel` 的分配 + 连续性包装。

    性能优化（2026-09-13，用户反馈"每步耗时过长、且计算效率对 CPU 核数
    不敏感"后的真实剖析结论）：此前两个公开入口都用 `np.matmul(D, X)` 的
    批量矩阵乘广播语义。数学上正确，但在本项目的真实形状上是最坏情况：
    P1 是逐 cell 的 (8,24)@(24,5)、P2 是 (27,81)@(81,5)——单次 gemm 只有
    几百到几千 FLOPs，79万个 cell 的批量循环几乎全是 BLAS 调用开销，且
    **numpy 对批量维度不做任何并行**（OpenBLAS 只会在单个 gemm 内部尝试
    多线程，这种尺寸下线程化只会更慢），所以整条体积项/梯度链路的耗时
    与 numba 线程数、与 CPU 核数完全无关——这正是用户观察到"加核不提速"
    的直接原因之一。
    换成按 cell `prange` 的显式三重循环后：79万单元 P1 形状实测
    163.5ms -> 34.6ms（4.7x）、P2 形状 1166ms -> 334ms（3.5x），且
    1->8 线程有 7.7 倍真实扩展性（277.9ms -> 35.9ms，16 线程 34.3ms 起
    受内存带宽限制）。数值上 P1 形状与原 `np.matmul` 路径**逐位完全相同**
    （随机张量实测最大误差 0.0），P2 形状 7.7e-16 相对误差（求和顺序不同
    导致的浮点重结合，与本模块文档记载的 einsum->matmul 那次替换是同一
    类、同一量级的现象）。
    """
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
        D: (F, J, M) —— 不依赖 cell 的共享算子（如 D_3d_tet/prism、D_fine）
        X: (C, J, M, V) —— 逐 cell 的场

    Returns:
        (C, F, V)
    """
    F, J, M = D.shape
    C, _, _, V = X.shape
    # J,M 相邻，合并成 K 轴是 no-copy view（小算子 D 同理）。
    return _contract_shared(D.reshape(F, J * M), X.reshape(C, J * M, V))

#: 过积分（去混叠）分块大小，与 `fr_residual/inviscid.py` 的强形式分支
#: 同一取值，理由见该处 P2 OOM 修复说明。
OVERINT_CHUNK_CELLS = 32768


def get_overintegration_context(mesh, ops):
    """取过积分所需的全部算子与细点度量；任一缺失则返回 None（调用方
    退回 coarse 路径）。

    三个消费方共用同一份：`fr_residual/inviscid.py`（无粘体积项，本项目
    唯一一直开着的）、`core/turbulence/transport.py`（k/omega 对流与
    扩散体积项）、`fr_residual/viscous_flux.py`（粘性体积项）。

    缺失的唯一正常情形是 `order == 0`：P0 是分片常数场、多项式导数恒为
    零，没有可去混叠的内容，`jacobians_fine` 与 overint 算子都不构造。

    Returns:
        dict 或 None。dict 只含 `segs`——
        `[(lo, hi, n_fine, det_fine, inv_fine, c2f, D_fine, f2c), ...]`
        两段（棱柱在前、四面体在后），与"棱柱在前"的单元存储顺序一致。
        每段自带**自己的** `n_fine` 与已按 `(seg_len, n_fine, ...)` 切好的
        细点度量。

    ## 为什么每段各自带 n_fine（2026-09-17 改动）

    此前返回的是**共享**的 `n_fine`/`det_fine`/`inv_fine`，因为棱柱与
    四面体的过积分算子都被填充到同一个 `(over_order+1)^3` 宽度。但 native
    四面体在 over_order 下只有 `(oo+1)(oo+2)(oo+3)/6` 个**真实**细点
    （P1: 10 vs 27，P2: 20 vs 64），填充槽位恒为零、对结果零贡献，却让
    整条过积分链在空点上白算——其中 `D_fine` 的收缩是 O(n_fine^2)，P2 上
    是 10.2 倍的无效 FLOPs。实测去掉细轴填充后 P1 加速 3.04x、P2 加速
    4.63x，最大相对差 1.4e-16 / 0.0（见 `fr/operators.py` 那段说明）。

    **共享键被刻意移除**（不做向后兼容别名）：漏改的消费点会直接
    `KeyError`，而不是静默用棱柱的 `n_fine` 去切四面体段的度量——后者
    会产生形状不匹配或（更糟）安静的错误答案。

    ## 四面体段的细点度量从**第 0 列广播**（2026-09-17 第二次改动）

    直边四面体的 Jacobian **逐单元为常数**（`compute_native_tet_jacobian`
    不依赖参考点位置），`compute_native_tet_jacobians` 把同一个常数写满
    该单元的全部细点槽位。所以这一段的度量只有 `n_tet` 个真实数值，
    取第 0 列再广播到 `n_fine_tet` 列与"在 native 真实细点上求值"恒等。

    第一版是"切前 `n_fine_tet` 列"，那等价但带来一条**多余的约束**
    `n_fine_tet <= n_fine_prism`——四面体的过积分阶数因此被棱柱的布局宽度
    夹住。实测代价在 P3 上很大：去混叠误差在 `oo = 2*order` 处断崖式下降
    （P3 oo=3 6.26e-2 / oo=4 2.66e-2 / oo=5 3.37e-3 / **oo=6 4.80e-6**），
    被夹到 5 只拿到 18.6 倍中的 13000 倍。改成广播后这条约束彻底消失，
    P3 可以直接取理想的 oo=6。

    **这条广播依赖的不变量在网格构造时被显式校验**（
    `high_order_mesh_order.build_order_geometry` 里的
    `_verify_tet_fine_metric_is_cellwise_constant`），不是默默假设——
    将来若引入曲边四面体，那条校验会当场失败而不是静默给出错误度量。

    顺带的事实（如实记录，不是本函数的问题）：`jacobians_fine` 里属于
    四面体的那 `n_tet * n_fine_prism * 10` 个浮点数现在对过积分路径完全
    冗余，只需要每单元 10 个。plate_demo（363,392 单元）P2 下这是约
    1.5 GB vs 24 MB。把那块压缩掉是一项独立的**内存**优化，要改
    `jacobians_fine` 的全局形状与 GPU/MPI/分布式加载共 9 处消费点，
    与本函数要解决的精度问题无关。
    """
    if getattr(mesh, "jacobians_fine", None) is None:
        return None
    for name in ("overint_interp_c2f_prism", "overint_D_fine_prism",
                 "overint_restrict_f2c_prism", "overint_interp_c2f_tet",
                 "overint_D_fine_tet", "overint_restrict_f2c_tet"):
        if getattr(ops, name, None) is None:
            return None
    n_fine_prism = mesh.n_sps_per_cell_fine
    n_cells = mesh.n_cells
    n_prism = mesh.n_prism_cells
    det_all = mesh.jacobians_fine["det_jacs"].reshape(n_cells, n_fine_prism)
    inv_all = mesh.jacobians_fine["inv_jacs"].reshape(n_cells, n_fine_prism, 3, 3)

    # n_fine_tet 从**矩阵自身的形状**推导（`overint_D_fine_tet` 现在是
    # (n_fine_tet, n_fine_tet, 3)），不读 `ops.overint_n_fine_tet` 那个
    # 字段——矩阵才是单一事实来源，从形状推导在结构上不可能与它不同步。
    # （那个字段仍然保留，供启动日志/诊断使用。）
    n_fine_tet = int(ops.overint_D_fine_tet.shape[0])
    _declared = int(getattr(ops, "overint_n_fine_tet", 0) or 0)
    if _declared and _declared != n_fine_tet:
        raise ValueError(
            f"ops.overint_n_fine_tet={_declared} 与 overint_D_fine_tet 的"
            f"形状 {ops.overint_D_fine_tet.shape} 不一致——算子构造有 bug，"
            f"不静默采用其中一个"
        )
    # 四面体段：逐单元常数 -> 取第 0 列广播到 n_fine_tet 宽。
    # `np.broadcast_to` 是零拷贝视图；消费方在分块时自己
    # `np.ascontiguousarray` 物化当前块（numba kernel 需要连续输入）。
    n_tet = n_cells - n_prism
    det_tet = np.broadcast_to(det_all[n_prism:, :1], (n_tet, n_fine_tet))
    inv_tet = np.broadcast_to(
        inv_all[n_prism:, :1], (n_tet, n_fine_tet, 3, 3))

    return dict(
        segs=(
            (0, n_prism, n_fine_prism,
             det_all[:n_prism], inv_all[:n_prism],
             ops.overint_interp_c2f_prism, ops.overint_D_fine_prism,
             ops.overint_restrict_f2c_prism),
            (n_prism, n_cells, n_fine_tet,
             det_tet, inv_tet,
             ops.overint_interp_c2f_tet, ops.overint_D_fine_tet,
             ops.overint_restrict_f2c_tet),
        ),
    )
