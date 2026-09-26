"""AutoFlowCFD V2.0 - 湍流标量输运的体积项 numba kernel（对流散度、扩散散度）。

面上的外插、质量通量与 DG 提升在 `transport/face_frames.py`（2026-09-26 改为
两侧各自坐标系，原先放在本文件的 owner 顺序版本已删除，理由见那里）。
"""

from numba import njit, prange


@njit(cache=True, parallel=True)
def scalar_convection_volume_kernel(phi, rho_u_tilde, op_D, out) -> None:
    """标量对流体积项散度：`out[c,s] = sum_m sum_j op_D[s,j,m] * phi[c,j] * rho_u_tilde[c,j,m]`。

    性能优化（2026-09-13，用户反馈"每步耗时过长、对 CPU 核数不敏感"后的
    真实剖析结论，见 `compute_scalar_convection_residual` 文档"共享几何量"
    一节）：此前体积项在 Python 层按块做
    `adj_j_chunk = det*inv_jacs` -> `rho_u_phi = rho*vel*phi` ->
    `np.matmul(adj_j_chunk, rho_u_phi)` -> 3 次 `np.tensordot` 累加，
    每一步都物化一份块大小的中间数组、且 matmul 那步是逐点 3x3@3x1 的
    批量微型 gemm（不随核数并行）。

    这里把整条链压进一个按 cell `prange` 的 kernel，配合调用方预先算好的
    **与标量无关**的逆变质量通量 `rho_u_tilde`（= adj(J) @ (rho*u)，k 和
    omega 两次调用共享同一份，见调用方文档）：
      F_tilde[c,j,m] = phi[c,j] * rho_u_tilde[c,j,m]
    （数学恒等式——phi 在该点是标量，可以从度量乘法里提出来）
    工作集只有逐 cell 的几十个 double，留在 L1 内；无任何大中间数组。

    Args:
        phi: (n_cells, n_sps) 标量场（k 或 omega）
        rho_u_tilde: (n_cells, n_sps, 3) 逆变质量通量 adj(J) @ (rho*u)
        op_D: (n_sps, n_sps, 3) 该单元类型的微分矩阵
        out: (n_cells, n_sps) 输出（调用方按 prism/tet 段分别传入切片）
    """
    C, S = phi.shape
    for c in prange(C):
        for s in range(S):
            tot = 0.0
            for m in range(3):
                acc = 0.0
                for j in range(S):
                    acc += op_D[s, j, m] * phi[c, j] * rho_u_tilde[c, j, m]
                tot += acc
            out[c, s] = tot


@njit(cache=True, parallel=True)
def scalar_volume_divergence_kernel(tilde, op_D, out) -> None:
    """逆变通量的体积项散度：`out[c,s] = sum_m sum_j op_D[s,j,m] * tilde[c,j,m]`。

    与 `scalar_convection_volume_kernel` 的区别只是没有 phi 因子——对流项
    可以把标量提到度量乘法外面（见该 kernel 文档），扩散项的逆变通量
    `adj(J) @ (Gamma*grad(phi))` 里 Gamma 与 grad(phi) 都随标量变化、提不
    出来，只能先算好 `tilde` 再收缩。

    性能优化（2026-09-13，与对流项同一次剖析）：原实现是 Python 层按块
    `adj_j = det*inv_jacs` -> `np.matmul(adj_j, G_phys[...,None])` ->
    3 次 `np.tensordot` 累加，逐块物化 3 份中间数组、且 matmul 那步是逐点
    3x3@3x1 微型 gemm（不随核数并行）。

    Args:
        tilde: (n_cells, n_sps, 3) 逆变通量（调用方用
            `volume_contract.contravariant_flux_from_metric` 算好）
        op_D: (n_sps, n_sps, 3) 该单元类型的微分矩阵
        out: (n_cells, n_sps) 输出（调用方按 prism/tet 段分别传入切片）
    """
    C, S = out.shape
    for c in prange(C):
        for s in range(S):
            tot = 0.0
            for m in range(3):
                acc = 0.0
                for j in range(S):
                    acc += op_D[s, j, m] * tilde[c, j, m]
                tot += acc
            out[c, s] = tot
