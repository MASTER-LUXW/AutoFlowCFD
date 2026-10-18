"""AutoFlowCFD V2.0 - 平衡壁面模型剪应力对平均流单元块 Jacobian 的贡献。

壁面模型（WMLES，或 RANS 自动壁面处理，`core/turbulence/wall_model/`）给 owner 单元的动量与能量残差加

    dU/dt[c, s, 1:5] += lift[k, s] / det[s] * area[k] * F_k(U_m)，   F = [tau, tau . u_wall]，U_m = E_m[k] . U[c]

（逐壁面通量点 k，`wall_model/stress.py::wall_stress_residual`；能量分量是移动壁面上剪应力做的功，静止壁为 0）。
它只依赖 owner 单元自身的解点，所以只进对角块：

    J[c][s, 1+i, t, b] += -lift[k, s] / det[s] * area[k] * (dF_i/dU_m,b) * E_m[k, t]，  dF = [dtau, u_wall . dtau]

（`R = -dU/dt`），启用低马赫预处理时再左乘行解点的 `Gamma_s`。`dtau/dU_m` 是解析导数
（`wall_model/law.py::wall_shear_stress_linearization`），与残差同一组匹配点几何（`EquilibriumWallModel.matching`）。
"""

import numpy as np

#: 每批壁面点的块元素数上限（float64 临时量约 8 字节/元素，256 MiB）
_CHUNK_ELEMENTS = 1 << 25


def add_wall_stress_blocks(ctx, K_prism, K_tet, U5, row_gamma, n_prism: int, npr: int, nte: int) -> None:
    """把壁面剪应力的导数加到收尾后的对角块上（`CellBlockJacobian` 布局 `(n, nr, 5, nr, 5)`，就地）。

    Args:
        ctx: `MeanFlowLinearization`（`ctx.wall_model` 非 None）
        U5: `(n_cells, n_sps, 5)` 守恒变量
        row_gamma: `row_gamma(cells, n) -> (len(cells), n, 5, 5)` 行解点的 `Gamma`；未启用低马赫预处理时为 None
    """
    m, dtau = ctx.wall_model.traction_jacobian(np, ctx.mesh, ctx.flat_geometry(), ctx.ghost_provider, U5, ctx.mu)
    if m is None:
        return
    owner = np.asarray(m.owner, dtype=np.int64)
    dflux = np.concatenate([dtau, np.einsum('ki,kib->kb', m.wall_velocity, dtau)[:, None, :]], axis=1)   # (m, 4, 5)
    for K, lo, n, pts in ((K_prism, 0, npr, np.flatnonzero(owner < n_prism)),
                          (K_tet, n_prism, nte, np.flatnonzero(owner >= n_prism))):
        chunk = max(1, _CHUNK_ELEMENTS // (n * n * 25))
        for p0 in range(0, pts.size, chunk):
            p = pts[p0:p0 + chunk]
            cells = owner[p]
            coef = -(m.lift[p, :n] * m.inv_det[p, :n]) * m.area[p, None]                     # (k, n)
            inner = (coef[:, :, None, None, None] * dflux[p][:, None, :, None, :]
                     * m.E_m[p, :n][:, None, None, :, None])                                 # (k, n, 4, n, 5)
            if row_gamma is None:
                out = np.zeros((p.size, n, 5, n, 5))
                out[:, :, 1:5] = inner
            else:
                out = np.einsum('ksai,ksitb->ksatb', row_gamma(cells, n)[..., 1:5], inner, optimize=True)
            # 同一单元可有多个壁面通量点：按单元累加
            np.add.at(K, cells - lo, out.astype(K.dtype))
