"""AutoFlowCFD V2.0 - 问题单元人工粘性对平均流单元块 Jacobian 的贡献。

人工粘性（`core/fr_operators/artificial_viscosity`）给每个守恒量加 `dU_k/dt += L_nu U_k`，
`L_nu = div(nu grad .)` 是冻结系数 `nu`（按步冻结，与残差同一份）的标量扩散算子，边界全为
齐次 Neumann（`turbulence/transport/diffusion.py::compute_scalar_diffusion_residual` 不传
壁面掩码）。它对守恒变量线性、变量之间不耦合，所以 Newton 块（`R = Gamma (-dU/dt)`）里

    B[s,a,t,d] += Gamma_s[a,d] * (-L_nu)[s,t]

不经过 `dQ/dU`。`-L_nu` 由湍流 k-omega 与人工粘性共用的标量输运装配
（`turbulence/jacobian/scalar_blocks.py`）以零源项、零质量通量、`rho = 1` 给出——与残差
同一个离散，只有一份 Jacobian 实现。
"""

import numpy as np

from .coupling import CouplingBlocks, CouplingGroup


def artificial_diffusion_blocks(mesh, ops, flat, nu, want_coupling: bool = False):
    """`-L_nu` 的标量块：`(prism (n_prism, npr, npr), tet (n_tet, nte, nte)[, CouplingBlocks])`。"""
    # 函数内导入：标量输运装配依赖本包的 coupling 模块（模块级导入成环）
    from autoflowcfd.core.turbulence.jacobian.pointwise import N_TURB_INPUTS
    from autoflowcfd.core.turbulence.jacobian.scalar_blocks import (
        assemble_scalar_pair_blocks, convection_ghost_affine,
    )
    from autoflowcfd.core.turbulence.transport.faces import boundary_diffusion_targets

    n_cells, n_sps = int(mesh.n_cells), int(mesh.n_sps_per_cell)
    nu = np.ascontiguousarray(np.asarray(nu, dtype=np.float64).reshape(n_cells, n_sps))
    zero_faces = np.zeros(flat.n_faces, dtype=bool)
    m = np.zeros(np.asarray(flat.owner_adj_row_exact).shape[:2])
    ghost_o, a_o = convection_ghost_affine(flat, "owner", m, zero_faces, zero_faces, zero_faces)
    ghost_n, a_n = convection_ghost_affine(flat, "neighbor", m, zero_faces, zero_faces, zero_faces)
    diff = {}
    for frame in ("owner", "neighbor"):
        b, d, t = boundary_diffusion_targets(np, flat, frame)
        diff[frame] = (np.ascontiguousarray(b), np.ascontiguousarray(np.stack([d, d])),
                       np.ascontiguousarray(np.stack([t, t])))
    out = assemble_scalar_pair_blocks(
        mesh, ops, flat, np.zeros((n_cells, n_sps, 2)), np.ascontiguousarray(np.stack([nu, nu], axis=-1)),
        np.zeros((n_cells, n_sps, 2, N_TURB_INPUTS)), np.zeros((n_cells, n_sps, 2, N_TURB_INPUTS)),
        np.ones((n_cells, n_sps)), np.zeros((n_cells, n_sps, 3)), np.zeros((n_cells, n_sps, 3)), m, m,
        (ghost_o, np.ascontiguousarray(a_o), ghost_n, np.ascontiguousarray(a_n)), diff, want_coupling)

    def first_var(blocks):
        n_pairs, rows, cols = blocks.shape
        return np.ascontiguousarray(blocks.reshape(n_pairs, rows // 2, 2, cols // 2, 2)[:, :, 0, :, 0])

    scalar = (first_var(out[0]), first_var(out[1]))
    if not want_coupling:
        return scalar
    groups = [CouplingGroup(row_is_prism=g.row_is_prism, col_is_prism=g.col_is_prism, rows=g.rows, cols=g.cols,
                            blocks=first_var(g.blocks)) for g in out[2].groups]
    return scalar + (CouplingBlocks(groups=groups),)


def add_scalar_kron_gamma(blocks, scalar, gamma, use_gamma: bool):
    """`blocks[k, s, a, t, d] += G_s[a, d] * scalar[k, s, t]`（`G` 为单位阵或逐行解点的 `gamma[k, s]`）。

    Args:
        blocks: `(n, nr, 5, ny, 5)` float32，就地更新。
        scalar: `(n, nr, ny)`。
        gamma: `(n, nr, 5, 5)`（`use_gamma` 时）。
    """
    if use_gamma:
        blocks += (gamma[:, :, :, None, :] * scalar[:, :, None, :, None]).astype(blocks.dtype)
    else:
        for a in range(5):
            blocks[:, :, a, :, a] += scalar.astype(blocks.dtype)


def add_scalar_coupling_kron_gamma(coupling: CouplingBlocks, scalar: CouplingBlocks, gamma, use_gamma: bool):
    """把 `-L_nu` 的耦合块并入平均流耦合块：同类（行/列单元类型）组内按 (行, 列) 单元对匹配。

    两者都来自同一组面邻居；平均流组另含混合拆分面配对等额外单元对，所以标量块的每个单元
    对都必须在平均流组里找到，找不到说明两份布局不一致，直接报错。

    Args:
        gamma: `(n_cells, n_sps, 5, 5)` 逐解点 `Gamma`（`use_gamma` 时），按行单元取。
    """
    by_type = {(g.row_is_prism, g.col_is_prism): g for g in coupling.groups}
    for s in scalar.groups:
        g = by_type.get((s.row_is_prism, s.col_is_prism))
        n_cells_key = int(max(s.rows.max(initial=0), s.cols.max(initial=0),
                              0 if g is None else max(g.rows.max(initial=0), g.cols.max(initial=0)))) + 1
        if g is None:
            raise RuntimeError("人工粘性耦合块的单元对类型在平均流耦合块里不存在")
        key_g = g.rows.astype(np.int64) * n_cells_key + g.cols
        key_s = s.rows.astype(np.int64) * n_cells_key + s.cols
        pos = np.searchsorted(key_g, key_s)
        if np.any(pos >= key_g.size) or np.any(key_g[np.minimum(pos, key_g.size - 1)] != key_s):
            raise RuntimeError("人工粘性耦合块的单元对不在平均流耦合块里（两份面邻居布局不一致）")
        n_pairs, nr5, ny5 = g.blocks.shape
        nr, ny = nr5 // 5, ny5 // 5
        target = g.blocks.reshape(n_pairs, nr, 5, ny, 5)
        sub = target[pos]
        add_scalar_kron_gamma(sub, s.blocks, gamma[s.rows, :nr] if use_gamma else None, use_gamma)
        target[pos] = sub
