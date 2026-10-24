"""平均流单元级粗块（`fr_residual/jacobian/coarse.py`，块 Jacobi 档的粗校正用）：界面核里流式约化的结果与
"先装配完整耦合块（块 ILU 档的那一份）、再按限制权重约化"一致，对角块不受影响。"""

import dataclasses

import numpy as np
import pytest

from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
from autoflowcfd.core.fr_residual.jacobian import MeanFlowLinearization, assemble_mean_flow_blocks
from autoflowcfd.fr.native_padding import real_sps_per_cell
from autoflowcfd.fr.operators import generate_fr_operators
from tests.unit.test_mean_flow_analytic_jacobian import MACH_REF, MU, _Residual, _av_field, _channel, _state


def _weights(mesh, seed):
    npr, nte = real_sps_per_cell(int(mesh.order))
    n_real = np.where(np.arange(mesh.n_cells) < mesh.n_prism_cells, npr, nte)
    real = np.arange(mesh.n_sps_per_cell)[None, :] < n_real[:, None]
    w = np.random.default_rng(seed).uniform(0.2, 1.8, real.shape) * 1e-6
    return np.where(real, w, 0.0), n_real


def _reference(bp, bt, coupling, w, n_real, n_prism):
    """完整块按权重约化：对角 `sum_s w_s sum_t J_cc[s,:,t,:]`，单元对同理。"""
    diag = np.zeros((n_real.size, 5, 5))
    for c in range(n_real.size):
        n = n_real[c]
        B = (bp[c] if c < n_prism else bt[c - n_prism]).reshape(n, 5, n, 5).astype(np.float64)
        diag[c] = np.einsum("s,satb->ab", w[c, :n], B)
    pairs = {}
    for g in coupling.groups:
        for k in range(g.rows.size):
            r, y = int(g.rows[k]), int(g.cols[k])
            B = g.blocks[k].reshape(n_real[r], 5, n_real[y], 5).astype(np.float64)
            pairs[r, y] = np.einsum("s,satb->ab", w[r, :n_real[r]], B)
    return diag, pairs


def _check(mesh, ops, provider, U, *, low_mach, nu_av=None, flat=None):
    res = _Residual(mesh, ops, provider, None, low_mach, flat, nu_av)
    r0 = res(U.reshape(-1, 5)).reshape(U.shape)
    ctx = MeanFlowLinearization(mesh=mesh, ops=ops, ghost_provider=provider, mu=MU, mach_ref=MACH_REF,
                                low_mach=low_mach, nu_av=nu_av, flat=flat)
    w, n_real = _weights(mesh, 3)
    bp, bt, coupling = assemble_mean_flow_blocks(ctx, U, residual=r0, want_coupling=True)
    bp2, bt2, coarse = assemble_mean_flow_blocks(ctx, U, residual=r0, coarse_weights=w)
    np.testing.assert_array_equal(bp2, bp)
    np.testing.assert_array_equal(bt2, bt)
    diag_ref, pairs_ref = _reference(bp, bt, coupling, w, n_real, mesh.n_prism_cells)
    assert pairs_ref, "测试网格必须有面邻居耦合"
    scale = max(np.abs(diag_ref).max(), max(np.abs(b).max() for b in pairs_ref.values()))
    np.testing.assert_allclose(coarse.diag, diag_ref, rtol=0, atol=2e-6 * scale)
    got = {(int(r), int(y)): coarse.blocks[k] for k, (r, y) in enumerate(zip(coarse.rows, coarse.cols))}
    assert set(got) == set(pairs_ref)
    worst = max(np.abs(got[k] - pairs_ref[k]).max() for k in got)
    assert worst <= 2e-6 * scale, worst / scale
    assert np.all(coarse.rows != coarse.cols)


@pytest.mark.parametrize("kind", ["tet", "prism"])
@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("low_mach", [True, False])
def test_streamed_coarse_blocks_match_reduced_full_coupling(kind, order, low_mach):
    mesh, provider = _channel(kind, order)
    _check(mesh, generate_fr_operators(order), provider, _state(mesh, 70 + order), low_mach=low_mach)


@pytest.mark.parametrize("kind", ["tet", "prism"])
def test_streamed_coarse_blocks_include_artificial_viscosity_coupling(kind):
    """人工粘性的标量耦合块（`-L_nu`，经 `Gamma_s` 作用到每个守恒变量）同样约化进单元对粗块。"""
    mesh, provider = _channel(kind, 2)
    _check(mesh, generate_fr_operators(2), provider, _state(mesh, 81), low_mach=True, nu_av=_av_field(mesh, 9))


def test_streamed_coarse_blocks_on_mixed_mesh_with_all_boundary_types():
    from tests.unit.test_boundary_ghost_states_batched_crosscheck import _build_multi_bc_provider
    from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh
    from tests.unit.test_mean_flow_analytic_jacobian import P_INF, RHO, U_INF

    mesh = _build_synthetic_mixed_mesh(2)
    ops = generate_fr_operators(2)
    provider = _build_multi_bc_provider(get_flat_face_geometry(mesh, ops), np.random.default_rng(2))
    rng = np.random.default_rng(4)
    shape = (mesh.n_cells, mesh.n_sps_per_cell)
    Q = np.stack([RHO * (1 + 0.03 * rng.standard_normal(shape)), U_INF + 3 * rng.standard_normal(shape),
                  3 * rng.standard_normal(shape), 3 * rng.standard_normal(shape),
                  P_INF * (1 + 0.01 * rng.standard_normal(shape))], axis=-1)
    U = Q.copy()
    U[..., 1:4] *= Q[..., :1]
    U[..., 4] = Q[..., 4] / 0.4 + 0.5 * Q[..., 0] * (Q[..., 1:4] ** 2).sum(-1)
    _check(mesh, ops, provider, U, low_mach=True)


def test_streamed_coarse_blocks_with_mixed_split_face_partner():
    """混合拆分面的配对边界面 owner 是另一个单元时，该来源是一个独立的单元对（`cross_slots` 的来源 2）。"""
    mesh, provider = _channel("prism", 1)
    ops = generate_fr_operators(1)
    flat = get_flat_face_geometry(mesh, ops)
    own = np.asarray(flat.owner_cell)
    bnd = np.asarray(flat.is_boundary).astype(bool)
    prim = np.asarray(flat.owner_is_primary).astype(bool)
    chosen = None
    for f in np.nonzero(~bnd & prim & np.asarray(flat.neighbor_is_primary).astype(bool))[0]:
        partners = np.nonzero(bnd & prim & (own != own[f]))[0]
        if partners.size:
            chosen = (int(f), int(partners[0]))
            break
    assert chosen is not None
    f, bf = chosen
    nb_partner = np.array(flat.mixed_nb_partner, copy=True)
    nb_mask = np.array(flat.mixed_nb_mask, copy=True)
    bnd_face = np.array(flat.mixed_bnd_face, copy=True)
    nb_partner[f] = bf
    nb_mask[f, : flat.n_fp // 2] = True
    bnd_face[bf] = True
    flat2 = dataclasses.replace(flat, mixed_nb_partner=nb_partner, mixed_nb_mask=nb_mask, mixed_bnd_face=bnd_face)
    _check(mesh, ops, provider, _state(mesh, 22), low_mach=True, flat=flat2)


def test_full_coupling_and_coarse_blocks_are_exclusive():
    mesh, provider = _channel("tet", 1)
    ops = generate_fr_operators(1)
    U = _state(mesh, 5)
    ctx = MeanFlowLinearization(mesh=mesh, ops=ops, ghost_provider=provider, mu=MU, mach_ref=MACH_REF, low_mach=False)
    with pytest.raises(ValueError):
        assemble_mean_flow_blocks(ctx, U, want_coupling=True, coarse_weights=_weights(mesh, 1)[0])
