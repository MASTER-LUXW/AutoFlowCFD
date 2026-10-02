"""平均流解析单元块 Jacobian（`core/fr_residual/jacobian`）逐块对照着色有限差分装配。

判据的依据：残差对单元自身自由度的依赖只经本单元与面邻居（梯度是单元内局部
梯度，没有 BR1 提升，模板是距离 1），距离 1 着色下差分装配出的 `J_cc` 是精确
的（误差只有差分截断 `O(sqrt(eps))`）。解析块应与之吻合到同一量级；逐子块
（5x5 变量对）只在差分本身能分辨的量级上比较（比块内最大元小 1e-6 以下的元素
差分已是噪声）。

参考残差在测试里按平均流 Newton 残差的定义显式拼出：
`R = Gamma(Q) * (-(无粘 + 粘性 + 人工粘性))`，涡粘与人工扩散系数冻结——与 `fr_solver/step.py` 同一约定，
但不依赖求解器对象，覆盖面更直接（边界类型、混合拆分面、涡粘）。
"""

import dataclasses

import numpy as np
import pytest

from autoflowcfd.core.fr_operators.artificial_viscosity import artificial_diffusion_residual
from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
from autoflowcfd.core.fr_operators.kernels import resolve_ausm_precond_mode
from autoflowcfd.core.fr_residual.inviscid import compute_inviscid_residual_fr, conserved_to_primitive
from autoflowcfd.core.fr_residual.jacobian import MeanFlowLinearization, assemble_mean_flow_blocks
from autoflowcfd.core.fr_residual.viscous_flux import compute_viscous_residual_fr
from autoflowcfd.core.time_integration.implicit.cell_blocks import CellBlockJacobian
from autoflowcfd.core.turbulence.transport import compute_scalar_diffusion_residual
from autoflowcfd.core.time_integration.implicit.coloring import greedy_cell_coloring
from autoflowcfd.core.utils.preconditioning import apply_low_mach_preconditioner
from autoflowcfd.fr.native_padding import real_sps_per_cell
from autoflowcfd.fr.operators import generate_fr_operators
from tests.validation._channel_mesh import (
    build_channel_mesh, build_channel_mesh_prism, build_face_exact_ghost_provider,
)

RHO, U_INF, P_INF = 1.225, 30.0, 101325.0
LX, H, LZ = 0.4, 0.1, 0.08
MU = 1.8e-5
MACH_REF = U_INF / np.sqrt(1.4 * P_INF / RHO)
SCALES = np.array([RHO, RHO * U_INF, RHO * U_INF, RHO * U_INF, P_INF / 0.4])


def _channel(kind, order):
    build = build_channel_mesh if kind == "tet" else build_channel_mesh_prism
    mesh = build(order, 3, 3, 2, LX, H, LZ)
    bc = {n: {"type": "SYMMETRY"} for n in ("z_min", "z_max")}
    for n in ("wall_bottom", "wall_top"):
        bc[n] = {"type": "WALL", "is_no_slip": True, "wall_velocity": [0.0, 0.0, 0.0]}
    for n in ("x_min", "x_max"):
        bc[n] = {"type": "FARFIELD", "Q_free": [RHO, U_INF, 0.0, 0.0, P_INF]}
    return mesh, build_face_exact_ghost_provider(mesh, LX, H, LZ, bc)


def _state(mesh, seed):
    """带壁面剪切与随机扰动的非平凡状态（守恒变量）。"""
    rng = np.random.default_rng(seed)
    y = np.asarray(mesh.sps_coords)[..., 1]
    shape = y.shape
    u = U_INF * 4.0 * (y / H) * (1.0 - y / H)
    Q = np.empty(shape + (5,))
    Q[..., 0] = RHO * (1.0 + 0.02 * rng.standard_normal(shape))
    Q[..., 1] = u * (1.0 + 0.05 * rng.standard_normal(shape))
    Q[..., 2] = 2.0 * rng.standard_normal(shape)
    Q[..., 3] = 2.0 * rng.standard_normal(shape)
    Q[..., 4] = P_INF * (1.0 + 0.01 * rng.standard_normal(shape))
    U = np.empty_like(Q)
    U[..., 0] = Q[..., 0]
    U[..., 1:4] = Q[..., :1] * Q[..., 1:4]
    U[..., 4] = Q[..., 4] / 0.4 + 0.5 * Q[..., 0] * (Q[..., 1:4] ** 2).sum(-1)
    return U


class _Residual:
    def __init__(self, mesh, ops, provider, mu_t, low_mach, flat, nu_av=None):
        self.mesh, self.ops, self.provider, self.mu_t = mesh, ops, provider, mu_t
        self.low_mach, self.flat, self.nu_av = low_mach, flat, nu_av
        self.shape = (mesh.n_cells, mesh.n_sps_per_cell, 5)

    def raw(self, U):
        inv = compute_inviscid_residual_fr(U, self.mesh, self.ops, boundary_ghost_provider=self.provider,
                                           mach_ref=MACH_REF, flat_face_override=self.flat)
        vis = compute_viscous_residual_fr(U, self.mesh, self.ops, MU, 0.72, mu_t_field=self.mu_t,
                                          boundary_ghost_provider=self.provider,
                                          flat_face_override=self.flat)
        if self.nu_av is not None:
            vis = vis + artificial_diffusion_residual(
                U, self.nu_av, lambda phi, gamma: compute_scalar_diffusion_residual(phi, gamma, self.mesh, self.ops))
        return -(inv + vis)

    def __call__(self, u_flat):
        U = u_flat.reshape(self.shape)
        r = self.raw(U)
        if self.low_mach:
            r = apply_low_mach_preconditioner(r, conserved_to_primitive(U), MACH_REF)
        return r.reshape(-1, 5)


def _compare(mesh, ops, provider, U, *, mu_t=None, low_mach=True, flat=None, nu_av=None):
    order = int(mesh.order)
    res = _Residual(mesh, ops, provider, mu_t, low_mach, flat, nu_av)
    u0 = U.reshape(-1, 5)
    r0 = res(u0)
    npr, nte = real_sps_per_cell(order)
    fl = flat if flat is not None else get_flat_face_geometry(mesh, ops)
    colors = greedy_cell_coloring(fl.owner_cell, fl.neighbor_cell, mesh.n_cells)
    cell_is_prism = np.arange(mesh.n_cells) < mesh.n_prism_cells
    fd = CellBlockJacobian(res, u0, r0, SCALES, n_sps=mesh.n_sps_per_cell, cell_is_prism=cell_is_prism,
                           n_real_prism=npr, n_real_tet=nte, colors=colors)
    ctx = MeanFlowLinearization(mesh=mesh, ops=ops, ghost_provider=provider, mu=MU, mach_ref=MACH_REF,
                                precond_mode=resolve_ausm_precond_mode(), low_mach=low_mach, mu_t=mu_t, flat=flat,
                                nu_av=nu_av)
    bp, bt = assemble_mean_flow_blocks(ctx, U, residual=r0.reshape(U.shape))
    worst = 0.0
    for A, B in ((bp, fd.blocks_prism), (bt, fd.blocks_tet)):
        if B.shape[0] == 0:
            continue
        A, B = A.astype(np.float64), B.astype(np.float64)
        n = A.shape[1] // 5
        scale = np.abs(B).max(axis=(1, 2))
        assert (np.abs(A - B).max(axis=(1, 2)) <= 2e-5 * scale).all()
        # 逐 5x5 变量对子块：只在差分能分辨的量级上比较
        a5, b5 = A.reshape(-1, n, 5, n, 5), B.reshape(-1, n, 5, n, 5)
        sub_scale = np.abs(b5).max(axis=(1, 3))
        resolved = sub_scale > 1e-5 * scale[:, None, None]
        err = np.abs(a5 - b5).max(axis=(1, 3))
        worst = max(worst, float((err[resolved] / sub_scale[resolved]).max()))
    assert worst < 5e-3, f"子块最大相对误差 {worst:.2e}"
    return worst


@pytest.mark.parametrize("kind", ["tet", "prism"])
@pytest.mark.parametrize("order", [1, 2])
def test_blocks_match_colored_finite_difference(kind, order):
    mesh, provider = _channel(kind, order)
    ops = generate_fr_operators(order)
    _compare(mesh, ops, provider, _state(mesh, 3 + order))


def test_blocks_match_without_low_mach_preconditioning():
    mesh, provider = _channel("prism", 1)
    _compare(mesh, generate_fr_operators(1), provider, _state(mesh, 11), low_mach=False)


def test_blocks_match_with_frozen_eddy_viscosity():
    mesh, provider = _channel("tet", 2)
    rng = np.random.default_rng(5)
    mu_t = 50.0 * MU * rng.uniform(0.5, 1.5, size=(mesh.n_cells, mesh.n_sps_per_cell))
    _compare(mesh, generate_fr_operators(2), provider, _state(mesh, 12), mu_t=mu_t)


def test_blocks_match_on_mixed_mesh_with_all_boundary_types():
    from tests.unit.test_boundary_ghost_states_batched_crosscheck import _build_multi_bc_provider
    from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh

    for order in (1, 2):
        mesh = _build_synthetic_mixed_mesh(order)
        ops = generate_fr_operators(order)
        provider = _build_multi_bc_provider(get_flat_face_geometry(mesh, ops), np.random.default_rng(order))
        rng = np.random.default_rng(order)
        shape = (mesh.n_cells, mesh.n_sps_per_cell)
        Q = np.stack([RHO * (1 + 0.03 * rng.standard_normal(shape)), U_INF + 3 * rng.standard_normal(shape),
                      3 * rng.standard_normal(shape), 3 * rng.standard_normal(shape),
                      P_INF * (1 + 0.01 * rng.standard_normal(shape))], axis=-1)
        U = Q.copy()
        U[..., 1:4] *= Q[..., :1]
        U[..., 4] = Q[..., 4] / 0.4 + 0.5 * Q[..., 0] * (Q[..., 1:4] ** 2).sum(-1)
        _compare(mesh, ops, provider, U)


def test_blocks_match_with_mixed_split_face_partner():
    """混合拆分面（B-8）：本面一部分通量点的另一侧取配对边界面的幽灵态。生产
    网格（plate_demo）上没有这类面，这里在棱柱通道上人为把一个内部面的一半通量点
    指向同一 owner 单元的一个边界面，检验"配对面 owner 就是本单元"这条链。"""
    mesh, provider = _channel("prism", 1)
    ops = generate_fr_operators(1)
    flat = get_flat_face_geometry(mesh, ops)
    own = np.asarray(flat.owner_cell)
    bnd = np.asarray(flat.is_boundary).astype(bool)
    prim = np.asarray(flat.owner_is_primary).astype(bool)
    chosen = None
    for f in np.nonzero(~bnd & prim & np.asarray(flat.neighbor_is_primary).astype(bool))[0]:
        partners = np.nonzero(bnd & prim & (own == own[f]))[0]
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
    _compare(mesh, ops, provider, _state(mesh, 21), flat=flat2)


def _av_field(mesh, seed):
    """问题单元式的人工扩散系数：大部分单元为零、少数单元 nu 与网格尺度相当（量级让人工
    扩散项与粘性项同阶或更大，差分对照才有分辨力）。"""
    rng = np.random.default_rng(seed)
    nu = np.zeros((mesh.n_cells, mesh.n_sps_per_cell))
    flagged = rng.random(mesh.n_cells) < 0.4
    nu[flagged] = 0.5 * rng.uniform(0.5, 1.5, size=(int(flagged.sum()), mesh.n_sps_per_cell))
    return nu


@pytest.mark.parametrize("kind", ["tet", "prism"])
@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("low_mach", [True, False])
def test_blocks_match_with_artificial_viscosity(kind, order, low_mach):
    """人工粘性 `div(nu grad U_k)`（nu 冻结）：对角块与着色差分一致（2026-10-02 之前解析
    装配不覆盖它，块 Jacobi 整体退回差分装配）。"""
    mesh, provider = _channel(kind, order)
    _compare(mesh, generate_fr_operators(order), provider, _state(mesh, 30 + order), low_mach=low_mach,
             nu_av=_av_field(mesh, order))


@pytest.mark.parametrize("kind", ["tet", "prism"])
def test_coupling_blocks_match_dense_difference_with_artificial_viscosity(kind):
    """耦合块（块 ILU / 多层预处理用）：对几个列单元逐自由度差分，比较其面邻居行。"""
    mesh, provider = _channel(kind, 1)
    ops = generate_fr_operators(1)
    U = _state(mesh, 41)
    nu_av = _av_field(mesh, 7)
    res = _Residual(mesh, ops, provider, None, True, None, nu_av)
    u0 = U.reshape(-1, 5)
    r0 = res(u0)
    ctx = MeanFlowLinearization(mesh=mesh, ops=ops, ghost_provider=provider, mu=MU, mach_ref=MACH_REF,
                                precond_mode=resolve_ausm_precond_mode(), low_mach=True, nu_av=nu_av)
    _, _, coupling = assemble_mean_flow_blocks(ctx, U, residual=r0.reshape(U.shape), want_coupling=True)
    npr, nte = real_sps_per_cell(1)
    n_sps = mesh.n_sps_per_cell
    worst = 0.0
    for col in (0, mesh.n_cells // 2):
        ny = npr if col < mesh.n_prism_cells else nte
        dense = {}
        for t in range(ny):
            for b in range(5):
                h = 1e-7 * SCALES[b]
                up = u0.copy()
                up[col * n_sps + t, b] += h
                dense[t, b] = ((res(up) - r0) / h).reshape(mesh.n_cells, n_sps, 5)
        for g in coupling.groups:
            for k in np.nonzero(g.cols == col)[0]:
                row = int(g.rows[k])
                nr = npr if row < mesh.n_prism_cells else nte
                blk = g.blocks[k].reshape(nr, 5, ny, 5).astype(np.float64)
                ref = np.stack([np.stack([dense[t, b][row, :nr] for b in range(5)], axis=-1)
                                for t in range(ny)], axis=2)
                worst = max(worst, float(np.abs(blk - ref).max() / np.abs(ref).max()))
    assert 0.0 < worst < 2e-4, f"耦合块最大相对误差 {worst:.2e}"
