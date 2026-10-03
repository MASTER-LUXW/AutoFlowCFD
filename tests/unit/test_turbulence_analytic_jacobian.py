"""k-ln(omega) 解析单元块 Jacobian（`core/turbulence/jacobian`）对照着色有限差分装配。

参考是紧耦合 Newton 残差的湍流子系统（`coupled_step.SubsystemResidual`：平均流冻结在
基态、零填充行置零），即耦合预处理里湍流那一块所近似的算子。湍流残差对单元自身自由度只经本单元与面邻居（梯度是单元内局部梯度），
距离 1 着色下差分装配的 `J_cc` 是精确的；耦合块对照稠密逐列差分 Jacobian，并检查
面邻居之外没有非零块。k/omega 场加 20% 随机扰动，让源项、F1 混合、realizability、
交叉扩散都处在非平凡状态。
"""


import numpy as np
import pytest

from autoflowcfd.core.fr_solver.turbulence.implicit import CpuCoupledBackend, single_machine_cell_colors
from autoflowcfd.core.time_integration.implicit.coupled_step import (
    TURB_COLUMNS, CoupledResidual, SubsystemResidual,
)
from autoflowcfd.core.time_integration.implicit.cell_blocks import CellBlockJacobian
from autoflowcfd.fr.native_padding import real_row_mask, real_sps_per_cell
from tests.validation._channel_mesh import (
    build_channel_mesh, build_channel_mesh_prism, build_face_exact_ghost_provider,
)

RHO, U, P = 1.225, 30.0, 101325.0
LX, H, LZ = 0.4, 0.1, 0.08


def _sst_solver(kind, order, n_warm=3):
    from autoflowcfd.core.fr_solver import FRSolver
    from autoflowcfd.core.time_integration import TimeIntegrationScheme

    mesh = (build_channel_mesh if kind == "tet" else build_channel_mesh_prism)(order, 3, 4, 2, LX, H, LZ)
    bc = {n: {"type": "SYMMETRY"} for n in ("z_min", "z_max")}
    for n in ("wall_bottom", "wall_top"):
        bc[n] = {"type": "WALL", "is_no_slip": True, "wall_velocity": [0.0, 0.0, 0.0]}
    for n in ("x_min", "x_max"):
        bc[n] = {"type": "FARFIELD", "Q_free": [RHO, U, 0.0, 0.0, P]}
    s = FRSolver(mesh=mesh, order=order, turb_model_name="SST", n_vars=7,
                 time_scheme=TimeIntegrationScheme.NEWTON_KRYLOV, rho_inf=RHO, vel_inf=U, p_inf=P,
                 mu_molecular=1.8e-5, bc_overrides=bc)
    s.order_continuation_enabled = False
    s.boundary_ghost_provider = build_face_exact_ghost_provider(mesh, LX, H, LZ, bc)
    y = np.asarray(mesh.sps_coords)[..., 1]
    s.wall_distance = np.ascontiguousarray(np.maximum(np.minimum(y, H - y), 1e-12))
    for _ in range(n_warm):
        s.step(1e-6)
    rng = np.random.default_rng(3)
    m = s.turb_model
    m.k_field = m.k_field * (1 + 0.2 * rng.standard_normal(m.k_field.shape))
    m.omega_field = m.omega_field * (1 + 0.2 * rng.standard_normal(m.omega_field.shape))
    return s


def _blocks(s):
    cb = CpuCoupledBackend(s)
    be = cb.turb
    m = s.turb_model
    x0 = cb.state()
    res = SubsystemResidual(CoupledResidual(cb, real_row_mask(cb.cell_is_prism, cb.n_sps, cb.order), x0),
                            x0, TURB_COLUMNS)
    # 未知量 (k, w = ln omega)，见 core/turbulence/sst/log_omega.py
    kw0 = np.ascontiguousarray(x0[:, TURB_COLUMNS])
    r0 = res(kw0)
    be.prepare_inputs()
    order = int(s.mesh.order)
    npr, nte = real_sps_per_cell(order)
    fd = CellBlockJacobian(res, kw0, r0, np.array([m.k_inf, 1.0]), n_sps=s.mesh.n_sps_per_cell,
                           cell_is_prism=np.arange(s.mesh.n_cells) < s.mesh.n_prism_cells,
                           n_real_prism=npr, n_real_tet=nte, colors=single_machine_cell_colors(s))
    asm = be.block_assembler()
    asm.want_coupling = True
    bp, bt, cp = asm(kw0, r0)
    return be, res, kw0, r0, fd, bp, bt, cp


def _assert_diag_match(fd, bp, bt):
    for A, B in ((bp, fd.blocks_prism), (bt, fd.blocks_tet)):
        if B.shape[0] == 0:
            continue
        A, B = A.astype(np.float64), B.astype(np.float64)
        err = np.abs(A - B).max(axis=(1, 2)) / np.abs(B).max(axis=(1, 2))
        assert err.max() < 1e-4, f"块最大相对误差 {err.max():.2e}"
        assert np.median(err) < 1e-5


@pytest.mark.parametrize("kind,order", [("prism", 1), ("prism", 2), ("tet", 1), ("tet", 2)])
def test_diagonal_blocks_match_colored_finite_difference(kind, order):
    _, _, _, _, fd, bp, bt, _ = _blocks(_sst_solver(kind, order))
    _assert_diag_match(fd, bp, bt)


def test_diagonal_blocks_match_without_overintegration(monkeypatch):
    monkeypatch.setenv("AFCFD_TURB_OVERINT", "off")
    _, _, _, _, fd, bp, bt, _ = _blocks(_sst_solver("prism", 1))
    _assert_diag_match(fd, bp, bt)


def test_coupling_blocks_match_dense_finite_difference():
    s = _sst_solver("tet", 1)
    be, res, kw0, r0, fd, bp, bt, cp = _blocks(s)
    nc, ns = s.mesh.n_cells, s.mesh.n_sps_per_cell
    npr, nte = real_sps_per_cell(1)
    nreal = np.where(np.arange(nc) < s.mesh.n_prism_cells, npr, nte)
    cols = [(c, sp, v) for c in range(nc) for sp in range(nreal[c]) for v in range(2)]
    cell_rows = {c: [] for c in range(nc)}
    flat_idx = []
    for k, (c, sp, v) in enumerate(cols):
        cell_rows[c].append(k)
        flat_idx.append((c * ns + sp) * 2 + v)
    flat_idx = np.array(flat_idx)
    u0, rr0 = kw0.reshape(-1), r0.reshape(-1)
    scale = np.array([s.turb_model.k_inf, 1.0])
    J = np.zeros((len(cols), len(cols)))
    for k in range(len(cols)):
        up = u0.copy()
        h = 1e-7 * max(abs(up[flat_idx[k]]), scale[flat_idx[k] % 2])
        up[flat_idx[k]] += h
        J[:, k] = (res(up.reshape(-1, 2)).reshape(-1)[flat_idx] - rr0[flat_idx]) / h
    covered = set()
    for g in cp.groups:
        for q in range(g.rows.size):
            c, y = int(g.rows[q]), int(g.cols[q])
            covered.add((c, y))
            ref = J[np.ix_(cell_rows[c], cell_rows[y])]
            assert np.abs(g.blocks[q] - ref).max() <= 1e-3 * (np.abs(ref).max() + 1e-300)
    jmax = np.abs(J).max()
    for c in range(nc):
        for y in range(nc):
            if c != y and (c, y) not in covered:
                assert np.abs(J[np.ix_(cell_rows[c], cell_rows[y])]).max() <= 1e-9 * jmax
