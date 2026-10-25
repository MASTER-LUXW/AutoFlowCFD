# -*- coding: utf-8 -*-
"""`FRSolver.compute_mean_flow_residual`：无粘与粘性共用一份原始变量与边界幽灵态（2026-10-15），结果与两项
各自从 `U` 重算时逐位相同；显式推进与隐式耦合步都经过这一处组装。"""

import numpy as np
import pytest

from tests.validation._channel_mesh import build_channel_mesh_prism, build_face_exact_ghost_provider
from autoflowcfd.core.time_integration import TimeIntegrationScheme

RHO, U, P = 1.225, 30.0, 101325.0
LX, H, LZ = 0.4, 0.1, 0.08


def _solver(order):
    from autoflowcfd.core.fr_solver import FRSolver

    mesh = build_channel_mesh_prism(order, nx=3, ny=4, nz=2, Lx=LX, H=H, Lz=LZ)
    bc = {n: {"type": "SYMMETRY"} for n in ("z_min", "z_max")}
    bc["wall_bottom"] = {"type": "WALL", "is_no_slip": True, "wall_velocity": [0.0, 0.0, 0.0]}
    bc["wall_top"] = {"type": "WALL", "is_no_slip": True, "wall_velocity": [U, 0.0, 0.0]}
    for n in ("x_min", "x_max"):
        bc[n] = {"type": "FARFIELD", "Q_free": [RHO, 0.5 * U, 0.0, 0.0, P]}
    solver = FRSolver(mesh=mesh, order=order, rho_inf=RHO, vel_inf=0.5 * U, p_inf=P,
                      mu_molecular=1.8e-5, bc_overrides=bc, wall_treatment="resolved", turb_model_name="SST",
                      time_scheme=TimeIntegrationScheme.SSP_RK3)
    solver.boundary_ghost_provider = build_face_exact_ghost_provider(mesh, LX, H, LZ, bc)
    rng = np.random.default_rng(order)
    Uc = np.array(solver.state.U, copy=True)
    Uc[..., :5] *= 1.0 + 0.02 * rng.normal(size=Uc[..., :5].shape)
    solver.state.U = Uc
    return solver


@pytest.mark.parametrize("order", [0, 1, 2])
def test_matches_separately_computed_terms_and_refreshes_primitives(order):
    solver = _solver(order)
    rng = np.random.default_rng(7)
    mu_t = rng.uniform(0.0, 5e-4, solver.state.U.shape[:2])
    got = solver.compute_mean_flow_residual(mu_t_turb=mu_t)
    Q_after = np.array(solver.state.Q, copy=True)

    visc = solver.compute_viscous_residual(mu_t_turb=mu_t)
    visc += solver.compute_inviscid_residual()
    assert np.array_equal(got, -visc)
    # `state.Q` 与 `state.U` 同步（低马赫预处理与湍流读它）
    assert np.array_equal(Q_after, solver.state.Q)


def test_shared_primitives_are_the_synced_state_without_reconverting(monkeypatch):
    """两项共用的原始变量就是刚同步的 `state.Q`：一次平均流残差里 U -> Q 只换算一次（`state._update_primitives`），
    不再另算一份新数组（cube_demo 粗网格 P2 上每次 206 MB 新分配加一次整场换算）。"""
    import autoflowcfd.core.fr_residual.inviscid as inviscid_mod

    solver = _solver(1)
    seen = []
    orig = inviscid_mod.mean_flow_primitives
    monkeypatch.setattr("autoflowcfd.core.fr_solver.solver.residuals.mean_flow_primitives",
                        lambda Q, *a: seen.append(Q) or orig(Q, *a))
    monkeypatch.setattr(inviscid_mod, "conserved_to_primitive",
                        lambda U: pytest.fail("平均流残差不应再从 U 重算原始变量"))
    solver.compute_mean_flow_residual(mu_t_turb=np.zeros(solver.state.U.shape[:2]))
    assert len(seen) == 1 and seen[0] is solver.state.Q
