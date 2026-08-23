"""Regression test for a real bug found 2026-08-22 (real-mesh resume,
asked directly by the user: "如果当前是P1，resume是从P1开始运算吗"):

`run_order_continuation` unconditionally reinitializes `solver.state` to a
uniform freestream and restarts the P0->target ramp from scratch whenever
`solver.state.U.shape[1] != 1` (not P0). This branch exists to handle a
*freshly constructed* FRSolver, whose default initial state is a uniform
freestream sized at the *target* order (not P0) - reinitializing it down
to a proper P0 starting point before ramping is correct there.

But `solve resume` reuses the exact same code path. A checkpoint saved
mid-ramp (e.g. genuinely converged physics at P1) also has
`state.U.shape[1] != 1`, and would hit the *identical* branch - silently
discarding the real, resumed solution and restarting the whole ramp from
a uniform P0 freestream, with no error or warning. Resuming a P1/P2
checkpoint would be indistinguishable from not resuming at all.

Fix: `rebuild_solver_from_checkpoint` now marks the solver with
`_resumed_from_checkpoint = True` right after loading the real state;
`run_order_continuation` skips the reinit-to-P0 branch when this flag is
set, and starts the `orders` ramp range from `solver.current_order`
(the checkpoint's real order) instead of always from P0.
"""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from autoflowcfd.core.utils.order_continuation import run_order_continuation


def _n_sps(order):
    return (order + 1) ** 3


def _fake_solver(current_order, target_order, resumed):
    n_sps = _n_sps(current_order)
    # Recognizable sentinel value - if the reinit-to-P0 branch fires, the
    # array gets replaced wholesale by a differently-shaped uniform state,
    # so any check that these exact values persist would fail.
    U = np.full((2, n_sps, 7), 42.0)

    mesh = SimpleNamespace(
        _order_geometry_cache={},
        set_order=lambda p: mesh.set_order_calls.append(p),
    )
    mesh.set_order_calls = []

    solver = SimpleNamespace(
        order=target_order,
        current_order=current_order,
        ops=SimpleNamespace(D_3d=np.zeros((n_sps, n_sps))),
        mesh=mesh,
        state=SimpleNamespace(U=U, n_cells=2, n_vars=7),
        freestream={"rho_inf": 1.225, "vel_inf": 33.33, "p_inf": 101325.0},
        turb_model=None,
        wall_distance=None,
        sgs_model=None,
        _resumed_from_checkpoint=resumed,
        step=lambda dt: 1e-9,  # always "converged" -> each phase finishes in 1 iteration
    )

    def _fake_interpolate(new_order):
        solver.current_order = new_order
        solver.state.U = np.full((2, _n_sps(new_order), 7), 42.0)

    solver._interpolate_to_new_order = _fake_interpolate
    return solver


def _fake_generate_ops(p):
    n = _n_sps(p)
    return SimpleNamespace(D_3d=np.zeros((n, n)))


class TestResumeSkipsP0Reinit:
    def test_resumed_p1_checkpoint_ramp_starts_at_p1_not_p0(self):
        solver = _fake_solver(current_order=1, target_order=2, resumed=True)

        with patch(
            "autoflowcfd.fr.operators.generate_fr_operators", side_effect=_fake_generate_ops,
        ), patch(
            "autoflowcfd.core.fr_solver.state.FRState"
        ) as mock_frstate:
            result = run_order_continuation(solver, max_iter=10, dt=1e-3, tol=1e-6)

        # The P0-reinit branch must never have run: FRState() is only
        # constructed there.
        mock_frstate.assert_not_called()
        # The ramp must never have visited P0 at all.
        assert 0 not in solver.mesh.set_order_calls
        assert solver.mesh.set_order_calls == [1, 2]
        assert result.converged is True

    def test_non_resumed_fresh_solver_still_restarts_from_p0(self):
        """Guards the primary (non-resume) `solve steady` path: unchanged
        behaviour when _resumed_from_checkpoint is absent/False."""
        solver = _fake_solver(current_order=2, target_order=2, resumed=False)

        with patch(
            "autoflowcfd.fr.operators.generate_fr_operators", side_effect=_fake_generate_ops,
        ):
            run_order_continuation(solver, max_iter=30, dt=1e-3, tol=1e-6)

        # P0 appears twice: once from the reinit-to-P0 branch itself, once
        # from the ramp loop's own first (P0) phase - pre-existing,
        # harmless redundancy, not something this fix changes.
        assert solver.mesh.set_order_calls == [0, 0, 1, 2]
