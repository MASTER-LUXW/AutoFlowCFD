"""Regression test for a real bug found 2026-08-23 (traced from the user's
P1-divergence root-cause request, after det(J)/face-misalignment/local-CFL
hypotheses were each checked against real cube_demo data and ruled out):

`_SolverGeometryMixin._get_metric_flux_scale`'s cache-validity check
compared only `cached.shape[0]` (n_cells) against `self.state.U.shape[0]`
(also n_cells) - both are invariant across an Order Continuation order
transition (only `shape[1]`, n_sps per cell, changes when P0->P1 etc.
happens). The check therefore always reported the cache as still valid
after any order transition, silently returning the *old* order's
`(n_cells, old_n_sps)` array. The consumer (cfl.py's `dt_geometric` term)
then computes `metric_flux_scale * wave_speed` where `wave_speed` is
`(n_cells, new_n_sps)` - numpy broadcasts the size-1-mismatched trailing
dimension without error, silently reusing the coarser old-order metric
scale for every SP of the new order. This is exactly the geometric CFL
term that was added specifically to catch degenerate-cell SP-level
stiffness (see cfl.py's own extensive documentation on it) - the bug
silently weakens that exact protection from the first post-transition
step onward, for the rest of the run (the cache is never explicitly
invalidated anywhere else in the codebase - confirmed by grepping for
all references to `_metric_flux_scale_cache`).

Fixed by comparing the full `(n_cells, n_sps)` shape instead of just
`shape[0]`.
"""

from types import SimpleNamespace

import numpy as np

from autoflowcfd.core.fr_solver.solver_geometry import _SolverGeometryMixin


class _FakeSolver(_SolverGeometryMixin):
    def __init__(self, n_cells, n_sps):
        self.state = SimpleNamespace(U=np.zeros((n_cells, n_sps, 7)))
        det_jacs = np.full((n_cells, n_sps), 1e-6)
        inv_jacs = np.tile(np.eye(3), (n_cells, n_sps, 1, 1))
        self.mesh = SimpleNamespace(
            n_cells=n_cells, n_sps_per_cell=n_sps,
            jacobians={"det_jacs": det_jacs, "inv_jacs": inv_jacs},
        )


class TestMetricFluxScaleCacheInvalidatesOnOrderTransition:
    def test_cache_recomputes_with_correct_shape_after_order_change(self):
        n_cells = 5
        solver = _FakeSolver(n_cells, n_sps=1)  # P0
        scale_p0 = solver._get_metric_flux_scale()
        assert scale_p0.shape == (n_cells, 1)

        # Simulate an Order Continuation transition to P1: n_cells stays
        # the same, n_sps changes (mesh.set_order + state interpolation).
        solver.mesh.n_sps_per_cell = 8
        solver.mesh.jacobians["det_jacs"] = np.full((n_cells, 8), 1e-6)
        solver.mesh.jacobians["inv_jacs"] = np.tile(np.eye(3), (n_cells, 8, 1, 1))
        solver.state.U = np.zeros((n_cells, 8, 7))

        scale_p1 = solver._get_metric_flux_scale()
        assert scale_p1.shape == (n_cells, 8), (
            f"expected the cache to invalidate and recompute at the new "
            f"n_sps=8, got stale shape {scale_p1.shape}"
        )

    def test_cache_is_actually_reused_when_shape_is_unchanged(self):
        """Guards against over-correcting into never caching at all."""
        n_cells = 5
        solver = _FakeSolver(n_cells, n_sps=8)
        first = solver._get_metric_flux_scale()
        # Mutate the mesh's raw jacobians in place without touching state.U
        # - if the cache were bypassed, this would produce a different
        # array object; if reused correctly, the same cached array comes
        # back regardless of the (now stale, but shape-unchanged) mesh data.
        solver.mesh.jacobians["det_jacs"] = solver.mesh.jacobians["det_jacs"] * 2
        second = solver._get_metric_flux_scale()
        assert second is first
