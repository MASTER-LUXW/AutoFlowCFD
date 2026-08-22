"""Unit tests for core/turbulence/transport.py's WALL k=0 Dirichlet fix.

`extrapolate_scalar_to_faces_kernel` previously always used a Neumann
(ghost=owner) default at every boundary face, including WALL, for k and
omega alike - a known, documented approximation (see the kernel's own
docstring history). k is exactly zero at a no-slip wall (a standard SST/
k-omega boundary condition, not an approximation), so for k specifically a
Dirichlet-zero ghost (ghost = -owner, mirroring the mean-flow no-slip wall
ghost state formula) is now applied wherever `_compute_wall_dirichlet_
face_mask` identifies a face as WALL-typed via the solver's
`boundary_ghost_provider`. omega is left on the Neumann default (its
analytic near-wall value needs additional wall-distance data, tracked as
separate future work).

These tests pin the kernel-level Dirichlet-zero mirroring in isolation and
the mask builder's face classification / defensive fallback.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from autoflowcfd.core.turbulence.transport_kernel import extrapolate_scalar_to_faces_kernel
from autoflowcfd.core.turbulence.transport import _compute_wall_dirichlet_face_mask


class TestExtrapolateScalarToFacesKernelWallDirichlet:
    def test_wall_face_mirrors_to_zero_non_wall_stays_neumann(self):
        """Two boundary faces on the same single cell (value 5.0): face 0 is
        flagged WALL-Dirichlet-zero, face 1 is not. Expect ghost = -owner for
        face 0 (enforces phi=0 at the wall) and ghost = owner for face 1
        (unchanged Neumann default)."""
        n_faces, n_fp, n_sps, n_prism = 2, 1, 1, 0
        scalar_sps = np.array([[5.0]])

        # boundary_extrap[celltype, axis, side_idx] -> (n_fp, n_sps); identity here.
        boundary_extrap = np.zeros((2, 3, 2, n_fp, n_sps))
        boundary_extrap[1, 0, 0] = np.array([[1.0]])  # tet(celltype=1), axis=0, side_idx=0 (side<=0)

        owner_cell = np.array([0, 0], dtype=np.int64)
        owner_axis = np.array([0, 0], dtype=np.int64)
        owner_side = np.array([-1.0, -1.0])

        # Both faces are boundary faces: no real neighbor source.
        neighbor_src0_cell = np.array([-1, -1], dtype=np.int64)
        neighbor_src0_mat = np.zeros((n_faces, n_fp, n_sps))
        neighbor_src1_idx = np.array([-1, -1], dtype=np.int64)
        neighbor_src1_cell = np.empty((0,), dtype=np.int64)
        neighbor_src1_mat = np.empty((0, n_fp, n_sps))

        wall_dirichlet_zero_face = np.array([True, False])

        phi_owner, phi_neighbor = extrapolate_scalar_to_faces_kernel(
            scalar_sps, boundary_extrap,
            neighbor_src0_cell, neighbor_src0_mat,
            neighbor_src1_idx, neighbor_src1_cell, neighbor_src1_mat,
            owner_cell, owner_axis, owner_side,
            n_prism, n_faces, n_fp, n_sps,
            wall_dirichlet_zero_face,
        )

        np.testing.assert_allclose(phi_owner, [[5.0], [5.0]])
        np.testing.assert_allclose(phi_neighbor[0], [-5.0])  # WALL: Dirichlet-zero mirror
        np.testing.assert_allclose(phi_neighbor[1], [5.0])   # non-WALL: unchanged Neumann

    def test_all_false_mask_reproduces_old_neumann_only_behavior(self):
        """A mask of all False must reproduce the pre-fix behavior exactly
        (ghost = owner at every boundary face) - guards against the new
        parameter silently changing existing (non-WALL) callers."""
        n_faces, n_fp, n_sps, n_prism = 1, 1, 1, 0
        scalar_sps = np.array([[3.0]])
        boundary_extrap = np.zeros((2, 3, 2, n_fp, n_sps))
        boundary_extrap[1, 0, 0] = np.array([[1.0]])
        owner_cell = np.array([0], dtype=np.int64)
        owner_axis = np.array([0], dtype=np.int64)
        owner_side = np.array([-1.0])
        neighbor_src0_cell = np.array([-1], dtype=np.int64)
        neighbor_src0_mat = np.zeros((n_faces, n_fp, n_sps))
        neighbor_src1_idx = np.array([-1], dtype=np.int64)
        neighbor_src1_cell = np.empty((0,), dtype=np.int64)
        neighbor_src1_mat = np.empty((0, n_fp, n_sps))
        wall_dirichlet_zero_face = np.array([False])

        _, phi_neighbor = extrapolate_scalar_to_faces_kernel(
            scalar_sps, boundary_extrap,
            neighbor_src0_cell, neighbor_src0_mat,
            neighbor_src1_idx, neighbor_src1_cell, neighbor_src1_mat,
            owner_cell, owner_axis, owner_side,
            n_prism, n_faces, n_fp, n_sps,
            wall_dirichlet_zero_face,
        )
        np.testing.assert_allclose(phi_neighbor, [[3.0]])


class TestComputeWallDirichletFaceMask:
    def test_identifies_wall_coded_faces_only(self):
        mesh = SimpleNamespace(face_connectivity=SimpleNamespace(n_faces=5))
        provider = SimpleNamespace(
            group_code=np.array([-1, 0, 1, 1, 2]),
            code_to_config={
                0: {"type": "FARFIELD"},
                1: {"type": "WALL"},
                2: {"type": "OUTLET"},
            },
        )
        solver = SimpleNamespace(mesh=mesh, boundary_ghost_provider=provider)

        mask = _compute_wall_dirichlet_face_mask(solver)

        np.testing.assert_array_equal(mask, [False, False, True, True, False])

    def test_no_wall_groups_returns_all_false(self):
        mesh = SimpleNamespace(face_connectivity=SimpleNamespace(n_faces=3))
        provider = SimpleNamespace(
            group_code=np.array([-1, 0, 0]),
            code_to_config={0: {"type": "FARFIELD"}},
        )
        solver = SimpleNamespace(mesh=mesh, boundary_ghost_provider=provider)

        mask = _compute_wall_dirichlet_face_mask(solver)

        np.testing.assert_array_equal(mask, [False, False, False])

    def test_missing_provider_metadata_falls_back_to_all_false(self):
        """A ghost provider without group_code/code_to_config (e.g. a plain
        callable used by some tests) must not raise - it degrades to the
        pre-fix Neumann-everywhere default, not a new failure mode."""
        mesh = SimpleNamespace(face_connectivity=SimpleNamespace(n_faces=4))
        solver = SimpleNamespace(mesh=mesh, boundary_ghost_provider=lambda f, q, n: q)

        mask = _compute_wall_dirichlet_face_mask(solver)

        np.testing.assert_array_equal(mask, [False, False, False, False])

    def test_no_ghost_provider_falls_back_to_all_false(self):
        mesh = SimpleNamespace(face_connectivity=SimpleNamespace(n_faces=2))
        solver = SimpleNamespace(mesh=mesh, boundary_ghost_provider=None)

        mask = _compute_wall_dirichlet_face_mask(solver)

        np.testing.assert_array_equal(mask, [False, False])


class TestTransportResidualOutlierSuppressionWrapping:
    """`compute_turbulence_transport_residual` wraps its (n_cells,n_sps)
    dk_dt/domega_dt transport residual as (n_cells,n_sps,1) to reuse
    `suppress_residual_outliers` (mechanism 3 - the same statistical
    outlier detection the mean-flow residual in inviscid.py/viscous_flux.py
    already uses, see troubled_cell.py's "mechanism 3" docs), then squeezes
    the trailing axis back off. These tests pin that reshape round-trip in
    isolation, on the exact scalar shape transport.py actually uses -
    `suppress_residual_outliers` itself is already covered by
    test_troubled_cell.py::TestSuppressResidualOutliers."""

    def test_outlier_sp_zeroed_siblings_and_shape_preserved(self):
        from autoflowcfd.core.fr_operators.troubled_cell import suppress_residual_outliers

        n_cells, n_sps = 2, 4
        dk_dt = np.full((n_cells, n_sps), 1.0)
        dk_dt[0, 2] = 1e6  # one wildly anomalous SP in cell 0
        k_field = np.ones((n_cells, n_sps)) * 1e-3

        result = suppress_residual_outliers(dk_dt[:, :, None], k_field[:, :, None])[:, :, 0]

        assert result.shape == dk_dt.shape
        assert result[0, 2] == 0.0
        # every other SP (both cells) must be untouched
        mask = np.ones_like(dk_dt, dtype=bool)
        mask[0, 2] = False
        np.testing.assert_array_equal(result[mask], dk_dt[mask])
