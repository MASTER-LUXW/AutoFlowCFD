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
        # 无混合拆分面（B-8）：partner 全 -1、掩码全 False，混合覆盖循环不生效。
        mixed_nb_partner = np.full(n_faces, -1, dtype=np.int64)
        mixed_nb_mask = np.zeros((n_faces, n_fp), dtype=np.bool_)
        # 无非零 Dirichlet 目标值（本测试只覆盖 k 的 Dirichlet-zero 分支）。
        has_wall_dirichlet_value = np.zeros(n_faces, dtype=np.bool_)
        wall_dirichlet_value_face = np.zeros((n_faces, n_fp), dtype=np.float64)
        # 纯 collapsed 场景（无 native 四面体）：owner_cube_face 全 <6，
        # boundary_extrap_native 空占位，见 kernel 文档 native 分支。
        owner_cube_face = np.zeros(n_faces, dtype=np.int64)
        boundary_extrap_native = np.zeros((0, n_fp, n_sps))

        phi_owner, phi_neighbor = extrapolate_scalar_to_faces_kernel(
            scalar_sps, boundary_extrap,
            neighbor_src0_cell, neighbor_src0_mat,
            neighbor_src1_idx, neighbor_src1_cell, neighbor_src1_mat,
            owner_cell, owner_axis, owner_side,
            n_prism, n_faces, n_fp, n_sps,
            wall_dirichlet_zero_face,
            mixed_nb_partner, mixed_nb_mask,
            has_wall_dirichlet_value, wall_dirichlet_value_face,
            owner_cube_face, boundary_extrap_native,
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
        mixed_nb_partner = np.full(n_faces, -1, dtype=np.int64)
        mixed_nb_mask = np.zeros((n_faces, n_fp), dtype=np.bool_)
        has_wall_dirichlet_value = np.zeros(n_faces, dtype=np.bool_)
        wall_dirichlet_value_face = np.zeros((n_faces, n_fp), dtype=np.float64)
        owner_cube_face = np.zeros(n_faces, dtype=np.int64)
        boundary_extrap_native = np.zeros((0, n_fp, n_sps))

        _, phi_neighbor = extrapolate_scalar_to_faces_kernel(
            scalar_sps, boundary_extrap,
            neighbor_src0_cell, neighbor_src0_mat,
            neighbor_src1_idx, neighbor_src1_cell, neighbor_src1_mat,
            owner_cell, owner_axis, owner_side,
            n_prism, n_faces, n_fp, n_sps,
            wall_dirichlet_zero_face,
            mixed_nb_partner, mixed_nb_mask,
            has_wall_dirichlet_value, wall_dirichlet_value_face,
            owner_cube_face, boundary_extrap_native,
        )
        np.testing.assert_allclose(phi_neighbor, [[3.0]])

    def test_wall_dirichlet_value_face_mirrors_to_target(self):
        """omega 解析壁面值分支：WALL 面标记 has_wall_dirichlet_value=True
        且给定 target=12.0 时，ghost 应满足 (ghost+owner)/2 == target
        （即 ghost = 2*target - owner），而不是 Dirichlet-zero 或 Neumann。"""
        n_faces, n_fp, n_sps, n_prism = 1, 1, 1, 0
        scalar_sps = np.array([[5.0]])
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
        mixed_nb_partner = np.full(n_faces, -1, dtype=np.int64)
        mixed_nb_mask = np.zeros((n_faces, n_fp), dtype=np.bool_)
        has_wall_dirichlet_value = np.array([True])
        wall_dirichlet_value_face = np.array([[12.0]])
        owner_cube_face = np.zeros(n_faces, dtype=np.int64)
        boundary_extrap_native = np.zeros((0, n_fp, n_sps))

        phi_owner, phi_neighbor = extrapolate_scalar_to_faces_kernel(
            scalar_sps, boundary_extrap,
            neighbor_src0_cell, neighbor_src0_mat,
            neighbor_src1_idx, neighbor_src1_cell, neighbor_src1_mat,
            owner_cell, owner_axis, owner_side,
            n_prism, n_faces, n_fp, n_sps,
            wall_dirichlet_zero_face,
            mixed_nb_partner, mixed_nb_mask,
            has_wall_dirichlet_value, wall_dirichlet_value_face,
            owner_cube_face, boundary_extrap_native,
        )

        np.testing.assert_allclose(phi_owner, [[5.0]])
        np.testing.assert_allclose(phi_neighbor, [[19.0]])  # 2*12.0 - 5.0
        face_avg = 0.5 * (phi_owner + phi_neighbor)
        np.testing.assert_allclose(face_avg, [[12.0]])


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


class _MockNodes:
    def __init__(self, coords):
        self._coords = coords

    def get_coordinates(self):
        return self._coords


class _MockCells:
    def __init__(self, connectivity):
        self.connectivity = connectivity


def _build_synthetic_mixed_mesh(order: int):
    """2 个共享面的四面体 + 2 个共享侧面的棱柱——与
    test_fr_residual_inviscid.py::_build_synthetic_mixed_mesh 完全相同
    的几何构造（本文件独立保留一份，避免跨测试文件的私有函数依赖）。"""
    from autoflowcfd.grid.high_order.high_order_mesh import HighOrderMesh

    nodes = np.array(
        [
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1],
            [10, 0, 0], [11, 0, 0], [10, 1, 0], [10, 0, 1], [11, 0, 1], [10, 1, 1],
        ],
        dtype=float,
    )
    tet_conn = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int32)
    nodes = np.vstack([nodes, [[9, -1, 0], [9, -1, 1]]])
    prism_conn = np.array(
        [
            [5, 6, 7, 8, 9, 10],
            [5, 7, 11, 8, 10, 12],
        ],
        dtype=np.int32,
    )

    mock_volume = SimpleNamespace(
        cell_count=len(tet_conn) + len(prism_conn),
        nodes=_MockNodes(nodes),
        cells=_MockCells(tet_conn),
        prism_cells=_MockCells(prism_conn),
    )

    mesh = HighOrderMesh(order=order)
    mesh.load_from_volume_mesh(mock_volume)
    return mesh


class TestComputeOmegaWallTarget:
    """真实 bug 回归测试（V2.0 专家组盲审发现）：omega 壁面解析式
    `omega_wall = 60*nu/(beta1*d1^2)`（Wilcox 标准公式），此前完全没有
    接入、omega 恒用 Neumann 默认。"""

    def test_matches_wilcox_formula_on_wall_faces_only(self):
        from autoflowcfd.core.turbulence.transport import _compute_omega_wall_target

        mesh = _build_synthetic_mixed_mesh(order=1)
        ops = mesh.operators
        n_faces = mesh.face_connectivity.n_faces
        n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell

        # 构造一个逐 SP 不同的合成壁面距离场（避免退化成常数掩盖 bug）。
        rng = np.random.default_rng(0)
        wall_distance = rng.uniform(0.001, 0.1, size=(n_cells, n_sps))

        # 挑 2 个真实存在的边界面标记为 WALL（不依赖真实 BoundaryGhostProvider，
        # 只测 _compute_omega_wall_target 本身的公式/取值逻辑）。
        boundary_idx = np.nonzero(mesh.face_connectivity.is_boundary)[0]
        assert len(boundary_idx) >= 2, "synthetic mesh 应该有边界面"
        wall_mask = np.zeros(n_faces, dtype=np.bool_)
        wall_mask[boundary_idx[:2]] = True

        mu = 1.8e-5
        rho = np.full((n_cells, n_sps), 1.2)
        beta1 = 0.075
        turb_model = SimpleNamespace(beta1=beta1)
        solver = SimpleNamespace(
            mesh=mesh, ops=ops, wall_distance=wall_distance, turb_model=turb_model,
        )

        omega_wall_value_face, has_value = _compute_omega_wall_target(solver, wall_mask, mu, rho)

        np.testing.assert_array_equal(has_value, wall_mask)
        # 非 WALL 面必须恒为 0（不会被消费，但仍应是良定义的值，不是 NaN/垃圾）。
        non_wall = ~wall_mask
        np.testing.assert_array_equal(omega_wall_value_face[non_wall], 0.0)

        from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
        flat = get_flat_face_geometry(mesh, ops)
        for f in np.nonzero(wall_mask)[0]:
            owner = flat.owner_cell[f]
            d1_expected = max(np.min(wall_distance[owner]), 1e-8)
            nu = mu / rho[owner].mean()
            omega_expected = 60.0 * nu / (beta1 * d1_expected**2)
            np.testing.assert_allclose(omega_wall_value_face[f], omega_expected, rtol=1e-10)
            assert np.isfinite(omega_expected) and omega_expected > 0

    def test_zero_wall_mask_gives_no_dirichlet_faces(self):
        from autoflowcfd.core.turbulence.transport import _compute_omega_wall_target

        mesh = _build_synthetic_mixed_mesh(order=1)
        n_faces = mesh.face_connectivity.n_faces
        n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell

        wall_mask = np.zeros(n_faces, dtype=np.bool_)
        solver = SimpleNamespace(
            mesh=mesh, ops=mesh.operators,
            wall_distance=np.full((n_cells, n_sps), 0.01),
            turb_model=SimpleNamespace(beta1=0.075),
        )

        omega_wall_value_face, has_value = _compute_omega_wall_target(
            solver, wall_mask, 1.8e-5, np.full((n_cells, n_sps), 1.2)
        )

        assert not np.any(has_value)
        np.testing.assert_array_equal(omega_wall_value_face, 0.0)


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
