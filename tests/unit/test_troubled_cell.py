"""Unit tests for core/fr_operators/troubled_cell.py's numba median kernel.

`_median_abs_over_sps_kernel` replaced `np.median(np.abs(residual), axis=1)`
inside `suppress_residual_outliers` - real perf bottleneck on production
meshes (numpy's generic n-dimensional `partition`-based reduction pays a lot
of per-call dispatch overhead when reducing millions of tiny (n_sps,) groups
independently). These tests pin the numba kernel against `np.median` to
guarantee it is bit-exact, not an approximation, across both parities of
`np.median`'s definition (odd n_sps takes the middle value, even n_sps
averages the two middle values) - P0/P2/P3 have odd n_sps ((order+1)^3 for
even order+1... actually (order+1)^3 is odd only when order+1 is odd, i.e.
order even), P1/P3 have even/odd depending on order+1's parity, so all four
production orders are covered explicitly below.
"""

import numpy as np
import pytest

from autoflowcfd.core.fr_operators.troubled_cell import (
    _median_abs_over_sps_kernel,
    precompute_cell_face_misalignment,
    suppress_residual_outliers,
)


class TestMedianAbsOverSpsKernel:
    @pytest.mark.parametrize("n_sps", [1, 8, 27, 64])  # P0, P1, P2, P3
    def test_matches_numpy_median(self, n_sps):
        rng = np.random.default_rng(0)
        residual = rng.standard_normal((500, n_sps, 5)) * 10.0

        expected = np.median(np.abs(residual), axis=1)
        actual = _median_abs_over_sps_kernel(residual)

        np.testing.assert_array_equal(actual, expected)

    def test_matches_numpy_median_with_negative_and_zero_values(self):
        """Explicit small case covering the sign-flip and zero-value paths
        in the kernel's inline abs()."""
        residual = np.array([
            [[-3.0], [1.0], [0.0], [-2.0]],  # even n_sps=4 -> average of 2 middle
        ])
        expected = np.median(np.abs(residual), axis=1)
        actual = _median_abs_over_sps_kernel(residual)
        np.testing.assert_array_equal(actual, expected)


class TestSuppressResidualOutliers:
    def test_no_outliers_returns_input_unchanged(self):
        rng = np.random.default_rng(2)
        residual = rng.standard_normal((50, 8, 5)) * 1e-3
        reference = np.ones((50, 8, 5))
        out = suppress_residual_outliers(residual, reference)
        np.testing.assert_array_equal(out, residual)

    def test_single_outlier_sp_zeroed_siblings_untouched(self):
        n_cells, n_sps, n_vars = 4, 8, 5
        rng = np.random.default_rng(3)
        residual = rng.standard_normal((n_cells, n_sps, n_vars)) * 1e-3
        reference = np.ones((n_cells, n_sps, n_vars))
        # Inject one wildly-out-of-scale value in cell 0, SP 0, var 1.
        residual[0, 0, 1] = 1e5
        out = suppress_residual_outliers(residual, reference)
        assert out[0, 0, 1] == 0.0
        # Every other entry must be untouched.
        mask = np.ones_like(residual, dtype=bool)
        mask[0, 0, 1] = False
        np.testing.assert_array_equal(out[mask], residual[mask])

    def test_whole_cell_uniform_blowup_is_a_known_undetected_gap(self):
        """已确认、暂不修复的限制的量化记录（2026-08-29，见
        tet_collapsed_coord_anisotropy 项目记忆与 suppress_residual_
        outliers 模块文档"调查记录"一节）：`ref_sibling`（同单元内其余
        SP 中位数）对"整个单元所有 SP 均匀放大"这类真实复现过的情形
        结构性失明——该单元自己的 median-of-siblings 会跟着一起被拖
        高，永远触不到相对阈值。

        曾尝试用全网格中位数（`ref_global`）独立堵住这个缺口，但在真实
        cube_demo 网格、从均匀自由流场初场起步时被证伪：真实残差分布
        天然高度不均匀（边界附近合法的大残差 vs 远场天然接近零），全局
        中位数会把边界驱动的真实物理也当异常清零（真实复现：RMS 残差
        从 3.5e8 骤降到 4.28e-4，F_pressure≈1.6e-11，求解器实质冻结在
        初场附近），已回退。这个测试不是要断言"这个缺口已修复"，而是
        如实记录"这个缺口确实存在、目前没有已知的安全修复方式"——避免
        未来有人在不知情的情况下重复踩坑。
        """
        n_cells, n_sps, n_vars = 10, 8, 5
        rng = np.random.default_rng(4)
        residual = rng.standard_normal((n_cells, n_sps, n_vars)) * 1e-3
        reference = np.ones((n_cells, n_sps, n_vars))
        # Cell 5: ALL SPs uniformly blown up (not a single spike) - the
        # cell's own sibling-median rises right along with it, so the
        # (still current, unmodified) local-only criterion lets it through.
        residual[5, :, :] = 1e5
        out = suppress_residual_outliers(residual, reference)
        assert np.array_equal(out[5], residual[5]), (
            "known gap: whole-cell uniform blowup is NOT caught by the current "
            "(local-only) criterion - see docstring for the falsified fix attempt"
        )

    def test_widespread_moderately_elevated_region_not_falsely_suppressed(self):
        """一大片（接近半个网格）合理地、均匀地比其余部分更大（比如
        靠近驻点/入口的真实物理区域），但幅度远低于安全倍数 `factor`，
        不应被误伤——局部（同单元）判据天然满足这一点，因为每个单元的
        参照量都随自己的实际残差量级自适应缩放，不依赖任何全网格统计量。
        """
        n_cells, n_sps, n_vars = 20, 8, 5
        rng = np.random.default_rng(5)
        residual = rng.standard_normal((n_cells, n_sps, n_vars)) * 1e-3
        reference = np.ones((n_cells, n_sps, n_vars))
        # Roughly half the mesh has a genuinely larger (but not pathological)
        # residual scale - only 50x the rest, far below the 1e4 safety factor.
        residual[:9] *= 50.0
        out = suppress_residual_outliers(residual, reference)
        np.testing.assert_array_equal(out, residual)


def _reference_cell_face_misalignment(mesh):
    """Independent per-face Python-loop oracle for
    `precompute_cell_face_misalignment`'s numba kernel (real perf bug this
    guards against: indexing `mesh.face_flux_points[f]` for every face
    lazily *constructs* a full FaceFluxPointGeometry object per access -
    1.88M such constructions measured to cost minutes on a production mesh,
    see `_cell_face_misalignment_kernel`'s docstring for the fix).

    Updated 2026-08-23 (see fr/face_flux_points_exact_normal.py module
    docstring): `own_dir_outward` used to be computed here via its own
    independent SP-grid Lagrange extrapolation of `adj_j` - an
    *approximation* of the true local metric direction, with real
    truncation error (worse at low order). The production kernel now reads
    the exact per-FP adj(J) row precomputed at mesh-load time instead of
    extrapolating - this reference must use the same exact values (still
    computed independently here, via `compute_exact_adj_rows` rather than
    the kernel's own precomputed arrays, so this remains a real oracle for
    the *reduction loop* logic, not a tautology) or it would be comparing
    the new kernel against a stale, less-accurate approximation of a
    different quantity.
    """
    from autoflowcfd.fr.face_flux_points_exact_normal import compute_exact_adj_rows

    fc = mesh.face_connectivity
    ffp_list = mesh.face_flux_points
    n_prism = mesh.n_prism_cells
    n1d = mesh.n_points_1d
    sps_1d = ffp_list._sps_1d

    owner_adj_row = compute_exact_adj_rows(
        fc.n_faces, n1d, sps_1d, n_prism,
        cell_arr=fc.owner_cell.astype(np.int64),
        axis_arr=ffp_list.owner_axis.astype(np.int64),
        side_arr=ffp_list.owner_side.astype(np.float64),
        prism_conn=mesh._fixed_prism_conn if mesh._fixed_prism_conn is not None else np.empty((0, 6), dtype=np.int64),
        tet_conn=mesh._fixed_tet_conn if mesh._fixed_tet_conn is not None else np.empty((0, 4), dtype=np.int64),
        node_coords=mesh._node_coords,
    )
    neighbor_adj_row = compute_exact_adj_rows(
        fc.n_faces, n1d, sps_1d, n_prism,
        cell_arr=fc.neighbor_cell.astype(np.int64),
        axis_arr=ffp_list.neighbor_axis.astype(np.int64),
        side_arr=ffp_list.neighbor_side.astype(np.float64),
        prism_conn=mesh._fixed_prism_conn if mesh._fixed_prism_conn is not None else np.empty((0, 6), dtype=np.int64),
        tet_conn=mesh._fixed_tet_conn if mesh._fixed_tet_conn is not None else np.empty((0, 4), dtype=np.int64),
        node_coords=mesh._node_coords,
        valid_mask=~fc.is_boundary,
    )

    def own_dir(row, side):
        mag = np.linalg.norm(row, axis=-1)
        return (row / np.maximum(mag[:, None], 1e-300)) * side

    cell_misalign = np.zeros(mesh.n_cells)
    for f in range(fc.n_faces):
        ffp = ffp_list[f]
        if ffp.owner_is_primary:
            owner_cell = int(fc.owner_cell[f])
            d = own_dir(owner_adj_row[f], ffp.owner_side)
            misalign = 1.0 - np.sum(d * ffp.true_normal, axis=-1)
            cell_misalign[owner_cell] = max(cell_misalign[owner_cell], float(misalign.max()))
        if (not fc.is_boundary[f]) and ffp.neighbor_is_primary:
            neighbor_cell = int(fc.neighbor_cell[f])
            d = own_dir(neighbor_adj_row[f], ffp.neighbor_side)
            misalign = 1.0 - np.sum(d * (-ffp.true_normal), axis=-1)
            cell_misalign[neighbor_cell] = max(cell_misalign[neighbor_cell], float(misalign.max()))
    return cell_misalign


class TestPrecomputeCellFaceMisalignment:
    @pytest.mark.parametrize("order", [0, 1, 2, 3])
    def test_matches_per_face_reference_loop(self, order):
        from .test_fr_residual_inviscid import _build_synthetic_mixed_mesh

        mesh = _build_synthetic_mixed_mesh(order)
        actual = precompute_cell_face_misalignment(mesh)
        expected = _reference_cell_face_misalignment(mesh)
        np.testing.assert_allclose(actual, expected, atol=1e-10)
