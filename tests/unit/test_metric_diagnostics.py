"""core/fr_operators/metric_diagnostics.py：面法向失配（与独立逐面参照实现对照）与网格度量报告的计数。"""

import numpy as np
import pytest

from autoflowcfd.core.fr_operators.metric_diagnostics import (
    SHAPE_QUALITY_WARN_THRESHOLD,
    log_mesh_metric_report,
    precompute_cell_face_misalignment,
)


def _reference_cell_face_misalignment(mesh):
    """`precompute_cell_face_misalignment` 的独立逐面 Python 循环参照实现：两侧 adj 行经 numpy 参照实现
    `compute_exact_adj_rows` 独立求出（不读核自己预先算好的数组），对归约循环是真正的参照。"""
    from autoflowcfd.fr.face_flux_points.exact_normal import compute_exact_adj_rows

    fc = mesh.face_connectivity
    ffp_list = mesh.face_flux_points
    n_prism = mesh.n_prism_cells
    n1d = mesh.n_points_1d
    sps_1d = ffp_list._sps_1d

    owner_adj_row = compute_exact_adj_rows(
        fc.n_faces, n1d, sps_1d, n_prism,
        cell_arr=fc.owner_cell.astype(np.int64),
        code_arr=fc.owner_cube_face.astype(np.int64),
        prism_conn=mesh._fixed_prism_conn if mesh._fixed_prism_conn is not None else np.empty((0, 6), dtype=np.int64),
        tet_conn=mesh._fixed_tet_conn if mesh._fixed_tet_conn is not None else np.empty((0, 4), dtype=np.int64),
        node_coords=mesh._node_coords,
    )
    neighbor_adj_row = compute_exact_adj_rows(
        fc.n_faces, n1d, sps_1d, n_prism,
        cell_arr=fc.neighbor_cell.astype(np.int64),
        code_arr=fc.neighbor_cube_face.astype(np.int64),
        prism_conn=mesh._fixed_prism_conn if mesh._fixed_prism_conn is not None else np.empty((0, 6), dtype=np.int64),
        tet_conn=mesh._fixed_tet_conn if mesh._fixed_tet_conn is not None else np.empty((0, 4), dtype=np.int64),
        node_coords=mesh._node_coords,
        valid_mask=~fc.is_boundary,
    )

    def own_dir(row):
        mag = np.linalg.norm(row, axis=-1)
        return row / np.maximum(mag[:, None], 1e-300)

    cell_misalign = np.zeros(mesh.n_cells)
    for f in range(fc.n_faces):
        if ffp_list.owner_is_primary[f]:
            owner_cell = int(fc.owner_cell[f])
            d = own_dir(owner_adj_row[f])
            misalign = 1.0 - np.sum(d * ffp_list.true_normal[f], axis=-1)
            cell_misalign[owner_cell] = max(cell_misalign[owner_cell], float(misalign.max()))
        if (not fc.is_boundary[f]) and ffp_list.neighbor_is_primary[f]:
            neighbor_cell = int(fc.neighbor_cell[f])
            d = own_dir(neighbor_adj_row[f])
            misalign = 1.0 - np.sum(d * (-ffp_list.true_normal[f]), axis=-1)
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


class TestMeshMetricReport:
    def test_counts_shape_and_misalignment_independently(self):
        q = np.full((5, 4), 0.8)
        q[1, 2] = 0.5 * SHAPE_QUALITY_WARN_THRESHOLD     # 单元 1 有一个解点形状质量低
        mis = np.zeros(5)
        mis[3] = 1.0 - np.cos(np.radians(5.0))           # 单元 3 法向失配 5°
        stats = log_mesh_metric_report(q, mis)
        assert stats == {"n_cells": 5, "n_shape_warn": 1, "n_misaligned": 1}

    def test_no_absolute_jacobian_size_criterion(self):
        """判据只看缩放雅可比（与单元物理尺寸无关）：贴壁薄单元形状正常就不报。"""
        stats = log_mesh_metric_report(np.full((3, 4), 0.7))
        assert stats["n_shape_warn"] == 0 and stats["n_misaligned"] == 0
