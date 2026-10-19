"""网格构造只算一遍 owner 侧 adj 行（I-35，2026-10-19，`face_metrics.py::build_face_metrics`）。

此前先用 numpy 参照实现（`compute_exact_face_normals_and_weights`，内部重算 owner adj 行，P2 峰值 ~5GB、初始化
10+ 分钟的那条路径）求法向 / 面积权重，随后 numba 版再算一遍 owner adj 行。现在法向 / 面积权重由 numba 的 adj 行
派生；numpy 版只作测试参照。
"""

from unittest.mock import patch

import numpy as np
import pytest

from autoflowcfd.fr.face_flux_points import exact_normal
from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh


@pytest.mark.parametrize("order", [1, 2])
def test_mesh_build_does_not_call_the_numpy_reference(order):
    def _forbidden(*a, **k):
        raise AssertionError("网格构造不应再调用 numpy 参照实现求 adj 行")

    with patch.object(exact_normal, "compute_exact_adj_rows", side_effect=_forbidden):
        mesh = _build_synthetic_mixed_mesh(order)
    assert mesh.face_flux_points.true_normal.shape[1] == (order + 1) ** 2


@pytest.mark.parametrize("order", [1, 2])
def test_normals_and_weights_match_the_numpy_reference(order):
    mesh = _build_synthetic_mixed_mesh(order)
    fc, ffp = mesh.face_connectivity, mesh.face_flux_points
    from autoflowcfd.fr.face_flux_points.geometry import CUBE_FACE_AXIS_SIDE
    from autoflowcfd.fr.quadrature_points import gauss_legendre

    sps_1d, w_1d = gauss_legendre(order + 1)
    codes = np.asarray(fc.owner_cube_face, dtype=np.int64)
    axis = np.array([v[0] for v in CUBE_FACE_AXIS_SIDE.values()])[codes]
    side = np.array([v[1] for v in CUBE_FACE_AXIS_SIDE.values()], dtype=float)[codes]
    ref_n, ref_w = exact_normal.compute_exact_face_normals_and_weights(
        n_faces=fc.n_faces, n1d=order + 1, sps_1d=sps_1d, weights_1d=w_1d, n_prism=mesh.n_prism_cells,
        owner_cell=np.asarray(fc.owner_cell, dtype=np.int64), owner_axis=axis.astype(np.int64), owner_side=side,
        owner_code=codes,
        prism_conn=mesh._fixed_prism_conn if mesh._fixed_prism_conn is not None else np.empty((0, 6), np.int64),
        tet_conn=mesh._fixed_tet_conn if mesh._fixed_tet_conn is not None else np.empty((0, 4), np.int64),
        node_coords=mesh._node_coords)
    np.testing.assert_allclose(ffp.true_normal, ref_n, rtol=0, atol=1e-13)
    np.testing.assert_allclose(ffp.true_area_weight, ref_w, rtol=1e-13, atol=1e-15 * np.abs(ref_w).max())
