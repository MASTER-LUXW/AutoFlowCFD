"""含棱柱的体网格质量报告（`MeshQualityValidator.validate_volume_mesh` -> `quality_validator_mixed.py`，I-205）。

真实网格都带边界层棱柱，求解前质量门走的就是这条路径；此前没有任何单元测试经过它。网格：一个直三棱柱（底面
直角三角形、高 0.1）加一个立在棱柱顶面上的四面体。
"""

import numpy as np

from autoflowcfd.grid.schema import (
    BoundaryMap, GridMetadata, NodeArray, PrismCells, TetrahedralCells, VolumeMeshData,
)
from autoflowcfd.grid.validation.quality_validator import MeshQualityValidator

_NODES = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                   [0.0, 0.0, 0.1], [1.0, 0.0, 0.1], [0.0, 1.0, 0.1],
                   [0.25, 0.25, 0.9]])
_PRISM_VOLUME = 0.5 * 0.1
_TET_VOLUME = 0.5 * 0.8 / 3.0            # 底面积 0.5、高 0.9 - 0.1


def _mesh(tet):
    return VolumeMeshData(
        nodes=NodeArray(x=_NODES[:, 0].copy(), y=_NODES[:, 1].copy(), z=_NODES[:, 2].copy()),
        cells=TetrahedralCells(connectivity=np.array([tet], dtype=np.int32), volumes=np.array([_TET_VOLUME])),
        boundaries=BoundaryMap(groups={}, bc_types={}),
        metadata=GridMetadata(node_count=7, cell_count=2, boundary_groups=[], file_format="test"),
        prism_cells=PrismCells(connectivity=np.array([[0, 1, 2, 3, 4, 5]], dtype=np.int32),
                               volumes=np.array([_PRISM_VOLUME])),
    )


def test_mixed_mesh_report_counts_both_cell_kinds():
    report = MeshQualityValidator().validate_volume_mesh(_mesh([3, 4, 5, 6]), check_overlap=False)
    assert report.n_cells == 2 and report.n_nodes == 7
    assert report.negative_volumes == 0
    np.testing.assert_allclose([report.min_volume, report.max_volume], [_PRISM_VOLUME, _TET_VOLUME], rtol=1e-12)
    np.testing.assert_allclose(report.volume_ratio, _TET_VOLUME / _PRISM_VOLUME, rtol=1e-12)
    # BL（棱柱）与 core（四面体）分区统计
    assert report.bl_max_aspect_ratio > report.core_max_aspect_ratio > 0.0
    np.testing.assert_allclose(report.max_aspect_ratio, report.bl_max_aspect_ratio)


def test_inverted_tet_is_reported():
    report = MeshQualityValidator().validate_volume_mesh(_mesh([3, 5, 4, 6]), check_overlap=False)
    assert report.negative_volumes == 1
    assert not report.passed
    np.testing.assert_allclose(report.min_volume, _PRISM_VOLUME, rtol=1e-12)   # 统计只取正体积单元
