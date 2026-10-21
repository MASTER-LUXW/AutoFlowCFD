"""四面体体积只有一份定义（I-134）：有符号体积 `quality_metrics.compute_tetrahedron_volumes`；单元属性存储的
物理体积是它的绝对值（`TetrahedralCells.compute_volumes`）。此前存储用的那份另写一遍行列式公式，验证器上还有两层
零附加逻辑的薄包装。面网格校验器的雅可比与三角形面积同一个公式。"""

import numpy as np

from autoflowcfd.grid.schema.grid_cells import TetrahedralCells
from autoflowcfd.grid.schema.grid_nodes import NodeArray
from autoflowcfd.grid.validation import quality_validator, quality_validator_metrics
from autoflowcfd.grid.validation.quality_metrics import compute_tetrahedron_volumes, compute_triangle_areas
from autoflowcfd.grid.validation.validator_jacobian import triangle_jacobians


def test_storage_volume_is_the_absolute_signed_volume():
    pts = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0.3, 0.2, -0.5]])
    conn = np.array([[0, 1, 2, 3], [0, 2, 1, 3], [0, 1, 2, 4]])        # 第二个与第一个反向
    signed = compute_tetrahedron_volumes(pts, conn)
    assert signed[0] > 0 > signed[1]
    nodes = NodeArray(x=pts[:, 0].copy(), y=pts[:, 1].copy(), z=pts[:, 2].copy())
    np.testing.assert_array_equal(TetrahedralCells.compute_volumes(nodes, conn), np.abs(signed))


def test_thin_wrappers_are_gone():
    assert not hasattr(quality_validator.MeshQualityValidator, "_compute_tetrahedron_volumes")
    assert not hasattr(quality_validator_metrics, "compute_tetrahedron_volumes")


def test_triangle_jacobian_is_twice_the_area():
    pts = np.random.default_rng(1).random((6, 3))
    tri = np.array([[0, 1, 2], [3, 4, 5], [1, 3, 5]])
    np.testing.assert_allclose(triangle_jacobians(pts, tri), 2.0 * compute_triangle_areas(pts, tri), rtol=1e-15)
