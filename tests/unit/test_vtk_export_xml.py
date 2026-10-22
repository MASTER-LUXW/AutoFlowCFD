"""低阶 VTK 导出在棱柱+四面体混合网格上的正确性。

- 单元：XML 输出必须同时含棱柱与四面体，顺序为棱柱在前、四面体在后（与单元场同序）。
- 节点（2026-10-17 修复）：单元 -> 节点插值此前只用四面体连接表去配"棱柱在前"的全部单元场——
  第 i 个四面体拿到第 i 个单元（其实是棱柱）的值，只属于棱柱的节点（边界层内部节点）没有数据、
  被填成兜底值。现在按混合连接逐单元体积加权。
"""

from types import SimpleNamespace

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from autoflowcfd.postprocess.solution_vector import SolutionVector  # noqa: E402
from autoflowcfd.postprocess._field_utils import cell_to_node  # noqa: E402
from autoflowcfd.postprocess.vtk_export import VTKExporter  # noqa: E402
from autoflowcfd.postprocess.vtk_export_xml import export_volume  # noqa: E402


def _mixed_grid():
    """1 个棱柱（节点 0..5）+ 1 个四面体（节点 3,4,5,6，与棱柱共享顶面）；体积 0.5 与 1/6。"""
    xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [0, 0, 2.0]])
    nodes = SimpleNamespace(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], count=len(xyz))
    vols = np.array([0.5, 1.0 / 6.0])
    return SimpleNamespace(
        nodes=nodes, node_count=len(xyz), cell_count=2,
        cells=SimpleNamespace(connectivity=np.array([[3, 4, 5, 6]])),
        prism_cells=SimpleNamespace(connectivity=np.array([[0, 1, 2, 3, 4, 5]])),
        get_cell_volumes=lambda: vols,
    )


def test_cells_are_prism_first_in_xml(tmp_path):
    out = tmp_path / "mixed.vtu"
    export_volume(out, _mixed_grid(), {"pressure": np.array([1.0, 2.0])}, {}, binary=True)
    result = pv.read(str(out))
    assert result.n_cells == 2
    assert result.celltypes[0] == int(pv.CellType.WEDGE)
    assert result.celltypes[1] == int(pv.CellType.TETRA)
    np.testing.assert_array_equal(result.cell_data["Pressure"], [1.0, 2.0])


def test_cell_to_node_uses_both_cell_types():
    grid = _mixed_grid()
    node = cell_to_node(grid, np.array([10.0, 40.0]))
    np.testing.assert_allclose(node[[0, 1, 2]], 10.0)            # 只属于棱柱
    np.testing.assert_allclose(node[6], 40.0)                     # 只属于四面体
    shared = (0.5 * 10.0 + (1.0 / 6.0) * 40.0) / (0.5 + 1.0 / 6.0)
    np.testing.assert_allclose(node[[3, 4, 5]], shared)           # 两者共享，按体积加权


@pytest.mark.parametrize("fmt, name", [("xml", "m.vtu"), ("legacy", "m.vtk")])
def test_point_data_on_mixed_mesh(tmp_path, fmt, name):
    """POINT_DATA 的压力：只属于棱柱的节点取棱柱值，只属于四面体的取四面体值。"""
    grid = _mixed_grid()
    gamma = 1.4
    p = np.array([101000.0, 99000.0])
    rho = np.array([1.2, 1.1])
    U = np.zeros((2, 5))
    U[:, 0] = rho
    U[:, 4] = p / (gamma - 1.0)
    path = VTKExporter(grid, SolutionVector(U)).export(
        str(tmp_path / name), fields=["pressure"], format=fmt, binary=False)
    result = pv.read(str(path))
    np.testing.assert_allclose(result.point_data["Pressure"][[0, 1, 2]], p[0], rtol=1e-6)
    np.testing.assert_allclose(result.point_data["Pressure"][6], p[1], rtol=1e-6)


def test_missing_requested_field_raises():
    grid = _mixed_grid()
    exporter = VTKExporter(grid, SolutionVector(np.ones((2, 5))))
    with pytest.raises(ValueError, match="湍流场"):
        exporter.export("unused.vtk", fields=["k"])
    with pytest.raises(ValueError, match="Q 准则"):
        exporter.export("unused.vtk", fields=["q_criterion"])
