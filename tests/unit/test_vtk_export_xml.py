"""postprocess/vtk_export_xml.py 处理棱柱+四面体混合网格的单元测试。

这里抓到的真实缺陷（2026-08-21 修复）：`export_xml` 只读
`grid_data.cells.connectivity`（四面体）并只用它构造 pyvista 网格——从不看
`grid_data.prism_cells`，于是每张带边界层棱柱挤出的网格（也就是本项目
实际产生的每张网格；棱柱边界层挤出是核心的、始终开启的功能）在导出为
默认/推荐的 .vtu 格式时静默丢失 100% 的棱柱单元。得到的 VTK 网格的单元数
（只有四面体）随后与解场数组（按全部单元，棱柱+四面体）对不上，被 VTK
自己的内部校验抓到并报错——不是一个静默的、看上去不对的文件，但对任何
混合网格仍是真实、可复现的失败（在真实的 cube_demo.nas 791492 单元算例上
确认：654512 个四面体 + 136980 个棱柱）。
"""

from types import SimpleNamespace

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from autoflowcfd.postprocess.vtk_export_xml import export_xml


def _build_mixed_grid_data(tmp_path):
    """一个很小的类 grid_data 对象，含 1 个棱柱 + 1 个四面体，与 `export_xml`
    读取的真实 VolumeMeshData 的结构一致（`.nodes.x/y/z`、
    `.cells.connectivity`、`.prism_cells.connectivity`）。
    """
    n_points = 11
    nodes = SimpleNamespace(
        x=np.linspace(0, 1, n_points),
        y=np.linspace(0, 1, n_points),
        z=np.linspace(0, 1, n_points),
        count=n_points,
    )
    tet_conn = np.array([[0, 1, 2, 3]], dtype=np.int64)
    prism_conn = np.array([[4, 5, 6, 7, 8, 9]], dtype=np.int64)
    return SimpleNamespace(
        nodes=nodes,
        cells=SimpleNamespace(connectivity=tet_conn),
        prism_cells=SimpleNamespace(connectivity=prism_conn),
    )


class _StubExporter:
    """VTKExporter 的最小替身——只有 `export_xml` 实际用到的属性/方法。"""

    _FIELD_LABELS = {"pressure": "Pressure"}

    def __init__(self, grid_data, n_cells):
        self.grid_data = grid_data
        self._n_cells = n_cells

    def _cell_fields(self, fields):
        return {"pressure": np.arange(self._n_cells, dtype=np.float64)}

    def _point_fields(self, cell_fields, n_points):
        return {}


class TestExportXmlMixedCells:
    def test_prism_and_tet_cells_both_present_in_output(self, tmp_path):
        grid_data = _build_mixed_grid_data(tmp_path)
        exporter = _StubExporter(grid_data, n_cells=2)  # 1 prism + 1 tet
        out_path = tmp_path / "mixed.vtu"

        export_xml(exporter, out_path, fields=["pressure"], binary=True)

        result = pv.read(str(out_path))
        assert result.n_cells == 2
        celltype_counts = {t: int((result.celltypes == t).sum()) for t in set(result.celltypes)}
        assert celltype_counts.get(int(pv.CellType.WEDGE), 0) == 1
        assert celltype_counts.get(int(pv.CellType.TETRA), 0) == 1
        # 全局顺序：先棱柱、后四面体（与本项目的单元编号约定一致，见
        # vtk_export.py::_VTK_WEDGE 文档）——单元 0 的场值（0.0）必须落在棱柱上，
        # 而不是四面体上。
        assert result.celltypes[0] == int(pv.CellType.WEDGE)
        assert result.celltypes[1] == int(pv.CellType.TETRA)

    def test_tet_only_mesh_still_exports_correctly(self, tmp_path):
        """没有 `prism_cells`（或为空）时必须退回原来只有四面体的路径，行为不变。"""
        n_points = 4
        nodes = SimpleNamespace(
            x=np.array([0.0, 1.0, 0.0, 0.0]),
            y=np.array([0.0, 0.0, 1.0, 0.0]),
            z=np.array([0.0, 0.0, 0.0, 1.0]),
            count=n_points,
        )
        tet_conn = np.array([[0, 1, 2, 3]], dtype=np.int64)
        grid_data = SimpleNamespace(
            nodes=nodes,
            cells=SimpleNamespace(connectivity=tet_conn),
            prism_cells=None,
        )
        exporter = _StubExporter(grid_data, n_cells=1)
        out_path = tmp_path / "tet_only.vtu"

        export_xml(exporter, out_path, fields=["pressure"], binary=True)

        result = pv.read(str(out_path))
        assert result.n_cells == 1
        assert result.celltypes[0] == int(pv.CellType.TETRA)
