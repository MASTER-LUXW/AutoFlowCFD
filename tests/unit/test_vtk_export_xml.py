"""Unit tests for postprocess/vtk_export_xml.py's mixed prism+tet handling.

Real bug caught here (fixed 2026-08-21): `export_xml` only ever read
`grid_data.cells.connectivity` (tetrahedra) and built the pyvista grid from
that alone - it never looked at `grid_data.prism_cells`, so every mesh with
a boundary-layer prism extrusion (which is every mesh this project actually
produces; prism BL extrusion is a core, always-on feature) silently lost
100% of its prism cells when exported to the default/recommended .vtu
format. The resulting VTK grid's cell count (tet-only) then mismatched the
solution field arrays (sized for all cells, prism+tet), which VTK's own
internal validation caught and raised on - not a silent wrong-looking file,
but still a real, reproducible failure for any hybrid mesh (confirmed on
the actual cube_demo.nas 791492-cell case: 654512 tets + 136980 prisms).
"""

from types import SimpleNamespace

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from autoflowcfd.postprocess.vtk_export_xml import export_xml


def _build_mixed_grid_data(tmp_path):
    """A tiny grid_data-like object with 1 prism + 1 tet cell, matching the
    real VolumeMeshData shape `export_xml` reads (`.nodes.x/y/z`,
    `.cells.connectivity`, `.prism_cells.connectivity`)."""
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
    """Minimal stand-in for VTKExporter - only the attributes/methods
    `export_xml` actually touches."""

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
        # Global ordering: prisms first, then tets (matches this project's
        # cell-index convention, see vtk_export.py::_VTK_WEDGE docs) - cell
        # 0's field value (0.0) must land on the prism, not the tet.
        assert result.celltypes[0] == int(pv.CellType.WEDGE)
        assert result.celltypes[1] == int(pv.CellType.TETRA)

    def test_tet_only_mesh_still_exports_correctly(self, tmp_path):
        """No `prism_cells` (or an empty one) must fall back to the
        original tet-only path, unchanged."""
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
