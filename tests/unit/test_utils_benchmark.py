"""Unit tests for cli/utils_commands.py::benchmark's mesh-construction and
residual-call wiring.

Real bug caught here (fixed 2026-08-22): `benchmark` called
`HighOrderMesh(grid_data, order=order)` - but `HighOrderMesh.__init__` takes
only `order` (no positional data argument at all), and `grid_data` here was
a *surface* GridData (from `NASParser.parse()`), not the VolumeMeshData
`load_from_volume_mesh` needs anyway. Every invocation crashed immediately
with `TypeError: HighOrderMesh.__init__() got multiple values for argument
'order'` before ever reaching the actual benchmark loop - this command had
never successfully run once. It also called `compute_inviscid_residual_fr
(mesh, U_init)`, the wrong argument order/arity for a function whose real
signature is `(U, mesh, ops, boundary_ghost_provider=None)`.

Mesh generation itself (BL extrusion + tetgen) is too heavy to run in a
unit test, so this patches `generate_volume_mesh_from_surface` to return a
tiny synthetic VolumeMeshData and asserts the *construction and call
pattern* is correct - the same pattern proven separately, many times, in
this session's real end-to-end CLI runs against cube_demo.nas.
"""

from unittest.mock import patch, MagicMock
from types import SimpleNamespace

import numpy as np
import pytest
from click.testing import CliRunner

from autoflowcfd.cli.main import cli
from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh


def _tiny_volume_mesh_data():
    """The exact node/connectivity data from `_build_synthetic_mixed_mesh`
    (2 tets + 2 prisms, already validated many times over elsewhere in this
    test suite - see test_fr_residual_inviscid.py, test_troubled_cell.py),
    wrapped as a stand-in VolumeMeshData."""
    nodes = np.array(
        [
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1],
            [10, 0, 0], [11, 0, 0], [10, 1, 0], [10, 0, 1], [11, 0, 1], [10, 1, 1],
            [9, -1, 0], [9, -1, 1],
        ],
        dtype=float,
    )
    tet_conn = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int32)
    prism_conn = np.array(
        [
            [5, 6, 7, 8, 9, 10],
            [5, 7, 11, 8, 10, 12],
        ],
        dtype=np.int32,
    )
    return SimpleNamespace(
        cell_count=len(tet_conn) + len(prism_conn),
        nodes=SimpleNamespace(get_coordinates=lambda: nodes),
        cells=SimpleNamespace(connectivity=tet_conn),
        prism_cells=SimpleNamespace(connectivity=prism_conn),
        boundaries=None,
    )


class TestBenchmarkMeshConstruction:
    def test_benchmark_builds_and_solves_without_crashing(self):
        """End-to-end through the CLI command (with mesh generation mocked
        out) - must reach the residual loop and print results, not crash on
        HighOrderMesh construction or the residual call signature."""
        with patch(
            "autoflowcfd.grid.nas_io.parser_core.NASParser.parse"
        ) as mock_parse, patch(
            "autoflowcfd.grid.nas_io.parser_core.NASParser.generate_volume_mesh_from_surface"
        ) as mock_gen:
            mock_parse.return_value = MagicMock()
            mock_gen.return_value = _tiny_volume_mesh_data()

            runner = CliRunner()
            # order=0 (P0) keeps the residual kernels cheap for a unit test.
            result = runner.invoke(
                cli,
                ["utils", "benchmark", __file__, "-n", "1", "-p", "1", "--json"],
            )

        assert result.exit_code == 0, result.output
        import json
        payload = json.loads(result.stdout)
        assert payload["status"] == "success"
        assert payload["n_cells"] == 4  # 2 tet + 2 prism
