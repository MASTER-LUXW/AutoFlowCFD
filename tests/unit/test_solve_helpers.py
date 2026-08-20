"""Unit tests for cli/solve_helpers.py's load_mesh_for_solver - the
dispatch logic between .pkl and .nas volume-mesh input, and the
求解前质量门 enforcement layered on top of both.

The underlying parsers this dispatches to (mesh_external_import.
import_external_volume_mesh, its own call into parse_volume_mesh_nas) have
no dedicated test coverage of their own and constructing a real valid
volume-mesh NAS fixture is out of scope here - these tests mock that call
boundary instead, so what's actually under test is load_mesh_for_solver's
OWN logic: which branch it takes per extension, that it enforces
--surface-mesh being required for .nas, and that the quality gate blocks
(unless skipped) regardless of which loading path produced the mesh.
"""

import pickle
from unittest.mock import MagicMock, patch

import click
import pytest

from autoflowcfd.cli.solve_helpers import load_mesh_for_solver


def _fake_volume_mesh_data():
    """A minimal stand-in good enough for HighOrderMesh.load_from_volume_mesh
    and the print()s in load_mesh_for_solver to not blow up - the solver/mesh
    machinery itself is mocked out in every test here, only load_mesh_for_
    solver's own dispatch/gate logic is under test."""
    vm = MagicMock()
    vm.nodes.count = 10
    vm.cell_count = 5
    return vm


def _fake_report(passed: bool):
    report = MagicMock()
    report.passed = passed
    report.summary.return_value = "<fake quality report>"
    return report


@pytest.fixture
def fake_pkl(tmp_path):
    """A .pkl file containing a genuinely picklable minimal VolumeMeshData
    (a MagicMock cannot be pickled), for tests that need a real file on disk
    for load_mesh_for_solver's own open()/pickle.load()."""
    import numpy as np
    from autoflowcfd.grid.structures import (
        BoundaryMap, GridMetadata, NodeArray, TetrahedralCells, VolumeMeshData,
    )
    vm = VolumeMeshData(
        nodes=NodeArray(x=np.zeros(4), y=np.zeros(4), z=np.zeros(4)),
        cells=TetrahedralCells(connectivity=np.array([[0, 1, 2, 3]], dtype=np.int32), volumes=np.array([1.0])),
        boundaries=BoundaryMap(groups={}, bc_types={}),
        metadata=GridMetadata(node_count=4, cell_count=1, boundary_groups=[], file_format='test'),
    )
    pkl_path = tmp_path / "volume.pkl"
    with open(pkl_path, 'wb') as f:
        pickle.dump(vm, f)
    return pkl_path, vm


class TestLoadMeshForSolverExtensionDispatch:
    def test_unsupported_extension_is_rejected(self, tmp_path):
        bogus = tmp_path / "mesh.su2"
        bogus.write_text("dummy")
        with pytest.raises(click.ClickException, match="generate-volume"):
            load_mesh_for_solver(str(bogus), order=2)

    def test_nas_without_surface_mesh_is_rejected(self, tmp_path):
        volume_nas = tmp_path / "volume.nas"
        volume_nas.write_text("dummy")
        with pytest.raises(click.ClickException, match="--surface-mesh"):
            load_mesh_for_solver(str(volume_nas), order=2)

    def test_pkl_not_a_volume_mesh_data_is_rejected(self, tmp_path):
        bad_pkl = tmp_path / "not_a_mesh.pkl"
        with open(bad_pkl, 'wb') as f:
            pickle.dump({"not": "a VolumeMeshData"}, f)
        with pytest.raises(click.ClickException, match="VolumeMeshData"):
            load_mesh_for_solver(str(bad_pkl), order=2)

    @patch("autoflowcfd.cli.solve_mesh_loader.HighOrderMesh")
    @patch("autoflowcfd.grid.mesh_gen.utils.mesh_external_import.import_external_volume_mesh")
    def test_nas_with_surface_mesh_dispatches_to_external_import(
        self, mock_import, mock_high_order_mesh, tmp_path
    ):
        volume_nas = tmp_path / "volume.nas"
        volume_nas.write_text("dummy")
        surface_nas = tmp_path / "surface.nas"
        surface_nas.write_text("dummy")
        mock_import.return_value = (_fake_volume_mesh_data(), _fake_report(passed=True))

        mesh, volume_data = load_mesh_for_solver(
            str(volume_nas), order=2, surface_mesh=str(surface_nas)
        )

        mock_import.assert_called_once_with(str(volume_nas), str(surface_nas))
        assert volume_data is mock_import.return_value[0]

    @patch("autoflowcfd.cli.solve_mesh_loader.HighOrderMesh")
    def test_pkl_path_loads_without_needing_surface_mesh(self, mock_high_order_mesh, fake_pkl):
        pkl_path, vm = fake_pkl
        with patch(
            "autoflowcfd.grid.validation.quality_validator.MeshQualityValidator"
        ) as mock_validator_cls:
            mock_validator_cls.return_value.validate_volume_mesh.return_value = _fake_report(passed=True)
            mesh, volume_data = load_mesh_for_solver(str(pkl_path), order=2)

        # pickle.load produces a new object, not the same identity as `vm` -
        # compare content instead.
        assert volume_data.nodes.count == vm.nodes.count
        assert volume_data.cell_count == vm.cell_count


class TestLoadMeshForSolverQualityGate:
    @patch("autoflowcfd.cli.solve_mesh_loader.HighOrderMesh")
    def test_failing_quality_gate_blocks_pkl_input(self, mock_high_order_mesh, fake_pkl):
        pkl_path, _vm = fake_pkl
        with patch(
            "autoflowcfd.grid.validation.quality_validator.MeshQualityValidator"
        ) as mock_validator_cls:
            mock_validator_cls.return_value.validate_volume_mesh.return_value = _fake_report(passed=False)
            with pytest.raises(click.ClickException, match="质量门"):
                load_mesh_for_solver(str(pkl_path), order=2)

    @patch("autoflowcfd.cli.solve_mesh_loader.HighOrderMesh")
    def test_skip_quality_check_bypasses_a_failing_gate(self, mock_high_order_mesh, fake_pkl):
        pkl_path, vm = fake_pkl
        # No MeshQualityValidator patch needed here - skip_quality_check=True
        # must short-circuit before it's ever constructed.
        mesh, volume_data = load_mesh_for_solver(str(pkl_path), order=2, skip_quality_check=True)
        assert volume_data.nodes.count == vm.nodes.count

    @patch("autoflowcfd.cli.solve_mesh_loader.HighOrderMesh")
    @patch("autoflowcfd.grid.mesh_gen.utils.mesh_external_import.import_external_volume_mesh")
    def test_failing_quality_gate_blocks_nas_input_too(
        self, mock_import, mock_high_order_mesh, tmp_path
    ):
        """The .nas path already gets a report back from import_external_
        volume_mesh itself - the gate must still apply to it, not just the
        freshly-computed report the .pkl path takes."""
        volume_nas = tmp_path / "volume.nas"
        volume_nas.write_text("dummy")
        surface_nas = tmp_path / "surface.nas"
        surface_nas.write_text("dummy")
        mock_import.return_value = (_fake_volume_mesh_data(), _fake_report(passed=False))

        with pytest.raises(click.ClickException, match="质量门"):
            load_mesh_for_solver(str(volume_nas), order=2, surface_mesh=str(surface_nas))
