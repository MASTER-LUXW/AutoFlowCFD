"""Unit tests for Python API."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

from autoflowcfd import AutoFlowCFDAPI, create_api, get_version


class TestAutoFlowCFDAPI:
    """Test suite for AutoFlowCFDAPI class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.api = AutoFlowCFDAPI()

    def test_initialization(self) -> None:
        """Test API initialization."""
        assert self.api.verbose is False
        assert self.api._config_loader is not None

    def test_initialization_verbose(self) -> None:
        """Test API initialization with verbose mode."""
        api = AutoFlowCFDAPI(verbose=True)
        assert api.verbose is True

    def test_get_version(self) -> None:
        """Test version retrieval."""
        version = self.api.get_version()
        assert version == "0.1.0"

    def test_check_environment(self) -> None:
        """Test environment check."""
        env_info = self.api.check_environment()
        assert "python_version" in env_info
        assert "platform" in env_info
        assert "autoflowcfd_version" in env_info
        assert "gpu_available" in env_info

    def test_create_steady_config(self) -> None:
        """Test steady config creation."""
        config = self.api.create_steady_config(
            backend="gpu",
            order=3,
            turbulence="sst_kw"
        )
        assert config.backend.value == "gpu"
        assert config.order == 3
        assert config.turbulence.value == "sst_kw"

    def test_create_transient_config(self) -> None:
        """Test transient config creation."""
        config = self.api.create_transient_config(
            backend="cpu",
            mode="ddes",
            dt=1e-4,
            total_time=0.3
        )
        assert config.backend.value == "cpu"
        assert config.dt == 1e-4
        assert config.total_time == 0.3

    @patch('autoflowcfd.api_grid_ops.NASParser')
    def test_load_grid(self, mock_parser_class: Mock) -> None:
        """Test grid loading."""
        # Mock parser and grid data
        mock_parser = MagicMock()
        mock_grid_data = MagicMock()
        mock_grid_data.get_node_count.return_value = 1000
        mock_grid_data.get_cell_count.return_value = 2000
        
        mock_parser.parse.return_value = mock_grid_data
        mock_parser_class.return_value = mock_parser
        
        # Test load_grid
        grid = self.api.load_grid("test.nas", validate=False)
        
        assert grid == mock_grid_data
        mock_parser_class.assert_called_once()

    def test_get_grid_info(self) -> None:
        """Test grid info extraction."""
        mock_grid = MagicMock()
        mock_grid.node_count = 1000
        mock_grid.cell_count = 2000
        mock_grid.boundaries.boundary_names = ["BODY", "INLET"]
        mock_grid.boundaries.get_boundary_cells.side_effect = lambda x: list(range(10))

        info = self.api.get_grid_info(mock_grid)

        assert info["node_count"] == 1000
        assert info["cell_count"] == 2000
        assert "boundary_groups" in info

    def test_validate_grid(self) -> None:
        """Test grid validation."""
        mock_grid = MagicMock()
        
        with patch('autoflowcfd.api_grid_ops.GridValidator') as mock_validator_class:
            mock_validator = MagicMock()
            mock_validator.validate.return_value = {
                "error_count": 0,
                "warning_count": 0
            }
            mock_validator_class.return_value = mock_validator
            
            report = self.api.validate_grid(mock_grid)
            
            assert report["error_count"] == 0

    @patch('autoflowcfd.cli.solve_wall_distance.compute_wall_distance_for_solver')
    @patch('autoflowcfd.grid.high_order.high_order_mesh.HighOrderMesh')
    @patch('autoflowcfd.api.FRSolver')
    def test_run_steady(
        self, mock_solver_class: Mock, mock_mesh_class: Mock, mock_wall_distance: Mock
    ) -> None:
        """Test steady simulation.

        run_steady 需要先把 volume_mesh 升格成 HighOrderMesh 再构造
        FRSolver（见 api.py::run_steady 文档字符串——此前这里直接把
        表面网格传给 FRSolver 是那个从未跑通过的 bug 本身），所以这里
        同时 mock 掉 HighOrderMesh 构造/wall distance 计算这两步，
        不依赖真实网格数据。
        """
        mock_volume_mesh = MagicMock()
        mock_mesh_class.return_value.load_from_volume_mesh.return_value = None

        mock_result = MagicMock()
        mock_result.converged = True
        mock_result.iterations = 1000

        mock_solver = MagicMock()
        mock_solver.solve.return_value = mock_result
        mock_solver_class.return_value = mock_solver

        result = self.api.run_steady(
            mock_volume_mesh,
            backend="cpu",
            order=2,
            max_iter=500
        )

        assert result.converged is True
        assert result.iterations == 1000
        mock_solver_class.assert_called_once()
        assert self.api.solver is mock_solver

    @patch('autoflowcfd.cli.solve_wall_distance.compute_wall_distance_for_solver')
    @patch('autoflowcfd.grid.high_order.high_order_mesh.HighOrderMesh')
    @patch('autoflowcfd.api.TransientSolver')
    def test_run_transient(
        self, mock_solver_class: Mock, mock_mesh_class: Mock, mock_wall_distance: Mock
    ) -> None:
        """Test transient simulation."""
        mock_volume_mesh = MagicMock()
        mock_mesh_class.return_value.load_from_volume_mesh.return_value = None

        mock_result = MagicMock()
        mock_result.converged = True
        mock_result.iterations = 3000

        mock_solver = MagicMock()
        mock_solver.solve.return_value = mock_result
        mock_solver_class.return_value = mock_solver

        result = self.api.run_transient(
            mock_volume_mesh,
            mode="ddes",
            physical_time=0.3,
            dt=1e-4
        )

        assert result.iterations == 3000
        mock_solver_class.assert_called_once()
        # physical_time/dt 决定 max_iter，solve() 应该按算出来的迭代数调用
        _, solve_kwargs = mock_solver.solve.call_args
        assert solve_kwargs["max_iter"] == int(0.3 / 1e-4)

    def test_resume_simulation_missing_checkpoint(self) -> None:
        """Test resume raises FileNotFoundError for missing checkpoint."""
        with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
            self.api.resume_simulation("nonexistent.h5")

    def test_calculate_coefficients_placeholder(self) -> None:
        """Test coefficient calculation placeholder."""
        mock_result = MagicMock()
        coeffs = self.api.calculate_coefficients(mock_result)
        
        assert "Cd" in coeffs
        assert "Cl" in coeffs
        assert coeffs["Cd"] == 0.0  # Placeholder value

    def test_export_vtk_without_prior_solve_raises(self) -> None:
        """export_vtk 需要先跑过 run_steady/run_transient（self.solver/
        self.volume_mesh 均已设置）——没有的话应该给出清楚的 ValueError，
        不是此前那种"传了错的 result.solution/grid_data 最终在别处
        抛不相关异常"（V2.0 专家组评审逐行核实此前实现从未跑通过）。
        """
        assert self.api.solver is None
        assert self.api.volume_mesh is None
        with pytest.raises(ValueError, match="run_steady/run_transient"):
            self.api.export_vtk(result=None, filename="output.vtk")

    def test_get_convergence_history_placeholder(self) -> None:
        """Test convergence history placeholder."""
        mock_result = MagicMock()
        history = self.api.get_convergence_history(mock_result)
        
        assert "iterations" in history
        assert "residuals" in history
        assert history["iterations"] == []

    def test_load_config(self, tmp_path: Path) -> None:
        """Test config loading."""
        # Create temporary config file
        config_file = tmp_path / "test.yaml"
        config_file.write_text("""
mode: steady
backend: cpu
order: 3
turbulence: sst_kw
max_iter: 1000
""")
        
        config = self.api.load_config(str(config_file))
        assert config is not None


class TestCreateAPI:
    """Test suite for create_api convenience function."""

    def test_create_api_default(self) -> None:
        """Test API creation with defaults."""
        api = create_api()
        assert isinstance(api, AutoFlowCFDAPI)
        assert api.verbose is False

    def test_create_api_verbose(self) -> None:
        """Test API creation with verbose mode."""
        api = create_api(verbose=True)
        assert api.verbose is True


class TestGetVersion:
    """Test suite for get_version function."""

    def test_version_format(self) -> None:
        """Test version string format."""
        version = get_version()
        assert isinstance(version, str)
        assert len(version) > 0

    def test_version_matches_module(self) -> None:
        """Test version matches module __version__."""
        import autoflowcfd
        assert get_version() == autoflowcfd.__version__


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
