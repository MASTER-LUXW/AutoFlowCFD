"""Unit tests for CLI commands."""

import pytest
from click.testing import CliRunner
from autoflowcfd.cli.main import cli


class TestCLIGridCommands:
    """Test suite for grid subcommands."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_grid_help(self) -> None:
        """Test grid command help."""
        result = self.runner.invoke(cli, ["grid", "--help"])
        assert result.exit_code == 0
        assert "parse" in result.output
        assert "validate" in result.output
        assert "info" in result.output

    def test_grid_parse_help(self) -> None:
        """Test grid parse help."""
        result = self.runner.invoke(cli, ["grid", "parse", "--help"])
        assert result.exit_code == 0
        assert "--output" in result.output
        assert "--streaming" in result.output

    def test_grid_validate_help(self) -> None:
        """Test grid validate help."""
        result = self.runner.invoke(cli, ["grid", "validate", "--help"])
        assert result.exit_code == 0
        assert "--report" in result.output
        assert "--threshold-aspect-ratio" in result.output

    def test_grid_info_help(self) -> None:
        """Test grid info help."""
        result = self.runner.invoke(cli, ["grid", "info", "--help"])
        assert result.exit_code == 0
        assert "--json" in result.output


class TestCLISolveCommands:
    """Test suite for solve subcommands."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_solve_help(self) -> None:
        """Test solve command help."""
        result = self.runner.invoke(cli, ["solve", "--help"])
        assert result.exit_code == 0
        assert "steady" in result.output
        assert "transient" in result.output
        assert "resume" in result.output

    def test_solve_steady_help(self) -> None:
        """Test solve steady help."""
        result = self.runner.invoke(cli, ["solve", "steady", "--help"])
        assert result.exit_code == 0
        assert "--backend" in result.output
        assert "--order" in result.output
        assert "--turbulence" in result.output
        assert "--max-iter" in result.output
        assert "--skip-quality-check" in result.output
        assert "--surface-mesh" in result.output

    def test_solve_transient_help(self) -> None:
        """Test solve transient help."""
        result = self.runner.invoke(cli, ["solve", "transient", "--help"])
        assert result.exit_code == 0
        assert "--physical-time" in result.output
        assert "--dt" in result.output
        assert "--time-method" in result.output
        assert "--skip-quality-check" in result.output
        assert "--surface-mesh" in result.output

    def test_solve_steady_rejects_unsupported_extension(self, tmp_path) -> None:
        """Neither .pkl nor .nas - solve steady must reject with a clear
        pointer to grid generate-volume/import-volume."""
        bogus_file = tmp_path / "mesh.su2"
        bogus_file.write_text("dummy\n")
        result = self.runner.invoke(cli, ["solve", "steady", str(bogus_file)])
        assert result.exit_code != 0
        assert "generate-volume" in result.output or "import-volume" in result.output

    def test_solve_transient_rejects_unsupported_extension(self, tmp_path) -> None:
        """Same as steady's own version of this check, for transient."""
        bogus_file = tmp_path / "mesh.su2"
        bogus_file.write_text("dummy\n")
        result = self.runner.invoke(cli, ["solve", "transient", str(bogus_file)])
        assert result.exit_code != 0
        assert "generate-volume" in result.output or "import-volume" in result.output

    def test_solve_steady_nas_without_surface_mesh_is_rejected(self, tmp_path) -> None:
        """A .nas volume mesh is now accepted by solve steady, but only
        together with --surface-mesh (needed to attribute WALL/INLET/OUTLET
        boundary groups) - passed alone it must be rejected, not silently
        solved with no boundary conditions at all."""
        volume_nas = tmp_path / "volume.nas"
        volume_nas.write_text("$ dummy nas file\n")
        result = self.runner.invoke(cli, ["solve", "steady", str(volume_nas)])
        assert result.exit_code != 0
        assert "--surface-mesh" in result.output

    def test_solve_transient_nas_without_surface_mesh_is_rejected(self, tmp_path) -> None:
        """Same as steady's own version of this check, for transient."""
        volume_nas = tmp_path / "volume.nas"
        volume_nas.write_text("$ dummy nas file\n")
        result = self.runner.invoke(cli, ["solve", "transient", str(volume_nas)])
        assert result.exit_code != 0
        assert "--surface-mesh" in result.output

    def test_solve_resume_help(self) -> None:
        """Test solve resume help."""
        result = self.runner.invoke(cli, ["solve", "resume", "--help"])
        assert result.exit_code == 0
        assert "checkpoint" in result.output.lower()

    def test_solve_status_help(self) -> None:
        """Test solve status help."""
        result = self.runner.invoke(cli, ["solve", "status", "--help"])
        assert result.exit_code == 0


class TestCLIPostCommands:
    """Test suite for post subcommands."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_post_help(self) -> None:
        """Test post command help."""
        result = self.runner.invoke(cli, ["post", "--help"])
        assert result.exit_code == 0
        assert "coefficients" in result.output
        assert "export-vtk" in result.output
        assert "convergence" in result.output

    def test_post_coefficients_help(self) -> None:
        """Test post coefficients help."""
        result = self.runner.invoke(cli, ["post", "coefficients", "--help"])
        assert result.exit_code == 0
        assert "--case" in result.output
        assert "--reference-area" in result.output

    def test_post_export_vtk_help(self) -> None:
        """Test post export-vtk help."""
        result = self.runner.invoke(cli, ["post", "export-vtk", "--help"])
        assert result.exit_code == 0
        assert "--output" in result.output

    def test_post_report_help(self) -> None:
        """Test post report help."""
        result = self.runner.invoke(cli, ["post", "report", "--help"])
        assert result.exit_code == 0
        assert "--format" in result.output

    def test_post_convergence_help(self) -> None:
        """Test post convergence help."""
        result = self.runner.invoke(cli, ["post", "convergence", "--help"])
        assert result.exit_code == 0

    def test_post_transient_mean_help(self) -> None:
        """Test post transient-mean help."""
        result = self.runner.invoke(cli, ["post", "transient-mean", "--help"])
        assert result.exit_code == 0

    def test_post_transient_rms_help(self) -> None:
        """Test post transient-rms help."""
        result = self.runner.invoke(cli, ["post", "transient-rms", "--help"])
        assert result.exit_code == 0

    def test_post_transient_psd_help(self) -> None:
        """Test post transient-psd help."""
        result = self.runner.invoke(cli, ["post", "transient-psd", "--help"])
        assert result.exit_code == 0
        assert "--probe-location" in result.output


class TestCLIConfigCommands:
    """Test suite for config subcommands."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_config_help(self) -> None:
        """Test config command help."""
        result = self.runner.invoke(cli, ["config", "--help"])
        assert result.exit_code == 0
        assert "init" in result.output
        assert "show" in result.output
        assert "validate" in result.output

    def test_config_init_help(self) -> None:
        """Test config init help."""
        result = self.runner.invoke(cli, ["config", "init", "--help"])
        assert result.exit_code == 0
        assert "--template" in result.output
        assert "steady" in result.output
        assert "transient" in result.output

    def test_config_show_help(self) -> None:
        """Test config show help."""
        result = self.runner.invoke(cli, ["config", "show", "--help"])
        assert result.exit_code == 0

    def test_config_validate_help(self) -> None:
        """Test config validate help."""
        result = self.runner.invoke(cli, ["config", "validate", "--help"])
        assert result.exit_code == 0


class TestCLIUtilsCommands:
    """Test suite for utils subcommands."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_utils_help(self) -> None:
        """Test utils command help."""
        result = self.runner.invoke(cli, ["utils", "--help"])
        assert result.exit_code == 0
        assert "version" in result.output
        assert "doctor" in result.output
        assert "benchmark" in result.output

    def test_utils_version(self) -> None:
        """Test utils version command."""
        result = self.runner.invoke(cli, ["utils", "version"])
        assert result.exit_code == 0
        assert "AutoFlowCFD" in result.output
        assert "0.1.0" in result.output

    def test_utils_version_json(self) -> None:
        """Test utils version with JSON output."""
        result = self.runner.invoke(cli, ["utils", "version", "--json"])
        assert result.exit_code == 0
        import json
        data = json.loads(result.output)
        assert "autoflowcfd" in data

    def test_utils_doctor(self) -> None:
        """Test utils doctor command."""
        result = self.runner.invoke(cli, ["utils", "doctor"])
        assert result.exit_code == 0
        assert "Python" in result.output or "python" in result.output

    def test_utils_doctor_json(self) -> None:
        """Test utils doctor with JSON output."""
        result = self.runner.invoke(cli, ["utils", "doctor", "--json"])
        assert result.exit_code == 0
        import json
        data = json.loads(result.output)
        assert "status" in data
        assert "info" in data

    def test_utils_benchmark_help(self) -> None:
        """Test utils benchmark help."""
        result = self.runner.invoke(cli, ["utils", "benchmark", "--help"])
        assert result.exit_code == 0
        assert "--backend" in result.output
        assert "--iterations" in result.output


class TestCLIGlobalOptions:
    """Test suite for global CLI options."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_main_help(self) -> None:
        """Test main help shows all command groups."""
        result = self.runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "grid" in result.output
        assert "solve" in result.output
        assert "post" in result.output
        assert "config" in result.output
        assert "utils" in result.output

    def test_version_flag(self) -> None:
        """Test --version flag."""
        result = self.runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "AutoFlowCFD" in result.output
        assert "0.1.0" in result.output

    def test_verbose_flag(self) -> None:
        """Test verbose flag."""
        result = self.runner.invoke(cli, ["-v", "--help"])
        assert result.exit_code == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
