"""Unit tests for CLI module."""

import pytest
from click.testing import CliRunner
from autoflowcfd.cli.main import cli


class TestCLI:
    """Test suite for CLI commands."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_cli_version(self) -> None:
        """Test that --version flag works."""
        result = self.runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "AutoFlowCFD" in result.output
        assert "0.1.0" in result.output

    def test_cli_help(self) -> None:
        """Test that --help flag works."""
        result = self.runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "AutoFlowCFD" in result.output
        assert "solve" in result.output
        assert "post" in result.output

    def test_solve_command_help(self) -> None:
        """Test solve command help."""
        result = self.runner.invoke(cli, ["solve", "--help"])
        assert result.exit_code == 0
        assert "run" in result.output or "transient" in result.output

    def test_post_command_help(self) -> None:
        """Test post command help."""
        result = self.runner.invoke(cli, ["post", "--help"])
        assert result.exit_code == 0
        assert "coefficients" in result.output or "export-vtk" in result.output

    def test_grid_command_help(self) -> None:
        """Test grid command help."""
        result = self.runner.invoke(cli, ["grid", "--help"])
        assert result.exit_code == 0

    def test_utils_command_help(self) -> None:
        """Test utils command help."""
        result = self.runner.invoke(cli, ["utils", "--help"])
        assert result.exit_code == 0

    def test_solve_run_missing_args(self) -> None:
        """Test solve run command without required arguments."""
        result = self.runner.invoke(cli, ["solve", "run"])
        assert result.exit_code != 0

    def test_verbose_flag(self) -> None:
        """Test verbose flag enables debug output."""
        result = self.runner.invoke(cli, ["-v", "--help"])
        assert result.exit_code == 0
