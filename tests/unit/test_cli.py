"""CLI 模块的单元测试。"""

import pytest
from click.testing import CliRunner
from autoflowcfd.cli.main import cli


class TestCLI:
    """CLI 命令的测试。"""

    def setup_method(self) -> None:
        """准备测试夹具。"""
        self.runner = CliRunner()

    def test_cli_version(self) -> None:
        """--version 选项可用。"""
        result = self.runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "AutoFlowCFD" in result.output
        assert "0.1.0" in result.output

    def test_cli_help(self) -> None:
        """--help 选项可用。"""
        result = self.runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "AutoFlowCFD" in result.output
        assert "solve" in result.output
        assert "post" in result.output

    def test_solve_command_help(self) -> None:
        """solve 命令的帮助。"""
        result = self.runner.invoke(cli, ["solve", "--help"])
        assert result.exit_code == 0
        assert "run" in result.output or "transient" in result.output

    def test_post_command_help(self) -> None:
        """post 命令的帮助。"""
        result = self.runner.invoke(cli, ["post", "--help"])
        assert result.exit_code == 0
        assert "coefficients" in result.output or "export-vtk" in result.output

    def test_grid_command_help(self) -> None:
        """grid 命令的帮助。"""
        result = self.runner.invoke(cli, ["grid", "--help"])
        assert result.exit_code == 0

    def test_utils_command_help(self) -> None:
        """utils 命令的帮助。"""
        result = self.runner.invoke(cli, ["utils", "--help"])
        assert result.exit_code == 0

    def test_solve_run_missing_args(self) -> None:
        """solve run 命令缺少必需参数。"""
        result = self.runner.invoke(cli, ["solve", "run"])
        assert result.exit_code != 0

    def test_verbose_flag(self) -> None:
        """verbose 选项打开调试输出。"""
        result = self.runner.invoke(cli, ["-v", "--help"])
        assert result.exit_code == 0
