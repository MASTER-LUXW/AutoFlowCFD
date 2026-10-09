"""CLI 命令的单元测试。"""

import pytest
from click.testing import CliRunner
from autoflowcfd.cli.main import cli


class TestCLIGridCommands:
    """grid 子命令的测试。"""

    def setup_method(self) -> None:
        """准备测试夹具。"""
        self.runner = CliRunner()

    def test_grid_help(self) -> None:
        """grid 命令的帮助。"""
        result = self.runner.invoke(cli, ["grid", "--help"])
        assert result.exit_code == 0
        assert "parse" in result.output
        assert "validate" in result.output
        assert "info" in result.output

    def test_grid_parse_help(self) -> None:
        """grid parse 的帮助。"""
        result = self.runner.invoke(cli, ["grid", "parse", "--help"])
        assert result.exit_code == 0
        assert "--output" in result.output
        assert "--streaming" in result.output

    def test_grid_validate_help(self) -> None:
        """grid validate 的帮助。"""
        result = self.runner.invoke(cli, ["grid", "validate", "--help"])
        assert result.exit_code == 0
        assert "--report" in result.output
        assert "--threshold-aspect-ratio" in result.output

    def test_grid_info_help(self) -> None:
        """grid info 的帮助。"""
        result = self.runner.invoke(cli, ["grid", "info", "--help"])
        assert result.exit_code == 0
        assert "--json" in result.output


class TestCLISolveCommands:
    """solve 子命令的测试。"""

    def setup_method(self) -> None:
        """准备测试夹具。"""
        self.runner = CliRunner()

    def test_solve_help(self) -> None:
        """solve 命令的帮助。"""
        result = self.runner.invoke(cli, ["solve", "--help"])
        assert result.exit_code == 0
        assert "steady" in result.output
        assert "transient" in result.output
        assert "resume" in result.output

    def test_solve_steady_help(self) -> None:
        """solve steady 的帮助。"""
        result = self.runner.invoke(cli, ["solve", "steady", "--help"])
        assert result.exit_code == 0
        assert "--backend" in result.output
        assert "--order" in result.output
        assert "--turbulence" in result.output
        assert "--max-iter" in result.output
        assert "--skip-quality-check" in result.output
        assert "--surface-mesh" in result.output

    def test_solve_transient_help(self) -> None:
        """solve transient 的帮助。"""
        result = self.runner.invoke(cli, ["solve", "transient", "--help"])
        assert result.exit_code == 0
        assert "--physical-time" in result.output
        assert "--dt" in result.output
        assert "--time-method" in result.output
        assert "--skip-quality-check" in result.output
        assert "--surface-mesh" in result.output

    def test_solve_steady_rejects_unsupported_extension(self, tmp_path) -> None:
        """既不是 .pkl 也不是 .nas：solve steady 必须拒绝，并明确指向
        grid generate-volume/import-volume。
        """
        bogus_file = tmp_path / "mesh.su2"
        bogus_file.write_text("dummy\n")
        result = self.runner.invoke(cli, ["solve", "steady", str(bogus_file)])
        assert result.exit_code != 0
        assert "generate-volume" in result.output or "import-volume" in result.output

    def test_solve_transient_rejects_unsupported_extension(self, tmp_path) -> None:
        """与 steady 的同一项检查，针对 transient。"""
        bogus_file = tmp_path / "mesh.su2"
        bogus_file.write_text("dummy\n")
        result = self.runner.invoke(cli, ["solve", "transient", str(bogus_file)])
        assert result.exit_code != 0
        assert "generate-volume" in result.output or "import-volume" in result.output

    def test_solve_steady_nas_without_surface_mesh_is_rejected(self, tmp_path) -> None:
        """solve steady 接受 .nas 体网格，但必须同时给 --surface-mesh（用来划分
        WALL/INLET/OUTLET 边界组）——单独传入必须被拒绝，而不是在没有任何边界
        条件的情况下静默求解。
        """
        volume_nas = tmp_path / "volume.nas"
        volume_nas.write_text("$ dummy nas file\n")
        result = self.runner.invoke(cli, ["solve", "steady", str(volume_nas)])
        assert result.exit_code != 0
        assert "--surface-mesh" in result.output

    def test_solve_transient_nas_without_surface_mesh_is_rejected(self, tmp_path) -> None:
        """与 steady 的同一项检查，针对 transient。"""
        volume_nas = tmp_path / "volume.nas"
        volume_nas.write_text("$ dummy nas file\n")
        result = self.runner.invoke(cli, ["solve", "transient", str(volume_nas)])
        assert result.exit_code != 0
        assert "--surface-mesh" in result.output

    def test_solve_resume_help(self) -> None:
        """solve resume 的帮助。"""
        result = self.runner.invoke(cli, ["solve", "resume", "--help"])
        assert result.exit_code == 0
        assert "checkpoint" in result.output.lower()

    def test_solve_status_help(self) -> None:
        """solve status 的帮助。"""
        result = self.runner.invoke(cli, ["solve", "status", "--help"])
        assert result.exit_code == 0


class TestCLIPostCommands:
    """post 子命令的测试。"""

    def setup_method(self) -> None:
        """准备测试夹具。"""
        self.runner = CliRunner()

    def test_post_help(self) -> None:
        """post 命令的帮助。"""
        result = self.runner.invoke(cli, ["post", "--help"])
        assert result.exit_code == 0
        assert "coefficients" in result.output
        assert "export-vtk" in result.output
        assert "convergence" in result.output

    def test_post_coefficients_help(self) -> None:
        """post coefficients 的帮助。"""
        result = self.runner.invoke(cli, ["post", "coefficients", "--help"])
        assert result.exit_code == 0
        assert "--case" in result.output
        assert "--reference-area" in result.output

    def test_post_export_vtk_help(self) -> None:
        """post export-vtk 的帮助。"""
        result = self.runner.invoke(cli, ["post", "export-vtk", "--help"])
        assert result.exit_code == 0
        assert "--output" in result.output

    def test_post_report_help(self) -> None:
        """Test post report help.

        `--format` 已于 2026-09-02 移除（此前的 markdown/html/pdf 选项
        从未真正实现，只会退回 JSON——用户确认没有这几种格式的需求后
        直接删除这个假选项，不再保留一个只有 json 一个真实取值的
        `--format`，见 post_commands.py::report 文档）。
        """
        result = self.runner.invoke(cli, ["post", "report", "--help"])
        assert result.exit_code == 0
        assert "--output" in result.output

    def test_post_convergence_help(self) -> None:
        """post convergence 的帮助。"""
        result = self.runner.invoke(cli, ["post", "convergence", "--help"])
        assert result.exit_code == 0

    def test_post_transient_mean_help(self) -> None:
        """post transient-mean 的帮助。"""
        result = self.runner.invoke(cli, ["post", "transient-mean", "--help"])
        assert result.exit_code == 0

    def test_post_transient_rms_help(self) -> None:
        """post transient-rms 的帮助。"""
        result = self.runner.invoke(cli, ["post", "transient-rms", "--help"])
        assert result.exit_code == 0

    def test_post_transient_psd_help(self) -> None:
        """post transient-psd 的帮助。"""
        result = self.runner.invoke(cli, ["post", "transient-psd", "--help"])
        assert result.exit_code == 0
        assert "--probe-location" in result.output


class TestCLIConfigCommands:
    """config 子命令的测试。"""

    def setup_method(self) -> None:
        """准备测试夹具。"""
        self.runner = CliRunner()

    def test_config_help(self) -> None:
        """config 命令的帮助。"""
        result = self.runner.invoke(cli, ["config", "--help"])
        assert result.exit_code == 0
        assert "init" in result.output
        assert "show" in result.output
        assert "validate" in result.output

    def test_config_init_help(self) -> None:
        """config init 的帮助。"""
        result = self.runner.invoke(cli, ["config", "init", "--help"])
        assert result.exit_code == 0
        assert "--template" in result.output
        assert "steady" in result.output
        assert "transient" in result.output

    def test_config_show_help(self) -> None:
        """config show 的帮助。"""
        result = self.runner.invoke(cli, ["config", "show", "--help"])
        assert result.exit_code == 0

    def test_config_validate_help(self) -> None:
        """config validate 的帮助。"""
        result = self.runner.invoke(cli, ["config", "validate", "--help"])
        assert result.exit_code == 0

    @pytest.mark.parametrize("template", ["steady", "transient"])
    def test_config_init_then_validate_round_trip_reports_correct_mode(
        self, tmp_path, template
    ) -> None:
        """`config init --template X` 生成的文件再过 `config validate` 必须干净地
        往返：退出码 0，并报告与请求相同的 mode。

        本测试抓到的真实缺陷（2026-08-21 修复）：`validate` 用
        `config_obj.mode if hasattr(config_obj, 'mode') else "unknown"` 取 `mode`，
        但 `mode` 是 YAML 顶层的路由键，在配置 dataclass 构造之前就被
        `ConfigLoader.load()` 消费掉了——它从来不是 SteadyConfig/TransientConfig
        的属性，于是 `hasattr` 恒为 False，任何配置文件不论实际 mode 是什么都
        报告 "unknown"。
        """
        out_file = tmp_path / f"{template}.yaml"
        init_result = self.runner.invoke(
            cli, ["config", "init", "--template", template, "-o", str(out_file)]
        )
        assert init_result.exit_code == 0, init_result.output

        validate_result = self.runner.invoke(
            cli, ["config", "validate", str(out_file), "--json"]
        )
        assert validate_result.exit_code == 0, validate_result.output

        # `--json` 的输出必须只靠 stdout 就能管道传递/解析（实际用法：
        # `autoflowcfd ... --json > out.json` 只重定向 stdout）——INFO 级日志属于
        # stderr，不能漏进 stdout 破坏 JSON 内容（见 cli/main.py 的
        # `logger.add(..., err=True)`）。
        import json
        payload = json.loads(validate_result.stdout)
        assert payload["status"] == "valid"
        assert payload["mode"] == template


class TestCLIUtilsCommands:
    """utils 子命令的测试。"""

    def setup_method(self) -> None:
        """准备测试夹具。"""
        self.runner = CliRunner()

    def test_utils_help(self) -> None:
        """utils 命令的帮助。"""
        result = self.runner.invoke(cli, ["utils", "--help"])
        assert result.exit_code == 0
        assert "version" in result.output
        assert "doctor" in result.output
        assert "benchmark" in result.output

    def test_utils_version(self) -> None:
        """utils version 命令。"""
        result = self.runner.invoke(cli, ["utils", "version"])
        assert result.exit_code == 0
        assert "AutoFlowCFD" in result.output
        assert "0.1.0" in result.output

    def test_utils_version_json(self) -> None:
        """utils version 的 JSON 输出。"""
        result = self.runner.invoke(cli, ["utils", "version", "--json"])
        assert result.exit_code == 0
        import json
        data = json.loads(result.output)
        assert "autoflowcfd" in data

    def test_utils_doctor(self) -> None:
        """utils doctor 命令。"""
        result = self.runner.invoke(cli, ["utils", "doctor"])
        assert result.exit_code == 0
        assert "Python" in result.output or "python" in result.output

    def test_utils_doctor_json(self) -> None:
        """utils doctor 的 JSON 输出。"""
        result = self.runner.invoke(cli, ["utils", "doctor", "--json"])
        assert result.exit_code == 0
        import json
        data = json.loads(result.output)
        assert "status" in data
        assert "info" in data

    def test_utils_benchmark_help(self) -> None:
        """utils benchmark 的帮助。"""
        result = self.runner.invoke(cli, ["utils", "benchmark", "--help"])
        assert result.exit_code == 0
        assert "--backend" in result.output
        assert "--iterations" in result.output


class TestCLIGlobalOptions:
    """CLI 全局选项的测试。"""

    def setup_method(self) -> None:
        """准备测试夹具。"""
        self.runner = CliRunner()

    def test_main_help(self) -> None:
        """主帮助列出全部命令组。"""
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


def test_cli_output_is_line_buffered_when_redirected(tmp_path):
    """输出被重定向到文件时 Python 默认整块缓冲：长程计算的日志会滞后几十分钟，进程异常退出时最后一段丢失
    （2026-10-08 cube_demo 稳态：日志停在第 53 步时 checkpoint 已写到第 150 步）。CLI 入口改为行缓冲。"""
    import subprocess
    import sys

    out = tmp_path / "log.txt"
    with open(out, "w") as f:
        subprocess.run([sys.executable, "-c",
                        "import sys, autoflowcfd.cli.main; print(sys.stdout.line_buffering, sys.stderr.line_buffering)"],
                       stdout=f, stderr=subprocess.DEVNULL, check=True)
    assert out.read_text().split() == ["True", "True"]
