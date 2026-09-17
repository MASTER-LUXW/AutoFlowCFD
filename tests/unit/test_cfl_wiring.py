"""CFL 参数接线回归测试（2026-09-07）：`AdaptiveCFLController` 的
`cfl_start`/`cfl_max` 此前恒用硬编码默认值（0.1/0.3），`SteadyConfig.
cfl_init`/`cfl_max` 这两个 config 字段从未真正接到控制器上，CLI 也没有
对应选项——稳态钝体绕流残差收敛慢时用户完全没有调 CFL 的口子。

现在：CLI `--cfl-start`/`--cfl-max`（solve steady / solve resume）
-> FRSolver 构造参数 -> AdaptiveCFLController；cfl_max 默认值从 0.3
上调到 0.5。
"""

from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from click.testing import CliRunner

from autoflowcfd.core.time_integration.adaptive_cfl import AdaptiveCFLController
from autoflowcfd.cli.main import cli


class TestAdaptiveCFLControllerParams:
    def test_controller_defaults_match_the_cli(self):
        """控制器构造默认值必须与 CLI 默认值是同一个数。

        2026-09-17 从 (0.1, 0.5) 重定为 (0.03, 0.06)，依据是三类实测（见
        `cli/solve_steady_command.py` 的 --cfl-max 帮助与
        `test_adaptive_cfl_bounds_consistency.py::
        TestConfigLayerCflDefaultsAreConsistent`）。这里不再硬编码那两个
        数字，而是从 click 元数据读——配置层与 CLI 默认值"相差 20 倍"
        那次事故（2026-09-15）的根源正是两处各自硬编码。
        """
        from autoflowcfd.cli.solve_steady_command import solve_steady

        want = {}
        for prm in solve_steady.params:
            for name, key in (("--cfl-start", "cfl_start"),
                              ("--cfl-max", "cfl_max"),
                              ("--cfl-min", "cfl_min")):
                if name in getattr(prm, "opts", []):
                    want[key] = float(prm.default)
        assert set(want) == {"cfl_start", "cfl_max", "cfl_min"}

        c = AdaptiveCFLController()
        assert c.cfl_start == want["cfl_start"]
        assert c.cfl_max == want["cfl_max"]
        assert c.cfl_min == want["cfl_min"]
        assert c.cfl_number == want["cfl_start"]   # 初始值 = cfl_start
        assert c.cfl_min < c.cfl_start < c.cfl_max, (
            "三者必须严格递增，否则控制器一步也动不了（cfl_start==cfl_max "
            "会被判定为固定 CFL，见 adaptive_cfl.py 构造函数）")

    def test_explicit_params_honored(self):
        c = AdaptiveCFLController(cfl_start=0.2, cfl_max=0.9)
        assert c.cfl_start == 0.2
        assert c.cfl_max == 0.9
        assert c.cfl_number == 0.2

    def test_grow_is_capped_at_cfl_max(self):
        """自适应放大不能越过 cfl_max（改上限后这个约束仍然成立）。"""
        c = AdaptiveCFLController(cfl_start=0.1, cfl_max=0.5,
                                  ramp_steps=0, cooldown_steps=0,
                                  growth_confirm_steps=1)
        # 残差持续快速下降 -> 触发 grow
        r = 1.0
        for _ in range(50):
            r *= 0.5
            c.update(r)
        assert c.cfl_number <= 0.5 + 1e-12


class TestResumeCflOptionForwarded:
    def _fake_solver(self):
        solver = MagicMock()
        solver.current_order = 0
        solver.order = 0
        solver._reference_area = None
        solver.solve.side_effect = lambda *a, **k: SimpleNamespace(
            iterations=5, final_residual=1e-3
        )
        return solver

    def test_cli_cfl_options_reach_rebuild_solver_from_checkpoint(self, tmp_path):
        ckpt = tmp_path / "checkpoint_iter_002000.h5"
        ckpt.write_bytes(b"")
        fake_solver = self._fake_solver()
        fake_meta = {
            "input_file": "volume.nas", "order": 0, "turbulence_model": "sst",
            "backend": "cpu", "surface_mesh": "surface.nas",
        }
        with patch(
            "autoflowcfd.cli.solve_commands.rebuild_solver_from_checkpoint",
            return_value=(fake_solver, 2000, fake_meta),
        ) as mock_rebuild, patch(
            "autoflowcfd.cli.solve_commands.save_results"
        ), patch(
            "autoflowcfd.cli.solve_commands.write_checkpoint"
        ):
            result = CliRunner().invoke(
                cli,
                ["solve", "resume", str(ckpt), "--max-iter", "5",
                 "--cfl-start", "0.2", "--cfl-max", "0.9"],
            )
        assert result.exit_code == 0, result.output
        assert mock_rebuild.call_args.kwargs["cfl_start"] == 0.2
        assert mock_rebuild.call_args.kwargs["cfl_max"] == 0.9

    def test_cli_cfl_options_default_when_not_passed(self, tmp_path):
        ckpt = tmp_path / "checkpoint_iter_002000.h5"
        ckpt.write_bytes(b"")
        fake_solver = self._fake_solver()
        fake_meta = {
            "input_file": "volume.nas", "order": 0, "turbulence_model": "sst",
            "backend": "cpu", "surface_mesh": "surface.nas",
        }
        with patch(
            "autoflowcfd.cli.solve_commands.rebuild_solver_from_checkpoint",
            return_value=(fake_solver, 2000, fake_meta),
        ) as mock_rebuild, patch(
            "autoflowcfd.cli.solve_commands.save_results"
        ), patch(
            "autoflowcfd.cli.solve_commands.write_checkpoint"
        ):
            result = CliRunner().invoke(
                cli, ["solve", "resume", str(ckpt), "--max-iter", "5"],
            )
        assert result.exit_code == 0, result.output
        # 不硬编码默认值：从 click 元数据读，与 CLI 保持单一事实来源
        from autoflowcfd.cli.solve_commands import resume as _resume_cmd

        _want = {}
        for prm in _resume_cmd.params:
            for name, key in (("--cfl-start", "cfl_start"),
                              ("--cfl-max", "cfl_max"),
                              ("--cfl-min", "cfl_min")):
                if name in getattr(prm, "opts", []):
                    _want[key] = float(prm.default)
        assert mock_rebuild.call_args.kwargs["cfl_start"] == _want["cfl_start"]
        assert mock_rebuild.call_args.kwargs["cfl_max"] == _want["cfl_max"]
        # rebuild_solver_from_checkpoint 内部把收到的 cfl_start/cfl_max
        # 原样传进 FRSolver(...) —— 这一步是源码里直接可见的
        # `cfl_start=cfl_start, cfl_max=cfl_max`（solve_checkpoint_io.py），
        # 不再单独 mock 验证（会因 FRSolver 的函数内延迟导入而变得脆弱）。
