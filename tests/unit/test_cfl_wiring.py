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
    def test_cli_leaves_cfl_defaults_to_the_control_law(self):
        """CLI 的三个 CFL 选项默认 `None`，由 `build_cfl_policy` 按时间格式取
        对应 CFL 律的签名默认值（2026-09-25 起）。

        历史：2026-09-17 以前 CLI 与控制器各自硬编码一份，2026-09-15 出过
        "配置层与 CLI 默认值相差 20 倍"；之后改成"CLI 默认值 == 控制器默认值"
        的一致性测试。引入隐式格式后同一组选项要服务两个默认值差两个数量级
        的 CFL 律（显式 0.03 / SER 5），CLI 再写死任何一个都会让另一个格式
        拿到错值，所以 CLI 不再持有默认值。
        """
        from autoflowcfd.cli.solve.commands import resume
        from autoflowcfd.cli.solve.steady import solve_steady
        from autoflowcfd.cli.solve.transient import transient

        for cmd in (solve_steady, resume, transient):
            seen = set()
            for prm in cmd.params:
                for name in ("--cfl-start", "--cfl-max", "--cfl-min"):
                    if name in getattr(prm, "opts", []):
                        assert prm.default is None, (cmd.name, name, prm.default)
                        seen.add(name)
            assert seen == {"--cfl-start", "--cfl-max", "--cfl-min"}, cmd.name

        c = AdaptiveCFLController()
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
            "autoflowcfd.cli.solve.commands.rebuild_solver_from_checkpoint",
            return_value=(fake_solver, 2000, fake_meta),
        ) as mock_rebuild, patch(
            "autoflowcfd.cli.solve.commands.save_results"
        ), patch(
            "autoflowcfd.cli.solve.commands.write_checkpoint"
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
            "autoflowcfd.cli.solve.commands.rebuild_solver_from_checkpoint",
            return_value=(fake_solver, 2000, fake_meta),
        ) as mock_rebuild, patch(
            "autoflowcfd.cli.solve.commands.save_results"
        ), patch(
            "autoflowcfd.cli.solve.commands.write_checkpoint"
        ):
            result = CliRunner().invoke(
                cli, ["solve", "resume", str(ckpt), "--max-iter", "5"],
            )
        assert result.exit_code == 0, result.output
        # 未传时原样传 None，由 build_cfl_policy 按续算所用时间格式取默认值
        for key in ("cfl_start", "cfl_max", "cfl_min"):
            assert mock_rebuild.call_args.kwargs[key] is None
        assert mock_rebuild.call_args.kwargs["time_scheme"] is None
        # rebuild_solver_from_checkpoint 内部把收到的 cfl_start/cfl_max
        # 原样传进 FRSolver(...) —— 这一步是源码里直接可见的
        # `cfl_start=cfl_start, cfl_max=cfl_max`（solve_checkpoint_io.py），
        # 不再单独 mock 验证（会因 FRSolver 的函数内延迟导入而变得脆弱）。
