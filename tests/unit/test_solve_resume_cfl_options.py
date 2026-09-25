"""
`solve resume` 的 CFL 选项必须真正到达求解器构造点。

## 真实缺口（2026-09-17 发现）

`solve steady` 早在 2026-09-15（提交 `34d9f39`）就把
`--cfl-start/--cfl-max/--cfl-min` 透传到了全部六个求解器构造点，包括
四条分布式路径——那次修复的注释就写在 `solve_steady_command.py` 里：
"真实缺口修复（2026-09-15）：`--cfl-start/--cfl-max/--cfl-min` 此前在
**全部分布式路径**上被静默丢弃"。

但 `solve resume` **整条命令被那次修复漏掉了**：

    单机 resume        有 --cfl-start/--cfl-max，**没有 --cfl-min**
    分布式 resume      三个 CFL 选项**一个都不传**到重建入口

后果是具体且致命的：控制器自身的 `cfl_min` 默认值是 0.05，而本项目在
两张真实 ANSA 网格上实测的稳定 CFL 约 0.03（见项目记忆
`adaptive_cfl_four_defects_and_soft_ceiling` 第 11/12 条：越界一次之后
收缩救不回来）。所以一条**原本固定 CFL 0.03 稳定收敛**的运行，只要
`solve resume` 接着跑，CFL 就会被下限抬到 0.05 并直接发散——而用户在
命令行传了 `--cfl-start 0.03` 也无济于事，因为下限压在上面。

这个缺口是在真实场景下撞到的：三条 160~216 步的运行被会话重启杀掉后，
要从 checkpoint 续算，才发现续算根本无法复现原运行的 CFL 配置。

## 方法论

与 `test_solve_resume_distributed.py` 同一套：mock 掉重建入口
（`rebuild_solver_from_checkpoint` / `rebuild_distributed_solver_from_
checkpoint`，它们各自的底层机制已有独立单元测试），只断言 **CLI 这一层
把用户传的值接力传了下去**——这正是缺口所在的那一层。
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from autoflowcfd.cli.main import cli


def _fake_solver():
    solver = MagicMock()
    solver.solve.return_value = SimpleNamespace(
        converged=False, iterations=10, final_residual=1.0)
    solver.order = 1
    solver.current_order = 1
    return solver


_META = {
    "input_file": "dummy.nas",
    "order": 1,
    "turbulence_model": "sst",
    "backend": "cpu",
    "surface_mesh": None,
}


class TestSingleMachineResume:
    """单机 resume：三个 CFL 选项都必须到达 `rebuild_solver_from_checkpoint`。"""

    def _run(self, tmp_path, extra_args):
        ck = tmp_path / "checkpoint_iter_000150.h5"
        ck.write_bytes(b"")
        # `save_results` 也要 mock：resume 收尾会把最终状态 pickle 到
        # final_state.pkl，而 MagicMock 替身不可 pickle（真实报错
        # `PicklingError: Can't pickle MagicMock`）。这与被测的 CLI 转发
        # 逻辑无关，是替身的固有限制。
        with patch("autoflowcfd.cli.solve.commands.rebuild_solver_from_checkpoint") as reb, \
             patch("autoflowcfd.cli.solve.commands.save_results"), \
             patch("autoflowcfd.cli.solve.commands.write_checkpoint"), \
             patch("autoflowcfd.cli.solve.commands._report_aerodynamic_coefficients"):
            reb.return_value = (_fake_solver(), 150, dict(_META))
            res = CliRunner().invoke(
                cli, ["solve", "resume", str(ck), "-n", "10"] + extra_args)
            return res, reb

    def test_cfl_min_option_exists_and_is_forwarded(self, tmp_path):
        """**核心回归**：`--cfl-min` 必须存在且真正传下去。

        缺口修复前这条会以 "no such option: --cfl-min" 失败。
        """
        res, reb = self._run(tmp_path, ["--cfl-min", "0.03"])
        assert res.exit_code == 0, res.output
        assert reb.call_args.kwargs["cfl_min"] == 0.03

    def test_all_three_cfl_values_forwarded(self, tmp_path):
        res, reb = self._run(
            tmp_path, ["--cfl-start", "0.03", "--cfl-max", "0.03",
                       "--cfl-min", "0.03"])
        assert res.exit_code == 0, res.output
        kw = reb.call_args.kwargs
        assert (kw["cfl_start"], kw["cfl_max"], kw["cfl_min"]) == (0.03, 0.03, 0.03)

    def test_cfl_min_default_matches_solve_steady(self, tmp_path):
        """默认值必须与 `solve steady --cfl-min` 一致。

        两条命令的默认值不一致，会让"同一组参数 resume 一次"悄悄改变
        数值行为——这正是本缺口造成真实发散的机制。
        """
        # 从 click 命令树上取，而不是 import 某个模块里的函数：取到的才是
        # 真正注册过的命令对象（带 .params）。
        steady = cli.commands["solve"].commands["steady"]
        steady_default = None
        for param in steady.params:
            if param.name == "cfl_min":
                steady_default = param.default
        assert steady_default is not None, "solve steady 没有 --cfl-min 了？"

        res, reb = self._run(tmp_path, [])
        assert res.exit_code == 0, res.output
        assert reb.call_args.kwargs["cfl_min"] == steady_default

    def test_a_fixed_cfl_config_is_reproducible_across_resume(self, tmp_path):
        """把真实事故场景钉住：固定 CFL 0.03 的运行必须能原样续算。

        即 cfl_min 必须能跟着一起降到 0.03——否则控制器下限 0.05 会把
        CFL 抬上去，而本项目实测越界一次之后收缩救不回来。
        """
        res, reb = self._run(
            tmp_path, ["--cfl-start", "0.03", "--cfl-max", "0.03",
                       "--cfl-min", "0.03"])
        assert res.exit_code == 0, res.output
        kw = reb.call_args.kwargs
        assert kw["cfl_min"] <= kw["cfl_start"] <= kw["cfl_max"], (
            f"续算的 CFL 三元组不自洽：{kw['cfl_min']} / {kw['cfl_start']} / "
            f"{kw['cfl_max']} —— 下限高于起始值时控制器会把 CFL 抬上去"
        )


class TestDistributedResume:
    """分布式 resume：三个 CFL 选项都必须到达
    `rebuild_distributed_solver_from_checkpoint`。修复前一个都不传。"""

    def _run(self, tmp_path, extra_args):
        ck = tmp_path / "checkpoint_iter_000150.h5"
        ck.write_bytes(b"")
        target = ("autoflowcfd.cli.solve.distributed_checkpoint_io"
                  ".rebuild_distributed_solver_from_checkpoint")
        with patch(target) as reb, \
             patch("autoflowcfd.core.mpi.distributed_checkpoint"
                   ".distributed_save_checkpoint"):
            reb.return_value = (_fake_solver(), 150, dict(_META))
            res = CliRunner().invoke(
                cli, ["solve", "resume", str(ck), "-n", "10",
                      "--n-ranks", "2"] + extra_args)
            return res, reb

    def test_cfl_values_reach_distributed_rebuild(self, tmp_path):
        """只断言 CLI 转发，不断言退出码。

        分布式 resume 的收尾保存会对 `solver.partition` / `state.U` 做真实
        索引，MagicMock 替身给不出自洽的形状（真实报错
        `IndexError: index 1 is out of bounds for axis 0 with size 1`）。
        那发生在**本测试要验证的转发之后**，且分布式保存本身已有
        `test_solve_resume_distributed.py` 专门覆盖，这里不重复。
        关键是重建入口确实被调用过、且拿到了正确的三个 CFL 值。
        """
        res, reb = self._run(
            tmp_path, ["--cfl-start", "0.03", "--cfl-max", "0.03",
                       "--cfl-min", "0.03"])
        assert reb.called, f"重建入口没有被调用：{res.output}"
        kw = reb.call_args.kwargs
        assert (kw["cfl_start"], kw["cfl_max"], kw["cfl_min"]) == (0.03, 0.03, 0.03)
        # 确认失败（若有）不是选项解析失败
        assert "no such option" not in res.output.lower()


class TestRebuildSignaturesAcceptCfl:
    """两个重建入口的签名都必须接收三个 CFL 参数。

    纯签名检查，不需要构造任何求解器：CLI 转发得再对，重建入口不接收
    也是白传（那正是修复前分布式那条的状态——CLI 解析了但没人接）。
    """

    def test_single_machine_rebuild(self):
        import inspect

        from autoflowcfd.cli.solve.checkpoint_io import (
            rebuild_solver_from_checkpoint,
        )

        params = inspect.signature(rebuild_solver_from_checkpoint).parameters
        for name in ("cfl_start", "cfl_max", "cfl_min"):
            assert name in params, f"rebuild_solver_from_checkpoint 缺 {name}"

    def test_distributed_rebuild(self):
        import inspect

        from autoflowcfd.cli.solve.distributed_checkpoint_io import (
            rebuild_distributed_solver_from_checkpoint,
        )

        params = inspect.signature(
            rebuild_distributed_solver_from_checkpoint).parameters
        for name in ("cfl_start", "cfl_max", "cfl_min"):
            assert name in params, (
                f"rebuild_distributed_solver_from_checkpoint 缺 {name}"
            )
