"""2026-08-21 修复的 CLI/core 日志缺陷的回归测试：

1. `cli/solve/commands.py` 与另外 6 个文件用了标准库的
   `logging.getLogger(__name__)`，而不是 `cli/main.py` 里配置的全项目
   `loguru` 日志器。本项目任何地方都没有配置过标准库的根日志器（没有
   `basicConfig`/handler），于是经它发出的每个
   `logger.info/warning/error(...)` 都被静默吞掉——`solve status` 什么都
   不打印，`core/fr_solver/step.py` 的
   `logger.error("Step failed with error: ...")` 从未真正出现过（只有紧随
   其后的 `traceback.print_exc()` 出现，容易被误认成同一条消息）。

2. `core/fr_solver/step.py::mean_flow_residual` 曾经从
   `core.fr_solver.solver` 惰性导入 `logger`；2026-09-25（solver.py 拆成包）
   起它直接导入 loguru 的 `logger`，这种跨模块耦合不再存在。
"""

from click.testing import CliRunner

from autoflowcfd.cli.main import cli


def test_solve_status_prints_content_on_stderr():
    """`solve status`（不带选项）必须真的把状态行打印到某处，而不是静默地
    什么都不做（修复之前的行为：退出码 0，两个流都是零字节）。
    """
    runner = CliRunner()
    result = runner.invoke(cli, ["solve", "status"])
    assert result.exit_code == 0
    assert "Ready" in result.output
    assert "Orders" in result.output


def test_solve_status_backend_prints_content():
    runner = CliRunner()
    result = runner.invoke(cli, ["solve", "status", "--backend"])
    assert result.exit_code == 0
    assert "cpu" in result.output.lower()


def test_no_stdlib_logging_getlogger_left_in_solve_or_solver_modules():
    """防止同样的错误再溜回来：本项目的每个文件的模块级日志器都必须用
    loguru，不能用 `logging.getLogger`（本项目从不配置标准库的根日志器，
    那里的标准库日志器永远静默无效）。
    """
    import autoflowcfd.cli.solve.commands as solve_commands
    import autoflowcfd.cli.solve.aero_coefficients as solve_aero_coefficients
    import autoflowcfd.core.fr_solver.solver.threads as solver_module

    for module in (solve_commands, solve_aero_coefficients, solver_module):
        assert hasattr(module, "logger")
        # loguru 的 Logger 单例有 `.opt`/`.bind`；标准库的 logging.Logger 没有
        # ——一个便宜、可靠的区分办法。
        assert hasattr(module.logger, "opt"), (
            f"{module.__name__}.logger looks like stdlib logging, not loguru"
        )
