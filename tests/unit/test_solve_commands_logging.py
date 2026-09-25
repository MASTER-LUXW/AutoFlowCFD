"""Regression tests for the CLI/core logging bugs fixed 2026-08-21:

1. `cli/solve/commands.py` and 6 other files used standard-library
   `logging.getLogger(__name__)` instead of the project-wide `loguru`
   logger configured in `cli/main.py`. The stdlib root logger is never
   configured (no `basicConfig`/handler) anywhere in this project, so
   every `logger.info/warning/error(...)` call through it was silently
   swallowed - `solve status` printed nothing at all, and
   `core/fr_solver/step.py`'s `logger.error("Step failed with error: ...")`
   never actually appeared (only `traceback.print_exc()` right after it
   did, easy to mistake for the same message).

2. `core/fr_solver/step.py::mean_flow_residual` used to import `logger`
   lazily from `core.fr_solver.solver`; since 2026-09-25 (solver.py split
   into a package) it imports loguru's `logger` directly, so that
   cross-module coupling no longer exists.
"""

from click.testing import CliRunner

from autoflowcfd.cli.main import cli


def test_solve_status_prints_content_on_stderr():
    """`solve status` (no flags) must actually print its status lines
    somewhere, not silently do nothing (the pre-fix behavior: exit 0,
    zero bytes on either stream)."""
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
    """Guards against the same mistake creeping back in: every file in
    this project must use loguru, not `logging.getLogger`, for its
    module-level logger (the project never configures the stdlib root
    logger, so a stdlib logger there is always silently inert)."""
    import autoflowcfd.cli.solve.commands as solve_commands
    import autoflowcfd.cli.solve.aero_coefficients as solve_aero_coefficients
    import autoflowcfd.core.fr_solver.solver.threads as solver_module

    for module in (solve_commands, solve_aero_coefficients, solver_module):
        assert hasattr(module, "logger")
        # loguru's Logger singleton exposes `.opt`/`.bind`; the stdlib
        # logging.Logger does not - a cheap, reliable discriminator.
        assert hasattr(module.logger, "opt"), (
            f"{module.__name__}.logger looks like stdlib logging, not loguru"
        )
