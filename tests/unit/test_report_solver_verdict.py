"""后处理报告取求解器自己的收敛结论、CSV 列不随阶段错位、未算的系数不写 0（I-159）。

此前报告的 converged 用写死的"全部残差绝对值 < 1e-4"重新判定，与求解器的判据（相对下降、阶段门控）无关；CSV 表头
取第一行的键、每行按自己的键写，湍流方程随阶段出现时列错位，首行无系数而后续有系数时系数列无表头；重放只恢复
Cd/Cl（实为压力部分 Cd_p/Cl_p），Cm/Cs/Cy/Cr 以 0.0 写进报告。
"""

import csv
import json

import numpy as np

from autoflowcfd.cli.post.helpers import _replay_history
from autoflowcfd.cli.solve.checkpoint_io.write import write_checkpoint
from autoflowcfd.core.fr_solver.state import solve_status
from autoflowcfd.core.time_integration import TimeIntegrationScheme
from autoflowcfd.core.utils.checkpoint import CheckpointManager
from autoflowcfd.postprocess import ConvergenceAnalyzer, SimulationReport
from tests.unit.test_convergence_history import _solver


def test_solve_status_values():
    assert solve_status(True, 1e-6) == "converged"
    assert solve_status(False, 1e-6) == "not_converged"
    assert solve_status(False, 0.0) == "finished"


def test_solver_records_its_verdict_in_the_checkpoint(tmp_path):
    s = _solver(TimeIntegrationScheme.SSP_RK3)
    s.solve(max_iter=2, tol=1e-30)                     # 两步达不到判据：未收敛
    assert s.solve_status == "not_converged"
    path = write_checkpoint(s, str(tmp_path), 2, "mesh.nas", order=1, turbulence_model="none", backend="cpu",
                            quiet=True)
    assert CheckpointManager().load(path)[3]["solve_status"] == "not_converged"
    s.solve(max_iter=2, tol=0.0)
    assert s.solve_status == "finished"


def test_report_converged_comes_from_the_solver_not_a_residual_threshold(tmp_path):
    analyzer = ConvergenceAnalyzer()
    analyzer.add_iteration(iteration=1, residuals={"rho": 1e-9})          # 绝对值很小，旧判据会说"已收敛"
    path = SimulationReport({}, analyzer, solve_status="not_converged").generate(str(tmp_path / "r.json"))
    summary = json.loads(path.read_text())["summary"]
    assert summary["converged"] is False and summary["solve_status"] == "not_converged"
    assert "convergence_criteria" not in summary
    assert analyzer.get_summary().converged is None                        # 不知道就是不知道


def test_csv_columns_are_the_union_over_the_whole_history(tmp_path):
    analyzer = ConvergenceAnalyzer()
    analyzer.add_iteration(iteration=1, residuals={"rho": 1.0, "rhoE": 2.0}, cfl=1.0)
    analyzer.add_iteration(iteration=2, residuals={"rho": 0.5, "rhoE": 1.0, "k": 3.0}, cfl=2.0,
                           coefficients={"Cd_p": 0.3})
    rows = list(csv.reader(open(analyzer.export_csv(str(tmp_path / "c.csv")), newline="")))
    assert rows[0] == ["iteration", "cfl", "k", "rho", "rhoE", "Cd_p"]
    assert rows[1] == ["1", "1.0", "", "1.0", "2.0", ""]
    assert rows[2] == ["2", "2.0", "3.0", "0.5", "1.0", "0.3"]


def test_replay_keeps_the_recorded_coefficient_names():
    history = {"iterations": [1, 2], "residuals": {"rho": [1.0, 0.5]}, "cfl_history": [1.0, 1.0],
               "coefficients": {"Cd_p": [np.nan, 0.31], "Cl_p": [np.nan, 0.02], "Cs_p": [np.nan, -0.01]}}
    analyzer = _replay_history(history)
    assert analyzer.history[0].coefficients == {}
    assert analyzer.history[1].coefficients == {"Cd_p": 0.31, "Cl_p": 0.02, "Cs_p": -0.01}
