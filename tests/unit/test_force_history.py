"""瞬态逐物理步的气动力历史 `forces.csv`（`core/probes/forces.py`，I-238）：每步一行总系数（压力 + 粘性）与压力
部分、与求解结束时的积分同一个函数、续算截断后接着写、分布式经整场汇总、参考面积缺失报错、CLI 开关进入运行参数。
"""

import csv

import numpy as np
import pytest

from autoflowcfd.core.probes import FORCE_FILE, FORCE_HISTORY_KEY, attach_force_history, attach_force_history_on_resume
from autoflowcfd.core.probes.forces import COLUMNS, HEADER
from autoflowcfd.postprocess.fr_coefficients import total_and_pressure_force_coefficients
from tests.unit.test_solver_probes import DT, _channel_solver

AREA = 0.4 * 0.08          # 通道下壁面积（量级合适即可）


def _rows(path):
    with open(path / FORCE_FILE, newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == list(HEADER)
    return np.array([[float(v) for v in r] for r in rows[1:]])


@pytest.mark.parametrize("backend", ["cpu", "distributed"])
def test_one_row_per_physical_step_with_viscous_and_pressure_parts(tmp_path, backend):
    solver = _channel_solver(backend)
    if backend == "cpu":
        solver._reference_area = AREA                # 每步日志的压力系数也算（与 forces.csv 的压力部分比对）
    attach_force_history(solver, AREA, tmp_path)
    solver.solve(max_iter=3, dt=DT, tol=0.0)
    rows = _rows(tmp_path)
    np.testing.assert_allclose(rows[:, 0], [DT, 2 * DT, 3 * DT], rtol=1e-12)
    np.testing.assert_array_equal(rows[:, 1], [1, 2, 3])
    if backend == "cpu":
        ref = total_and_pressure_force_coefficients(solver, AREA)
    else:
        from autoflowcfd.core.mpi.global_view import gather_global_view

        ref = total_and_pressure_force_coefficients(gather_global_view(solver), AREA)
    np.testing.assert_allclose(rows[-1, 2:], [ref[k] for k in COLUMNS], rtol=1e-12)
    # 通道壁面摩擦：粘性部分不为零（总阻力 != 压力阻力）
    assert abs(rows[-1, 2] - rows[-1, 5]) > 1e-3 * abs(rows[-1, 2])
    if backend == "cpu":
        # 压力部分与每步日志 / 收敛历史的同一份
        last = solver.convergence_history.last_coefficients
        np.testing.assert_allclose(rows[-1, 5:], [last["Cd_p"], last["Cl_p"], last["Cs_p"]], rtol=1e-12)


def test_resume_truncates_and_continues(tmp_path):
    from autoflowcfd.cli.solve.checkpoint_io.write import write_checkpoint
    from autoflowcfd.core.utils.checkpoint import CheckpointManager

    solver = _channel_solver()
    attach_force_history(solver, AREA, tmp_path)
    solver.solve(max_iter=3, dt=DT, tol=0.0)
    meta = CheckpointManager().load(write_checkpoint(solver, str(tmp_path), 3, "mesh.nas", order=1,
                                                     turbulence_model="none", backend="cpu", quiet=True))[3]
    assert meta[FORCE_HISTORY_KEY] == pytest.approx(AREA)
    # 模拟第 2 步的 checkpoint、中断于第 3 步之后：续算时第 3 步那行被截掉，按记录的参考面积接着写
    meta["physical_time"] = 2 * DT
    resumed = _channel_solver()
    resumed.time_integrator.current_time = 2 * DT
    assert attach_force_history_on_resume(resumed, meta, tmp_path, requested=False, reference_area=None) is not None
    assert _rows(tmp_path)[:, 0].tolist() == pytest.approx([DT, 2 * DT])
    resumed.solve(max_iter=2, dt=DT, tol=0.0)
    np.testing.assert_allclose(_rows(tmp_path)[:, 0], [DT, 2 * DT, 3 * DT, 4 * DT], rtol=1e-12)
    # 原算例没有写、本次也没要求：不挂
    meta.pop(FORCE_HISTORY_KEY)
    other = _channel_solver()
    assert attach_force_history_on_resume(other, meta, tmp_path / "x", requested=False, reference_area=AREA) is None


def test_missing_reference_area_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="参考面积"):
        attach_force_history(_channel_solver(), None, tmp_path)


def test_cli_flag_reaches_the_run(tmp_path):
    from tests.unit.test_solve_transient_distributed_cli import _invoke

    assert _invoke(tmp_path, "--force-history", "--reference-area", "0.2").force_history is True
    assert _invoke(tmp_path).force_history is False
