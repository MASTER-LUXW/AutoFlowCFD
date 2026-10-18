"""湍流平板：RANS 近壁处理（`core/turbulence/wall_model/treatment.py`）在对数律层网格上的摩阻。

算例 `_flat_plate_case.py`（P1 SST、隐式稳态），近壁第一层 2 mm（贴壁单元形心 y+ 约 190，对数律层）：

* ``auto``（平衡壁面模型）：局部摩阻与 White 关联式一致到几个百分点（2026-10-18 实测 +0.5% ~ +3.3%）；
* ``resolved``（低雷诺数壁面处理）：贴壁单元越出粘性底层，壁面剪应力只来自贴壁单元内的速度梯度，摩阻低 97%——
  这正是 auto 要解决的问题；收尾的 y+ 报告（`postprocess/wall_yplus.py`）对这种组合报警。

同一算例在壁面解析网格上（第一层 2e-5 m，形心 y+ 约 1.8；单次 6~10 分钟，未列为常规测试）实测：
resolved -13.2%（231 步）、auto -6.4%（215 步）。
"""

import numpy as np
import pytest

from autoflowcfd.core.time_integration.base import STEADY_DT
from tests.validation._flat_plate_case import build_flat_plate_solver, wall_cf, wall_model_cf, white_cf

DY_WALL, NY = 2e-3, 12


def _solve(treatment):
    solver, _ = build_flat_plate_solver(1, ny=NY, dy_wall=DY_WALL, turb_model="SST", wall_treatment=treatment)
    result = solver.solve(max_iter=400, dt=STEADY_DT, tol=1e-6)
    x, cf = wall_model_cf(solver) if treatment == "auto" else wall_cf(solver)
    sel = (x > 0.5) & (x < 1.8)
    return solver, result, (cf[sel] - white_cf(x[sel])) / white_cf(x[sel])


@pytest.fixture(scope="module")
def auto_run():
    return _solve("auto")


def test_auto_matches_white_on_a_log_layer_mesh(auto_run):
    _, result, err = auto_run
    assert result.converged
    assert abs(err.mean()) < 0.04 and np.abs(err).max() < 0.06, err


def test_auto_y_plus_report_is_in_the_log_layer_without_warning(auto_run, monkeypatch):
    from autoflowcfd.postprocess import wall_yplus

    solver, _, _ = auto_run
    warnings = []
    monkeypatch.setattr(wall_yplus.logger, "warning", lambda msg: warnings.append(msg))
    yp = wall_yplus.report_wall_y_plus(solver, "SST", "auto")
    on_plate = yp.position[:, 0] > 0.5
    assert 30.0 < np.median(yp.y_plus[on_plate]) < wall_yplus.LOG_LAYER_Y_PLUS
    assert not warnings


def test_resolved_underpredicts_and_the_report_warns(monkeypatch):
    from autoflowcfd.postprocess import wall_yplus

    solver, result, err = _solve("resolved")
    assert result.converged
    assert err.mean() < -0.5                              # 越出粘性底层：壁面剪应力严重偏低
    warnings = []
    monkeypatch.setattr(wall_yplus.logger, "warning", lambda msg: warnings.append(msg))
    wall_yplus.report_wall_y_plus(solver, "SST", "resolved")
    assert warnings
