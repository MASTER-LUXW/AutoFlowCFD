"""CFL 步长计算里逐解点的几何 / 粘性量必须与当前状态同形（I-76）。

此前状态与网格的解点数不一致时把度量、涡粘平铺 / 裁剪到状态宽度：不同阶数的解点下标互不对应，那样拼出的量
没有意义（换阶时网格与状态同步切换，这条分支本不可达，可达即是错误）。
"""

import numpy as np
import pytest

from autoflowcfd.core.fr_solver.cfl import compute_local_time_step
from tests.unit.test_fixed_cfl_is_honoured import _channel_solver


def test_viscosity_field_on_another_set_of_solution_points_is_rejected(monkeypatch):
    solver = _channel_solver(0.1, adaptive=False)
    n_cells, n_sps = solver.state.U.shape[:2]
    monkeypatch.setattr(solver, "_get_cfl_viscosity_field", lambda: np.full((n_cells, 1), 1e-4))
    with pytest.raises(ValueError, match="涡粘"):
        compute_local_time_step(solver)


def test_consistent_shapes_pass():
    solver = _channel_solver(0.1, adaptive=False)
    dt = np.asarray(compute_local_time_step(solver))
    assert dt.shape == solver.state.U.shape[:2] and np.all(dt > 0.0)
