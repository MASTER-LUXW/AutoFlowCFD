"""守恒量 -> 原始量换算与声速各只有一份、不设量纲压力下限（I-116 / I-81(1)）。

此前 CPU 求解器状态另有一份换算核、压力钳到 1 Pa；CPU CFL 按 a^2 >= 1e-10 兜底，GPU CFL 按 p >= 1 Pa、rho >= 1e-9
兜底——同一状态在两个后端给出不同步长，钳位掩盖非物理态，无量纲算例（p_inf ~ 1/GAMMA）被直接改写。
"""

import numpy as np

from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive, primitive_to_conserved
from autoflowcfd.core.fr_solver.state import FRState
from autoflowcfd.core.thermo import GAMMA, sound_speed
from tests.unit._module_source import module_source


def test_state_primitives_use_the_residual_conversion_without_a_pressure_floor():
    state = FRState(2, 3)
    Q = np.array([1.0, 0.3, -0.2, 0.1, 1.0 / GAMMA])          # 无量纲算例：p_inf = 1/GAMMA < 1 Pa
    state.U[:] = primitive_to_conserved(Q)
    state.U[1, 2] = primitive_to_conserved(np.array([1.0, 0.0, 0.0, 0.0, 0.25]))
    state._update_primitives()
    np.testing.assert_array_equal(state.Q, conserved_to_primitive(state.U))
    assert abs(state.Q[0, 0, 4] - 1.0 / GAMMA) < 1e-14 and abs(state.Q[1, 2, 4] - 0.25) < 1e-14


def test_sound_speed_has_no_dimensional_floor():
    np.testing.assert_allclose(sound_speed(np, np.array([1.0]), np.array([1.0 / GAMMA])), [1.0])
    with np.errstate(invalid="ignore"):
        assert np.isnan(sound_speed(np, np.array([1.0]), np.array([-1.0]))[0])     # 非物理态不被兜底


def test_both_cfl_paths_use_the_shared_sound_speed():
    # 全部后端的 CFL 步长经同一个函数（`cfl_limits.local_time_step_limits`，I-115），声速在里面按数组模块算；
    # GPU 后端先用与残差同一个换算得到原始变量
    assert "sound_speed(xp," in module_source("autoflowcfd.core.fr_solver.cfl_limits")
    for mod in ("autoflowcfd.core.gpu.solver.gpu_solver", "autoflowcfd.core.gpu.distributed.gpu_distributed"):
        src = module_source(mod)
        assert "local_time_step_limits(" in src and "conserved_to_primitive_gpu(self.U_gpu" in src, mod
