"""湍流标量梯度不再有有量纲绝对上限（I-42）。

修复前：SST 源项（CPU、单 GPU、多 GPU）、输运、SA 源项与解析 Jacobian 把 grad k、grad ln(omega)、grad nu_tilde 的
模长截到 1e6。近壁 |grad ln omega| ~ 2/y，y < 2e-6 m 时真实梯度被截，改变交叉扩散 CD_kw 与 ln omega 方程的
(mu + sigma mu_t)|grad w|^2 源项。这里取 ln omega 沿 y 线性、斜率 3e6 /m 的场（P1 精确表示），源项收到的
grad omega 必须等于 omega * 3e6 * e_y。
"""

import inspect

import numpy as np
import pytest

from autoflowcfd.core.freestream_defaults import MU_MOLECULAR, P_INF, RHO_INF
from autoflowcfd.core.time_integration import TimeIntegrationScheme
from tests.unit.test_gpu_solver_order_continuation import _patch_gpu_modules  # noqa: F401（自动夹具）
from tests.validation._channel_mesh import build_channel_mesh_prism, channel_wall_source

SLOPE = 3.0e6                    # d ln(omega) / dy，1/m
H = 1.0e-5                       # 通道高度：ln omega 跨通道变化 30


def _mesh():
    m = build_channel_mesh_prism(1, 3, 3, 2, 4 * H, H, 0.8 * H)
    m.boundary_bc_types = {"x_min": "VELOCITY_INLET", "x_max": "PRESSURE_OUTLET", "wall_bottom": "WALL",
                           "wall_top": "WALL", "z_min": "SYMMETRY", "z_max": "SYMMETRY"}
    return m


def _solver(cls):
    from autoflowcfd.core.fr_solver import FRSolver
    from autoflowcfd.core.fr_solver.turbulence.wall_distance import apply_wall_distance_source

    wall = channel_wall_source(4 * H, H, 0.8 * H)
    kw = dict(mesh=_mesh(), order=1, turb_model_name="SST", time_scheme=TimeIntegrationScheme.SSP_RK3, vel_inf=30.0,
              rho_inf=RHO_INF, p_inf=P_INF, mu_molecular=MU_MOLECULAR)
    if cls is FRSolver:
        s = cls(**kw)
        apply_wall_distance_source(s, wall)
    else:
        s = cls(wall_distance_source=wall, **kw)
    s.order_continuation_enabled = False
    return s


def _set_log_linear_omega(model, y):
    y = np.asarray(y)
    omega = 10.0 * np.exp(SLOPE * (y - y.min()))
    model.omega_field = omega
    return omega


def _recorded_grad_omega(model, run):
    seen = {}
    orig = model.compute_source_terms

    def recording(*a, **kw):
        seen["grad_omega"] = np.asarray(inspect.signature(orig).bind(*a, **kw).arguments["grad_omega"]).copy()
        return orig(*a, **kw)
    model.compute_source_terms = recording
    try:
        run()
    finally:
        model.compute_source_terms = orig
    return seen["grad_omega"]


def _real(s):
    from autoflowcfd.fr.native_padding import real_row_mask

    n_cells, n_sps = s.mesh.sps_coords.shape[:2]
    return real_row_mask(np.arange(n_cells) < s.mesh.n_prism_cells, n_sps, 1).reshape(n_cells, n_sps)


def _check(grad_omega, omega, real):
    g, w = grad_omega[real], omega[real]
    expected = np.zeros(g.shape)
    expected[:, 1] = SLOPE * w
    np.testing.assert_allclose(g, expected, rtol=1e-8, atol=1e-8 * np.abs(expected).max())
    assert (np.abs(g[:, 1]) / w).min() > 2.9e6        # 修复前被截到 1e6


def test_cpu_source_receives_the_exact_log_omega_gradient():
    from autoflowcfd.core.fr_solver import FRSolver
    from autoflowcfd.core.fr_solver.turbulence.source import evaluate_turbulence_rates, prepare_turbulence_inputs

    s = _solver(FRSolver)
    omega = _set_log_linear_omega(s.turb_model, s.mesh.sps_coords[..., 1])
    Q, grad_vel, d, mu = prepare_turbulence_inputs(s)
    g = _recorded_grad_omega(s.turb_model, lambda: evaluate_turbulence_rates(s, Q, grad_vel, d, mu))
    _check(g, omega, _real(s))


def test_single_gpu_source_receives_the_exact_log_omega_gradient():
    from autoflowcfd.core.gpu.solver.gpu_solver import GPUFRSolver

    s = _solver(GPUFRSolver)
    omega = _set_log_linear_omega(s.turb_model_gpu, s.mesh.sps_coords[..., 1])
    g = _recorded_grad_omega(s.turb_model_gpu, lambda: s._evaluate_turbulence_rates_gpu(*s._prepare_turbulence_inputs_gpu()))
    _check(g, omega, _real(s))


def test_clip_is_gone_from_every_backend():
    """多 GPU 源项与 SA / 解析 Jacobian 路径没有可在单进程里跑的求解器，按源码钉住：不再引用已删除的上限。"""
    import importlib

    from tests.unit._module_source import module_source

    for name in ["autoflowcfd.core.gpu.distributed.gpu_distributed_init.turb_source",
                 "autoflowcfd.core.gpu.turbulence.gpu_scalar_transport.residual",
                 "autoflowcfd.core.turbulence.jacobian.pointwise", "autoflowcfd.core.turbulence.sa.linearization",
                 "autoflowcfd.core.turbulence.sa.rates", "autoflowcfd.core.turbulence.transport.residual"]:
        assert "clip_gradient_magnitude" not in module_source(importlib.import_module(name)), name
    with pytest.raises(ImportError):
        from autoflowcfd.core.turbulence.limits import MAX_GRADIENT_MAGNITUDE  # noqa: F401
