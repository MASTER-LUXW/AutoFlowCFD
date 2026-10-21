"""AUSM+up 通量不带有量纲的状态下限（I-18，2026-10-19）。

此前 CPU 核、GPU 批量核与 P0 的 CUDA 核在入口把 rho 钳到 1e-6、p 钳到 10 Pa，GPU 的守恒→原始变量换算另把 p 钳到
10 Pa：有量纲空气算例从不触发，无量纲算例（p ~ 1/GAMMA）则被静默改写。判据是 Euler 通量的量纲一致性：把两侧状态
无量纲化（rho/rho0、u/u0、p/(rho0 u0^2)），通量按 (rho0 u0, rho0 u0^2, rho0 u0^3) 还原后必须与有量纲通量相同
（同一个参考马赫数下 AUSM+up 的全部系数都是无量纲的）。
"""

import numpy as np

from autoflowcfd.core.fr_operators.kernels import compute_ausm_up_flux
from autoflowcfd.core.thermo import GAMMA

RHO0, U0 = 1.225, 340.0
SCALE = np.array([RHO0 * U0, RHO0 * U0 ** 2, RHO0 * U0 ** 2, RHO0 * U0 ** 2, RHO0 * U0 ** 3])
NORMAL = np.array([0.6, 0.8, 0.0])
MACH_REF = 0.1


def _states():
    qL = np.array([1.225, 30.0, -4.0, 2.0, 101325.0])
    qR = np.array([1.180, 26.0, -1.0, 3.5, 100100.0])
    return qL, qR


def _nondim(q):
    return np.array([q[0] / RHO0, q[1] / U0, q[2] / U0, q[3] / U0, q[4] / (RHO0 * U0 ** 2)])


def test_cpu_flux_is_dimensionally_consistent():
    qL, qR = _states()
    f_dim = compute_ausm_up_flux(qL, qR, NORMAL, MACH_REF)
    f_nd = compute_ausm_up_flux(_nondim(qL), _nondim(qR), NORMAL, MACH_REF)
    assert _nondim(qL)[4] < 1.0                           # 无量纲压力约 0.7：旧的 10 Pa 下限会改写它
    np.testing.assert_allclose(f_nd * SCALE, f_dim, rtol=1e-12, atol=1e-12 * np.abs(f_dim).max())


def test_gpu_batch_flux_matches_cpu_without_floors(monkeypatch):
    from autoflowcfd.core.gpu.residual import gpu_inviscid
    from tests.unit._gpu_cupy_shim import patch_module_get_cupy
    from tests.unit._numpy_as_cupy import NumpyAsCupy

    patch_module_get_cupy(monkeypatch, gpu_inviscid, NumpyAsCupy())
    from autoflowcfd.core.gpu.residual.gpu_inviscid.flux import _ausm_up_flux_batch_gpu

    qL, qR = (_nondim(q) for q in _states())
    got = _ausm_up_flux_batch_gpu(qL[None, None], qR[None, None], NORMAL[None, None], MACH_REF)[0, 0]
    np.testing.assert_allclose(got, compute_ausm_up_flux(qL, qR, NORMAL, MACH_REF), rtol=1e-13, atol=1e-15)


def test_gpu_primitive_conversion_has_no_pressure_floor(monkeypatch):
    from autoflowcfd.core.gpu.residual import gpu_flux
    from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive, primitive_to_conserved
    from tests.unit._gpu_cupy_shim import patch_module_get_cupy
    from tests.unit._numpy_as_cupy import NumpyAsCupy

    patch_module_get_cupy(monkeypatch, gpu_flux, NumpyAsCupy())
    U = primitive_to_conserved(np.array([[1.0, 0.1, 0.0, 0.0, 1.0 / GAMMA]]))
    np.testing.assert_allclose(gpu_flux.conserved_to_primitive_gpu(U), conserved_to_primitive(U), rtol=1e-14)


def test_nonpositive_state_is_exposed_not_masked():
    """非正状态说明上游正性保证失效：通量给出 NaN（由发散检测报告），而不是被钳成某个看似合理的值。"""
    qL, qR = _states()
    qL = qL.copy()
    qL[4] = -1.0
    assert not np.all(np.isfinite(compute_ausm_up_flux(qL, qR, NORMAL, MACH_REF)))
