"""单 GPU 的局部时间步长与 CPU 同一口径（2026-10-09）。

此前单 GPU 的对流步长限制里，面积取的是 `inviscid_p0.py::_extract_p0_face_geometry` 的"面积权重"——每面第一个
通量点的求积权重，只在 P0 等于整面面积。P1 下各单元的面积和只有真实值的 1/3.35（棱柱）/ 1/2.54（四面体），
P2 只有 1/10.3 / 1/7.30，步长随之放大同样的倍数：单 GPU 的实际 CFL 是名义值的 3~10 倍，而 CPU、CPU 分布式与
多 GPU 取的都是 `face_connectivity` 的物理面积。
"""

import numpy as np
import pytest

from autoflowcfd.core.time_integration import TimeIntegrationScheme
from tests.unit.test_gpu_solver_order_continuation import _patch_gpu_modules  # noqa: F401（自动夹具）
from tests.validation._channel_mesh import build_channel_mesh, build_channel_mesh_prism

LX, H, LZ = 0.4, 0.1, 0.08
_BC = {"x_min": "VELOCITY_INLET", "x_max": "PRESSURE_OUTLET", "wall_bottom": "WALL", "wall_top": "WALL",
       "z_min": "SYMMETRY", "z_max": "SYMMETRY"}


def _pair(kind, order, scheme):
    from autoflowcfd.core.fr_solver import FRSolver
    from autoflowcfd.core.gpu.solver.gpu_solver import GPUFRSolver

    def mesh():
        m = (build_channel_mesh if kind == "tet" else build_channel_mesh_prism)(order, 3, 3, 2, LX, H, LZ)
        m.boundary_bc_types = dict(_BC)
        return m
    kw = dict(order=order, turb_model_name="NONE", time_scheme=scheme, vel_inf=30.0)
    cpu, gpu = FRSolver(mesh=mesh(), **kw), GPUFRSolver(mesh=mesh(), **kw)
    for s in (cpu, gpu):
        s.order_continuation_enabled = False
    return cpu, gpu


@pytest.mark.parametrize("kind", ["prism", "tet"])
@pytest.mark.parametrize("order", [0, 1, 2])
@pytest.mark.parametrize("scheme", [TimeIntegrationScheme.SSP_RK3, TimeIntegrationScheme.DUAL_TIME])
def test_single_gpu_local_time_step_equals_cpu(kind, order, scheme):
    """SSP-RK3 带低马赫预处理（两份步长都比），DUAL_TIME 不带。CPU 返回逐解点的步长，GPU 取单元内真实解点的
    最小值；均匀来流下两者应一致到舍入。"""
    from autoflowcfd.fr.native_padding import reduce_per_cell_over_real_sps

    cpu, gpu = _pair(kind, order, scheme)
    dt_mean, dt_phys = cpu._compute_local_time_step(return_physical_too=True)
    g_mean, g_phys = gpu._compute_local_time_step_gpu(return_physical_too=True)
    n_prism = int(cpu.mesh.n_prism_cells)
    for name, c, g in (("平均流", dt_mean, g_mean), ("物理波速", dt_phys, g_phys)):
        c = reduce_per_cell_over_real_sps(np.asarray(c), n_prism, order, 'min')
        np.testing.assert_allclose(np.asarray(g), c, rtol=1e-12, err_msg=f"{kind} P{order} {scheme.value} {name}")


def test_face_geometry_is_uploaded_once():
    _cpu, gpu = _pair("prism", 1, TimeIntegrationScheme.SSP_RK3)
    first = gpu._face_geometry_gpu()
    gpu._compute_local_time_step_gpu()
    assert gpu._face_geometry_gpu() is first


@pytest.mark.parametrize("kind", ["prism", "tet"])
@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("scheme", [TimeIntegrationScheme.SSP_RK3, TimeIntegrationScheme.DUAL_TIME])
def test_time_step_definition_is_the_same_on_a_nonuniform_state(kind, order, scheme):
    """非均匀状态下 CPU 与单 GPU 仍一致，且 CPU 的步长在单元内为常数。

    2026-10-09 以前 CPU 的对流谱半径只取每个单元第 0 个解点的速度与声速（GPU 逐解点），并且返回逐解点各不
    相同的步长（GPU 在单元的真实解点上取最小）——均匀来流下看不出差别。"""
    from autoflowcfd.fr.native_padding import real_row_mask

    cpu, gpu = _pair(kind, order, scheme)
    rng = np.random.default_rng(11)
    n_cells, n_sps = cpu.state.U.shape[:2]
    real = real_row_mask(np.arange(n_cells) < int(cpu.mesh.n_prism_cells), n_sps, order).reshape(n_cells, n_sps)
    U = cpu.state.U.copy()
    U[..., 0] *= 1.0 + 0.05 * rng.standard_normal((n_cells, n_sps))
    U[..., 1:4] += 0.5 * cpu.state.U[..., 1:2].max() * rng.standard_normal((n_cells, n_sps, 3))
    U[..., 4] *= 1.0 + 0.05 * rng.standard_normal((n_cells, n_sps))
    U[~real] = cpu.state.U[~real]                 # 填充槽位保持初值（生产中它们冻结在初值）
    cpu.state.U = U
    cpu.state._update_primitives()
    with gpu.edit_host_state() as host:
        host.state.U = U.copy()
        host.state._update_primitives()

    dt_mean, dt_phys = cpu._compute_local_time_step(return_physical_too=True)
    g_mean, g_phys = gpu._compute_local_time_step_gpu(return_physical_too=True)
    for name, c, g in (("平均流", dt_mean, g_mean), ("物理波速", dt_phys, g_phys)):
        c = np.asarray(c)
        assert np.ptp(c, axis=1).max() == 0.0, f"{name}：CPU 步长在单元内不是常数"
        np.testing.assert_allclose(np.asarray(g), c[:, 0], rtol=1e-11, err_msg=f"{kind} P{order} {scheme.value} {name}")
    assert np.ptp(np.asarray(dt_phys)[:, 0]) > 0.0     # 状态确实非均匀（各单元步长不同）
