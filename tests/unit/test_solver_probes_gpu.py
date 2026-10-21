"""GPU 后端的探针采样（I-170）：单 GPU 求解器（numpy 充当 cupy）真实求解写出时间序列；多 GPU 只从设备拷探针所在的
本 rank 行（`U_gpu` 为本 rank local 单元、原生排列）。"""

from types import SimpleNamespace

import numpy as np

from autoflowcfd.core.freestream_defaults import MU_MOLECULAR, P_INF, RHO_INF
from autoflowcfd.core.probes import read_probe_series
from autoflowcfd.core.probes.recorder import _state_rows, attach_probes
from tests.unit.test_gpu_solver_order_continuation import _patch_gpu_modules  # noqa: F401（自动夹具）
from tests.validation._channel_mesh import build_channel_mesh_prism, channel_wall_source

_BC = {"x_min": "VELOCITY_INLET", "x_max": "PRESSURE_OUTLET", "wall_bottom": "WALL", "wall_top": "WALL",
       "z_min": "SYMMETRY", "z_max": "SYMMETRY"}


def test_single_gpu_solve_records_probe_series(tmp_path):
    from autoflowcfd.core.gpu.solver.gpu_solver import GPUFRSolver
    from autoflowcfd.core.probes.recorder import _interpolation_weights
    from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive

    mesh = build_channel_mesh_prism(1, 3, 3, 2, 0.4, 0.1, 0.08)
    mesh.boundary_bc_types = dict(_BC)
    solver = GPUFRSolver(mesh=mesh, order=1, turb_model_name="NONE", vel_inf=10.0, rho_inf=RHO_INF, p_inf=P_INF,
                         mu_molecular=MU_MOLECULAR, wall_distance_source=channel_wall_source(0.4, 0.1, 0.08),
                         time_scheme="ssp_rk3", time_accurate=True)
    solver.order_continuation_enabled = False
    rec = attach_probes(solver, [[0.13, 0.05, 0.03]], tmp_path)
    solver.solve(max_iter=2, dt=1e-5, tol=0.0)
    _points, times, values = read_probe_series(tmp_path)
    np.testing.assert_allclose(times, [1e-5, 2e-5], rtol=1e-12)
    U = np.asarray(solver.host_view().state.U)[rec.locations.cells, :, :5]
    expected = conserved_to_primitive(np.einsum("ks,ksv->kv", _interpolation_weights(1, rec.locations, U.shape[1]), U))
    np.testing.assert_allclose(values["p"][-1], expected[:, 4], rtol=1e-14)


def test_multi_gpu_rows_come_from_the_local_device_array():
    U_local = np.random.default_rng(3).random((4, 6, 5))
    solver = SimpleNamespace(U_gpu=U_local, state=SimpleNamespace(U=np.zeros_like(U_local)))   # 主机副本未同步
    np.testing.assert_array_equal(_state_rows(solver, np.array([3, 1])), U_local[[3, 1]])
