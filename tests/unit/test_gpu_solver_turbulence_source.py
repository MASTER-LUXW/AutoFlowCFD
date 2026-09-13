"""AutoFlowCFD V2.0 - 单机 GPU SST/DDES/IDDES 源项完整流程端到端验证
（2026-09-02）。

真实 bug 发现记录：排查多GPU分布式SST移植时，对照单机 GPU 路径
`gpu_solver_io.py::compute_turbulence_source_gpu` 发现它把 `grad_U`
（5 变量物理梯度，(n_cells,n_sps,5,3)）直接传给 `turb_model_gpu.
compute_source_terms_gpu`——但该方法（及其内部
`compute_strain_rate_magnitude_gpu`）按文档/CPU 版 `SSTModelFR.
compute_source_terms` 同名参数实际期望的是**速度梯度**
(n_cells,n_sps,3,3)，本方法自己在前几行就已经算出了正确的
`grad_vel = grad_U[...,1:4,:]`（供 DDES/IDDES 长度尺度用），却没有
把它传给 `compute_source_terms_gpu`。这是一个真实的、此前从未被任何
测试（哪怕 numpy 替身级别）覆盖到的 shape 不匹配 bug——真实 CUDA 环境
下，任何 `--backend gpu --turbulence-model sst` 的生产运行，第一次
调用 `compute_turbulence_source_gpu` 就会在 `cp.transpose(grad_u,
(0,1,3,2))` 处因广播失败而崩溃。已修复（`grad_U`→`grad_vel`）。

DDES/IDDES 扩展（2026-09-02 续查"GPU侧DDES/IDDES/WMLES/LES是否有同类
未发现的bug"）：DDES/IDDES 的长度尺度替换代码（`apply_to_sst_model_
gpu`/`apply_to_sst_model_iddes_gpu`）本身调用点用的就是正确的
`grad_vel`（不是这次修复的那个 bug），但两者共享同一个此前有 bug 的
`compute_source_terms_gpu` 调用——本次一并做端到端验证，确认它们也
真的受益于上面的修复，且没有各自独立的其他问题。

本测试用同一套"把 numpy 伪装成 CuPy"的替身模式（本机无真实 CuPy），
对完整的 `compute_turbulence_source_gpu` 方法本体（不是它内部的某个
子函数）做端到端调用，验证：(1) 不再因 shape 不匹配崩溃；(2) 产出的
k/omega/nu_t 更新结果与 CPU 单机路径 `compute_turbulence_source`
逐位一致（两者是同一套数值算法，只是张量库不同）。
"""

import types

import numpy as np
import pytest

from autoflowcfd.fr.operators import generate_fr_operators
from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
from autoflowcfd.core.fr_residual.inviscid import primitive_to_conserved, conserved_to_primitive
from autoflowcfd.core.turbulence.sst import SSTModelFR
from autoflowcfd.core.turbulence.des import DDESModel, IDDESModel, compute_h_max_and_h_wn
from autoflowcfd.core.gpu.turbulence.gpu_turbulence_sst import GPUTurbulenceSST
from autoflowcfd.core.gpu.turbulence.gpu_turbulence_des import GPUDDESModel, GPUIDDESModel
from autoflowcfd.core.gpu.solver.gpu_solver_io import _GPUSolverIOMixin
from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh

import autoflowcfd.core.gpu.solver.gpu_solver_io as gpu_solver_io_mod
import autoflowcfd.core.gpu.solver.gpu_solver as gpu_solver_mod
import autoflowcfd.core.gpu.residual.gpu_gradients as gpu_gradients_mod
import autoflowcfd.core.gpu.residual.gpu_volume_contract as gpu_volume_contract_mod
import autoflowcfd.core.gpu.residual.gpu_flux as gpu_flux_mod
import autoflowcfd.core.gpu.turbulence.gpu_scalar_transport as gst_mod
import autoflowcfd.core.gpu.turbulence.gpu_turbulence_sst as gpu_turbulence_sst_mod
import autoflowcfd.core.gpu.turbulence.gpu_turbulence_des as gpu_turbulence_des_mod
import autoflowcfd.core.gpu.gpu_modal_filter as gpu_modal_filter_mod


class _NumpyAsCupy:
    def __getattr__(self, name):
        return getattr(np, name)

    def scatter_add(self, a, indices, b):
        np.add.at(a, indices, b)

    def asnumpy(self, x):
        return np.asarray(x)

    class cuda:
        class Device:
            def __init__(self, device_id):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False


@pytest.fixture(autouse=True)
def _patch_get_cupy(monkeypatch):
    shim = _NumpyAsCupy()
    for mod in (gpu_solver_io_mod, gpu_solver_mod, gpu_gradients_mod, gpu_volume_contract_mod,
                gpu_flux_mod, gst_mod, gpu_turbulence_sst_mod, gpu_turbulence_des_mod,
                gpu_modal_filter_mod):
        monkeypatch.setattr(mod, "get_cupy", lambda: shim)
    monkeypatch.setattr(gpu_turbulence_sst_mod, "gpu_available", True)
    monkeypatch.setattr(gpu_turbulence_des_mod, "gpu_available", True)


def _prepare_mesh_ops_data(mesh, ops):
    n_cells = mesh.n_cells
    n_sps = mesh.n_sps_per_cell
    det_jacs = mesh.jacobians['det_jacs'].reshape(n_cells, n_sps)
    inv_jacs = mesh.jacobians['inv_jacs'].reshape(n_cells, n_sps, 3, 3)
    adj_j = det_jacs[..., None, None] * inv_jacs
    mesh_data = {
        'det_jacs': det_jacs, 'inv_jacs': inv_jacs, 'adj_j': adj_j,
        'n_cells': n_cells, 'n_prism': mesh.n_prism_cells,
        'cell_volumes': mesh.cell_volumes,
        'D_3d_prism': ops.D_3d_prism, 'D_3d_tet': ops.D_3d_tet,
    }
    return mesh_data


def _nonuniform_state(mesh, rng):
    rho_inf, u_inf, v_inf, w_inf, p_inf = 1.225, 30.0, 5.0, -3.0, 101325.0
    n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
    Q = np.zeros((n_cells, n_sps, 5))
    Q[..., 0] = rho_inf * (1.0 + rng.uniform(-0.02, 0.02, size=(n_cells, n_sps)))
    Q[..., 1] = u_inf + rng.uniform(-3.0, 3.0, size=(n_cells, n_sps))
    Q[..., 2] = v_inf + rng.uniform(-2.0, 2.0, size=(n_cells, n_sps))
    Q[..., 3] = w_inf + rng.uniform(-2.0, 2.0, size=(n_cells, n_sps))
    Q[..., 4] = p_inf * (1.0 + rng.uniform(-0.01, 0.01, size=(n_cells, n_sps)))
    U = np.stack(
        [primitive_to_conserved(Q[c, s]) for c in range(n_cells) for s in range(n_sps)]
    ).reshape(n_cells, n_sps, 5)
    k_field = 1.0 * (1.0 + rng.uniform(-0.3, 0.3, size=(n_cells, n_sps)))
    omega_field = 500.0 * (1.0 + rng.uniform(-0.3, 0.3, size=(n_cells, n_sps)))
    return U, k_field, omega_field


def _make_ddes_model_cpu(turb_model_name):
    if turb_model_name == "SST":
        return None
    if turb_model_name == "DDES":
        return DDESModel()
    if turb_model_name == "IDDES":
        return IDDESModel()
    raise ValueError(turb_model_name)


def _make_ddes_model_gpu(turb_model_name):
    if turb_model_name == "SST":
        return None
    if turb_model_name == "DDES":
        return GPUDDESModel()
    if turb_model_name == "IDDES":
        return GPUIDDESModel()
    raise ValueError(turb_model_name)


@pytest.mark.parametrize("turb_model_name", ["SST", "DDES", "IDDES"])
def test_compute_turbulence_source_gpu_matches_cpu_single_machine(turb_model_name):
    order = 1
    mesh = _build_synthetic_mixed_mesh(order)
    ops = generate_fr_operators(order)
    n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
    mu = 1.8e-5
    rng = np.random.default_rng(4242)

    U, k_field, omega_field = _nonuniform_state(mesh, rng)
    d_wall = np.full((n_cells, n_sps), 0.05)

    h_max = h_wn = None
    if turb_model_name == "IDDES":
        h_max, h_wn = compute_h_max_and_h_wn(mesh)

    # ---- GPU（numpy 替身）：调用完整的 compute_turbulence_source_gpu ----
    mesh_data = _prepare_mesh_ops_data(mesh, ops)
    turb_gpu = GPUTurbulenceSST(n_cells, n_sps, device_id=0)
    turb_gpu.k_field = k_field.copy()
    turb_gpu.omega_field = omega_field.copy()

    flat = get_flat_face_geometry(mesh, ops)  # numpy，充当"flat_face_gpu"

    dt_used = 1e-3  # 见 TestDdesModelDistinctFromSst 同一处 dt 放大注释

    stub = types.SimpleNamespace(
        mesh=mesh, mesh_data=mesh_data, ops_data=mesh_data, ops=ops,
        U_gpu=U, Q_gpu=conserved_to_primitive(U[..., :5]),
        mu_molecular=mu, turb_model_gpu=turb_gpu, sgs_model_gpu=None,
        ddes_model_gpu=_make_ddes_model_gpu(turb_model_name),
        _iddes_h_max_gpu=h_max, _iddes_h_wn_gpu=h_wn,
        wall_distance_gpu=d_wall,
        turb_model_name=turb_model_name, flat_face_gpu=flat,
        _wall_mask_k_gpu=np.zeros(flat.n_faces, dtype=bool),
        time_integrator=types.SimpleNamespace(cfl=0.5),
        order=order,
        # 产项渐变因子已完成状态（两侧 CPU/GPU 必须显式设成同一个值才
        # 能逐位对照——GPU 版 `_update_production_ramp_gpu` 在
        # `_turb_production_ramp_steps` 缺失时默认 50（渐变中，
        # production_factor 在前 50 步内 <1），CPU 版
        # `_update_production_ramp` 缺失时 `getattr(...,0)` 默认 0
        # （渐变立即完成，production_factor=1.0）——两者默认值不同，
        # 若两侧测试 stub 都不显式设置会造成一个纯测试搭建失误导致的
        # 虚假数值差异，不是真实的 GPU/CPU 行为不一致（生产路径下
        # `init_turbulence_models`/GPU 对应构造都会显式设 50，不会
        # 依赖这个容易踩坑的默认值分叉）。
        _turb_ramp_step=10 ** 9, _turb_production_ramp_steps=0,
    )
    # `SimpleNamespace` 不会自动绑定方法——`_update_production_ramp_gpu`
    # 以 `self.xxx()` 形式被内部调用，需要显式绑定成 stub 的方法。
    # `_compute_local_time_step_gpu` 直接换成返回常量 dt 的替身——它
    # 本身是一整条独立的 CFL 几何/度量计算链路（这里不是本测试要验证
    # 的对象，本测试关注的是 grad_U/grad_vel 这一处 bug；用真实 CFL
    # 计算只会引入一堆无关的 get_cupy() patch 需求），与 CPU 参照两侧
    # 用同一个常量 dt，保证 SST 数值更新公式本身可以逐位对照。
    stub._update_production_ramp_gpu = types.MethodType(
        _GPUSolverIOMixin._update_production_ramp_gpu, stub
    )
    stub._compute_local_time_step_gpu = lambda: np.full(n_cells, dt_used)

    mu_t_gpu = _GPUSolverIOMixin.compute_turbulence_source_gpu(stub)

    # ---- CPU 单机参照 ----
    from autoflowcfd.core.fr_solver.turbulence import compute_turbulence_source

    turb_cpu = SSTModelFR(n_cells, n_sps)
    turb_cpu.k_field = k_field.copy()
    turb_cpu.omega_field = omega_field.copy()

    class _CpuStub:
        def __init__(self_inner):
            self_inner.mesh = mesh
            self_inner.ops = ops
            self_inner.turb_model = turb_cpu
            self_inner.turb_model_name = turb_model_name
            self_inner.ddes_model = _make_ddes_model_cpu(turb_model_name)
            self_inner._iddes_h_max = h_max
            self_inner._iddes_h_wn = h_wn
            self_inner.wall_distance = d_wall
            self_inner.mu_molecular = mu
            self_inner.state = types.SimpleNamespace(
                U=U, Q=conserved_to_primitive(U[..., :5]), n_cells=n_cells, n_sps=n_sps,
            )
            self_inner._turbulence_flat_face_override = None
            self_inner._turb_ramp_step = 10 ** 9
            self_inner._turb_production_ramp_steps = 0

        def _compute_gradients(self_inner):
            from autoflowcfd.core.fr_operators.gradients import compute_physical_gradient
            return compute_physical_gradient(U[..., :5], mesh, ops)

        def _get_cell_volumes(self_inner):
            return mesh.cell_volumes

    # dt_local 用同一个常量（见上方 stub._compute_local_time_step_gpu
    # 替身文档），两侧严格用同一个 dt 才能做逐位对照。
    compute_turbulence_source(_CpuStub(), np.full((n_cells, n_sps), dt_used))

    assert np.all(np.isfinite(turb_gpu.k_field)), (
        f"GPU {turb_model_name} 源项计算不应该产生非有限值（回归 grad_U/grad_vel bug 的直接症状）"
    )
    np.testing.assert_allclose(turb_gpu.k_field, turb_cpu.k_field, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(turb_gpu.omega_field, turb_cpu.omega_field, rtol=1e-8, atol=1e-6)
    np.testing.assert_allclose(turb_gpu.nu_t, turb_cpu.nu_t, rtol=1e-8, atol=1e-12)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
