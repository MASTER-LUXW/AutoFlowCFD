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
import autoflowcfd.core.gpu.turbulence.gpu_implicit_turbulence as gpu_implicit_turb_mod
from tests.unit._gpu_cupy_shim import patch_module_get_cupy


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
    patch_module_get_cupy(monkeypatch, [
        gpu_solver_io_mod, gpu_solver_mod, gpu_gradients_mod, gpu_volume_contract_mod,
        gpu_flux_mod, gst_mod, gpu_turbulence_sst_mod, gpu_turbulence_des_mod,
        gpu_modal_filter_mod, gpu_implicit_turb_mod], shim)
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
    # 补齐生产 GPU 路径会上传、替身容易漏掉的键，见 _gpu_standin_helpers
    # 模块文档。本文件把算子与网格数据放在同一个 dict 里，两个参数传同
    # 一个对象。
    from ._gpu_standin_helpers import complete_gpu_standin
    complete_gpu_standin(mesh, ops, mesh_data, mesh_data)
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


def _build_standins(turb_model_name):
    """同一份非均匀状态上的 GPU（numpy 替身）与 CPU 单机湍流求解器替身。

    两侧用同一个常量 dt、同一个产项渐变状态，保证可以逐位对照（见下方
    各处注释）。显式源项测试与隐式 k-omega Newton 测试共用。
    """
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
        _open_mask_gpu=np.zeros(flat.n_faces, dtype=bool),
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
    # 2026-09-25：湍流源项拆成 prepare/evaluate/finalize 三个件（显式与隐式
    # k-omega 更新共用），替身同样绑定真实类里的这几个方法。
    for _name in ("_prepare_turbulence_inputs_gpu", "_evaluate_turbulence_rates_gpu",
                  "_finalize_turbulence_update_gpu", "_turbulent_mu_t_gpu"):
        setattr(stub, _name, types.MethodType(getattr(_GPUSolverIOMixin, _name), stub))
    # 签名随 2026-09-14 低马赫数预处理接入 GPU 而变化：湍流场更新必须
    # 取**物理**波速算出的那一份 dt（`return_physical_too=True` 的第二个
    # 返回值），不能跟着平均流的预处理步长放大约 7 倍——k/omega 的显式
    # 更新刻意没有 point-implicit 阻尼。替身这里两份都给同一个常量，
    # 与 CPU 参照严格对齐，仍然满足本测试"逐位对照 SST 更新公式"的目的。
    def _dt_stub(return_physical_too=False):
        dt_arr = np.full(n_cells, dt_used)
        return (dt_arr, dt_arr) if return_physical_too else dt_arr

    stub._compute_local_time_step_gpu = _dt_stub

    # ---- CPU 单机参照 ----

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

    cpu = _CpuStub()
    cpu.order = order
    return types.SimpleNamespace(gpu=stub, cpu=cpu, turb_gpu=turb_gpu, turb_cpu=turb_cpu,
                                 n_cells=n_cells, n_sps=n_sps, dt_used=dt_used)


@pytest.mark.parametrize("turb_model_name", ["SST", "DDES", "IDDES"])
def test_compute_turbulence_source_gpu_matches_cpu_single_machine(turb_model_name):
    from autoflowcfd.core.fr_solver.turbulence import compute_turbulence_source

    b = _build_standins(turb_model_name)
    stub, turb_gpu, turb_cpu = b.gpu, b.turb_gpu, b.turb_cpu
    n_cells, n_sps, dt_used = b.n_cells, b.n_sps, b.dt_used
    # 逐单元不同的局部步长：GPU 与 CPU 必须逐点用同一个 dt（2026-09-25 以前
    # GPU 取全场均值，在非均匀 dt 上与 CPU 不一致）
    dt_cells = dt_used * (1.0 + np.arange(n_cells) / n_cells)
    _GPUSolverIOMixin.compute_turbulence_source_gpu(stub, dt_cells[:, None])
    compute_turbulence_source(b.cpu, np.repeat(dt_cells[:, None], n_sps, axis=1))

    assert np.all(np.isfinite(turb_gpu.k_field)), (
        f"GPU {turb_model_name} 源项计算不应该产生非有限值（回归 grad_U/grad_vel bug 的直接症状）"
    )
    np.testing.assert_allclose(turb_gpu.k_field, turb_cpu.k_field, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(turb_gpu.omega_field, turb_cpu.omega_field, rtol=1e-8, atol=1e-6)
    np.testing.assert_allclose(turb_gpu.nu_t, turb_cpu.nu_t, rtol=1e-8, atol=1e-12)




@pytest.mark.parametrize("turb_model_name", ["SST", "DDES"])
def test_implicit_turbulence_newton_gpu_matches_cpu(turb_model_name):
    """隐式 k-omega Newton 步：GPU 适配器（numpy 替身，走块 Jacobi 的 cupy
    分支——批量 `xp.linalg.inv`）与 CPU 适配器（numba 分支）在同一状态上
    给出同一个结果。算法只有一份（`fr_solver/turbulence/implicit.py`），
    两侧只差求值件与数组模块，所以差异只能来自那里。"""
    from autoflowcfd.core.fr_solver.turbulence.implicit import (
        CpuTurbulenceBackend,
        step_turbulence_newton,
    )
    from autoflowcfd.core.gpu.turbulence.gpu_implicit_turbulence import GpuTurbulenceBackend

    b = _build_standins(turb_model_name)
    b.gpu._newton_turb_state = None
    b.cpu._newton_turb_state = None
    dtau = np.full((b.n_cells, b.n_sps), 50.0 * b.dt_used)   # 远超显式极限
    step_turbulence_newton(GpuTurbulenceBackend(b.gpu), dtau)
    step_turbulence_newton(CpuTurbulenceBackend(b.cpu), dtau)

    ig = b.gpu._newton_turb_state["last_info"]
    ic = b.cpu._newton_turb_state["last_info"]
    assert ig["theta"] > 0.0 and ic["theta"] > 0.0
    assert ig["gmres_iters"] == ic["gmres_iters"]
    np.testing.assert_allclose(b.turb_gpu.k_field, b.turb_cpu.k_field, rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(b.turb_gpu.omega_field, b.turb_cpu.omega_field, rtol=1e-6)
    np.testing.assert_allclose(b.turb_gpu.nu_t, b.turb_cpu.nu_t, rtol=1e-6, atol=1e-14)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
