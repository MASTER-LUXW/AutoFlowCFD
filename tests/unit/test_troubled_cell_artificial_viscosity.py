"""问题单元人工粘性（熵残差判据 + 守恒变量拉普拉斯，2026-10-01）的判据性测试。

钉住的性质（判据 r = (h/p) max |u_hat . grad s|，s = ln(p/rho^gamma)）：

1. 斜坡：r<=R_LOW 为 0、r>=R_FULL 为 1、两端导数为零（C¹）、单调。
2. **不该触发的地方不触发**：均匀流；等熵变化（p 与 rho 按 p ~ rho^gamma 大幅
   变化，熵处处相同）；熵只沿**壁面法向**变化、流动沿壁面（边界层的物理熵
   梯度形态）。
3. 等压熵尖峰（plate_demo 锐边热斑的形态）触发，且系数恰为
   `ramp * alpha * |u|_mean * h / p`（运动粘度）。
4. 系数随尖峰幅度**连续**变化（门限处从 0 起步）。
6. 施加形式：五个守恒量的拉普拉斯、边界齐次 Neumann —— 每个守恒量的
   全域积分变化恒为零（质量、动量、能量都守恒），且对单点扰动是耗散的。
5. order 0 返回全零。
"""

import numpy as np
import pytest

from autoflowcfd.core.fr_operators.artificial_viscosity import (
    ENTROPY_SENSOR_FULL, ENTROPY_SENSOR_LOW, compute_artificial_diffusivity,
    compute_entropy_sensor, entropy_sensor_ramp,
)
from autoflowcfd.core.fr_residual.viscous import compute_scalar_gradient
from autoflowcfd.core.fr_solver.solver import FRSolver
from autoflowcfd.core.time_integration import TimeIntegrationScheme
from tests.validation._channel_mesh import (
    build_channel_mesh_prism, build_face_exact_ghost_provider)

H, LX, LZ = 1.0e-2, 2.0e-2, 2.5e-3
NX, NY, NZ = 6, 6, 1
RHO, P, U_INF = 1.225, 101325.0, 30.0
GAMMA = 1.4
BC = {
    "wall_bottom": {"type": "SYMMETRY"}, "wall_top": {"type": "SYMMETRY"},
    "z_min": {"type": "SYMMETRY"}, "z_max": {"type": "SYMMETRY"},
    "x_min": {"type": "OUTLET", "p_outlet": P},
    "x_max": {"type": "OUTLET", "p_outlet": P},
}


class TestRamp:
    def test_endpoints_midpoint_and_monotone(self):
        r = np.linspace(0.0, 0.2, 4001)
        w = entropy_sensor_ramp(r)
        assert np.all(w[r <= ENTROPY_SENSOR_LOW] == 0.0)
        assert np.all(w[r >= ENTROPY_SENSOR_FULL] == 1.0)
        mid = 0.5 * (ENTROPY_SENSOR_LOW + ENTROPY_SENSOR_FULL)
        assert abs(entropy_sensor_ramp(np.array([mid]))[0] - 0.5) < 1e-15
        assert np.all(np.diff(w) >= 0.0)

    def test_derivative_vanishes_at_both_ends(self):
        d = 1e-7
        for r0 in (ENTROPY_SENSOR_LOW, ENTROPY_SENSOR_FULL):
            for side in (-1.0, 1.0):
                slope = abs(entropy_sensor_ramp(np.array([r0 + side * d]))[0]
                            - entropy_sensor_ramp(np.array([r0]))[0]) / d
                assert slope < 1e-3, f"r={r0} 侧 {side} 斜率 {slope:.3e}（应为 O(d)）"

    def test_rejects_bad_interval(self):
        with pytest.raises(ValueError):
            entropy_sensor_ramp(np.array([0.0]), 0.1, 0.1)


def _build(order=1, alpha=1.0):
    mesh = build_channel_mesh_prism(order, nx=NX, ny=NY, nz=NZ, Lx=LX, H=H, Lz=LZ)
    solver = FRSolver(
        mesh=mesh, order=order, turb_model_name="NONE",
        time_scheme=TimeIntegrationScheme.SSP_RK3,
        rho_inf=RHO, vel_inf=U_INF, p_inf=P, mu_molecular=1.8e-5,
        bc_overrides=BC, n_threads=1,
        artificial_viscosity_enabled=True, artificial_viscosity_alpha=alpha,
    )
    solver.order_continuation_enabled = False
    solver.boundary_ghost_provider = build_face_exact_ghost_provider(mesh, LX, H, LZ, BC)
    return solver, mesh


def _xyz(solver):
    n, ns = solver.state.U.shape[:2]
    return np.asarray(solver.mesh.sps_coords).reshape(n, ns, 3)


def _set(solver, rho, p, u=U_INF):
    U = solver.state.U
    U[..., 0] = rho
    U[..., 1] = rho * u
    U[..., 2] = 0.0
    U[..., 3] = 0.0
    U[..., 4] = p / (GAMMA - 1.0) + 0.5 * rho * u ** 2
    solver.state._update_primitives()


def _eps(solver):
    return solver.compute_artificial_diffusivity_field()


class TestDoesNotTriggerOnPhysicalFields:
    def test_uniform_flow(self):
        solver, _ = _build()
        _set(solver, RHO, P)
        assert np.all(_eps(solver) == 0.0)

    def test_isentropic_variation_of_large_amplitude(self):
        """p 与 rho 变化 30%，但 p ~ rho^gamma，熵处处相同。"""
        solver, _ = _build()
        x = _xyz(solver)[..., 0]
        rho = RHO * (1.0 + 0.3 * np.sin(2.0 * np.pi * x / LX))
        _set(solver, rho, P * (rho / RHO) ** GAMMA)
        assert np.all(_eps(solver) == 0.0)

    def test_entropy_gradient_normal_to_the_flow(self):
        """熵（温度）沿 y 强烈变化、流动沿 x：边界层的物理熵梯度形态，u.grad s = 0。"""
        solver, _ = _build()
        y = _xyz(solver)[..., 1]
        T = 288.0 * (1.0 + 0.5 * y / H)
        _set(solver, P / (287.0 * T), P)
        assert np.all(_eps(solver) == 0.0)


def _spike(solver, amp):
    """一个内部单元里沿流向的等压温度尖峰（压力不变、密度下降）。"""
    xyz = _xyz(solver)
    xc = xyz[..., 0].mean(axis=1)
    yc = xyz[..., 1].mean(axis=1)
    c = int(np.argmin((xc - 0.5 * LX) ** 2 + (yc - 0.5 * H) ** 2))
    T = np.full(solver.state.U.shape[:2], 288.0)
    x = xyz[c, :, 0]
    T[c] = 288.0 * (1.0 + amp * (x - x.min()) / max(np.ptp(x), 1e-300))
    _set(solver, P / (287.0 * T), P)
    return c


def test_isobaric_entropy_spike_triggers_with_exact_coefficient():
    from autoflowcfd.fr.native_padding import real_sps_per_cell

    alpha = 0.7
    solver, mesh = _build(alpha=alpha)
    c = _spike(solver, 0.5)
    eps = _eps(solver)
    U = solver.state.U
    cip = np.arange(mesh.n_cells) < mesh.n_prism_cells
    r = compute_entropy_sensor(U, 1, mesh.cell_volumes, cip,
                               lambda f: compute_scalar_gradient(f, solver.ops, mesh))
    assert r[c] > ENTROPY_SENSOR_FULL and eps[c, 0] > 0.0
    others = np.ones(mesh.n_cells, dtype=bool)
    others[c] = False
    assert np.all(eps[others] == 0.0), "尖峰以外的单元不应触发（判据是单元局部的）"

    n_real = real_sps_per_cell(1)[0]
    spd_m = (np.linalg.norm(U[c, :n_real, 1:4], axis=1) / U[c, :n_real, 0]).mean()
    expect = alpha * spd_m * np.cbrt(mesh.cell_volumes[c]) / 1
    np.testing.assert_allclose(eps[c], expect, rtol=1e-14)


def test_coefficient_is_continuous_in_spike_amplitude():
    solver, _ = _build()
    vals = []
    for a in np.linspace(0.0, 0.3, 121):
        c = _spike(solver, a)
        vals.append(_eps(solver)[c, 0])
    vals = np.array(vals)
    assert vals[0] == 0.0 and vals[-1] > 0.0
    first = int(np.flatnonzero(vals > 0.0)[0])
    assert vals[first] < 0.05 * vals.max(), "门限处应从 0 连续起步（C¹ 斜坡）"
    assert np.max(np.abs(np.diff(vals))) < 0.1 * vals.max()


def test_order_zero_returns_zero():
    U = np.zeros((4, 1, 5))
    U[..., 0] = RHO
    U[..., 4] = P / (GAMMA - 1.0)
    eps = compute_artificial_diffusivity(U, 0, np.full(4, 1e-6), np.ones(4, dtype=bool),
                                       lambda f: np.zeros(f.shape + (3,)))
    assert eps.shape == (4, 1) and np.all(eps == 0.0)


def test_laplacian_conserves_every_variable_and_is_dissipative():
    """DG 质量矩阵下的全域积分变化 1^T M dU_k/dt 对 k=0..4 都为零（齐次 Neumann
    边界），且对被扰动的守恒量是耗散的（<dU, M L dU> < 0）。"""
    from autoflowcfd.fr.native_padding import real_sps_per_cell
    from tests.unit.test_viscous_heat_conduction_sign import _dg_mass_matrix

    solver, mesh = _build()
    c = _spike(solver, 0.5)
    nu = _eps(solver)
    assert nu[c, 0] > 0.0
    R = solver._artificial_diffusion_residual(nu)
    n_real = real_sps_per_cell(1)[0]
    M = _dg_mass_matrix(mesh, 1, 1)
    for k in range(5):
        r = R[:, :n_real, k].ravel()
        mr = M @ r
        total, scale = float(mr.sum()), float(np.abs(mr).sum())
        assert scale > 0.0 or k in (2, 3), f"守恒量 {k} 的扩散项恒为零，用例失去区分力"
        assert abs(total) <= 1e-10 * max(scale, 1e-300), (
            f"守恒量 {k} 的全域积分变化 {total:.3e}（相对 {abs(total) / max(scale, 1e-300):.2e}）")
    d = (solver.state.U[:, :n_real, 0] - RHO).ravel()
    assert float(d @ (M @ R[:, :n_real, 0].ravel())) < 0.0, "质量扩散对扰动不是耗散的"


class _NumpyAsCupy:
    """numpy 充当 CuPy（与 test_gpu_scalar_transport 同一种替身）：跑的是 GPU 生产函数本身。"""

    def __getattr__(self, name):
        return getattr(np, name)

    def scatter_add(self, a, indices, b):
        np.add.at(a, indices, b)

    def asnumpy(self, x):
        return np.asarray(x)


def test_single_gpu_path_matches_cpu(monkeypatch):
    """单 GPU 的系数场与人工扩散残差，与 CPU 同一算例逐点一致（numpy 替身跑 GPU 生产代码）。"""
    from types import SimpleNamespace

    import autoflowcfd.core.gpu.residual.gpu_gradients as gg
    import autoflowcfd.core.gpu.residual.gpu_volume_contract as gvc
    import autoflowcfd.core.gpu.solver.gpu_solver.residual as gres
    import autoflowcfd.core.gpu.solver.gpu_solver.timestep as gts
    import autoflowcfd.core.gpu.turbulence.gpu_scalar_transport as gst
    from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
    from tests.unit._gpu_cupy_shim import patch_module_get_cupy
    from tests.unit._gpu_standin_helpers import complete_gpu_standin

    patch_module_get_cupy(monkeypatch, [gg, gvc, gres, gts, gst], _NumpyAsCupy())
    solver, mesh = _build(alpha=0.7)
    _spike(solver, 0.5)
    U = solver.state.U.copy()
    nu_cpu = solver.compute_artificial_diffusivity_field()
    res_cpu = solver._artificial_diffusion_residual(nu_cpu)
    assert np.abs(nu_cpu).max() > 0.0

    n, ns = U.shape[:2]
    det = mesh.jacobians["det_jacs"].reshape(n, ns)
    inv = mesh.jacobians["inv_jacs"].reshape(n, ns, 3, 3)
    mesh_data = {"det_jacs": det, "inv_jacs": inv, "adj_j": det[..., None, None] * inv,
                 "n_cells": n, "n_prism": mesh.n_prism_cells}
    ops_data = {"D_3d_prism": solver.ops.D_3d_prism, "D_3d_tet": solver.ops.D_3d_tet}
    complete_gpu_standin(mesh, solver.ops, ops_data, mesh_data)
    stub = SimpleNamespace(
        mesh=mesh, mesh_data=mesh_data, ops_data=ops_data, current_order=1,
        flat_face_gpu=get_flat_face_geometry(mesh, solver.ops), U_gpu=U,
        artificial_viscosity_enabled=True, artificial_viscosity_alpha=0.7)
    stub._cell_volumes_gpu = lambda: gts._GPUSolverTimeStepMixin._cell_volumes_gpu(stub)
    nu_gpu = gres._GPUSolverResidualMixin.compute_artificial_diffusivity_field_gpu(stub, U)
    np.testing.assert_allclose(nu_gpu, nu_cpu, rtol=1e-12, atol=0.0)
    res_gpu = gres._GPUSolverResidualMixin._artificial_diffusion_residual_gpu(stub, U, nu_gpu)
    scale = np.abs(res_cpu).max(axis=(0, 1))
    for k in range(5):
        np.testing.assert_allclose(res_gpu[..., k], res_cpu[..., k], rtol=0.0,
                                   atol=1e-11 * max(scale[k], 1e-300), err_msg=f"分量 {k}")


class _FakeHaloExchange:
    """用全局真值切出的扩展状态代替 MPI halo 交换（本机无 mpi4py），其余生产逻辑
    （compact 重排、适配器、标量扩散装配、换回原生排列）照常执行。"""

    def __init__(self, U_extended):
        self._U_extended = U_extended

    def exchange(self, U_local):
        return self._U_extended


@pytest.mark.parametrize("rank", [0, 1])
def test_cpu_mpi_path_matches_single_machine(rank):
    from autoflowcfd.core.mpi.distributed_artificial_viscosity import (
        distributed_artificial_diffusion_dudt, distributed_artificial_diffusivity, local_from_compact,
    )
    from autoflowcfd.core.mpi.distributed_flat_face import build_distributed_flat_face
    from autoflowcfd.core.mpi.partition import build_distributed_partition

    solver, mesh = _build(alpha=0.7)
    _spike(solver, 0.5)
    U = solver.state.U.copy()
    nu_ref = solver.compute_artificial_diffusivity_field()
    res_ref = solver._artificial_diffusion_residual(nu_ref)
    assert np.abs(nu_ref).max() > 0.0

    cell_partition = (np.arange(mesh.n_cells) % 2).astype(np.int32)
    part = build_distributed_partition(mesh.face_connectivity, cell_partition, rank=rank, n_ranks=2)
    dist_fc = build_distributed_flat_face(mesh, solver.ops, part, cell_partition=cell_partition)
    native_ids = np.concatenate([part.local_cells, part.halo_cells]).astype(np.int64)
    halo = _FakeHaloExchange(U[native_ids])
    U_local = U[part.local_cells]

    nu_c = distributed_artificial_diffusivity(U_local, part, halo, dist_fc, mesh, solver.ops,
                                              order=1, alpha_av=0.7)
    nu_local = local_from_compact(nu_c, dist_fc, part.n_local_cells)
    np.testing.assert_allclose(nu_local, nu_ref[part.local_cells], rtol=1e-12, atol=0.0)
    # halo 单元的系数（用 halo 数据算）与拥有方逐位一致：单元局部判据
    np.testing.assert_allclose(nu_c[dist_fc.inv_perm][part.n_local_cells:],
                               nu_ref[part.halo_cells], rtol=1e-12, atol=0.0)

    dudt = distributed_artificial_diffusion_dudt(U_local, nu_c, part, halo, dist_fc, mesh, solver.ops)
    ref = res_ref[part.local_cells]
    scale = np.abs(res_ref).max(axis=(0, 1))
    for k in range(5):
        np.testing.assert_allclose(dudt[..., k], ref[..., k], rtol=0.0,
                                   atol=1e-11 * max(scale[k], 1e-300), err_msg=f"分量 {k}")


@pytest.mark.parametrize("rank", [0, 1])
def test_multi_gpu_path_matches_single_machine(monkeypatch, rank):
    """多 GPU：compact 空间的系数与人工扩散（numpy 替身跑生产方法），local 段与单机一致。"""
    from types import SimpleNamespace

    import autoflowcfd.core.gpu.distributed.gpu_distributed.residual as mres
    import autoflowcfd.core.gpu.residual.gpu_gradients as gg
    import autoflowcfd.core.gpu.residual.gpu_volume_contract as gvc
    import autoflowcfd.core.gpu.turbulence.gpu_scalar_transport as gst
    from autoflowcfd.core.mpi.distributed_compute import DistributedMeshAdapter
    from autoflowcfd.core.mpi.distributed_flat_face import build_distributed_flat_face
    from autoflowcfd.core.mpi.partition import build_distributed_partition
    from tests.unit._gpu_cupy_shim import patch_module_get_cupy
    from tests.unit._gpu_standin_helpers import complete_gpu_standin

    patch_module_get_cupy(monkeypatch, [gg, gvc, gst, mres], _NumpyAsCupy())
    solver, mesh = _build(alpha=0.7)
    _spike(solver, 0.5)
    U = solver.state.U.copy()
    nu_ref = solver.compute_artificial_diffusivity_field()
    res_ref = solver._artificial_diffusion_residual(nu_ref)

    cell_partition = (np.arange(mesh.n_cells) % 2).astype(np.int32)
    part = build_distributed_partition(mesh.face_connectivity, cell_partition, rank=rank, n_ranks=2)
    dist_fc = build_distributed_flat_face(mesh, solver.ops, part, cell_partition=cell_partition)
    adapter = DistributedMeshAdapter(part, dist_fc, mesh, solver.ops)
    native_ids = np.concatenate([part.local_cells, part.halo_cells]).astype(np.int64)
    n_c, ns = len(native_ids), U.shape[1]
    det = np.asarray(adapter.jacobians["det_jacs"]).reshape(n_c, ns)
    inv = np.asarray(adapter.jacobians["inv_jacs"]).reshape(n_c, ns, 3, 3)
    md = {"det_jacs": det, "inv_jacs": inv, "adj_j": det[..., None, None] * inv,
          "cell_volumes": np.asarray(adapter.cell_volumes), "n_cells": n_c,
          "n_prism": int(dist_fc.base_flat.n_prism),
          "D_3d_prism": solver.ops.D_3d_prism, "D_3d_tet": solver.ops.D_3d_tet}
    complete_gpu_standin(mesh, solver.ops, md, md, compact_ids=dist_fc.compact_global_ids)
    stub = SimpleNamespace(
        U_gpu=U[part.local_cells], gpu_halo=_FakeHaloExchange(U[native_ids]),
        _perm_gpu=np.asarray(dist_fc.perm), dist_flat_face=dist_fc, mesh_data=md,
        flat_face_gpu=dist_fc.base_flat, current_order=1,
        artificial_viscosity_enabled=True, artificial_viscosity_alpha=0.7)
    stub._permute_to_compact = lambda a: a[stub._perm_gpu]
    nu_c = mres._MultiGPUResidualMixin.compute_artificial_diffusivity_compact_gpu(stub)
    nu_local = nu_c[dist_fc.inv_perm][:part.n_local_cells]
    np.testing.assert_allclose(nu_local, nu_ref[part.local_cells], rtol=1e-12, atol=0.0)

    U_compact = U[native_ids][dist_fc.perm]
    res_c = mres._MultiGPUResidualMixin._artificial_diffusion_compact_gpu(stub, U_compact, nu_c)
    got = res_c[dist_fc.inv_perm][:part.n_local_cells]
    ref = res_ref[part.local_cells]
    scale = np.abs(res_ref).max(axis=(0, 1))
    for k in range(5):
        np.testing.assert_allclose(got[..., k], ref[..., k], rtol=0.0,
                                   atol=1e-11 * max(scale[k], 1e-300), err_msg=f"分量 {k}")
