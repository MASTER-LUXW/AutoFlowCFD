# -*- coding: utf-8 -*-
"""多 GPU 分布式的 SA-neg（`gpu/distributed/gpu_distributed_init/turb_source.py`，numpy 替身）与 CPU 分布式
（`core/mpi/distributed_turbulence.py`）在同一份非均匀状态上的显式湍流段逐位一致。

两者是同一份算法（`core/turbulence/sa/rates.py`）注入不同后端的标量输运原语，紧凑视图都由
`TransportedTurbulence.like` 构造。状态里放了负的 `nu_tilde`（SA-neg 负支）与精确为 0 的壁距（壁面解点的
强 Dirichlet），两侧都要处理到。方法与 `test_gpu_distributed_turbulence.py` 的 SST 对照相同。
"""

import types
import numpy as np
import pytest

import autoflowcfd.core.gpu.distributed.gpu_distributed as gd_mod
import autoflowcfd.core.gpu.distributed.gpu_distributed_init as gdi_mod
import autoflowcfd.core.gpu.gpu_modal_filter as gpu_modal_filter_mod
import autoflowcfd.core.gpu.residual.gpu_flux as gpu_flux_mod
import autoflowcfd.core.gpu.residual.gpu_gradients as gpu_gradients_mod
import autoflowcfd.core.gpu.residual.gpu_volume_contract as gpu_volume_contract_mod
import autoflowcfd.core.gpu.turbulence.gpu_scalar_transport as gst_mod
from autoflowcfd.core.gpu.distributed.gpu_distributed_init import _GPUDistributedInitMixin
from autoflowcfd.core.mpi.distributed_flat_face import build_distributed_flat_face
from autoflowcfd.core.mpi.distributed_turbulence import distributed_compute_turbulence_source_and_viscosity
from autoflowcfd.core.mpi.partition import build_distributed_partition
from autoflowcfd.core.turbulence.sa import SAModel
from autoflowcfd.fr.operators import generate_fr_operators

from tests.unit._fake_halo import ShapeKeyedFakeHalo
from tests.unit._gpu_cupy_shim import patch_module_get_cupy
from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh
from tests.unit.test_gpu_distributed_fully_distributed import gpu_shim  # noqa: F401  （夹具）
from tests.unit._gpu_distributed_turbulence_common import (
    _bind_turb_source,
    _nonuniform_state,
    _prepare_compact_mesh_data,
)
from tests.unit._numpy_as_cupy import NumpyAsCupy

MU, RHO = 1.8e-5, 1.225


@pytest.fixture(autouse=True)
def _patch_get_cupy(monkeypatch):
    patch_module_get_cupy(monkeypatch, [gdi_mod, gd_mod, gpu_gradients_mod, gpu_volume_contract_mod, gpu_flux_mod,
                                        gst_mod, gpu_modal_filter_mod], NumpyAsCupy())


@pytest.mark.parametrize("rank", [0, 1])
def test_gpu_distributed_sa_matches_cpu_distributed_sa(rank):
    order = 1
    mesh = _build_synthetic_mixed_mesh(order)
    ops = generate_fr_operators(order)
    n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
    dt = 1e-5
    rng = np.random.default_rng(4242)
    U, _, _ = _nonuniform_state(mesh, rng)
    nu_ref = MU / RHO
    nu_tilde = nu_ref * (3.0 + rng.uniform(-1.0, 6.0, size=(n_cells, n_sps)))
    nu_tilde[rng.random(nu_tilde.shape) < 0.15] *= -0.4              # SA-neg 负支
    d_wall = rng.uniform(0.01, 0.08, size=(n_cells, n_sps))
    d_wall[:, 0] = 0.0                                                # 壁面解点

    cell_partition = np.array([0, 1, 0, 1], dtype=np.int32)
    partition = build_distributed_partition(mesh.face_connectivity, cell_partition, rank=rank, n_ranks=2)
    dist_fc = build_distributed_flat_face(mesh, ops, partition, cell_partition=cell_partition)
    compact_ids = dist_fc.compact_global_ids
    n_local = partition.n_local_cells
    native_ids = (np.concatenate([partition.local_cells, partition.halo_cells]) if partition.n_halo > 0
                  else partition.local_cells)
    d_compact = d_wall[compact_ids]
    d_local = d_wall[partition.local_cells]

    def model(xp):
        m = SAModel(n_local, n_sps, nu_ref, viscosity_ratio=3.0, xp=xp)
        m.nu_tilde_field = nu_tilde[partition.local_cells].copy()
        m.apply_wall_distance(d_local, nu_ref)
        return m

    # ---- CPU 分布式参照 ----
    turb_cpu = model(np)
    halo_cpu = ShapeKeyedFakeHalo(U[native_ids][..., :5], np.where(d_wall == 0.0, 0.0, nu_tilde)[native_ids][..., None])
    mu_t_cpu = distributed_compute_turbulence_source_and_viscosity(
        U[partition.local_cells], partition, halo_cpu, dist_fc, mesh, ops, turb_cpu, MU, d_compact,
        np.full((n_local, n_sps), dt), ramp_owner=types.SimpleNamespace(_turb_ramp_step=10, _turb_production_ramp_steps=50, _turb_production_ramp_complete=False),
        turb_model_name="SA")

    # ---- 多 GPU（numpy 替身）----
    mesh_data = _prepare_compact_mesh_data(mesh, ops, compact_ids)
    mesh_data['n_prism'] = dist_fc.base_flat.n_prism
    mesh_data['cell_volumes'] = mesh.cell_volumes[compact_ids]
    turb_gpu = model(NumpyAsCupy())
    halo_gpu = ShapeKeyedFakeHalo(U[native_ids], np.where(d_wall == 0.0, 0.0, nu_tilde)[native_ids][..., None])
    n_faces = dist_fc.base_flat.n_faces
    stub = types.SimpleNamespace(
        rank=rank, device_id=0, mu_molecular=MU, boundary_ghost_provider=None, partition=partition, mesh=mesh,
        dist_flat_face=dist_fc, mesh_data=mesh_data, ops_data=mesh_data, ops=ops, flat_face_gpu=dist_fc.base_flat,
        turb_model_gpu=turb_gpu, turb_model_name="SA", gpu_halo=halo_gpu, _perm_gpu=dist_fc.perm,
        _inv_perm_gpu=dist_fc.inv_perm, n_compact=len(compact_ids), wall_distance_gpu=d_compact,
        _wall_mask_k_gpu=np.zeros(n_faces, dtype=bool), _open_mask_gpu=np.zeros(n_faces, dtype=bool),
        U_gpu=U[partition.local_cells], ddes_model_gpu=None, sgs_model_gpu=None,
        iddes_h_max_compact=None, iddes_h_wn_compact=None, _turb_ramp_step=10, _turb_production_ramp_steps=50,
        _turb_production_ramp_complete=False)
    stub._permute_to_compact = lambda arr: arr[stub._perm_gpu]
    stub._unpermute_from_compact = lambda arr: arr[stub._inv_perm_gpu]
    _bind_turb_source(stub)
    mu_t_gpu = _GPUDistributedInitMixin._compute_turbulence_source_distributed(stub, dt)

    assert turb_gpu.production_factor == pytest.approx(10 / 50)
    np.testing.assert_allclose(turb_gpu.nu_tilde_field, turb_cpu.nu_tilde_field, rtol=1e-10,
                               atol=1e-12 * nu_ref)
    np.testing.assert_allclose(turb_gpu.nu_t, turb_cpu.nu_t, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(mu_t_gpu[dist_fc.inv_perm][:n_local], mu_t_cpu[dist_fc.inv_perm][:n_local],
                               rtol=1e-10, atol=1e-14)
    # 壁面解点两侧都保持 0；负值解点推进后仍可为负（不裁剪）
    wall = d_local == 0.0
    assert np.all(turb_cpu.nu_tilde_field[wall] == 0.0) and np.all(turb_gpu.nu_tilde_field[wall] == 0.0)
    assert np.any(turb_cpu.nu_tilde_field < 0.0)


def test_fully_distributed_build_wires_sa(monkeypatch, gpu_shim):
    """多 GPU 完全分布式加载构造 SA-neg：模型来自与单机同一个工厂（来流值）、root 算好的壁距原样上传、
    本 rank local 壁面解点经壁距钩子置零。构造期依赖 CUDA 的数组管理/面几何/halo 用该文件同一组替身。"""
    import autoflowcfd.core.gpu.turbulence.gpu_scalar_transport as scalar_transport_mod
    from autoflowcfd.core.fr_solver.boundary import build_boundary_ghost_provider
    from autoflowcfd.core.gpu.distributed.gpu_distributed import MultiGPUDistributedSolver
    from autoflowcfd.core.gpu.distributed.gpu_distributed_fully_distributed import (
        build_multi_gpu_solver_from_fully_distributed_package,
    )
    from autoflowcfd.core.mpi.distributed_mesh_loader import build_fully_distributed_rank_package
    from autoflowcfd.core.mpi.distributed_turbulence import local_part
    from autoflowcfd.core.turbulence.sa.constants import chi_for_viscosity_ratio
    from tests.unit._patch_pkg import patch_pkg_attr
    from tests.unit._wall_source import synthetic_wall_source

    patch_pkg_attr(monkeypatch, scalar_transport_mod, "compute_turbulence_face_masks_gpu",
                   lambda stub, provider: (np.zeros(stub.face_connectivity.n_faces, dtype=bool),
                                           np.zeros(stub.face_connectivity.n_faces, dtype=bool)))

    mesh = _build_synthetic_mixed_mesh(1)
    ops = generate_fr_operators(1)
    freestream = {"rho_inf": RHO, "vel_inf": 33.33, "p_inf": 101325.0}
    stub = types.SimpleNamespace(mesh=mesh, freestream=freestream, turb_model_name="SA", wmles_model=None)
    package = build_fully_distributed_rank_package(
        mesh, ops, mesh.face_connectivity, np.zeros(mesh.n_cells, dtype=np.int32), 0, 1,
        build_boundary_ghost_provider(stub, bc_overrides={}), freestream, mu_molecular=MU, mach_ref=0.2,
        order=mesh.order, enable_viscous=True, turb_model_name="SA", viscosity_ratio=3.0,
        wall_distance_source=synthetic_wall_source(mesh))
    solver = build_multi_gpu_solver_from_fully_distributed_package(
        MultiGPUDistributedSolver, package, n_ranks=1, device_id=0, rank=0, root_context=None)

    m = solver.turb_model_gpu
    assert isinstance(m, SAModel)
    assert m.nu_tilde_inf == pytest.approx(chi_for_viscosity_ratio(3.0) * MU / RHO, rel=1e-14)
    np.testing.assert_allclose(np.asarray(solver.wall_distance_gpu), package['wall_distance_compact'])
    wall = np.asarray(local_part(solver, solver.wall_distance_gpu)) == 0.0
    assert wall.any() and np.all(np.asarray(m.nu_tilde_field)[wall] == 0.0)
