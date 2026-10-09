"""AutoFlowCFD V2.0 - 多GPU分布式 SST 湍流模型端到端验证（2026-09-02）。

背景：`MultiGPUDistributedSolver`（`--multi-gpu`）此前把 `turbulence_
model` 收紧为只接受 'none'——`_compute_turbulence_source_distributed`
虽然有函数体，但：(1) 引用了本类从未定义过的 `self.Q_gpu`/
`self.ops_data`；(2) 直接用 `self.U_gpu`（native 排列，只有 n_local，
没有 halo）算梯度，没有做 halo 交换+compact 索引空间重排；(3) 完全
没有调用 k/omega 输运（对流+扩散）；(4) `_init_wall_distance_
distributed` 对全局 sps_coords 整体 reshape 再切 (n_local,n_sps)，
n_ranks>1 时形状不匹配。本次真正实现，与 CPU MPI 路径
（`core/mpi/distributed_turbulence.py`）同一个设计：复用单机
`compute_source_terms_gpu`/`compute_turbulence_transport_residual_gpu`/
`update_fields` 的数值逻辑，只把 halo 交换+compact 索引空间重排
接上。

验证方式：本机没有真实 CuPy/CUDA 设备，用把 `get_cupy()` 替换成"返回
numpy 模块（外加 cuda.Device 空上下文管理器）"的 monkeypatch，直接跑
*生产函数本身*（不是重新实现一份），对同一个真实混合棱柱+四面体网格，
与 CPU 单机路径 `compute_turbulence_source`（间接，通过对照
`SSTModelFR.compute_source_terms`/`compute_turbulence_transport_
residual` 的同一份底层公式）——但更直接、更贴近本项目已建立方法论
的判据是：把这条新路径的输出与 CPU MPI 分布式路径（`distributed_
turbulence.py::distributed_compute_turbulence_source_and_viscosity`，
本次会话早些时候已经决定性验证过与单机逐位一致）在同一份非均匀状态
下对照——两者数值上应该完全一致（都是同一套 SST 数值算法，只是张量
库从 numpy 换成"伪装成 numpy 的 cupy"，不应该有任何数值差异）。
"""

import types
import numpy as np
import pytest

from autoflowcfd.fr.operators import generate_fr_operators
from autoflowcfd.core.mpi.partition import build_distributed_partition
from autoflowcfd.core.mpi.distributed_flat_face import build_distributed_flat_face
from autoflowcfd.core.mpi.distributed_turbulence import (
    distributed_compute_turbulence_source_and_viscosity,
)
from autoflowcfd.core.turbulence.sst import SSTModelFR
from autoflowcfd.core.turbulence.des import compute_h_max_and_h_wn
from autoflowcfd.core.gpu.distributed.gpu_distributed_init import _GPUDistributedInitMixin
from autoflowcfd.core.gpu.turbulence.gpu_turbulence_sst import GPUTurbulenceSST

from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh
from tests.unit._fake_halo import ShapeKeyedFakeHalo
from tests.unit._gpu_distributed_turbulence_common import (
    _bind_turb_source,
    _make_ddes_cpu,
    _make_ddes_gpu,
    _nonuniform_state,
    _prepare_compact_mesh_data,
)
from tests.unit._gpu_distributed_turbulence_common import _patch_get_cupy  # noqa: F401  autouse 夹具：进入本模块命名空间才生效


@pytest.mark.parametrize("turb_model_name", ["SST", "DDES", "IDDES"])
@pytest.mark.parametrize("rank", [0, 1])
def test_gpu_distributed_sst_matches_cpu_distributed_sst(rank, turb_model_name):
    """决定性判据：GPU 分布式 SST/DDES/IDDES（numpy 替身）与 CPU 分布式
    同名模型（本会话早些时候已经与单机逐位验证过，含 des_length_scale
    halo 交换修复）在同一份非均匀状态下必须逐位一致——两者是同一套
    数值算法的两个后端实现，不应该有任何差异。"""
    order = 1
    mesh = _build_synthetic_mixed_mesh(order)
    ops = generate_fr_operators(order)
    n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
    mu = 1.8e-5
    dt = 1e-5
    rng = np.random.default_rng(4242)

    U, k_field, omega_field = _nonuniform_state(mesh, rng)
    d_wall = np.full((n_cells, n_sps), 0.05)

    h_max_global = h_wn_global = None
    if turb_model_name == "IDDES":
        h_max_global, h_wn_global = compute_h_max_and_h_wn(mesh)

    cell_partition = np.array([0, 1, 0, 1], dtype=np.int32)
    fc = mesh.face_connectivity
    partition = build_distributed_partition(fc, cell_partition, rank=rank, n_ranks=2)
    dist_fc = build_distributed_flat_face(mesh, ops, partition, cell_partition=cell_partition)
    compact_global_ids = dist_fc.compact_global_ids
    n_compact = len(compact_global_ids)
    n_local = partition.n_local_cells
    n_halo = partition.n_halo
    native_ids = (
        np.concatenate([partition.local_cells, partition.halo_cells])
        if n_halo > 0 else partition.local_cells
    )

    iddes_h_max_compact = iddes_h_wn_compact = None
    if turb_model_name == "IDDES":
        iddes_h_max_compact = h_max_global[compact_global_ids]
        iddes_h_wn_compact = h_wn_global[compact_global_ids]

    # ---- CPU 分布式参照（本会话早些时候已验证与单机逐位一致）----
    turb_cpu = SSTModelFR(n_local, n_sps)
    turb_cpu.k_field = k_field[partition.local_cells].copy()
    turb_cpu.omega_field = omega_field[partition.local_cells].copy()
    U_local = U[partition.local_cells]
    k_omega_extended = np.stack([k_field[native_ids], omega_field[native_ids]], axis=-1)
    fake_halo_cpu = ShapeKeyedFakeHalo(U[native_ids][..., :5], k_omega_extended)
    d_wall_compact = d_wall[compact_global_ids]
    dt_local_local = np.full((n_local, n_sps), dt)

    mu_t_compact_cpu = distributed_compute_turbulence_source_and_viscosity(
        U_local, partition, fake_halo_cpu, dist_fc, mesh, ops,
        turb_cpu, mu, d_wall_compact, dt_local_local,
        # 产生项渐变进行中（第 10/50 步）：两侧必须按同一个计数器渐变（多 GPU
        # 2026-09-25 以前从不推进渐变，production_factor 恒为 1）
        ramp_owner=types.SimpleNamespace(_turb_ramp_step=10, _turb_production_ramp_steps=50, _turb_production_ramp_complete=False),
        turb_model_name=turb_model_name, ddes_model=_make_ddes_cpu(turb_model_name),
        iddes_h_max_compact=iddes_h_max_compact, iddes_h_wn_compact=iddes_h_wn_compact,
    )

    # ---- GPU 分布式（numpy 替身）----
    mesh_data = _prepare_compact_mesh_data(mesh, ops, compact_global_ids)
    mesh_data['n_prism'] = dist_fc.base_flat.n_prism
    mesh_data['cell_volumes'] = mesh.cell_volumes[compact_global_ids]

    turb_gpu = GPUTurbulenceSST(n_local, n_sps, device_id=0)
    turb_gpu.k_field = k_field[partition.local_cells].copy()
    turb_gpu.omega_field = omega_field[partition.local_cells].copy()

    fake_halo_gpu = ShapeKeyedFakeHalo(U[native_ids], k_omega_extended)  # 平均流 5 变量 + k/omega

    stub = types.SimpleNamespace(
        rank=rank, device_id=0, mu_molecular=mu, boundary_ghost_provider=None,
        partition=partition, mesh=mesh, dist_flat_face=dist_fc,
        mesh_data=mesh_data, ops_data=mesh_data, ops=ops,
        flat_face_gpu=dist_fc.base_flat,
        turb_model_gpu=turb_gpu, turb_model_name=turb_model_name, gpu_halo=fake_halo_gpu,
        _perm_gpu=dist_fc.perm, _inv_perm_gpu=dist_fc.inv_perm, n_compact=n_compact,
        wall_distance_gpu=d_wall_compact,
        _wall_mask_k_gpu=np.zeros(dist_fc.base_flat.n_faces, dtype=bool),
        _open_mask_gpu=np.zeros(dist_fc.base_flat.n_faces, dtype=bool),
        U_gpu=U_local,
        ddes_model_gpu=_make_ddes_gpu(turb_model_name),
        iddes_h_max_compact=iddes_h_max_compact, iddes_h_wn_compact=iddes_h_wn_compact,
        _turb_ramp_step=10, _turb_production_ramp_steps=50, _turb_production_ramp_complete=False,
    )
    stub._permute_to_compact = lambda arr: arr[stub._perm_gpu]
    stub._unpermute_from_compact = lambda arr: arr[stub._inv_perm_gpu]
    _bind_turb_source(stub)

    mu_t_compact_gpu = _GPUDistributedInitMixin._compute_turbulence_source_distributed(stub, dt)
    assert turb_gpu.production_factor == pytest.approx(10 / 50)

    np.testing.assert_allclose(turb_gpu.k_field, turb_cpu.k_field, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(turb_gpu.omega_field, turb_cpu.omega_field, rtol=1e-10, atol=1e-8)
    np.testing.assert_allclose(turb_gpu.nu_t, turb_cpu.nu_t, rtol=1e-10, atol=1e-14)

    mu_t_native_gpu = mu_t_compact_gpu[dist_fc.inv_perm]
    mu_t_native_cpu = mu_t_compact_cpu[dist_fc.inv_perm]
    np.testing.assert_allclose(
        mu_t_native_gpu[:n_local], mu_t_native_cpu[:n_local], rtol=1e-10, atol=1e-14
    )


@pytest.mark.parametrize("turb_model_name", ["DDES", "IDDES"])
def test_gpu_distributed_ddes_two_consecutive_calls_does_not_crash(turb_model_name):
    """真实 bug 回归测试（2026-09-02，移植自 CPU 侧同一处修复——本次
    移植 GPU 分布式 DDES/IDDES 时把这个修复直接一并做进去了，但仍然
    需要一个决定性测试锁定它，不能只靠"移植时抄对了"这个假设）：
    `des_length_scale` 是跨步持久状态，只有 1 个分量，若被直接原样
    setattr 到 compact 大小的临时视图上而不做 halo 交换+重排，第二次
    调用会因 (n_local,n_sps) 与 (n_compact,n_sps) 形状不匹配崩溃——见
    `core/mpi/distributed_turbulence.py` 同名 bug 的完整记录。这里只
    验证"连续调用两次不崩溃、des_length_scale 形状始终正确"，不重复
    做两步数值一致性对照（CPU 侧已经用真实跨 rank 数据交换决定性
    验证过数值正确性，GPU 侧走的是完全同构的代码路径）。"""
    order = 1
    mesh = _build_synthetic_mixed_mesh(order)
    ops = generate_fr_operators(order)
    n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
    mu = 1.8e-5
    dt = 1e-3
    rng = np.random.default_rng(4242)

    U, k_field, omega_field = _nonuniform_state(mesh, rng)
    d_wall = np.full((n_cells, n_sps), 0.05)

    h_max_global = h_wn_global = None
    if turb_model_name == "IDDES":
        h_max_global, h_wn_global = compute_h_max_and_h_wn(mesh)

    rank = 1
    cell_partition = np.array([0, 1, 0, 1], dtype=np.int32)
    fc = mesh.face_connectivity
    partition = build_distributed_partition(fc, cell_partition, rank=rank, n_ranks=2)
    dist_fc = build_distributed_flat_face(mesh, ops, partition, cell_partition=cell_partition)
    compact_global_ids = dist_fc.compact_global_ids
    n_compact = len(compact_global_ids)
    n_local = partition.n_local_cells
    n_halo = partition.n_halo
    native_ids = (
        np.concatenate([partition.local_cells, partition.halo_cells])
        if n_halo > 0 else partition.local_cells
    )

    iddes_h_max_compact = iddes_h_wn_compact = None
    if turb_model_name == "IDDES":
        iddes_h_max_compact = h_max_global[compact_global_ids]
        iddes_h_wn_compact = h_wn_global[compact_global_ids]

    mesh_data = _prepare_compact_mesh_data(mesh, ops, compact_global_ids)
    mesh_data['n_prism'] = dist_fc.base_flat.n_prism
    mesh_data['cell_volumes'] = mesh.cell_volumes[compact_global_ids]

    turb_gpu = GPUTurbulenceSST(n_local, n_sps, device_id=0)
    turb_gpu.k_field = k_field[partition.local_cells].copy()
    turb_gpu.omega_field = omega_field[partition.local_cells].copy()

    d_wall_compact = d_wall[compact_global_ids]
    U_local = U[partition.local_cells]
    k_omega_extended = np.stack([k_field[native_ids], omega_field[native_ids]], axis=-1)
    fake_halo_gpu = ShapeKeyedFakeHalo(U[native_ids], k_omega_extended)

    stub = types.SimpleNamespace(
        rank=rank, device_id=0, mu_molecular=mu, boundary_ghost_provider=None,
        partition=partition, mesh=mesh, dist_flat_face=dist_fc,
        mesh_data=mesh_data, ops_data=mesh_data, ops=ops,
        flat_face_gpu=dist_fc.base_flat,
        turb_model_gpu=turb_gpu, turb_model_name=turb_model_name, gpu_halo=fake_halo_gpu,
        _turb_ramp_step=0, _turb_production_ramp_steps=0, _turb_production_ramp_complete=False,   # 不做产生项渐变
        _perm_gpu=dist_fc.perm, _inv_perm_gpu=dist_fc.inv_perm, n_compact=n_compact,
        wall_distance_gpu=d_wall_compact,
        _wall_mask_k_gpu=np.zeros(dist_fc.base_flat.n_faces, dtype=bool),
        _open_mask_gpu=np.zeros(dist_fc.base_flat.n_faces, dtype=bool),
        U_gpu=U_local,
        ddes_model_gpu=_make_ddes_gpu(turb_model_name),
        iddes_h_max_compact=iddes_h_max_compact, iddes_h_wn_compact=iddes_h_wn_compact,
    )
    stub._permute_to_compact = lambda arr: arr[stub._perm_gpu]
    stub._unpermute_from_compact = lambda arr: arr[stub._inv_perm_gpu]
    _bind_turb_source(stub)

    # call 1：des_length_scale 还是 None，第一次调用不会触发它的 halo 交换
    _GPUDistributedInitMixin._compute_turbulence_source_distributed(stub, dt)
    assert turb_gpu.des_length_scale is not None
    assert turb_gpu.des_length_scale.shape == (n_local, n_sps)

    # call 2：这次 des_length_scale 非 None，会真正触发 halo 交换分支——
    # 用一个"halo 部分填 0，local 部分是本 rank 自己刚写回的值"的假
    # halo（不追求跨 rank 数值真实性，只验证 shape 路径不崩溃、结果
    # 形状始终正确——数值正确性已经由 CPU 侧同构代码路径的两步测试
    # 决定性验证过）。
    des_ext = np.zeros((len(native_ids), n_sps))
    des_ext[:n_local] = turb_gpu.des_length_scale
    fake_halo_gpu.set(des_ext)
    _GPUDistributedInitMixin._compute_turbulence_source_distributed(stub, dt)
    assert turb_gpu.des_length_scale.shape == (n_local, n_sps)
    assert np.all(np.isfinite(turb_gpu.k_field))


@pytest.mark.parametrize("rank", [0, 1])
def test_gpu_distributed_les_matches_single_machine_wale(rank):
    """决定性判据：GPU 分布式 LES（纯 WALE，numpy 替身）算出的 local
    cells mu_t，必须与单机 CPU 路径 `WALEModel.compute_eddy_viscosity`
    对同一批全局单元逐位一致——WALE 是纯代数模型（不依赖跨步状态），
    分布式路径直接用当前 halo 交换后的速度场现算，理论上不需要任何
    "跨步 lag"这类复杂度，这里用真实计算验证这个简化确实成立。"""
    from autoflowcfd.core.turbulence.sgs import WALEModel
    from autoflowcfd.core.gpu.turbulence.gpu_sgs import GPUWALEModel
    from autoflowcfd.core.fr_operators.gradients import compute_physical_gradient as cpu_compute_physical_gradient

    order = 1
    mesh = _build_synthetic_mixed_mesh(order)
    ops = generate_fr_operators(order)
    n_sps = mesh.n_sps_per_cell
    rng = np.random.default_rng(4242)
    U, _, _ = _nonuniform_state(mesh, rng)

    # ---- CPU 单机参照 ----
    grad_U_cpu = cpu_compute_physical_gradient(U[..., :5], mesh, ops)
    grad_vel_cpu = grad_U_cpu[..., 1:4, :]
    delta_cpu = np.abs(mesh.cell_volumes) ** (1.0 / 3.0)
    delta_cpu = np.tile(delta_cpu[:, None], (1, n_sps))
    wale_cpu = WALEModel()
    nu_t_cpu = wale_cpu.compute_eddy_viscosity(grad_vel_cpu, delta_cpu)
    from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive
    rho_cpu = conserved_to_primitive(U[..., :5])[..., 0]
    mu_t_cpu_global = rho_cpu * nu_t_cpu

    # ---- GPU 分布式（numpy 替身）----
    cell_partition = np.array([0, 1, 0, 1], dtype=np.int32)
    fc = mesh.face_connectivity
    partition = build_distributed_partition(fc, cell_partition, rank=rank, n_ranks=2)
    dist_fc = build_distributed_flat_face(mesh, ops, partition, cell_partition=cell_partition)
    compact_global_ids = dist_fc.compact_global_ids
    n_local = partition.n_local_cells
    n_halo = partition.n_halo
    native_ids = (
        np.concatenate([partition.local_cells, partition.halo_cells])
        if n_halo > 0 else partition.local_cells
    )

    mesh_data = _prepare_compact_mesh_data(mesh, ops, compact_global_ids)
    mesh_data['n_prism'] = dist_fc.base_flat.n_prism
    mesh_data['cell_volumes'] = mesh.cell_volumes[compact_global_ids]

    n_sps_val = n_sps
    delta_compact = np.abs(mesh_data['cell_volumes']) ** (1.0 / 3.0)
    grid_scale_compact = np.tile(delta_compact[:, None], (1, n_sps_val))

    U_local = U[partition.local_cells]
    fake_halo_gpu = ShapeKeyedFakeHalo(U[native_ids])

    stub = types.SimpleNamespace(
        rank=rank, device_id=0, mesh=mesh, boundary_ghost_provider=None,
        partition=partition, dist_flat_face=dist_fc, flat_face_gpu=dist_fc.base_flat,
        mesh_data=mesh_data, ops_data=mesh_data,
        turb_model_gpu=None, sgs_model_gpu=GPUWALEModel(),
        gpu_halo=fake_halo_gpu,
        _perm_gpu=dist_fc.perm, _inv_perm_gpu=dist_fc.inv_perm,
        U_gpu=U_local, _grid_scale_compact=grid_scale_compact,
    )
    stub._permute_to_compact = lambda arr: arr[stub._perm_gpu]
    stub._unpermute_from_compact = lambda arr: arr[stub._inv_perm_gpu]
    _bind_turb_source(stub)

    mu_t_compact_gpu = _GPUDistributedInitMixin._compute_turbulence_source_distributed(stub, dt=1e-5)

    mu_t_native_gpu = mu_t_compact_gpu[dist_fc.inv_perm]
    mu_t_local_gpu = mu_t_native_gpu[:n_local]
    expected = mu_t_cpu_global[partition.local_cells]
    np.testing.assert_allclose(mu_t_local_gpu, expected, rtol=1e-9, atol=1e-14)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
