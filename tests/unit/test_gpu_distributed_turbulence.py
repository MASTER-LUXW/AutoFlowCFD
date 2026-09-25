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
`update_fields_gpu` 的数值逻辑，只把 halo 交换+compact 索引空间重排
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
from tests.unit._wall_source import synthetic_wall_source
import pytest

from tests.unit._patch_pkg import patch_pkg_attr

from autoflowcfd.fr.operators import generate_fr_operators
from autoflowcfd.core.mpi.partition import build_distributed_partition
from autoflowcfd.core.mpi.distributed_flat_face import build_distributed_flat_face
from autoflowcfd.core.mpi.distributed_turbulence import (
    distributed_compute_turbulence_source_and_viscosity,
)
from autoflowcfd.core.turbulence.sst import SSTModelFR
from autoflowcfd.core.turbulence.des import DDESModel, IDDESModel, compute_h_max_and_h_wn
from autoflowcfd.core.fr_residual.inviscid import primitive_to_conserved
from autoflowcfd.core.gpu.distributed.gpu_distributed_init import _GPUDistributedInitMixin
from autoflowcfd.core.gpu.turbulence.gpu_turbulence_sst import GPUTurbulenceSST
from autoflowcfd.core.gpu.turbulence.gpu_turbulence_des import GPUDDESModel, GPUIDDESModel
from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh

import autoflowcfd.core.gpu.distributed.gpu_distributed_init as gdi_mod
import autoflowcfd.core.gpu.distributed.gpu_distributed as gd_mod
import autoflowcfd.core.gpu.residual.gpu_gradients as gpu_gradients_mod
import autoflowcfd.core.gpu.residual.gpu_volume_contract as gpu_volume_contract_mod
import autoflowcfd.core.gpu.residual.gpu_flux as gpu_flux_mod
import autoflowcfd.core.gpu.turbulence.gpu_scalar_transport as gst_mod
import autoflowcfd.core.gpu.turbulence.gpu_turbulence_sst as gpu_turbulence_sst_mod
import autoflowcfd.core.gpu.turbulence.gpu_turbulence_des as gpu_turbulence_des_mod
import autoflowcfd.core.gpu.turbulence.gpu_sgs as gpu_sgs_mod
import autoflowcfd.core.gpu.gpu_modal_filter as gpu_modal_filter_mod
from tests.unit._gpu_cupy_shim import patch_module_get_cupy


class _NumpyAsCupy:
    """把 numpy 伪装成 CuPy 模块接口，供各 gpu_*.py 生产函数在没有真实
    CUDA 设备的机器上直接运行（不是重新实现，是给同一份代码换一个张量
    库后端）——与本会话此前 `test_gpu_scalar_transport.py` 同一个模式，
    额外加了 `cuda.Device` 空上下文管理器（`GPUTurbulenceSST.__init__`
    需要）。"""

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
        gdi_mod, gd_mod, gpu_gradients_mod, gpu_volume_contract_mod, gpu_flux_mod,
        gst_mod, gpu_turbulence_sst_mod, gpu_turbulence_des_mod, gpu_sgs_mod,
        gpu_modal_filter_mod], shim)
    # `GPUTurbulenceSST.__init__`/`GPUDDESModel.__init__`/`GPUWALEModel.
    # __init__` 单独检查模块级 `gpu_available` 标志（与 `get_cupy()` 是
    # 否被换成替身无关），本机没有真实 CuPy 时恒为 False，需要一并
    # patch 掉才能在没有真实 CUDA 设备时构造这些类。
    monkeypatch.setattr(gpu_turbulence_sst_mod, "gpu_available", True)
    monkeypatch.setattr(gpu_turbulence_des_mod, "gpu_available", True)
    monkeypatch.setattr(gpu_sgs_mod, "gpu_available", True)


class _FakeHalo:
    def __init__(self, extended):
        self._extended = extended

    def exchange(self, _local):
        return self._extended


def _prepare_compact_mesh_data(mesh, ops, compact_global_ids):
    """构造 compact 索引空间的 mesh_data/ops_data（与 `_CompactMeshDataView`
    /`GPUArrayManager.upload_mesh_data` 语义一致，只是数组仍是 numpy）。"""
    n_sps = mesh.n_sps_per_cell
    det_jacs = mesh.jacobians['det_jacs'].reshape(mesh.n_cells, n_sps)[compact_global_ids]
    inv_jacs = mesh.jacobians['inv_jacs'].reshape(mesh.n_cells, n_sps, 3, 3)[compact_global_ids]
    adj_j = det_jacs[..., None, None] * inv_jacs
    mesh_data = {
        'det_jacs': det_jacs, 'inv_jacs': inv_jacs, 'adj_j': adj_j,
        'n_cells': len(compact_global_ids), 'n_prism': None,  # 调用方会覆盖 n_prism
        'D_3d_prism': ops.D_3d_prism, 'D_3d_tet': ops.D_3d_tet,
    }
    # 补齐生产 GPU 路径会上传、替身容易漏掉的键，见
    # _gpu_standin_helpers 模块文档。本文件算子与网格数据在同一个
    # dict，且细点度量要按 compact 索引空间切。
    from ._gpu_standin_helpers import complete_gpu_standin
    complete_gpu_standin(mesh, ops, mesh_data, mesh_data,
                         compact_ids=compact_global_ids)
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


def _make_ddes_gpu(turb_model_name):
    if turb_model_name == "SST":
        return None
    if turb_model_name == "DDES":
        return GPUDDESModel()
    if turb_model_name == "IDDES":
        return GPUIDDESModel()
    raise ValueError(turb_model_name)


def _make_ddes_cpu(turb_model_name):
    if turb_model_name == "SST":
        return None
    if turb_model_name == "DDES":
        return DDESModel()
    if turb_model_name == "IDDES":
        return IDDESModel()
    raise ValueError(turb_model_name)


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
    fake_halo_5var_cpu = _FakeHalo(U[native_ids][..., :5])
    k_omega_local = np.stack([k_field[partition.local_cells], omega_field[partition.local_cells]], axis=-1)
    k_omega_extended = np.stack([k_field[native_ids], omega_field[native_ids]], axis=-1)
    fake_halo_turb_cpu = _FakeHalo(k_omega_extended)
    d_wall_compact = d_wall[compact_global_ids]
    dt_local_local = np.full((n_local, n_sps), dt)

    mu_t_compact_cpu, _ = distributed_compute_turbulence_source_and_viscosity(
        U_local, partition, fake_halo_5var_cpu, fake_halo_turb_cpu, dist_fc, mesh, ops,
        turb_cpu, mu, d_wall_compact, dt_local_local,
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

    fake_halo_5var_gpu = _FakeHalo(U[native_ids])  # gpu_halo.exchange(U_gpu) 吃 5 变量全量
    fake_halo_turb_gpu = _FakeHalo(k_omega_extended)

    stub = types.SimpleNamespace(
        rank=rank, device_id=0, mu_molecular=mu,
        partition=partition, mesh=mesh, dist_flat_face=dist_fc,
        mesh_data=mesh_data, ops_data=mesh_data, ops=ops,
        flat_face_gpu=dist_fc.base_flat,
        turb_model_gpu=turb_gpu, turb_halo_gpu=fake_halo_turb_gpu, gpu_halo=fake_halo_5var_gpu,
        _perm_gpu=dist_fc.perm, _inv_perm_gpu=dist_fc.inv_perm, n_compact=n_compact,
        wall_distance_gpu=d_wall_compact,
        _wall_mask_k_gpu=np.zeros(dist_fc.base_flat.n_faces, dtype=bool),
        _open_mask_gpu=np.zeros(dist_fc.base_flat.n_faces, dtype=bool),
        U_gpu=U_local,
        ddes_model_gpu=_make_ddes_gpu(turb_model_name),
        iddes_h_max_compact=iddes_h_max_compact, iddes_h_wn_compact=iddes_h_wn_compact,
        des_length_scale_halo_gpu=_FakeHalo(np.zeros((len(native_ids), n_sps, 1))),  # 第一次调用不会被读取
    )
    stub._permute_to_compact = lambda arr: arr[stub._perm_gpu]
    stub._unpermute_from_compact = lambda arr: arr[stub._inv_perm_gpu]

    mu_t_compact_gpu = _GPUDistributedInitMixin._compute_turbulence_source_distributed(stub, dt)

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
    fake_halo_5var_gpu = _FakeHalo(U[native_ids])
    k_omega_extended = np.stack([k_field[native_ids], omega_field[native_ids]], axis=-1)
    fake_halo_turb_gpu = _FakeHalo(k_omega_extended)

    stub = types.SimpleNamespace(
        rank=rank, device_id=0, mu_molecular=mu,
        partition=partition, mesh=mesh, dist_flat_face=dist_fc,
        mesh_data=mesh_data, ops_data=mesh_data, ops=ops,
        flat_face_gpu=dist_fc.base_flat,
        turb_model_gpu=turb_gpu, turb_halo_gpu=fake_halo_turb_gpu, gpu_halo=fake_halo_5var_gpu,
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

    # call 1：des_length_scale 还是 None，第一次调用不会触发 halo 交换
    # 分支，随便给个占位 halo（不会被读取）。
    stub.des_length_scale_halo_gpu = _FakeHalo(np.zeros((len(native_ids), n_sps, 1)))
    _GPUDistributedInitMixin._compute_turbulence_source_distributed(stub, dt)
    assert turb_gpu.des_length_scale is not None
    assert turb_gpu.des_length_scale.shape == (n_local, n_sps)

    # call 2：这次 des_length_scale 非 None，会真正触发 halo 交换分支——
    # 用一个"halo 部分填 0，local 部分是本 rank 自己刚写回的值"的假
    # halo（不追求跨 rank 数值真实性，只验证 shape 路径不崩溃、结果
    # 形状始终正确——数值正确性已经由 CPU 侧同构代码路径的两步测试
    # 决定性验证过）。
    des_ext = np.zeros((len(native_ids), n_sps, 1))
    des_ext[:n_local, :, 0] = turb_gpu.des_length_scale
    stub.des_length_scale_halo_gpu = _FakeHalo(des_ext)
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
    n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
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
    fake_halo_5var_gpu = _FakeHalo(U[native_ids])

    stub = types.SimpleNamespace(
        rank=rank, device_id=0, mesh=mesh,
        mesh_data=mesh_data, ops_data=mesh_data,
        turb_model_gpu=None, sgs_model_gpu=GPUWALEModel(),
        gpu_halo=fake_halo_5var_gpu,
        _perm_gpu=dist_fc.perm, _inv_perm_gpu=dist_fc.inv_perm,
        U_gpu=U_local, _grid_scale_compact=grid_scale_compact,
    )
    stub._permute_to_compact = lambda arr: arr[stub._perm_gpu]
    stub._unpermute_from_compact = lambda arr: arr[stub._inv_perm_gpu]

    mu_t_compact_gpu = _GPUDistributedInitMixin._compute_turbulence_source_distributed(stub, dt=1e-5)

    mu_t_native_gpu = mu_t_compact_gpu[dist_fc.inv_perm]
    mu_t_local_gpu = mu_t_native_gpu[:n_local]
    expected = mu_t_cpu_global[partition.local_cells]
    np.testing.assert_allclose(mu_t_local_gpu, expected, rtol=1e-9, atol=1e-14)


class TestWallDistanceDistributedGpu:
    """`_init_wall_distance_distributed`：compact 索引空间（local + halo）解点上
    按壁面距离来源查询。2026-09-02 修过的缺陷是对全局 sps_coords reshape 后按
    n_local 切（n_ranks>1 时元素总数对不上）；2026-09-25 起它与 CPU 分布式共用
    同一个函数，且删除了"没有壁面就退回单元特征长度"的兜底。"""

    @pytest.mark.parametrize("rank", [0, 1])
    def test_compact_shaped_query_from_the_source(self, rank):
        order = 1
        mesh = _build_synthetic_mixed_mesh(order)
        ops = generate_fr_operators(order)
        cell_partition = np.array([0, 1, 0, 1], dtype=np.int32)
        fc = mesh.face_connectivity
        partition = build_distributed_partition(fc, cell_partition, rank=rank, n_ranks=2)
        dist_fc = build_distributed_flat_face(mesh, ops, partition, cell_partition=cell_partition)
        compact_global_ids = dist_fc.compact_global_ids
        source = synthetic_wall_source(mesh)

        stub = types.SimpleNamespace(
            rank=rank, mesh=mesh, dist_flat_face=dist_fc, turb_model_name="SST",
            _wall_distance_source=source,
        )
        _GPUDistributedInitMixin._init_wall_distance_distributed(stub)

        assert stub.wall_distance_gpu.shape == (len(compact_global_ids), mesh.n_sps_per_cell)
        np.testing.assert_allclose(stub.wall_distance_gpu,
                                   source.query(mesh.sps_coords[compact_global_ids]))

    def test_missing_source_is_an_error_not_an_estimate(self):
        mesh = _build_synthetic_mixed_mesh(1)
        ops = generate_fr_operators(1)
        cell_partition = np.zeros(mesh.n_cells, dtype=np.int32)
        partition = build_distributed_partition(mesh.face_connectivity, cell_partition, rank=0, n_ranks=1)
        dist_fc = build_distributed_flat_face(mesh, ops, partition, cell_partition=cell_partition)
        stub = types.SimpleNamespace(rank=0, mesh=mesh, dist_flat_face=dist_fc, turb_model_name="SST")
        with pytest.raises(RuntimeError):
            _GPUDistributedInitMixin._init_wall_distance_distributed(stub)


class TestWmlesDistributedGpu:
    """多GPU分布式 WMLES 壁面剪应力修正端到端验证（2026-09-02）——与
    `TestWallDistanceDistributedGpu`/上面 SST 系列同一套 numpy 替身
    方法论，直接调用 `MultiGPUDistributedSolver.compute_viscous_
    residual_gpu` 这个真实生产方法（用 stub 对象承载它需要的属性），
    只 mock 掉与本次改动无关的 `compute_viscous_residual_fr_gpu`
    （GPU 粘性通量核，本次没有改动，返回全零隔离出 WMLES 修正项本身），
    判据：stub 路径算出的修正与直接调用 CPU 核心函数
    `compute_wmles_wall_stress_correction`（用同一份 compact 数据）算出
    的参照值逐位一致。"""

    @pytest.mark.parametrize("rank", [0, 1])
    def test_wmles_correction_matches_direct_cpu_call(self, rank, monkeypatch):
        from autoflowcfd.core.turbulence.wmles import WMLESModel
        from autoflowcfd.core.fr_solver.boundary import build_boundary_ghost_provider
        from autoflowcfd.core.utils.solver_helpers import compute_wmles_wall_stress_correction
        import autoflowcfd.core.gpu.residual.gpu_viscous as gpu_viscous_mod

        order = 1
        mesh = _build_synthetic_mixed_mesh(order)
        ops = generate_fr_operators(order)
        n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
        mu = 1.8e-5
        rho_inf = 1.225
        rng = np.random.default_rng(777)
        U, _k, _omega = _nonuniform_state(mesh, rng)

        fc = mesh.face_connectivity
        boundary_face = int(np.nonzero(fc.is_boundary)[0][0])
        wall_cell = int(fc.owner_cell[boundary_face])
        mesh.boundary_groups = {"wall_group": np.array([wall_cell], dtype=np.int64)}
        mesh.boundary_bc_types = {"wall_group": "WALL"}

        root_stub = types.SimpleNamespace(
            mesh=mesh, freestream={"rho_inf": rho_inf, "vel_inf": 30.0, "p_inf": 101325.0},
            turb_model_name="WMLES", wmles_model=object(),
        )
        provider_global = build_boundary_ghost_provider(root_stub, bc_overrides={})
        wmles_model = WMLESModel(nu=mu / rho_inf)
        wall_distance = np.full((n_cells, n_sps), 1e-2)

        cell_partition = np.array([0, 1, 0, 1], dtype=np.int32)
        partition = build_distributed_partition(fc, cell_partition, rank=rank, n_ranks=2)
        dist_fc = build_distributed_flat_face(mesh, ops, partition, cell_partition=cell_partition)
        compact_global_ids = dist_fc.compact_global_ids

        import copy as copy_mod
        provider_local = copy_mod.copy(provider_global)
        provider_local.group_code = provider_global.group_code[partition.local_faces]
        wall_distance_compact = wall_distance[compact_global_ids]

        mesh_data = _prepare_compact_mesh_data(mesh, ops, compact_global_ids)
        mesh_data['n_prism'] = dist_fc.base_flat.n_prism

        compact_mesh_view = types.SimpleNamespace(
            n_prism_cells=dist_fc.base_flat.n_prism, n_points_1d=mesh.n_points_1d,
            jacobians={'det_jacs': mesh_data['det_jacs'], 'inv_jacs': mesh_data['inv_jacs']},
        )

        n_halo = partition.n_halo
        native_ids = (
            np.concatenate([partition.local_cells, partition.halo_cells])
            if n_halo > 0 else partition.local_cells
        )
        U_extended = U[native_ids]

        # 与本次改动无关的 GPU 粘性通量核返回全零，隔离出 WMLES 修正项
        # 本身（该核函数本次未改动，不是要重新验证的对象）。
        patch_pkg_attr(monkeypatch, 
            gpu_viscous_mod, "compute_viscous_residual_fr_gpu",
            lambda *a, **k: np.zeros((len(compact_global_ids), n_sps, 5)),
        )

        stub = types.SimpleNamespace(
            rank=rank, device_id=0, mesh=mesh, mu_molecular=mu,
            mesh_data=mesh_data, ops=ops,
            boundary_ghost_provider=provider_local,
            flat_face_gpu=None,
            dist_flat_face=dist_fc,
            wall_distance_gpu=wall_distance_compact,
            wmles_model=wmles_model,
            _compact_mesh_view=compact_mesh_view,
            U_extended_gpu=U_extended,
            partition=partition,
        )
        stub._permute_to_compact = lambda arr: arr[dist_fc.perm]
        stub._unpermute_from_compact = lambda arr: arr[dist_fc.inv_perm]

        residual_local = gd_mod.MultiGPUDistributedSolver.compute_viscous_residual_gpu(stub)

        # 参照值：直接调用 CPU 核心函数，用同一份 compact 数据。
        from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive
        U_compact = U_extended[dist_fc.perm]
        Q_compact = conserved_to_primitive(U_compact[..., :5])
        facade = types.SimpleNamespace(
            wmles_model=wmles_model, mesh=compact_mesh_view, ops=ops,
            wall_distance=wall_distance_compact,
            state=types.SimpleNamespace(U=U_compact, Q=Q_compact),
            boundary_ghost_provider=provider_local,
        )
        expected_compact = compute_wmles_wall_stress_correction(facade, flat_face_override=dist_fc.base_flat)
        if expected_compact is None:
            # 这个 rank 的 local+halo compact 区域里恰好不含任何真实 WALL
            # 面（4 单元合成网格切成 2 rank 时完全可能发生——WALL 单元
            # 落在另一个 rank）：GPU 侧同样应该完全不叠加任何修正，
            # 残差退化为 mock 的粘性核返回值（全零）。
            np.testing.assert_allclose(residual_local, np.zeros_like(residual_local), atol=1e-10)
            return
        expected_native = expected_compact[dist_fc.inv_perm]
        expected_local = expected_native[:partition.n_local_cells, :, :5]

        np.testing.assert_allclose(residual_local, expected_local, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
