"""AutoFlowCFD V2.0 - 多GPU"完全分布式加载"构造入口验证（问题清单 #1，
2026-09-02）。

背景：`MultiGPUDistributedSolver` 此前只支持"传统模式"（每个 rank 独立
加载完整全局网格），完全没有"完全分布式加载"（只有 root 加载完整网格）
构造入口——见 `gpu_distributed_fully_distributed.py` 模块文档。

方法论（与本项目一贯的"无法本地验证"原则一致，本机没有真实 CUDA/MPI
环境）：
1. `build_fully_distributed_rank_package`（root 侧纯函数，CPU 计算，
   与后端无关）用真实合成网格算出一份真实 package——不是手搓的假
   package，这是本次改动真正的集成风险点（package 字段是否被正确
   消费）。
2. GPU 硬件相关的类（`GPUArrayManager`/`GPUHaloExchange`/
   `build_gpu_flat_face`/`GPUTurbulenceSST` 等）都要求真实 CuPy/CUDA
   （`if not gpu_available: raise` 或 `if cp is None: raise`），本机
   无法真实构造——这些类本身的数值逻辑已由其他测试文件独立验证过
   （`test_gpu_distributed_turbulence.py` 等），不是本次改动的对象，
   这里用轻量 Fake 替换，只验证"我的新代码是否用正确的参数调用了
   正确的构造函数、是否把 package 里的字段接到了正确的属性上"——与
   `test_gpu_distributed_dual_time.py`/`test_gpu_distributed_order_
   continuation.py` 同一个既定方法论（不重新验证既有 GPU kernel）。
3. `redistribute_multi_gpu_fully_distributed_for_new_order`（Order
   Continuation）用同一个真实合成网格 + 真实 root_context，决定性
   验证插值数学（升阶精确 Lagrange 延拓 / 降阶重置为均匀流场）与
   `current_order`/mesh/partition 更新。
"""

import types

import numpy as np
import pytest

from autoflowcfd.core.mpi.distributed_mesh_loader import (
    build_fully_distributed_rank_package,
    distributed_mesh_load_v2,
)
from autoflowcfd.core.mpi.distributed_flat_face import build_distributed_flat_face
from autoflowcfd.core.mpi.partition import build_distributed_partition
from autoflowcfd.fr.operators import generate_fr_operators
from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh

import autoflowcfd.core.gpu.distributed.gpu_distributed_fully_distributed as gdfd_mod
import autoflowcfd.core.gpu.array_manager as array_manager_mod
import autoflowcfd.core.gpu.gpu_face_geometry as gpu_face_geometry_mod


class _NumpyAsCupy:
    def __getattr__(self, name):
        return getattr(np, name)

    def asnumpy(self, x):
        return np.asarray(x)

    class cuda:
        class Device:
            def __init__(self, device_id=0):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        class runtime:
            @staticmethod
            def getDeviceCount():
                return 1


class _FakeArrayManager:
    """替身：真实 `GPUArrayManager` 要求真实 CuPy/CUDA 设备
    （`if not gpu_available: raise`），本机无法构造。`upload_mesh_data`
    的真实数值逻辑已由其他测试独立验证（GPU 侧无法在本机跑，但其纯
    numpy 输入/输出的对应 CPU 逻辑早已验证），这里只需要返回一个带
    `cell_volumes` 键的 dict，供本类后续代码消费（NONE/SST 分支都会
    读取 `mesh_data.items()` 构造 `ops_data`，LES 分支会读
    `mesh_data['cell_volumes']`）。"""

    def __init__(self, device_id=0):
        self.device_id = device_id

    def upload_mesh_data(self, mesh, ops):
        return {'cell_volumes': np.asarray(mesh.cell_volumes)}


class _FakeHaloExchange:
    def __init__(self, *args, **kwargs):
        pass


def _fake_build_gpu_flat_face(flat_face, device_id):
    return types.SimpleNamespace(n_faces=flat_face.n_faces)


@pytest.fixture()
def gpu_shim(monkeypatch):
    """把本次改动新模块 + 其调用的 GPU 硬件相关类全部换成替身，
    见模块文档"方法论"一节。"""
    shim = _NumpyAsCupy()
    monkeypatch.setattr(gdfd_mod, "get_cupy", lambda: shim)
    monkeypatch.setattr(gdfd_mod, "GPUHaloExchange", _FakeHaloExchange)
    monkeypatch.setattr(array_manager_mod, "GPUArrayManager", _FakeArrayManager)
    monkeypatch.setattr(gpu_face_geometry_mod, "build_gpu_flat_face", _fake_build_gpu_flat_face)
    return shim


@pytest.fixture(scope="module")
def mesh_and_ops():
    order = 1
    mesh = _build_synthetic_mixed_mesh(order)
    ops = generate_fr_operators(order)
    assert mesh.n_cells == 4
    return mesh, ops


def _build_none_package(mesh, ops, rank=0, n_ranks=1):
    fc = mesh.face_connectivity
    cell_partition = np.zeros(mesh.n_cells, dtype=np.int32)
    freestream = {"rho_inf": 1.225, "vel_inf": 33.33, "p_inf": 101325.0}

    from autoflowcfd.core.fr_solver.boundary import build_boundary_ghost_provider
    root_stub = types.SimpleNamespace(
        mesh=mesh, freestream=freestream, turb_model_name="NONE", wmles_model=None,
    )
    boundary_ghost_provider = build_boundary_ghost_provider(root_stub, bc_overrides={})

    return build_fully_distributed_rank_package(
        mesh, ops, fc, cell_partition, rank, n_ranks,
        boundary_ghost_provider, freestream, mu_molecular=1.8e-5, mach_ref=0.2,
        order=mesh.order, enable_viscous=True, turb_model_name="NONE",
    )


def _build_sst_package(mesh, ops, rank=0, n_ranks=1):
    fc = mesh.face_connectivity
    cell_partition = np.zeros(mesh.n_cells, dtype=np.int32)
    freestream = {"rho_inf": 1.225, "vel_inf": 33.33, "p_inf": 101325.0}

    from autoflowcfd.core.fr_solver.boundary import build_boundary_ghost_provider
    root_stub = types.SimpleNamespace(
        mesh=mesh, freestream=freestream, turb_model_name="SST", wmles_model=None,
    )
    boundary_ghost_provider = build_boundary_ghost_provider(root_stub, bc_overrides={})

    return build_fully_distributed_rank_package(
        mesh, ops, fc, cell_partition, rank, n_ranks,
        boundary_ghost_provider, freestream, mu_molecular=1.8e-5, mach_ref=0.2,
        order=mesh.order, enable_viscous=True, turb_model_name="SST",
        turbulence_intensity=0.02, viscosity_ratio=8.0,
    )


class TestBuildFromFullyDistributedPackageNoneTurbulence:
    def test_wires_geometry_and_freestream_correctly(self, mesh_and_ops, gpu_shim):
        from autoflowcfd.core.gpu.distributed.gpu_distributed import MultiGPUDistributedSolver
        from autoflowcfd.core.gpu.distributed.gpu_distributed_fully_distributed import (
            build_multi_gpu_solver_from_fully_distributed_package,
        )

        mesh, ops = mesh_and_ops
        package = _build_none_package(mesh, ops)

        solver = build_multi_gpu_solver_from_fully_distributed_package(
            MultiGPUDistributedSolver, package, n_ranks=1, device_id=0, rank=0, root_context=None,
        )

        assert solver._is_fully_distributed is True
        assert solver.cell_partition is None
        assert solver.rank == 0
        assert solver.n_ranks == 1
        assert solver.current_order == mesh.order
        assert solver.order == mesh.order

        # partition/dist_flat_face 必须原样来自 package（root 已经预先
        # 按 compact 索引空间切好，不应该被重新计算）。
        assert solver.partition is package['partition']
        assert solver.dist_flat_face is package['dist_fc']
        assert solver.mesh is package['precompacted_mesh']
        assert solver.n_compact == package['precompacted_mesh'].n_cells

        # boundary_ghost_provider：直接取自 package（root 已经把
        # group_code 重映射到本 rank 的 compact 索引空间），不应该在
        # 这里重新调用 build_boundary_ghost_provider。
        assert solver.boundary_ghost_provider is package['boundary_ghost_provider']

        # freestream 必须无条件设置（不只是 SST/DDES/IDDES 分支），
        # AUSM+up mach_ref 全部 turb_model_name 都要用到。
        assert solver.freestream["rho_inf"] == pytest.approx(1.225)
        assert solver.freestream["mach_ref"] == pytest.approx(0.2)
        assert solver.mu_molecular == pytest.approx(1.8e-5)

        # U_gpu 均匀自由流场初始化。
        rho_inf, vel_inf, p_inf = 1.225, 33.33, 101325.0
        n_local = solver.partition.n_local_cells
        n_sps = package['precompacted_mesh'].n_sps_per_cell
        assert solver.U_gpu.shape == (n_local, n_sps, 5)
        np.testing.assert_allclose(solver.U_gpu[:, :, 0], rho_inf)
        np.testing.assert_allclose(solver.U_gpu[:, :, 1], rho_inf * vel_inf)
        expected_e = p_inf / 0.4 + 0.5 * rho_inf * vel_inf ** 2
        np.testing.assert_allclose(solver.U_gpu[:, :, 4], expected_e)

        # NONE：不应该构造任何湍流/SGS/WMLES 模型。
        assert solver.turb_model_gpu is None
        assert solver.sgs_model_gpu is None
        assert solver.wmles_model is None
        assert solver.wall_distance_gpu is None

    def test_classmethod_delegates_to_builder(self, mesh_and_ops, gpu_shim, monkeypatch):
        """`MultiGPUDistributedSolver.from_fully_distributed_package`
        classmethod 必须把参数原样转发给
        `build_multi_gpu_solver_from_fully_distributed_package`（不是
        重新实现一遍逻辑）。"""
        from autoflowcfd.core.gpu.distributed.gpu_distributed import MultiGPUDistributedSolver
        import autoflowcfd.core.gpu.distributed.gpu_distributed as gd_mod

        captured = {}

        def _fake_builder(cls, package, n_ranks, device_id=None, rank=None, root_context=None):
            captured.update(cls=cls, package=package, n_ranks=n_ranks,
                             device_id=device_id, rank=rank, root_context=root_context)
            return "SENTINEL"

        monkeypatch.setattr(gdfd_mod, "build_multi_gpu_solver_from_fully_distributed_package", _fake_builder)

        mesh, ops = mesh_and_ops
        package = _build_none_package(mesh, ops)
        result = MultiGPUDistributedSolver.from_fully_distributed_package(
            package, n_ranks=1, device_id=2, rank=0, root_context="ROOT_CTX",
        )

        assert result == "SENTINEL"
        assert captured["cls"] is MultiGPUDistributedSolver
        assert captured["package"] is package
        assert captured["n_ranks"] == 1
        assert captured["device_id"] == 2
        assert captured["rank"] == 0
        assert captured["root_context"] == "ROOT_CTX"


class TestBuildFromFullyDistributedPackageSstTurbulence:
    def test_wires_sst_turbulence_from_package(self, mesh_and_ops, gpu_shim, monkeypatch):
        """SST：`wall_distance_compact`（root 预先算好）必须原样上传成
        `wall_distance_gpu`；k_inf/omega_inf 必须按 Tu/VR 公式算对；
        turb_halo_gpu 必须构造。真实 `GPUTurbulenceSST`/
        `compute_wall_dirichlet_mask_gpu` 要求真实 CUDA，这里用最小
        Fake 隔离（不重新验证它们的数值逻辑，那是其他测试文件的范围）。
        """
        import autoflowcfd.core.gpu.turbulence.gpu_turbulence_sst as sst_mod
        import autoflowcfd.core.gpu.turbulence.gpu_scalar_transport as scalar_transport_mod

        class _FakeGPUTurbulenceSST:
            def __init__(self, n_cells, n_sps, device_id, k_inf=1e-6, omega_inf=1.0):
                self.n_cells = n_cells
                self.n_sps = n_sps
                self.k_inf = k_inf
                self.omega_inf = omega_inf
                self.k_field = np.full((n_cells, n_sps), k_inf)
                self.omega_field = np.full((n_cells, n_sps), omega_inf)
                self.k_max = None
                self.omega_max = None

        monkeypatch.setattr(sst_mod, "GPUTurbulenceSST", _FakeGPUTurbulenceSST)
        monkeypatch.setattr(
            scalar_transport_mod, "compute_wall_dirichlet_mask_gpu",
            lambda compact_mesh_stub, provider: np.zeros(compact_mesh_stub.face_connectivity.n_faces, dtype=bool),
        )

        from autoflowcfd.core.gpu.distributed.gpu_distributed import MultiGPUDistributedSolver
        from autoflowcfd.core.gpu.distributed.gpu_distributed_fully_distributed import (
            build_multi_gpu_solver_from_fully_distributed_package,
        )

        mesh, ops = mesh_and_ops
        package = _build_sst_package(mesh, ops)
        assert package['wall_distance_compact'] is not None

        solver = build_multi_gpu_solver_from_fully_distributed_package(
            MultiGPUDistributedSolver, package, n_ranks=1, device_id=0, rank=0, root_context=None,
        )

        assert solver.turb_model_gpu is not None
        assert solver.turb_halo_gpu is not None
        assert solver.wall_distance_gpu is not None
        np.testing.assert_allclose(solver.wall_distance_gpu, package['wall_distance_compact'])

        vel_inf, Tu, VR = 33.33, 0.02, 8.0
        rho_inf, mu = 1.225, 1.8e-5
        nu = mu / rho_inf
        k_inf_expected = 1.5 * (vel_inf * Tu) ** 2
        omega_inf_expected = k_inf_expected / (VR * nu)
        assert solver.turb_model_gpu.k_inf == pytest.approx(k_inf_expected)
        assert solver.turb_model_gpu.omega_inf == pytest.approx(omega_inf_expected)
        assert solver.turb_model_gpu.k_max == pytest.approx(0.5 * vel_inf ** 2)
        assert solver.turb_model_gpu.omega_max == pytest.approx(1e6)

        assert solver._wall_mask_k_gpu is not None


class TestRedistributeMultiGpuFullyDistributedForNewOrder:
    """Order Continuation：`redistribute_multi_gpu_fully_distributed_
    for_new_order` 用真实合成网格 + 真实 root_context，决定性验证插值
    数学与阶数切换后的 mesh/partition/current_order 更新（与 CPU 版
    `redistribute_fully_distributed_for_new_order` 同一套判据）。"""

    def _make_root_context(self, mesh, ops):
        fc = mesh.face_connectivity
        cell_partition = np.zeros(mesh.n_cells, dtype=np.int32)
        freestream = {"rho_inf": 1.225, "vel_inf": 33.33, "p_inf": 101325.0}
        import types as _types
        from autoflowcfd.core.fr_solver.boundary import build_boundary_ghost_provider
        root_stub = _types.SimpleNamespace(
            mesh=mesh, freestream=freestream, turb_model_name="NONE", wmles_model=None,
        )
        boundary_ghost_provider_global = build_boundary_ghost_provider(root_stub, bc_overrides={})
        return {
            'mesh': mesh, 'ops': ops, 'fc': fc, 'cell_partition': cell_partition,
            'boundary_ghost_provider_global': boundary_ghost_provider_global,
            'freestream': freestream, 'mu_molecular': 1.8e-5, 'mach_ref': 0.2,
            'enable_viscous': True, 'turb_model_name': 'NONE',
            'wall_node_indices': None, 'h_max_global': None, 'h_wn_global': None,
            'turbulence_intensity': 0.01, 'viscosity_ratio': 5.0,
            'bc_overrides': {}, 'n_ranks': 1,
            'time_scheme': None, 'dual_time_inner_iter': 20,
        }

    def test_p0_to_p1_upgrade_interpolates_uniform_field_exactly(self, mesh_and_ops, gpu_shim):
        from autoflowcfd.core.gpu.distributed.gpu_distributed_fully_distributed import (
            redistribute_multi_gpu_fully_distributed_for_new_order,
        )

        mesh, ops = mesh_and_ops
        root_context = self._make_root_context(mesh, ops)

        # 先在 P0 上构造一个"传统模式"意义下的均匀自由流状态 solver stub
        # （只暴露该函数真正读取的属性——见模块文档"方法论"一节）。
        rho_inf, vel_inf, p_inf = 1.225, 33.33, 101325.0
        gamma = 1.4
        e = p_inf / ((gamma - 1.0) * rho_inf) + 0.5 * vel_inf ** 2

        solver = types.SimpleNamespace()
        solver.rank = 0
        solver.n_ranks = 1
        solver.device_id = 0
        solver.current_order = 0
        solver._root_context = root_context
        solver._package_freestream = root_context['freestream']
        solver.freestream = {**root_context['freestream'], "mach_ref": root_context['mach_ref']}
        solver.mu_molecular = root_context['mu_molecular']
        n_local = mesh.n_cells  # 单 rank：全部 local
        solver.partition = types.SimpleNamespace(n_local_cells=n_local)
        # **升阶插值要读局部棱柱数**（2026-09-20）：延拓算子按基与单元
        # 类型分派（见 `fr/order_interp.py`）。真实实例上这个属性来自
        # `PrecompactedMeshData.n_prism_cells`（compact 索引空间、棱柱在
        # 前）；stub 只暴露被真正读取的属性，所以这里补上它。
        solver.mesh = types.SimpleNamespace(
            n_prism_cells=int(mesh.n_prism_cells))
        solver.U_gpu = np.zeros((n_local, 1, 5))
        solver.U_gpu[:, :, 0] = rho_inf
        solver.U_gpu[:, :, 1] = rho_inf * vel_inf
        solver.U_gpu[:, :, 4] = rho_inf * e
        solver.turb_model_gpu = None
        solver.ddes_model_gpu = None
        solver.sgs_model_gpu = None
        solver.wmles_model = None
        solver.array_mgr = _FakeArrayManager(device_id=0)
        solver._init_modal_filter_distributed = types.MethodType(lambda self: None, solver)

        redistribute_multi_gpu_fully_distributed_for_new_order(solver, 1)

        assert solver.current_order == 1
        new_n_sps = (1 + 1) ** 3
        assert solver.U_gpu.shape == (n_local, new_n_sps, 5)
        # 均匀自由流场：升阶插值必须精确保持（Lagrange 延拓的定义性质）。
        np.testing.assert_allclose(solver.U_gpu[:, :, 0], rho_inf)
        np.testing.assert_allclose(solver.U_gpu[:, :, 1], rho_inf * vel_inf)
        np.testing.assert_allclose(solver.U_gpu[:, :, 4], rho_inf * e)
        assert solver.mesh.n_sps_per_cell == new_n_sps
        assert solver.partition.n_local_cells == n_local
        assert solver.boundary_ghost_provider is not None

    def test_reset_to_p0_when_downgrading(self, mesh_and_ops, gpu_shim):
        from autoflowcfd.core.gpu.distributed.gpu_distributed_fully_distributed import (
            redistribute_multi_gpu_fully_distributed_for_new_order,
        )

        mesh, ops = mesh_and_ops
        root_context = self._make_root_context(mesh, ops)
        rho_inf, vel_inf, p_inf = 1.225, 33.33, 101325.0
        n_local = mesh.n_cells

        solver = types.SimpleNamespace()
        solver.rank = 0
        solver.n_ranks = 1
        solver.device_id = 0
        solver.current_order = 1
        solver._root_context = root_context
        solver._package_freestream = root_context['freestream']
        solver.freestream = {**root_context['freestream'], "mach_ref": root_context['mach_ref']}
        solver.mu_molecular = root_context['mu_molecular']
        n_sps_old = (1 + 1) ** 3
        solver.partition = types.SimpleNamespace(n_local_cells=n_local)
        solver.U_gpu = np.random.default_rng(3).uniform(1, 2, size=(n_local, n_sps_old, 5))
        solver.turb_model_gpu = None
        solver.ddes_model_gpu = None
        solver.sgs_model_gpu = None
        solver.wmles_model = None
        solver.array_mgr = _FakeArrayManager(device_id=0)
        solver._init_modal_filter_distributed = types.MethodType(lambda self: None, solver)

        redistribute_multi_gpu_fully_distributed_for_new_order(solver, 0)

        assert solver.current_order == 0
        assert solver.U_gpu.shape == (n_local, 1, 5)
        np.testing.assert_allclose(solver.U_gpu[:, :, 0], rho_inf)
        np.testing.assert_allclose(solver.U_gpu[:, :, 1], rho_inf * vel_inf)

    def test_missing_root_context_raises(self, gpu_shim):
        from autoflowcfd.core.gpu.distributed.gpu_distributed_fully_distributed import (
            redistribute_multi_gpu_fully_distributed_for_new_order,
        )
        solver = types.SimpleNamespace(current_order=0, rank=0, _root_context=None)
        with pytest.raises(NotImplementedError):
            redistribute_multi_gpu_fully_distributed_for_new_order(solver, 1)


class TestGpuInterpolateDispatchesFullyDistributed:
    """`gpu_interpolate_to_new_order` 必须按 `_is_fully_distributed`
    分派到本次新增的重建函数，而不是落到"传统模式"的 `cell_partition
    is None` fail-fast 分支（那是给 `__init__` 的 `partition_info` 遗留
    分支用的，两者是不同的东西，见 `gpu_distributed_order_continuation.py`
    模块文档更正说明）。"""

    def test_fully_distributed_flag_dispatches_to_redistribute(self, monkeypatch):
        from autoflowcfd.core.gpu.distributed.gpu_distributed_order_continuation import (
            gpu_interpolate_to_new_order,
        )
        import autoflowcfd.core.gpu.distributed.gpu_distributed_order_continuation as oc_mod

        captured = {}

        def _fake_redistribute(solver, target_p):
            captured['solver'] = solver
            captured['target_p'] = target_p

        monkeypatch.setattr(
            "autoflowcfd.core.gpu.distributed.gpu_distributed_fully_distributed."
            "redistribute_multi_gpu_fully_distributed_for_new_order",
            _fake_redistribute,
        )

        solver = types.SimpleNamespace(current_order=0, _is_fully_distributed=True, cell_partition=None)
        gpu_interpolate_to_new_order(solver, 1)

        assert captured['solver'] is solver
        assert captured['target_p'] == 1

    def test_traditional_mode_without_cell_partition_still_raises(self):
        """回归防护：非"完全分布式加载"（`_is_fully_distributed` 缺失/
        False）且 `cell_partition is None` 的旧路径必须继续 fail-fast，
        不能被本次改动误放行。"""
        from autoflowcfd.core.gpu.distributed.gpu_distributed_order_continuation import (
            gpu_interpolate_to_new_order,
        )
        solver = types.SimpleNamespace(current_order=0, cell_partition=None)
        with pytest.raises(NotImplementedError):
            gpu_interpolate_to_new_order(solver, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
