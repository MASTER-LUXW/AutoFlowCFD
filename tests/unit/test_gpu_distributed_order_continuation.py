"""多GPU分布式 Order Continuation 验证（2026-09-02）。

背景：`MultiGPUDistributedSolver` 之前完全没有 Order Continuation
机制（与 CPU `DistributedFRSolver` 同一处缺口，见 core/gpu/distributed/
gpu_distributed_order_continuation.py 模块文档）。

验证范围/方法论说明（与本项目一贯的"无法本地验证"原则一致）：本机
没有真实 CUDA/MPI 环境。用 numpy-as-cupy 替身可以让
`MultiGPUDistributedSolver.__init__` 走得比此前任何测试都更远（过程中
真实发现并修复了 `_CompactMeshDataView` 缺失 `n_sps_per_cell_fine`
属性的真实 bug——任何启用了过积分的 GPU 分布式构造都会因此崩溃，
此前从未被任何测试捕捉到，因为所有既有 GPU 分布式测试都只测试更底层
的独立函数），但会在 `GPUHaloExchange.__init__` 的 CUDA-aware MPI
探测处（`is_cuda_aware_mpi()` 需要真正的 `mpi4py.MPI.Comm` 对象，不是
本次改动能力范围内可以合理伪造的东西）撞到一个更深、与 Order
Continuation 本身无关的既有障碍——因此本文件不测试完整
`MultiGPUDistributedSolver.__init__`，改用与既有
`test_gpu_distributed_dual_time.py` 同一个方法论：构造一个只暴露
`gpu_interpolate_to_new_order` 真正读取/写入的属性的最小 stub，
决定性验证这个函数本身的插值数学与重建时序（阶数切换前后 n_sps
是否正确、`current_order` 是否正确更新、插值矩阵的精确 Lagrange
延拓性质——与单机 `order_continuation.py` 同一份验证方式，见
`test_order_continuation.py`）。
"""

import types

import numpy as np
import pytest


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


@pytest.fixture()
def gpu_oc_module(monkeypatch):
    import autoflowcfd.core.gpu.distributed.gpu_distributed_order_continuation as mod
    import autoflowcfd.core.gpu as core_gpu_mod
    shim = _NumpyAsCupy()
    # `gpu_interpolate_to_new_order` 内部用 `from autoflowcfd.core.gpu
    # import get_cupy`（函数内延迟导入，每次调用都重新从源模块取），
    # patch 源模块的属性即可，本模块自身不持有这个名字。
    monkeypatch.setattr(core_gpu_mod, "get_cupy", lambda: shim)
    return mod


def _make_mesh_stub(order, n_cells=2):
    """只暴露 `gpu_interpolate_to_new_order` 真正读取的 mesh 属性：
    `_order_geometry_cache`/`set_order`/`face_connectivity`。不使用真实
    `HighOrderMesh`——本测试的目标是验证重建编排本身（数组形状/属性
    赋值顺序），不是重新验证 `HighOrderMesh.set_order`（已有独立测试
    覆盖，见 test_high_order_mesh_order_continuation 等既有文件）。"""
    mesh = types.SimpleNamespace()
    mesh._order_geometry_cache = {order: object()}
    mesh.order = order
    mesh.face_connectivity = object()
    n_sps = (order + 1) ** 3
    mesh.n_sps_per_cell = n_sps
    mesh.n_points_1d = order + 1
    mesh.n_cells = n_cells
    # **局部棱柱数**（2026-09-20）：延拓算子按基与单元类型分派，需要知道
    # compact 索引空间里"棱柱在前"的前缀长度（生产里由
    # `_GPUDistributedMeshAdapter` 从 `base_flat.n_prism` 赋值）。这里的
    # stub 代表一个全棱柱的局部分区。
    mesh.n_prism_cells = n_cells
    mesh.jacobians = None
    mesh.jacobians_fine = None
    mesh.cell_volumes = np.ones(n_cells)

    def _set_order(new_order):
        mesh.order = new_order
        mesh.n_sps_per_cell = (new_order + 1) ** 3
        mesh.n_points_1d = new_order + 1
        mesh._order_geometry_cache.setdefault(new_order, object())
    mesh.set_order = _set_order
    return mesh


class TestGpuInterpolateToNewOrderCoreMath:
    """不构造完整 `MultiGPUDistributedSolver`（见模块文档"验证范围"
    一节），只验证 `gpu_interpolate_to_new_order` 对 U/湍流场的插值
    数学与 n_sps/current_order 更新是否正确——用 monkeypatch 短路掉
    几何重建部分（`build_distributed_partition`/`_init_distributed_
    face_geometry`/`array_mgr.upload_mesh_data` 等，这些都已经是本
    项目其余测试独立验证过的既有基础设施，不是本次改动的对象）。"""

    def test_p0_to_p1_upgrade_interpolates_uniform_field_exactly(self, gpu_oc_module, monkeypatch):
        from autoflowcfd.core.gpu.distributed.gpu_distributed_order_continuation import (
            gpu_interpolate_to_new_order,
        )

        n_local = 3
        rho_inf, vel_inf, p_inf = 1.225, 33.33, 101325.0
        gamma = 1.4
        e = p_inf / ((gamma - 1.0) * rho_inf) + 0.5 * vel_inf ** 2

        solver = types.SimpleNamespace()
        solver.current_order = 0
        solver.cell_partition = np.array([0, 0, 0])
        solver.mesh = _make_mesh_stub(0, n_local)
        solver.rank = 0
        solver.n_ranks = 1
        solver.device_id = 0
        solver.flux_type = "radau"
        solver.freestream = {"rho_inf": rho_inf, "vel_inf": vel_inf, "p_inf": p_inf}
        solver.partition = types.SimpleNamespace(n_local_cells=n_local, local_faces=np.array([], dtype=np.int64))
        solver.U_gpu = np.zeros((n_local, 1, 5))
        solver.U_gpu[:, :, 0] = rho_inf
        solver.U_gpu[:, :, 1] = rho_inf * vel_inf
        solver.U_gpu[:, :, 4] = rho_inf * e
        solver.turb_model_gpu = None
        solver.ddes_model_gpu = None
        solver.sgs_model_gpu = None
        solver.wmles_model = None

        # 短路掉几何/分区/上传相关的重型基础设施——已由本项目其余测试
        # 独立验证过，不是本次改动的对象。
        new_partition = types.SimpleNamespace(n_local_cells=n_local, n_halo=0, n_total_cells=n_local, local_faces=np.array([], dtype=np.int64))
        monkeypatch.setattr(gpu_oc_module, "build_distributed_partition", lambda *a, **k: new_partition) \
            if hasattr(gpu_oc_module, "build_distributed_partition") else None

        import autoflowcfd.core.mpi.partition as partition_mod
        monkeypatch.setattr(partition_mod, "build_distributed_partition", lambda *a, **k: new_partition)

        def _init_face_geometry(self):
            self.dist_flat_face = types.SimpleNamespace(
                compact_global_ids=np.arange(n_local),
                base_flat=types.SimpleNamespace(n_prism=0, n_faces=0),
                perm=np.arange(n_local), inv_perm=np.arange(n_local),
            )
        solver._init_distributed_face_geometry = types.MethodType(_init_face_geometry, solver)
        solver._init_modal_filter_distributed = types.MethodType(lambda self: None, solver)
        solver._init_wall_distance_distributed = types.MethodType(lambda self: None, solver)

        class _FakeArrayMgr:
            def upload_mesh_data(self, mesh, ops):
                return {"cell_volumes": np.ones(n_local)}
        solver.array_mgr = _FakeArrayMgr()

        monkeypatch.setattr(
            "autoflowcfd.core.fr_solver.boundary.build_boundary_ghost_provider",
            lambda solver_, bc_overrides=None: None,
        )

        class _FakeOps:
            D_3d = np.zeros((8, 8))
        import autoflowcfd.fr.operators as ops_mod
        monkeypatch.setattr(ops_mod, "generate_fr_operators", lambda order, **k: _FakeOps())

        class _FakeHalo:
            def __init__(self, *a, **k):
                pass
        import autoflowcfd.core.gpu.distributed.gpu_halo_exchange as ghe_mod
        monkeypatch.setattr(ghe_mod, "GPUHaloExchange", _FakeHalo)

        gpu_interpolate_to_new_order(solver, 1)

        assert solver.current_order == 1
        # 均匀自由流场：升阶插值必须精确保持（新 n_sps=8，见 _FakeOps）。
        assert solver.U_gpu.shape == (n_local, 8, 5)
        np.testing.assert_allclose(solver.U_gpu[:, :, 0], rho_inf)
        np.testing.assert_allclose(solver.U_gpu[:, :, 1], rho_inf * vel_inf)
        np.testing.assert_allclose(solver.U_gpu[:, :, 4], rho_inf * e)

    def test_reset_to_p0_when_downgrading(self, gpu_oc_module, monkeypatch):
        """降阶（reset 场景，见 CPU 版同名文档）：不做插值，直接重置为
        均匀自由流场。"""
        from autoflowcfd.core.gpu.distributed.gpu_distributed_order_continuation import (
            gpu_interpolate_to_new_order,
        )

        n_local = 2
        rho_inf, vel_inf, p_inf = 1.225, 33.33, 101325.0

        solver = types.SimpleNamespace()
        solver.current_order = 2
        solver.cell_partition = np.array([0, 0])
        solver.mesh = _make_mesh_stub(2, n_local)
        solver.rank = 0
        solver.n_ranks = 1
        solver.device_id = 0
        solver.flux_type = "radau"
        solver.freestream = {"rho_inf": rho_inf, "vel_inf": vel_inf, "p_inf": p_inf}
        solver.partition = types.SimpleNamespace(n_local_cells=n_local, local_faces=np.array([], dtype=np.int64))
        solver.U_gpu = np.random.default_rng(1).uniform(1, 2, size=(n_local, 27, 5))
        solver.turb_model_gpu = None
        solver.ddes_model_gpu = None
        solver.sgs_model_gpu = None
        solver.wmles_model = None

        new_partition = types.SimpleNamespace(n_local_cells=n_local, n_halo=0, n_total_cells=n_local, local_faces=np.array([], dtype=np.int64))
        import autoflowcfd.core.mpi.partition as partition_mod
        monkeypatch.setattr(partition_mod, "build_distributed_partition", lambda *a, **k: new_partition)

        def _init_face_geometry(self):
            self.dist_flat_face = types.SimpleNamespace(
                compact_global_ids=np.arange(n_local),
                base_flat=types.SimpleNamespace(n_prism=0, n_faces=0),
                perm=np.arange(n_local), inv_perm=np.arange(n_local),
            )
        solver._init_distributed_face_geometry = types.MethodType(_init_face_geometry, solver)
        solver._init_modal_filter_distributed = types.MethodType(lambda self: None, solver)
        solver._init_wall_distance_distributed = types.MethodType(lambda self: None, solver)

        class _FakeArrayMgr:
            def upload_mesh_data(self, mesh, ops):
                return {"cell_volumes": np.ones(n_local)}
        solver.array_mgr = _FakeArrayMgr()

        monkeypatch.setattr(
            "autoflowcfd.core.fr_solver.boundary.build_boundary_ghost_provider",
            lambda solver_, bc_overrides=None: None,
        )

        class _FakeOps:
            D_3d = np.zeros((1, 1))
        import autoflowcfd.fr.operators as ops_mod
        monkeypatch.setattr(ops_mod, "generate_fr_operators", lambda order, **k: _FakeOps())

        class _FakeHalo:
            def __init__(self, *a, **k):
                pass
        import autoflowcfd.core.gpu.distributed.gpu_halo_exchange as ghe_mod
        monkeypatch.setattr(ghe_mod, "GPUHaloExchange", _FakeHalo)

        gpu_interpolate_to_new_order(solver, 0)

        assert solver.current_order == 0
        assert solver.U_gpu.shape == (n_local, 1, 5)
        np.testing.assert_allclose(solver.U_gpu[:, :, 0], rho_inf)

    def test_cell_partition_none_raises(self, gpu_oc_module):
        """'分布式加载'（partition_info 构造）模式：本 rank 没有完整
        全局网格，必须 fail-fast，不能假装能用。"""
        from autoflowcfd.core.gpu.distributed.gpu_distributed_order_continuation import (
            gpu_interpolate_to_new_order,
        )
        solver = types.SimpleNamespace(current_order=0, cell_partition=None)
        with pytest.raises(NotImplementedError):
            gpu_interpolate_to_new_order(solver, 1)


class TestResumeCeilingFractionResetHeuristicGpu:
    """真实完整性缺口修复回归测试（2026-09-05）：多 GPU 分布式（以及
    单机 `GPUFRSolver`，鸭子类型复用同一份实现，见
    `distributed_order_continuation.py::run_distributed_order_
    continuation` 模块文档"三条后端统一走本模块"一节）此前完全没有
    单机 CPU 早就有的"resume 时检测湍流场是否被上界大面积钳制"安全网，
    见 `_reset_turbulence_if_resumed_field_exploded` 文档完整推导。

    这里只需要验证被测函数本身只用 `.sum()`/`.size`/`>=`/`float(...)`
    这几个 numpy 与 CuPy 语义完全一致的数组 API（函数体内没有任何
    `get_cupy()`/`cp.xxx` 调用，直接对传入的 `turb_model_gpu.k_field`
    数组操作），不需要真正的 CUDA 设备就能决定性验证——用一个只暴露
    `turb_model_gpu`/`freestream`/`mu_molecular` 这几个被读取属性的
    最小 stub（同本文件其余测试类的既有方法论），数组用 numpy 构造
    （不经过 `_NumpyAsCupy` shim 也可以，因为函数本身不调用 `get_cupy()`；
    这里仍然用 shim 构造只是与本文件既有风格保持一致）。
    """

    def _make_solver_stub(self, k_value, omega_value, n_cells=8, n_sps=8,
                           k_max=555.44, omega_max=1e6):
        turb = types.SimpleNamespace(
            k_field=np.full((n_cells, n_sps), k_value, dtype=np.float64),
            omega_field=np.full((n_cells, n_sps), omega_value, dtype=np.float64),
            nu_t=np.full((n_cells, n_sps), 0.5, dtype=np.float64),
            k_max=k_max, omega_max=omega_max,
        )
        return types.SimpleNamespace(
            turb_model_gpu=turb,
            freestream={"rho_inf": 1.225, "vel_inf": 33.33, "p_inf": 101325.0},
            mu_molecular=1.8e-5,
            _turbulence_intensity=0.01,
            _viscosity_ratio=5.0,
        )

    def test_healthy_high_turbulence_field_is_not_falsely_reset(self):
        from autoflowcfd.core.mpi.distributed_order_continuation import (
            _reset_turbulence_if_resumed_field_exploded,
        )
        solver = self._make_solver_stub(k_value=38.11, omega_value=28459.5)

        _reset_turbulence_if_resumed_field_exploded(solver)

        np.testing.assert_allclose(solver.turb_model_gpu.k_field, 38.11)
        np.testing.assert_allclose(solver.turb_model_gpu.omega_field, 28459.5)

    def test_genuinely_clamped_field_still_gets_reset(self):
        from autoflowcfd.core.mpi.distributed_order_continuation import (
            _reset_turbulence_if_resumed_field_exploded,
        )
        solver = self._make_solver_stub(k_value=555.44, omega_value=1e6)

        _reset_turbulence_if_resumed_field_exploded(solver)

        k_inf_expected = 1.5 * (33.33 * 0.01) ** 2
        assert not np.allclose(solver.turb_model_gpu.k_field, 555.44)
        np.testing.assert_allclose(solver.turb_model_gpu.k_field, k_inf_expected, rtol=1e-6)
        assert np.all(solver.turb_model_gpu.nu_t == 0.0)

    def test_partial_clamping_below_threshold_is_not_reset(self):
        from autoflowcfd.core.mpi.distributed_order_continuation import (
            _reset_turbulence_if_resumed_field_exploded,
        )
        solver = self._make_solver_stub(k_value=1.0, omega_value=100.0, n_cells=100, n_sps=8)

        flat = solver.turb_model_gpu.k_field.reshape(-1)
        n_clamp = max(1, int(0.05 * flat.size))
        flat[:n_clamp] = solver.turb_model_gpu.k_max

        _reset_turbulence_if_resumed_field_exploded(solver)

        assert np.any(np.isclose(solver.turb_model_gpu.k_field, 1.0))
