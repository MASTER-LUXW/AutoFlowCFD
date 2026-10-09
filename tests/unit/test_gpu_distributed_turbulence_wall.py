"""多 GPU 分布式湍流：壁面距离与 WMLES 壁面模型（从 test_gpu_distributed_turbulence.py 拆出，背景见该文件模块文档）。"""

import types
import numpy as np
import pytest

from autoflowcfd.fr.operators import generate_fr_operators
from autoflowcfd.core.mpi.partition import build_distributed_partition
from autoflowcfd.core.mpi.distributed_flat_face import build_distributed_flat_face
from autoflowcfd.core.gpu.distributed.gpu_distributed_init import _GPUDistributedInitMixin
import autoflowcfd.core.gpu.distributed.gpu_distributed as gd_mod

from tests.unit._wall_source import synthetic_wall_source
from tests.unit._patch_pkg import patch_pkg_attr
from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh
from tests.unit._gpu_distributed_turbulence_common import (
    _bind_turb_source,
    _nonuniform_state,
    _prepare_compact_mesh_data,
)
from tests.unit._gpu_distributed_turbulence_common import _patch_get_cupy  # noqa: F401  autouse 夹具：进入本模块命名空间才生效
from tests.unit._boundary_groups import tag_wall_cells


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
        tag_wall_cells(mesh, [wall_cell])

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
        _bind_turb_source(stub)

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
