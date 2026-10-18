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
    只 mock 掉 GPU 粘性通量核（返回全零，隔离出 WMLES 修正项本身）。
    判据：本 rank 的修正与单机在全局网格上的修正逐位一致（壁面面完全
    属于 owner 单元所在的 rank）。"""

    @pytest.mark.parametrize("rank", [0, 1])
    def test_wmles_correction_matches_direct_cpu_call(self, rank, monkeypatch):
        from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
        from autoflowcfd.core.fr_solver.boundary import build_boundary_ghost_provider
        from autoflowcfd.core.gpu.distributed.gpu_distributed.compact_view import _CompactMeshDataView
        from autoflowcfd.core.turbulence.wall_model import EquilibriumWallModel
        import autoflowcfd.core.gpu.residual.gpu_viscous as gpu_viscous_mod

        order = 1
        mesh = _build_synthetic_mixed_mesh(order)
        ops = generate_fr_operators(order)
        n_sps = mesh.n_sps_per_cell
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
            turb_model_name="WMLES", wall_model=object(),
        )
        provider_global = build_boundary_ghost_provider(root_stub, bc_overrides={})

        cell_partition = np.array([0, 1, 0, 1], dtype=np.int32)
        partition = build_distributed_partition(fc, cell_partition, rank=rank, n_ranks=2)
        dist_fc = build_distributed_flat_face(mesh, ops, partition, cell_partition=cell_partition)
        compact_global_ids = dist_fc.compact_global_ids

        import copy as copy_mod
        provider_local = copy_mod.copy(provider_global)
        provider_local.group_code = provider_global.group_code[partition.local_faces]

        mesh_data = _prepare_compact_mesh_data(mesh, ops, compact_global_ids)
        mesh_data['n_prism'] = dist_fc.base_flat.n_prism

        compact_mesh_view = _CompactMeshDataView(mesh, compact_global_ids, dist_fc.base_flat.n_prism)

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
            wall_model=EquilibriumWallModel(),
            _compact_mesh_view=compact_mesh_view,
            U_extended_gpu=U_extended,
            partition=partition,
        )
        stub._permute_to_compact = lambda arr: arr[dist_fc.perm]
        stub._unpermute_from_compact = lambda arr: arr[dist_fc.inv_perm]
        _bind_turb_source(stub)

        residual_local = gd_mod.MultiGPUDistributedSolver.compute_viscous_residual_gpu(stub)

        # 参照值：单机在全局网格上的修正，取本 rank 的 local 单元
        expected_global = EquilibriumWallModel().residual_correction(
            np, mesh, get_flat_face_geometry(mesh, ops), provider_global, U, mu)
        assert expected_global is not None
        expected_local = expected_global[partition.local_cells]

        np.testing.assert_allclose(residual_local, expected_local, atol=1e-10)
