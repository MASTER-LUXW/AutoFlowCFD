"""分布式路径逐单元局部 CFL 步长的决定性验证（2026-09-14）。

## 这个测试要钉住什么

分布式路径此前用**全局固定 dt**、不做逐 cell 局部 CFL，源码里记作
"已接受的简化"。用户明确指出本项目不接受简化，本轮补齐
（`core/mpi/distributed_cfl.py`）。补齐的方式是构造一个 solver 视图、
调用**与单机完全同一个** `cfl.py::compute_local_time_step`，因此最强的
判据就是：

**分布式路径算出的"某个 rank 的 local cells 的 dt"，必须与单机路径在
同一网格上算出的、对应同一批全局单元的 dt 一致。**

这比"dt 都是正数"或"量级差不多"强得多——它能抓出紧凑索引空间重排
（"棱柱在前"）、面归属（分区边界面必须按内部面两侧累加）、halo 对齐、
几何量抽取（按 compact_global_ids 重排 jacobians/volumes）上的任何错误。
而这几处恰好是本项目分布式代码反复出过真实 bug 的地方（见
`distributed_flat_face.py` 与 `distributed_compute.py` 的 bug 记录）。

## 为什么可以在本机跑

本机没有 mpi4py，测不了真正的点对点通信；但局部 dt 的计算逻辑完全不
依赖通信——它只需要一份"local+halo 扩展状态"。所以沿用本项目既有的
假 halo 交换范式（见 `test_distributed_compute_residual.py` 模块文档）：
直接从全局真值里按 `local_cells + halo_cells` 切出扩展状态，其余全部
走真实生产代码。

## 非均匀流场是必须的

均匀自由流场下所有单元的波速相同，dt 的差异只来自几何——那样即使
owner/neighbor 索引整体错位也可能"看起来对"。所以用一个随空间变化的
流场：速度/密度/压力都逐单元不同，任何索引错位都会让某些单元的谱半径
求和读到错误邻居、dt 立刻对不上。
"""

import numpy as np
import pytest

from autoflowcfd.core.fr_residual.inviscid import primitive_to_conserved
from autoflowcfd.core.fr_solver.cfl import compute_local_time_step
from autoflowcfd.core.mpi.distributed_cfl import (
    DistributedCFLView,
    compute_distributed_local_time_step,
)
from autoflowcfd.core.mpi.distributed_flat_face import build_distributed_flat_face
from autoflowcfd.core.mpi.partition import build_distributed_partition
from autoflowcfd.fr.operators import generate_fr_operators
from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh

FREESTREAM = {"rho_inf": 1.225, "vel_inf": 30.0, "p_inf": 101325.0, "mach_ref": 0.1}
MU = 1.8e-5


@pytest.fixture(scope="module")
def mesh_ops():
    order = 2
    mesh = _build_synthetic_mixed_mesh(order)
    ops = generate_fr_operators(order)
    return mesh, ops


def _nonuniform_state(mesh):
    """逐单元不同的流场——均匀场无法判别索引错位（见模块文档）。"""
    n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
    rng = np.random.default_rng(20260914)
    U = np.empty((n_cells, n_sps, 5))
    for c in range(n_cells):
        rho = 1.225 * (1.0 + 0.15 * rng.random())
        u = 30.0 * (1.0 + 0.4 * rng.random())
        v = 5.0 * (rng.random() - 0.5)
        w = -3.0 * (rng.random() - 0.5)
        p = 101325.0 * (1.0 + 0.1 * rng.random())
        U[c, :, :] = primitive_to_conserved(np.array([rho, u, v, w, p]))
    return U


def _conserved_to_primitive_field(U):
    gamma = 1.4
    rho = np.maximum(U[..., 0], 1e-30)
    u, v, w = U[..., 1] / rho, U[..., 2] / rho, U[..., 3] / rho
    q2 = u * u + v * v + w * w
    p = (gamma - 1.0) * (U[..., 4] - 0.5 * rho * q2)
    return np.stack([rho, u, v, w, p], axis=-1)


class _SingleMachineView:
    """单机参照：用同一个 `compute_local_time_step`，喂完整全局网格。

    刻意也走"视图"而不是构造一个真 FRSolver：构造 FRSolver 会额外初始化
    湍流/边界/算子等一大堆与本测试无关的东西，而本测试要比较的恰恰是
    **同一个函数**在两种索引空间下的输出，视图能把变量控制得最干净。
    """

    class _Mesh:
        def __init__(self, mesh):
            self._m = mesh
            self.face_connectivity = mesh.face_connectivity
            self.jacobians = mesh.jacobians
            self.n_sps_per_cell = mesh.n_sps_per_cell
            self.n_cells = mesh.n_cells

        def get_all_cell_volumes(self):
            return self._m.get_all_cell_volumes()

    class _State:
        def __init__(self, U, Q):
            self.U, self.Q = U, Q

    def __init__(self, mesh, U, order, low_mach_precond_enabled=False):
        self.mesh = self._Mesh(mesh)
        self.state = self._State(U, _conserved_to_primitive_field(U))
        self.mu_molecular = MU
        self.freestream = FREESTREAM
        self._cfl_controller = None
        self.current_order = order
        self.low_mach_precond_enabled = low_mach_precond_enabled
        self._cache = None

    def _get_metric_flux_scale(self):
        if self._cache is not None:
            return self._cache
        n_cells, n_sps = self.mesh.n_cells, self.mesh.n_sps_per_cell
        det = self.mesh.jacobians["det_jacs"].reshape(n_cells, n_sps)
        inv = self.mesh.jacobians["inv_jacs"].reshape(n_cells, n_sps, 3, 3)
        adj = det[..., None, None] * inv
        self._cache = np.sum(np.linalg.norm(adj, axis=-1), axis=-1)
        return self._cache

    def _get_turbulent_viscosity_field(self):
        return None


def _distributed_dt_for_rank(mesh, ops, U, rank, n_ranks, order, **kw):
    """走真实生产代码算出某个 rank 的 local cells 的 dt + 对应全局单元号。"""
    fc = mesh.face_connectivity
    cell_partition = np.arange(mesh.n_cells) % n_ranks
    partition = build_distributed_partition(fc, cell_partition, rank=rank, n_ranks=n_ranks)
    dist_fc = build_distributed_flat_face(mesh, ops, partition,
                                          cell_partition=cell_partition)

    # 假 halo 交换：扩展状态直接从全局真值切片（原生排列：local 在前、
    # halo 在后），再按 dist_fc.perm 重排到"棱柱在前"紧凑排列——这正是
    # 生产路径 (gpu_distributed / distributed_compute) 的做法。
    n_halo = partition.n_halo
    native_ids = (np.concatenate([partition.local_cells, partition.halo_cells])
                  if n_halo > 0 else partition.local_cells)
    U_native = U[native_ids]
    U_compact = U_native[dist_fc.perm]
    Q_compact = _conserved_to_primitive_field(U_compact)

    n_sps = mesh.n_sps_per_cell
    cgi = dist_fc.compact_global_ids
    det = mesh.jacobians["det_jacs"].reshape(mesh.n_cells, n_sps)[cgi]
    inv = mesh.jacobians["inv_jacs"].reshape(mesh.n_cells, n_sps, 3, 3)[cgi]
    jac = {"det_jacs": det, "inv_jacs": inv}
    vols = mesh.get_all_cell_volumes()[cgi]

    dt = compute_distributed_local_time_step(
        U_compact, Q_compact, dist_fc, mesh,
        jacobians=jac, cell_volumes=vols,
        n_local_cells=partition.n_local_cells,
        mu_molecular=MU, freestream=FREESTREAM,
        current_order=order, **kw)
    # 返回的 dt 已经换回**原生**排列并切出 local 段，所以对应的全局单元号
    # 就是 partition.local_cells 自身顺序——**不是** cgi[:n_local]
    # （紧凑排列把 local 与 halo 混在一起重排了，见 distributed_cfl.py
    # 模块文档"切片必须先换回原生排列"）。
    return dt, partition.local_cells


class TestDistributedDtMatchesSingleMachine:
    """最强的一条：逐单元与单机路径比对。"""

    @pytest.mark.parametrize("n_ranks", [2, 3])
    def test_matches_single_machine_per_cell(self, mesh_ops, n_ranks):
        mesh, ops = mesh_ops
        order = 2
        U = _nonuniform_state(mesh)

        ref = compute_local_time_step(_SingleMachineView(mesh, U, order))
        assert ref.shape == (mesh.n_cells, mesh.n_sps_per_cell)
        assert np.all(ref > 0)

        seen = np.zeros(mesh.n_cells, dtype=bool)
        for rank in range(n_ranks):
            dt, gids = _distributed_dt_for_rank(mesh, ops, U, rank, n_ranks, order)
            assert dt.shape[0] == len(gids)
            seen[gids] = True
            got = dt
            exp = ref[gids]
            rel = np.abs(got - exp) / np.maximum(np.abs(exp), 1e-300)
            assert rel.max() <= 1e-12, (
                f"rank {rank}/{n_ranks}: 分布式 dt 与单机不一致，最大相对差 "
                f"{rel.max():.3e}\n  分布式={got.ravel()[:6]}\n  单机  ={exp.ravel()[:6]}")
        assert seen.all(), "各 rank 的 local cells 并集没有覆盖全部单元"

    def test_matches_with_preconditioner_enabled(self, mesh_ops):
        """开启低马赫数预处理后（dt 按预处理波速放大）同样必须一致——
        这条同时验证了预处理在分布式路径上真的接上了。"""
        mesh, ops = mesh_ops
        order = 2
        U = _nonuniform_state(mesh)
        ref_mean, ref_phys = compute_local_time_step(
            _SingleMachineView(mesh, U, order, low_mach_precond_enabled=True),
            return_physical_too=True)
        assert np.all(ref_mean >= ref_phys - 1e-30), "预处理后的 dt 不应小于物理 dt"
        assert ref_mean.max() > ref_phys.max() * 1.5, (
            "本测试需要预处理真的放大了 dt 才有判别力")

        for rank in range(2):
            out, gids = _distributed_dt_for_rank(
                mesh, ops, U, rank, 2, order,
                low_mach_precond_enabled=True, return_physical_too=True)
            dt_mean, dt_phys = out
            for name, got, exp in (("mean_flow", dt_mean, ref_mean[gids]),
                                   ("physical", dt_phys, ref_phys[gids])):
                rel = np.abs(got - exp) / np.maximum(np.abs(exp), 1e-300)
                assert rel.max() <= 1e-12, (
                    f"rank {rank} 的 {name} dt 与单机不一致：{rel.max():.3e}")

    def test_turbulent_viscosity_only_needs_local_part(self, mesh_ops):
        """涡粘场只给 local 部分（halo 段填 0）不影响 local cells 的 dt——
        这条钉住模块文档里"dt_visc 是纯逐单元量、不需要为 mu_t 再加一次
        halo 交换"这个论证。"""
        mesh, ops = mesh_ops
        order = 2
        U = _nonuniform_state(mesh)
        n_sps = mesh.n_sps_per_cell
        rng = np.random.default_rng(5)
        mu_t_global = 1e-3 * (1.0 + rng.random((mesh.n_cells, n_sps)))

        class _ViewWithMut(_SingleMachineView):
            def _get_turbulent_viscosity_field(self_inner):
                return mu_t_global

        ref = compute_local_time_step(_ViewWithMut(mesh, U, order))
        for rank in range(2):
            fc = mesh.face_connectivity
            cp_ = np.arange(mesh.n_cells) % 2
            part = build_distributed_partition(fc, cp_, rank=rank, n_ranks=2)
            dist_fc = build_distributed_flat_face(mesh, ops, part, cell_partition=cp_)
            mu_t_local = mu_t_global[part.local_cells]
            dt, gids = _distributed_dt_for_rank(
                mesh, ops, U, rank, 2, order, mu_t_local=mu_t_local)
            rel = np.abs(dt - ref[gids]) / np.maximum(np.abs(ref[gids]), 1e-300)
            assert rel.max() <= 1e-12, f"rank {rank}: 带涡粘的 dt 不一致 {rel.max():.3e}"


class TestFaceClassificationSemantics:
    """分区边界面必须按**内部面**处理（两侧都累加谱半径）。

    如果误当成物理边界面，owner 侧会丢掉该面的邻居贡献、谱半径偏小、
    dt 被**高估**——这正是"局部 CFL 在分布式路径上写错"最可能的形态，
    而且在均匀流场下不容易暴露。上面逐单元比对已经覆盖了它，这里再单独
    钉一条更直接的：分区边界面在视图里不得被标成边界。
    """

    def test_partition_boundary_is_not_treated_as_boundary(self, mesh_ops):
        mesh, ops = mesh_ops
        cp_ = np.arange(mesh.n_cells) % 2
        part = build_distributed_partition(mesh.face_connectivity, cp_, rank=0, n_ranks=2)
        dist_fc = build_distributed_flat_face(mesh, ops, part, cell_partition=cp_)
        assert dist_fc.partition_boundary_mask.any(), (
            "这个分区方式下没有分区边界面，本测试失去判别力")

        n_sps = mesh.n_sps_per_cell
        cgi = dist_fc.compact_global_ids
        U = _nonuniform_state(mesh)
        U_compact = U[np.concatenate([part.local_cells, part.halo_cells])
                      if part.n_halo > 0 else part.local_cells][dist_fc.perm]
        view = DistributedCFLView(
            U_compact, _conserved_to_primitive_field(U_compact), dist_fc, mesh,
            mu_molecular=MU, freestream=FREESTREAM, cfl_controller=None,
            current_order=2, low_mach_precond_enabled=False,
            n_local_cells=part.n_local_cells,
            jacobians={
                "det_jacs": mesh.jacobians["det_jacs"].reshape(mesh.n_cells, n_sps)[cgi],
                "inv_jacs": mesh.jacobians["inv_jacs"].reshape(
                    mesh.n_cells, n_sps, 3, 3)[cgi],
            },
            cell_volumes=mesh.get_all_cell_volumes()[cgi],
        )
        is_bnd = view.mesh.face_connectivity.is_boundary
        assert not np.any(is_bnd & dist_fc.partition_boundary_mask), (
            "分区边界面被当成了物理边界面——owner 侧会丢掉邻居的谱半径贡献，"
            "dt 被高估")
        # 物理边界面必须仍然是边界
        assert np.all(is_bnd[dist_fc.physical_boundary_mask]), (
            "物理边界面没有被标成边界")

    def test_halo_owner_faces_are_treated_as_interior(self, mesh_ops):
        """第四类面（"halo"：本 rank 只持有 neighbor 侧）也必须按内部面处理。

        这一类在 `FaceClassification` 里由 `halo_owner_mask` 覆盖，**三个
        掩码（interior/partition/physical）全为 False**——实测 rank 1 的
        全局面 3/9/12 正是如此，而且 `_remap_mixed_partner` 会刻意把
        **local** 单元放到 owner 槽位、远端单元放到 neighbor 槽位。
        历史上这一类面曾经根本没被选进 `local_faces`（见 partition.py 的
        `halo_owner_mask` bug 记录），是分布式残差长期丢贡献的根源。

        已用破坏性验证确认这一类真的被逐单元比对覆盖：把它们当成边界面
        后，rank 1 的 dt 与单机偏差 33%（`test_matches_single_machine_
        per_cell[2]` 失败）。本用例把这个结构性事实直接钉住，避免将来
        有人"顺手"把非 interior 的面一律当边界。
        """
        mesh, ops = mesh_ops
        cp_ = np.arange(mesh.n_cells) % 2
        part = build_distributed_partition(mesh.face_connectivity, cp_,
                                           rank=1, n_ranks=2)
        dist_fc = build_distributed_flat_face(mesh, ops, part, cell_partition=cp_)
        halo_cat = (~dist_fc.interior_mask & ~dist_fc.partition_boundary_mask
                    & ~dist_fc.physical_boundary_mask)
        assert halo_cat.any(), (
            "rank 1 上没有出现第四类(halo)面，本用例失去判别力——"
            "上游面分类若改变语义请更新这里")
        assert np.all(dist_fc.neighbor_cell_local[halo_cat] >= 0), (
            "第四类面应当有真实邻居（在 halo 里）")

        n_sps = mesh.n_sps_per_cell
        cgi = dist_fc.compact_global_ids
        U = _nonuniform_state(mesh)
        native = (np.concatenate([part.local_cells, part.halo_cells])
                  if part.n_halo > 0 else part.local_cells)
        U_compact = U[native][dist_fc.perm]
        view = DistributedCFLView(
            U_compact, _conserved_to_primitive_field(U_compact), dist_fc, mesh,
            mu_molecular=MU, freestream=FREESTREAM, cfl_controller=None,
            current_order=2, low_mach_precond_enabled=False,
            n_local_cells=part.n_local_cells,
            jacobians={
                "det_jacs": mesh.jacobians["det_jacs"].reshape(mesh.n_cells, n_sps)[cgi],
                "inv_jacs": mesh.jacobians["inv_jacs"].reshape(
                    mesh.n_cells, n_sps, 3, 3)[cgi],
            },
            cell_volumes=mesh.get_all_cell_volumes()[cgi],
        )
        assert not np.any(view.mesh.face_connectivity.is_boundary[halo_cat]), (
            "第四类(halo)面被当成了边界面——本 rank 的 local 单元会丢掉这个面"
            "的谱半径贡献，dt 被高估")
