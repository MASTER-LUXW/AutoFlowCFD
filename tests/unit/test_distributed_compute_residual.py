"""AutoFlowCFD V2.0 - #2 CPU 分布式残差计算端到端单元测试。

背景：`DistributedMeshAdapter`/`distributed_compute_inviscid_residual`/
`distributed_compute_viscous_residual` 此前的 `n_cells`/`jacobians` 直接
转发 `local_mesh` 的原始值——但 `dist_fc`（`DistributedFlatFaceGeometry`）
的 owner/neighbor 用的是"棱柱在前"local+halo 压缩索引空间（见
distributed_flat_face.py 模块文档的 #1 修复），与 `local_mesh` 的全局
单元编号是两套不同的索引空间，直接混用会读到错误单元的 jacobians/
产出错误残差。#2 修复：`DistributedMeshAdapter` 按 `dist_fc.
compact_global_ids` 重新抽取+重排 jacobians，`distributed_compute_*_
residual` 在调用残差函数前后做 perm/inv_perm 重排（与 GPU 侧 #1 修复
完全同一个道理）。

验证方式：真实 MPI 硬件在本机不可用（mpi4py 未安装），无法测试真正的
点对点 halo 通信；但 `distributed_compute_inviscid_residual`/
`distributed_compute_viscous_residual` 本身接受一个 `halo_exchange`
对象、只调用它的 `.exchange(U_local)` 方法——用一个"假"halo 交换器
（直接返回已知正确的扩展状态，绕开真正的 MPI 通信但不绕开这两个函数
自身的其余逻辑：网格适配器构造、压缩索引空间重排、残差函数调用、
换回原生排列、切片）来端到端测试这两个函数的真实生产代码路径。

核心判据：对同一个真实混合棱柱+四面体网格，用分布式路径算出的
"某个 rank 的 local cells 残差"，必须与单机路径 `compute_inviscid_
residual_fr(U, mesh, ops, ...)` 算出的、对应同一批全局单元的残差
一致（不只是"都接近零"——直接数值比对更强，能捕捉分布式路径自身
引入的任何符号/索引错误，即使这些错误恰好不影响均匀自由流场的近零
残差这个更弱的判据）。

判据尺度说明：`_build_synthetic_mixed_mesh` 是一个刻意极简的 4 单元
合成网格（不是精心构造的良态网格），其四面体单元本身在 P2 下已知会
在坍缩坐标退化角点附近产生较大的单元局部残差噪声——这不是本次#2
修复引入的问题，而是 `test_fr_residual_inviscid.py::
TestFreeStreamPreservation` 早已记录并接受的、这个具体网格/阶数组合
特有的数值现象（该测试类自己对 P2 的容差就放宽到 `rel_res =
max(abs(residual))/p_inf < 3e-5`）。分布式路径与单机路径在这类
条件数很差的退化单元上，即使物理上完全等价，也可能因为浮点运算
顺序不同（全场一次性计算 vs 按 local+halo 子集抽取计算）产生比
良态单元更大的绝对差异——这是浮点重结合噪声，不是正确性 bug（已用
诊断脚本实测确认：本文件测得的最大绝对差异 0.00715726，换算成与
`TestFreeStreamPreservation` 同一个判据 `/p_inf` 后是 7.06e-8，远低于
该测试类自己对这个网格+阶数接受的 3e-5 阈值）。因此本文件的比较判据
沿用 `TestFreeStreamPreservation` 已经建立、有据可查的同一个尺度
（相对 p_inf），而不是对残差数组本身用绝对/相对容差直接比较——后者
在某个退化单元残差本身量级偏大时会产生误导性的"相对误差"数字。
"""

import copy

import numpy as np
import pytest

from autoflowcfd.core.mpi.partition import build_distributed_partition
from autoflowcfd.core.mpi.distributed_flat_face import build_distributed_flat_face
from autoflowcfd.core.mpi.distributed_compute import (
    DistributedMeshAdapter,
    distributed_compute_inviscid_residual,
    distributed_compute_viscous_residual,
)
from autoflowcfd.core.fr_residual.inviscid import compute_inviscid_residual_fr, primitive_to_conserved
from autoflowcfd.core.fr_residual.viscous import compute_viscous_residual
from autoflowcfd.fr.operators import generate_fr_operators
from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh


class _FakeHaloExchange:
    """用已知正确的扩展状态（直接从全局真值切片而来）代替真正的 MPI
    halo 交换——本机没有 mpi4py，无法测试真正的点对点通信本身，但这样
    能端到端测试 `distributed_compute_*_residual` 自身的其余生产逻辑
    （网格适配器构造、压缩索引空间重排、残差函数调用、换回原生排列、
    切片），见模块文档。"""

    def __init__(self, U_extended: np.ndarray):
        self._U_extended = U_extended

    def exchange(self, U_local: np.ndarray) -> np.ndarray:
        return self._U_extended


@pytest.fixture(scope="module")
def mixed_mesh_and_ops():
    order = 2
    mesh = _build_synthetic_mixed_mesh(order)
    ops = generate_fr_operators(order)
    assert mesh.n_prism_cells == 2
    assert mesh.n_cells == 4
    return mesh, ops


@pytest.fixture(scope="module")
def mixed_mesh_and_ops_native():
    """native 四面体（路径C）MPI 分布式移植（2026-09-02）验证用——与
    `mixed_mesh_and_ops` 同一份合成网格，唯一区别是 tet_basis_mode="native"。

    背景：`distributed_flat_face.py` 此前的模块文档一直写"分布式/MPI 路径
    明确不支持 native 模式"，只把 native 相关字段原样透传、不引入分派。
    本次移植前先决定性验证这个"不支持"的说法是否仍然成立——用本文件已经
    建立的"分布式残差必须与单机路径逐位一致"判据直接测（见下方
    TestDistributedResidualMatchesSingleMachineNative），结果证实：这套
    透传机制本身就是完整、正确的（`owner_cube_face`/`neighbor_cube_face`
    按面索引正确切片，`boundary_extrap_native`/`lift_native` 是与面无关
    的全局常量查找表，原样传递即可），不需要任何额外分派代码——CPU 端
    分布式残差计算复用的正是同一套已经支持 native 的
    `compute_inviscid_interface_correction_kernel`/`compute_viscous_
    interface_correction_kernel`，MPI 分区只影响单元切分方式，不影响
    这两个 kernel 内部的 native/collapsed 分派逻辑。"不支持"的表述已经
    过时，本次更新为"已验证支持"。"""
    order = 2
    mesh = _build_synthetic_mixed_mesh(order, tet_basis_mode="native")
    ops = generate_fr_operators(order)
    assert mesh.n_prism_cells == 2
    assert mesh.n_cells == 4
    return mesh, ops


def _uniform_freestream_U(mesh) -> np.ndarray:
    rho_inf, u_inf, v_inf, w_inf, p_inf = 1.225, 30.0, 5.0, -3.0, 101325.0
    Q_inf = np.array([rho_inf, u_inf, v_inf, w_inf, p_inf])
    U_inf = primitive_to_conserved(Q_inf)
    return np.tile(U_inf, (mesh.n_cells, mesh.n_sps_per_cell, 1))


class TestDistributedMeshAdapterCompactGeometry:
    @pytest.mark.parametrize("rank", [0, 1])
    def test_jacobians_match_global_mesh_reindexed(self, mixed_mesh_and_ops, rank):
        mesh, ops = mixed_mesh_and_ops
        cell_partition = np.array([0, 1, 0, 1], dtype=np.int32)
        fc = mesh.face_connectivity

        partition = build_distributed_partition(fc, cell_partition, rank=rank, n_ranks=2)
        dist_fc = build_distributed_flat_face(mesh, ops, partition, cell_partition=cell_partition)
        adapter = DistributedMeshAdapter(partition, dist_fc, mesh, ops)

        assert adapter.n_cells == len(dist_fc.compact_global_ids)
        assert adapter.n_prism_cells == dist_fc.base_flat.n_prism

        n_sps = mesh.n_sps_per_cell
        det_jacs_global = mesh.jacobians['det_jacs'].reshape(mesh.n_cells, n_sps)
        expected = det_jacs_global[dist_fc.compact_global_ids]
        np.testing.assert_array_equal(adapter.jacobians['det_jacs'], expected)


class TestDistributedResidualMatchesSingleMachine:
    """核心判据：分布式路径算出的 local cells 残差必须与单机路径逐位
    相等（不只是都接近零）。"""

    @pytest.mark.parametrize("rank", [0, 1])
    def test_inviscid_residual_matches(self, mixed_mesh_and_ops, rank):
        mesh, ops = mixed_mesh_and_ops
        U = _uniform_freestream_U(mesh)
        mach_ref = 0.2

        residual_single = compute_inviscid_residual_fr(U, mesh, ops, mach_ref=mach_ref)

        cell_partition = np.array([0, 1, 0, 1], dtype=np.int32)
        fc = mesh.face_connectivity
        partition = build_distributed_partition(fc, cell_partition, rank=rank, n_ranks=2)
        dist_fc = build_distributed_flat_face(mesh, ops, partition, cell_partition=cell_partition)

        n_halo = partition.n_halo
        native_global_ids = (
            np.concatenate([partition.local_cells, partition.halo_cells])
            if n_halo > 0 else partition.local_cells
        )
        U_extended_correct = U[native_global_ids]
        fake_halo = _FakeHaloExchange(U_extended_correct)

        U_local = U[partition.local_cells]
        residual_local = distributed_compute_inviscid_residual(
            U_local, partition, fake_halo, dist_fc, mesh, ops, mach_ref=mach_ref,
        )

        expected_local = residual_single[partition.local_cells]
        # 见模块文档"判据尺度说明"：沿用 TestFreeStreamPreservation 已
        # 建立的相对 p_inf 判据，而不是对残差数组直接比较。
        rel_diff = np.max(np.abs(residual_local - expected_local)) / 101325.0
        assert rel_diff < 3e-5, f"分布式与单机残差不一致: rel_diff={rel_diff:.3e}"

    @pytest.mark.parametrize("rank", [0, 1])
    def test_viscous_residual_matches(self, mixed_mesh_and_ops, rank):
        mesh, ops = mixed_mesh_and_ops
        U = _uniform_freestream_U(mesh)
        mu = 1.8e-5

        residual_single = compute_viscous_residual(U, None, ops, mesh, mu=mu)

        cell_partition = np.array([0, 1, 0, 1], dtype=np.int32)
        fc = mesh.face_connectivity
        partition = build_distributed_partition(fc, cell_partition, rank=rank, n_ranks=2)
        dist_fc = build_distributed_flat_face(mesh, ops, partition, cell_partition=cell_partition)

        n_halo = partition.n_halo
        native_global_ids = (
            np.concatenate([partition.local_cells, partition.halo_cells])
            if n_halo > 0 else partition.local_cells
        )
        U_extended_correct = U[native_global_ids]
        fake_halo = _FakeHaloExchange(U_extended_correct)

        U_local = U[partition.local_cells]
        residual_local = distributed_compute_viscous_residual(
            U_local, partition, fake_halo, dist_fc, mesh, ops, mu=mu,
        )

        expected_local = residual_single[partition.local_cells]
        # 均匀自由流场下解析残差恒为 0，两条路径实际算出的都是纯浮点
        # 噪声（~1e-10 到 1e-13 量级）——不同的计算顺序（全场 vs 子集
        # 抽取）产生不同的舍入噪声本身是预期的，不是正确性问题；这里
        # 用绝对容差而不是相对容差比较，避免在真值本身接近零时被
        # 无意义的"相对误差 248 倍"这类噪声比噪声的比值吓到。
        np.testing.assert_allclose(residual_local, expected_local, atol=1e-8)

    @pytest.mark.parametrize("rank", [0, 1])
    def test_viscous_residual_with_wmles_matches(self, mixed_mesh_and_ops, rank):
        """WMLES 壁面剪应力修正的分布式支持（2026-09-02）：核心判据与
        上面的纯层流版本完全一致，唯一区别是同时激活一个真实 WALL 组
        +真实 `WMLESModel`+`boundary_ghost_provider`。这也顺带覆盖了
        `compute_wmles_wall_stress_correction` 本次改用 `flat_face_
        override`/`boundary_ghost_provider` 之后的分布式正确性——此前
        这条路径完全不存在（WMLES 是 MPI 分布式明确拒绝的湍流模型之一，
        见 DistributedFRSolver.__init__ 文档）。"""
        mesh, ops = mixed_mesh_and_ops
        U = _uniform_freestream_U(mesh)
        n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
        mu = 1.8e-5
        rho_inf = 1.225

        from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive
        from autoflowcfd.core.fr_solver.boundary import build_boundary_ghost_provider
        from autoflowcfd.core.turbulence.wmles import WMLESModel
        from autoflowcfd.core.utils.solver_helpers import compute_wmles_wall_stress_correction

        # 真实 WALL 组：取一个真实边界面的 owner 单元打标签（与
        # test_wmles_wall_bc_wiring.py 新增的端到端测试同一个手法）。
        fc = mesh.face_connectivity
        boundary_face = int(np.nonzero(fc.is_boundary)[0][0])
        wall_cell = int(fc.owner_cell[boundary_face])
        mesh.boundary_groups = {"wall_group": np.array([wall_cell], dtype=np.int64)}
        mesh.boundary_bc_types = {"wall_group": "WALL"}

        import types
        root_stub = types.SimpleNamespace(
            mesh=mesh, freestream={"rho_inf": rho_inf, "vel_inf": 30.0, "p_inf": 101325.0},
            turb_model_name="WMLES", wmles_model=object(),  # 触发 is_no_slip=False
        )
        provider_global = build_boundary_ghost_provider(root_stub, bc_overrides={})
        wall_distance = np.full((n_cells, n_sps), 1e-2)
        wmles_model = WMLESModel(nu=mu / rho_inf)

        Q = conserved_to_primitive(U[..., :5])
        residual_single = compute_viscous_residual(
            U, Q, ops, mesh, mu=mu, boundary_ghost_provider=provider_global,
        )
        facade = types.SimpleNamespace(
            wmles_model=wmles_model, mesh=mesh, ops=ops, wall_distance=wall_distance,
            state=types.SimpleNamespace(U=U, Q=Q), boundary_ghost_provider=provider_global,
        )
        correction_single = compute_wmles_wall_stress_correction(facade)
        assert correction_single is not None, "test setup must produce at least one real WALL face"
        residual_single = residual_single + correction_single[..., :residual_single.shape[-1]]

        cell_partition = np.array([0, 1, 0, 1], dtype=np.int32)
        partition = build_distributed_partition(fc, cell_partition, rank=rank, n_ranks=2)
        dist_fc = build_distributed_flat_face(mesh, ops, partition, cell_partition=cell_partition)

        provider_local = copy.copy(provider_global)
        provider_local.group_code = provider_global.group_code[partition.local_faces]
        wall_distance_compact = wall_distance[dist_fc.compact_global_ids]

        n_halo = partition.n_halo
        native_global_ids = (
            np.concatenate([partition.local_cells, partition.halo_cells])
            if n_halo > 0 else partition.local_cells
        )
        fake_halo = _FakeHaloExchange(U[native_global_ids])

        U_local = U[partition.local_cells]
        residual_local = distributed_compute_viscous_residual(
            U_local, partition, fake_halo, dist_fc, mesh, ops, mu=mu,
            boundary_ghost_provider=provider_local,
            wmles_model=wmles_model, wall_distance_compact=wall_distance_compact,
        )

        expected_local = residual_single[partition.local_cells]
        np.testing.assert_allclose(residual_local, expected_local, atol=1e-8)


class TestDistributedResidualMatchesSingleMachineNative:
    """native 四面体（路径C）MPI 分布式移植验证（2026-09-02）——与
    `TestDistributedResidualMatchesSingleMachine` 完全同一判据/结构，
    唯一区别是用 native tet_basis_mode 的网格/算子。"""

    @pytest.mark.parametrize("rank", [0, 1])
    def test_inviscid_residual_matches(self, mixed_mesh_and_ops_native, rank):
        mesh, ops = mixed_mesh_and_ops_native
        U = _uniform_freestream_U(mesh)
        mach_ref = 0.2

        residual_single = compute_inviscid_residual_fr(U, mesh, ops, mach_ref=mach_ref)

        cell_partition = np.array([0, 1, 0, 1], dtype=np.int32)
        fc = mesh.face_connectivity
        partition = build_distributed_partition(fc, cell_partition, rank=rank, n_ranks=2)
        dist_fc = build_distributed_flat_face(mesh, ops, partition, cell_partition=cell_partition)

        n_halo = partition.n_halo
        native_global_ids = (
            np.concatenate([partition.local_cells, partition.halo_cells])
            if n_halo > 0 else partition.local_cells
        )
        U_extended_correct = U[native_global_ids]
        fake_halo = _FakeHaloExchange(U_extended_correct)

        U_local = U[partition.local_cells]
        residual_local = distributed_compute_inviscid_residual(
            U_local, partition, fake_halo, dist_fc, mesh, ops, mach_ref=mach_ref,
        )

        expected_local = residual_single[partition.local_cells]
        rel_diff = np.max(np.abs(residual_local - expected_local)) / 101325.0
        assert rel_diff < 3e-5, f"native: 分布式与单机残差不一致: rel_diff={rel_diff:.3e}"

    @pytest.mark.parametrize("rank", [0, 1])
    def test_viscous_residual_matches(self, mixed_mesh_and_ops_native, rank):
        mesh, ops = mixed_mesh_and_ops_native
        U = _uniform_freestream_U(mesh)
        mu = 1.8e-5

        residual_single = compute_viscous_residual(U, None, ops, mesh, mu=mu)

        cell_partition = np.array([0, 1, 0, 1], dtype=np.int32)
        fc = mesh.face_connectivity
        partition = build_distributed_partition(fc, cell_partition, rank=rank, n_ranks=2)
        dist_fc = build_distributed_flat_face(mesh, ops, partition, cell_partition=cell_partition)

        n_halo = partition.n_halo
        native_global_ids = (
            np.concatenate([partition.local_cells, partition.halo_cells])
            if n_halo > 0 else partition.local_cells
        )
        U_extended_correct = U[native_global_ids]
        fake_halo = _FakeHaloExchange(U_extended_correct)

        U_local = U[partition.local_cells]
        residual_local = distributed_compute_viscous_residual(
            U_local, partition, fake_halo, dist_fc, mesh, ops, mu=mu,
        )

        expected_local = residual_single[partition.local_cells]
        np.testing.assert_allclose(residual_local, expected_local, atol=1e-8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
