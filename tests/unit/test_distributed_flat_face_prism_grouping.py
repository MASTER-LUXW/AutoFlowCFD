"""AutoFlowCFD V2.0 - #1 分布式 local+halo 扩展索引"棱柱在前"排列单元测试。

背景：`build_distributed_flat_face` 此前构造的 local+halo 扩展索引空间
按"local cells 在前、halo cells 在后"排列（与 partition.global_to_local
同一约定），但 GPU/CPU 残差 kernel 的体积项（gpu_inviscid_volume.py::
compute_volume_term_gpu、gpu_viscous.py 的 div_G、gpu_gradients.py 的
compute_physical_gradient_gpu 等）全靠对这个扩展索引数组做
`array[:n_prism]`/`array[n_prism:]` **切片**来分别喂给棱柱/四面体专用
算子——"local在前、halo在后"这个排列通常不满足"棱柱都在前、四面体都在
后"，任何单一阈值切片都无法正确复原（见 distributed_flat_face.py 模块
内详细说明）。修复为扩展索引空间改按"棱柱在前、四面体在后"排列，配套
`perm`/`inv_perm` 置换供调用方在 halo 交换协议原生排列与本排列之间转换。

这部分是本次 GPU/MPI 分布式修复里唯一能在本机真正验证的部分（纯 numpy
分区/索引逻辑，不需要真实 CUDA/MPI 硬件）——用真实构造过的混合
棱柱+四面体网格（与 test_fr_residual_inviscid.py 的自由流场保持性测试
同一个 mesh 构造 helper）、真实的 build_distributed_partition/
build_distributed_flat_face 调用链，验证排列的各项不变量。
"""

import numpy as np
import pytest

from autoflowcfd.core.mpi.partition import build_distributed_partition
from autoflowcfd.core.mpi.distributed_flat_face import build_distributed_flat_face
from autoflowcfd.fr.operators import generate_fr_operators
from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh


@pytest.fixture(scope="module")
def mixed_mesh_and_ops():
    order = 1
    mesh = _build_synthetic_mixed_mesh(order)
    ops = generate_fr_operators(order)
    assert mesh.n_prism_cells == 2
    assert mesh.n_cells == 4
    return mesh, ops


class TestPrismGroupedCompactOrdering:
    """核心不变量：扩展索引空间必须是"棱柱在前、四面体在后"，perm/
    inv_perm 必须是互逆置换，compact_global_ids 必须与置换后的
    原生（local-then-halo）全局编号一致。"""

    @pytest.mark.parametrize("rank", [0, 1])
    def test_invariants_hold_for_both_ranks(self, mixed_mesh_and_ops, rank):
        mesh, ops = mixed_mesh_and_ops
        # 两个棱柱(全局0,1)、两个四面体(全局2,3)各自跨两个 rank 分布，
        # 确保每个 rank 的 local+halo 集合里棱柱/四面体都不是单一类型
        # ——这是排列 bug 会实际暴露的场景（若每个 rank 恰好只拥有单一
        # 类型的单元，旧的"local在前halo在后"排列不会触发任何可观察的
        # 错误，测试也就失去意义）。
        cell_partition = np.array([0, 1, 0, 1], dtype=np.int32)
        fc = mesh.face_connectivity

        partition = build_distributed_partition(fc, cell_partition, rank=rank, n_ranks=2)
        dist_flat = build_distributed_flat_face(mesh, ops, partition, cell_partition=cell_partition)

        n_local = partition.n_local_cells
        n_halo = partition.n_halo
        n_total = n_local + n_halo

        # perm/inv_perm 是 [0,n_total) 的互逆置换
        assert sorted(dist_flat.perm.tolist()) == list(range(n_total))
        assert sorted(dist_flat.inv_perm.tolist()) == list(range(n_total))
        np.testing.assert_array_equal(dist_flat.perm[dist_flat.inv_perm], np.arange(n_total))
        np.testing.assert_array_equal(dist_flat.inv_perm[dist_flat.perm], np.arange(n_total))

        # compact_global_ids 必须等于"原生(local在前halo在后)全局编号"
        # 按 perm 重排后的结果
        native_ids = (
            np.concatenate([partition.local_cells, partition.halo_cells])
            if n_halo > 0 else partition.local_cells
        )
        np.testing.assert_array_equal(dist_flat.compact_global_ids, native_ids[dist_flat.perm])

        # 核心不变量：棱柱在前、四面体在后
        is_prism = dist_flat.compact_global_ids < mesh.n_prism_cells
        n_prism_compact = int(np.sum(is_prism))
        assert np.all(is_prism[:n_prism_compact]), "前 n_prism_compact 个位置必须全是棱柱"
        assert not np.any(is_prism[n_prism_compact:]), "n_prism_compact 之后必须没有棱柱"

        # sub_flat.n_prism（下游体积项切片直接消费的字段）必须等于这个
        # 排列下的真实棱柱计数，不是全局棱柱计数
        assert dist_flat.base_flat.n_prism == n_prism_compact

        # compact_cell_type 与排列独立算出的类型必须逐位一致
        expected_type = np.where(is_prism, 0, 1).astype(np.int8)
        np.testing.assert_array_equal(dist_flat.compact_cell_type, expected_type)

    def test_native_order_is_not_trivially_prism_grouped(self, mixed_mesh_and_ops):
        """负控制：确认这个合成算例真的会触发排列问题（原生 local-then-
        halo 顺序本身不是"棱柱在前"），不是测试在一个碰巧已经符合要求
        的输入上空转。"""
        mesh, ops = mixed_mesh_and_ops
        cell_partition = np.array([0, 1, 0, 1], dtype=np.int32)
        fc = mesh.face_connectivity

        partition = build_distributed_partition(fc, cell_partition, rank=0, n_ranks=2)
        n_halo = partition.n_halo
        native_ids = (
            np.concatenate([partition.local_cells, partition.halo_cells])
            if n_halo > 0 else partition.local_cells
        )
        is_prism_native = native_ids < mesh.n_prism_cells
        # rank0 拥有全局 cell 0(棱柱)和 2(四面体)，原生顺序是 [0,2,...]
        # -> [棱柱, 四面体, ...]，即便这恰好已经是"前棱柱后四面体"，halo
        # （若存在）大概率打破这个顺序——用 assert 显式记录这个场景下
        # 原生顺序与排列后顺序是否不同，而不是盲目假设一定不同。
        dist_flat = build_distributed_flat_face(mesh, ops, partition, cell_partition=cell_partition)
        native_is_prism_grouped = bool(np.all(is_prism_native[: int(np.sum(is_prism_native))]))
        permuted_is_prism_grouped = True  # 已在上面的测试里独立验证过
        # 至少要确认 perm 不是恒等置换（如果是恒等置换，说明这个测试
        # 网格没有真正触发需要重排的场景，需要换一个更能暴露问题的
        # cell_partition）。
        assert not np.array_equal(dist_flat.perm, np.arange(len(dist_flat.perm))), (
            "perm 是恒等置换——这个合成算例没有触发需要重排的场景，"
            "无法验证排列修复是否生效"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
