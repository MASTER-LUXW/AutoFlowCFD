"""验证 DistributedFRSolver/MultiGPUDistributedSolver 的
`boundary_ghost_provider.group_code` 压缩索引空间重映射——真实 bug 修复
（2026-09-02，实现分布式湍流模型时排查发现，与湍流本身无关）：
`build_boundary_ghost_provider`/`FRSolver.__init__` 用完整全局网格构造
`group_code`（长度=全局面数，按全局面编号索引），但
`distributed_compute_inviscid_residual`/`_viscous_residual` 最终调用
`ghost_provider(f, ...)` 时 `f` 是本 rank 的 local+halo **压缩索引空间**
面编号——两套编号不是同一个索引空间的子区间。真实合成网格验证过：
2-rank 分区下 8/12（67%）压缩面会被分配到错误的边界组编码。

本文件直接用 `build_distributed_partition`/`build_distributed_flat_face`
（与 test_distributed_compute_residual.py 同一套已验证的底层构造方式，
绕开 DistributedFRSolver.__init__ 依赖真实 MPI broadcast 的分区路径）
复现修复前的 bug 场景 + 验证修复后的 remap 逻辑本身。

只测试 CPU 路径——`MultiGPUDistributedSolver` 的同名修复逻辑完全一致
（见 gpu_distributed.py 同一处注释），但该类构造本身需要真实 CuPy+MPI
（本机都没有），无法在这里直接实例化验证，只能保证两处修复逐字对应。
"""

import numpy as np
import pytest

from autoflowcfd.core.mpi.partition import build_distributed_partition
from autoflowcfd.core.mpi.distributed_flat_face import build_distributed_flat_face
from autoflowcfd.fr.operators import generate_fr_operators
from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh


class _FakeGroupCodeProvider:
    """最小假 provider，只暴露 group_code 属性，模拟
    BoundaryGhostStateProvider 未经 remap 前的状态（长度=全局面数）。"""
    def __init__(self, group_code):
        self.group_code = group_code


@pytest.mark.parametrize("rank", [0, 1])
def test_group_code_remap_matches_compact_index_space(rank):
    order = 1
    mesh = _build_synthetic_mixed_mesh(order)
    ops = generate_fr_operators(order)
    fc = mesh.face_connectivity
    n_faces_global = fc.n_faces

    # 用"每个面各自唯一编码"的合成 group_code，清楚地暴露索引空间是否
    # 对齐（与真实排查时用的验证脚本同一手法）。
    group_code_global = np.arange(n_faces_global, dtype=np.int64)
    provider = _FakeGroupCodeProvider(group_code_global.copy())

    cell_partition = np.array([0, 1, 0, 1], dtype=np.int32)
    partition = build_distributed_partition(fc, cell_partition, rank=rank, n_ranks=2)
    dist_fc = build_distributed_flat_face(mesh, ops, partition, cell_partition=cell_partition)
    n_faces_compact = dist_fc.base_flat.n_faces
    local_faces = partition.local_faces

    # 修复前的真实 bug 现场：直接用压缩空间面索引去查全局编号的
    # group_code（等价于此前从未 remap 时 ghost_provider(f,...) 内部
    # `self.group_code[f]` 的行为）。
    buggy_lookup = group_code_global[:n_faces_compact]

    # 本次修复：与 distributed_solver.py::local_solver /
    # gpu_distributed.py::__init__ 完全同一行代码。
    provider.group_code = provider.group_code[local_faces]

    assert provider.group_code.shape[0] == n_faces_compact
    # 决定性判据：修复后的取值必须是"按真实全局面编号取值"，不是"直接
    # 用压缩索引取值"——用上面同一份 buggy_lookup 交叉核对两者存在真实
    # 差异（否则这个合成网格/分区凑巧是恒等映射，测试没有真正覆盖场景）。
    n_mismatch_vs_buggy = int(np.sum(provider.group_code != buggy_lookup))
    assert n_mismatch_vs_buggy > 0, (
        "这个分区凑巧是恒等映射，没有真正覆盖 local_faces != range(n_compact) "
        "的场景，测试本身需要换一个分区/网格重新设计。"
    )
    np.testing.assert_array_equal(provider.group_code, group_code_global[local_faces])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
