"""`build_distributed_partition` 的 send_lists 向量化重写等价性验证
（2026-09-14）。

## 背景

这段代码原先是"外层按 rank、内层按面"的双层 Python 循环，并且用
`if local_idx not in send_cells` 对一个 **list** 做成员测试。注释把它
标为"首版简化：通过面连接直接判断"——那个标注本身是错的：从
`(face_connectivity, cell_partition)` 推导"哪些 local cell 是 rank r 的
halo"是**精确**的（每个 rank 都已持有这两份全局数据，不需要通信、
没有精度损失），绕开全局通信是数据可得性的结果，不是用近似换简单。

真实缺陷是性能：O(n_ranks × n_faces) 次解释器迭代 + O(n_send^2) 的列表
成员测试。79 万单元 cube_demo（188 万面）在 16 rank 下仅这一步就是
3000 万次迭代，外加数亿次比较。已改为纯 numpy。

## 本文件的判据

1. **逐位等价**：对多种网格/分区/rank 数，向量化结果必须与原逐面循环
   的参考实现给出**完全相同**的数组（含顺序）。参考实现在本文件内按
   原算法重写一份，不依赖已被替换的生产代码。
2. **通信协议约定**：send_lists 必须按**全局 cell id 升序**排列。这一条
   是 `halo.py::HaloExchange.exchange()` 的硬约束（两端不传 id、完全
   依赖同序摆放），历史上出过真实 bug（发送方顺序
   [22,23,18,21,19,20] vs 接收方期望 [18,19,20,21,22,23]，集合相同
   顺序不同 -> halo 数据静默对应到错误单元）。
3. **与 recv_lists 互为镜像**：rank a 发给 rank b 的单元集合，必须正好
   等于 rank b 从 rank a 接收的单元集合（按全局 id 比较）。这是"不需要
   通信也能算对"这个论断的直接检验。
4. **性能量级**：在一个足够大的合成连接关系上，向量化实现必须远快于
   参考实现——把"这是性能修复"这件事也钉住，避免将来有人"为了可读性"
   改回逐面循环。
"""

import time

import numpy as np
import pytest

from autoflowcfd.core.mpi.partition import build_distributed_partition


class _FakeFC:
    """最小面连接关系替身：`build_distributed_partition` 只用到这四个字段
    （owner_cell/neighbor_cell/is_boundary/n_faces）。用替身而不是真实
    网格，是为了能自由构造 rank 数/分区形态，把判别力集中在本次改动上。"""

    def __init__(self, owner, neighbor, n_cells):
        self.owner_cell = np.asarray(owner, dtype=np.int64)
        self.neighbor_cell = np.asarray(neighbor, dtype=np.int64)
        self.is_boundary = self.neighbor_cell < 0
        self.n_faces = len(self.owner_cell)
        self.n_cells = n_cells


def _reference_send_lists(fc, cell_partition, rank, n_ranks,
                          global_to_local, local_cells):
    """按**原**算法（逐面循环 + 列表去重 + 末尾按全局 id 排序）重算一份
    参考值。刻意保留原实现的全部细节，包括那个 O(n^2) 的列表成员测试——
    本文件要验证的正是"新实现与它逐位等价"。"""
    out = {}
    for other_rank in range(n_ranks):
        if other_rank == rank:
            continue
        send_cells = []
        for f in range(fc.n_faces):
            if fc.is_boundary[f]:
                continue
            oc = int(fc.owner_cell[f])
            nc = int(fc.neighbor_cell[f])
            if nc < 0:
                continue
            oc_part = cell_partition[oc]
            nc_part = cell_partition[nc]
            if oc_part == rank and nc_part == other_rank:
                li = int(global_to_local[oc])
                if li not in send_cells:
                    send_cells.append(li)
            if nc_part == rank and oc_part == other_rank:
                li = int(global_to_local[nc])
                if li not in send_cells:
                    send_cells.append(li)
        if send_cells:
            arr = np.array(send_cells, dtype=np.int64)
            order = np.argsort(local_cells[arr])
            out[other_rank] = arr[order]
    return out


def _chain_mesh(n_cells):
    """一维链式连接：cell i 与 i+1 相邻，两端各一个物理边界面。"""
    owner = list(range(n_cells - 1)) + [0, n_cells - 1]
    neighbor = list(range(1, n_cells)) + [-1, -1]
    return _FakeFC(owner, neighbor, n_cells)


def _grid_mesh(nx, ny):
    """二维结构化网格的面连接（4 邻域），跨 rank 面更丰富。"""
    owner, neighbor = [], []

    def gid(i, j):
        return i + nx * j

    for j in range(ny):
        for i in range(nx):
            if i + 1 < nx:
                owner.append(gid(i, j))
                neighbor.append(gid(i + 1, j))
            else:
                owner.append(gid(i, j))
                neighbor.append(-1)
            if j + 1 < ny:
                owner.append(gid(i, j))
                neighbor.append(gid(i, j + 1))
            else:
                owner.append(gid(i, j))
                neighbor.append(-1)
    return _FakeFC(owner, neighbor, nx * ny)


CASES = [
    ("chain12_r2_block", _chain_mesh(12), 2, "block"),
    ("chain12_r3_rr", _chain_mesh(12), 3, "roundrobin"),
    ("chain25_r4_rr", _chain_mesh(25), 4, "roundrobin"),
    ("grid6x6_r3_block", _grid_mesh(6, 6), 3, "block"),
    ("grid6x6_r3_rr", _grid_mesh(6, 6), 3, "roundrobin"),
    ("grid7x5_r4_rr", _grid_mesh(7, 5), 4, "roundrobin"),
]


def _partition_of(kind, n_cells, n_ranks):
    if kind == "block":
        return (np.arange(n_cells) * n_ranks // n_cells).astype(np.int32)
    return (np.arange(n_cells) % n_ranks).astype(np.int32)


@pytest.mark.parametrize("name,fc,n_ranks,pkind",
                         CASES, ids=[c[0] for c in CASES])
class TestSendListsEquivalence:
    def test_matches_reference_elementwise(self, name, fc, n_ranks, pkind):
        cp_ = _partition_of(pkind, fc.n_cells, n_ranks)
        for rank in range(n_ranks):
            part = build_distributed_partition(fc, cp_, rank=rank, n_ranks=n_ranks)
            ref = _reference_send_lists(
                fc, cp_, rank, n_ranks, part.global_to_local, part.local_cells)
            got = part.send_lists
            assert set(got.keys()) == set(ref.keys()), (
                f"{name} rank{rank}: 接收方 rank 集合不同 "
                f"{sorted(got)} vs {sorted(ref)}")
            for r in ref:
                np.testing.assert_array_equal(
                    got[r], ref[r],
                    err_msg=f"{name} rank{rank}->{r}: send_list 与参考实现不同")

    def test_sorted_by_global_cell_id(self, name, fc, n_ranks, pkind):
        """通信协议硬约束：必须按全局 cell id 升序（见模块文档）。"""
        cp_ = _partition_of(pkind, fc.n_cells, n_ranks)
        for rank in range(n_ranks):
            part = build_distributed_partition(fc, cp_, rank=rank, n_ranks=n_ranks)
            for r, lst in part.send_lists.items():
                g = part.local_cells[np.asarray(lst, dtype=np.int64)]
                assert np.all(np.diff(g) > 0), (
                    f"{name} rank{rank}->{r}: send_list 未按全局 id 严格升序"
                    f"（{g}）——halo 交换会把数据对应到错误的单元上")

    def test_send_and_recv_are_mirrors(self, name, fc, n_ranks, pkind):
        """rank a 发给 b 的集合 == rank b 从 a 收的集合（按全局 id）。

        这是"不需要全局通信也能算对"这个论断的直接检验：两端各自独立
        从同一份全局数据推导，结果必须严格互为镜像。
        """
        cp_ = _partition_of(pkind, fc.n_cells, n_ranks)
        parts = [build_distributed_partition(fc, cp_, rank=r, n_ranks=n_ranks)
                 for r in range(n_ranks)]
        for a in range(n_ranks):
            for b in range(n_ranks):
                if a == b:
                    continue
                sent = parts[a].send_lists.get(b)
                if sent is not None and len(sent):
                    sent_g = parts[a].local_cells[np.asarray(sent, dtype=np.int64)]
                else:
                    sent_g = np.array([], dtype=np.int64)
                recv_g = np.asarray(
                    parts[b].recv_lists.get(a, np.array([], dtype=np.int64)),
                    dtype=np.int64)
                np.testing.assert_array_equal(
                    np.sort(sent_g), np.sort(recv_g),
                    err_msg=f"{name}: rank{a}->{b} 发送集合与 rank{b} 接收集合不镜像")


class TestSendListsPerformance:
    """把"这是一次性能修复"钉住：向量化必须远快于逐面循环参考实现。

    不断言绝对耗时（本机负载会波动），只断言**比值**——原实现在这个规模
    上是 O(n_ranks × n_faces) 迭代 + O(n_send^2) 列表查找，向量化后是若干
    次 numpy 扫描，差距应当在一个数量级以上。取 3 倍作为阈值，远低于
    实测差距、又足以在有人改回逐面循环时失败。
    """

    def test_vectorized_is_much_faster(self):
        fc = _grid_mesh(60, 60)          # 3600 单元、约 7200 面
        n_ranks = 8
        cp_ = _partition_of("roundrobin", fc.n_cells, n_ranks)

        t0 = time.perf_counter()
        parts = [build_distributed_partition(fc, cp_, rank=r, n_ranks=n_ranks)
                 for r in range(n_ranks)]
        t_vec = time.perf_counter() - t0

        t0 = time.perf_counter()
        for r in range(n_ranks):
            _reference_send_lists(fc, cp_, r, n_ranks,
                                  parts[r].global_to_local, parts[r].local_cells)
        t_ref = time.perf_counter() - t0

        assert t_ref > 3.0 * t_vec, (
            f"向量化实现没有明显更快（参考 {t_ref:.3f}s vs 向量化 "
            f"{t_vec:.3f}s，含分区构造的其余开销）——是否有人改回了逐面循环？")
