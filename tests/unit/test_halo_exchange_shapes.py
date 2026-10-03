# -*- coding: utf-8 -*-
"""`HaloExchange.exchange` 的 MPI 分支：任意逐单元形状、两个 rank 真实打包/收发/回填。

本机没有 MPI；两个线程各扮演一个 rank，假的通信子用"邮箱 + 条件变量"实现 Isend/Irecv/
Waitall（Isend 立即投递一份拷贝，Irecv 的 Waitall 阻塞到对方投递为止），走的是生产代码
里 `mpi_available` 为真时的同一段打包与按 `halo_cells` 回填逻辑。判据：每个 rank 扩展数组的
halo 段等于从全局真值按 `halo_cells` 切出来的值——标量、`(n_sps,)`、`(n_sps, n_vars)`、
`(n_sps, 3, 3)` 四种逐单元形状经同一个实例交换（此前按形状各建一个交换器、另有一份
标量专用的 `exchange_scalar`）。
"""

import threading
from collections import defaultdict, deque

import numpy as np

import autoflowcfd.core.mpi.halo as halo_mod
from autoflowcfd.core.mpi.halo import HaloExchange
from autoflowcfd.core.mpi.partition import build_distributed_partition
from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh


class _Mailbox:
    """按 (源, 目的, tag) 先进先出——MPI 对同一 (源, 目的, tag, 通信子) 保证按发送顺序匹配
    （non-overtaking），连续几次交换都用 tag 0 依赖的正是这一点。"""

    def __init__(self):
        self.messages, self.cond = defaultdict(deque), threading.Condition()

    def post(self, key, data):
        with self.cond:
            self.messages[key].append(np.array(data, copy=True))
            self.cond.notify_all()

    def take(self, key):
        with self.cond:
            assert self.cond.wait_for(lambda: len(self.messages[key]) > 0, timeout=30), f"等不到消息 {key}"
            return self.messages[key].popleft()


class _Request:
    def __init__(self, on_wait=None):
        self.on_wait = on_wait

    def wait(self):
        if self.on_wait is not None:
            self.on_wait()


class _Comm:
    def __init__(self, rank, box):
        self.rank, self.box = rank, box

    def Isend(self, buf, dest, tag):
        self.box.post((self.rank, dest, tag), buf)
        return _Request()

    def Irecv(self, buf, source, tag):
        def fill():
            buf[...] = self.box.take((source, self.rank, tag))
        return _Request(fill)


class _MPI:
    class Request:
        @staticmethod
        def Waitall(reqs):
            for r in reqs:
                r.wait()


def test_exchange_any_cell_shape_two_ranks(monkeypatch):
    mesh = _build_synthetic_mixed_mesh(1)
    fc = mesh.face_connectivity
    cell_partition = np.array([0, 1, 0, 1], dtype=np.int32)
    parts = [build_distributed_partition(fc, cell_partition, rank=r, n_ranks=2) for r in (0, 1)]
    assert all(p.n_halo > 0 and p.neighbor_ranks for p in parts), "合成网格的两个分区必须互为 halo"

    n_sps = 4
    rng = np.random.default_rng(0)
    fields = [rng.standard_normal((4,) + shape) for shape in ((), (n_sps,), (n_sps, 5), (n_sps, 3, 3))]

    box = _Mailbox()
    local = threading.local()
    monkeypatch.setattr(halo_mod, "mpi_available", True)
    monkeypatch.setattr(halo_mod, "get_comm", lambda: local.comm)
    monkeypatch.setattr(halo_mod, "get_mpi", lambda: _MPI)

    results, errors = {}, []

    def run(r):
        try:
            local.comm = _Comm(r, box)
            ex = HaloExchange(parts[r], n_sps, 5)
            results[r] = [ex.exchange(f[parts[r].local_cells]) for f in fields]
        except Exception as e:  # pragma: no cover - 线程内异常转交主线程断言
            errors.append(e)

    threads = [threading.Thread(target=run, args=(r,)) for r in (0, 1)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not errors, errors

    for r in (0, 1):
        p = parts[r]
        native = np.concatenate([p.local_cells, p.halo_cells])
        for f, ext in zip(fields, results[r]):
            np.testing.assert_array_equal(ext, f[native])
