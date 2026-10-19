"""分布式 checkpoint 按 rank 分发（I-103，2026-10-19）：多线程模拟多 rank 的点对点通信（本机没有 MPI）。

* `distribute_from_root`：root 把全局逐单元数组按各 rank 的全局单元索引点对点发出，每个 rank 只拿自己的段；
  此前全局数组用 pickle 的 bcast 发给每个 rank（内存 O(全局)、单条消息 >2 GiB 失败）。
* `_root_read`：root 上读取 / 校验失败时异常广播给全部 rank 一起抛出；此前 `distributed_load_checkpoint` 不传播，
  其余 rank 卡在 bcast 里。
"""

import queue
import threading

import numpy as np
import pytest

from autoflowcfd.core.mpi.distributed_checkpoint import load as load_mod
from autoflowcfd.core.mpi.distributed_checkpoint import state as state_mod

_TIMEOUT = 20.0     # 秒：任何一个 rank 卡住时测试失败而不是挂起


class _ThreadP2P:
    """n 个线程（rank）之间的 bcast / Send / Recv，语义与 mpi4py 相同（Recv 写进调用方给的缓冲区）。"""

    def __init__(self, n):
        self.n = n
        self.local = threading.local()
        self.queues = {}
        self.lock = threading.Lock()

    def _q(self, src, dst, tag):
        with self.lock:
            return self.queues.setdefault((src, dst, tag), queue.Queue())

    def rank(self):
        return self.local.rank

    def bcast(self, obj, root=0):
        me = self.rank()
        if me == root:
            for r in range(self.n):
                if r != root:
                    self._q(root, r, "bcast").put(obj)
            return obj
        return self._q(root, me, "bcast").get(timeout=_TIMEOUT)

    def Send(self, buf, dest, tag):
        self._q(self.rank(), dest, tag).put(np.array(buf, copy=True))

    def Recv(self, buf, source, tag):
        data = self._q(source, self.rank(), tag).get(timeout=_TIMEOUT)
        np.copyto(buf, data.reshape(buf.shape))

    def run(self, fn):
        out, errors = [None] * self.n, [None] * self.n

        def work(r):
            self.local.rank = r
            try:
                out[r] = fn(r)
            except Exception as e:
                errors[r] = e

        threads = [threading.Thread(target=work, args=(r,)) for r in range(self.n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=2 * _TIMEOUT)
            assert not t.is_alive(), "有 rank 卡住"
        return out, errors


@pytest.fixture
def comm(monkeypatch):
    c = _ThreadP2P(3)
    for mod in (state_mod, load_mod):
        monkeypatch.setattr(mod, "get_comm", lambda: c)
        monkeypatch.setattr(mod, "get_rank", c.rank)
        monkeypatch.setattr(mod, "get_size", lambda: c.n)
    return c


_LOCAL_CELLS = [np.array([5, 0, 3]), np.array([1, 6]), np.array([2, 4, 7])]   # 非连续、非排序


@pytest.mark.parametrize("tail", [(), (4,), (4, 5)])
def test_each_rank_receives_exactly_its_cells(comm, tail):
    rng = np.random.default_rng(0)
    g = rng.standard_normal((8,) + tail)

    def fn(r):
        return state_mod.distribute_from_root(g if r == 0 else None, _LOCAL_CELLS[r])

    out, errors = comm.run(fn)
    assert errors == [None] * 3
    for r in range(3):
        np.testing.assert_array_equal(out[r], g[_LOCAL_CELLS[r]])
        assert out[r].shape == (len(_LOCAL_CELLS[r]),) + tail


def test_non_root_never_receives_the_global_array(comm):
    """非 root 收到的消息总量只有自己的段（此前每个 rank 收到整个全局数组）。"""
    g = np.arange(8 * 6, dtype=float).reshape(8, 6)
    received = []
    orig = comm.Recv

    def counting_recv(buf, source, tag):
        orig(buf, source, tag)
        if comm.rank() != 0 and buf.dtype == np.float64:
            received.append((comm.rank(), buf.size))

    comm.Recv = counting_recv
    comm.run(lambda r: state_mod.distribute_from_root(g if r == 0 else None, _LOCAL_CELLS[r]))
    assert sorted(received) == [(1, 2 * 6), (2, 3 * 6)]


def test_root_failure_is_raised_on_every_rank(comm):
    def fn(r):
        def read():
            raise ValueError("U_sps 形状与求解器不符")
        return load_mod._root_read(read)

    out, errors = comm.run(fn)
    assert all(isinstance(e, ValueError) and "形状" in str(e) for e in errors)


def test_root_result_is_shared_and_arrays_stay_on_root(comm):
    def fn(r):
        return load_mod._root_read(lambda: (("meta", 7), {"U_sps": np.ones(3)}))

    out, errors = comm.run(fn)
    assert errors == [None] * 3
    assert all(o[0] == ("meta", 7) for o in out)
    assert "U_sps" in out[0][1] and out[1][1] == {} and out[2][1] == {}
