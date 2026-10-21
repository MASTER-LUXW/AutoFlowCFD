"""完全分布式加载的紧凑包传输（I-106）：pickle 字节分块收发（单条消息超过 2^31-1 字节时 MPI 计数溢出）；root 逐个
rank 构建、发出即释放（此前 n_ranks 份包同时在 root 内存里），初次加载与换阶共用一个分发函数。"""

from collections import defaultdict, deque

import numpy as np
import pytest

import autoflowcfd.core.mpi.comm as comm_mod
from autoflowcfd.core.mpi.distributed_mesh_loader import fully_distributed


class _LoopbackComm:
    """按 (dest, tag) 排队的点对点消息；Recv 不区分 source（测试里只有 root 发送）。"""

    def __init__(self):
        self.queues = defaultdict(deque)
        self.sizes = defaultdict(list)
        self.dest = None

    def Send(self, buf, dest, tag):
        data = np.array(buf, copy=True)
        self.queues[(dest, tag)].append(data)
        self.sizes[(dest, tag)].append(data.nbytes)

    def Recv(self, buf, source, tag):
        data = self.queues[(self.dest, tag)].popleft()
        assert data.nbytes == buf.nbytes
        buf[...] = data.view(buf.dtype).reshape(buf.shape)


@pytest.fixture()
def loopback(monkeypatch):
    comm = _LoopbackComm()
    monkeypatch.setattr(comm_mod, "get_comm", lambda: comm)
    monkeypatch.setattr(comm_mod, "MESSAGE_CHUNK_BYTES", 7)
    return comm


def test_pickled_object_round_trips_in_bounded_chunks(loopback):
    obj = {"a": np.arange(50, dtype=np.float64), "b": "压缩包", "c": (1, None)}
    comm_mod.send_pickled(obj, 3, (10, 11))
    assert all(n <= 7 for n in loopback.sizes[(3, 11)]) and len(loopback.sizes[(3, 11)]) > 1
    loopback.dest = 3
    got = comm_mod.recv_pickled(0, (10, 11))
    np.testing.assert_array_equal(got["a"], obj["a"])
    assert got["b"] == obj["b"] and got["c"] == obj["c"]


def test_root_builds_and_sends_one_rank_at_a_time(loopback, monkeypatch):
    events = []

    def fake_build(*args, **kwargs):
        rank = args[4]
        events.append(("build", rank))
        return {"rank": rank, "order": args[10], "turb": kwargs["turb_model_name"]}

    real_send = fully_distributed.send_pickled

    def recording_send(obj, dest, tags):
        events.append(("send", dest))
        real_send(obj, dest, tags)

    monkeypatch.setattr(fully_distributed, "build_fully_distributed_rank_package", fake_build)
    monkeypatch.setattr(fully_distributed, "send_pickled", recording_send)
    root_context = {k: None for k in ("mesh", "ops", "fc", "cell_partition", "boundary_ghost_provider_global",
                                      "freestream", "mu_molecular", "mach_ref", "enable_viscous",
                                      "wall_distance_source", "h_max_global", "h_wn_global", "global_cell_colors",
                                      "global_cell_colors_d2")}
    root_context.update({name: None for name in fully_distributed.RUN_OPTION_NAMES}, turb_model_name="SST")

    mine = fully_distributed._distribute_packages(root_context, 2, 3, (20, 21))
    assert events == [("build", 1), ("send", 1), ("build", 2), ("send", 2), ("build", 0)]
    assert mine == {"rank": 0, "order": 2, "turb": "SST"}
    for rank in (1, 2):
        loopback.dest = rank
        received = fully_distributed._distribute_packages(None, 2, 3, (20, 21))
        assert received == {"rank": rank, "order": 2, "turb": "SST"}


def test_core_loader_does_not_import_the_cli():
    from tests.unit._module_source import module_source

    assert "autoflowcfd.cli" not in module_source("autoflowcfd.core.mpi.distributed_mesh_loader")
