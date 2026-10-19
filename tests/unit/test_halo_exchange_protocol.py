"""halo 交换协议（CPU `core/mpi/halo.py::HaloExchange` 与 GPU `core/gpu/distributed/gpu_halo_exchange.py`）。

真实分区（3 个 rank）各开一个线程，经线程局部的替身通信器收发；扩展数组必须逐位等于从全局场按
`local_cells + halo_cells` 切出的真值。修复前（I-95 / I-122）：接收位置每次交换逐单元 `searchsorted`（CPU）/
逐单元设备赋值（GPU），找不到的单元静默跳过；扩展数组强制 float64；GPU CUDA-aware 路径用了 CuPy 不存在的
`MemoryPointer.from_device`，一执行就 AttributeError，打包后也没有流同步。
"""

import queue
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from autoflowcfd.core.mpi.partition import build_distributed_partition, partition_mesh
from tests.unit._numpy_as_cupy import NumpyAsCupy
from tests.validation._channel_mesh import build_channel_mesh_prism

N_RANKS = 3


class _Comm:
    """线程局部通信器替身：(源, 目的) 一条 FIFO 队列，Isend 立即拷贝入队，Irecv 在 Wait 时出队写入 buffer。"""

    def __init__(self, rank, boxes):
        self.rank, self.boxes = rank, boxes

    def Get_rank(self):
        return self.rank

    def Isend(self, buf, dest, tag=0):
        self.boxes[(self.rank, dest)].put(np.array(buf, copy=True))
        return SimpleNamespace(wait=lambda: None)

    def Irecv(self, buf, source, tag=0):
        def wait():
            buf[...] = self.boxes[(source, self.rank)].get(timeout=30)
        return SimpleNamespace(wait=wait)


_MPI = SimpleNamespace(Request=SimpleNamespace(Waitall=lambda reqs: [r.wait() for r in reqs]))


@pytest.fixture
def world(monkeypatch):
    """3 个 rank 的真实分区 + 线程局部通信器（CPU 与 GPU 两个模块都替换）。"""
    import autoflowcfd.core.gpu.distributed.gpu_halo_exchange as gmod
    import autoflowcfd.core.mpi.halo as cmod

    mesh = build_channel_mesh_prism(1, 6, 3, 2, 0.6, 0.1, 0.08)
    fc = mesh.face_connectivity
    cell_part = partition_mesh(fc, N_RANKS, n_cells=mesh.n_cells)
    parts = [build_distributed_partition(fc, cell_part, r, N_RANKS) for r in range(N_RANKS)]
    assert all(p.neighbor_ranks for p in parts) and max(len(p.neighbor_ranks) for p in parts) >= 2
    boxes = {(a, b): queue.Queue() for a in range(N_RANKS) for b in range(N_RANKS)}
    local = threading.local()
    for m in (cmod, gmod):
        monkeypatch.setattr(m, "get_comm", lambda: local.comm)
        monkeypatch.setattr(m, "get_mpi", lambda: _MPI)
        monkeypatch.setattr(m, "mpi_available", lambda require=False: True)
    monkeypatch.setattr(gmod, "gpu_available", True)
    monkeypatch.setattr(gmod, "get_cupy", lambda: NumpyAsCupy())

    def run(make_exchanger, field):
        out, errors = [None] * N_RANKS, []

        def body(r):
            try:
                local.comm = _Comm(r, boxes)
                ex = make_exchanger(parts[r])
                out[r] = ex.exchange(field[parts[r].local_cells])
            except Exception as e:          # 线程里的失败带回主线程
                errors.append(e)
        threads = [threading.Thread(target=body, args=(r,)) for r in range(N_RANKS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(60)
        if errors:
            raise errors[0]
        return out

    return SimpleNamespace(mesh=mesh, parts=parts, run=run, gmod=gmod)


def _cpu(part):
    from autoflowcfd.core.mpi.halo import HaloExchange
    return HaloExchange(part, n_sps=8, n_vars=5)


def _gpu(cuda_aware):
    def make(part):
        import autoflowcfd.core.gpu.distributed.gpu_halo_exchange as gmod
        gmod._cuda_aware_mpi_cache = cuda_aware
        return gmod.GPUHaloExchange(part, n_sps=8, n_vars=5)
    return make


@pytest.mark.parametrize("backend", ["cpu", "gpu_cuda_aware", "gpu_staging"])
@pytest.mark.parametrize("cell_shape,dtype", [((8, 5), np.float64), ((8, 2), np.float64), ((), np.float64),
                                              ((8,), np.int64)])
def test_extended_array_equals_global_truth(world, monkeypatch, backend, cell_shape, dtype):
    if backend != "cpu":
        monkeypatch.setattr(world.gmod, "_cuda_aware_mpi_cache", None)
    make = {"cpu": _cpu, "gpu_cuda_aware": _gpu(True), "gpu_staging": _gpu(False)}[backend]
    rng = np.random.default_rng(3)
    field = (rng.standard_normal((world.mesh.n_cells,) + cell_shape) * 100).astype(dtype)
    out = world.run(make, field)
    for r, part in enumerate(world.parts):
        truth = field[np.concatenate([part.local_cells, part.halo_cells])]
        assert out[r].dtype == dtype                      # 类型跟随输入（此前强制 float64）
        np.testing.assert_array_equal(out[r], truth)


def test_inconsistent_recv_list_is_refused(world):
    """接收列表里有不在 halo_cells 中的单元：构造期硬失败（此前那一行 halo 静默留在未初始化的 empty 里）。"""
    from autoflowcfd.core.mpi.halo import halo_recv_positions

    part = world.parts[0]
    r = next(iter(part.recv_lists))
    bad = SimpleNamespace(**{k: getattr(part, k) for k in ("rank", "halo_cells", "n_local_cells")},
                          recv_lists={r: np.append(np.asarray(part.recv_lists[r]), part.local_cells[0])})
    with pytest.raises(ValueError, match="不在本 rank 的 halo_cells"):
        halo_recv_positions(bad)


def test_cpu_halo_without_mpi_but_with_neighbors_is_refused(monkeypatch, world):
    import autoflowcfd.core.mpi.halo as cmod
    monkeypatch.setattr(cmod, "mpi_available", lambda require=False: False)
    with pytest.raises(RuntimeError, match="MPI 不可用"):
        _cpu(world.parts[0]).exchange(np.zeros((world.parts[0].n_local_cells, 8, 5)))
