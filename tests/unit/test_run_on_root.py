"""root 单独执行的操作（写 checkpoint / 最终结果）把成功或失败广播给全部 rank（I-102）。

此前 root 在 gather 之后、barrier 之前抛异常（写盘失败、磁盘满）时，其余 rank 永远卡在 barrier。
"""

import pytest

import autoflowcfd.core.mpi.comm as comm
from autoflowcfd.core.mpi.comm import run_on_root
from tests.unit._module_source import module_source


def _world(monkeypatch, rank, root_message):
    """模拟 rank 号与 root 的广播内容（root 上 bcast 原样返回自己传入的值）。"""
    sent = []

    def bcast(obj):
        sent.append(obj)
        return obj if rank == 0 else root_message

    monkeypatch.setattr(comm, "get_rank", lambda: rank)
    monkeypatch.setattr(comm, "bcast_from_root", bcast)
    return sent


def test_root_failure_is_broadcast_and_reraised(monkeypatch):
    sent = _world(monkeypatch, 0, None)

    def fail():
        raise OSError("磁盘已满")

    with pytest.raises(OSError, match="磁盘已满"):
        run_on_root(fail)
    assert sent == ["OSError: 磁盘已满"]          # 先告知其余 rank 再抛出


def test_other_ranks_fail_consistently(monkeypatch):
    _world(monkeypatch, 1, "OSError: 磁盘已满")
    called = []
    with pytest.raises(RuntimeError, match="磁盘已满"):
        run_on_root(lambda: called.append(1))
    assert called == []                            # 非 root 不执行


def test_success_returns_the_result_on_root_only(monkeypatch):
    _world(monkeypatch, 0, None)
    assert run_on_root(lambda: "ckpt.h5") == "ckpt.h5"
    _world(monkeypatch, 2, None)
    assert run_on_root(lambda: "ckpt.h5") is None


def test_distributed_writers_use_it():
    src = module_source("autoflowcfd.core.mpi.distributed_checkpoint.save")
    assert src.count("run_on_root(") == 2 and "barrier()" not in src
