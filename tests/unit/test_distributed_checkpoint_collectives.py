"""分布式 checkpoint 的湍流字段收集：root 与非 root 必须做同样多次集体收集（I-199d）。

`gather_global_state` 是点对点收发（root 逐 rank Recv、其余 rank Send）。此前 `gather_turbulence_fields` 在非 root
上收完输运场就返回，root 却还要收第二次（单元积分平均）：root 等不到发送而挂起，或把非 root 接下来那次收集
（上一时间层）的数据当成积分平均收下。
"""

from types import SimpleNamespace

import numpy as np

import autoflowcfd.core.mpi.distributed_checkpoint.turbulence as turb_ckpt
import autoflowcfd.core.turbulence.output as turb_out


class _Model:
    TRANSPORTED_FIELDS = ("k", "omega")

    def __init__(self, n_cells: int, n_sps: int):
        self.nu_t = np.full((n_cells, n_sps), 3.0)
        self._fields = [np.full((n_cells, n_sps), 1.0), np.full((n_cells, n_sps), 2.0)]

    def transported_fields(self):
        return self._fields


def _gather(monkeypatch, is_root: bool):
    calls = []

    def fake_gather(U_local, local_cells, n_global):
        calls.append(U_local.shape)
        return np.zeros((n_global,) + U_local.shape[1:]) if is_root else None

    monkeypatch.setattr(turb_ckpt, "gather_global_state", fake_gather)
    monkeypatch.setattr(turb_out, "turbulence_cell_means", lambda model, W: {"k": np.ones(4), "omega": np.ones(4)})
    solver = SimpleNamespace(turb_model=_Model(4, 3),
                             partition=SimpleNamespace(n_local_cells=4, local_cells=np.arange(4)))
    return turb_ckpt.gather_turbulence_fields(solver, 8, np.ones((4, 3))), calls


def test_every_rank_takes_part_in_every_gather(monkeypatch):
    out_root, calls_root = _gather(monkeypatch, is_root=True)
    out_other, calls_other = _gather(monkeypatch, is_root=False)
    assert out_other is None
    assert set(out_root) == {"k", "omega", "nu_t", turb_out.CHECKPOINT_PREFIX + "k",
                             turb_out.CHECKPOINT_PREFIX + "omega"}
    assert len(calls_root) == 2 and calls_other == calls_root
