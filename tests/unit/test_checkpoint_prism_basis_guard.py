"""checkpoint 必须记录棱柱基，跨基 resume 必须**硬失败**（2026-09-20）。

## 这份测试钉的是一个真实危险组合

两条棱柱基的每单元解点布局不同：原生基 P1 每单元只有前 6 个槽位是自由度
（其余按 `fr/native_padding.py` 约定复制真实 SP#0），坍缩基 8 个全是自由
度。同一份 `(n_cells, n_sps, n_vars)` 数组在两条基下**含义不同**，而
**形状恰好相同** —— 所以跨基 resume 既不会报形状不符、也不会报任何错，
只是静默给出一个看不出异常的错解。

这个组合在 2026-09-20 默认值从 `collapsed` 改成 `native` 的那一刻变成
现实：磁盘上所有既有 checkpoint 都是 collapsed 时代写的。

判据分三条：写入、同基放行、跨基拒绝；再加一条"属性缺失"的兼容规则
（缺失时只有 `collapsed` 放行，因为那之前唯一的默认值就是它）。
"""

import h5py
import numpy as np
import pytest

from autoflowcfd.core.utils.checkpoint import CheckpointManager
from autoflowcfd.core.utils.checkpoint_load import load_checkpoint


def _manager(tmp_path):
    """最小 CheckpointManager：`config` 只被 `_compute_config_hash` 读，
    用 `getattr(..., 默认值)` 取，所以给一个空对象即可。"""
    from types import SimpleNamespace

    return CheckpointManager(config=SimpleNamespace(),
                             output_dir=str(tmp_path), quiet=True)


def _write(tmp_path, monkeypatch, basis):
    monkeypatch.setenv("AFCFD_PRISM_BASIS", basis)
    mgr = _manager(tmp_path)
    path = mgr.save(
        solution=np.ones((3, 5)),
        # history 的 schema 见 `CheckpointManager.save` 文档：
        # residuals 是 {方程名: [值]} 的字典，不是列表。
        history={"iterations": [1, 2],
                 "residuals": {"rho": [1.0, 0.5]},
                 "coefficients": {},
                 "cfl_history": [0.03, 0.03]},
        iteration=7,
    )
    assert path is not None
    return path


@pytest.mark.parametrize("basis", ["collapsed", "native"])
def test_basis_is_recorded(tmp_path, monkeypatch, basis):
    path = _write(tmp_path, monkeypatch, basis)
    with h5py.File(path, "r") as f:
        saved = f["metadata"].attrs["prism_basis"]
    if isinstance(saved, bytes):
        saved = saved.decode("utf-8")
    assert saved == basis


@pytest.mark.parametrize("basis", ["collapsed", "native"])
def test_same_basis_loads(tmp_path, monkeypatch, basis):
    path = _write(tmp_path, monkeypatch, basis)
    mgr = _manager(tmp_path)
    sol, hist, it, _meta = load_checkpoint(mgr, path)
    assert it == 7
    assert sol.shape == (3, 5)


@pytest.mark.parametrize("saved,current", [("collapsed", "native"),
                                           ("native", "collapsed")])
def test_cross_basis_resume_raises(tmp_path, monkeypatch, saved, current):
    """**核心判据**：跨基 resume 必须抛错，不是 warning。

    与 `config_hash` 不一致刻意不同级别：配置哈希不一致只是"可能影响
    结果"（例如 CFL 上限变了），棱柱基不一致是"这份数据的含义变了"。
    """
    path = _write(tmp_path, monkeypatch, saved)
    monkeypatch.setenv("AFCFD_PRISM_BASIS", current)
    mgr = _manager(tmp_path)
    with pytest.raises(RuntimeError, match="棱柱基"):
        load_checkpoint(mgr, path)


def test_missing_attr_allows_collapsed_but_refuses_native(tmp_path,
                                                          monkeypatch):
    """属性缺失（2026-09-20 之前写的 checkpoint）的兼容规则。

    那之前**唯一**的默认值是 collapsed，所以"缺失 + 当前 collapsed"是
    兼容组合（放行）；"缺失 + 当前 native"正是危险组合（拒绝）。
    """
    path = _write(tmp_path, monkeypatch, "collapsed")
    with h5py.File(path, "r+") as f:
        del f["metadata"].attrs["prism_basis"]

    mgr = _manager(tmp_path)
    monkeypatch.setenv("AFCFD_PRISM_BASIS", "collapsed")
    sol, _hist, it, _meta = load_checkpoint(mgr, path)
    assert it == 7 and sol.shape == (3, 5)

    monkeypatch.setenv("AFCFD_PRISM_BASIS", "native")
    with pytest.raises(RuntimeError, match="没有记录棱柱基"):
        load_checkpoint(mgr, path)
