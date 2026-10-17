"""四面体 Warp & Blend 优化参数的下标（I-26）与 checkpoint 的四面体节点集版本。

Nodes3D.m：`alpopt = [0 0 0 0.1002 1.1332 ...]; if N <= 15, alpha = alpopt(N)`（MATLAB 1 起下标）。移植成 0 起下标的
`ALPHA_STORE[p]` 后 p >= 3 错位一档（P3 用了 0.1002，应为 0）；P1/P2 两种取法恰好都是 0。节点集改变之后，P3+
（实际是 P4+，P3 的 alpha 不影响节点）checkpoint 里的解点值在旧、新节点上含义不同，必须硬失败而不是静默重解释。
"""

import h5py
import numpy as np
import pytest

from autoflowcfd.fr.warp_blend_nodes import TET_NODE_SET, warp_blend_alpha, warp_blend_nodes_3d


@pytest.mark.parametrize("p,alpha", [(1, 0.0), (2, 0.0), (3, 0.0), (4, 0.1002), (5, 1.1332), (15, 0.9655),
                                     (16, 1.0)])
def test_alpha_follows_nodes3d(p, alpha):
    assert warp_blend_alpha(p) == pytest.approx(alpha)


@pytest.mark.parametrize("p,changes", [(2, False), (3, False), (4, True), (5, True)])
def test_only_p4_and_above_nodes_change(p, changes):
    """修复前后的节点：P1~P3 相同（P3 没有内部节点、面内只有面心一点，blend 为零），P4 起不同。"""
    from autoflowcfd.fr import warp_blend_nodes as wb

    new = np.stack(warp_blend_nodes_3d(p))
    old_alpha = wb.ALPHA_STORE[p] if p <= 14 else 1.0          # 修复前的取法
    saved = wb.warp_blend_alpha
    try:
        wb.warp_blend_alpha = lambda q: old_alpha if q == p else saved(q)
        old = np.stack(warp_blend_nodes_3d(p))
    finally:
        wb.warp_blend_alpha = saved
    assert bool(np.abs(new - old).max() > 1e-6) == changes


def _write(tmp_path, solution):
    from types import SimpleNamespace

    from autoflowcfd.core.utils.checkpoint import CheckpointManager

    mgr = CheckpointManager(config=SimpleNamespace(), output_dir=str(tmp_path), quiet=True)
    path = mgr.save(solution=solution, history={"iterations": [1], "residuals": {"rho": [1.0]},
                                                "coefficients": {}, "cfl_history": [0.03]}, iteration=3)
    return mgr, path


def test_node_set_is_recorded_and_p3_without_it_is_refused(tmp_path):
    from autoflowcfd.core.utils.checkpoint_load import load_checkpoint

    mgr, path = _write(tmp_path, np.ones((2, 125, 5)))          # P4：(4+1)^3 个解点
    with h5py.File(path, "r") as f:
        saved = f["metadata"].attrs["tet_node_set"]
    assert (saved.decode() if isinstance(saved, bytes) else saved) == TET_NODE_SET
    load_checkpoint(mgr, path)
    with h5py.File(path, "r+") as f:
        del f["metadata"].attrs["tet_node_set"]
    with pytest.raises(Exception, match="四面体节点集"):
        load_checkpoint(mgr, path)


def test_p3_checkpoint_without_node_set_still_loads(tmp_path):
    """P1~P3 的节点没有变：旧 checkpoint 照常加载。"""
    from autoflowcfd.core.utils.checkpoint_load import load_checkpoint

    mgr, path = _write(tmp_path, np.ones((2, 64, 5)))
    with h5py.File(path, "r+") as f:
        del f["metadata"].attrs["tet_node_set"]
    load_checkpoint(mgr, path)
