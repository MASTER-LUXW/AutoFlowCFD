"""非流形空腔局部重铺只有一个实现（I-142，`mesh_repair_nonmanifold_mixed.py`）：纯四面体网格按"没有棱柱"调用它。
结果带出原四面体的保留掩码（保留者按原顺序在前、新铺的在后），调用方据此变换逐四面体的附带数组（Stage B' 的坏单元
掩码）。此前纯四面体另有一份把全部种子合成一个空腔、无焊接、无质量门的实现。"""

import importlib

import numpy as np
import pytest

from autoflowcfd.grid.mesh_gen.repair.mesh_repair_nonmanifold_mixed import CavityPatch, patch_nonmanifold_cavity_mixed
from tests.unit.test_cavity_weld_conformity import _cube_tets, _torn_cube

_NO_PRISMS = (np.empty((0, 6), dtype=np.int64), np.ones(0, dtype=bool))


def _patch(nodes, tets, keep):
    return patch_nonmanifold_cavity_mixed(nodes, _NO_PRISMS[0], tets, _NO_PRISMS[1], keep)


def test_kept_mask_orders_survivors_first():
    nodes, tets, a, _b = _torn_cube()
    keep = np.ones(len(tets), dtype=bool)
    keep[np.flatnonzero((tets == a).any(axis=1))[0]] = False
    patched = _patch(nodes, tets, keep)
    assert isinstance(patched, CavityPatch) and patched.tet_kept.shape == (len(tets),)
    k = int(patched.tet_kept.sum())
    assert k < len(tets) and len(patched.tet_cells) > k              # 有单元被替换、有新铺单元
    # 保留者原样在前（焊接只改引用的节点号：坐标比对容许被合并的那一个点）
    kept_old = nodes[tets[patched.tet_kept]]
    kept_new = patched.nodes[patched.tet_cells[:k]]
    assert np.mean(np.all(np.isclose(kept_old, kept_new), axis=(1, 2))) > 0.95


def test_unchanged_mesh_keeps_everything():
    nodes, tets, _ = _cube_tets(2)
    patched = _patch(nodes, tets, np.ones(len(tets), dtype=bool))
    assert patched.nodes is nodes and patched.tet_cells is tets and patched.tet_kept.all()


def test_pure_tet_copy_is_gone():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("autoflowcfd.grid.mesh_gen.repair.mesh_repair_nonmanifold_patch")
    import autoflowcfd.grid.mesh_gen.repair.mesh_repair_cavity as cavity

    assert not hasattr(cavity, "patch_nonmanifold_cavity")
