# -*- coding: utf-8 -*-
"""棱柱四边形侧面被两个不同四面体的三角形覆盖时，棱柱一侧通量点的两个来源（`fr/face_flux_points/merge.py`）。

四边形侧面恒按对角线拆成两条三角形子面记录；只有分组首条（primary）做外插，它的通量点按对角线分到两半：
首条子面那一半从它的邻居（src0）插值，另一半从另一条子面的邻居（src1）插值。2026-10-15 之前 src1 的**插值
矩阵**按另一条子面的邻居构造，记录的**单元**却是首条子面自己的邻居——另一半通量点用的是 A 四面体的解点值
配 B 四面体的插值权重。两侧的公共通量因此不相等（四面体那一侧读的是棱柱的值），离散不守恒。棱柱-棱柱的
共形四边形面上两条子面的邻居是同一个棱柱，看不出来；cube_demo 粗网格（28.6 万单元）上有 297 个棱柱四边形
侧面受影响（边界层边缘处棱柱侧面挨着四面体），表现为湍流块耦合模式里 297 个单向耦合。

判据：位置场是线性的，棱柱与四面体的插值都精确，于是棱柱侧每个通量点由两个来源插值出的"另一侧"坐标
必须等于它自己外插出的坐标（标准与翻转两种对角线走向各测一次，order 1、2）。
"""

from types import SimpleNamespace

import numpy as np
import pytest

from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
from autoflowcfd.grid.high_order import HighOrderMesh
from tests.unit.test_fr_residual_inviscid import _MockCells, _MockNodes

#: 棱柱局部节点坐标；四边形侧面是局部节点 (0, 1, 4, 3)，在 y = 0 平面上，四面体顶点在 y < 0 一侧
PRISM_COORDS = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1]]
APEX = [0.5, -0.8, 0.5]
#: 棱柱四边形侧面的对角线由全局节点编号决定（`face_flux_points/data.py::_classify_half`）：两种编号各给出
#: "标准"（局部 0-4）与"翻转"（局部 1-3）对角线，四面体按同一条对角线剖分（共形）。
#: 每项：棱柱局部节点 -> 全局编号、按局部节点写的两个四面体（6 为顶点）
LAYOUTS = {"standard": ([0, 1, 2, 3, 4, 5], [[0, 1, 4, 6], [0, 4, 3, 6]]),
           "flipped": ([1, 0, 2, 3, 4, 5], [[0, 1, 3, 6], [1, 4, 3, 6]])}


def _mesh(order, layout):
    gid, tets_local = LAYOUTS[layout]
    nodes = np.empty((7, 3))
    for local, g in enumerate(gid):
        nodes[g] = PRISM_COORDS[local]
    nodes[6] = APEX
    to_global = gid + [6]
    tets = np.array([[to_global[i] for i in t] for t in tets_local], dtype=np.int32)
    mesh = HighOrderMesh(order=order)
    mesh.load_from_volume_mesh(SimpleNamespace(
        cell_count=3, nodes=_MockNodes(nodes), cells=_MockCells(tets),
        prism_cells=_MockCells(np.array([gid], dtype=np.int32))))
    fc = mesh.face_connectivity
    mesh.boundary_groups = {"farfield": np.unique(fc.owner_cell[fc.is_boundary]).astype(np.int64)}
    mesh.boundary_bc_types = {"farfield": "FARFIELD"}
    return mesh


@pytest.mark.parametrize("layout", sorted(LAYOUTS))
@pytest.mark.parametrize("order", [1, 2])
def test_prism_side_sources_reproduce_its_own_flux_point_positions(order, layout):
    mesh = _mesh(order, layout)
    flat = get_flat_face_geometry(mesh, mesh.operators)
    X = np.asarray(mesh.sps_coords).reshape(mesh.n_cells, mesh.n_sps_per_cell, 3)
    prism, tets = 0, {1, 2}
    checked = 0
    for f in range(flat.n_faces):
        if flat.is_boundary[f] or flat.owner_cell[f] != prism or not flat.owner_is_primary[f]:
            continue
        if flat.neighbor_cell[f] not in tets:
            continue
        i1 = int(flat.neighbor_src1_idx[f])
        c0 = int(flat.neighbor_src0_cell[f])
        assert i1 >= 0, "两个四面体覆盖的四边形侧面必须有第二个来源"
        c1 = int(flat.neighbor_src1_cell[i1])
        assert {c0, c1} == tets, f"两个来源应是两个不同的四面体，实为 {c0}, {c1}"
        own = flat.boundary_extrap_native[flat.owner_face_op[f]] @ X[prism]
        other = (flat.neighbor_src0_tpl[flat.neighbor_src0_tid[f]] @ X[c0]
                 + flat.neighbor_src1_mat[i1] @ X[c1])
        np.testing.assert_allclose(other, own, atol=1e-10)
        checked += 1
    assert checked == 1
