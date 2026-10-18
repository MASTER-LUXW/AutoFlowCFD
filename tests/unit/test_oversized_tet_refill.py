"""超大四面体的处理（`grid/mesh_gen/tetgen/mesh_tetgen_postprocess.py::fill_without_oversized_tetrahedra`，I-145）。

修复前：对体积超限的四面体做质心 1 拆 4 细分——贴原四面体某面的子四面体底面不变、高度每层缩为 1/4，k 层后纵横比
约 4^k，用必然制造薄片的方式修"体积过大"。现在在超大四面体内部按目标间距插点、交回 tetgen 重新 Delaunay。
"""

import numpy as np
import pytest

from autoflowcfd.grid.mesh_gen.tetgen.mesh_tetgen_core import fill_core_volume
from autoflowcfd.grid.mesh_gen.tetgen.mesh_tetgen_postprocess import (
    _tet_volumes, boundary_limited_tetrahedra, fill_without_oversized_tetrahedra, interior_lattice_points)

_CUBE_PTS = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
_CUBE_TRIS = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                       [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], np.int32)


def _min_dihedral_deg(nodes, tets):
    """每个四面体六条棱处二面角的最小值（度）。"""
    P = nodes[tets]
    out = np.full(len(tets), 180.0)
    faces = [(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)]
    normals = []
    for f, (a, b, c) in enumerate(faces):
        n = np.cross(P[:, b] - P[:, a], P[:, c] - P[:, a])
        inward = P[:, f] - P[:, a]
        n *= np.sign(np.einsum("ij,ij->i", n, inward))[:, None]       # 指向对顶点（内法向）
        normals.append(n / np.linalg.norm(n, axis=1, keepdims=True))
    for i in range(4):
        for j in range(i + 1, 4):
            ang = 180.0 - np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", normals[i], normals[j]), -1, 1)))
            out = np.minimum(out, ang)
    return out


def _fill(extra):
    return fill_core_volume(_CUBE_PTS, _CUBE_TRIS, background_points=extra, verbose=False)


def test_lattice_points_keep_clear_of_the_outer_faces_and_existing_nodes():
    """单个四面体（四个面都是外表面）：点离每个面 >= spacing/2、离顶点 >= spacing/2；两个共面四面体的并集内部共享面
    不留间隙，且点阵全域一致（同一点不会在共享面两侧各出现一次）。"""
    nodes = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2], [2, 2, 2]], float)
    pts = interior_lattice_points(nodes, np.array([[0, 1, 2, 3]]), 0.2)
    assert len(pts) > 10
    dist = np.stack([pts[:, 0], pts[:, 1], pts[:, 2], (2 - pts.sum(axis=1)) / np.sqrt(3)], axis=1)
    assert dist.min() >= 0.1 - 1e-12
    two = interior_lattice_points(nodes, np.array([[0, 1, 2, 3], [1, 2, 3, 4]]), 0.2)
    assert len(np.unique(np.round(two, 12), axis=0)) == len(two)
    shared = np.abs(two.sum(axis=1) - 2) / np.sqrt(3)               # 到共享面 x+y+z=2 的距离
    assert shared.min() < 0.1                                       # 共享面附近照样放点
    tiny = np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]], float)
    np.testing.assert_allclose(interior_lattice_points(tiny, np.array([[0, 1, 2, 3]]), 1.0), tiny.mean(axis=0)[None])


def _refined_cube(m):
    """边界每边剖成 m 段的单位立方体 PLC。"""
    g = np.linspace(0, 1, m + 1)
    pts, idx, tris = [], {}, []

    def pid(p):
        k = tuple(np.round(p, 12))
        if k not in idx:
            idx[k] = len(pts)
            pts.append(p)
        return idx[k]
    for ax in range(3):
        for val in (0.0, 1.0):
            o = [x for x in range(3) if x != ax]
            for i in range(m):
                for j in range(m):
                    q = []
                    for di, dj in ((0, 0), (1, 0), (1, 1), (0, 1)):
                        p = np.zeros(3)
                        p[ax], p[o[0]], p[o[1]] = val, g[i + di], g[j + dj]
                        q.append(pid(p))
                    tris += [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]
    return np.array(pts), np.array(tris, np.int32)


def _interior_worst(nodes, tets, max_volume):
    vols = _tet_volumes(nodes, tets)
    over = np.flatnonzero(vols > max_volume)
    interior = over[~boundary_limited_tetrahedra(nodes, tets[over], max_volume)]
    return float(vols[interior].max()) / max_volume if len(interior) else 0.0


def test_refill_removes_interior_outliers_with_far_fewer_slivers_than_centroid_subdivision():
    """与生产同一配置（区域体积精化 + 保边界 nobisect，阈值为区域目标的 5 倍）：边界比目标尺寸粗 2.4 倍，tetgen
    留下最大 25 倍上限的四面体。修复前的质心 1 拆 4 细分在此造出 719 个二面角 < 5 度的薄片（tetgen 自身 2 个）；
    重填后薄片 <= 1.5%（实测 20 个），体内最大超限倍数降到 1/3 以下（剩余的挨着粗边界，tetgen 在 nobisect 下
    不能在那里插点，再重填不再改善即停止）。"""
    P, F = _refined_cube(2)
    rv = 1e-4
    max_volume = 5 * rv

    def fill(extra):
        return fill_core_volume(P, F, regions=[(np.full(3, 0.5), 1, rv)], background_points=extra, verbose=False,
                                force_preserve_boundary=True)
    n0, t0 = fill(None)
    assert _tet_volumes(n0, t0).max() > 20 * max_volume
    nodes, tets = fill_without_oversized_tetrahedra(fill, None, max_volume, (6 * rv) ** (1 / 3))
    vols = _tet_volumes(nodes, tets)
    assert vols.sum() == pytest.approx(1.0, rel=1e-12)
    np.testing.assert_array_equal(nodes[:len(P)], P)                     # 边界点原样保留
    assert _interior_worst(nodes, tets, max_volume) < _interior_worst(n0, t0, max_volume) / 3   # 20.4 倍 -> 5.3 倍
    assert (_min_dihedral_deg(nodes, tets) < 5.0).mean() <= 0.015


def test_boundary_limited_tetrahedra_are_left_alone():
    """边界面本身大到其上的正四面体都超限：不在体内插点（只会造出薄片），原样返回。"""
    nodes0, tets0 = _fill(None)                                          # 8 点立方体：边界三角形面积 0.5
    calls = []

    def fill(extra):
        calls.append(extra)
        return nodes0, tets0
    nodes, tets = fill_without_oversized_tetrahedra(fill, None, 0.004, target_edge_length=0.15)
    assert len(calls) == 1 and tets is tets0
    assert boundary_limited_tetrahedra(nodes0, tets0, 0.004).all()


def test_stops_when_refilling_no_longer_helps():
    """重填后最大超限倍数不再下降：退回最好的一轮，不无限重跑。"""
    P, F = _refined_cube(2)
    n0, t0 = fill_core_volume(P, F, verbose=False)
    calls = []

    def fill(extra):
        calls.append(extra)
        return n0, t0                                                    # 忽略插入的点
    nodes, tets = fill_without_oversized_tetrahedra(fill, None, 1e-4, target_edge_length=0.05)
    assert len(calls) == 2 and tets is t0
