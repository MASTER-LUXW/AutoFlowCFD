"""BL 挤出前的尖锐拐角拆分（`grid/mesh_gen/utils/mesh_corner_split.py::split_sharp_corners`，I-205）。

立方体表面（每面 2 个三角形）：8 个顶点都是 3 个曲面片交汇的拐角、12 条棱都是硬边。拆分后每个拐角每个曲面片
一份副本（与原顶点同位置），每条硬边一条倒角（2 个三角形）、每个拐角一个帽盖三角形，结果拓扑水密。倒角 / 帽盖的
环绕方向不参与下游（棱柱按排序后的底面构造、体积取绝对值，tetgen 的 PLC 不要求朝向，挤出法向只用真实面），
所以这里不断言朝向。
"""

from collections import Counter

import numpy as np
import pytest

from autoflowcfd.grid.mesh_gen.utils.mesh_corner_split import split_sharp_corners


def _cube():
    nodes = np.array([[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)])
    faces = []
    for axis in range(3):
        u, v = [a for a in range(3) if a != axis]
        for side in (0, 1):
            quad = []
            for a, b in ((0, 0), (1, 0), (1, 1), (0, 1)):
                c = [0, 0, 0]
                c[axis], c[u], c[v] = side, a, b
                quad.append(4 * c[0] + 2 * c[1] + c[2])
            faces += [[quad[0], quad[1], quad[2]], [quad[0], quad[2], quad[3]]]
    return nodes, np.array(faces)


def _undirected_edge_counts(faces):
    return Counter(tuple(sorted(e)) for f in faces for e in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])))


def test_cube_corners_split_per_patch_and_stay_watertight():
    nodes, faces = _cube()
    new_nodes, topo, real, orig = split_sharp_corners(nodes, faces)

    assert len(new_nodes) == 8 + 8 * 2                     # 每个拐角 3 份（原顶点 + 2 个副本）
    np.testing.assert_array_equal(np.bincount(orig), np.full(8, 3))
    np.testing.assert_array_equal(new_nodes, nodes[orig])   # 副本从原顶点位置出发
    np.testing.assert_array_equal(new_nodes[:8], nodes)     # 只追加、原索引不变

    assert real.tolist() == [True] * 12 + [False] * 32      # 12 条倒角 x 2 + 8 个帽盖
    np.testing.assert_array_equal(orig[topo[:12]], faces)   # 真实面只重映射到本曲面片的副本
    patch_nodes = [set(topo[2 * k:2 * k + 2].ravel()) for k in range(6)]
    assert all(not (patch_nodes[i] & patch_nodes[j]) for i in range(6) for j in range(i + 1, 6))
    caps = [f for f in topo[12:] if len(set(orig[f])) == 1]
    assert len(caps) == 8 and sorted(int(orig[f[0]]) for f in caps) == list(range(8))

    assert set(_undirected_edge_counts(topo).values()) == {2}   # 水密：每条边恰好两个面


def test_smooth_surface_is_left_alone():
    nodes, faces = _cube()
    new_nodes, topo, real, orig = split_sharp_corners(nodes, faces, threshold=np.deg2rad(120.0))
    np.testing.assert_array_equal(new_nodes, nodes)
    np.testing.assert_array_equal(topo, faces)
    assert real.all() and np.array_equal(orig, np.arange(8))


@pytest.mark.parametrize("radius, split", [(0.5, False), (1.0, True)])
def test_min_feature_radius_separates_coarse_curves_from_creases(radius, split):
    """单位立方体棱：弦长 1、二面角 90 度，隐含半径 1/(2 sin 45°) = 0.707；不小于 min_feature_radius 时按光滑曲面处理。"""
    nodes, faces = _cube()
    new_nodes = split_sharp_corners(nodes, faces, min_feature_radius=radius)[0]
    assert (len(new_nodes) > 8) is split


def test_empty_input():
    nodes, _ = _cube()
    new_nodes, topo, real, orig = split_sharp_corners(nodes, np.zeros((0, 3), dtype=np.int64))
    np.testing.assert_array_equal(new_nodes, nodes)
    assert topo.shape == (0, 3) and real.shape == (0,)
    np.testing.assert_array_equal(orig, np.arange(8))
