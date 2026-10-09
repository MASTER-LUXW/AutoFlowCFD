"""边界面预细分必须共形（2026-10-09，`mesh_background_merge_utils._refine_large_boundary_faces`）。

此前每个面只二分自己的最长边、各自新建中点：相邻面不在同一点剖分（悬挂节点），两个面都拆同一条边时还会出现
两个重合的中点——交给 tetgen 的表面不封闭。现在每条超长边一个共享中点，面按被标记的边数拆成 2/3/4 个子三角形。
"""

import numpy as np

from autoflowcfd.grid.mesh_gen.background.mesh_background_merge_utils import _refine_large_boundary_faces


def _closed_box():
    v = np.array([[x, y, z] for x in (0.0, 3.0) for y in (0.0, 1.0) for z in (0.0, 0.4)])
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    tris = []
    for a, b, c, d in quads:
        tris += [(a, b, c), (a, c, d)]
    return v, np.array(tris, dtype=np.int64)


def _edge_use(faces):
    e = np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    return counts


def _area_and_volume(v, f):
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    cr = np.cross(b - a, c - a)
    return 0.5 * np.linalg.norm(cr, axis=1).sum(), np.einsum("ij,ij->i", a, cr).sum() / 6.0


def test_refined_closed_surface_stays_watertight_and_consistent():
    v, f = _closed_box()
    area0, vol0 = _area_and_volume(v, f)
    nv, nf = _refine_large_boundary_faces(v, f, 0.35)
    assert (_edge_use(nf) == 2).all()                         # 封闭流形：无悬挂节点、无重合中点
    assert len(np.unique(np.round(nv, 12), axis=0)) == len(nv)  # 没有重合的点
    area1, vol1 = _area_and_volume(nv, nf)
    assert abs(area1 - area0) < 1e-12 * area0                 # 细分不改变表面
    assert abs(vol1 - vol0) < 1e-12 * abs(vol0)               # 有向体积不变 => 绕向保持一致
    e = np.vstack([nf[:, [0, 1]], nf[:, [1, 2]], nf[:, [2, 0]]])
    assert np.linalg.norm(nv[e[:, 0]] - nv[e[:, 1]], axis=1).max() <= 0.35


def test_no_long_edges_is_a_no_op():
    v, f = _closed_box()
    nv, nf = _refine_large_boundary_faces(v, f, 10.0)
    assert len(nv) == len(v) and np.array_equal(np.sort(nf, axis=1), np.sort(f, axis=1))
