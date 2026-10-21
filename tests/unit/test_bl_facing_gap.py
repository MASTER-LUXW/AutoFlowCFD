"""BL 厚度上限按"法向前方最近面片"测量（I-148，`mesh_gen/tetgen/facing_gap.py`）。

此前 `compute_local_thickness_limit` 按"锥内最近节点"测量：对面是大三角形、顶点都在搜索半径或锥外时（稀疏远场面、
细长间隙对面的粗面片）完全检测不到，挤出层可以穿透对面。
"""

import numpy as np

from autoflowcfd.grid.mesh_gen.tetgen.facing_gap import (
    _ray_hit_distance, closest_points_on_triangles, facing_gap,
)
from autoflowcfd.grid.mesh_gen.tetgen.mesh_tetgen_seam import compute_local_thickness_limit


def _plate_under_big_slab(rng, n_junk=20):
    """细分平板（z=0，外法向 +z）+ 上方只有两个三角形的大斜板（顶点远在搜索半径外）+ n_junk 个随机杂乱三角形。"""
    gx, gy = np.meshgrid(np.linspace(-1, 1, 9), np.linspace(-1, 1, 9), indexing="ij")
    plate = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(81)])
    pf = []
    for i in range(8):
        for j in range(8):
            v = i * 9 + j
            pf += [[v, v + 9, v + 10], [v, v + 10, v + 1]]
    slab = np.array([[-50, -50, -2.2], [50, -50, 2.8], [50, 50, 2.8], [-50, 50, -2.2]])   # z = 0.3 + 0.05 x
    junk = rng.uniform(-2, 2, size=(3 * n_junk, 3))
    nodes = np.vstack([plate, slab, junk])
    faces = np.vstack([pf, [[81, 82, 83], [81, 83, 84]], 85 + np.arange(3 * n_junk).reshape(n_junk,
                                                                                            3)]).astype(np.int64)
    return nodes, faces, np.array(pf, dtype=np.int64)


def _brute(nodes, faces, idx, normals, radius, angle):
    cos_th = np.cos(np.radians(angle))
    a, b, c = nodes[faces[:, 0]], nodes[faces[:, 1]], nodes[faces[:, 2]]
    out = np.full(len(idx), np.inf)
    for k, i in enumerate(idx):
        keep = ~(faces == i).any(axis=1)
        p = np.repeat(nodes[i][None], keep.sum(), axis=0)
        n = np.repeat(normals[k][None], keep.sum(), axis=0)
        d = closest_points_on_triangles(p, a[keep], b[keep], c[keep]) - p
        dist = np.linalg.norm(d, axis=1)
        ahead = (dist > 0) & (np.einsum('ij,ij->i', d, n) >= cos_th * dist) & (dist <= radius)
        hit = _ray_hit_distance(p, n, a[keep], b[keep], c[keep])
        out[k] = min(np.where(ahead, dist, np.inf).min(), np.where(hit <= radius, hit, np.inf).min())
    return out


def test_candidate_search_is_exact_against_brute_force():
    rng = np.random.default_rng(0)
    nodes, faces, _ = _plate_under_big_slab(rng)
    idx = np.arange(81)
    normals = np.tile([0.0, 0.0, 1.0], (81, 1))
    for radius, angle in ((1.0, 60.0), (0.5, 45.0), (2.0, 75.0)):
        gap, _ = facing_gap(nodes, faces, idx, normals, radius, angle)
        ref = _brute(nodes, faces, idx, normals, radius, angle)
        assert np.array_equal(np.isinf(gap), np.isinf(ref))
        assert np.allclose(gap[np.isfinite(ref)], ref[np.isfinite(ref)], rtol=0, atol=1e-14)


def test_large_facing_triangle_with_far_vertices_caps_the_bl():
    """斜板距平板中心 0.3 cos(atan 0.05) ≈ 0.2996，四个顶点都在 50 m 外：必须限制 BL 厚度（此前按最近节点为 inf）。"""
    nodes, faces, plate_faces = _plate_under_big_slab(np.random.default_rng(0), n_junk=0)
    domain = 10.0                                                     # 搜索半径 0.08 * 10 = 0.8
    limit = compute_local_thickness_limit(nodes, faces, plate_faces, np.arange(81), domain)
    assert np.isclose(limit[40], 0.45 * 0.3 / np.sqrt(1 + 0.05 ** 2), rtol=1e-12)
