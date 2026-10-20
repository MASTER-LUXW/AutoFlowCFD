"""棱柱侧面四边形对角线与 3 四面体拆分的唯一规则（I-139，`mesh_gen/utils/prism_split.py`）。

此前规则是"底面顶点按编号排序后小号底点连大号对应顶点"，只在相邻棱柱对共享侧面的底 / 顶分配一致时协调；外部
导入的体网格（棱柱底 / 顶方向任意）上相邻棱柱可能选不同对角线，共享侧面变成四个单侧"边界"三角形。现在过四边形
最小全局编号顶点，与底 / 顶指定无关；在上层编号大于下层的网格（本项目挤出的 BL）上与旧规则逐位相同。
"""

import numpy as np

from autoflowcfd.grid.mesh_gen.extraction.face_extractor import FaceExtractor
from autoflowcfd.grid.mesh_gen.utils.prism_split import prism_face_triangles, split_prisms_to_tets
from autoflowcfd.grid.schema.grid_nodes import NodeArray

_XYZ = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 0], [1, 1, 1]], float)


def test_neighbour_with_swapped_bottom_and_top_shares_the_side_quad():
    """B 与 A 共享侧面 (1,2,5,4)，但 B 的底 / 顶与 A 相反（外部网格常见）：两侧必须拆出同一对三角形。"""
    prisms = np.array([[0, 1, 2, 3, 4, 5], [4, 5, 7, 1, 2, 6]], dtype=np.int32)
    faces = FaceExtractor.extract_faces_mixed(prisms, np.empty((0, 4), np.int32), NodeArray.from_array(_XYZ),
                                              strict=True)
    assert (faces.n_interior_faces, faces.n_boundary_faces) == (2, 12)


def _tet_boundary(tets):
    faces = {}
    for t in tets:
        for drop in range(4):
            key = tuple(sorted(np.delete(t, drop).tolist()))
            faces[key] = faces.get(key, 0) + 1
    return {k for k, n in faces.items() if n == 1}


def test_split_tiles_the_prism_and_matches_its_face_triangles_for_any_numbering():
    rng = np.random.default_rng(0)
    xyz = _XYZ[:6]
    volume = 0.5
    for _ in range(200):
        label = rng.permutation(6) + rng.integers(0, 50)                # 任意全局编号
        nodes = np.zeros((label.max() + 1, 3))
        nodes[label] = xyz
        prism = label[None, :]
        if rng.random() < 0.5:                                          # 底 / 顶互换存储
            prism = prism[:, [3, 4, 5, 0, 1, 2]]
        tets = split_prisms_to_tets(prism)
        p = nodes[tets]
        vol = np.einsum('ij,ij->i', p[:, 1] - p[:, 0], np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0])) / 6
        assert np.all(np.abs(vol) > 1e-12) and np.isclose(np.abs(vol).sum(), volume)
        assert _tet_boundary(tets) == {tuple(sorted(f)) for f in prism_face_triangles(prism)[0].tolist()}


def test_identical_to_the_previous_rule_when_top_numbers_exceed_bottom():
    """挤出的 BL：上层编号 = 下层编号 + 每层节点数。输出（含面与四面体顺序）与旧规则逐位相同。"""
    rng = np.random.default_rng(1)
    base = np.array([rng.permutation(1000)[:3] for _ in range(500)])
    prisms = np.hstack([base, base + 1000])
    sb = np.sort(base, axis=1)
    v0, v1, v2 = sb.T
    w0, w1, w2 = (sb + 1000).T
    old_tets = np.concatenate([np.stack(t, axis=1) for t in
                               ((v0, v1, v2, w2), (v0, v1, w1, w2), (v0, w0, w1, w2))])
    old_faces = np.stack([np.stack(f, axis=1) for f in
                          ((v0, v1, v2), (w0, w1, w2), (v0, v1, w1), (v0, w0, w1), (v1, v2, w2), (v1, w1, w2),
                           (v0, v2, w2), (v0, w0, w2))], axis=1)
    assert np.array_equal(split_prisms_to_tets(prisms), old_tets)
    assert np.array_equal(prism_face_triangles(prisms), old_faces)
