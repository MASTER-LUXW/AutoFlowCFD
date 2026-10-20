"""validation/mesh_overlap_check.py 的单元测试。

巨大离群面的回归来自一次真实卡死：单个尺寸离群的巨大边界面（cube_demo 粗的远场/计算域外壳面片之一）的粗筛
搜索半径随它自己巨大的尺寸放大，一次查询就能返回几十万个候选（实测 142,944 个，整个检查 6 分钟以上、数 GB
内存）。此前的对策是把任何一个面的候选截断到最近的 2000 个——截断本身可能丢掉真实缺陷。现在广相位
（overlap_geometry.candidate_face_pairs，I-189）只从外接半径较小的一侧查询，大面不查小面，候选数有界且完备。
"""

import numpy as np

from autoflowcfd.grid.validation.mesh_overlap_check import check_face_overlap_and_proximity
from autoflowcfd.grid.schema.grid_nodes import NodeArray
from autoflowcfd.grid.mesh_gen.extraction.face_extractor import FaceExtractor

# 两个在三维里真正相交、不共享顶点的三角形（与
# test_overlap_geometry.py / test_mesh_front_collision.py 用的是同一个夹具）。
A0, A1, A2 = np.array([-2., -2., 0.]), np.array([2., -2., 0.]), np.array([0., 2., 0.])
B0, B1, B2 = np.array([0., 0., -2.]), np.array([0., 0., 2.]), np.array([0., 3., 0.])


def _cap_tet(p0, p1, p2, eps=1e-3):
    """一个薄四面体，其中一个面恰好是 (p0, p1, p2)——另外 3 个面是薄片，
    除了存在之外与测试无关。
    """
    centroid = (p0 + p1 + p2) / 3.0
    normal = np.cross(p1 - p0, p2 - p0)
    normal = normal / np.linalg.norm(normal)
    p3 = centroid + eps * normal
    return np.array([p0, p1, p2, p3])


def _tiny_tet(center, scale=0.01):
    return center + np.array([
        [0.0, 0.0, 0.0],
        [scale, 0.0, 0.0],
        [0.0, scale, 0.0],
        [0.0, 0.0, scale],
    ])


def _huge_tet(center, scale=200.0):
    return _tiny_tet(center, scale=scale)


# 沿 z 放在干扰单元那条线的远下方（干扰单元从不占据那里），所以它自身的
# 范围在几何上碰不到任何其它四面体——只有它的搜索半径（随它自己巨大的尺寸
# 放大）能碰到。经验调出的参数（见 scratchpad/tune_huge_tet.py）：scale=200、
# z 偏移 -600 让它的粗筛候选数远超一个调小的上限，同时不产生任何真实的
# 三角形-三角形相交。
HUGE_TET_CENTER = np.array([40.0, 0.0, -600.0])
HUGE_TET_SCALE = 200.0


def _build_mesh(tets):
    """tets：(4,3) 节点数组的列表，每个是孤立的四面体（四面体之间不共享节点
    ——让每个面都是边界面）。
    """
    all_nodes = np.concatenate(tets, axis=0)
    cells = np.arange(len(all_nodes), dtype=np.int64).reshape(-1, 4)
    return all_nodes, cells


def _report(tets):
    nodes, cells = _build_mesh(tets)
    node_arr = NodeArray(x=nodes[:, 0].copy(), y=nodes[:, 1].copy(), z=nodes[:, 2].copy())
    faces = FaceExtractor.extract_faces(cells.astype(np.int32), node_arr)
    return check_face_overlap_and_proximity(nodes, cells, faces=faces)


class TestHugeOutlierFace:
    def test_small_overlap_found_and_huge_face_does_not_explode_the_candidates(self):
        """远处一张巨大的面（不接触任何东西）：小面对小面的真实重叠照常检出，巨大单元不被牵连，候选对数有界
        （巨大面只与同档或更大的面配对，不会把全部小面当候选）。"""
        tet_a = _cap_tet(A0, A1, A2)
        tet_b = _cap_tet(B0, B1, B2)
        distractors = [_tiny_tet(np.array([10.0 * i, 0.0, 0.0])) for i in range(1, 9)]
        huge = _huge_tet(HUGE_TET_CENTER, scale=HUGE_TET_SCALE)
        tets = [tet_a, tet_b] + distractors + [huge]

        report = _report(tets)

        assert report.has_overlaps
        assert 0 in report.overlapping_cell_ids and 1 in report.overlapping_cell_ids
        assert len(tets) - 1 not in report.overlapping_cell_ids
        n_faces = 4 * len(tets)
        assert report.n_candidate_pairs < n_faces * (n_faces - 1) // 2


class TestSlenderFaces:
    def test_needle_faces_crossing_near_their_tips_are_reported(self):
        """细长面顶点离质心远大于 sqrt(面积)：此前 3 x sqrt(面积) 的搜索半径两侧都查不到对方。"""
        needle_a = _cap_tet(np.array([0., -0.005, 0.]), np.array([0., 0.005, 0.]), np.array([1., 0., 0.]), eps=1e-4)
        needle_b = _cap_tet(np.array([0.95, -0.002, -1.]), np.array([0.95, 0.002, -1.]), np.array([0.95, 0., 1.]),
                            eps=1e-4)
        report = _report([needle_a, needle_b])
        assert set(report.overlapping_cell_ids.tolist()) == {0, 1}


class TestNoOverlapCleanMesh:
    def test_well_separated_tets_report_no_overlaps(self):
        distractors = [_tiny_tet(np.array([10.0 * i, 0.0, 0.0])) for i in range(8)]
        nodes, cells = _build_mesh(distractors)
        node_arr = NodeArray(x=nodes[:, 0].copy(), y=nodes[:, 1].copy(), z=nodes[:, 2].copy())
        faces = FaceExtractor.extract_faces(cells.astype(np.int32), node_arr)

        report = check_face_overlap_and_proximity(nodes, cells, faces=faces)

        assert not report.has_overlaps
        assert len(report.overlapping_cell_ids) == 0
