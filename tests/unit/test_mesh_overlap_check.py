"""validation/mesh_overlap_check.py 的单元测试。

CANDIDATE_CAP_PER_FACE 的测试是一次真实卡死的回归：单个尺寸离群的巨大
边界面（cube_demo 粗的远场/计算域外壳面片之一）的粗筛搜索半径随它自己
巨大的尺寸放大，**一次**查询就能返回几十万个候选——实测 142,944 个，把
一个 500 面的分块撑到 558 万候选对，整个检查要 6 分钟以上、数 GB 内存。
修复是把任何一个面的候选集截断到最近的 CAP 个邻居。这些测试确认：网格里
同时存在一张不相干的巨大面时，截断不会让真正的小面对小面的重叠被漏掉。
"""

import numpy as np
import pytest

from autoflowcfd.grid.validation import mesh_overlap_check
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


class TestCandidateCapPreservesCorrectness:
    def test_small_overlap_still_found_next_to_a_huge_distractor_face(self, monkeypatch):
        """一张巨大的面（在远处，不接触任何东西）迫使它自己的某次粗筛查询被截断；
        网格里别处真正的小面对小面重叠仍必须被检出。
        """
        monkeypatch.setattr(mesh_overlap_check, "CANDIDATE_CAP_PER_FACE", 3)

        tet_a = _cap_tet(A0, A1, A2)
        tet_b = _cap_tet(B0, B1, B2)

        # 几个彼此分开、互不重叠的小四面体，用来填充那张巨大面自己那次查询的
        # 候选数。
        distractors = [_tiny_tet(np.array([10.0 * i, 0.0, 0.0])) for i in range(1, 9)]

        # 放在"簇中心"附近（在巨大面自己过大的搜索半径之内、靠近其余所有单元），
        # 但离任何单个四面体都足够远，在几何上从不与之相交。
        huge = _huge_tet(HUGE_TET_CENTER, scale=HUGE_TET_SCALE)

        tets = [tet_a, tet_b] + distractors + [huge]
        nodes, cells = _build_mesh(tets)

        node_arr = NodeArray(x=nodes[:, 0].copy(), y=nodes[:, 1].copy(), z=nodes[:, 2].copy())
        faces = FaceExtractor.extract_faces(cells.astype(np.int32), node_arr)

        report = check_face_overlap_and_proximity(nodes, cells, faces=faces)

        assert report.has_overlaps
        # tet_a 是单元 0，tet_b 是单元 1（按 `cells` 的构造顺序）。
        assert 0 in report.overlapping_cell_ids
        assert 1 in report.overlapping_cell_ids
        # 巨大的干扰单元不接触任何东西，不能因为截断引起的误报而被牵连进来。
        huge_cell_id = len(tets) - 1
        assert huge_cell_id not in report.overlapping_cell_ids

    def test_cap_actually_engages_for_the_huge_face(self, monkeypatch):
        """对测试夹具本身的检查：不截断时，巨大面自己的查询确实超过一个小的上限
        （否则上面的测试根本没有走到截断路径）。
        """
        monkeypatch.setattr(mesh_overlap_check, "CANDIDATE_CAP_PER_FACE", 3)

        tet_a = _cap_tet(A0, A1, A2)
        tet_b = _cap_tet(B0, B1, B2)
        distractors = [_tiny_tet(np.array([10.0 * i, 0.0, 0.0])) for i in range(1, 9)]
        huge = _huge_tet(HUGE_TET_CENTER, scale=HUGE_TET_SCALE)

        tets = [tet_a, tet_b] + distractors + [huge]
        nodes, cells = _build_mesh(tets)
        node_arr = NodeArray(x=nodes[:, 0].copy(), y=nodes[:, 1].copy(), z=nodes[:, 2].copy())
        faces = FaceExtractor.extract_faces(cells.astype(np.int32), node_arr)

        boundary_idx = faces.get_boundary_face_indices()
        centroids = faces.center[boundary_idx]
        face_size = np.sqrt(np.maximum(faces.area[boundary_idx], 1e-300))
        from scipy.spatial import cKDTree
        tree = cKDTree(centroids)
        search_radius = 3.0 * face_size
        counts = np.array([
            len(tree.query_ball_point(centroids[i], r=search_radius[i]))
            for i in range(len(boundary_idx))
        ])
        assert counts.max() > 3, "fixture must produce a face whose uncapped candidate count exceeds the patched cap"


class TestNoOverlapCleanMesh:
    def test_well_separated_tets_report_no_overlaps(self):
        distractors = [_tiny_tet(np.array([10.0 * i, 0.0, 0.0])) for i in range(8)]
        nodes, cells = _build_mesh(distractors)
        node_arr = NodeArray(x=nodes[:, 0].copy(), y=nodes[:, 1].copy(), z=nodes[:, 2].copy())
        faces = FaceExtractor.extract_faces(cells.astype(np.int32), node_arr)

        report = check_face_overlap_and_proximity(nodes, cells, faces=faces)

        assert not report.has_overlaps
        assert len(report.overlapping_cell_ids) == 0
