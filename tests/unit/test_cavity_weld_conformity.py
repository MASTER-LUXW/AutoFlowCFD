"""混合网格空腔修补的焊接必须保持网格封闭（2026-10-09，`mesh_repair_nonmanifold_mixed_weld.py`）。

此前空腔边界上的近重合点对只在空腔自己的重铺里合并，空腔外引用被合并点的单元不变：两侧的面对不上，各剩
一个外部面；被丢弃的退化面在外部单元一侧也成了外部面。修补照样报告成功。cube_demo 实测体网格 39574 个外部
面、面网格 39488 个三角形，多出的 86 个里 27 对共享两顶点、第三顶点相距 1.2~1.7 mm（= 焊接容差，局部中位
棱长的 10%）。

这里用结构化四面体立方体复现：把一个内部节点挪到相邻节点 5% 棱长处（撕裂点对），把含它的一个单元标记为坏，
空腔修补之后外表面必须与修补前逐面相同。
"""

import numpy as np

from autoflowcfd.grid.mesh_gen.repair.mesh_repair_cavity_shared import (
    _glued_face_pairs,
    _weld_near_coincident_boundary_points,
    retile_is_conformal,
)
from autoflowcfd.grid.mesh_gen.repair.mesh_repair_nonmanifold_mixed import patch_nonmanifold_cavity_mixed
from autoflowcfd.grid.mesh_gen.tetgen.mesh_prism_to_tet import orient_tetrahedra

_KUHN = [(0, 1, 3, 7), (0, 1, 5, 7), (0, 2, 3, 7), (0, 2, 6, 7), (0, 4, 5, 7), (0, 4, 6, 7)]


def _cube_tets(n):
    """[0,1]^3 结构化网格，每个六面体按 Kuhn 剖分成 6 个四面体（全局一致，共形）。"""
    g = np.linspace(0.0, 1.0, n + 1)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    nodes = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    def nid(i, j, k):
        return (i * (n + 1) + j) * (n + 1) + k

    tets = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                c = [nid(i + (b >> 2 & 1), j + (b >> 1 & 1), k + (b & 1)) for b in range(8)]
                tets += [[c[t[0]], c[t[1]], c[t[2]], c[t[3]]] for t in _KUHN]
    return nodes, np.array(tets, dtype=np.int64), nid


def _exterior(tets):
    f = np.vstack([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]], tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]])
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 0] != f[:, 2]) & (f[:, 1] != f[:, 2])]
    uniq, counts = np.unique(np.sort(f, axis=1), axis=0, return_counts=True)
    return {tuple(r) for r in uniq[counts == 1]}, int((counts > 2).sum())


def _torn_cube():
    n = 6
    nodes, tets, nid = _cube_tets(n)
    a, b = nid(2, 3, 3), nid(3, 3, 3)
    nodes[a] = nodes[b] - np.array([0.05 / n, 0.0, 0.0])      # 撕裂点对：相距 5% 棱长
    return nodes, orient_tetrahedra(nodes, tets), a, b


def test_cavity_patch_with_weld_keeps_the_mesh_closed():
    nodes, tets, a, b = _torn_cube()
    before, _ = _exterior(tets)
    keep = np.ones(len(tets), dtype=bool)
    keep[np.flatnonzero((tets == a).any(axis=1))[0]] = False
    new_nodes, _, new_tets = patch_nonmanifold_cavity_mixed(
        nodes, np.empty((0, 6), dtype=np.int64), tets, np.ones(0, dtype=bool), keep)
    after, n_nonmanifold = _exterior(np.asarray(new_tets, dtype=np.int64))
    assert n_nonmanifold == 0
    assert after == before                       # 修补前 432 个外部面；旧实现修补后 436 个
    referenced = set(np.unique(new_tets))
    assert len({a, b} & referenced) == 1         # 撕裂点对合并成一个点，全部单元都改引用幸存点
    p0 = new_nodes[new_tets[:, 0]]
    vol = np.einsum("ij,ij->i", new_nodes[new_tets[:, 1]] - p0,
                    np.cross(new_nodes[new_tets[:, 2]] - p0, new_nodes[new_tets[:, 3]] - p0)) / 6.0
    assert abs(vol.sum() - 1.0) < 1e-12          # 体积守恒（单位立方体）
    assert (np.abs(vol) > 0).all()


def test_protected_points_survive_and_two_protected_points_never_merge():
    pts = np.array([[0, 0, 0], [1e-3, 0, 0], [1, 0, 0], [0, 1, 0], [1 + 1e-3, 0, 0]], dtype=float)
    faces = np.array([[0, 2, 3], [1, 3, 2], [2, 3, 4]], dtype=np.int32)
    gpts = np.array([10, 11, 12, 13, 14])
    protected = np.zeros(20, dtype=bool)
    protected[[11, 12, 14]] = True
    _, _, _, removed, survivor = _weld_near_coincident_boundary_points(pts, faces, gpts, 0.1, protected)
    assert list(removed) == [10] and list(survivor) == [11]      # 受保护的 11 当幸存者
    assert 12 not in removed and 14 not in removed              # 12、14 都受保护：不合并


def test_glued_pairs_need_opposite_orientation():
    faces = np.array([[0, 1, 2], [0, 2, 1], [3, 4, 5], [3, 4, 5]])
    assert list(_glued_face_pairs(faces)) == [True, True, False, False]   # 同向重合是折叠，不粘合


def test_retile_conformity_rejects_a_split_boundary_face():
    # 单个四面体：外表面就是输入的 4 个面
    tet = np.array([[0, 1, 2, 3]])
    faces = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])
    assert retile_is_conformal(tet, 4, faces)
    # 在面 (1,2,3) 上插一个 Steiner 点 4（tetgen 不保边界时的样子）：外表面多了点 4 的面
    split = np.array([[0, 1, 2, 4], [0, 2, 3, 4], [0, 3, 1, 4]])
    assert not retile_is_conformal(split, 4, faces)
