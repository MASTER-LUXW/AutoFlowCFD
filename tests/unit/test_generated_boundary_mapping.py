"""生成器边界映射：外部面落在面网格三角形上、恰好覆盖输入表面（2026-10-09，`mesh_boundary.map_generated_boundaries`）。

此前只用四面体求"只出现一次的面"、按节点编号去匹配面网格：贴着棱柱顶面的四面体面被当成边界面，重合点去重
又压缩了节点编号——cube_demo 实测 24689/40966 个"边界单元"进了 UNCLASSIFIED（按 WALL），网格缺口处的外部面
有的当壁面、有的没有任何分组。现在外部面在完整混合网格上求、按几何包含关系对应面网格三角形并校验覆盖面积
（tetgen 在边界上插点时外部面是面网格三角形的细分，同样要能通过），对不上就报错。
"""

import numpy as np
import pytest

from autoflowcfd.grid.mesh_gen.utils.mesh_boundary import map_generated_boundaries
from autoflowcfd.grid.schema.grid_boundaries import BoundaryMap
from tests.unit.test_cavity_weld_conformity import _cube_tets, _exterior


def _cube_with_surface(n=3):
    nodes, tets, _nid = _cube_tets(n)
    faces = np.array(sorted(_exterior(tets)[0]))
    # 面网格：外表面三角形另起一份节点编号（打乱顺序），按所在的立方体面分组
    perm = np.random.default_rng(0).permutation(len(nodes))
    inv = np.argsort(perm)
    surf_nodes = nodes[perm]
    surf_faces = inv[faces]
    c = nodes[faces].mean(axis=1)
    groups, bc = {}, {}
    for axis, name in enumerate("xyz"):
        for side, value in (("min", 0.0), ("max", 1.0)):
            key = f"{name}_{side}"
            groups[key] = np.flatnonzero(np.isclose(c[:, axis], value)).astype(np.int32)
            bc[key] = "WALL" if name == "z" else "SYMMETRY"
    return nodes, tets, surf_nodes, surf_faces, BoundaryMap(groups=groups, bc_types=bc)


def test_groups_follow_coordinates_not_node_numbering():
    nodes, tets, sn, sf, sb = _cube_with_surface()
    bm = map_generated_boundaries(nodes, np.empty((0, 6), np.int64), tets, sn, sf, sb)
    assert set(bm.groups) == set(sb.groups)
    assert bm.bc_types == sb.bc_types
    centers = nodes[tets].mean(axis=1)
    for name, cells in bm.groups.items():
        axis, value = "xyz".index(name[0]), (0.0 if name.endswith("min") else 1.0)
        # 拥有该组外部面的单元都贴着那个立方体面（单元中心离它不到一个网格步长）
        assert np.all(np.abs(centers[cells, axis] - value) < 1.0 / 3)


def test_interior_gap_is_an_error():
    nodes, tets, sn, sf, sb = _cube_with_surface()
    interior = np.flatnonzero(np.all((nodes[tets] > 0.0) & (nodes[tets] < 1.0), axis=(1, 2)))
    assert len(interior)
    with pytest.raises(ValueError, match="中 4 个不在任何面网格三角形上") as err:
        map_generated_boundaries(nodes, np.empty((0, 6), np.int64), np.delete(tets, interior[0], axis=0),
                                 sn, sf, sb)
    assert "覆盖面积" not in str(err.value)    # 内部缺口不改变外表面的覆盖


def test_moved_surface_point_is_an_error():
    nodes, tets, sn, sf, sb = _cube_with_surface()
    moved = nodes.copy()
    k = int(np.flatnonzero(np.isclose(nodes[:, 2], 0.0) & (nodes[:, 0] > 0) & (nodes[:, 1] > 0))[0])
    moved[k, 0] += 1e-4                     # 修补在表面内移动了一个表面点：仍在平面上，但覆盖关系变了
    with pytest.raises(ValueError, match="覆盖面积与自身面积不符") as err:
        map_generated_boundaries(moved, np.empty((0, 6), np.int64), tets, sn, sf, sb)
    assert "不在任何面网格三角形上" not in str(err.value)
    moved = nodes.copy()
    moved[k, 2] += 1e-4                     # 移出表面
    with pytest.raises(ValueError, match="不在任何面网格三角形上"):
        map_generated_boundaries(moved, np.empty((0, 6), np.int64), tets, sn, sf, sb)


def test_subdivided_boundary_faces_are_accepted():
    """tetgen 在边界上插 Steiner 点时外部面是面网格三角形的细分：仍然恰好覆盖，必须通过、组不变。"""
    nodes, tets, sn, sf, sb = _cube_with_surface()
    ext = sorted(_exterior(tets)[0])
    face = np.array(ext[0])
    owner = int(np.flatnonzero(np.isin(tets, face).sum(axis=1) == 3)[0])
    apex = [v for v in tets[owner] if v not in face][0]
    steiner = len(nodes)
    nodes = np.vstack([nodes, nodes[face].mean(axis=0)])
    a, b, c = face
    split = np.array([[a, b, steiner, apex], [b, c, steiner, apex], [c, a, steiner, apex]])
    tets = np.vstack([np.delete(tets, owner, axis=0), split])
    bm = map_generated_boundaries(nodes, np.empty((0, 6), np.int64), tets, sn, sf, sb)
    assert set(bm.groups) == set(sb.groups)
