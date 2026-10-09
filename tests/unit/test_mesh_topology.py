"""生成的体网格的拓扑与有效性测试。

这些测试针对网格审计里发现的网格生成缺陷：

* 棱柱 -> 四面体的拆分必须**协调**（每个内部面恰好被两个单元共享）。
  盲目套用固定模板会产生悬挂面，有限体积的面提取器会把它们当成边界面，
  把内部区域静默地变成壁面。
* 生成的四面体必须有**正的有符号体积**，这样翻转的单元才能被检出，
  而不是被 ``abs()`` 掩盖。
"""

import numpy as np
import pytest

from autoflowcfd.grid.mesh_gen.tetgen.mesh_prism_to_tet import (
    convert_layers_to_tetrahedra, orient_tetrahedra,
)


def _flat_patch(nx=3, ny=3, lx=1.0, ly=1.0):
    """z=0 平面上三角化的平面片。"""
    xs = np.linspace(0.0, lx, nx + 1)
    ys = np.linspace(0.0, ly, ny + 1)
    nodes, idx = [], {}
    for j in range(ny + 1):
        for i in range(nx + 1):
            idx[(i, j)] = len(nodes)
            nodes.append((xs[i], ys[j], 0.0))
    nodes = np.array(nodes, dtype=np.float64)

    faces = []
    for j in range(ny):
        for i in range(nx):
            a = idx[(i, j)]
            b = idx[(i + 1, j)]
            c = idx[(i + 1, j + 1)]
            d = idx[(i, j + 1)]
            # 每个四边形拆成两个三角形，故意混用绕向，让测试不依赖整齐的输入顺序。
            faces.append([a, b, c])
            faces.append([c, d, a])
    return nodes, np.array(faces, dtype=np.int64)


def _stack_layers(base_nodes, n_layers=4, dz=0.1):
    """把这片堆成 n_layers 层，返回 all_nodes。"""
    layers = [base_nodes + np.array([0.0, 0.0, k * dz]) for k in range(n_layers)]
    return np.vstack(layers)


def _face_occurrence_counts(tets):
    """统计每个三角形面属于几个单元。"""
    templates = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64)
    faces = tets[:, templates].reshape(-1, 3)
    faces = np.sort(faces, axis=1)
    _, counts = np.unique(faces, axis=0, return_counts=True)
    return counts


class TestPrismSplitConformality:
    # 注意：下面的 layer_conn 总是 n_layers - 1 项（每个挤出**步**一项），
    # 不是 n_layers 项（每层节点一项）——见 convert_layers_to_tetrahedra 的
    # layer_connectivity 文档。这里传 n_layers 项曾让内部的 nodes_per_layer 静默
    # 算错（差一），协调性/封闭面检查恰好注意不到，但下面的体积检查能发现。
    def test_every_face_shared_by_at_most_two_cells(self):
        """决定性的协调性检查。

        有效的体网格里，一个面要么在边界上（1 个单元）要么在内部（2 个单元）。
        计数 3 以上说明相邻棱柱在对角线上不一致，即网格不协调。
        """
        base_nodes, base_faces = _flat_patch()
        n_layers = 4
        all_nodes = _stack_layers(base_nodes, n_layers)
        layer_conn = [base_faces.copy() for _ in range(n_layers - 1)]

        tets, _face_of_tet = convert_layers_to_tetrahedra(all_nodes, layer_conn, base_faces)
        counts = _face_occurrence_counts(tets)

        assert counts.max() <= 2, (
            f"non-conformal mesh: {int(np.count_nonzero(counts > 2))} faces are "
            f"shared by more than 2 cells (max={counts.max()})"
        )

    def test_boundary_faces_form_closed_surface(self):
        """只属于一个单元的面必须围成整个体积。

        封闭面的外向面积矢量之和为零。拆分不协调时，内部会出现伪"边界"面，
        这个和就不会抵消。
        """
        base_nodes, base_faces = _flat_patch()
        n_layers = 4
        all_nodes = _stack_layers(base_nodes, n_layers)
        layer_conn = [base_faces.copy() for _ in range(n_layers - 1)]
        tets, _face_of_tet = convert_layers_to_tetrahedra(all_nodes, layer_conn, base_faces)

        templates = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]],
                             dtype=np.int64)
        faces = tets[:, templates].reshape(-1, 3)
        keys = np.sort(faces, axis=1)
        uniq, inverse, counts = np.unique(keys, axis=0, return_inverse=True,
                                          return_counts=True)
        boundary_positions = np.where(counts[inverse] == 1)[0]
        bf = faces[boundary_positions]

        p0, p1, p2 = all_nodes[bf[:, 0]], all_nodes[bf[:, 1]], all_nodes[bf[:, 2]]
        raw = np.cross(p1 - p0, p2 - p0)          # 2 * area * normal
        areas = 0.5 * np.linalg.norm(raw, axis=1)
        unit = raw / np.maximum(2.0 * areas, 1e-30)[:, None]

        # 把每个边界面相对所属单元定向为朝外。
        owner = np.repeat(np.arange(len(tets)), 4)[boundary_positions]
        centroids = all_nodes[tets].mean(axis=1)
        fc = (p0 + p1 + p2) / 3.0
        flip = np.einsum('ij,ij->i', unit, fc - centroids[owner]) < 0
        unit[flip] *= -1.0

        total = (unit * areas[:, None]).sum(axis=0)
        scale = areas.sum()
        assert np.linalg.norm(total) / scale < 1e-10, (
            f"boundary faces do not form a closed surface: residual "
            f"{np.linalg.norm(total)/scale:.3e} (mesh is leaking)"
        )

    def test_all_volumes_positive(self):
        """朝向修复之后有符号体积必须为正。"""
        base_nodes, base_faces = _flat_patch()
        n_layers = 3
        all_nodes = _stack_layers(base_nodes, n_layers)
        layer_conn = [base_faces.copy() for _ in range(n_layers - 1)]
        tets, _face_of_tet = convert_layers_to_tetrahedra(all_nodes, layer_conn, base_faces)

        p0, p1, p2, p3 = (all_nodes[tets[:, i]] for i in range(4))
        signed = np.einsum('ij,ij->i', p1 - p0,
                           np.cross(p2 - p0, p3 - p0)) / 6.0
        assert np.all(signed > 0), (
            f"{int(np.count_nonzero(signed <= 0))} tets have non-positive "
            f"signed volume"
        )

    def test_volume_sums_to_slab_volume(self):
        """网格总体积必须等于挤出平板的解析体积。"""
        base_nodes, base_faces = _flat_patch(nx=3, ny=3, lx=1.0, ly=1.0)
        n_layers, dz = 4, 0.1
        all_nodes = _stack_layers(base_nodes, n_layers, dz)
        layer_conn = [base_faces.copy() for _ in range(n_layers - 1)]
        tets, _face_of_tet = convert_layers_to_tetrahedra(all_nodes, layer_conn, base_faces)

        p0, p1, p2, p3 = (all_nodes[tets[:, i]] for i in range(4))
        vol = np.abs(np.einsum('ij,ij->i', p1 - p0,
                               np.cross(p2 - p0, p3 - p0))) / 6.0
        expected = 1.0 * 1.0 * dz * (n_layers - 1)
        assert vol.sum() == pytest.approx(expected, rel=1e-12)


class TestOrientTetrahedra:
    def test_flips_inverted_cell(self):
        nodes = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        # 故意颠倒的顺序（有符号体积为负）。
        tets = np.array([[0, 1, 3, 2]], dtype=np.int64)
        p0, p1, p2, p3 = (nodes[tets[:, i]] for i in range(4))
        before = np.einsum('ij,ij->i', p1 - p0, np.cross(p2 - p0, p3 - p0))
        assert before[0] < 0

        fixed = orient_tetrahedra(nodes, tets.copy())
        q0, q1, q2, q3 = (nodes[fixed[:, i]] for i in range(4))
        after = np.einsum('ij,ij->i', q1 - q0, np.cross(q2 - q0, q3 - q0))
        assert after[0] > 0
        assert set(fixed[0]) == {0, 1, 2, 3}, "vertex set must be preserved"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
