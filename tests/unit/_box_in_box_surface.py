"""合成面网格：远场长方体（FARFIELD 组 "tunnel"）里放一个立方体物面（WALL 组 "body"），供体网格生成管线的
端到端单元测试使用（单元数几千、几秒内生成）。两个盒子的三角形都朝流体域外（物面朝外、远场朝外）。
"""

import numpy as np

from autoflowcfd.grid.schema import BoundaryMap, CellArray, GridData, GridMetadata, NodeArray


def _box_surface(lo, hi, n):
    """轴对齐长方体表面：每面 n x n 个四边形、每个拆成 2 个三角形，法向朝外。返回 (nodes, faces)。"""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    nodes, faces, index = [], [], {}

    def node(p):
        key = tuple(np.round(p, 12))
        if key not in index:
            index[key] = len(nodes)
            nodes.append(p)
        return index[key]

    t = np.linspace(0.0, 1.0, n + 1)
    for axis in range(3):
        u, v = [a for a in range(3) if a != axis]
        for side in (0, 1):
            for i in range(n):
                for j in range(n):
                    quad = []
                    for a, b in ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)):
                        p = np.empty(3)
                        p[axis] = (lo, hi)[side][axis]
                        p[u] = lo[u] + t[a] * (hi[u] - lo[u])
                        p[v] = lo[v] + t[b] * (hi[v] - lo[v])
                        quad.append(node(p))
                    tris = [[quad[0], quad[1], quad[2]], [quad[0], quad[2], quad[3]]]
                    faces += tris
    nodes, faces = np.array(nodes), np.array(faces)
    # 统一朝外：按三角形质心相对盒子中心判断
    center = 0.5 * (lo + hi)
    tri = nodes[faces]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    inward = np.einsum("ij,ij->i", normal, tri.mean(axis=1) - center) < 0.0
    faces[inward] = faces[inward][:, [0, 2, 1]]
    return nodes, faces


def box_in_box_surface(body_n: int = 3, tunnel_n: int = 2) -> GridData:
    """物面：边长 1 m 的立方体（中心在原点）；远场：[-3, 5] x [-2, 2] x [-2, 2] m。"""
    body_nodes, body_faces = _box_surface([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5], body_n)
    tun_nodes, tun_faces = _box_surface([-3.0, -2.0, -2.0], [5.0, 2.0, 2.0], tunnel_n)
    nodes = np.vstack([body_nodes, tun_nodes])
    faces = np.vstack([body_faces, tun_faces + len(body_nodes)]).astype(np.int32)
    n_body = len(body_faces)
    boundaries = BoundaryMap(
        groups={"body": np.arange(n_body, dtype=np.int32),
                "tunnel": np.arange(n_body, len(faces), dtype=np.int32)},
        bc_types={"body": "WALL", "tunnel": "FARFIELD"})
    bbox = (float(nodes[:, 0].min()), float(nodes[:, 0].max()), float(nodes[:, 1].min()),
            float(nodes[:, 1].max()), float(nodes[:, 2].min()), float(nodes[:, 2].max()))
    return GridData(
        nodes=NodeArray(x=nodes[:, 0].copy(), y=nodes[:, 1].copy(), z=nodes[:, 2].copy()),
        cells=CellArray(connectivity=faces, cell_type=np.zeros(len(faces), dtype=np.int32)),
        boundaries=boundaries,
        metadata=GridMetadata(node_count=len(nodes), cell_count=len(faces), boundary_groups=["body", "tunnel"],
                              file_format="synthetic", bounding_box=bbox),
    )
