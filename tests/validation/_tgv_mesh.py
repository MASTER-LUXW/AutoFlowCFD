"""构造三方向（x/y/z）周期的四面体立方体网格，供 Taylor-Green 涡
（TGV）衰减基准使用。

TGV 标准算例要求三方向周期的立方体域，这与等熵涡（`_isentropic_vortex.py`）
只需单方向周期不同——棱柱只能沿挤出轴给出单一记录、两端天然全等的
封盖面（`_periodic_mesh.py` 采用的方案），无法同时在 3 个方向都提供
这种"封盖"关系；若改用棱柱挤出 + 侧面周期，两端的四边形侧面会被
`FaceExtractor` 按它自己的（基于全局节点号的）对角线规则各自独立拆分成
2 个三角形子面，两端的拆分方向不保证互为平移镜像，配对会因为三角化
不全等而失败（本项目周期边界条件开发过程中在单方向棱柱网格上真实
复现过这个失败模式）。

四面体没有这个问题：所有面天生就是三角形，不存在"哪条对角线"的选择
歧义；只要用一个**与 (i,j,k) 无关的固定局部拆分模板**（每个六面体
都用同一套局部角点 v0..v7 -> 6 个子四面体的映射，不依赖全局最小-最大
节点编号），六面体网格边界上任意一个方向的两侧邊界面三角化天然互为
平移镜像，可以同时在 x/y/z 三个方向配对——直接复用
`_channel_mesh.py::build_channel_mesh` 已经验证过的固定局部拆分模板。

代价（详见项目记忆 tet_collapsed_coord_anisotropy）：四面体坍缩坐标
P2 方案的参考轴 (a,b,c) 权重天然不对称，约 1/3 的单元若局部梯度方向
压在单一参考轴上，残差会被放大 6-7 个数量级——这是 TGV 这个三维旋转
流场、必须用四面体才能达成三方向周期的固有代价，与 Couette/等熵涡
特意选用棱柱规避这个问题不同。本模块的使用方需要自行监控残差分布、
不能假设四面体网格上的残差和棱柱网格一样干净（见 test_tgv.py 里对此
的显式检查）。
"""
from types import SimpleNamespace

import numpy as np

from autoflowcfd.grid.high_order.high_order_mesh import HighOrderMesh
from autoflowcfd.grid.schema.grid_boundaries import BoundaryMap

from ._channel_mesh import _MockCells, _MockNodes


def build_triply_periodic_tet_mesh(order, n, L):
    """n x n x n 六面体（各 6 拆分为四面体）立方体网格，边长 L，
    x/y/z 三个方向都标记为一对周期面。
    """
    n1 = n + 1

    def gid(i, j, k):
        return i + n1 * j + n1 * n1 * k

    ii, jj, kk = np.meshgrid(np.arange(n1), np.arange(n1), np.arange(n1), indexing="ij")
    node_id = (ii + n1 * jj + n1 * n1 * kk).ravel()
    order_idx = np.argsort(node_id)
    xs = (ii.ravel() / n * L)[order_idx]
    ys = (jj.ravel() / n * L)[order_idx]
    zs = (kk.ravel() / n * L)[order_idx]
    nodes = np.column_stack([xs, ys, zs])

    tets = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                v0 = gid(i, j, k)
                v1 = gid(i + 1, j, k)
                v2 = gid(i + 1, j + 1, k)
                v3 = gid(i, j + 1, k)
                v4 = gid(i, j, k + 1)
                v5 = gid(i + 1, j, k + 1)
                v6 = gid(i + 1, j + 1, k + 1)
                v7 = gid(i, j + 1, k + 1)
                tets.extend([
                    [v0, v1, v2, v6],
                    [v0, v2, v3, v6],
                    [v0, v3, v7, v6],
                    [v0, v7, v4, v6],
                    [v0, v4, v5, v6],
                    [v0, v5, v1, v6],
                ])

    tet_conn = np.array(tets, dtype=np.int32)
    n_tets = len(tet_conn)

    tol = 1e-9 * max(L, 1.0)

    def face_on_plane(coord_vals, target):
        return (np.abs(coord_vals - target) < tol).sum(axis=1) >= 3

    tet_coords = nodes[tet_conn]
    groups = {
        "x_min": np.flatnonzero(face_on_plane(tet_coords[:, :, 0], 0.0)),
        "x_max": np.flatnonzero(face_on_plane(tet_coords[:, :, 0], L)),
        "y_min": np.flatnonzero(face_on_plane(tet_coords[:, :, 1], 0.0)),
        "y_max": np.flatnonzero(face_on_plane(tet_coords[:, :, 1], L)),
        "z_min": np.flatnonzero(face_on_plane(tet_coords[:, :, 2], 0.0)),
        "z_max": np.flatnonzero(face_on_plane(tet_coords[:, :, 2], L)),
    }
    bc_types = {name: "PERIODIC" for name in groups}
    boundary_map = BoundaryMap(
        groups={k: v.astype(np.int32) for k, v in groups.items()},
        bc_types=bc_types,
        parameters={
            "x_min": {"paired_with": "x_max", "translation": [L, 0.0, 0.0]},
            "y_min": {"paired_with": "y_max", "translation": [0.0, L, 0.0]},
            "z_min": {"paired_with": "z_max", "translation": [0.0, 0.0, L]},
        },
    )

    empty_prisms = np.zeros((0, 6), dtype=np.int32)
    mock_volume = SimpleNamespace(
        cell_count=n_tets,
        nodes=_MockNodes(nodes),
        cells=_MockCells(tet_conn),
        prism_cells=_MockCells(empty_prisms),
        boundaries=boundary_map,
    )

    mesh = HighOrderMesh(order=order)
    mesh.load_from_volume_mesh(mock_volume, build_faces=True)
    return mesh
