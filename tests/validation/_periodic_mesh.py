"""构造 x 方向周期的小型棱柱通道网格，供周期边界条件自一致性验证使用。

棱柱沿 x 方向挤出（(y,z) 平面三角化，x_min/x_max 因此是棱柱的封盖三角形
面——每个面只有一条记录，且两端三角化天然全等）。这个朝向是刻意选择的：
若改用沿 y/z 挤出、x_min/x_max 变成四边形侧面（会被 FaceExtractor 恒定
按对角线方向拆成 2 个三角形子记录），两端网格线的固定对角线拆分方向在
物理上不再镜像对称，`pair_periodic_boundary_faces` 会因为两侧三角化不
全等而找不到一一对应的配对点，属于网格本身的拓扑缺陷（而不是配对算法
的 bug）——沿周期方向挤出、让周期面本身是单一记录的封盖面，是规避这个
问题最直接的网格设计选择。
"""
from types import SimpleNamespace

import numpy as np

from autoflowcfd.grid.high_order_mesh import HighOrderMesh
from autoflowcfd.grid.schema.grid_boundaries import BoundaryMap

from ._channel_mesh import _MockCells, _MockNodes, build_face_exact_ghost_provider


def build_periodic_channel_mesh_x(order, nx, ny, nz, Lx, H, Lz, side_bc_type="SYMMETRY"):
    """棱柱沿 x 方向挤出，x_min/x_max 标记为一对周期面（translation=[Lx,0,0]），
    y/z 方向的 4 个侧面（wall_bottom/wall_top/z_min/z_max）留作普通边界面，
    由调用方通过 `build_face_exact_ghost_provider` 按面自身物理坐标分类
    （绕开 BoundaryMap.groups 按 owner 单元分组在角点单元上的固有歧义，
    见 _channel_mesh.py 模块文档——这个歧义与本文件要验证的周期配对是
    两个独立问题，本网格的角点单元同时贴着 x_min/x_max（周期面）和
    wall_bottom/z_min 等侧面，必须用 face-exact 分类才能避免侧面被误标）。

    Args:
        side_bc_type: 4 个侧面在 `BoundaryMap.bc_types` 里标注的类型
            （仅影响元数据/日志，实际生效的 BC 公式由调用方通过
            face-exact ghost provider 显式指定，与这里的标注无关）。
    """
    nx1, ny1, nz1 = nx + 1, ny + 1, nz + 1

    def gid(i, j, k):
        return i + nx1 * j + nx1 * ny1 * k

    ii, jj, kk = np.meshgrid(np.arange(nx1), np.arange(ny1), np.arange(nz1), indexing="ij")
    node_id = (ii + nx1 * jj + nx1 * ny1 * kk).ravel()
    order_idx = np.argsort(node_id)
    xs = (ii.ravel() / nx * Lx)[order_idx]
    ys = (jj.ravel() / ny * H)[order_idx]
    zs = (kk.ravel() / nz * Lz)[order_idx]
    nodes = np.column_stack([xs, ys, zs])

    prisms = []
    for j in range(ny):
        for k in range(nz):
            tri_a = [(j, k), (j + 1, k), (j + 1, k + 1)]
            tri_b = [(j, k), (j + 1, k + 1), (j, k + 1)]
            for tri in (tri_a, tri_b):
                for i in range(nx):
                    v = [gid(i, tj, tk) for tj, tk in tri]
                    w = [gid(i + 1, tj, tk) for tj, tk in tri]
                    prisms.append(v + w)

    prism_conn = np.array(prisms, dtype=np.int32)
    n_prisms = len(prism_conn)

    tol = 1e-9 * max(Lx, H, Lz, 1.0)

    def face_on_plane(coord_vals, target):
        return (np.abs(coord_vals - target) < tol).sum(axis=1) >= 3

    prism_coords = nodes[prism_conn]
    groups = {
        "wall_bottom": np.flatnonzero(face_on_plane(prism_coords[:, :, 1], 0.0)),
        "wall_top": np.flatnonzero(face_on_plane(prism_coords[:, :, 1], H)),
        "z_min": np.flatnonzero(face_on_plane(prism_coords[:, :, 2], 0.0)),
        "z_max": np.flatnonzero(face_on_plane(prism_coords[:, :, 2], Lz)),
        "x_min": np.flatnonzero(face_on_plane(prism_coords[:, :, 0], 0.0)),
        "x_max": np.flatnonzero(face_on_plane(prism_coords[:, :, 0], Lx)),
    }
    bc_types = {
        "wall_bottom": side_bc_type, "wall_top": side_bc_type,
        "z_min": side_bc_type, "z_max": side_bc_type,
        "x_min": "PERIODIC", "x_max": "PERIODIC",
    }
    boundary_map = BoundaryMap(
        groups={k: v.astype(np.int32) for k, v in groups.items()},
        bc_types=bc_types,
        parameters={"x_min": {"paired_with": "x_max", "translation": [Lx, 0.0, 0.0]}},
    )

    empty_tets = np.zeros((0, 4), dtype=np.int32)
    mock_volume = SimpleNamespace(
        cell_count=n_prisms,
        nodes=_MockNodes(nodes),
        cells=_MockCells(empty_tets),
        prism_cells=_MockCells(prism_conn),
        boundaries=boundary_map,
    )

    mesh = HighOrderMesh(order=order)
    mesh.load_from_volume_mesh(mock_volume, build_faces=True)
    return mesh


def build_periodic_symmetry_ghost_provider(mesh, Lx, H, Lz):
    """wall_bottom/wall_top/z_min/z_max 四个非周期侧面统一按 SYMMETRY 处理，
    按面自身物理坐标分类（见 build_periodic_channel_mesh_x 文档）。
    x_min/x_max 配对后已经是内部面，不会出现在这里的边界面集合里。
    """
    bc_by_plane = {
        "wall_bottom": {"type": "SYMMETRY"}, "wall_top": {"type": "SYMMETRY"},
        "z_min": {"type": "SYMMETRY"}, "z_max": {"type": "SYMMETRY"},
    }
    return build_face_exact_ghost_provider(mesh, Lx, H, Lz, bc_by_plane)
