"""构造结构化平面通道网格（Couette/Poiseuille 定量精度基准共用），以及
按面物理位置精确分类边界条件的 ghost state provider。

网格：结构化六面体网格，每个六面体沿主对角线（局部角点 (i,j,k)->(i+1,j+1,k+1)，
在结构化编号下恰好是该六面体全局节点编号最小->最大的角点）拆成 6 个四面体
——这个拆分规则保证相邻六面体共享面上用的对角线只取决于该四边形面自身 4 个
角点的全局编号（min->max），与 fr/collapsed_basis.py 等模块已经验证过的
"任意四边形面按全局编号 min->max 取对角线"规则完全一致，因此是全局自洽、
无缝拼接的四面体网格，不需要单独的六面体几何映射支持。
"""
from types import SimpleNamespace

import numpy as np

from autoflowcfd.grid.high_order.high_order_mesh import HighOrderMesh
from autoflowcfd.boundary.fr_ghost_state import BoundaryGhostStateProvider


def build_face_exact_ghost_provider(mesh, Lx, H, Lz, bc_by_plane, tol_scale=1e-6):
    """绕开 mesh.boundaries.groups（按 owner 单元索引分组）在角/棱单元上的
    固有歧义——真实发现：owner-cell 分组机制下，同一个单元只要同时拥有
    属于 2 个不同命名边界组的边界面（角落/棱上的单元，任何有限盒子网格
    都无法避免），tag_boundary_groups 会把该单元的**全部**边界面都打上
    最后一次匹配到的组的标签，与这些面各自真实所在的物理边界面无关。
    对本算例（nz=1 时每个单元同时贴着 z_min 和 z_max）这个歧义命中了
    108 个边界面里的 20 个（约18.5%），把部分 y=0/y=H 壁面误标成
    x 端的 OUTLET，导致求解在第 0 步就以 O(1e9) 量级发散——不是
    求解器本身的 bug，是这个手工测试网格绕过 BoundaryGhostStateProvider
    时必须自己保证的前提条件。这里直接按每个边界面自身的物理位置
    （face center 坐标）分类，是与 owner-cell 分组完全等价、但没有
    歧义的实现，因为 BoundaryGhostStateProvider 本身只需要一个
    per-face 的 group_code 数组，不要求这个数组必须来自
    tag_boundary_groups。
    """
    fc = mesh.face_connectivity
    tol = tol_scale * max(Lx, H, Lz, 1.0)
    group_code = np.full(fc.n_faces, -1, dtype=np.int32)
    code_to_config = {}
    bidx = fc.get_boundary_face_indices()
    centers = fc.center[bidx]

    def classify(c):
        if abs(c[1] - 0.0) < tol:
            return "wall_bottom"
        if abs(c[1] - H) < tol:
            return "wall_top"
        if abs(c[2] - 0.0) < tol:
            return "z_min"
        if abs(c[2] - Lz) < tol:
            return "z_max"
        if abs(c[0] - 0.0) < tol:
            return "x_min"
        if abs(c[0] - Lx) < tol:
            return "x_max"
        raise ValueError(f"boundary face center {c} not on any known domain plane")

    name_to_code = {}
    for f, c in zip(bidx, centers):
        name = classify(c)
        if name not in name_to_code:
            name_to_code[name] = len(name_to_code)
            code_to_config[name_to_code[name]] = bc_by_plane[name]
        group_code[f] = name_to_code[name]

    default_config = {"type": "FARFIELD", "Q_free": [1.225, 0.0, 0.0, 0.0, 101325.0]}
    return BoundaryGhostStateProvider(group_code, code_to_config, default_config)


class _MockNodes:
    def __init__(self, coords):
        self._coords = coords

    def get_coordinates(self):
        return self._coords


class _MockCells:
    def __init__(self, connectivity):
        self.connectivity = connectivity


class _MockBoundaries:
    def __init__(self, groups, bc_types):
        self.groups = groups
        self.bc_types = bc_types


def build_channel_mesh(order, nx, ny, nz, Lx, H, Lz):
    nx1, ny1, nz1 = nx + 1, ny + 1, nz + 1
    ii, jj, kk = np.meshgrid(np.arange(nx1), np.arange(ny1), np.arange(nz1), indexing="ij")
    node_id = (ii + nx1 * jj + nx1 * ny1 * kk).ravel()
    order_idx = np.argsort(node_id)  # 保证 node 数组下标本身就是全局编号
    xs = (ii.ravel() / nx * Lx)[order_idx]
    ys = (jj.ravel() / ny * H)[order_idx]
    zs = (kk.ravel() / nz * Lz)[order_idx]
    nodes = np.column_stack([xs, ys, zs])

    def gid(i, j, k):
        return i + nx1 * j + nx1 * ny1 * k

    tets = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                v0 = gid(i, j, k)
                v1 = gid(i + 1, j, k)
                v2 = gid(i + 1, j + 1, k)
                v3 = gid(i, j + 1, k)
                v4 = gid(i, j, k + 1)
                v5 = gid(i + 1, j, k + 1)
                v6 = gid(i + 1, j + 1, k + 1)
                v7 = gid(i, j + 1, k + 1)
                sub_tets = [
                    [v0, v1, v2, v6],
                    [v0, v2, v3, v6],
                    [v0, v3, v7, v6],
                    [v0, v7, v4, v6],
                    [v0, v4, v5, v6],
                    [v0, v5, v1, v6],
                ]
                tets.extend(sub_tets)

    tet_conn = np.array(tets, dtype=np.int32)
    n_tets = len(tet_conn)

    # 边界分组：按每个 tet 的 4 个节点是否有 >=3 个落在目标边界平面上
    # （即该 tet 恰有一个面完全位于边界平面），纯几何判据，不依赖对
    # 6-tet 分解组合规律的手工推导。
    tol = 1e-9 * max(Lx, H, Lz, 1.0)

    def face_on_plane(coord_vals, target):
        on_plane = np.abs(coord_vals - target) < tol
        return on_plane.sum(axis=1) >= 3

    tet_coords = nodes[tet_conn]  # (n_tets,4,3)
    groups = {
        "wall_bottom": np.flatnonzero(face_on_plane(tet_coords[:, :, 1], 0.0)),
        "wall_top": np.flatnonzero(face_on_plane(tet_coords[:, :, 1], H)),
        "z_min": np.flatnonzero(face_on_plane(tet_coords[:, :, 2], 0.0)),
        "z_max": np.flatnonzero(face_on_plane(tet_coords[:, :, 2], Lz)),
        "x_min": np.flatnonzero(face_on_plane(tet_coords[:, :, 0], 0.0)),
        "x_max": np.flatnonzero(face_on_plane(tet_coords[:, :, 0], Lx)),
    }
    bc_types = {name: "WALL" for name in groups}  # 占位，真正类型由 bc_overrides 决定

    mock_volume = SimpleNamespace(
        cell_count=n_tets,
        nodes=_MockNodes(nodes),
        cells=_MockCells(tet_conn),
        prism_cells=None,
        boundaries=_MockBoundaries(groups, bc_types),
    )

    mesh = HighOrderMesh(order=order)
    mesh.load_from_volume_mesh(mock_volume, build_faces=True)
    return mesh


def build_channel_mesh_prism(order, nx, ny, nz, Lx, H, Lz):
    """棱柱通道网格：(x,z) 平面 nx*nz 个矩形各拆 2 个三角形，沿 y 方向
    （壁面法向）整层挤出 ny 层，共 2*nx*nz*ny 个棱柱。

    背景（见项目记忆 tet_collapsed_coord_anisotropy）：四面体坍缩坐标
    P2 方案的参考轴 (a,b,c) 权重天然不对称，约 1/3 的四面体单元若主
    梯度方向压在单一参考轴上，残差会被放大 6-7 个数量级，与网格质量/
    尺度无关——这正是 AutoFlowCFD 网格架构本身要求近壁/高剪切区用棱柱
    （不用四面体）的原因：棱柱挤出方向 (c 轴) 用完全无权重的普通
    Legendre 基，且对直壁挤出物理 y 是 c 的精确线性函数，Couette/
    Poiseuille 这类沿壁面法向变化的解析解复合后是 c 的精确多项式，
    插值截断误差为零。验证近壁剪切物理必须用棱柱网格，不能用纯四面体。
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
    for i in range(nx):
        for k in range(nz):
            # (x,z) 平面矩形 (i,k)-(i+1,k)-(i+1,k+1)-(i,k+1) 拆成两个三角形，
            # 所有矩形用同一条对角线方向拆分，保证整张网格是流形三角剖分。
            tri_a = [(i, k), (i + 1, k), (i + 1, k + 1)]
            tri_b = [(i, k), (i + 1, k + 1), (i, k + 1)]
            for tri in (tri_a, tri_b):
                for j in range(ny):
                    v = [gid(ti, j, tk) for ti, tk in tri]
                    w = [gid(ti, j + 1, tk) for ti, tk in tri]
                    prisms.append(v + w)

    prism_conn = np.array(prisms, dtype=np.int32)
    n_prisms = len(prism_conn)

    tol = 1e-9 * max(Lx, H, Lz, 1.0)

    def face_on_plane(coord_vals, target):
        on_plane = np.abs(coord_vals - target) < tol
        return on_plane.sum(axis=1) >= 3

    prism_coords = nodes[prism_conn]  # (n_prisms,6,3)
    groups = {
        "wall_bottom": np.flatnonzero(face_on_plane(prism_coords[:, :, 1], 0.0)),
        "wall_top": np.flatnonzero(face_on_plane(prism_coords[:, :, 1], H)),
        "z_min": np.flatnonzero(face_on_plane(prism_coords[:, :, 2], 0.0)),
        "z_max": np.flatnonzero(face_on_plane(prism_coords[:, :, 2], Lz)),
        "x_min": np.flatnonzero(face_on_plane(prism_coords[:, :, 0], 0.0)),
        "x_max": np.flatnonzero(face_on_plane(prism_coords[:, :, 0], Lx)),
    }
    bc_types = {name: "WALL" for name in groups}

    empty_tets = np.zeros((0, 4), dtype=np.int32)
    mock_volume = SimpleNamespace(
        cell_count=n_prisms,
        nodes=_MockNodes(nodes),
        cells=_MockCells(empty_tets),
        prism_cells=_MockCells(prism_conn),
        boundaries=_MockBoundaries(groups, bc_types),
    )

    mesh = HighOrderMesh(order=order)
    mesh.load_from_volume_mesh(mock_volume, build_faces=True)
    return mesh
