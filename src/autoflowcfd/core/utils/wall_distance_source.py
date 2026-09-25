"""AutoFlowCFD V2.0 - 壁面距离的唯一来源（全部后端、全部阶数共用）。

## 为什么需要（2026-09-25 查出的真实缺陷）

壁面距离此前有**两套**构造：

* 单机 CPU：CLI 从体网格的 WALL 组**边界面**取壁面节点
  （`wall_nodes_from_boundary_faces`，2026-09-15 修复的一阶物理错误），
  KD-Tree 或 Eikonal 求距离、映射到 SP，换阶时用缓存的壁面几何重新查询；
* CPU 分布式（传统 / 完全分布式加载 / 换阶重建）、单 GPU、多 GPU：各自一份
  拷贝，遍历 `mesh.boundary_groups` 时把每个值当字典读
  `bg.get('type')` / `bg.get('node_indices')`。

而 `BoundaryMap.groups` 的契约是 `Dict[str, np.ndarray]`（单元索引数组）。
于是凡是组名里不含 "WALL" 的真实网格（plate_demo 的 body/inlet/outlet/
tunnel、cube_demo 同样），这些后端**在构造湍流求解器时直接 AttributeError**；
组名碰巧含 "WALL" 时又把单元索引当节点索引（正是单机 09-15 修掉的那个
错误）。另外单 GPU 与 CPU 分布式在找不到壁面时会静默退回"用单元特征
长度当壁距"，而单机在同一情形下明确报错。

## 做法

壁面距离只由一个与阶数、分区、后端都无关的几何对象决定：

    WallDistanceSource.kdtree(wall_coords)          壁面节点物理坐标
    WallDistanceSource.eikonal(mesh_nodes, dist)     节点级 Eikonal 距离

`query(points)` 给出任意一组点到壁面的距离。单机、分布式各 rank（查自己
compact 空间的 SP 坐标）、GPU、各次换阶都调用同一个 `query`，差别只在
"查哪些点"。来源只由 CLI 从体网格构造一次（`from_volume_data`），找不到
壁面时报错，不退化成任何估计值。
"""

from typing import Tuple

import numpy as np


def wall_nodes_from_boundary_faces(volume_data, bm) -> Tuple[np.ndarray, int]:
    """从 WALL 组的**边界面**取出真正位于壁面上的节点索引。

    **这是 2026-09-15 发现的一处一阶物理错误的修复。** 原实现用
    `max(indices) >= n_nodes` 去猜 `BoundaryMap` 存的是单元还是节点索引；
    它按契约**恒定是单元索引**，那个判据恰好只在 WALL 组上猜错（壁面的边界
    单元是边界层棱柱，索引落在 `[0, n_prism)`，两张真实网格都满足
    `n_prism < n_nodes`）：

        cube_demo : body  range=[0,136974]  n_nodes=187702  -> 猜错
        plate_demo: body  range=[0, 65235]  n_nodes= 88496  -> 猜错

    后果是壁距变成"到按编号散布在全域的任意节点的距离"（plate_demo max
    壁距 0.976 m，真实 4.359 m）。

    **为什么用边界面而不是"单元的全部节点"**：边界层棱柱有 3 个节点在壁面
    上、3 个在第一层之外，全部算作壁面节点会让近壁壁距虚胖一层。

    已知边际情形：`map_boundaries_by_geometry` 把逐面匹配结果折叠成
    `{owner_cell: group}`，同一单元若同时拥有属于不同组的边界面（只发生在
    组交界棱上），这里会把它的全部边界面都计入——可忽略的过包含。

    Returns:
        `(wall_node_indices, n_wall_faces)`。

    Raises:
        ValueError: BoundaryMap 与体网格不匹配，或面数据缺少节点连接。
    """
    n_cells = volume_data.cell_count
    wall_cells = set()
    for bc_name, bc_type in bm.bc_types.items():
        if bc_type != 'WALL' or not bm.has_boundary(bc_name):
            continue
        idx = np.asarray(bm.get_cell_indices(bc_name))
        if idx.size == 0:
            continue
        if int(idx.max()) >= n_cells:
            raise ValueError(
                f"边界组 '{bc_name}' 的单元索引最大值 {int(idx.max())} 超出体网格单元数 "
                f"{n_cells}——BoundaryMap 与体网格不匹配（壁面距离场会整体错位）。")
        wall_cells.update(int(c) for c in idx)
    if not wall_cells:
        return np.empty(0, dtype=np.int64), 0

    faces = volume_data.ensure_faces_exist()
    if faces.node_connectivity is None:
        raise ValueError(
            "体网格的面数据缺少 node_connectivity，无法从边界面取壁面节点——不能退回"
            "'把单元全部节点当壁面'（那会让近壁壁距虚胖一层）。")
    bidx = faces.get_boundary_face_indices()
    if len(bidx) == 0:
        return np.empty(0, dtype=np.int64), 0
    owner = faces.connectivity[bidx, 0]
    keep = np.fromiter((int(o) in wall_cells for o in owner), dtype=bool, count=len(owner))
    sel = bidx[keep]
    if len(sel) == 0:
        return np.empty(0, dtype=np.int64), 0
    nodes = faces.node_connectivity[sel].ravel()
    nodes = nodes[nodes >= 0]
    return np.unique(nodes).astype(np.int64), int(len(sel))


class WallDistanceSource:
    """与阶数、分区、后端都无关的壁面距离来源；`query(points)` 给出距离。"""

    __slots__ = ("kind", "_tree", "_mesh_node_tree", "_node_distances", "n_wall_nodes")

    def __init__(self, kind, *, tree=None, mesh_node_tree=None, node_distances=None, n_wall_nodes=0):
        self.kind = kind
        self._tree = tree
        self._mesh_node_tree = mesh_node_tree
        self._node_distances = node_distances
        self.n_wall_nodes = int(n_wall_nodes)

    @classmethod
    def kdtree(cls, wall_coords: np.ndarray) -> "WallDistanceSource":
        """到壁面节点的欧氏最近距离（默认）。"""
        from scipy.spatial import cKDTree

        wall_coords = np.asarray(wall_coords, dtype=np.float64).reshape(-1, 3)
        if wall_coords.shape[0] == 0:
            raise ValueError("壁面节点集合为空，无法构造壁面距离来源")
        return cls("kdtree", tree=cKDTree(wall_coords), n_wall_nodes=wall_coords.shape[0])

    @classmethod
    def eikonal(cls, mesh_nodes: np.ndarray, node_distances: np.ndarray,
                n_wall_nodes: int) -> "WallDistanceSource":
        """节点级 Eikonal（沿网格拓扑传播的）距离；查询点取最近网格节点的值
        ——不能对查询点重新做欧氏查询，那等于丢掉 Eikonal 的全部意义。"""
        from scipy.spatial import cKDTree

        return cls("eikonal", mesh_node_tree=cKDTree(np.asarray(mesh_nodes, dtype=np.float64)),
                   node_distances=np.asarray(node_distances, dtype=np.float64),
                   n_wall_nodes=n_wall_nodes)

    @classmethod
    def from_nodes(cls, mesh_nodes, wall_indices, *, connectivity=None, use_eikonal=False):
        """由网格节点坐标与壁面节点索引构造（KD-Tree 或 Eikonal）。"""
        mesh_nodes = np.asarray(mesh_nodes, dtype=np.float64)
        wall_indices = np.asarray(wall_indices, dtype=np.int64)
        if wall_indices.size == 0:
            raise ValueError("没有壁面节点，无法构造壁面距离来源")
        if not use_eikonal:
            return cls.kdtree(mesh_nodes[wall_indices])
        if connectivity is None:
            raise ValueError("Eikonal 壁面距离需要节点邻接表（build_node_adjacency）")
        from autoflowcfd.core.utils.wall_distance import compute_wall_distance

        dist = compute_wall_distance(mesh_nodes, wall_indices, connectivity=connectivity,
                                     use_eikonal=True)
        return cls.eikonal(mesh_nodes, dist, wall_indices.size)

    @classmethod
    def from_volume_data(cls, volume_data, *, use_eikonal: bool = False) -> "WallDistanceSource":
        """CLI 的唯一构造入口：WALL 组边界面上的节点 -> 来源。

        Raises:
            ValueError: 网格里没有任何 WALL 边界面（湍流模型需要壁距时不能
                退化成估计值继续求解）。
        """
        bm = volume_data.boundaries
        wall_nodes, n_faces = wall_nodes_from_boundary_faces(volume_data, bm)
        if wall_nodes.size == 0:
            raise ValueError(
                "网格里没有任何 WALL 类型边界面（boundaries.bc_types 中无 WALL 项，或 WALL "
                "组没有边界面）——湍流模型的屏蔽函数、omega 壁面值、壁面模型都依赖壁距，"
                "不能退化为估计值继续求解。请检查体网格的边界分组。")
        mesh_nodes = volume_data.nodes.get_coordinates()
        connectivity = None
        if use_eikonal:
            from autoflowcfd.grid.connectivity.node_connectivity import build_node_adjacency

            tet = volume_data.cells.connectivity if volume_data.cells else None
            prism = volume_data.prism_cells.connectivity if volume_data.prism_cells else None
            connectivity = build_node_adjacency(volume_data.node_count, tet_connectivity=tet,
                                                prism_connectivity=prism)
        return cls.from_nodes(mesh_nodes, wall_nodes, connectivity=connectivity,
                              use_eikonal=use_eikonal)

    def query(self, points: np.ndarray) -> np.ndarray:
        """`points (..., 3)` -> 距离 `(...)`。"""
        pts = np.asarray(points, dtype=np.float64)
        flat = pts.reshape(-1, 3)
        if self.kind == "kdtree":
            d, _ = self._tree.query(flat, k=1)
        else:
            _, nearest = self._mesh_node_tree.query(flat, k=1)
            d = self._node_distances[nearest]
        return np.asarray(d, dtype=np.float64).reshape(pts.shape[:-1])
