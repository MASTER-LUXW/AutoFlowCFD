"""AutoFlowCFD V2.0 - 壁面距离的唯一来源（全部后端、全部阶数共用）。

## 精确的点到壁面距离（2026-10-04）

此前的来源是"到最近**壁面节点**的距离"（KD-Tree）或节点级 Eikonal 值取最近网格节点。
高阶解点不在网格节点上：贴壁解点到最近壁面节点的距离约为 `sqrt(d^2 + s^2)`（`s` 是解点
在壁面切向到最近顶点的偏移，与壁面网格尺度同量级），在壁面加密网格上可以比真实壁距
大几个数量级。plate_demo 实测：P2 最贴壁 5% 的解点被高估中位数 4.1 倍、最多 76 倍（P1
的解点恰好与节点几何相近，约 1.0 倍）。SST 的壁面 omega 值 `60 nu / (beta1 d1^2)` 与 SA 的
耗散项 `cw1 fw (nu_tilde / d)^2` 都按平方反比放大这一误差。

现在按壁面三角形求**精确**最近距离（`aabb_tree.py`）。Eikonal 来源同时删除：到壁面的欧氏
最近距离就是流体域内到壁面的测地距离——最近壁面点的连线若穿过另一处壁面，穿过点会更近，
矛盾——所以离散 Eikonal（节点图上的最短路径松弛）只是同一个量的一个更慢、更不准的近似。

## 壁面上的点

直边映射下，落在壁面面片上的解点（原生四面体的解点含顶点、棱点与面点）与壁面三角形
只差舍入误差。`query` 把距离不超过浮点重合判据（`WALL_COINCIDENCE_ULPS` 个 ulp，相对
点坐标量级与最近三角形尺度）的点精确置为 0：**`d == 0` 当且仅当该点位于壁面上**——
SA-neg 据此在这些点上施加强 Dirichlet 条件（`turbulence/sa` 模块文档）。

## 后端与阶数

`query(points)` 给出任意一组点到壁面的距离。单机、分布式各 rank（查自己 compact 空间的
解点坐标）、GPU、各次换阶都调用同一个 `query`，差别只在"查哪些点"。来源只由 CLI 从体网格
构造一次（`from_volume_data`），找不到壁面时报错，不退化成任何估计值。
"""

import numpy as np

from .aabb_tree import build_aabb_tree, nearest_triangle_distance
from .surface import wall_triangles

#: 判定"点位于壁面上"的浮点重合判据：距离不超过这么多个 ulp（相对 `|p|_inf` 与最近三角形
#: 最长边之和）。直边映射算出的解点坐标与壁面三角形之间只差几个 ulp 的舍入。
WALL_COINCIDENCE_ULPS = 1024


class WallDistanceSource:
    """与阶数、分区、后端都无关的壁面距离来源；`query(points)` 给出精确最近距离。"""

    __slots__ = ("_tri", "_tree", "_edge", "n_wall_faces")

    def __init__(self, triangles: np.ndarray):
        tri = np.ascontiguousarray(np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3))
        if tri.shape[0] == 0:
            raise ValueError("壁面三角形集合为空，无法构造壁面距离来源")
        self._tri = tri
        self._tree = build_aabb_tree(tri)
        edges = np.stack([tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 1], tri[:, 0] - tri[:, 2]], axis=1)
        self._edge = np.linalg.norm(edges, axis=2).max(axis=1)
        self.n_wall_faces = int(tri.shape[0])

    def __getstate__(self):
        # 分布式把来源发给各 rank：只传三角形，树在接收端重建（numba 结果是普通数组，
        # 但重建比传输更便宜且不依赖发送端的编译产物）
        return {"triangles": self._tri}

    def __setstate__(self, state):
        WallDistanceSource.__init__(self, state["triangles"])

    @classmethod
    def from_volume_data(cls, volume_data) -> "WallDistanceSource":
        """CLI 的唯一构造入口：WALL 组的边界面 -> 来源。

        Raises:
            ValueError: 网格里没有任何 WALL 边界面（湍流模型需要壁距时不能退化成估计值继续求解）。
        """
        tri = wall_triangles(volume_data)
        if tri.shape[0] == 0:
            raise ValueError(
                "网格里没有任何 WALL 类型边界面（boundaries.bc_types 中无 WALL 项，或 WALL "
                "组没有边界面）——湍流模型的屏蔽函数、omega 壁面值、壁面模型都依赖壁距，"
                "不能退化为估计值继续求解。请检查体网格的边界分组。")
        return cls(tri)

    def query(self, points: np.ndarray) -> np.ndarray:
        """`points (..., 3)` -> 距离 `(...)`；位于壁面上的点精确为 0（模块文档"壁面上的点"）。"""
        pts = np.asarray(points, dtype=np.float64)
        flat = np.ascontiguousarray(pts.reshape(-1, 3))
        if flat.shape[0] == 0:
            return np.zeros(pts.shape[:-1])
        d, nearest = nearest_triangle_distance(flat, self._tri, *self._tree)
        scale = np.abs(flat).max(axis=1) + self._edge[nearest]
        d = np.where(d <= WALL_COINCIDENCE_ULPS * np.finfo(np.float64).eps * scale, 0.0, d)
        return d.reshape(pts.shape[:-1])
