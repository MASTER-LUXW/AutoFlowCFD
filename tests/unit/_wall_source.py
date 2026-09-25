# -*- coding: utf-8 -*-
"""测试共用：合成网格的壁面距离来源。

生产路径的来源由 CLI 从体网格的 WALL 边界面构造
（`core/utils/wall_distance_source.py::WallDistanceSource.from_volume_data`）。
合成测试网格没有带 bc_types 的 BoundaryMap，这里取网格最低 z 平面上的解点当
"壁面"——确定、真实的几何来源，与生产路径走同一个 `query`。2026-09-25 以前
这类测试靠"没有壁面就退回单元特征长度"的兜底跑通，那条兜底已删除。
"""

import numpy as np

from autoflowcfd.core.utils.wall_distance_source import WallDistanceSource


def synthetic_wall_source(mesh) -> WallDistanceSource:
    """取网格最低 z 平面上的解点作为"壁面"点集（`HighOrderMesh` 公开的几何只有
    解点坐标）。"""
    pts = np.asarray(mesh.sps_coords, dtype=np.float64).reshape(-1, 3)
    z0 = pts[:, 2].min()
    on_floor = pts[:, 2] <= z0 + 1e-9 * max(1.0, float(np.ptp(pts[:, 2])))
    return WallDistanceSource.kdtree(pts[on_floor])
