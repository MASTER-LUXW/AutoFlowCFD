# -*- coding: utf-8 -*-
"""测试共用：合成网格的壁面距离来源。

生产路径的来源由 CLI 从体网格的 WALL 边界面构造
（`core/utils/wall_distance::WallDistanceSource.from_volume_data`）。合成测试网格没有带
bc_types 的 BoundaryMap，这里把网格最低 z 平面当作"壁面"：用覆盖整个包围盒（四周各外扩
一倍尺度）的两个三角形表示该平面，与生产路径走同一个 `query`（精确点到面距离，平面上的
解点精确为 0）。
"""

import numpy as np

from autoflowcfd.core.utils.wall_distance import WallDistanceSource, triangles_from_faces


def synthetic_wall_source(mesh) -> WallDistanceSource:
    """网格最低 z 平面（`HighOrderMesh` 公开的几何只有解点坐标，取其包围盒）。"""
    pts = np.asarray(mesh.sps_coords, dtype=np.float64).reshape(-1, 3)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    pad = max(1.0, float((hi - lo).max()))
    x0, x1 = lo[0] - pad, hi[0] + pad
    y0, y1 = lo[1] - pad, hi[1] + pad
    z0 = lo[2]
    corners = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]])
    return WallDistanceSource(triangles_from_faces(corners, np.array([[0, 1, 2], [0, 2, 3]])))
