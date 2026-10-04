"""AutoFlowCFD V2.0 - 壁面距离：壁面三角形（`surface.py`）、AABB 树精确最近距离
（`aabb_tree.py`）、全部后端共用的来源对象（`source.py`）。"""

from .source import WALL_COINCIDENCE_ULPS, WallDistanceSource
from .surface import triangles_from_faces, wall_face_nodes, wall_triangles

__all__ = ["WALL_COINCIDENCE_ULPS", "WallDistanceSource", "triangles_from_faces", "wall_face_nodes",
           "wall_triangles"]
