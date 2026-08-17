"""mesh_corner_split.py 用到的纯几何辅助函数 (从该文件拆分)。

从 mesh_corner_split.py 拆出来（该文件原有 446 行，超过 400 行硬性
拆分阈值）：这三个函数都是无副作用的纯几何/拓扑计算，不依赖
split_sharp_corners 内部任何状态，只被它调用，独立成文件是最干净的
拆分点。纯代码搬移，不改变任何行为。

注意：`_face_normals` 与 mesh_tetgen_seam.py 里的同名私有函数、
mesh_utils.py::compute_face_normals 是三份独立实现（拆分前就已如此，
不是本次拆分引入的重复），本次只搬移 mesh_corner_split.py 自己这一份，
不合并/不改动其它两处。
"""

import numpy as np


def _face_normals(nodes: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = nodes[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    cross_norm = np.linalg.norm(cross, axis=1)
    return cross / np.maximum(cross_norm, 1e-300)[:, np.newaxis]


def _implied_edge_radius(edge_length: float, dihedral_angle: float) -> float:
    """局部半径由一个三角化边隐含的曲率，将该边视为未知半径的圆弧的弦，
    圆弧扫过 `dihedral_angle`（标准弦到半径关系：弦 c，对角 theta，
    半径 r = c / (2 sin(theta/2))）。

    这个估计的要点：对于真正的尖锐 CAD 折痕（两个平面以固定的 G0 不连续
    角相交），该角是两个平面自身的属性——它不会随网格细化而缩小，所以
    这个公式隐含的半径与边长成正比缩小（r = c / const）。对于一个真实的
    物理半径 R 的曲面，只是欠三角化（曲线上只有几个面片），相同的公式
    恢复大约 R 本身，与边长无关，前提是边长相对于 R 较小（精细曲线上
    的大弦开始明显低估 R——这里不是问题，因为这个仅在已注册为局部“尖锐”
    的边上评估，即按构造就是小弦）。区分两者正是“隐含半径与探测边长
    本身可比（随其缩小到 0——真实折痕）还是远大于它（大致与分辨率无关——
    真实但粗糙的曲线）”——见 split_sharp_corners 自己的 min_feature_radius
    参数了解这个如何实际使用。
    """
    half_angle = dihedral_angle / 2.0
    sin_half = np.sin(half_angle)
    if sin_half < 1e-9:
        return np.inf
    return edge_length / (2.0 * sin_half)


def _unique_edges(faces: np.ndarray):
    """每条不同的无向边产出 (v0, v1, face_idx_array)——补丁边界边 1 个面，
    内部边 2 个面，非流形输入更多。
    """
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    edge_face_idx = np.tile(np.arange(len(faces)), 3)
    sorted_edges = np.sort(edges, axis=1)

    order = np.lexsort((sorted_edges[:, 1], sorted_edges[:, 0]))
    se = sorted_edges[order]
    efi = edge_face_idx[order]

    is_new = np.ones(len(se), dtype=bool)
    is_new[1:] = np.any(se[1:] != se[:-1], axis=1)
    boundaries = np.flatnonzero(is_new)
    boundaries = np.append(boundaries, len(se))

    for i in range(len(boundaries) - 1):
        lo, hi = boundaries[i], boundaries[i + 1]
        yield se[lo, 0], se[lo, 1], efi[lo:hi]
