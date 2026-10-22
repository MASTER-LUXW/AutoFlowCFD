"""尖锐边判定用的几何辅助：一条边隐含的局部曲率半径（`implied_edge_radius`）。

面法向用 `mesh_utils.py::compute_face_normals`，唯一边与边-面关联用 `mesh_utils.py::face_edge_incidence`（各自的
唯一实现）。
"""

import numpy as np


def implied_edge_radius(edge_length, dihedral_angle):
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

    逐元素（标量或数组）；`sin(theta/2) < 1e-9`（两面近乎共面）时半径为无穷。
    """
    sin_half = np.sin(np.asarray(dihedral_angle, dtype=np.float64) / 2.0)
    flat = sin_half < 1e-9
    radius = np.where(flat, np.inf, np.asarray(edge_length, dtype=np.float64) / (2.0 * np.where(flat, 1.0, sin_half)))
    return radius if radius.ndim else float(radius)
