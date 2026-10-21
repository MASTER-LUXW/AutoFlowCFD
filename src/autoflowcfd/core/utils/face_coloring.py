"""
AutoFlowCFD V2.0 - 面图着色（界面核 scatter-add 无写冲突的并行）

界面核把每个面的修正量累加到它触及的单元：内部面对 owner 与 neighbor 各做一次 `correction[cell] += ...`，边界面
只对 owner。面按颜色分组，同色的任意两个面不触及同一个单元，于是每个颜色组内可以 `prange`、直接写共享的
`correction`，没有写冲突、不需要每线程缓冲（界面核只有这一种并行写法，`inviscid_kernel.py` /
`viscous_flux_kernel.py` / 标量提升核按 `FlatFaceGeometry.color_face_indices` 逐色循环）。

冲突判定必须同时覆盖两侧的写入：只按 owner 着色会把"A 的 owner 是 B 的 neighbor"的两个面分进同一色，同色并行
循环里对同一单元非原子并发写入，静默丢失更新。

算法：贪心着色——按面序遍历，每个面取它的冲突面（与它触及同一单元的面）尚未使用的最小颜色。保持面序（不重排），
同一网格上逐位确定。典型 4~8 色（四面体网格）到 10~15 色（混合网格）。
"""

import numpy as np
from numba import njit


@njit(cache=True)
def _greedy_coloring_kernel(face_cells: np.ndarray, cell_ptr: np.ndarray, cell_faces: np.ndarray,
                            max_conflicts: int) -> np.ndarray:
    """`face_cells (n_faces, 2)`：面触及的单元（没有第二个时为 -1）；`cell_ptr / cell_faces`：单元 -> 触及它的面
    （CSR）。`seen[c] == f` 表示面 f 的某个冲突面已用颜色 c。"""
    n_faces = face_cells.shape[0]
    colors = np.full(n_faces, -1, dtype=np.int32)
    seen = np.full(max_conflicts + 1, -1, dtype=np.int64)
    for f in range(n_faces):
        for k in range(2):
            cell = face_cells[f, k]
            if cell < 0:
                continue
            for p in range(cell_ptr[cell], cell_ptr[cell + 1]):
                g = cell_faces[p]
                if g != f and colors[g] >= 0:
                    seen[colors[g]] = f
        c = 0
        while seen[c] == f:
            c += 1
        colors[f] = c
    return colors


def greedy_face_coloring(owner_cell: np.ndarray, neighbor_cell: np.ndarray, is_boundary: np.ndarray) -> np.ndarray:
    """贪心面图着色，`(n_faces,)` int32 颜色编号 0..n_colors-1。

    Args:
        owner_cell: (n_faces,) 每个面的 owner 单元
        neighbor_cell: (n_faces,) 每个面的 neighbor 单元（边界面处的值不读）
        is_boundary: (n_faces,) bool，边界面没有 neighbor 侧写入
    """
    owner = np.asarray(owner_cell, dtype=np.int64)
    neighbor = np.asarray(neighbor_cell, dtype=np.int64)
    boundary = np.asarray(is_boundary, dtype=np.bool_)
    n_faces = len(owner)
    if n_faces == 0:
        return np.zeros(0, dtype=np.int32)
    second = np.where(boundary | (neighbor < 0) | (neighbor == owner), -1, neighbor)
    face_cells = np.ascontiguousarray(np.stack([owner, second], axis=1))

    touched = face_cells.ravel()
    faces = np.repeat(np.arange(n_faces, dtype=np.int64), 2)
    keep = touched >= 0
    touched, faces = touched[keep], faces[keep]
    order = np.argsort(touched, kind="stable")
    n_cells = int(touched.max()) + 1
    degree = np.bincount(touched, minlength=n_cells)
    cell_ptr = np.zeros(n_cells + 1, dtype=np.int64)
    np.cumsum(degree, out=cell_ptr[1:])
    # 一个面最多触及两个单元：冲突面数不超过两侧单元的度之和
    return _greedy_coloring_kernel(face_cells, cell_ptr, np.ascontiguousarray(faces[order]), int(2 * degree.max()))
