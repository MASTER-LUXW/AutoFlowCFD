"""混合网格空腔修补的全局焊接状态与共形检查（`patch_nonmanifold_cavity_mixed` 专用）。

空腔边界上的近重合点对被焊接（`_weld_near_coincident_boundary_points`）之后，被合并掉的点必须在**全部**
单元里改引用幸存点，否则空腔外单元的面与重铺结果的边界面对不上、网格不再封闭（2026-10-09 以前就是这样：
cube_demo 体网格因此多出 86 个外部面，修补却报告成功）。本模块维护这份全局节点重映射，并给出两项检查：

- 空腔外受影响单元（引用了被合并点）换点后不能退化、不能翻转朝向；
- 重剖分结果的外表面必须与交给 tetgen 的边界逐面相同（`mesh_repair_cavity_shared.retile_is_conformal`）。
"""

import numpy as np

from .mesh_repair_cavity_shared import _repeated_node_rows
from .mesh_repair_nonmanifold_mixed_demote import _split_prisms_to_tets

# 棱柱 (v0,v1,v2,w0,w1,w2) 按槽位（不按全局索引排序）的 3 四面体分解：只用来判断换点前后朝向是否一致，
# 与全局索引无关，换点后仍是同一组槽位，前后可比。
_PRISM_SLOT_TETS = np.array([[0, 1, 2, 5], [0, 1, 4, 5], [0, 3, 4, 5]])


def _signed_dets(nodes: np.ndarray, tets: np.ndarray) -> np.ndarray:
    p0 = nodes[tets[..., 0]]
    return np.einsum('...i,...i->...', nodes[tets[..., 1]] - p0,
                     np.cross(nodes[tets[..., 2]] - p0, nodes[tets[..., 3]] - p0))


class GlobalWeldState:
    """全部已接受空腔的焊接累积成的全局节点重映射 + 受保护节点。

    受保护节点 = 域外边界节点（移动它会改变物体/外边界几何）∪ 已接受重铺引用的边界节点（移动它会改变一份
    已经过质量门的重铺）。焊接时受保护点只当幸存者、两个受保护点不合并。
    """

    def __init__(self, n_nodes: int, prism_cells: np.ndarray, tet_cells: np.ndarray, exterior_nodes: np.ndarray):
        self.node_remap = np.arange(n_nodes, dtype=np.int64)
        self.locked = np.zeros(n_nodes, dtype=bool)
        self.exterior = np.zeros(n_nodes, dtype=bool)
        self.exterior[exterior_nodes] = True
        self._prism = np.asarray(prism_cells, dtype=np.int64)
        self._tet = np.asarray(tet_cells, dtype=np.int64)
        self.n_prism = len(prism_cells)
        self._cur_prism = self._prism
        self._cur_tet = self._tet

    def protected(self) -> np.ndarray:
        return self.exterior | self.locked

    def current(self, cell_ids: np.ndarray) -> np.ndarray:
        """全局单元号（棱柱 [0,n_prism)、四面体其后）在当前重映射下的连接关系，作为四面体列表
        （棱柱按 `_split_prisms_to_tets` 拆成 3 个）。"""
        cell_ids = np.asarray(cell_ids, dtype=np.int64)
        prisms = cell_ids[cell_ids < self.n_prism]
        tets = cell_ids[cell_ids >= self.n_prism] - self.n_prism
        parts = []
        if len(prisms):
            parts.append(_split_prisms_to_tets(self._cur_prism[prisms]))
        if len(tets):
            parts.append(self._cur_tet[tets])
        return np.vstack(parts) if parts else np.empty((0, 4), dtype=np.int64)

    def affected_outside(self, removed: np.ndarray, in_cavity: np.ndarray) -> np.ndarray:
        """空腔外引用了 `removed` 中任一节点（当前重映射下）的全局单元号。"""
        hit_p = np.flatnonzero(np.isin(self._cur_prism, removed).any(axis=1)) if self.n_prism else \
            np.empty(0, dtype=np.int64)
        hit_t = np.flatnonzero(np.isin(self._cur_tet, removed).any(axis=1)) + self.n_prism
        hit = np.concatenate([hit_p, hit_t])
        return hit[~in_cavity[hit]]

    def _rows(self, cell_ids: np.ndarray, step: np.ndarray = None):
        cell_ids = np.asarray(cell_ids, dtype=np.int64)
        prisms = self._cur_prism[cell_ids[cell_ids < self.n_prism]]
        tets = self._cur_tet[cell_ids[cell_ids >= self.n_prism] - self.n_prism]
        if step is not None:
            prisms, tets = step[prisms], step[tets]
        return prisms, tets

    def degenerate_after(self, cell_ids: np.ndarray, step: np.ndarray) -> np.ndarray:
        """换点 `step` 之后出现重复节点的单元（全局单元号）。"""
        cell_ids = np.asarray(cell_ids, dtype=np.int64)
        prisms, tets = self._rows(cell_ids, step)
        bad = np.concatenate([_repeated_node_rows(prisms) if len(prisms) else np.zeros(0, dtype=bool),
                              _repeated_node_rows(tets) if len(tets) else np.zeros(0, dtype=bool)])
        ordered = np.concatenate([cell_ids[cell_ids < self.n_prism], cell_ids[cell_ids >= self.n_prism]])
        return ordered[bad]

    def orientation_preserved(self, nodes: np.ndarray, cell_ids: np.ndarray, step: np.ndarray) -> bool:
        """换点前后每个单元（棱柱按槽位分解）的有向体积同号且换点后非零。"""
        before_p, before_t = self._rows(cell_ids)
        after_p, after_t = self._rows(cell_ids, step)
        for before, after in ((before_p[:, _PRISM_SLOT_TETS] if len(before_p) else None,
                               after_p[:, _PRISM_SLOT_TETS] if len(after_p) else None),
                              (before_t if len(before_t) else None, after_t if len(after_t) else None)):
            if before is None:
                continue
            d0, d1 = _signed_dets(nodes, before), _signed_dets(nodes, after)
            if np.any(d1 == 0.0) or np.any(np.sign(d0) != np.sign(d1)):
                return False
        return True

    def as_tets_after(self, cell_ids: np.ndarray, step: np.ndarray) -> np.ndarray:
        prisms, tets = self._rows(cell_ids, step)
        parts = [_split_prisms_to_tets(prisms)] if len(prisms) else []
        if len(tets):
            parts.append(tets)
        return np.vstack(parts) if parts else np.empty((0, 4), dtype=np.int64)

    def accept(self, removed: np.ndarray, survivor: np.ndarray, boundary_pts: np.ndarray) -> None:
        """接受一个空腔：把它的焊接并进全局重映射，锁定它的边界点。"""
        if len(removed):
            step = np.arange(len(self.node_remap), dtype=np.int64)
            step[removed] = survivor
            self.node_remap = step[self.node_remap]
            self._cur_prism = self.node_remap[self._prism]
            self._cur_tet = self.node_remap[self._tet]
        self.locked[self.node_remap[boundary_pts]] = True
