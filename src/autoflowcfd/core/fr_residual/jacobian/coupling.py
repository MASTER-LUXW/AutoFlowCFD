"""AutoFlowCFD V2.0 - 面邻居耦合块 `J_cy = dR_c/dU_y` 的槽位布局与收尾。

`faces.py` 对每个 primary 面侧、每个来源（src0 / src1 / 混合拆分面配对边界面的
owner）输出一个耦合块，写进扁平 float32 数组里预先分好的槽位。槽位按
(行单元类型, 列单元类型) 分组连续排布，收尾时逐组整体向量化处理：

    除以 -det_s（R = -dU/dt）、右乘列单元的 dQ/dU、左乘行单元的 Gamma

然后把同一 (行, 列) 单元对的多个槽位（四边形面的两条 sources 记录常指向同一
单元）求和，得到按单元对去重的耦合块（`CouplingBlocks`，块 ILU 用）。
"""

from dataclasses import dataclass
from typing import List

import numpy as np

from .faces import N_CROSS_SOURCES
from .transform import finalize_pairs_kernel


@dataclass
class CouplingGroup:
    """一组 (行类型, 列类型) 的去重耦合块：`blocks[k]` 是 `dR_rows[k] / dU_cols[k]`。"""
    row_is_prism: bool
    col_is_prism: bool
    rows: np.ndarray        # (n_pairs,) int64 单元号
    cols: np.ndarray        # (n_pairs,) int64 单元号
    blocks: np.ndarray      # (n_pairs, n_row*5, n_col*5) float32


@dataclass
class CouplingBlocks:
    groups: List[CouplingGroup]

    @property
    def n_pairs(self) -> int:
        return int(sum(g.rows.size for g in self.groups))


def cross_layout(flat, n_prism, n_real_prism, n_real_tet):
    """槽位布局：返回 `(offset, expected_col, total_size, slots)`。

    `offset[f, side, src]`（`(n_faces,2,3)` int64，-1 表示该槽位不存在）是槽位在
    扁平数组里的起点，`expected_col` 是该槽位的列单元（界面核写回的列单元必须
    与它一致）；`slots` 是按分组排好序的 `(row_cell, col_cell, offset)` 与分组边界。
    """
    own = np.asarray(flat.owner_cell, dtype=np.int64)
    nei = np.asarray(flat.neighbor_cell, dtype=np.int64)
    bnd = np.asarray(flat.is_boundary, dtype=bool)
    rows, cols, keys = [], [], []

    def add(mask, row_cell, col_cell, side, src):
        f = np.nonzero(mask)[0]
        rows.append(row_cell[f])
        cols.append(col_cell[f])
        keys.append(np.stack([f, np.full(f.size, side), np.full(f.size, src)], axis=1))

    for side, primary, row_cell, pref, mixed_partner, mixed_mask in (
            (0, np.asarray(flat.owner_is_primary, dtype=bool), own, "neighbor",
             np.asarray(flat.mixed_nb_partner), np.asarray(flat.mixed_nb_mask)),
            (1, np.asarray(flat.neighbor_is_primary, dtype=bool) & ~bnd, nei, "owner",
             np.asarray(flat.mixed_ow_partner), np.asarray(flat.mixed_ow_mask))):
        interior = primary & ~bnd
        s0 = np.asarray(getattr(flat, pref + "_src0_cell"), dtype=np.int64)
        add(interior & (s0 >= 0), row_cell, s0, side, 0)
        i1 = np.asarray(getattr(flat, pref + "_src1_idx"), dtype=np.int64)
        c1_all = np.asarray(getattr(flat, pref + "_src1_cell"), dtype=np.int64)
        s1 = np.where(i1 >= 0, c1_all[np.maximum(i1, 0)] if c1_all.size else -1, -1)
        add(interior & (s1 >= 0), row_cell, s1, side, 1)
        has_mixed = interior & (mixed_partner >= 0) & mixed_mask.any(axis=1)
        partner_owner = np.where(mixed_partner >= 0, own[np.maximum(mixed_partner, 0)], -1)
        add(has_mixed & (partner_owner != row_cell), row_cell, partner_owner, side, 2)

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    keys = np.concatenate(keys)
    n_row = np.where(rows < n_prism, n_real_prism, n_real_tet)
    n_col = np.where(cols < n_prism, n_real_prism, n_real_tet)
    group = 2 * (rows >= n_prism) + (cols >= n_prism)          # 0: P-P, 1: P-T, 2: T-P, 3: T-T
    order = np.lexsort((cols, rows, group))
    rows, cols, keys, group = rows[order], cols[order], keys[order], group[order]
    sizes = 25 * n_row[order] * n_col[order]
    offs = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
    offset = -np.ones((flat.n_faces, 2, N_CROSS_SOURCES), dtype=np.int64)
    offset[keys[:, 0], keys[:, 1], keys[:, 2]] = offs
    expected_col = -np.ones_like(offset)
    expected_col[keys[:, 0], keys[:, 1], keys[:, 2]] = cols
    bounds = np.searchsorted(group, np.arange(5))
    return offset, expected_col, int(sizes.sum()), (rows, cols, offs, bounds)


def finalize_coupling(data, slots, det, TQ, gamma, use_gamma, n_real_prism, n_real_tet):
    """原始变量空间的耦合槽位 -> 守恒变量空间、按单元对去重的 `CouplingBlocks`。"""
    rows, cols, offs, bounds = slots
    groups = []
    for g in range(4):
        lo, hi = int(bounds[g]), int(bounds[g + 1])
        row_p, col_p = g < 2, g % 2 == 0
        nr = n_real_prism if row_p else n_real_tet
        ny = n_real_prism if col_p else n_real_tet
        if hi <= lo:
            continue
        r, c = rows[lo:hi], cols[lo:hi]
        # 组内已按 (行, 列) 排序：同一单元对的槽位相邻，逐段求和即去重
        new_pair = np.ones(hi - lo, dtype=bool)
        new_pair[1:] = (r[1:] != r[:-1]) | (c[1:] != c[:-1])
        first = np.nonzero(new_pair)[0].astype(np.int64)
        out = np.empty((first.size, nr, 5, ny, 5), dtype=np.float32)
        finalize_pairs_kernel(data, int(offs[lo]), nr, ny, first, hi - lo, r, c, det, TQ, gamma,
                              bool(use_gamma), out)
        groups.append(CouplingGroup(row_is_prism=row_p, col_is_prism=col_p, rows=r[first], cols=c[first],
                                    blocks=out.reshape(first.size, nr * 5, ny * 5)))
    return CouplingBlocks(groups=groups)
