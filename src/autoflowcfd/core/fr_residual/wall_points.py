"""壁面通量点的选取（壁面力积分与 WMLES 壁面模型共用同一组点）。

混合拆分面（棱柱四边形侧面一半在边界）上落在边界半区的通量点同样计入：它们的通量由配对的内部面记录
在掩码通量点上施加（`fr/face_flux_points/merge.py`）。
"""

from typing import Sequence

import numpy as np


def wall_flux_points(flat, group_code: np.ndarray, codes: Sequence[int]):
    """选出属于 `codes` 的边界面通量点：返回 (取本侧数据的面记录, 取幽灵态的面记录, 通量点下标, 组编码)。"""
    in_group = np.isin(group_code, np.asarray(list(codes), dtype=np.int64))
    n_fp = flat.n_fp
    rec, ghost, fp, code = [], [], [], []
    primary = np.flatnonzero(flat.is_boundary & flat.owner_is_primary & in_group)
    if primary.size:
        rec.append(np.repeat(primary, n_fp))
        ghost.append(np.repeat(primary, n_fp))
        fp.append(np.tile(np.arange(n_fp), primary.size))
        code.append(np.repeat(group_code[primary], n_fp))
    partner = np.asarray(flat.mixed_nb_partner)
    mixed = np.flatnonzero((partner >= 0) & flat.owner_is_primary & ~flat.is_boundary)
    mixed = mixed[in_group[partner[mixed]]]
    for f in mixed:
        pts = np.flatnonzero(flat.mixed_nb_mask[f])
        rec.append(np.full(pts.size, f))
        ghost.append(np.full(pts.size, partner[f]))
        fp.append(pts)
        code.append(np.full(pts.size, group_code[partner[f]]))
    if not rec:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty, empty
    return (np.concatenate(rec).astype(np.int64), np.concatenate(ghost).astype(np.int64),
            np.concatenate(fp).astype(np.int64), np.concatenate(code).astype(np.int64))
