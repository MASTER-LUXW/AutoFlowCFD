"""AutoFlowCFD V2.0 - 平均流单元块装配的粗层约化（块 Jacobi 档的单元级粗校正用）。

粗校正（`time_integration/implicit/coarse/cell_coarse.py`）要的是

    A_c[c, y] = sum_s w[c, s] sum_t J_cy[s, :, t, :]        （面邻居单元对，5x5）
    A_c[c, c] = sum_s w[c, s] sum_t J_cc[s, :, t, :]

完整耦合块（块 ILU 档，`coupling.py`）在 P2 大网格上几十 GB，这里不存它：界面核每算出一个耦合槽位就当场约化成
5x5 并累加到所属单元对（`transform.py::coarse_reduce_slot`；同色面不共享单元，行单元在一个颜色内唯一，累加
无写冲突），对角粗块由收尾后的 `J_cc` 求和（`coarse_diag_kernel`），人工粘性的标量耦合块（规模是平均流
耦合块的 1/25）按单元对约化后并入。

槽位与单元对：`coupling.py::cross_slots` 给出全部槽位，同一单元对（四边形面的两条来源记录、两侧都是同一列单元
的来源）合并成一个粗块。
"""

from dataclasses import dataclass

import numpy as np

from .faces import N_CROSS_SOURCES


@dataclass
class CoarseBlocks:
    """一次装配的粗层块：对角 `(n_cells, 5, 5)`，面邻居单元对 `rows/cols`（不含对角）与块 `(n_pairs, 5, 5)`。"""
    diag: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    blocks: np.ndarray


@dataclass
class CoarsePairLayout:
    """界面核的粗块去向：`pair[f, side, src]` 为单元对编号（-1 为该槽位不存在），`expected_col` 为该槽位的列单元
    （界面核写回的必须与它一致），`rows/cols` 为各单元对的行 / 列单元（可含对角，收尾时并入对角块）。"""
    pair: np.ndarray
    expected_col: np.ndarray
    rows: np.ndarray
    cols: np.ndarray


def coarse_pair_layout(flat, n_cells: int) -> CoarsePairLayout:
    from .coupling import cross_slots

    rows, cols, keys = cross_slots(flat, include_partner=True)
    key = rows * int(n_cells) + cols
    uniq, inv = np.unique(key, return_inverse=True)
    pair = -np.ones((flat.n_faces, 2, N_CROSS_SOURCES), dtype=np.int64)
    pair[keys[:, 0], keys[:, 1], keys[:, 2]] = inv
    expected = -np.ones_like(pair)
    expected[keys[:, 0], keys[:, 1], keys[:, 2]] = cols
    return CoarsePairLayout(pair, expected, (uniq // n_cells).astype(np.int64), (uniq % n_cells).astype(np.int64))


def add_scalar_coupling_coarse(out, layout: CoarsePairLayout, scalar, w, gamma, use_gamma: bool, n_cells: int):
    """标量耦合块（`-L_nu`，Newton 空间、作用到每个守恒变量：`B[s,a,t,d] += G_s[a,d] scalar[s,t]`）约化并累加到
    单元对粗块 `out`：`out[p] += sum_s w_s G_s sum_t scalar[s, t]`。单元对必须在 `layout` 里。"""
    key_all = layout.rows * int(n_cells) + layout.cols
    for g in scalar.groups:
        if not g.rows.size:
            continue
        p = np.searchsorted(key_all, g.rows * int(n_cells) + g.cols)
        if np.any(p >= key_all.size) or np.any(key_all[np.minimum(p, key_all.size - 1)] != g.rows * n_cells + g.cols):
            raise RuntimeError("标量耦合块的单元对不在平均流粗块布局里")
        nr = g.blocks.shape[1]
        row_sum = np.asarray(g.blocks, dtype=np.float64).sum(axis=2) * w[g.rows, :nr]     # (n_g, nr)
        if use_gamma:
            contrib = np.einsum("ks,ksad->kad", row_sum, np.asarray(gamma[g.rows, :nr], dtype=np.float64))
        else:
            contrib = row_sum.sum(axis=1)[:, None, None] * np.eye(5)
        np.add.at(out, p, contrib)


def finish_coarse_blocks(diag, layout: CoarsePairLayout, pair_blocks) -> CoarseBlocks:
    """单元对里行列相同的（来源单元就是行单元本身）并入对角，其余作为耦合块。"""
    on = layout.rows == layout.cols
    if np.any(on):
        np.add.at(diag, layout.rows[on], pair_blocks[on])
    off = ~on
    return CoarseBlocks(diag=diag, rows=layout.rows[off], cols=layout.cols[off],
                        blocks=np.ascontiguousarray(pair_blocks[off]))
