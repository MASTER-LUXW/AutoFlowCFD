"""AutoFlowCFD V2.0 - 按面算子索引排好的原生面外插 / 提升整表（全部后端的残差核共用）。

面算子索引 `op = (code - 6) + FACE_OP_STRIDE * slot`：`code` 是原生面编码（四面体 6~9、
棱柱 10~14），`slot` 是三角形面的坍缩顶点槽位（`fr/triangle_apex.py`，四边形面恒 0）。
`slot = 0` 时 `op = code - 6`，与旧的按编码查表逐位一致。

四边形面（棱柱侧面）不存在的槽位填 NaN：一旦某处给四边形面算出了非零槽位，残差立刻
变成 NaN 暴露出来，而不是静默用错矩阵。
"""

import numpy as np

from ..native_padding import pad_native_matrix_to_global
from ..triangle_apex import N_TRI_SLOTS

#: 原生面编码个数（四面体 4 + 棱柱 5），即槽位之间的索引跨度。
FACE_OP_STRIDE = 9
#: 面算子总数。
N_FACE_OPS = FACE_OP_STRIDE * N_TRI_SLOTS
#: 原生面编码起点（与 `grid/connectivity/face_connectivity.py::NATIVE_FACE_CODE_BASE` 一致）。
_CODE_BASE = 6
#: 棱柱面编码起点。
_PRISM_CODE_BASE = 10


def face_op_index(code, slot):
    """面算子索引（标量或数组）。"""
    return np.asarray(code) - _CODE_BASE + FACE_OP_STRIDE * np.asarray(slot)


def is_triangle_face_code(code) -> np.ndarray:
    """原生面编码是否三角形面（四面体面 6~9、棱柱封盖 10~11）。"""
    code = np.asarray(code)
    return (code >= _CODE_BASE) & (code < _PRISM_CODE_BASE + 2)


def build_face_op_tables(tet_extrap, tet_lift, prism_extrap, prism_lift, n_sps: int):
    """由逐（面, 槽位）的未填充算子组装两张整表。

    Args:
        tet_extrap / tet_lift: `{(excluded_vertex, slot): 矩阵}`。
        prism_extrap / prism_lift: `{(face_id, slot): 矩阵}`。
        n_sps: 全局统一解点宽度（填充目标）。

    Returns:
        `(extrap (N_FACE_OPS, n_fp, n_sps), lift (N_FACE_OPS, n_sps, n_fp))`。
    """
    n_fp = next(iter(tet_extrap.values())).shape[0]
    E = np.full((N_FACE_OPS, n_fp, n_sps), np.nan)
    L = np.full((N_FACE_OPS, n_sps, n_fp), np.nan)
    for base, ex, li in ((0, tet_extrap, tet_lift), (_PRISM_CODE_BASE - _CODE_BASE, prism_extrap, prism_lift)):
        for (k, slot), mat in ex.items():
            op = base + k + FACE_OP_STRIDE * slot
            E[op] = pad_native_matrix_to_global(mat, n_sps, pad_axes=(1,))
            L[op] = pad_native_matrix_to_global(li[(k, slot)], n_sps, pad_axes=(0,))
    return np.ascontiguousarray(E), np.ascontiguousarray(L)
