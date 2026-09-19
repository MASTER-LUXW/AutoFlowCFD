"""AutoFlowCFD V2.0 - 三条基的节点 Vandermonde 逆矩阵与模态索引。

`build_fp_newton_parallel` / `build_ms_interp_parallel` 两个 numba kernel
在内部构造"面点 -> 体积节点"插值矩阵时需要 `V_sps^{-1}`（以及原生基的
模态索引三元组）。这里一次性备好，kernel 只做查表与矩阵乘。

从 `merge.py`（原 670 行）拆出（2026-09-19，项目"单文件不超 500 行"
规范）。整段原样搬运，逻辑一字未改。

## 三条基，两个"自动探测 + 零占位"原则

坍缩棱柱与坍缩四面体恒构造（任何网格都可能有这两类面）。两类**原生**
面则按 `face_conn` 里是否真的出现对应编码区间来决定：没出现就传零长度
数组，kernel 内对应分支永远不会被执行，既有行为逐位不变。

## 一个必须一致的约定

`lu_factor(V.T)` 之后再 `.T` —— 也就是 `v_sps_inv = V^{-1}`，**不是**
`V^{-T}`。三条基必须用同一个约定：搞反了不会报错，只会让插值矩阵变成
另一个矩阵（实测相对误差 5.44，而正确时是 1.0e-15）。
"""

from dataclasses import dataclass

import numpy as np

from autoflowcfd.grid.connectivity.face_connectivity import (
    NATIVE_PRISM_FACE_CODE_RANGE,
    FRFaceConnectivity,
)


@dataclass
class BasisInverses:
    """字段名与 `merge.py` 里原来的局部变量**同名**，便于逐行核对。"""

    v_sps_inv_tet: np.ndarray
    v_sps_inv_prism: np.ndarray
    v_sps_inv_native: np.ndarray
    native_mode_i: np.ndarray
    native_mode_j: np.ndarray
    native_mode_k: np.ndarray
    v_sps_inv_np: np.ndarray
    np_mode_i: np.ndarray
    np_mode_j: np.ndarray
    np_mode_k: np.ndarray


def build_basis_inverses(face_conn: FRFaceConnectivity, n1d: int,
                         sps_1d: np.ndarray) -> BasisInverses:
    """备好三条基的 `V_sps^{-1}` 与原生基的模态索引。"""
    from scipy.linalg import lu_factor, lu_solve

    # 预计算 V_sps 逆矩阵（用于 kernel 内插值矩阵构建）
    from .geometry import _get_v_sps_lu
    from scipy.linalg import lu_factor, lu_solve
    n_sps = n1d ** 3
    # 四面体 V_sps_inv
    lu_tet = _get_v_sps_lu("tet", n1d, sps_1d)
    I_n = np.eye(n_sps)
    v_sps_inv_tet = np.ascontiguousarray(lu_solve(lu_tet, I_n).T)
    # 棱柱 V_sps_inv
    lu_prism = _get_v_sps_lu("prism", n1d, sps_1d)
    v_sps_inv_prism = np.ascontiguousarray(lu_solve(lu_prism, I_n).T)
    # native 四面体（路径C）V_sps_inv + 模态索引——只在 face_conn 里真的
    # 出现过 native 四面体面（cube face code>=6，见
    # grid/connectivity/face_connectivity.py::with_native_face_codes）时
    # 才计算；不含 native 四面体的既有网格（默认坍缩坐标路径）传零长度
    # 占位数组，kernel 内对应分支（判据同样是 code>=6）永远不会被执行，
    # 不改变任何现有行为——这是本函数自动探测是否启用 native 分支的唯一
    # 入口，不需要单独的 tet_basis_mode 参数贯穿调用链（Part7 文档"实现
    # 顺序建议"里"求解器/网格加载路径接入 tet_basis_mode"仍是独立的、
    # 尚未做的后续工作，见该文档；这里只保证一旦上游把 face_conn 换成
    # `.with_native_face_codes()` 翻译后的版本，这条 numba 路径立即可用）。
    has_native_tet = bool(np.any(np.asarray(face_conn.owner_cube_face) >= 6)) or bool(
        np.any(np.asarray(face_conn.neighbor_cube_face) >= 6)
    )
    if has_native_tet:
        from .geometry import _get_v_sps_lu_native
        order_native = n1d - 1
        lu_native, modes_native = _get_v_sps_lu_native(order_native)
        n_native = len(modes_native)
        v_sps_inv_native = np.ascontiguousarray(lu_solve(lu_native, np.eye(n_native)).T)
        native_mode_i = np.array([m[0] for m in modes_native], dtype=np.int32)
        native_mode_j = np.array([m[1] for m in modes_native], dtype=np.int32)
        native_mode_k = np.array([m[2] for m in modes_native], dtype=np.int32)
    else:
        v_sps_inv_native = np.zeros((0, 0))
        native_mode_i = np.zeros(0, dtype=np.int32)
        native_mode_j = np.zeros(0, dtype=np.int32)
        native_mode_k = np.zeros(0, dtype=np.int32)

    # 原生**棱柱**（编码 [10,15)）的 Vandermonde 逆与模态索引 —— 与上面
    # 四面体那一段同一个"自动探测 + 零占位"原则：面编码里没出现原生棱柱面
    # 时传零长度数组，kernel 内对应分支（判据 `code >= _NATIVE_PRISM_LO`）
    # 永远不会被执行，既有行为逐位不变。
    #
    # `lu_factor(V.T)` 之后再 `.T` —— 与四面体那条**同一个约定**
    # （`v_sps_inv = V^{-1}`，不是 `V^{-T}`）。这一点必须一致：搞反了不会
    # 报错，只会让插值矩阵变成另一个矩阵（实测相对误差 5.44，而正确时是
    # 1.0e-15）。
    _pf_lo, _pf_hi = NATIVE_PRISM_FACE_CODE_RANGE
    _oc = np.asarray(face_conn.owner_cube_face)
    _nc = np.asarray(face_conn.neighbor_cube_face)
    has_native_prism = bool(
        np.any((_oc >= _pf_lo) & (_oc < _pf_hi))
        or np.any((_nc >= _pf_lo) & (_nc < _pf_hi)))
    if has_native_prism:
        from autoflowcfd.fr.native_prism.basis import (
            build_native_prism_nodes,
            build_native_prism_vandermonde,
            restricted_prism_modes,
        )

        _order_np = n1d - 1
        _ref_np = build_native_prism_nodes(_order_np)
        _V_np, _, _, _ = build_native_prism_vandermonde(_order_np, _ref_np)
        _modes_np = restricted_prism_modes(_order_np)
        _n_np = len(_modes_np)
        if _V_np.shape != (_n_np, _n_np):
            raise ValueError(
                f"原生棱柱节点 Vandermonde 形状 {_V_np.shape} 应为方阵 "
                f"({_n_np}, {_n_np}) —— 节点数与模态数理论上必须相等")
        v_sps_inv_np = np.ascontiguousarray(
            lu_solve(lu_factor(_V_np.T), np.eye(_n_np)).T)
        np_mode_i = np.array([m[0] for m in _modes_np], dtype=np.int32)
        np_mode_j = np.array([m[1] for m in _modes_np], dtype=np.int32)
        np_mode_k = np.array([m[2] for m in _modes_np], dtype=np.int32)
    else:
        v_sps_inv_np = np.zeros((0, 0))
        np_mode_i = np.zeros(0, dtype=np.int32)
        np_mode_j = np.zeros(0, dtype=np.int32)
        np_mode_k = np.zeros(0, dtype=np.int32)

    return BasisInverses(
        v_sps_inv_tet=v_sps_inv_tet,
        v_sps_inv_prism=v_sps_inv_prism,
        v_sps_inv_native=v_sps_inv_native,
        native_mode_i=native_mode_i,
        native_mode_j=native_mode_j,
        native_mode_k=native_mode_k,
        v_sps_inv_np=v_sps_inv_np,
        np_mode_i=np_mode_i,
        np_mode_j=np_mode_j,
        np_mode_k=np_mode_k,
    )
