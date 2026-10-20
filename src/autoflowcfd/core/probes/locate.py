"""AutoFlowCFD V2.0 - 探针定位：找到包含探针点的单元，并求出它在该单元里的参考坐标。

几何是直边单元（`grid/high_order`）：四面体映射是仿射的，参考坐标由重心坐标直接给出；棱柱映射对底面重心坐标与
挤出参数是双线性的，用 Newton 迭代反解。参考坐标约定见 `fr/point_interpolation.py`。

候选单元取包围盒含探针点的全部单元（逐单元比较，完整而非近邻启发式），逐个做精确包含判定；落在多个单元的公共
面 / 棱 / 顶点上时取全局编号最小的那个（FR 解跨单元间断，任一侧都是该点上的合法取值，取最小编号使结果确定）。
不在任何单元内的探针报错——探针放到物体里或计算域外是算例设置错误，不能静默挪到最近的单元。
"""

from dataclasses import dataclass

import numpy as np

#: 包含判定的容差（重心坐标 / 参考坐标，无量纲）：舍入误差量级之上、远小于任何真实的单元尺度
INSIDE_TOLERANCE = 1e-9
#: 棱柱参考坐标 Newton 反解的收敛阈值与迭代上限（直边棱柱的映射是双线性的，单元内的点几步即收敛）
_NEWTON_TOL = 1e-13
_NEWTON_MAX_ITER = 50


@dataclass(frozen=True)
class ProbeLocations:
    """探针所在单元与参考坐标。

    Attributes:
        points: `(k, 3)` 探针物理坐标
        cells: `(k,)` 全局单元编号（棱柱在前、四面体在后，与求解器状态的行顺序一致）
        is_prism: `(k,)` 所在单元是否为棱柱
        rst: `(k, 3)` 参考坐标
    """
    points: np.ndarray
    cells: np.ndarray
    is_prism: np.ndarray
    rst: np.ndarray


def _tet_rst(vertices: np.ndarray, point: np.ndarray) -> np.ndarray:
    """四面体参考坐标 `(r, s, t) = 2 (L1, L2, L3) - 1`（L 为顶点 p1~p3 的重心坐标）。"""
    A = (vertices[1:] - vertices[0]).T
    return 2.0 * np.linalg.solve(A, point - vertices[0]) - 1.0


def _tet_contains(rst: np.ndarray) -> bool:
    L = 0.5 * (rst + 1.0)
    return bool(L.min() >= -INSIDE_TOLERANCE and L.sum() <= 1.0 + INSIDE_TOLERANCE)


def _prism_map(vertices: np.ndarray, rst: np.ndarray):
    """直边棱柱映射 x(r,s,t) 与雅可比矩阵 dx/d(r,s,t)（顶点顺序 v0,v1,v2,w0,w1,w2，与
    `fr/native_prism/basis.py::map_native_prism_to_physical` 同一约定）。"""
    r, s, t = rst
    lam = np.array([-(r + s) / 2.0, (1.0 + r) / 2.0, (1.0 + s) / 2.0])
    dlam_dr = np.array([-0.5, 0.5, 0.0])
    dlam_ds = np.array([-0.5, 0.0, 0.5])
    bottom, top = vertices[:3], vertices[3:]
    wb, wt = 0.5 * (1.0 - t), 0.5 * (1.0 + t)
    x = wb * lam @ bottom + wt * lam @ top
    J = np.column_stack([wb * dlam_dr @ bottom + wt * dlam_dr @ top,
                         wb * dlam_ds @ bottom + wt * dlam_ds @ top,
                         0.5 * (lam @ top - lam @ bottom)])
    return x, J


def _prism_rst(vertices: np.ndarray, point: np.ndarray):
    """Newton 反解棱柱参考坐标；不收敛（点远在单元外、映射在该点附近不可逆）时返回 None。"""
    rst = np.array([-1.0 / 3.0, -1.0 / 3.0, 0.0])           # 单元形心
    scale = np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))
    for _ in range(_NEWTON_MAX_ITER):
        x, J = _prism_map(vertices, rst)
        residual = point - x
        if np.linalg.norm(residual) <= _NEWTON_TOL * scale:
            return rst
        try:
            rst = rst + np.linalg.solve(J, residual)
        except np.linalg.LinAlgError:
            return None
        if np.abs(rst).max() > 10.0:                          # 已远离参考单元，不可能在单元内
            return None
    return None


def _prism_contains(rst) -> bool:
    if rst is None:
        return False
    r, s, t = rst
    lam = np.array([-(r + s) / 2.0, (1.0 + r) / 2.0, (1.0 + s) / 2.0])
    return bool(lam.min() >= -INSIDE_TOLERANCE and abs(t) <= 1.0 + 2.0 * INSIDE_TOLERANCE)


def locate_probes(mesh, points) -> ProbeLocations:
    """在全局网格 `mesh`（`_node_coords`、`_fixed_prism_conn`、`_fixed_tet_conn`、`n_prism_cells`）里定位探针。

    Raises:
        ValueError: 探针不在任何单元内（列出探针与离它最近的单元形心距离）
    """
    points = np.atleast_2d(np.asarray(points, dtype=np.float64))
    if points.shape[1] != 3:
        raise ValueError(f"探针坐标应为 (k, 3)，收到 {points.shape}")
    nodes = np.asarray(mesh._node_coords, dtype=np.float64)
    n_prism = int(mesh.n_prism_cells)
    blocks = []                                   # (全局编号偏移, 连接表, 是否棱柱, 包围盒下界, 上界)
    for offset, conn, prism in ((0, mesh._fixed_prism_conn if n_prism > 0 else None, True),
                                (n_prism, mesh._fixed_tet_conn, False)):
        if conn is None or len(conn) == 0:
            continue
        conn = np.asarray(conn, dtype=np.int64)
        corners = nodes[conn]
        lo, hi = corners.min(axis=1), corners.max(axis=1)
        pad = INSIDE_TOLERANCE * np.linalg.norm(hi - lo, axis=1)[:, None]
        blocks.append((offset, conn, prism, lo - pad, hi + pad))

    cells = np.empty(len(points), dtype=np.int64)
    is_prism = np.empty(len(points), dtype=bool)
    rst_all = np.empty((len(points), 3))
    for k, p in enumerate(points):
        found = None
        for offset, conn, prism, lo, hi in blocks:
            for c in np.flatnonzero(np.all((lo <= p) & (p <= hi), axis=1)):
                vertices = nodes[conn[c]]
                if prism:
                    rst = _prism_rst(vertices, p)
                    ok = _prism_contains(rst)
                else:
                    rst = _tet_rst(vertices, p)
                    ok = _tet_contains(rst)
                if ok:
                    found = (offset + int(c), prism, rst)
                    break
            if found is not None:
                break
        if found is None:
            nearest = min(np.linalg.norm(0.5 * (lo + hi) - p, axis=1).min() for _, _, _, lo, hi in blocks)
            raise ValueError(f"探针 {k} {tuple(p)} 不在计算域的任何单元内（离它最近的单元包围盒中心 {nearest:.4g} m）："
                             "检查坐标单位（米）以及探针是否放进了物体内部或计算域外")
        cells[k], is_prism[k], rst_all[k] = found
    return ProbeLocations(points=points, cells=cells, is_prism=is_prism, rst=rst_all)
