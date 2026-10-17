"""WMLES 壁面模型的匹配点（Kawai & Larsson 2012：壁面模型输入取在域内距壁 h_wm 处的 LES 解，不取壁面本身）。

每个壁面通量点 x_fp（单位外法向 n，流体 -> 壁）沿内法向取匹配点

    x_m = x_fp - h_wm n，  h_wm = d - n . c

c 为 owner 单元的积分形心、`n . x = d` 为壁面面记录所在平面（`FlatFaceGeometry.face_plane`），即匹配点与形心
同高（贴壁单元的中部）。h_wm 取自平面而不是通量点坐标：P0 每面一个通量点，解点外插只能得到单元中心。匹配点在 owner 单元里的参考坐标由解点坐标的
插值多项式反解（直边单元的几何映射在该多项式空间里，P>=1 时插值精确；四面体一次 Newton 即精确，棱柱为双线性、
几步收敛）。斜单元上匹配点若越出 owner 单元，沿 x_fp -> x_m 线段二分到单元内最远点，h_wm 按实际取值。
P0 每单元一个解点（单元值），插值行即该解点。

通量点集合与壁面力积分同一个（`core/fr_residual/wall_points.py::wall_flux_points`）：含棱柱四边形侧面一半在
边界时、由配对内部面记录施加的边界半区通量点——WMLES 下这些点的幽灵态同样是滑移镜像，漏掉它们就没有壁面剪应力。

几何不随状态变化：插值行、h_wm、法向、面积权重、提升列与 1/det 在构造时算好，每次残差求值只做小矩阵乘。
"""

from dataclasses import dataclass

import numpy as np

_INSIDE_TOL = 1e-10
_NEWTON_TOL = 1e-13
_NEWTON_MAX_ITER = 30
_BISECT_ITER = 40

#: 放到设备上的逐点数组（`WallMatching.to`）
DEVICE_FIELDS = ("owner", "E_m", "h_wm", "normal", "area", "lift", "inv_det")


@dataclass
class WallMatching:
    """逐壁面通量点（m 个）的匹配点几何。"""
    face: np.ndarray        # (m,) 施加通量的面记录（边界主记录，或混合拆分面的内部配对记录）
    fp: np.ndarray          # (m,) 面内通量点下标
    code: np.ndarray        # (m,) 所属边界组编码
    owner: np.ndarray       # (m,) owner 单元
    E_m: np.ndarray         # (m, n_sps) 解点 -> 匹配点插值行
    h_wm: np.ndarray        # (m,) 匹配点距壁距离
    normal: np.ndarray      # (m, 3) 单位外法向（流体 -> 壁）
    area: np.ndarray        # (m,) 物理面积权重
    lift: np.ndarray        # (m, n_sps) owner 面提升算子的该通量点列
    inv_det: np.ndarray     # (m, n_sps) owner 单元解点上的 1/det(J)

    def to(self, xp) -> "WallMatching":
        """数组放到 xp 所在的设备上（numpy 原样返回）。"""
        if xp.__name__ == "numpy":
            return self
        fields = {name: getattr(self, name) for name in self.__dataclass_fields__}
        fields.update({name: xp.asarray(getattr(self, name)) for name in DEVICE_FIELDS})
        return WallMatching(**fields)


def _nodal_basis(order: int, prism: bool, ref_pts: np.ndarray):
    """原生节点基（解点上的拉格朗日基）在任意参考点处的值 `(n, n_native)` 与梯度 `(n, n_native, 3)`。"""
    if prism:
        from autoflowcfd.fr.native_prism.basis import build_native_prism_nodes, build_native_prism_vandermonde

        V_sps = build_native_prism_vandermonde(order, build_native_prism_nodes(order))[0]
        V, Vr, Vs, Vt = build_native_prism_vandermonde(order, ref_pts)
    else:
        from autoflowcfd.fr.native_tet.basis import (
            build_native_tet_operators, restricted_tet_modes, rst_to_abc, simplex3d_grad, simplex3d_value,
        )

        nodes, _ = build_native_tet_operators(order)
        modes = restricted_tet_modes(order)
        a0, b0, c0 = rst_to_abc(nodes[:, 0], nodes[:, 1], nodes[:, 2])
        a, b, c = rst_to_abc(ref_pts[:, 0], ref_pts[:, 1], ref_pts[:, 2])
        V_sps = np.column_stack([simplex3d_value(a0, b0, c0, *m) for m in modes])
        V = np.column_stack([simplex3d_value(a, b, c, *m) for m in modes])
        grads = [simplex3d_grad(a, b, c, *m) for m in modes]
        Vr = np.column_stack([g[0] for g in grads])
        Vs = np.column_stack([g[1] for g in grads])
        Vt = np.column_stack([g[2] for g in grads])
    inv = np.linalg.inv(V_sps)
    return V @ inv, np.stack([Vr @ inv, Vs @ inv, Vt @ inv], axis=-1)


def _inside(prism: bool, xi: np.ndarray) -> np.ndarray:
    r, s, t = xi[:, 0], xi[:, 1], xi[:, 2]
    if prism:
        return (r >= -1 - _INSIDE_TOL) & (s >= -1 - _INSIDE_TOL) & (r + s <= _INSIDE_TOL) & (np.abs(t) <= 1 + _INSIDE_TOL)
    return (r >= -1 - _INSIDE_TOL) & (s >= -1 - _INSIDE_TOL) & (t >= -1 - _INSIDE_TOL) & (r + s + t <= -1 + _INSIDE_TOL)


def _locate(order: int, prism: bool, X: np.ndarray, x: np.ndarray) -> np.ndarray:
    """逐点 Newton 反解：X (n, n_native, 3) 为各点所在单元的解点坐标，x (n, 3) 目标物理点 -> 参考坐标 (n, 3)。"""
    xi = np.tile(np.array([-1.0 / 3.0, -1.0 / 3.0, 0.0]) if prism else np.array([-0.5, -0.5, -0.5]), (len(x), 1))
    scale = np.maximum(np.max(np.abs(X - X.mean(axis=1, keepdims=True)), axis=(1, 2)), 1e-300)
    for _ in range(_NEWTON_MAX_ITER):
        phi, dphi = _nodal_basis(order, prism, xi)
        resid = np.einsum('ns,nsd->nd', phi, X) - x
        if np.all(np.linalg.norm(resid, axis=1) <= _NEWTON_TOL * scale):
            return xi
        J = np.einsum('nsk,nsd->ndk', dphi, X)
        xi = xi - np.linalg.solve(J, resid[..., None])[..., 0]
    phi, _ = _nodal_basis(order, prism, xi)
    worst = float(np.max(np.linalg.norm(np.einsum('ns,nsd->nd', phi, X) - x, axis=1) / scale))
    raise RuntimeError(f"WMLES 匹配点参考坐标反解 {_NEWTON_MAX_ITER} 次未收敛（最大相对残差 {worst:.3e}）")


def build_wall_matching(mesh, flat, points, cell_is_prism: np.ndarray, W: np.ndarray, det: np.ndarray,
                        order: int) -> WallMatching:
    """壁面通量点的匹配点几何（状态无关）。

    Args:
        mesh: 提供 `sps_coords (n_cells, n_sps, 3)`（单机网格或分布式紧凑空间适配器）
        flat: 面几何（与残差组装同一个索引空间）
        points: `wall_flux_points` 的 `(face, ghost, fp, code)`
        cell_is_prism: (n_cells,) 单元类型
        W: 解点积分权重 (n_cells, n_sps)（`fr/native_padding.py::sp_integration_weights`），求积分形心用
        det: 解点 det(J) (n_cells, n_sps)
        order: 多项式阶数
    """
    from autoflowcfd.fr.native_padding import real_sps_per_cell

    face, ghost, fp, code = points
    owner = np.asarray(flat.owner_cell)[face].astype(np.int64)
    face_op = np.asarray(flat.owner_face_op)[face].astype(np.int64)
    X_sp = np.asarray(mesh.sps_coords, dtype=np.float64)
    m, n_sps = len(face), X_sp.shape[1]

    E_o = np.asarray(flat.boundary_extrap_native)[face_op, fp]                 # (m, n_sps)
    x_fp = np.einsum('ks,ksd->kd', E_o, X_sp[owner])
    plane = np.asarray(flat.face_plane)[ghost]                                 # 壁面边界记录的平面
    normal = np.ascontiguousarray(plane[:, :3])
    area = np.asarray(flat.true_area_weight)[face, fp]
    lift = np.ascontiguousarray(np.asarray(flat.lift_native)[face_op, :, fp])  # (m, n_sps)
    inv_det = np.where(W[owner] > 0.0, 1.0 / det[owner], 0.0)                 # 填充槽位（权重 0）不写残差
    centroid = np.einsum('cs,csd->cd', W, X_sp) / W.sum(axis=1)[:, None]
    h_wm = plane[:, 3] - np.einsum('kd,kd->k', normal, centroid[owner])
    if np.any(h_wm <= 0.0):
        raise ValueError(f"{int(np.count_nonzero(h_wm <= 0.0))} 个壁面通量点的 owner 单元形心不在流体一侧（法向朝向错误）")

    E_m = np.zeros((m, n_sps))
    if order == 0:
        E_m[:, 0] = 1.0
    else:
        n_real_prism, n_real_tet = real_sps_per_cell(order)
        is_prism_pt = np.asarray(cell_is_prism)[owner]
        for prism, n_real in ((True, n_real_prism), (False, n_real_tet)):
            sel = np.flatnonzero(is_prism_pt == prism)
            if sel.size == 0:
                continue
            X = X_sp[owner[sel], :n_real]
            p0 = x_fp[sel]
            dirn = h_wm[sel, None] * normal[sel]
            lam = np.ones(sel.size)
            xi = _locate(order, prism, X, p0 - dirn)
            out = ~_inside(prism, xi)
            if out.any():
                lo, hi = np.zeros(out.sum()), np.ones(out.sum())
                Xo, po, do = X[out], p0[out], dirn[out]
                for _ in range(_BISECT_ITER):
                    mid = 0.5 * (lo + hi)
                    ok = _inside(prism, _locate(order, prism, Xo, po - mid[:, None] * do))
                    lo = np.where(ok, mid, lo)
                    hi = np.where(ok, hi, mid)
                lam[out] = lo
                xi[out] = _locate(order, prism, Xo, po - lo[:, None] * do)
            E_m[sel, :n_real] = _nodal_basis(order, prism, xi)[0]
            h_wm[sel] = h_wm[sel] * lam
    return WallMatching(face=face, fp=fp, code=code, owner=owner, E_m=E_m, h_wm=h_wm, normal=normal, area=area,
                        lift=lift, inv_det=inv_det)
