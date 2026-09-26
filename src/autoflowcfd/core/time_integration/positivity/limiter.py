"""AutoFlowCFD V2.0 - 正性保持限制器对象：从几何与算子构建，逐 RK stage 调用。

数值核心见同目录 `zhang_shu.py`；本模块只负责准备它需要的静态数据：

* 守恒权重 `W_cs = w_s * det(J)_cs`：棱柱 `w` 来自
  `fr/native_prism/quadrature.build_native_prism_sp_weights`，四面体来自
  `fr/native_tet/quadrature.build_native_tet_sp_weights`（两者同一构造）；
* 每类单元解点之外全部通量求值点的插值行：各面通量点（`fr_operators/
  face_kernels.native_face_extrap_stack`，与平面面几何同一个组装处；棱柱面
  编码 10~14、四面体 6~9，按 `code - 6` 取）与过积分细点（`ops.overint_
  interp_c2f_*`）——同一点集也供隐式 Newton 步的物理性限幅使用
  （`PositivityLimiter.density_pressure_limits`）；
* 真实解点数（`fr/native_padding.real_sps_per_cell`，补零槽位不参与）；
* 逐单元类型掩码 `cell_is_prism` —— **不假设单元顺序**：单机网格棱柱在前，
  分布式本地编号里两类单元交错，两者用同一个核。

阶数切换后必须重建（权重、外插行、解点数都随阶数变），见
`get_positivity_limiter` 的缓存键。
"""

import numpy as np

from autoflowcfd.core.fr_operators.flux_kernels import GAMMA

from .zhang_shu import zhang_shu_numba, zhang_shu_xp


class PositivityLimiter:
    """就地限制扁平守恒量 `U_flat` `(n_cells*n_sps, n_vars)` 的可调用对象。

    Attributes:
        n_calls / n_limited_total / min_theta: 累计统计（诊断用）。
    """

    def __init__(self, W, E_prism, E_tet, cell_is_prism, n_real_prism, n_real_tet,
                 xp=np, gamma=GAMMA):
        self.xp = xp
        self.W = W
        self.E_prism = E_prism
        self.E_tet = E_tet
        self.cell_is_prism = cell_is_prism
        self.n_real_prism = int(n_real_prism)
        self.n_real_tet = int(n_real_tet)
        self.gamma = float(gamma)
        self.n_cells, self.n_sps = W.shape
        self.n_calls = 0
        self.n_limited_total = 0
        self.min_theta = 1.0
        if xp is np:
            self._theta = np.ones(self.n_cells)
            self._bad = np.zeros(self.n_cells, dtype=np.bool_)

    def __call__(self, U_flat):
        xp = self.xp
        n_vars = U_flat.shape[-1]
        U3 = U_flat.reshape(self.n_cells, self.n_sps, n_vars)
        if xp is np:
            zhang_shu_numba(U3, self.W, self.E_prism, self.E_tet, self.cell_is_prism,
                            self.n_real_prism, self.n_real_tet, self.gamma,
                            self._theta, self._bad)
            theta, bad = self._theta, self._bad
        else:
            theta, bad = zhang_shu_xp(xp, U3, self.W, self.E_prism, self.E_tet,
                                      self.cell_is_prism, self.n_real_prism,
                                      self.n_real_tet, self.gamma)
        self.n_calls += 1
        n_lim = int((theta < 1.0).sum())
        if n_lim:
            self.n_limited_total += n_lim
            self.min_theta = min(self.min_theta, float(theta.min()))
        if bool(bad.any()):
            from autoflowcfd.core.fr_solver.residual_diagnostics import SolverDivergedError

            cells = xp.where(bad)[0]
            first = [int(c) for c in cells[:8].tolist()]
            raise SolverDivergedError(
                f"{int(bad.sum())} 个单元的**单元均值**已不可容许（密度或压力 <= 0，"
                f"或非有限）—— 例如单元 {first}。均值的正性由格式在 CFL 条件下"
                f"保证，它坏了就是真正的发散，正性限制器不能也不应修补它"
                f"（向一个不可容许的均值收缩没有意义）。常见成因：CFL 超出稳定"
                f"边界、网格质量门未通过而强行求解。")
        return U_flat

    def density_pressure_limits(self, u0_flat, du_flat, red):
        """隐式 Newton 步的物理性限幅，在**与本限制器同一个点集**上求值（解点 +
        面通量点 + 过积分细点），返回逐行 `alpha`（同一单元的行取单元内最小值）。

        规则本身是 `implicit/jfnk.py::density_pressure_row_limits`（rho 与 p 单步
        相对变化对数对称不超过 50%），这里只把求值点从解点扩到全部通量求值点：
        Newton 步从可容许态出发、每个点都限相对变化，就不会走进任何求值点上
        rho、p 非正的状态。此前只看解点，plate_demo 锐边贴壁单元解点都正常而一个
        细点 rho 被逐步推到 7e-10、p<0，体积通量在那里奇异，平均流从此每步被拒。
        """
        from autoflowcfd.core.time_integration.implicit.physicality import density_pressure_row_limits

        xp = self.xp
        n_var = u0_flat.shape[1]
        U0 = u0_flat.reshape(self.n_cells, self.n_sps, n_var)
        D = du_flat.reshape(self.n_cells, self.n_sps, n_var)
        alpha_cell = xp.ones(self.n_cells)
        for sel, nr, E in ((self.cell_is_prism, self.n_real_prism, self.E_prism),
                           (~self.cell_is_prism, self.n_real_tet, self.E_tet)):
            idx = xp.nonzero(sel)[0]
            if idx.size == 0:
                continue
            Ec = E[:, :nr]
            u0c, dc = U0[idx, :nr], D[idx, :nr]
            P0 = xp.concatenate([u0c, xp.einsum("qs,csv->cqv", Ec, u0c)], axis=1)
            PD = xp.concatenate([dc, xp.einsum("qs,csv->cqv", Ec, dc)], axis=1)
            a = density_pressure_row_limits(P0.reshape(-1, n_var), PD.reshape(-1, n_var), red)
            alpha_cell[idx] = a.reshape(idx.size, -1).min(axis=1)
        return xp.repeat(alpha_cell, self.n_sps)

    def summary(self) -> str:
        return (f"positivity limiter: {self.n_calls} stage 调用，累计限制 "
                f"{self.n_limited_total} 个单元次，最小 θ={self.min_theta:.4g}")


def build_positivity_limiter_from_arrays(det_jacs, cell_is_prism, ops, order, xp=np):
    """按逐单元几何构建限制器（单机、分布式共用的唯一构造处）。

    Args:
        det_jacs: `(n_cells, n_sps)` 解点上的 det(J)，与 `U` 同一单元顺序。
        cell_is_prism: `(n_cells,)` 布尔，同一单元顺序。
        ops: `FROperators`（当前阶数）。
        order: 当前阶数。
        xp: 数组模块（numpy 或 cupy）。cupy 时静态数组搬上设备、走向量化版。
    """
    from autoflowcfd.core.fr_operators.face_kernels import native_face_extrap_stack
    from autoflowcfd.fr.native_padding import real_sps_per_cell
    from autoflowcfd.fr.native_prism.quadrature import build_native_prism_sp_weights
    from autoflowcfd.fr.native_tet.quadrature import build_native_tet_sp_weights

    order = int(order)
    det = np.asarray(det_jacs, dtype=np.float64)
    is_prism = np.ascontiguousarray(np.asarray(cell_is_prism, dtype=np.bool_))
    n_cells, n_sps = det.shape
    if is_prism.shape != (n_cells,):
        raise ValueError(f"cell_is_prism 形状 {is_prism.shape} 与 det_jacs 的单元数 {n_cells} 不符")
    n_real_prism, n_real_tet = real_sps_per_cell(order)

    w = np.zeros((n_cells, n_sps))
    w[is_prism, :n_real_prism] = build_native_prism_sp_weights(order)[None, :]
    w[~is_prism, :n_real_tet] = build_native_tet_sp_weights(order)[None, :]
    W = np.ascontiguousarray(w * det)

    Ebn = native_face_extrap_stack(ops, n_sps)
    E_tet = [Ebn[0:4].reshape(-1, n_sps)]       # 面编码 6..9
    E_prism = [Ebn[4:9].reshape(-1, n_sps)]     # 面编码 10..14
    # 过积分细点：无粘（与湍流输运）体积通量就在这些点上求值，必须同样可容许
    # （2026-09-26：plate_demo 锐边贴壁单元一个细点 rho=7e-10、p<0，解点与通量点
    # 都正常，体积通量在那里按 1/rho 奇异，残差对 1e-5 的扰动跳 13 个数量级）
    for rows, name in ((E_prism, "overint_interp_c2f_prism"), (E_tet, "overint_interp_c2f_tet")):
        c2f = getattr(ops, name, None)
        if c2f is not None:
            rows.append(np.asarray(c2f)[:, :n_sps])
    E_prism = np.ascontiguousarray(np.vstack(E_prism))
    E_tet = np.ascontiguousarray(np.vstack(E_tet))

    if xp is not np:
        W, E_prism, E_tet, is_prism = (xp.asarray(a) for a in (W, E_prism, E_tet, is_prism))
    return PositivityLimiter(W, E_prism, E_tet, is_prism, n_real_prism, n_real_tet, xp=xp)


def build_positivity_limiter(mesh, ops, order=None, xp=np):
    """按单机 `HighOrderMesh`（棱柱在前 `[0, n_prism)`）构建限制器。"""
    n_cells = int(mesh.n_cells)
    det = np.asarray(mesh.jacobians["det_jacs"]).reshape(n_cells, int(mesh.n_sps_per_cell))
    return build_positivity_limiter_from_arrays(
        det, np.arange(n_cells) < int(mesh.n_prism_cells), ops,
        int(mesh.order if order is None else order), xp=xp)


def get_positivity_limiter(solver, xp=np, geometry=None):
    """取（必要时重建）挂在求解器上的限制器。全部后端共用这一个取用入口。

    静态数据只随**阶数**与网格变化，所以按 `(阶数, 每单元解点数, 网格对象,
    算子对象, 数组模块)` 缓存；Order Continuation 切阶后自动重建。

    Args:
        geometry: 可选回调 `() -> (det_jacs, cell_is_prism)`，给出与状态
            数组同一单元顺序的几何（分布式本地编号用）；缺省按单机网格。
            只在缓存失效时调用。
    """
    mesh = solver.mesh
    # 当前阶数：`current_order` 为 None（GPU 求解器未走 Order Continuation 时）即目标阶数
    order = getattr(solver, "current_order", None)
    if order is None:
        order = getattr(solver, "order", getattr(mesh, "order", 1))
    order = int(order)
    key = (order, int(mesh.n_sps_per_cell), id(mesh), id(solver.ops), xp.__name__)
    cached = getattr(solver, "_positivity_limiter", None)
    if cached is not None and getattr(solver, "_positivity_limiter_key", None) == key:
        return cached
    if geometry is None:
        lim = build_positivity_limiter(mesh, solver.ops, order=order, xp=xp)
    else:
        det_jacs, cell_is_prism = geometry()
        lim = build_positivity_limiter_from_arrays(det_jacs, cell_is_prism, solver.ops, order, xp=xp)
    solver._positivity_limiter = lim
    solver._positivity_limiter_key = key
    return lim
