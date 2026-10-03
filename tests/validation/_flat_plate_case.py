# -*- coding: utf-8 -*-
"""湍流平板（NASA TMR 2DZP 同型）：定常解稳定的 RANS 验证算例。

## 为什么需要它

现有真实网格（plate_demo、cube_demo、ahmed_demo）都是钝体或大分离尾迹，Re~1e6 的钝体
物理上涡脱落、定常 RANS 解不稳定，伪时间推进会跟随不稳定模态（项目记忆
`p1_newton_k_crossing_nonlinearity`）——工业做法是那类问题改用非定常（URANS/DES）。
"P2/P3 快速稳定收敛"要在**定常解稳定**的外流上考核；零压梯度湍流平板是工业 RANS 求解器
的标准验证算例（NASA Turbulence Modeling Resource 2DZP），有公认的摩阻参考。

## 算例

    来流 U = 30 m/s，rho = 1.225，p = 101325；Re = 5e6 / m（与 TMR 同一雷诺数）
    -> mu = rho U / 5e6 = 7.35e-6 Pa s（空气的约 0.4 倍：命中目标雷诺数而不改来流，
       与 Blasius 算例同一做法）

    区域 x in [X_IN, L] = [-1/3, 2]，y in [0, H] = [0, 1]
    底边：x < 0 对称面（来流引入段），x >= 0 无滑移绝热壁（平板）
    入口 x = X_IN：INLET（来流态）；出口 x = L：OUTLET（静压）；上边界 y = H：FARFIELD
    展向两面 SYMMETRY，nz = 2（(x,z) 三角剖分对 z 中面精确镜像对称，见 `_channel_mesh.py`）

网格：张量积坐标线（`_channel_mesh.build_prism_mesh_from_lines`），流向在前缘两侧几何加密，
壁面法向几何拉伸，棱柱沿壁面法向挤出（与真实边界层网格同一拓扑）。壁面距离用解析式
（平板上为 y，引入段为到前缘的距离），经生产入口 `apply_wall_distance_source` 施加
（omega 上界随之按最小壁距给定，换阶时按同一来源重查）。

边界类型按边界面中心逐面分派（平板与它前方的对称面在同一个平面上，按单元记录的网格
分组在角点单元上有歧义）。Order Continuation 换阶时按 `bc_overrides` 重建幽灵态提供者，
这里把重建入口指向同一个逐面提供者——它只依赖面拓扑，与阶数无关。

## 参照

湍流平板局部摩阻的 White 关联式 `cf = 0.455 / ln^2(0.06 Re_x)`（零压梯度、全湍流），
工程上与 SST 等两方程模型的结果吻合到几个百分点（TMR 2DZP 的 SST 解在 x = 0.97 处与它
相差约 6%），判据只在前缘转捩/欠解析段下游（默认 Re_x >= 1e6）核对。
"""

import numpy as np

RHO_INF = 1.225
U_INF = 30.0
P_INF = 101325.0
RE_PER_M = 5.0e6
MU = RHO_INF * U_INF / RE_PER_M
X_IN = -1.0 / 3.0
L_PLATE = 2.0
H_DOMAIN = 1.0


def geometric_lines(length: float, n: int, first: float) -> np.ndarray:
    """`[0, length]` 上 `n` 段几何级数分布的坐标线（从 0 端起第一段长 `first`）。

    比值 `r` 由 `first (r^n - 1) / (r - 1) = length` 二分求出；`first * n >= length` 时
    无需加密（返回均匀）。
    """
    if first * n >= length:
        return np.linspace(0.0, length, n + 1)
    lo, hi = 1.0 + 1e-12, 10.0
    for _ in range(200):
        r = 0.5 * (lo + hi)
        if first * (r ** n - 1.0) / (r - 1.0) > length:
            hi = r
        else:
            lo = r
    r = 0.5 * (lo + hi)
    seg = first * r ** np.arange(n)
    lines = np.concatenate([[0.0], np.cumsum(seg)])
    lines *= length / lines[-1]
    return lines


def plate_lines(nx_up: int, nx_plate: int, ny: int, dx_le: float, dy_wall: float):
    """流向（前缘两侧加密）与壁面法向（几何拉伸）坐标线 `(x_lines, y_lines)`。"""
    up = -geometric_lines(-X_IN, nx_up, dx_le)[::-1]          # [X_IN, 0]，靠 0 端加密
    plate = geometric_lines(L_PLATE, nx_plate, dx_le)          # [0, L]，靠 0 端加密
    x_lines = np.concatenate([up, plate[1:]])
    y_lines = geometric_lines(H_DOMAIN, ny, dy_wall)
    return x_lines, y_lines


def analytic_wall_distance(xyz: np.ndarray) -> np.ndarray:
    """平板上为 y，引入段（x < 0）为到前缘 (0, 0) 的距离。"""
    x, y = xyz[..., 0], xyz[..., 1]
    return np.ascontiguousarray(np.where(x >= 0.0, y, np.hypot(x, y)))


class AnalyticWallDistance:
    """解析壁距来源（`analytic_wall_distance`），与生产来源同样的 `kind` / `query` 接口。"""

    kind = "analytic"

    def query(self, points):
        return analytic_wall_distance(np.asarray(points))


def build_flat_plate_solver(order: int, *, nx_up: int = 8, nx_plate: int = 24, ny: int = 24,
                            dx_le: float = 2e-3, dy_wall: float = 2e-5, lz: float = 0.05,
                            turb_model: str = "SST", turbulence_intensity: float = 1e-3,
                            viscosity_ratio: float = 1.0, order_continuation: bool = False):
    """构造湍流平板隐式稳态（NK）求解器，均匀来流初场；返回 `(solver, meta)`。

    `order_continuation=True` 时 `solver.solve()` 走生产的逐阶爬坡（P0 -> ... -> `order`），
    否则直接在 `order` 阶上推进（`solver.step`）。
    """
    from autoflowcfd.core.fr_solver import FRSolver
    from autoflowcfd.core.fr_solver.turbulence.wall_distance import apply_wall_distance_source
    from autoflowcfd.core.time_integration import TimeIntegrationScheme
    from tests.validation._channel_mesh import build_ghost_provider_by_classifier, build_prism_mesh_from_lines

    x_lines, y_lines = plate_lines(nx_up, nx_plate, ny, dx_le, dy_wall)
    mesh = build_prism_mesh_from_lines(order, x_lines, y_lines, np.array([0.0, 0.5 * lz, lz]))
    q_free = [RHO_INF, U_INF, 0.0, 0.0, P_INF]
    bc = {
        "inlet": {"type": "INLET", "Q_inlet": q_free},
        "outlet": {"type": "OUTLET", "p_outlet": P_INF},
        "farfield": {"type": "FARFIELD", "Q_free": q_free},
        "symmetry_upstream": {"type": "SYMMETRY"},
        "plate": {"type": "WALL", "is_no_slip": True, "wall_velocity": [0.0, 0.0, 0.0]},
        "span": {"type": "SYMMETRY"},
    }
    tol = 1e-9

    def classify(c):
        if abs(c[2]) < tol or abs(c[2] - lz) < tol:
            return "span"
        if abs(c[1]) < tol:
            return "plate" if c[0] > 0.0 else "symmetry_upstream"
        if abs(c[1] - H_DOMAIN) < tol:
            return "farfield"
        if abs(c[0] - X_IN) < tol:
            return "inlet"
        if abs(c[0] - L_PLATE) < tol:
            return "outlet"
        raise ValueError(f"边界面中心 {c} 不在任何区域边界上")

    # 网格自带的分组只是占位（构造需要），真正的边界类型由逐面分类的提供者给出
    placeholder = {name: {"type": "SYMMETRY"} for name in
                   ("x_min", "x_max", "wall_bottom", "wall_top", "z_min", "z_max")}
    solver = FRSolver(mesh=mesh, order=order, turb_model_name=turb_model, n_vars=7,
                      time_scheme=TimeIntegrationScheme.NEWTON_KRYLOV, rho_inf=RHO_INF, vel_inf=U_INF,
                      p_inf=P_INF, mu_molecular=MU, bc_overrides=placeholder,
                      turbulence_intensity=turbulence_intensity, viscosity_ratio=viscosity_ratio)
    solver.order_continuation_enabled = order_continuation
    provider = build_ghost_provider_by_classifier(mesh, classify, bc)
    solver.boundary_ghost_provider = provider
    solver._build_boundary_ghost_provider = lambda _bc_overrides: provider
    apply_wall_distance_source(solver, AnalyticWallDistance())
    solver._reference_area = L_PLATE * lz
    meta = dict(order=order, n_cells=int(mesh.n_cells), nx_up=nx_up, nx_plate=nx_plate, ny=ny,
                dx_le=dx_le, dy_wall=dy_wall, lz=lz, y_lines=y_lines, x_lines=x_lines)
    return solver, meta


def white_cf(x: np.ndarray) -> np.ndarray:
    """White 湍流平板局部摩阻关联式（模块文档"参照"）。"""
    re_x = RE_PER_M * np.asarray(x, dtype=float)
    return 0.455 / np.log(0.06 * re_x) ** 2


def wall_cf(solver):
    """平板（x > 0）上逐壁面通量点的局部摩阻 `(x, cf)`，按 x 排序。

    用贴壁单元多项式在壁面通量点上的速度梯度：单元内物理梯度乘以生产代码里同一个面外插
    矩阵（`boundary_extrap_native[owner_face_op]`），各阶都是该单元多项式的精确壁面梯度
    （Blasius 算例的 `wall_shear_profile` 用前两层解点差分，P>=2 时会低估）。`tau_w =
    mu du/dy`（平板法向即 y）。
    """
    from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
    from autoflowcfd.core.fr_operators.gradients import compute_physical_gradient

    mesh, flat = solver.mesh, get_flat_face_geometry(solver.mesh, solver.ops)
    Q = np.asarray(solver.state.Q)
    grad_u = compute_physical_gradient(np.ascontiguousarray(Q[..., 1:2]), mesh, solver.ops)[:, :, 0, 1]
    xyz = np.asarray(mesh.sps_coords)
    xs, cfs = [], []
    for f in np.nonzero(flat.is_boundary & flat.owner_is_primary)[0]:
        c = flat.owner_cell[f]
        E = flat.boundary_extrap_native[flat.owner_face_op[f]]
        fp = E @ xyz[c]
        if np.abs(fp[:, 1]).max() > 1e-9 or fp[:, 0].min() <= 0.0:
            continue
        xs.append(fp[:, 0])
        cfs.append(solver.mu_molecular * (E @ grad_u[c]) / (0.5 * RHO_INF * U_INF ** 2))
    x, cf = np.concatenate(xs), np.concatenate(cfs)
    o = np.argsort(x)
    return x[o], cf[o]
