# -*- coding: utf-8 -*-
"""Blasius 平板层流边界层验证算例（构造与解析参考量）。

## 为什么需要这个算例

本项目此前**没有一个能真正收敛、且有精确解可定量核对的算例**：

  * `test_couette.py` 的稳态由**粘性扩散时标** `H^2/nu` 支配，比对流时标
    大三个数量级，显式格式需约 27 万步——该文件自己的文档就承认"无法在
    合理预算内定量收敛"，只验证了误差方向正确。
  * 三张真实 ANSA 网格（plate_demo / plate_demo_les / cube_demo）都是钝体
    正对来流，**本来就没有稳态解**（涡脱落），而且 plate_demo 走完一个绕板
    特征时间需约 2 万步（见 `core/fr_solver/pseudotime_budget.py`）。
  * `test_isentropic_vortex.py` 是无粘周期问题，验证空间精度阶数，不涉及
    壁面/粘性/收敛。

平板边界层补上的正是缺的那一块：

  1. **稳态由对流时标 `L/U` 决定**（信息只往下游传），所以收敛步数是
     `(L/h_x)/CFL` 量级而不是粘性时标那种天文数字；
  2. 有经典精确解：局部摩阻 `cf = 0.664 / sqrt(Re_x)`（Blasius，平板层流，
     零压梯度），速度剖面 `u/U = f'(eta)`，`eta = y*sqrt(U/(nu*x))`；
  3. 同时给出几个**精确**守恒判据：进出口质量流量必须相等、绝热壁热通量
     必须为零、上下壁摩阻必须对称。

## 算例参数与为什么这样取

    U = 30 m/s   L = 1.0 m   Re_L = 1e5（层流，Blasius 适用上限约 5e5）
    -> nu = U*L/Re_L = 3.0e-4 m^2/s，mu = rho*nu = 3.675e-4 Pa*s

`mu` 是空气的约 20 倍——这是**刻意**的：验证算例要的是命中目标雷诺数，
而不是复现空气的绝对粘度（标准做法）。若改用空气真实 mu，同样的 Re_L 需要
`L = 0.05 m` 的板与更细的近壁网格，收敛步数反而更多。

边界层厚度 `delta(x) ~ 5*sqrt(nu*x/U)`，在 x=L 处是 15.8 mm。通道半高取
H = 0.1 m（约 6.3 delta，外流不受上壁干扰），y 方向均匀 ny 层。

    ny = 64  ->  dy = 1.56 mm  ->  x=L 处 delta 内约 10 层
    ny = 32  ->  dy = 3.12 mm  ->  x=L 处 delta 内约  5 层

`cf` 只在 `delta(x)` 被至少 4 层解析的下游段核对（默认 x >= 0.2 L），
上游段本来就欠解析，把它一起平均会得出没有意义的数。

## 一条必须说明的边界条件取法

`_channel_mesh.build_channel_mesh_prism` 的 y=0 与 y=H 两个面在本算例里
**物理含义不同**：y=0 是平板（无滑移壁），y=H 是外场。把 y=H 取成
`SLIP_WALL`（法向不可穿透、切向自由）而不是无滑移壁，是因为 H 处的外流
应当自由滑移；取无滑移会在上壁再长一层边界层，把"零压梯度"这个 Blasius
前提破坏掉（通道两壁边界层增长会挤压核心、产生顺流压降）。

x=0 取 INLET（均匀来流），x=L 取 OUTLET（固定静压）；z 两面取 SYMMETRY
（nz=1，等效二维）。
"""

from typing import Dict, Tuple

import numpy as np

#: 算例参数（见模块文档"算例参数"一节）
RHO_INF = 1.225
U_INF = 30.0
P_INF = 101325.0
L_PLATE = 1.0

#: 默认板长雷诺数。取 2e4 而不是更"标准"的 1e5 是**成本**决定的，不是
#: 物理决定的（2e4 仍远低于转捩的 5e5，Blasius 完全适用）：
#:
#: 显式格式下一个流过时间需要的步数是
#:     steps = (L/U) / (CFL*dy/(u+c_pre)) ~ 2*(L/dy)/CFL
#: 而"delta(L) 内至少 8 层"要求 `dy <= delta(L)/8 = 0.61*sqrt(nu*L/U)`，
#: 于是
#:     L/dy >= 1.64*sqrt(Re_L)      ->   steps ~ 3.3*sqrt(Re_L)/CFL
#:
#: Re_L=1e5 时是每流过时间约 10,400 步（CFL 0.1），2e4 时约 4,600 步。
#: 精度判据完全不受影响（cf 的相对误差只取决于 delta 内的层数），所以
#: 这里选成本低的那个。要换回 1e5 只需给 `build_blasius_solver(re_l=1e5)`。
RE_L_DEFAULT = 1.0e4


def nu_for(re_l: float = RE_L_DEFAULT) -> float:
    """命中给定板长雷诺数所需的运动粘度。"""
    return U_INF * L_PLATE / float(re_l)


#: 模块级默认（向后兼容既有引用）
RE_L = RE_L_DEFAULT
NU = nu_for(RE_L_DEFAULT)
MU = RHO_INF * NU

#: Blasius 层流平板局部摩阻系数的系数：`cf = _CF_COEF / sqrt(Re_x)`。
#: 0.664 来自 `cf = 2*tau_w/(rho*U^2)` 与 `f''(0) = 0.332`
#: （Blasius 方程的标准数值解）：`cf = 2*0.332/sqrt(Re_x)`。
_CF_COEF = 0.664

#: 位移厚度与动量厚度（同样来自 Blasius 解）：
#:   delta* = 1.7208 * sqrt(nu x / U)
#:   theta  = 0.6641 * sqrt(nu x / U)
#: 99% 厚度常用 `delta99 = 4.91 * sqrt(nu x / U)`（有时记作 5.0）。
_DELTA_STAR_COEF = 1.7208
_THETA_COEF = 0.6641
_DELTA99_COEF = 4.91


def blasius_cf(x: np.ndarray, nu: float = None) -> np.ndarray:
    """局部摩阻系数 `cf(x) = 0.664/sqrt(Re_x)`；x<=0 处返回 nan。"""
    nu = NU if nu is None else float(nu)
    x = np.asarray(x, dtype=float)
    re_x = U_INF * x / nu
    out = np.full_like(x, np.nan)
    m = re_x > 0.0
    out[m] = _CF_COEF / np.sqrt(re_x[m])
    return out


def blasius_tau_wall(x: np.ndarray, nu: float = None) -> np.ndarray:
    """壁面剪应力 `tau_w = 0.5*rho*U^2*cf`。"""
    return 0.5 * RHO_INF * U_INF ** 2 * blasius_cf(x, nu)


def blasius_delta99(x: np.ndarray, nu: float = None) -> np.ndarray:
    """99% 边界层厚度。"""
    nu = NU if nu is None else float(nu)
    x = np.asarray(x, dtype=float)
    return _DELTA99_COEF * np.sqrt(np.maximum(x, 0.0) * nu / U_INF)


def blasius_thicknesses(x: float, nu: float = None) -> Dict[str, float]:
    """给定 x 处的三种厚度（delta99 / 位移厚度 / 动量厚度）。"""
    nu = NU if nu is None else float(nu)
    s = np.sqrt(max(x, 0.0) * nu / U_INF)
    return {
        "delta99": _DELTA99_COEF * s,
        "delta_star": _DELTA_STAR_COEF * s,
        "theta": _THETA_COEF * s,
    }


def _blasius_shoot(eta_max: float, n: int = 20000):
    """打靶积分 Blasius 方程，返回 `(grid, eta_grid)`。

    Blasius 方程 `f''' + 0.5 f f'' = 0`，边条件 `f(0)=f'(0)=0`、
    `f'(inf)=1`。打靶量是 `f''(0)`（已知解约 0.33206）；这里用二分把
    `f'(eta_max)` 打到 1，所以结果不依赖任何硬编码表。

    `grid[:, 0:3]` 依次是 `f`、`f'`、`f''`。

    **单独提出来**（2026-09-19）：原先这段积分内嵌在 `blasius_profile`
    里、只把 `f'` 返回出来。而无前缘奇点档的入口需要横向速度
    `v = 0.5 sqrt(nu U / x) (eta f' - f)`，它要 `f` 本身。

    （那时曾用 `v = 0` 近似并把它记成"不影响下游 cf"——**被测量否掉**：
    `v = 0` 与连续性方程不相容，入口面上被迫产生一个大的 v 修正，实测
    让 `le_offset=0.5` 的残差比含奇点那档还差 4.5 倍（2.39e6 vs
    5.31e5）、`|v|/U` 大 4 倍（0.265 vs 0.067）。所以那不是"可接受的
    近似"，是错的。）
    """
    h = eta_max / n

    def rhs(v):
        return np.array([v[1], v[2], -0.5 * v[0] * v[2]])

    def shoot(fpp0):
        y = np.array([0.0, 0.0, fpp0])           # f, f', f''
        grid = np.empty((n + 1, 3))
        grid[0] = y
        for i in range(n):
            k1 = rhs(y)
            k2 = rhs(y + 0.5 * h * k1)
            k3 = rhs(y + 0.5 * h * k2)
            k4 = rhs(y + h * k3)
            y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            grid[i + 1] = y
        return grid

    lo, hi = 0.1, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if shoot(mid)[-1, 1] < 1.0:
            lo = mid
        else:
            hi = mid
    return shoot(0.5 * (lo + hi)), np.linspace(0.0, eta_max, n + 1)


def blasius_f_and_fp(eta: np.ndarray):
    """返回 `(f(eta), f'(eta))`。

    `f'` 就是 `u/U`；`f` 只有横向速度 `v` 需要。
    """
    eta = np.asarray(eta, dtype=float)
    eta_max = max(float(eta.max()) if eta.size else 0.0, 10.0)
    grid, gx = _blasius_shoot(eta_max)
    return np.interp(eta, gx, grid[:, 0]), np.interp(eta, gx, grid[:, 1])


def blasius_profile(eta: np.ndarray) -> np.ndarray:
    """`u/U = f'(eta)`：现场积分 Blasius 方程（RK4 + 打靶），不查表。"""
    return blasius_f_and_fp(eta)[1]


def blasius_v_over_u(eta: np.ndarray, re_x: float) -> np.ndarray:
    """横向速度 `v/U_inf = (eta f' - f) / (2 sqrt(Re_x))`。

    由 `v = 0.5 sqrt(nu U / x) (eta f' - f)` 除以 `U`、再用
    `sqrt(nu/(U x)) = 1/sqrt(Re_x)` 化简得到。

    `eta -> inf` 时 `eta f' - f -> 1.7208`（正是 `delta*` 的系数），
    于是外缘 `v/U -> 0.8604/sqrt(Re_x)` —— 与教科书那个 `0.86/sqrt(Re_x)`
    一致。这条恰好能当实现自检，见
    `tests/validation/test_blasius.py::TestReferenceSolutionItself`。
    """
    f, fp = blasius_f_and_fp(eta)
    return (np.asarray(eta, dtype=float) * fp - f) / (2.0 * np.sqrt(re_x))


def blasius_fpp0() -> float:
    """打靶得到的 `f''(0)`（应为约 0.33206）——独立核对积分器本身。"""
    lo, hi = 0.1, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        eta = np.array([10.0])
        # 复用 blasius_profile 的打靶：这里只需要 f'(eta_max) 的符号
        val = _shoot_fp_at_inf(mid)
        if val < 1.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _shoot_fp_at_inf(fpp0: float, eta_max: float = 10.0, n: int = 20000) -> float:
    h = eta_max / n
    y = np.array([0.0, 0.0, fpp0])

    def rhs(v):
        return np.array([v[1], v[2], -0.5 * v[0] * v[2]])

    for _ in range(n):
        k1 = rhs(y)
        k2 = rhs(y + 0.5 * h * k1)
        k3 = rhs(y + 0.5 * h * k2)
        k4 = rhs(y + h * k3)
        y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return float(y[1])


def build_blasius_solver(
    order: int = 1,
    nx: int = 32,
    ny: int = 0,
    nz: int = 1,
    cfl: float = 0.10,
    re_l: float = RE_L_DEFAULT,
    cells_in_delta: float = 6.0,
    delta_margin: float = 6.0,
    turb_model: str = "NONE",
    lz_over_h: float = 0.25,
    le_offset: float = 0.0,
):
    """构造平板边界层求解器；返回 `(solver, meta)`。

    通道高度 `H` 与 `ny` **由分辨率要求反推**，不写死：

        dy = delta99(L) / cells_in_delta
        H  = delta_margin * delta99(L)      （外场不受上壁干扰）
        ny = round(H / dy) = delta_margin * cells_in_delta

    所以 `ny` 入参只在它与反推值不一致时用来**覆盖**（用于网格收敛研究：
    同一个 H 下加倍 ny）。`meta` 里会同时报反推值与实际用的值。

    Args:
        order: 多项式阶数
        nx: 流向单元数（每个矩形拆 2 个棱柱）
        ny: 壁面法向层数；传 0 表示用反推值
        nz: 展向层数（1 = 等效二维，两侧取 SYMMETRY）
        cfl: 固定 CFL
        re_l: 板长雷诺数
        cells_in_delta: x=L 处 delta99 内要求的层数
        delta_margin: 通道高度是 delta99(L) 的多少倍
        lz_over_h: 展向宽度与通道高度之比，`Lz = lz_over_h * H`。默认
            0.25（原先写死的值，所以默认行为不变）。

            **为什么做成参数（2026-09-18）**：本算例的三角形在 **x–z
            平面**（挤出沿壁面法向 y，理由见 `_channel_mesh.py`），所以
            `Lz` 决定三角形的**流向/展向长宽比**。展向 w 长到来流 4~5%
            这个开放问题的下一个待验证方向就是"流向强梯度通过坍缩三角形
            基耦合出 w"，而判别它需要扫这个长宽比——写死的常数扫不了。
        turb_model: 湍流模型名（层流验证用 "NONE"）
        le_offset: **虚拟原点偏移**，单位是板长 `L_PLATE` 的倍数。

            `0.0`（默认，保持既有行为）：无滑移壁面从 `x=0` 开始，而
            `x=0` 就是入口面 —— 于是**平板前缘落在入口面上**，那里
            Blasius 解有 `du/dy -> inf` 的可积奇点。

            `> 0`：域被当成虚拟前缘**下游** `le_offset * L_PLATE` 处的
            一段，入口携带该处的**解析 Blasius 剖面**，精确解在域内处处
            是 `x_v = x + le_offset*L_PLATE` 处的 Blasius 解。域内**不含
            前缘**，因此存在真正的稳态。

            ## 为什么这个参数是必须的（实测依据，2026-09-19）

            `le_offset=0` 那一档**不是**一个可用于稳态收敛/摩阻定量验证
            的算例。推进 400 步后按位置统计残差：

                x/L in [0.000,0.125)   占残差总平方和 99.5%
                其余 7 个 x 箱          各 0.1%
                y/H in [0.000,0.125)   占 100.0%

            最大残差点在 `x/L=0.0000, y/H=0.0059`（能量方程）——正是前缘
            角。后果是：残差在显式 CFL 0.03 下跑 2000 步仍单调上升到
            6.6e5、从未回落；P2 最终发散，且发散**时刻与 dt 完全无关**

                CFL 0.03   第 2834 步
                CFL 0.02   第 4251 步   = 2834 x 1.5
                CFL 0.01   第 8501 步   = 2834 x 3（事先预言 8502）

            这同时解释了项目记忆 `blasius_spanwise_w_open` 里那条
            "**不收敛**（均匀加密不降反升）"——奇点的固有行为。

            **两档都保留**：`le_offset=0` 是那条 P2 发散的最小复现
            （1728 单元、一分钟一轮），删掉它会失去这个诊断入口；
            `le_offset>0` 才是稳态收敛与 cf 定量验证该用的那一档。
    """
    import sys
    from pathlib import Path

    tests_dir = str(Path(__file__).resolve().parents[1])
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    from validation._channel_mesh import (
        build_channel_mesh_prism, build_face_exact_ghost_provider,
    )

    from autoflowcfd.core.fr_solver import FRSolver
    from autoflowcfd.core.time_integration import TimeIntegrationScheme

    nu = nu_for(re_l)
    mu = RHO_INF * nu
    d99_L = float(blasius_delta99(np.array([L_PLATE]), nu)[0])
    H = delta_margin * d99_L
    ny_needed = int(round(delta_margin * cells_in_delta))
    ny_use = int(ny) if ny else ny_needed
    Lz = float(lz_over_h) * H

    mesh = build_channel_mesh_prism(order, nx=nx, ny=ny_use, nz=nz,
                                    Lx=L_PLATE, H=H, Lz=Lz)
    bc_overrides = {
        # y=0 是平板本体（无滑移绝热壁）
        "wall_bottom": {"type": "WALL", "is_no_slip": True,
                        "wall_velocity": [0.0, 0.0, 0.0]},
        # y=H 是外场：滑移壁（法向不可穿透、切向自由）。取无滑移会在上壁
        # 再长一层边界层、挤压核心产生顺流压降，破坏 Blasius 的零压梯度
        # 前提（见模块文档）。底层 BC 名是 "WALL" + is_no_slip=False
        # （"SLIP_WALL" 只是 `build_boundary_ghost_provider` 的上层别名，
        # `bc_overrides` 走底层名，见该函数的 type_map）。
        "wall_top": {"type": "WALL", "is_no_slip": False},
        "z_min": {"type": "SYMMETRY"},
        "z_max": {"type": "SYMMETRY"},
        # INLET 需要显式给 `Q_inlet`（原始变量 rho,u,v,w,p）：`bc_overrides`
        # 走底层 BC 名，不经过 `VELOCITY_INLET -> ("INLET", {...})` 那层映射。
        # `le_offset > 0` 时这个常量只是占位：下面会把这一组换成逐通量点
        # 的解析 Blasius 剖面（见 `le_offset` 参数文档）。
        "x_min": {"type": "INLET",
                  "Q_inlet": [RHO_INF, U_INF, 0.0, 0.0, P_INF]},
        "x_max": {"type": "OUTLET", "p_outlet": P_INF},
    }
    solver = FRSolver(
        mesh=mesh, order=order, turb_model_name=turb_model, n_vars=5,
        time_scheme=TimeIntegrationScheme.SSP_RK3,
        rho_inf=RHO_INF, vel_inf=U_INF, p_inf=P_INF,
        mu_molecular=mu, bc_overrides=bc_overrides,
        adaptive_cfl=False, cfl_start=cfl, cfl_max=cfl, cfl_min=cfl,
    )
    solver.order_continuation_enabled = False
    # **必须**用逐面精确的幽灵态 provider：默认的 owner-cell 分组在
    # nz=1 时把每个单元同时算作贴着 z_min 与 z_max，会把约 18.5% 的
    # 壁面面误标成 x 端的 OUTLET，第 0 步就以 O(1e9) 量级发散（这是
    # `_channel_mesh.build_face_exact_ghost_provider` 文档记录的真实
    # 现象，`test_couette.py` 同样必须这么接）。
    solver.boundary_ghost_provider = build_face_exact_ghost_provider(
        mesh, L_PLATE, H, Lz, bc_overrides)

    x0 = float(le_offset) * L_PLATE
    if x0 > 0.0:
        # 无前缘奇点档：入口换成虚拟原点下游 x0 处的解析 Blasius 剖面。
        from validation._channel_mesh import wrap_inlet_with_profile

        re_x0 = U_INF * x0 / nu

        def _inlet_profile(positions):
            """`(n_fp,3) -> (n_fp,5)`：该批通量点处的 Blasius 入口状态。

            `u` 与 `v` **都**给精确值：

                u/U = f'(eta)
                v/U = (eta f' - f) / (2 sqrt(Re_x0))

            `v` 必须给：它与 `u` 剖面由连续性方程绑死。令 `v=0` 会在
            入口面上违反连续性、逼出一个大的局部修正 —— 实测那一版让
            本档的残差比含前缘奇点那档还**差 4.5 倍**（2.39e6 vs
            5.31e5）、`|v|/U` 大 4 倍（0.265 vs 0.067）。
            """
            y = np.asarray(positions)[:, 1]
            eta = y / np.sqrt(nu * x0 / U_INF)
            q = np.empty((len(y), 5))
            q[:, 0] = RHO_INF
            q[:, 1] = U_INF * blasius_profile(eta)
            q[:, 2] = U_INF * blasius_v_over_u(eta, re_x0)
            q[:, 3] = 0.0
            q[:, 4] = P_INF
            return q

        solver.boundary_ghost_provider = wrap_inlet_with_profile(
            solver.boundary_ghost_provider, solver, "x_min",
            bc_overrides, _inlet_profile)

    solver._reference_area = L_PLATE * Lz

    dy = H / ny_use
    meta = {
        "nu": nu, "mu": mu, "Re_L": float(re_l),
        "H": H, "Lz": Lz, "dx": L_PLATE / nx, "dy": dy,
        "nx": nx, "ny": ny_use, "ny_needed": ny_needed, "nz": nz,
        "delta99_at_L": d99_L,
        "cells_in_delta_at_L": d99_L / dy,
        "flow_through_time": L_PLATE / U_INF,
        "n_cells": int(mesh.n_cells),
        # 虚拟原点偏移（0 = 前缘在入口面上、含奇点，见 le_offset 文档）
        "le_offset": float(le_offset),
        "x_virtual_origin": x0,
    }
    return solver, meta


def wall_shear_profile(solver, mesh, *, wall_y: float = 0.0,
                       tol_rel: float = 1e-6):
    """从解里提取底壁（y=wall_y）上逐 x 的壁面剪应力与摩阻系数。

    ## 做法：用**胞内多项式在壁面处的梯度**，不是 `u_1/y_1`

    第一版写的是 `tau_w = mu * u_1 / y_1`（第一层解点的速度除以它到壁面的
    距离）。**那是错的**：它隐含假设 `u(0) = 0`，而 DG/FR 的无滑移是通过
    **通量弱施加**的——胞内多项式在壁面处的值一般不为零（这正是弱施加与
    强施加的区别）。实测后果：cf 比 Blasius 低 83%，一度被误判成求解器
    的近壁耗散问题。

    P1 下贴壁单元在壁面法向有 2 个解点，胞内是线性的，所以壁面梯度就是
    这两点的斜率：

        tau_w = mu * (u_2 - u_1) / (y_2 - y_1)

    这对 P1 是**精确**的（线性函数的斜率处处相同），不引入任何近似。
    order>=2 时同一段代码仍只用前两个解点，那时它是一阶差分近似、会低估
    壁面梯度——所以 `meta` 里报了实际用到的两个点位，判据要按分辨率给
    容差，不能给一个与阶数无关的固定容差。

    Returns:
        `(x, cf_num, tau_num, y1)`，前三个是一维数组（按 x 排序、同一 x 的
        多个解点已取平均），`y1` 是第一层解点到壁面的距离。
    """
    xyz = np.asarray(mesh.sps_coords)
    Q = np.asarray(solver.state.Q)
    y = xyz[:, :, 1]
    x = xyz[:, :, 0]
    u = Q[:, :, 1]

    y_min = y.min()
    scale = max(float(y.max() - y_min), 1e-300)
    first = np.abs(y - y_min) <= tol_rel * scale
    if not first.any():
        raise ValueError("找不到贴壁第一层解点")
    y1 = float(y[first][0] - wall_y)
    if y1 <= 0.0:
        raise ValueError(f"贴壁第一层解点到壁面的距离非正：{y1}")

    # 第二层解点：同一批贴壁单元里 y 次小的那一层
    cells = np.unique(np.nonzero(first)[0])
    y_sub = y[cells]
    uniq = np.unique(np.round(y_sub, 12))
    if uniq.size < 2:
        raise ValueError("贴壁单元在壁面法向只有一个解点，取不到梯度")
    y2 = float(uniq[1] - wall_y)
    second = np.abs(y - (uniq[1])) <= tol_rel * scale

    mu = float(solver.mu_molecular)
    # 逐单元取两层解点，按 x 配对
    x1, u1 = x[first], u[first]
    x2, u2 = x[second], u[second]
    o1, o2 = np.argsort(x1, kind="stable"), np.argsort(x2, kind="stable")
    x1, u1 = x1[o1], u1[o1]
    x2, u2 = x2[o2], u2[o2]
    if x1.shape != x2.shape or not np.allclose(x1, x2, atol=1e-9):
        raise ValueError("两层解点的 x 排布不一致，无法逐点配对")

    dudy = (u2 - u1) / (y2 - y1)
    tau = mu * dudy
    cf = tau / (0.5 * RHO_INF * U_INF ** 2)

    ux, inv = np.unique(np.round(x1, 12), return_inverse=True)
    cf_m = np.zeros_like(ux)
    tau_m = np.zeros_like(ux)
    cnt = np.zeros_like(ux)
    np.add.at(cf_m, inv, cf)
    np.add.at(tau_m, inv, tau)
    np.add.at(cnt, inv, 1.0)
    return ux, cf_m / cnt, tau_m / cnt, y1


def inlet_outlet_mass_flux(solver, mesh, Lx: float, H: float, Lz: float):
    """进出口质量流量（逐面积分），返回 `(m_in, m_out, rel_imbalance)`。

    稳态下两者必须相等——这是一条**精确**判据（与解析解无关），任何不等
    都是离散守恒或边界处理的问题。
    """
    fc = mesh.face_connectivity
    ops = solver.ops
    Q = np.asarray(solver.state.Q)
    ctr = np.asarray(fc.center)
    bidx = np.asarray(fc.get_boundary_face_indices())
    tol = 1e-8 * max(Lx, H, Lz)

    def flux_through(sel):
        total = 0.0
        for f in bidx[sel]:
            ffp = mesh.face_flux_points[f]
            if not ffp.owner_is_primary:
                continue
            owner = int(fc.owner_cell[f])
            oc = int(fc.owner_cube_face[f])
            if oc >= 6:
                E = ops.boundary_extrap_native_tet[oc - 6]
                sl = slice(0, E.shape[1])
            else:
                E = ops.boundary_extrap_prism[(ffp.owner_axis, ffp.owner_side)]
                sl = slice(None)
            rho = E @ Q[owner, sl, 0]
            vel = np.column_stack([E @ Q[owner, sl, c] for c in (1, 2, 3)])
            n = ffp.true_normal
            aw = ffp.true_area_weight
            total += float(np.sum(rho * np.sum(vel * n, axis=1) * aw))
        return total

    # `ctr` 是**全部**面的中心，而 `flux_through` 收到的掩码是用来索引
    # `bidx`（只含边界面）的——必须先按 bidx 取子集，否则形状不匹配
    # （第一版就是拿全部面的中心去索引边界面数组，直接 IndexError）。
    x_b = ctr[bidx, 0]
    m_in = -flux_through(np.abs(x_b - 0.0) < tol)      # 法向朝外，入流为负
    m_out = flux_through(np.abs(x_b - Lx) < tol)
    denom = max(abs(m_in), abs(m_out), 1e-300)
    return m_in, m_out, abs(m_out - m_in) / denom
