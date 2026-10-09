"""Blasius 层流平板边界层的解析解与算例常数（`_blasius_case.py` 的求解器搭建与各验证测试共用）。

来流参数、板长雷诺数到粘度的换算、Blasius 方程（f 的三阶导数 + f 乘二阶导数 / 2 = 0）的打靶数值解，以及由它
得到的局部摩阻系数、各种边界层厚度与法向速度剖面。
"""

import functools
import math
from typing import Dict

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


#: 默认板长雷诺数下的运动粘度（各解析函数的 `nu` 缺省时取它）
NU = nu_for(RE_L_DEFAULT)

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


@functools.lru_cache(maxsize=16)
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
    # 打靶结果按积分上限缓存（上限取整）：Blasius 解与位置无关，而入口 / 上边界的解析剖面
    # 作为幽灵态回调在**每次残差求值**里都会调用这里——不缓存时每次调用做一遍纯 Python
    # 打靶（二分 80 次 x 2 万步 RK4），隐式 NK 的每个 GMRES 迭代都要求值一次残差，
    # 1152 单元的 P1 第一步跑 10 分钟以上。eta <= 10 与此前逐位相同；eta > 10 时 f'
    # 早已饱和到 1，上限取整带来的差别在 1e-8 以下。
    grid, gx = _blasius_shoot(float(math.ceil(eta_max)))
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
