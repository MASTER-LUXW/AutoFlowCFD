"""AutoFlowCFD V2.0 - SA-neg 模型常数与来流值。

## 模型（Allmaras, Johnson & Spalart 2012, ICCFD7-1902；NASA Turbulence Modeling Resource "SA-neg"）

SA-neg 以标准 SA（不含转捩项 `f_t2`，即 TMR 的 "SA-noft2"）为基础，对 `nu_tilde < 0` 给出一支
连续的方程，使离散解越过零时方程仍良态——这正是它为高阶离散设计的原因：高阶多项式在边界层
外缘欠冲出负的 `nu_tilde` 时，标准 SA 的 `f_v1`、`S_tilde` 等在零附近失去意义，而 SA-neg 的负支
是线性耗散的，解自行回到非负（TMR："the negative branch ... is designed to drive nu_tilde back
to zero"）。

    nu_tilde >= 0：  P = c_b1 S_tilde nu_tilde,           D = c_w1 f_w (nu_tilde/d)^2,   nu_t = nu_tilde f_v1
    nu_tilde <  0：  P = c_b1 (1 - c_t3) Omega nu_tilde,   D = -c_w1 (nu_tilde/d)^2,       nu_t = 0

`S_tilde` 用 Allmaras 2012 的修正（避免 `S_tilde` 变负，`c_v2`、`c_v3`）；扩散系数在负支乘
`f_n = (c_n1 + chi^3)/(c_n1 - chi^3)`。可压缩形式取 TMR 推荐的写法（见 `pointwise.py`）。
"""

import math

#: 标准 SA 常数（TMR）
C_B1 = 0.1355
SIGMA = 2.0 / 3.0
C_B2 = 0.622
KAPPA = 0.41
C_W2 = 0.3
C_W3 = 2.0
C_V1 = 7.1
C_W1 = C_B1 / KAPPA ** 2 + (1.0 + C_B2) / SIGMA

#: SA-neg 的负支与 S_tilde 修正常数（Allmaras 2012）
C_T3 = 1.2
C_V2 = 0.7
C_V3 = 0.9
C_N1 = 16.0

#: f_w 的参数 r 的上限（标准 SA）
R_LIM = 10.0

#: TMR 推荐的来流 `chi = nu_tilde / nu`（"fully turbulent" 来流），对应粘性比约 0.21。
CHI_FREESTREAM_TMR = 3.0


def eddy_viscosity_ratio(chi: float) -> float:
    """来流（`nu_tilde >= 0`）上 `mu_t / mu = chi f_v1(chi)`。"""
    chi3 = chi ** 3
    return chi * chi3 / (chi3 + C_V1 ** 3)


def chi_for_viscosity_ratio(ratio: float) -> float:
    """反解 `chi f_v1(chi) = ratio`（左端在 `chi >= 0` 上严格单调递增），得到来流 `chi`。

    `ratio` 是求解器的来流涡粘比参数（`viscosity_ratio`，SST 用它定 omega_inf），两种模型的
    来流按同一个物理量给定。牛顿迭代从 `max(ratio, chi_0)` 起步，二十步以内收敛到机器精度；
    不收敛视为输入非法（非正、非有限）并报错。
    """
    if not (math.isfinite(ratio) and ratio > 0.0):
        raise ValueError(f"来流涡粘比必须为正有限值，收到 {ratio!r}")
    chi = max(float(ratio), 1.0)
    cv13 = C_V1 ** 3
    for _ in range(50):
        chi3 = chi ** 3
        f = chi * chi3 / (chi3 + cv13) - ratio
        df = (4.0 * chi3 * (chi3 + cv13) - 3.0 * chi ** 6) / (chi3 + cv13) ** 2
        step = f / df
        chi = max(chi - step, 0.5 * chi)
        if abs(step) <= 1e-14 * max(chi, 1.0):
            return chi
    raise RuntimeError(f"来流 chi 反解不收敛（ratio={ratio!r}）")
