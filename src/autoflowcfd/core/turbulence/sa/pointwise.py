"""AutoFlowCFD V2.0 - SA-neg 的逐点函数（唯一定义；numpy / cupy 共用，不修改模型状态）。

## 方程（与 SST 的 k 同一套对流形式，见 `turbulence/transport/convection.py`）

TMR 推荐的可压缩形式，扩散系数按 SA-neg 乘 `f_n`：

    rho D(nu_t~)/Dt = rho (P - D) + (1/sigma) [ div(rho (nu + nu_t~ f_n) grad nu_t~)
                                               + c_b2 rho |grad nu_t~|^2 - (nu + nu_t~ f_n) grad rho . grad nu_t~ ]

于是在本项目的残差约定（`R = -[S + conv + diff] / rho`）下：

    扩散系数  Gamma = (mu + rho nu_t~ f_n) / sigma                         `sa_diffusivity`
    逐点项    [c_b2 rho |grad nu_t~|^2 - (nu + nu_t~ f_n) grad rho . grad nu_t~] / sigma   `sa_gradient_source`
    源项      rho (P - D)                                                  `sa_source_terms`

最后一项（密度梯度项）是把 `div(rho (nu+nu_t~ f_n) grad)` 写成守恒形式时多出来的修正，使方程在
变密度下与不可压 SA 一致（TMR "compressible form"）。

## 两支

`nu_t~ >= 0`（SA-noft2）：

    chi = nu_t~/nu,  f_v1 = chi^3/(chi^3 + c_v1^3),  f_v2 = 1 - chi/(1 + chi f_v1)
    S_bar = nu_t~ f_v2 / (kappa^2 d^2)
    S_tilde = Omega + S_bar                                         若 S_bar >= -c_v2 Omega
            = Omega + Omega (c_v2^2 Omega + c_v3 S_bar) / ((c_v3 - 2 c_v2) Omega - S_bar)   否则
    r = min(nu_t~ / (S_tilde kappa^2 d^2), r_lim)，S_tilde <= 0 时取 r_lim
    g = r + c_w2 (r^6 - r),  f_w = g [(1 + c_w3^6)/(g^6 + c_w3^6)]^(1/6)
    P = c_b1 S_tilde nu_t~,  D = c_w1 f_w (nu_t~/d)^2,  nu_t = nu_t~ f_v1,  f_n = 1

`nu_t~ < 0`（SA-neg 负支）：

    P = c_b1 (1 - c_t3) Omega nu_t~,  D = -c_w1 (nu_t~/d)^2,  nu_t = 0,  f_n = (c_n1 + chi^3)/(c_n1 - chi^3)

`S_tilde` 的修正式在两段交界 `S_bar = -c_v2 Omega` 处值与一阶导数都连续（Allmaras 2012）。
"""

from .constants import (
    C_B1, C_B2, C_N1, C_T3, C_V1, C_V2, C_V3, C_W1, C_W2, C_W3, KAPPA, R_LIM, SIGMA,
)

#: 壁面距离的绝对下限（壁面解点 d = 0 时 `1/d^2` 的除零保护；真实解点的壁距远大于它）。
WALL_DISTANCE_FLOOR = 1e-12

_CV1_3 = C_V1 ** 3
_CW3_6 = C_W3 ** 6


def vorticity_magnitude(grad_vel, xp):
    """`|Omega| = sqrt(2 Omega_ij Omega_ij)`，`Omega_ij` 为速度梯度的反对称部分（即旋度的模）。"""
    g = grad_vel
    c1 = g[..., 2, 1] - g[..., 1, 2]
    c2 = g[..., 0, 2] - g[..., 2, 0]
    c3 = g[..., 1, 0] - g[..., 0, 1]
    return xp.sqrt(c1 * c1 + c2 * c2 + c3 * c3)


def _safe(x, xp):
    """除法分母：零替换为 1（被 `where` 屏蔽的分支不产生 inf/nan 警告）。"""
    return xp.where(x != 0.0, x, 1.0)


def sa_eddy_viscosity(nu_tilde, nu, xp):
    """`nu_t = nu_t~ f_v1`（负支为 0）。"""
    chi = nu_tilde / nu
    chi3 = chi * chi * chi
    return xp.where(nu_tilde > 0.0, nu_tilde * chi3 / (chi3 + _CV1_3), 0.0)


def sa_source_terms(nu_tilde, nu, d_wall, omega_mag, production_factor, xp):
    """单位质量的源项 `P - D` 与点隐式阻尼系数 `c >= 0`（显式更新 `phi += dt S/(1 + dt c)` 用）。

    `production_factor` 是求解器的产生项斜坡因子（与 SST 同一机制）。阻尼系数取源项中"随
    `nu_t~` 线性衰减"那部分的系数：正支 `D / nu_t~ = c_w1 f_w nu_t~/d^2`，负支
    `-(P - D)/nu_t~ = c_b1 (c_t3 - 1) Omega + c_w1 |nu_t~|/d^2`（两项都使负值回到零）。
    """
    d = xp.maximum(d_wall, WALL_DISTANCE_FLOOR)
    chi = nu_tilde / nu
    chi3 = chi * chi * chi
    f_v1 = chi3 / (chi3 + _CV1_3)
    f_v2 = 1.0 - chi / (1.0 + chi * f_v1)
    kd2 = (KAPPA * d) ** 2
    s_bar = nu_tilde * f_v2 / kd2
    denom = (C_V3 - 2.0 * C_V2) * omega_mag - s_bar
    s_mod = omega_mag + omega_mag * (C_V2 * C_V2 * omega_mag + C_V3 * s_bar) / _safe(denom, xp)
    s_tilde = xp.where(s_bar >= -C_V2 * omega_mag, omega_mag + s_bar, s_mod)
    pos_s = s_tilde > 0.0
    r = xp.where(pos_s, xp.minimum(nu_tilde / _safe(s_tilde * kd2, xp), R_LIM), R_LIM)
    g = r + C_W2 * (r ** 6 - r)
    f_w = g * ((1.0 + _CW3_6) / (g ** 6 + _CW3_6)) ** (1.0 / 6.0)
    ratio2 = (nu_tilde / d) ** 2
    pos = nu_tilde >= 0.0
    production = xp.where(pos, C_B1 * s_tilde * nu_tilde, C_B1 * (1.0 - C_T3) * omega_mag * nu_tilde)
    destruction = xp.where(pos, C_W1 * f_w * ratio2, -C_W1 * ratio2)
    damping = xp.where(pos, C_W1 * f_w * nu_tilde / (d * d),
                       C_B1 * (C_T3 - 1.0) * omega_mag + C_W1 * xp.abs(nu_tilde) / (d * d))
    return production_factor * production - destruction, damping


def sa_negative_diffusion_factor(nu_tilde, nu, xp):
    """`f_n`：正支为 1，负支 `(c_n1 + chi^3)/(c_n1 - chi^3)`。"""
    chi = nu_tilde / nu
    chi3 = chi * chi * chi
    return xp.where(nu_tilde >= 0.0, 1.0, (C_N1 + chi3) / (C_N1 - chi3))


def sa_diffusivity(nu_tilde, rho, mu, xp):
    """`Gamma = (mu + rho nu_t~ f_n) / sigma`（`f_n` 保证负支上它仍为正，Allmaras 2012）。"""
    f_n = sa_negative_diffusion_factor(nu_tilde, mu / rho, xp)
    return (mu + rho * nu_tilde * f_n) / SIGMA


def sa_gradient_source(nu_tilde, rho, mu, grad_nu_tilde, grad_rho, xp):
    """依赖梯度的逐点项 `[c_b2 rho |grad nu_t~|^2 - (nu + nu_t~ f_n) grad rho . grad nu_t~] / sigma`
    （带 rho，与源项同一量纲）。"""
    nu = mu / rho
    f_n = sa_negative_diffusion_factor(nu_tilde, nu, xp)
    g2 = xp.sum(grad_nu_tilde * grad_nu_tilde, axis=-1)
    gr = xp.sum(grad_rho * grad_nu_tilde, axis=-1)
    return (C_B2 * rho * g2 - (nu + nu_tilde * f_n) * gr) / SIGMA
