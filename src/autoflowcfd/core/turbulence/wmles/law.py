"""壁面律：Spalding（1961）全 y+ 范围的单一关系式（粘性底层、缓冲层、对数层连续）。

    y+ = u+ + e^(-kappa B) [ e^(kappa u+) - 1 - kappa u+ - (kappa u+)^2/2 - (kappa u+)^3/6 ]

给定匹配点切向速度 U、壁距 y、运动粘度 nu，Re_y = U y / nu = u+ y+(u+)。f(u+) = u+ y+(u+) - Re_y 在 u+ > 0 上
严格单调递增（y+ 与 dy+/du+ 均为正），对 u+ 做带区间保护的 Newton：每步保持根所在区间 [lo, hi]，Newton 步落到
区间外时取中点，二次收敛且不会发散。u_tau = U / u+，tau_w = rho u_tau^2。

数组模块无关（`xp` 为 numpy 或 cupy）。
"""

KAPPA = 0.41
B_LOG = 5.2

#: u+ 的区间上限：u+ y+(u+) 在 u+ = 200 处约 1e37，远超任何物理 Re_y（Re_y = 1e12 时 u+ 约 72），
#: 夹住它只是让 exp(kappa u+) 在迭代早期不溢出
U_PLUS_CAP = 200.0


def spalding_y_plus(xp, u_plus, kappa: float = KAPPA, B: float = B_LOG):
    """y+(u+) 与 dy+/du+。"""
    ku = kappa * u_plus
    c = xp.exp(-kappa * B)
    e = xp.exp(ku)
    y_plus = u_plus + c * (e - 1.0 - ku - 0.5 * ku * ku - ku * ku * ku / 6.0)
    dy = 1.0 + c * kappa * (e - 1.0 - ku - 0.5 * ku * ku)
    return y_plus, dy


def solve_u_plus(xp, re_y, kappa: float = KAPPA, B: float = B_LOG, tol: float = 1e-13, max_iter: int = 60):
    """解 u+ y+(u+) = Re_y（Re_y >= 0）。返回 (u+, 未收敛点数)。Re_y = 0 处 u+ = 0。

    对 g(u+) = ln(u+ y+(u+)) - ln(Re_y) 做 Newton（对数层里 u+ y+ 随 u+ 指数增长，取对数后近似线性，
    远离根时也是大步收敛），每步保持根所在区间、步子落到区间外时取中点。
    """
    re_y = xp.maximum(xp.asarray(re_y, dtype=xp.float64), 0.0)
    active = re_y > 0.0
    log_re = xp.log(xp.where(active, re_y, 1.0))
    # 区间：y+ >= u+ 给出 u+^2 <= u+ y+ = Re_y，故根在 (0, min(sqrt(Re_y), U_PLUS_CAP)]
    hi = xp.minimum(xp.sqrt(re_y), U_PLUS_CAP)
    lo = xp.zeros_like(re_y)
    # 初值：粘性底层（Re_y < 100）取 sqrt(Re_y)，否则取对数律估计 ln(Re_y)/kappa（都夹在区间内）
    u = xp.where(re_y < 100.0, hi, xp.minimum(hi, xp.log(xp.maximum(re_y, 1.0)) / kappa))
    u = xp.where(active, u, 1.0)
    for _ in range(max_iter):
        y_plus, dy = spalding_y_plus(xp, u, kappa, B)
        g = xp.log(u * y_plus) - log_re
        lo = xp.where(g < 0.0, u, lo)
        hi = xp.where(g > 0.0, u, hi)
        u_new = u - g / (1.0 / u + dy / y_plus)
        outside = (u_new <= lo) | (u_new >= hi)
        u_new = xp.where(outside, 0.5 * (lo + hi), u_new)
        done = (xp.abs(u_new - u) <= tol * xp.maximum(u_new, 1.0)) | ~active
        u = xp.where(active, u_new, 1.0)
        if bool(xp.all(done)):
            break
    u = xp.where(active, u, 0.0)
    y_plus, _ = spalding_y_plus(xp, u, kappa, B)
    residual = xp.abs(u * y_plus - re_y)
    n_bad = int(xp.count_nonzero(active & (residual > 1e-10 * xp.maximum(re_y, 1e-300))))
    return u, n_bad


def wall_shear_stress(xp, u_tangent, y, rho, nu, kappa: float = KAPPA, B: float = B_LOG):
    """匹配点切向速度矢量 `u_tangent (..., 3)`、壁距 y、密度 rho、运动粘度 nu -> (tau_w (..., 3), y+ (...))。

    tau_w 是壁面对流体的剪应力矢量（与切向速度反向），量值 rho u_tau^2。
    """
    U = xp.sqrt(xp.sum(u_tangent * u_tangent, axis=-1))
    u_plus, n_bad = solve_u_plus(xp, U * y / nu, kappa, B)
    if n_bad:
        raise RuntimeError(f"壁面律 Newton 在 {n_bad} 个点上未收敛")
    moving = u_plus > 0.0
    u_tau = xp.where(moving, U / xp.where(moving, u_plus, 1.0), 0.0)
    unit = xp.where(moving[..., None], u_tangent / xp.where(moving, U, 1.0)[..., None], 0.0)
    tau = -(rho * u_tau * u_tau)[..., None] * unit
    return tau, u_tau * y / nu
