"""SST 基 DDES 的混合长度尺度（Gritskevich, Garbaruk, Schütze & Menter 2012, Flow Turbul. Combust. 88:431-449）。

    l_DDES = l_RANS - f_d max(0, l_RANS - l_LES)
    l_RANS = sqrt(k) / (beta* omega)，   l_LES = C_DES Delta，   C_DES = 0.78 F1 + 0.61 (1 - F1)
    f_d    = 1 - tanh[(C_d1 r_d)^C_d2]，  r_d = (nu_t + nu) / (kappa^2 d_w^2 sqrt(0.5 (S^2 + Omega^2)))
    C_d1 = 20，C_d2 = 3

高阶单元的分辨尺度：P 阶 FR/DG 单元在每个方向上有 P+1 个自由度，Delta = h_max / (P+1)（P0 即有限体积的 h_max）。
k 方程的破坏项用 `des_length_scale`：D_k = rho k^(3/2) / l_DDES。

数组模块无关（`xp` 为 numpy 或 cupy），CPU、单 GPU、CPU 分布式、多 GPU 共用这一份实现。
"""

KAPPA = 0.41
C_DES_KOMEGA = 0.78
C_DES_KEPS = 0.61
C_D1 = 20.0
C_D2 = 3.0

#: 屏蔽函数里比值 r 的饱和截断：tanh((20 r)^3) 在 r = 10 时已严格等于 1，截断只为避免幂运算溢出
R_SATURATION = 10.0


def des_constant(F1):
    """C_DES = 0.78 F1 + 0.61 (1 - F1)（k-omega 区与 k-epsilon 区各自的标定值按 SST 混合函数过渡）。"""
    return C_DES_KOMEGA * F1 + C_DES_KEPS * (1.0 - F1)


def velocity_gradient_scale(xp, grad_u):
    """sqrt(0.5 (S^2 + Omega^2))，S = sqrt(2 S_ij S_ij)、Omega = sqrt(2 Omega_ij Omega_ij)；
    恒等于速度梯度张量的 Frobenius 范数。`grad_u[..., i, j] = du_i/dx_j`。"""
    return xp.sqrt(xp.sum(grad_u * grad_u, axis=(-2, -1)))


def rans_length(xp, k, omega, beta_star):
    """l_RANS = sqrt(k) / (beta* omega)（k 取非负部分；omega 由对数形式保证为正）。"""
    return xp.sqrt(xp.maximum(k, 0.0)) / (beta_star * omega)


def wall_ratio(xp, numerator, d_w, grad_scale):
    """r = numerator / (kappa^2 d_w^2 sqrt(0.5 (S^2 + Omega^2)))，饱和截断到 R_SATURATION。

    壁面解点（d_w = 0）与静止流体（梯度为零）处分母为零，r 取饱和值——屏蔽函数在那里取 RANS 分支。
    """
    denom = (KAPPA * KAPPA) * d_w * d_w * grad_scale
    positive = denom > 0.0
    r = xp.where(positive, numerator / xp.where(positive, denom, 1.0), R_SATURATION)
    return xp.minimum(r, R_SATURATION)


def resolution_scale(h, order: int):
    """P 阶单元的分辨尺度 h / (P+1)。"""
    return h / (int(order) + 1)


def _cell_to_points(xp, h, n_sps: int):
    """逐单元几何量 `(n_cells,)` 广播到解点 `(n_cells, n_sps)`。"""
    return xp.broadcast_to(h[:, None], (h.shape[0], n_sps))


class DDESModel:
    """SST-DDES（常数见模块文档）。"""

    def length_scale(self, xp, k, omega, nu_t, nu, d_w, grad_u, F1, h_max, order: int, beta_star: float):
        """解点上的 l_DDES。`h_max` 为逐单元最大边长 `(n_cells,)`。"""
        g = velocity_gradient_scale(xp, grad_u)
        f_d = 1.0 - xp.tanh((C_D1 * wall_ratio(xp, nu_t + nu, d_w, g)) ** C_D2)
        l_rans = rans_length(xp, k, omega, beta_star)
        l_les = des_constant(F1) * _cell_to_points(xp, resolution_scale(h_max, order), k.shape[1])
        return l_rans - f_d * xp.maximum(l_rans - l_les, 0.0)

    def apply(self, model, *, d_w, nu, grad_u, h_max, h_wn, xp):
        """按本步状态更新 `model.des_length_scale`（须在 SST 源项求值之后调用：用的 F1 与 nu_t 是那一步的结果）。

        Args:
            model: SST 模型（CPU 或 GPU 实现，提供 k_field / omega_field / nu_t / _last_F1 / beta_star）
            d_w: 解点壁距；nu: 解点分子运动粘度；grad_u: 解点速度梯度 (..., 3, 3)
            h_max / h_wn: 逐单元最大边长 / 壁法向尺度（DDES 不用 h_wn）
            xp: 数组模块
        阶数由解点数推出（`order_from_n_sps`），与场本身一致。
        """
        from autoflowcfd.fr.native_padding import order_from_n_sps

        order = order_from_n_sps(model.k_field.shape[1])
        F1 = getattr(model, "_last_F1", None)
        if F1 is None:
            raise RuntimeError("DES 长度尺度需要本步的 F1：须在 SST 源项求值之后调用")
        model.des_length_scale = self.length_scale(
            xp, model.k_field, model.omega_field, model.nu_t, nu, d_w, grad_u, F1, h_max, order, model.beta_star)
