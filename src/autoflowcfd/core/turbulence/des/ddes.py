"""SST 基 DDES 的混合长度尺度（Gritskevich, Garbaruk, Schütze & Menter 2012, Flow Turbul. Combust. 88:431-449）。

    l_DDES = l_RANS - f_d max(0, l_RANS - l_LES)
    l_RANS = sqrt(k) / (beta* omega)，   l_LES = C_DES Delta，   C_DES = 0.78 F1 + 0.61 (1 - F1)
    f_d    = 1 - tanh[(C_d1 r_d)^C_d2]，  r_d = (nu_t + nu) / (kappa^2 d_w^2 sqrt(0.5 (S^2 + Omega^2)))
    C_d1 = 20，C_d2 = 3

高阶单元的分辨尺度：P 阶 FR/DG 单元在每个方向上有 P+1 个自由度，Delta = h_max / (P+1)（P0 即有限体积的 h_max）。
k 方程的破坏项 D_k = rho k^(3/2) / l_DDES，在 SST 源项求值里用同一次求值的 F1 与 nu_t 算出
（`DESLengthScale`，`core/turbulence/sst/source.py`）。

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

    def length_scale(self, xp, k, omega, nu_t, nu, d_w, grad_u, F1, h_max, order: int, beta_star: float,
                     h_wn=None):
        """解点上的 l_DDES。`h_max` 为逐单元最大边长 `(n_cells,)`；DDES 不用 `h_wn`。"""
        g = velocity_gradient_scale(xp, grad_u)
        f_d = 1.0 - xp.tanh((C_D1 * wall_ratio(xp, nu_t + nu, d_w, g)) ** C_D2)
        l_rans = rans_length(xp, k, omega, beta_star)
        l_les = des_constant(F1) * _cell_to_points(xp, resolution_scale(h_max, order), k.shape[1])
        return l_rans - f_d * xp.maximum(l_rans - l_les, 0.0)

    def bind(self, h_max, h_wn=None) -> "DESLengthScale":
        """绑定与湍流场同一索引空间的逐单元网格尺度（单机为全局、分布式为紧凑空间）。"""
        return DESLengthScale(self, h_max, h_wn)


class DESLengthScale:
    """SST 源项求值时调用的 DES 长度尺度：模型 + 逐单元网格尺度（最大边长、壁法向尺度）。

    阶数由解点数推出（`order_from_n_sps`），与场本身一致，换阶数后无需重建。
    """

    __slots__ = ("model", "h_max", "h_wn")

    def __init__(self, model: DDESModel, h_max, h_wn=None):
        if h_max is None:
            raise ValueError("DES 长度尺度需要逐单元最大边长 h_max")
        if getattr(model, "REQUIRES_WALL_NORMAL_SCALE", False) and h_wn is None:
            raise ValueError(f"{type(model).__name__} 需要逐单元壁法向尺度 h_wn")
        self.model, self.h_max, self.h_wn = model, h_max, h_wn

    def __call__(self, xp, k, omega, nu_t, nu, d_w, grad_u, F1, beta_star: float):
        from autoflowcfd.fr.native_padding import order_from_n_sps

        return self.model.length_scale(xp, k, omega, nu_t, nu, d_w, grad_u, F1, self.h_max,
                                       order_from_n_sps(k.shape[1]), beta_star, h_wn=self.h_wn)


def des_length_for(ddes_model, h_max, h_wn):
    """求解器持有的 DES 模型（可为 None）-> 源项用的长度尺度求值器（None 表示纯 RANS）。"""
    return None if ddes_model is None else ddes_model.bind(h_max, h_wn)
