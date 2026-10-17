"""SST 基 IDDES 的混合长度尺度（Shur, Spalart, Strelets & Travin 2008；SST 形式与常数见
Gritskevich, Garbaruk, Schütze & Menter 2012, Flow Turbul. Combust. 88:431-449）。

    l_IDDES = f~_d (1 + f_e) l_RANS + (1 - f~_d) l_LES，    l_LES = C_DES Delta
    Delta   = min{max[C_w d_w, C_w h_max, h_wn], h_max}，      C_w = 0.15
    f~_d    = max{1 - f_dt, f_B}，  f_dt = 1 - tanh[(C_dt1 r_dt)^C_dt2]，C_dt1 = 20、C_dt2 = 3
    f_B     = min{2 exp(-9 alpha^2), 1}，  alpha = 0.25 - d_w / h_max
    f_e     = max{f_e1 - 1, 0} f_e2，
    f_e1    = 2 exp(-11.09 alpha^2)（alpha >= 0），2 exp(-9 alpha^2)（alpha < 0）
    f_e2    = 1 - max{f_t, f_l}，f_t = tanh[(c_t^2 r_dt)^3]，f_l = tanh[(c_l^2 r_dl)^10]，c_t = 1.87、c_l = 5
    r_dt    = nu_t / (kappa^2 d_w^2 sqrt(0.5 (S^2 + Omega^2)))，r_dl 同式、分子为 nu

f~_d 的 DDES 分支（1 - f_dt）在附着边界层里保持 RANS、使 IDDES 在没有入口湍流内容时退化为 DDES；f_B 与 f_e
构成壁面模化 LES 分支。高阶单元的网格尺度（h_max、h_wn）一律取分辨尺度 h / (P+1)（见 `ddes.py`）。
"""

from .ddes import (
    DDESModel, KAPPA, R_SATURATION, _cell_to_points, des_constant, rans_length, resolution_scale,
    velocity_gradient_scale, wall_ratio,
)

C_W = 0.15
C_DT1 = 20.0
C_DT2 = 3.0
C_T = 1.87
C_L = 5.0


class IDDESModel(DDESModel):
    """SST-IDDES（常数见模块文档）。"""

    def length_scale(self, xp, k, omega, nu_t, nu, d_w, grad_u, F1, h_max, order: int, beta_star: float,
                     h_wn=None):
        n_sps = k.shape[1]
        hmax = _cell_to_points(xp, resolution_scale(h_max, order), n_sps)
        hwn = _cell_to_points(xp, resolution_scale(h_wn, order), n_sps)
        delta = xp.minimum(xp.maximum(xp.maximum(C_W * d_w, C_W * hmax), hwn), hmax)

        g = velocity_gradient_scale(xp, grad_u)
        r_dt = wall_ratio(xp, nu_t, d_w, g)
        r_dl = wall_ratio(xp, nu, d_w, g)
        f_dt = 1.0 - xp.tanh((C_DT1 * r_dt) ** C_DT2)

        alpha = 0.25 - d_w / hmax
        f_b = xp.minimum(2.0 * xp.exp(-9.0 * alpha * alpha), 1.0)
        f_d_tilde = xp.maximum(1.0 - f_dt, f_b)

        f_e1 = xp.where(alpha >= 0.0, 2.0 * xp.exp(-11.09 * alpha * alpha), 2.0 * xp.exp(-9.0 * alpha * alpha))
        f_e2 = 1.0 - xp.maximum(xp.tanh((C_T * C_T * r_dt) ** 3), xp.tanh((C_L * C_L * r_dl) ** 10))
        f_e = xp.maximum(f_e1 - 1.0, 0.0) * f_e2

        l_rans = rans_length(xp, k, omega, beta_star)
        return f_d_tilde * (1.0 + f_e) * l_rans + (1.0 - f_d_tilde) * des_constant(F1) * delta

    def apply(self, model, *, d_w, nu, grad_u, h_max, h_wn, xp):
        from autoflowcfd.fr.native_padding import order_from_n_sps

        order = order_from_n_sps(model.k_field.shape[1])
        F1 = getattr(model, "_last_F1", None)
        if F1 is None:
            raise RuntimeError("DES 长度尺度需要本步的 F1：须在 SST 源项求值之后调用")
        if h_wn is None:
            raise ValueError("IDDES 需要逐单元壁法向尺度 h_wn")
        model.des_length_scale = self.length_scale(
            xp, model.k_field, model.omega_field, model.nu_t, nu, d_w, grad_u, F1, h_max, order, model.beta_star,
            h_wn=h_wn)


__all__ = ["IDDESModel", "C_W", "C_DT1", "C_DT2", "C_T", "C_L", "KAPPA", "R_SATURATION"]
