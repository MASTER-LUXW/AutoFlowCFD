"""亚格子模型：WALE（Nicoud & Ducros 1999, Flow Turbul. Combust. 62:183-200）与高阶单元的 LES 滤波宽度。

    nu_sgs = (C_w Delta)^2 (S^d_ij S^d_ij)^(3/2) / [ (S_ij S_ij)^(5/2) + (S^d_ij S^d_ij)^(5/4) ]
    S^d_ij = S_ik S_kj + Omega_ik Omega_kj - (1/3) delta_ij (S_mn S_mn - Omega_mn Omega_mn)，C_w = 0.325

WALE 在纯剪切与壁面处自然给出 nu_sgs -> 0（近壁按 y^3 衰减），不需要壁面阻尼函数。

滤波宽度：P 阶单元的分辨尺度 Delta = (V / N_p)^(1/3)，N_p 为该单元的真实自由度数（四面体 (P+1)(P+2)(P+3)/6、
棱柱 (P+1)^2(P+2)/2）——单元体积平均分给每个自由度的体积的立方根。

数组模块无关（`xp` 为 numpy 或 cupy），CPU、单 GPU、CPU 分布式、多 GPU 共用这一份实现。
"""

import numpy as np

C_WALE = 0.325


def les_filter_width(volumes, cell_is_prism, order: int, xp=np):
    """逐单元 LES 滤波宽度 `(n_cells,)`：Delta = (V / N_p)^(1/3)。"""
    from autoflowcfd.fr.native_padding import real_sps_per_cell

    n_real_prism, n_real_tet = real_sps_per_cell(int(order))
    n_dof = xp.where(xp.asarray(cell_is_prism), float(n_real_prism), float(n_real_tet))
    return (xp.abs(xp.asarray(volumes, dtype=xp.float64)) / n_dof) ** (1.0 / 3.0)


class WALEModel:
    """WALE 亚格子涡粘。"""

    def __init__(self, c_wale: float = C_WALE):
        self.c_wale = c_wale
        self.nu_t = None

    def compute_eddy_viscosity(self, grad_u, delta, xp=np):
        """解点上的亚格子运动涡粘 `(n_cells, n_sps)`。

        Args:
            grad_u: `(n_cells, n_sps, 3, 3)`，`[..., i, j] = du_i/dx_j`
            delta: 逐单元滤波宽度 `(n_cells,)`（`les_filter_width`）
            xp: 数组模块
        """
        g = grad_u
        gt = xp.swapaxes(g, -1, -2)
        S = 0.5 * (g + gt)
        W = 0.5 * (g - gt)
        S_sq = xp.sum(S * S, axis=(-2, -1))
        Sd = S @ S + W @ W
        trace = xp.trace(Sd, axis1=-2, axis2=-1)
        Sd = Sd - (trace / 3.0)[..., None, None] * xp.eye(3)
        Sd_sq = xp.sum(Sd * Sd, axis=(-2, -1))
        denom = S_sq ** 2.5 + Sd_sq ** 1.25
        positive = denom > 0.0
        ratio = xp.where(positive, Sd_sq ** 1.5 / xp.where(positive, denom, 1.0), 0.0)
        nu_t = (self.c_wale * delta[:, None]) ** 2 * ratio
        self.nu_t = nu_t
        return nu_t
