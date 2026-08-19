"""动态 Smagorinsky 亚格子模型 (从 turbulence_sgs.py 拆分)。

从 turbulence_sgs.py 拆出来（该文件原有 406 行，超过 400 行硬性拆分
阈值）：`DynamicSmagorinskyModel` 是 `SmagorinskyModel` 的一个可选
变体，与文件里主要在用的 WALE/Smagorinsky 模型没有其它耦合，独立成
文件最清晰。纯代码搬移，不改变任何行为。
"""

import numpy as np

from .sgs import SmagorinskyModel


class DynamicSmagorinskyModel(SmagorinskyModel):
    """
    动态 Smagorinsky 模型。

    通过 Germano 恒等式动态计算 C_s，无需预设常数。
    """

    def __init__(self):
        super().__init__(c_s=0.1)  # 初始值，会被动态更新
        self.c_s_dynamic = None

    def compute_dynamic_cs(self, grad_u_coarse: np.ndarray,
                          grad_u_fine: np.ndarray,
                          delta_coarse: float,
                          delta_fine: float) -> float:
        """
        通过测试滤波器和 Germano 恒等式动态计算 C_s。

        基于 Lilly 的最小二乘法，最小化 Germano 恒等式的误差：

        L_ij = T_ij - τ̂_ij = 2C_s^2 (Δ_f^2 |S̃| S̃_ij - Δ_c^2 |Ŝ| Ŝ_îj)

        其中 T_ij 是测试滤波器尺度的应力，τ̂_ij 是粗网格应力的滤波。

        Args:
            grad_u_coarse: 粗网格速度梯度，形状 (3, 3)
            grad_u_fine: 细网格速度梯度，形状 (3, 3)
            delta_coarse: 粗网格尺度 Δ_c
            delta_fine: 细网格尺度 Δ_f

        Returns:
            c_s: 动态计算的 Smagorinsky 常数
        """
        # 计算应变率张量
        S_coarse = 0.5 * (grad_u_coarse + grad_u_coarse.T)
        S_fine = 0.5 * (grad_u_fine + grad_u_fine.T)

        # 应变率模
        S_mag_coarse = np.sqrt(2.0 * np.sum(S_coarse**2))
        S_mag_fine = np.sqrt(2.0 * np.sum(S_fine**2))

        # 构造 Leonard 应力 L_ij（简化：使用速度梯度的差异）
        # 实际应该对细网格应力进行滤波
        tau_fine = 2.0 * (delta_fine**2) * S_mag_fine * S_fine
        tau_coarse_filtered = 2.0 * (delta_coarse**2) * S_mag_coarse * S_coarse

        # 假设滤波操作近似为恒等（单点动态模型）
        L_ij = tau_fine - tau_coarse_filtered

        # 构造 M_ij = Δ_f^2 |S̃| S̃_ij - Δ_c^2 |Ŝ| Ŝ_îj
        M_ij = (delta_fine**2) * S_mag_fine * S_fine - (delta_coarse**2) * S_mag_coarse * S_coarse

        # Lilly 的最小二乘解：C_s^2 = <L_ij M_ij> / <M_ij M_ij>
        numerator = np.sum(L_ij * M_ij)
        denominator = np.sum(M_ij * M_ij)

        if denominator > 1e-10:
            c_s_squared = numerator / denominator

            # 限制 C_s 的范围以避免数值不稳定
            # 典型范围：0.0 - 0.2
            c_s_squared = max(0.0, min(c_s_squared, 0.04))  # 0.2^2 = 0.04

            c_s = np.sqrt(c_s_squared)
        else:
            # 分母太小，使用默认值
            c_s = self.c_s

        return c_s
