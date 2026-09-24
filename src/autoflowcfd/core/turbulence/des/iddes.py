"""AutoFlowCFD V2.0 - IDDES 模型(Shur et al. 2008)

从 `src/autoflowcfd/core/turbulence/des.py`(原 657 行)拆出(2026-09-24, 项目"单文件不超 500 行"规范)。**纯搬家, 逻辑未改**。
"""

import numpy as np

from typing import Optional

from loguru import logger
from .ddes import DDESModel


class IDDESModel(DDESModel):
    """
    IDDES (Improved Delayed Detached Eddy Simulation) 模型，按
    Shur, Spalart, Strelets & Travin (2008)《A hybrid RANS-LES approach
    with delayed-DES and wall-modelled LES capabilities》(Int. J. Heat
    Fluid Flow 29(6):1638-1649) 的思路、并采用 Gritskevich, Garbaruk,
    Schütze & Menter (2012)《Development of DDES and IDDES Formulations
    for the k-ω Shear Stress Transport Model》(Flow Turbul. Combust.
    88(3):431-449) 给出的 SST k-ω 版本常数重新实现。

    取代此前的死代码版本（自造的 `1-exp(-y_plus/10)` 屏蔽函数修正，不
    对应任何标准文献公式，见 git 历史该类此前的文档字符串）。

    本次实现（第 4 次专家评审第 4 轮修复）的验证置信度说明——按项目
    "no simplification, verify facts" 惯例明确标注，供后续复核：
    - **高置信度**（多个独立信源交叉确认一致，或可从已验证的 DDESModel
      推出）：alpha = 0.25 - d_w/h_max 混合坐标定义；f_B 混合权重公式
      min(2*exp(-9*alpha^2), 1)；f_e1 的 alpha>=0/<0 分段公式；
      l_IDDES = f_B*(1+f_e)*l_RANS + (1-f_B)*l_LES 混合公式结构；
      C_DES=0.78（SST kw 分支标准值，注意与 DDESModel 基类默认的 0.65——
      那是 SA 模型标定值——不同，两者不可混用）；C_w=0.15。
    - **已独立核实**（2026-09-02，此前是"中等置信度：本会话仅凭训练
      记忆写出，多次网络检索均未能独立复核到原始文献 Table 1 数值"）：
      f_e2 公式里的经验常数 c_t=1.87、c_l=5.0，已对照 OpenFOAM 生产级
      `kOmegaSSTIDDES` 实现（`Ct_=1.87`/`Cl_=5`，该实现本身就是
      Gritskevich et al. 2012 SST-IDDES 公式的产业界广泛使用的独立
      实现）逐值核对一致，不再是训练记忆孤证。核实来源：
      github.com/fertinaz/kOmegaSSTIDDES（OpenFOAM 官方
      `kOmegaSSTIDDES` 的社区移植版，常数定义在
      `kOmegaSSTIDDES.C::Ct_`/`Cl_` 的
      `dimensioned<scalar>::lookupOrAddToDict` 默认值参数）。附带
      验证：本类 `c_w1=20.0` 与该实现的 `Cdt1_=20` 也一致（同一常数，
      DDES 屏蔽函数延迟参数，本来就已经是高置信度项，这里是额外
      交叉确认）。
    - **本项目场景下的必要近似**（非文献规定做法）：Shur et al. 2008
      的网格尺度 Δ 公式假设结构化网格、可直接给出"壁面法向网格间距"
      h_wn；本项目是非结构化棱柱+四面体混合网格，h_wn 通过
      compute_h_max_and_h_wn() 从连接关系反推（棱柱竖直边最小值/
      四面体退化为 h_max），是合理但非文献规定的替代方案，见该函数
      文档的完整说明。
    - C_DES 本身在标准 SST-DES 框架下应该按 SST 的 F1 混合函数在 kw
      分支值(0.78)和 k-epsilon 分支值(0.61)间插值——本实现固定用
      kw 分支值 0.78，不做 F1 混合，这与本文件 DDESModel 基类既有的
      简化处理方式一致（该基类也是固定单一 C_des，从未做 F1 混合），
      是刻意保持两个模型内部一致性的选择，不是遗漏。

    Attributes（新增于 DDESModel 基类的 c_des/c_w1/psi/l_eff）:
        c_w: IDDES 网格尺度公式里的常数，标准值 0.15。
        c_t: f_e2 中 f_t 分量的常数（已独立核实，见上）。
        c_l: f_e2 中 f_l 分量的常数（已独立核实，见上）。
    """

    def __init__(self, c_des: float = 0.78, c_w1: float = 20.0,
                 c_w: float = 0.15, c_t: float = 1.87, c_l: float = 5.0):
        """
        初始化 IDDES 模型。

        Args:
            c_des: IDDES 的 C_DES 常数，SST kw 分支标准值 0.78（不是
                DDESModel 基类默认的 0.65——那是 SA 模型的标定值，见类
                文档字符串）。
            c_w1: 共享 DDESModel 基类的延迟参数，默认沿用同一 SST 重新
                标定值 20.0；本类实际未调用基类的 compute_shielding_function
                （IDDES 的 f_B/f_e 是独立于 DDES f_d 的公式，见类文档），
                保留此参数只是为了维持与基类构造签名一致，不在本类逻辑
                中使用。
            c_w: 网格尺度 Δ 公式中的常数，标准值 0.15。
            c_t: f_e2 中 f_t 分量的常数（中等置信度，见类文档）。
            c_l: f_e2 中 f_l 分量的常数（中等置信度，见类文档）。
        """
        super().__init__(c_des, c_w1)
        self.c_w = c_w
        self.c_t = c_t
        self.c_l = c_l

    def compute_grid_scale_iddes(self, d_w: np.ndarray, h_max: np.ndarray,
                                  h_wn: np.ndarray) -> np.ndarray:
        """IDDES 专用网格尺度 Δ（Gritskevich et al. 2012 式对应
        Shur et al. 2008 原文的 Δ 定义）：

            Δ = min( max(C_w*d_w, C_w*h_max, h_wn), h_max )

        区别于 DDESModel 基类的 compute_grid_scale（纯 cube_root(V)
        各向同性尺度，公式结构完全不同）——故意不复用/不修改基类方法，
        避免影响已验证正确的 DDESModel 行为。

        Args:
            d_w: 壁面距离
            h_max: 逐单元最大边长（见 compute_h_max_and_h_wn），需已广播
                到与 d_w 相同形状
            h_wn: 逐单元近壁法向间距估计（见 compute_h_max_and_h_wn），
                需已广播到与 d_w 相同形状

        Returns:
            delta: IDDES 网格尺度，形状与 d_w 相同
        """
        inner = np.maximum(np.maximum(self.c_w * d_w, self.c_w * h_max), h_wn)
        return np.minimum(inner, h_max)

    def compute_alpha(self, d_w: np.ndarray, h_max: np.ndarray) -> np.ndarray:
        """alpha = 0.25 - d_w/h_max（Gritskevich et al. 2012 混合坐标，
        决定 f_B/f_e1 的近壁-远壁位置）。"""
        return 0.25 - d_w / np.maximum(h_max, 1e-12)

    def compute_f_b(self, alpha: np.ndarray) -> np.ndarray:
        """f_B = min(2*exp(-9*alpha^2), 1)：RANS(f_B->1 近壁)/
        LES(f_B->0 远离壁面) 混合权重。"""
        return np.minimum(2.0 * np.exp(-9.0 * alpha**2), 1.0)

    def compute_f_e1(self, alpha: np.ndarray) -> np.ndarray:
        """f_e1 分段公式（Gritskevich et al. 2012）：
        alpha>=0 时 2*exp(-11.09*alpha^2)，alpha<0 时 2*exp(-9*alpha^2)。
        """
        return np.where(
            alpha >= 0.0,
            2.0 * np.exp(-11.09 * alpha**2),
            2.0 * np.exp(-9.0 * alpha**2),
        )

    def compute_f_e2(self, nu_t: np.ndarray, nu: np.ndarray, d_w: np.ndarray,
                     S_Omega_mag: np.ndarray, kappa: float = 0.41) -> np.ndarray:
        """f_e2 = 1 - max(f_t, f_l)（elevating function 的第二因子，
        防止边界层内网格局部加密时被误判进入 LES 分支——中等置信度
        常数 c_t/c_l，见类文档）。

        r_dt = nu_t / (kappa^2 * d_w^2 * S_Omega_mag)（纯涡粘尺度，
            区别于 DDESModel.compute_shielding_function 里 r_d 用的
            nu_t+nu 联合尺度）
        r_dl = nu   / (kappa^2 * d_w^2 * S_Omega_mag)（纯分子粘度尺度）
        f_t = tanh[(c_t^2 * r_dt)^3]
        f_l = tanh[(c_l^2 * r_dl)^10]
        """
        d_w_safe = np.maximum(d_w, 1e-6)
        S_Omega_safe = np.maximum(S_Omega_mag, 1e-6)
        denom = kappa**2 * d_w_safe**2 * S_Omega_safe
        r_dt = nu_t / denom
        r_dl = nu / denom
        f_t = np.tanh((self.c_t**2 * r_dt) ** 3)
        f_l = np.tanh((self.c_l**2 * r_dl) ** 10)
        return 1.0 - np.maximum(f_t, f_l)

    def compute_effective_length_scale_iddes(self, k: np.ndarray, omega: np.ndarray,
                                              beta_star: float, delta_iddes: np.ndarray,
                                              f_b: np.ndarray, f_e: np.ndarray) -> np.ndarray:
        """l_IDDES = f_B*(1+f_e)*l_RANS + (1-f_B)*l_LES（Gritskevich et
        al. 2012 凸组合混合，注意与 DDESModel 基类的 min 型公式
        l_eff = l_rans - f_d*max(0,l_rans-l_les) 结构不同，因此不复用/
        不重写基类的 compute_effective_length_scale）。

        l_RANS = sqrt(k)/(beta_star*omega)：与 DDES 相同定义，SST 模型
            自身隐含的湍流长度尺度。
        l_LES = C_DES * Δ_IDDES：Δ_IDDES 是 compute_grid_scale_iddes 的
            结果，不是 DDES 用的 cube_root(V)。
        """
        omega_safe = np.maximum(omega, 1e-10)
        k_safe = np.maximum(k, 0.0)
        l_rans = np.sqrt(k_safe) / (beta_star * omega_safe)
        l_les = self.c_des * delta_iddes

        l_iddes = f_b * (1.0 + f_e) * l_rans + (1.0 - f_b) * l_les
        l_iddes = np.maximum(l_iddes, 1e-10)

        self.l_eff = l_iddes
        return l_iddes

    def apply_to_sst_model_iddes(self, sst_model, d_w: np.ndarray,
                                 h_max: np.ndarray, h_wn: np.ndarray,
                                 nu: np.ndarray,
                                 grad_u: Optional[np.ndarray] = None) -> None:
        """把 IDDES 应用到 SST k-ω 模型（对应 DDESModel 基类的
        apply_to_sst_model，但 IDDES 的 Δ/f_B/f_e/l_IDDES 公式结构与
        DDES 完全不同，见类文档，因此单独实现而非复用/重写基类方法）。

        Args:
            sst_model: SSTModelFR 实例
            d_w: 壁面距离，形状 (n_cells, n_sps)
            h_max, h_wn: 逐单元几何量（见 compute_h_max_and_h_wn），形状
                (n_cells,)，内部广播到 (n_cells, n_sps) 后与 d_w 运算
            nu: 分子运动粘度场 mu/rho，形状 (n_cells, n_sps)
            grad_u: 速度梯度张量，形状 (n_cells, n_sps, 3, 3)，用于
                |S|/|Ω| 联合尺度（与 DDESModel.compute_shielding_function
                同一套公式，None 时退化用 omega 近似）
        """
        n_sps = sst_model.k_field.shape[1]
        h_max_b = np.tile(h_max[:, np.newaxis], (1, n_sps))
        h_wn_b = np.tile(h_wn[:, np.newaxis], (1, n_sps))

        k = sst_model.k_field
        omega = sst_model.omega_field
        nu_t = sst_model.get_turbulent_viscosity()

        if grad_u is not None:
            S_mag = self.compute_strain_rate_magnitude(grad_u)
            Omega_mag = self.compute_vorticity_magnitude(grad_u)
            S_Omega_mag = np.sqrt(0.5 * (S_mag**2 + Omega_mag**2))
        else:
            S_Omega_mag = omega.copy()
        S_Omega_mag = np.maximum(S_Omega_mag, 1e-6)

        alpha = self.compute_alpha(d_w, h_max_b)
        f_b = self.compute_f_b(alpha)
        f_e1 = self.compute_f_e1(alpha)
        f_e2 = self.compute_f_e2(nu_t, nu, d_w, S_Omega_mag)
        f_e = np.maximum(f_e1 - 1.0, 0.0) * f_e2

        delta_iddes = self.compute_grid_scale_iddes(d_w, h_max_b, h_wn_b)

        l_iddes = self.compute_effective_length_scale_iddes(
            k, omega, sst_model.beta_star, delta_iddes, f_b, f_e
        )
        sst_model.des_length_scale = l_iddes

        logger.debug(
            f"IDDES applied: f_B range [{f_b.min():.4f}, {f_b.max():.4f}], "
            f"f_e range [{f_e.min():.4f}, {f_e.max():.4f}], "
            f"l_iddes range [{l_iddes.min():.4e}, {l_iddes.max():.4e}]"
        )
