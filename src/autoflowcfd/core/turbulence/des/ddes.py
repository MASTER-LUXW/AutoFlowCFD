"""AutoFlowCFD V2.0 - DDES 模型

从 `src/autoflowcfd/core/turbulence/des.py`(原 657 行)拆出(2026-09-24, 项目"单文件不超 500 行"规范)。**纯搬家, 逻辑未改**。
"""

import numpy as np

from typing import Optional

from loguru import logger


class DDESModel:
    """
    DDES 混合模型处理器。
    
    基于 Spalart-Allmaras 或 SST k-ω 模型的 DDES 实现：
    - 在边界层内：F_d ≈ 0，使用 RANS 长度尺度 l_RANS = d_w
    - 在分离区：F_d ≈ 1，使用 LES 长度尺度 l_LES = C_DES * Δ

    Attributes:
        c_des: DES 常数（通常取 0.65）
        c_w1: DDES 延迟参数（SST-DDES 取 20.0——Gritskevich et al. 2012
            为 SST 框架专门重新标定的值，不是 SA-DES97 原始的 8.0；用
            SA 的 8.0 会让屏蔽函数在 SST 框架下过早从 RANS 切到 LES，
            丧失原本应有的"延迟"保护，见 compute_shielding_function 文档）
        psi: 屏蔽函数值场
        l_eff: 有效长度尺度场
    """

    def __init__(self, c_des: float = 0.65, c_w1: float = 20.0):
        """
        初始化 DDES 模型。

        Args:
            c_des: DES 常数
            c_w1: DDES 延迟参数（控制屏蔽函数的敏感度）。默认 20.0 —— 第四次
                评审第三轮修正：此前默认 8.0 是 SA-DES97（Spalart 2006）的
                标定值，直接套用到 SST 框架会让屏蔽函数在 SST 模型下过早
                判定为 LES 区，达不到 DDES"延迟"应有的保护效果。Gritskevich
                et al. 2012（"Development of DDES and IDDES Formulations for
                the k-ω Shear Stress Transport Model"）明确指出：为了让
                SST-DDES 获得与 SA-DDES 相当的保护水平，C_d1 必须重新标定
                为 20（已用 OpenFOAM kOmegaSSTDDES 官方实现的 Cd1_ 默认值
                交叉核实）。
        """
        self.c_des = c_des
        self.c_w1 = c_w1
        self.psi = None  # 屏蔽函数场
        self.l_eff = None  # 有效长度尺度场
        
    def compute_grid_scale(self, cell_volumes: np.ndarray,
                          method: str = 'cube_root',
                          h_max: Optional[np.ndarray] = None) -> np.ndarray:
        """
        计算网格尺度 Δ。

        Args:
            cell_volumes: 单元体积数组
            method: 计算方法
                - 'cube_root': Δ = V^(1/3)（各向同性假设，扁平棱柱单元
                  会被严重低估 Δ）
                - 'max_edge': Δ = 逐单元最大边长（各向异性棱柱边界层
                  网格的标准做法，见下方说明），需要提供 `h_max`
            h_max: (n_cells,) 逐单元最大边长（`compute_h_max_and_h_wn`
                的第一个返回值），method='max_edge' 时必需

        Returns:
            delta: 网格尺度

        Note（2026-09-02 实现）：`max_edge` 此前长期声称"尚未实现，需要
            单元边长几何数据"——但 `compute_h_max_and_h_wn(mesh)`（本
            模块下方）早就实现了这个几何量的计算（IDDES 的 Δ_IDDES 公式
            本来就要用它），只是从未把它接到 DDES 基类这个入口。现在
            直接复用同一个函数的输出：调用方（`apply_to_sst_model`）
            按与 IDDES 完全一致的方式，在求解器初始化时算好 `h_max`
            （与流场状态无关，只依赖网格几何，算一次缓存即可）并传入。
            `'wurz'`（Chapman/Scotti 各向异性修正）仍未实现——没有已知
            调用方要求它，不在本次范围内。
        """
        if method == 'cube_root':
            return cell_volumes ** (1.0 / 3.0)
        elif method == 'max_edge':
            if h_max is None:
                raise ValueError(
                    "compute_grid_scale: method='max_edge' 需要提供 h_max"
                    "（compute_h_max_and_h_wn(mesh) 的第一个返回值）。"
                )
            return h_max
        else:
            raise NotImplementedError(
                f"compute_grid_scale: method='{method}' 尚未实现——只有 "
                f"'cube_root'/'max_edge' 已实现，请勿静默退化为其中之一。"
            )

    def compute_strain_rate_magnitude(self, grad_u: np.ndarray) -> np.ndarray:
        """
        计算应变率张量的模 |S|。

        Args:
            grad_u: 速度梯度张量，形状 (n_cells, n_sps, 3, 3)

        Returns:
            S_mag: 应变率模，形状 (n_cells, n_sps)
        """
        # S_ij = 0.5 * (∂u_i/∂x_j + ∂u_j/∂x_i)
        S_ij = 0.5 * (grad_u + np.transpose(grad_u, (0, 1, 3, 2)))

        # |S| = sqrt(2 * S_ij * S_ij)
        S_mag = np.sqrt(2.0 * np.einsum('nijm,nijm->ni', S_ij, S_ij))

        return S_mag

    def compute_vorticity_magnitude(self, grad_u: np.ndarray) -> np.ndarray:
        """
        计算涡量（旋转率）张量的模 |Ω|。

        第四次评审第三轮新增：Gritskevich et al. 2012 的 SST-DDES 屏蔽
        函数 r_d 公式用的是应变率与涡量的联合尺度
        sqrt(0.5*(|S|²+|Ω|²))，不是单独的 |S|——此前
        compute_shielding_function 只用了 |S|，在纯旋转（无应变）流动
        区域会系统性低估 r_d，见该方法文档的完整说明。

        Args:
            grad_u: 速度梯度张量，形状 (n_cells, n_sps, 3, 3)

        Returns:
            Omega_mag: 涡量模，形状 (n_cells, n_sps)，与 S_mag 同一约定
                （|Ω| = sqrt(2*Ω_ij*Ω_ij)）
        """
        # Ω_ij = 0.5 * (∂u_i/∂x_j - ∂u_j/∂x_i)
        Omega_ij = 0.5 * (grad_u - np.transpose(grad_u, (0, 1, 3, 2)))

        # |Ω| = sqrt(2 * Ω_ij * Ω_ij)
        Omega_mag = np.sqrt(2.0 * np.einsum('nijm,nijm->ni', Omega_ij, Omega_ij))

        return Omega_mag

    def compute_shielding_function(self, d_w: np.ndarray, nu_t: np.ndarray,
                                   omega: np.ndarray, nu: np.ndarray, kappa: float = 0.41,
                                   grad_u: Optional[np.ndarray] = None) -> np.ndarray:
        """
        计算 DDES 屏蔽函数 F_d。

        F_d = 1 - tanh[(C_d1 * r_d)^C_d2]，C_d2 = 3（标准值，SA/SST 通用）

        其中 r_d = (ν_t + ν) / (κ² * d_w² * sqrt(0.5*(|S|²+|Ω|²)))
        （Gritskevich et al. 2012, SST-DDES；已用 OpenFOAM kOmegaSSTDDES
        官方实现交叉核实：分母用应变率 |S| 与涡量 |Ω| 的联合尺度
        sqrt(0.5*(S²+Ω²))，不是单独的 |S|）。

        第四次评审第三轮修正：此前这里只用 |S|（纯应变），遗漏了 |Ω|
        （涡量）项——在纯旋转、无应变的流动区域（例如刚体式漩涡核心）
        会系统性低估分母、从而高估 r_d，让屏蔽函数在这类区域过早判定
        为 RANS（f_d 偏小），掩盖了本该被 LES 解析的分离涡结构。同时
        c_w1（对应文献里的 C_d1）默认值也从 SA-DES97 的 8.0 改为
        Gritskevich et al. 2012 为 SST 框架重新标定的 20.0（见 __init__
        文档），两处改动必须同时生效才能恢复文献声称的屏蔽保护水平。

        Args:
            d_w: 壁面距离
            nu_t: 湍流涡粘系数
            omega: 比耗散率
            nu: 分子运动粘度（此前遗漏此项会导致粘性底层 nu_t->0 时 r_d->0,
                f_d->1，恰好在最需要屏蔽为 RANS 的粘性底层误判为 LES 区）
            kappa: Von Karman 常数
            grad_u: 速度梯度张量（可选，用于精确计算 |S|/|Ω|）

        Returns:
            f_d: 屏蔽函数，范围 [0, 1]
                 - F_d ≈ 0: 边界层内（RANS 模式）
                 - F_d ≈ 1: 分离区（LES 模式）
        """
        # 防止除以零
        d_w = np.maximum(d_w, 1e-6)
        omega = np.maximum(omega, 1e-6)
        nu_t = np.maximum(nu_t, 1e-10)

        # 计算应变率+涡量联合尺度 sqrt(0.5*(|S|²+|Ω|²))
        if grad_u is not None:
            S_mag = self.compute_strain_rate_magnitude(grad_u)
            Omega_mag = self.compute_vorticity_magnitude(grad_u)
            S_Omega_mag = np.sqrt(0.5 * (S_mag**2 + Omega_mag**2))
        else:
            # 简化：没有速度梯度时用 omega 近似（量纲一致，仅作退化兜底）
            S_Omega_mag = omega.copy()

        S_Omega_mag = np.maximum(S_Omega_mag, 1e-6)

        # 计算 r_d（含分子粘度项，含应变+涡量联合尺度）
        r_d = (nu_t + nu) / (kappa**2 * d_w**2 * S_Omega_mag)

        # 限制 r_d 的范围
        r_d = np.minimum(r_d, 10.0)

        # 计算屏蔽函数
        f_d = 1.0 - np.tanh((self.c_w1 * r_d)**3)
        
        # 存储供后续使用
        self.psi = f_d
        
        return f_d

    def compute_effective_length_scale(self, k: np.ndarray, omega: np.ndarray,
                                      beta_star: float,
                                      delta: np.ndarray,
                                      f_d: np.ndarray,
                                      c_des: Optional[float] = None) -> np.ndarray:
        """
        计算 DDES 有效长度尺度。

        Args:
            k: 湍动能场，形状 (n_cells, n_sps)
            omega: 比耗散率场，形状 (n_cells, n_sps)
            beta_star: SST 模型常数（通常 0.09）
            delta: 网格尺度，形状 (n_cells,) 或 (n_cells, n_sps)
            f_d: 屏蔽函数，形状 (n_cells, n_sps)
            c_des: DES 常数（None 使用默认值）

        Returns:
            l_eff: 有效长度尺度，形状 (n_cells, n_sps)
        """
        if c_des is None:
            c_des = self.c_des

        # 确保 delta 与 k 维度一致
        if delta.ndim == 1:
            n_sps = k.shape[1]
            delta = np.tile(delta[:, np.newaxis], (1, n_sps))

        # l_RANS = sqrt(k)/(beta_star*omega)：SST 模型自身隐含的湍流长度尺度
        # （Gritskevich et al. 2012, SST-DDES），不是几何壁面距离——用 d_w
        # 顶替会在厚边界层/尾流等仍应保持 RANS 的区域使用与真实湍流尺度
        # 无关的耗散长度，破坏 SST-DES 自洽性。
        omega_safe = np.maximum(omega, 1e-10)
        k_safe = np.maximum(k, 0.0)
        l_rans = np.sqrt(k_safe) / (beta_star * omega_safe)
        l_les = c_des * delta

        # DDES 公式
        l_eff = l_rans - f_d * np.maximum(0.0, l_rans - l_les)

        # 确保非负
        l_eff = np.maximum(l_eff, 1e-10)

        self.l_eff = l_eff

        return l_eff

    def apply_to_sst_model(self, sst_model, d_w: np.ndarray,
                          cell_volumes: np.ndarray,
                          nu: np.ndarray,
                          grad_u: Optional[np.ndarray] = None,
                          h_max: Optional[np.ndarray] = None):
        """
        将 DDES 模型应用到 SST k-ω 模型。

        修改 SST 模型的耗散项，使用 l_eff 替代 d_w。

        Args:
            sst_model: SSTModelFR 实例
            d_w: 壁面距离，形状 (n_cells, n_sps)
            cell_volumes: 单元体积，形状 (n_cells,)——`h_max` 提供时只用于
                兜底（`h_max is None` 时退化为各向同性 `cube_root(V)`），
                否则不参与网格尺度计算
            nu: 分子运动粘度场 mu/rho，形状 (n_cells, n_sps)（屏蔽函数 r_d
                公式需要，见 compute_shielding_function）
            grad_u: 速度梯度张量，形状 (n_cells, n_sps, 3, 3)
            h_max: (n_cells,) 逐单元最大边长（2026-09-02 新增，见
                `compute_grid_scale` 文档"Note"一节）。生产路径（`fr_
                solver/turbulence.py`）总会提供它，改用各向异性感知的
                `max_edge` 网格尺度——本项目高度依赖棱柱边界层网格，
                `cube_root(V)` 会系统性低估这类扁平单元的 Δ。None 时
                （例如脱离 solver 直接构造 `DDESModel` 的场景，见本文件
                `__main__` 测试块）退化为 `cube_root`，不报错也不警告，
                因为这条路径本来就没有网格几何可用。
        """
        # 计算网格尺度：h_max 可用时用各向异性感知的 max_edge（标准做法，
        # 见 compute_grid_scale 文档），否则退化为各向同性 cube_root。
        if h_max is not None:
            delta = self.compute_grid_scale(cell_volumes, method='max_edge', h_max=h_max)
        else:
            delta = self.compute_grid_scale(cell_volumes)

        # 获取湍流变量
        k = sst_model.k_field
        omega = sst_model.omega_field
        nu_t = sst_model.get_turbulent_viscosity()

        # 计算屏蔽函数
        f_d = self.compute_shielding_function(d_w, nu_t, omega, nu, grad_u=grad_u)

        # 计算有效长度尺度，直接替换 SST k 方程耗散项里的长度尺度
        # （turbulence_sst.py::compute_source_terms 里 D_k=rho*k^1.5/l_eff），
        # 而不是用启发式系数缩放 beta_star——那样做既不是标准 DES 公式，
        # 也会因为原地修改 sst_model.beta_star 这个"常数"而在多步迭代下
        # 产生复合误差（每次都把上一步已经改过的值当 original 再改一次）。
        # beta_star 本身保持不变，供纯 RANS 场合复用。
        l_eff = self.compute_effective_length_scale(k, omega, sst_model.beta_star, delta, f_d)
        sst_model.des_length_scale = l_eff

        logger.debug(
            f"DDES applied: f_d range [{f_d.min():.4f}, {f_d.max():.4f}], "
            f"l_eff range [{l_eff.min():.4e}, {l_eff.max():.4e}]"
        )
