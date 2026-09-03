"""
AutoFlowCFD V2.0 - DDES/IDDES 混合湍流模型 (T-04)

本模块实现 Delayed Detached Eddy Simulation (DDES) 逻辑，
通过屏蔽函数在边界层内保持 RANS，在分离区切换为 LES。

核心功能:
1. DDES 屏蔽函数 F_d 计算
2. 有效长度尺度 l_eff 计算（RANS/LES 切换）
3. IDDES 改进型延迟分离涡模拟
4. 与 SST k-ω 模型的无缝集成
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


def compute_h_max_and_h_wn(mesh) -> tuple:
    """逐单元 IDDES 网格尺度所需的两个几何量：h_max（最大边长）与
    h_wn（近壁法向网格间距的估计）。

    复用 grid/validation/quality_metrics.py 中已验证的边长计算函数
    （tetrahedron_edge_lengths/prism_edge_lengths），不重新实现单元边长
    几何——这两个函数原本用于网格质量的长宽比计算，这里直接对同一批
    (nodes, cells) 数据复用。单元顺序遵循 HighOrderMesh 自身的约定
    （见 grid/high_order/high_order_mesh.py：cell_type = "prism" if
    i < n_prism_cells else "tet"）：前 n_prism_cells 个是棱柱，其余是
    四面体。

    h_wn（wall-normal spacing）取值约定：
    - 棱柱单元（本项目里唯一真正各向异性设计的单元类型，边界层挤出层）：
      prism_edge_lengths 返回的 9 条边中，第 6/7/8 列（索引 6:9）是竖直边
      （v_i -> w_i，边界层挤出方向，近似壁面法向），取这 3 条边的最小值
      作为该单元的法向间距估计——法向间距应该是"最薄"的那个方向。
    - 四面体单元（核心/尾流 LES 区域，本项目网格生成上设计为接近各向
      同性，不像棱柱层刻意拉伸）：没有明确的"法向"方向概念，取
      h_wn = h_max（各向同性单元下两者退化为同一个量，是标准简化，
      Shur et al. 2008 原文的结构化网格假设在此处不适用于本项目的
      非结构化四面体核心区，这是该假设不成立时的合理近似而非缺陷）。

    Args:
        mesh: HighOrderMesh 实例（需要 n_cells/n_prism_cells/_node_coords/
            _fixed_prism_conn/_fixed_tet_conn，均为该类已建立的内部
            几何属性，fr/face_flux_points*.py 已有跨模块访问这些属性的
            先例）

    Returns:
        (h_max, h_wn): 均为形状 (n_cells,) 的数组
    """
    from autoflowcfd.grid.validation.quality_metrics import (
        prism_edge_lengths, tetrahedron_edge_lengths,
    )

    n_cells = mesh.n_cells
    n_prism = mesh.n_prism_cells
    h_max = np.zeros(n_cells, dtype=np.float64)
    h_wn = np.zeros(n_cells, dtype=np.float64)

    if n_prism > 0:
        prism_edges = prism_edge_lengths(mesh._node_coords, mesh._fixed_prism_conn)
        h_max[:n_prism] = np.max(prism_edges, axis=1)
        h_wn[:n_prism] = np.min(prism_edges[:, 6:9], axis=1)

    n_tet = n_cells - n_prism
    if n_tet > 0:
        tet_edges = tetrahedron_edge_lengths(mesh._node_coords, mesh._fixed_tet_conn)
        tet_h_max = np.max(tet_edges, axis=1)
        h_max[n_prism:] = tet_h_max
        h_wn[n_prism:] = tet_h_max

    return h_max, h_wn


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


if __name__ == "__main__":
    # 测试代码
    from autoflowcfd.core.turbulence.sst import SSTModelFR
    
    # 创建测试数据
    n_cells = 100
    n_sps = 8
    
    d_w = np.random.rand(n_cells, n_sps) * 0.01
    cell_volumes = np.ones(n_cells) * 1e-6
    nu_t = np.random.rand(n_cells, n_sps) * 1e-4
    omega = np.random.rand(n_cells, n_sps) * 100
    
    # 创建 SST 模型
    sst = SSTModelFR(n_cells, n_sps)
    sst.k_field = np.random.rand(n_cells, n_sps) * 1e-4
    
    # 应用 DDES
    ddes = DDESModel()
    ddes.apply_to_sst_model(sst, d_w, cell_volumes, nu=np.full((n_cells, n_sps), 1.5e-5))
    
    print("DDES model test completed.")
