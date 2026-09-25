"""AutoFlowCFD V2.0 - omega 壁面边界处理（Dirichlet 掩码、目标值、松弛）。

从 `core/turbulence/transport.py` 拆出（2026-09-24）。纯搬家，逻辑未改。

这一层与对流/扩散残差是**并列**的第三件事：SST 的 omega 在壁面上有解析
值（Wilcox `omega_wall = 60*nu/(beta1*d1^2)`），它既要进面外插的 ghost
规则（`_compute_wall_dirichlet_face_mask`），也要在每步之后单独施加一次
松弛（`enforce_omega_wall_relaxation`）。
"""

import os
import numpy as np
from typing import Tuple


from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry


def _compute_wall_dirichlet_face_mask(solver) -> np.ndarray:
    """算出哪些面是**真实无滑移**WALL 边界面，供 k 场的 Dirichlet-zero
    ghost 及 omega 壁面解析式（Wilcox omega wall function）使用（见
    extrapolate_scalar_to_faces_kernel 文档）。

    数据来源：`solver.boundary_ghost_provider`——真实求解路径下是
    `boundary.fr_ghost_state.BoundaryGhostStateProvider`，持有
    `group_code`（每个面所属边界组的整数编码，-1 表示内部面/未匹配）和
    `code_to_config`（编码 -> {'type': 'WALL', 'is_no_slip': bool,...}）。用
    `code_to_config` 里显式标记为 WALL 的编码集合对 `group_code` 做一次
    向量化匹配（`np.isin`），成本是对 187 万面级别网格的一次数组比较，
    不是逐面 Python 循环。

    真实 bug 修复（2026-09-12，cube_demo 791,492 单元真实网格 P1 阶段
    omega 独立于历史、确定性地在特定单元收敛到同一个数值~984312 的排查
    发现，完整推导见 `enforce_omega_wall_relaxation` 文档）：此前这里把
    `cfg.get("type")=="WALL"` 的编码**不加区分**全部当作需要 Wilcox 近壁
    omega 解析式（`omega_wall=60*nu/(beta1*d1^2)`，专为*真实粘性无滑移*
    边界层设计）处理的壁面——但本项目的 WALL 类型边界组同时覆盖两种物理
    上完全不同的情形（见 `boundary/fr_ghost_state.py::build_wall_ghost_
    state` 的 `is_no_slip` 参数）：`is_no_slip=True`（真实固壁，如
    cube_demo 的 "body"）与 `is_no_slip=False`（滑移壁，如 cube_demo 的
    风洞外壁 "tunnel"，用于近似远场/对称边界，物理上零剪切、不产生真实
    边界层）。滑移壁没有真实的近壁粘性子层，Wilcox 公式在这里没有物理
    意义；真实复现：cube_demo 的 "tunnel" 边界组配置为
    `is_no_slip=False`，其 owner 单元因为不属于任何 BL 棱柱加密区、`d1`
    （该单元到最近 body 表面的距离）经常很小，代入公式得到远超合理量级
    的值，被 `_compute_omega_wall_target` 内部的 `omega_max` 安全上限
    钳成 1,000,000，随后 `enforce_omega_wall_relaxation` 每步把这些
    "滑移壁"owner 单元的 omega 强行按固定 relax=0.5 拉向这个物理上荒谬
    的目标（0.5*2267.82+0.5*1e6=501133.91，与真实观测的单步跳变值精确
    吻合，决定性验证：手术式重置 omega 场后单步复现同一批单元同一数值），
    与流场是否真的发展出边界层完全无关——这些单元的平均流速度全程精确
    等于来流值（无滑移损耗），却因为这个 bug 被反复拉向 1e6 附近，最终
    稳定在耗散项与松弛项相互竞争的一个不动点 984312.6254，与该单元是否
    真的靠近任何真实固壁毫无关系。修复：只把 `is_no_slip` 非 False
    （默认 True，与 `fr_ghost_state.py` 的默认值一致）的 WALL 编码计入——
    真实无滑移壁（body）行为完全不变，滑移壁（tunnel）的 k/omega 现在
    正确地退回默认 Neumann（零梯度）处理，与它"物理上应表现得像对称面/
    远场"这个建模意图一致。

    防御性回退：如果 `boundary_ghost_provider` 不是这个类型（例如某些
    测试用的自定义 ghost provider 只是一个普通 callable，没有
    group_code/code_to_config），拿不到分组信息时返回全 False——退回
    调用方原有的 Neumann 默认，不是新的静默 bug（这是修复前唯一的行为，
    对这些没有分组信息的场景数值结果不变）。
    """
    return wall_dirichlet_face_mask(
        getattr(solver, "boundary_ghost_provider", None), solver.mesh.face_connectivity.n_faces)


def wall_dirichlet_face_mask(provider, n_faces: int) -> np.ndarray:
    """`_compute_wall_dirichlet_face_mask` 的纯函数形式（CPU 与 GPU 共用，
    判据见该函数文档）。"""
    group_code = getattr(provider, "group_code", None)
    code_to_config = getattr(provider, "code_to_config", None)
    if group_code is None or code_to_config is None:
        return np.zeros(n_faces, dtype=np.bool_)

    wall_codes = [
        code for code, cfg in code_to_config.items()
        if cfg.get("type") == "WALL" and cfg.get("is_no_slip", True)
    ]
    if not wall_codes:
        return np.zeros(n_faces, dtype=np.bool_)
    return np.isin(group_code, wall_codes)


def _compute_open_boundary_face_mask(solver, flat) -> np.ndarray:
    """哪些面是**开放边界**（流入/流出：FARFIELD、INLET、OUTLET 等），
    供 k/omega 对流的来流条件使用（`compute_scalar_convection_residual` 的
    `open_boundary_face` 参数）。

    分类读的是平均流热边界条件那一份唯一来源
    `boundary/fr_ghost_state.py::ADIABATIC_THERMAL_BC_TYPES`（WALL/
    SYMMETRY）：它的文档写明"INLET/OUTLET/FARFIELD 是流入/流出边界"，
    这里取其补集，不另建一张 BC 类型表。

    与 `_compute_wall_dirichlet_face_mask` 同一个鸭子类型约定：provider 没有
    `group_code/code_to_config`（测试用的普通 callable）时返回全 False，
    保持那些场景的既有行为。
    """
    is_boundary = np.asarray(flat.is_boundary, dtype=np.bool_)
    return open_boundary_code_mask(
        getattr(solver, "boundary_ghost_provider", None), is_boundary.shape[0]) & is_boundary


def open_boundary_code_mask(provider, n_faces: int) -> np.ndarray:
    """按边界组类型逐面标记"开放边界"（未匹配任何组的面按
    `default_config`）。**不含**"是否真边界面"这一条（内部面的
    `group_code` 也是 -1）：CPU 与调用方的 `is_boundary` 求与，GPU 与
    自己从面邻居源推出的真边界掩码求与（`gpu_scalar_transport`）。
    """
    from autoflowcfd.boundary.fr_ghost_state import ADIABATIC_THERMAL_BC_TYPES

    mask = np.zeros(n_faces, dtype=np.bool_)
    group_code = getattr(provider, "group_code", None)
    code_to_config = getattr(provider, "code_to_config", None)
    if group_code is None or code_to_config is None:
        return mask
    default_config = getattr(provider, "default_config", None)
    group_code = np.asarray(group_code)
    for code in np.unique(group_code):
        cfg = code_to_config.get(int(code), default_config)
        if cfg is not None and cfg.get("type") not in ADIABATIC_THERMAL_BC_TYPES:
            mask[group_code == code] = True
    return mask


#: omega 壁面目标值的公式族，由 `AFCFD_OMEGA_WALL_MODE` 选择：
#:   "amplified"（默认，既有行为逐位不变）—— Menter 的放大式
#:       omega_wall = 10 * 6*nu/(beta1*d1^2)
#:   "blended"                          —— Menter 的二项混合式
#:       omega_vis = 6*nu/(beta1*d1^2)
#:       omega_log = sqrt(k)/(C_mu^0.25 * kappa * d1)
#:       omega_wall = sqrt(omega_vis^2 + omega_log^2)
#:
#: **文献依据**（2026-09-15 联网核实，B-6）：混合式与两个渐近支的形式、
#: 以及常数取值，与 OpenFOAM `omegaWallFunction` 的实现逐项一致——
#: `omegaVis = 6*nuw/(beta1_*sqr(y))`、
#: `omegaLog = sqrt(k)/(Cmu25*kappa_*y)`、
#: `omega = sqrt(sqr(omegaVis) + sqr(omegaLog))`，其中 `Cmu25 = pow025(Cmu)`。
#: 默认常数：`beta1_ = 0.075`（该类构造函数）、`Cmu = 0.09`、`kappa = 0.41`
#: （`nutWallFunction` 的默认值，omegaWallFunction 经 nutw 取用）。较新
#: 版本把二项混合换成了分数权重线性混合（`lamFrac*omegaVis +
#: turbFrac*omegaLog`），本项目实现的是经典二项式——它是 Menter 原始
#: 形式，且在两个极限下都渐近正确。
#:
#: **为什么混合式在原理上更可靠**：`amplified` 档那个 10 倍是 Menter 为
#: **有限体积、近壁不解析** 的情形设计的数值手段（把 omega 抬得足够高，
#: 以在粗近壁网格上强制出正确的渐近行为），不是一个物理值；混合式没有
#: 任何这类自由因子，两个支各自是渐近精确解。注意在**壁面解析**（低 Re）
#: 网格上粘性支占绝对主导，于是两档的差别基本就是那个 10 倍。
#:
#: **默认值没有改**：这是湍流模型的物理改动。本项目在 omega 壁面处理上
#: 已经有两次"数学上更对但被真实数据证伪"的先例（显式 SIPG 罚项、点隐式
#: 动态松弛系数，均见 `enforce_omega_wall_relaxation` 文档），所以这里
#: 只提供开关与判据，改默认值必须有真实长程数据。
#:
#: 另注：本档与 `AFCFD_OMEGA_WALL_D1`（长度尺度口径 min|mean）是**两个
#: 独立的维度**，且两者的偏差会**相乘**——`min` 口径在 order=1 上已经
#: 高估 5.60 倍，叠加 10 倍放大就是约 56 倍。
_OMEGA_WALL_MODES = ("amplified", "blended")


#: 混合式的经验常数（来源见上）
_OMEGA_WALL_CMU = 0.09


_OMEGA_WALL_KAPPA = 0.41


def resolve_omega_wall_mode() -> str:
    """读取 `AFCFD_OMEGA_WALL_MODE` 并校验取值（非法值显式报错，不静默
    退回默认——与本项目其余开关同一条约定）。"""
    mode = os.environ.get("AFCFD_OMEGA_WALL_MODE", "amplified").lower()
    if mode not in _OMEGA_WALL_MODES:
        raise ValueError(
            f"AFCFD_OMEGA_WALL_MODE={mode!r} 不是合法取值"
            f"（{' | '.join(_OMEGA_WALL_MODES)}）。"
            f"'amplified' 是既有行为（Menter 放大式 10*6nu/(beta1*d1^2)），"
            f"'blended' 是 Menter 二项混合式 sqrt(omega_vis^2+omega_log^2)，"
            f"见 transport.py 里 _OMEGA_WALL_MODES 一节的文献依据。")
    return mode


def _omega_wall_formula(solver, owner_cells, nu_owner, d1, beta1):
    """按 `AFCFD_OMEGA_WALL_MODE` 算 omega 壁面目标值（未做 omega_max 钳制，
    由调用方统一钳）。

    Args:
        solver: 需要 `turb_model.k_field`（仅 blended 档用到）
        owner_cells: (n_wall_faces,) 各 WALL 面的 owner 单元索引
        nu_owner: (n_wall_faces,) 该单元的运动粘度 mu/rho
        d1: (n_wall_faces,) 近壁特征长度（口径由 AFCFD_OMEGA_WALL_D1 决定）
        beta1: SST 内层 beta 系数
    """
    omega_vis = 6.0 * nu_owner / (beta1 * d1 ** 2)
    if resolve_omega_wall_mode() == "amplified":
        # 10 * 6nu/(beta1*d1^2)——与改动前逐位一致
        return 10.0 * omega_vis
    # blended：需要 owner 单元的 k。取该单元**真实自由度**上的均值，
    # 与本文件 rho_owner 同一处理（native 四面体的零填充槽位冻结在初值，
    # 混进来会带偏；见 fr/native_padding.py）。
    from autoflowcfd.fr.native_padding import (
        order_from_n_sps, reduce_rows_over_real_sps,
    )
    k_field = getattr(getattr(solver, "turb_model", None), "k_field", None)
    if k_field is None:
        raise RuntimeError(
            "AFCFD_OMEGA_WALL_MODE=blended 需要 solver.turb_model.k_field "
            "（对数支 omega_log = sqrt(k)/(Cmu^0.25*kappa*d1) 用到它），"
            "但当前湍流模型没有这个场——不接受静默退回 amplified 档。")
    k_rows = np.asarray(k_field)[owner_cells]
    k_owner = reduce_rows_over_real_sps(
        k_rows, owner_cells < solver.mesh.n_prism_cells,
        order_from_n_sps(k_rows.shape[1]), 'mean')
    omega_log = (np.sqrt(np.maximum(k_owner, 0.0))
                 / (_OMEGA_WALL_CMU ** 0.25 * _OMEGA_WALL_KAPPA * d1))
    return np.sqrt(omega_vis ** 2 + omega_log ** 2)


def _compute_omega_wall_target(
    solver, wall_mask: np.ndarray, mu: float, rho: np.ndarray, flat_face_override=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """按 Wilcox 解析式计算 WALL 面各自的 omega 目标值（真实修复，
    V2.0 专家组盲审发现，2026-08-28）：

        omega_wall = 60*nu / (beta1 * d1^2)

    （Wilcox《Turbulence Modeling for CFD》标准公式，beta1=0.075 是
    SST 内层 beta 系数，omega/sst.py::SSTModelFR.beta1）。

    `d1` 的取法（与直接对 `solver.wall_distance` 做面外插的方案不同，
    是本次实现有意的选择，不是疏漏）：不能用
    `_extrapolate_scalar_to_faces(solver.wall_distance, ...)` 把
    wall_distance 外插到 WALL 面本身——wall_distance 是"到最近壁面的
    距离"，在几何上就位于壁面的这个面自身，外插值会趋于 0，代入公式
    会让 omega_wall 发散到无穷大，不是"精度稍差"而是量纲上完全错误。
    Wilcox 公式里的 d1 本来就是"近壁第一层网格点到壁面的距离"，不是
    "壁面到自身的距离"——这里直接取该面 owner 单元自身 SPs 上
    wall_distance 场的最小值，作为该单元的近壁特征距离（该单元里离墙
    最近的 SP 到墙的真实距离），比在面上外插整个场更贴近公式原意，
    也从根本上避免了除以零。

    Args:
        solver: FRSolver 实例（需要 solver.wall_distance 已计算）
        wall_mask: (n_faces,) bool，WALL 边界面掩码（
            _compute_wall_dirichlet_face_mask 的返回值）
        mu: 分子动力粘度
        rho: (n_cells, n_sps) 密度场，用于取 owner 单元的代表密度算 nu
        flat_face_override: 显式传入时优先使用，不再调用
            `get_flat_face_geometry(solver.mesh, solver.ops)`（真实修复，
            2026-09-02，见 `compute_turbulence_transport_residual` 同名参数
            文档——本函数此前是该文件里唯一没有跟随 flat_face_override 传参
            约定的函数，分布式场景下 `solver.mesh.face_connectivity` 是
            `DistributedFlatFaceGeometry`，不具备 `owner_cube_face` 等字段，
            "完全分布式加载"模式下会直接崩溃；"传统模式"不崩溃是因为
            `get_flat_face_geometry` 按 `mesh.face_flux_points` 对象身份
            缓存，而"传统模式"下这个身份与构造 `dist_fc` 时已经缓存过的
            全局 mesh 是同一个对象，命中缓存返回的是**全局**（非 compact）
            FlatFaceGeometry——`wall_face_idx`（来自 compact 空间的
            wall_mask）被当成全局面索引使用，语义上是错的，只是在
            现有测试里因为从未真正匹配到 WALL 边界组（wall_mask 恒为全
            False，见 `_compute_wall_dirichlet_face_mask` 提前返回分支）
            而没有被触发）

    Returns:
        (omega_wall_value_face, has_value_face)：
        - omega_wall_value_face: (n_faces, n_fp) float，WALL 面上恒为
          该面 owner 单元算出的标量（对该面所有 FP 广播同一个值，不是
          外插得到的逐 FP 不同值——d1 本身就是单元级别的代表量，不需要
          逐 FP 精细区分），非 WALL 面为 0（不会被使用，has_value_face
          对应位置为 False）
        - has_value_face: (n_faces,) bool，与 wall_mask 相同

    真实 bug 修复（2026-09-05，真实网格验证决定性发现）：`d1 = np.maximum(
    d1, 1e-8)` 只防止除零，不防止结果本身失控——cube_demo 791,492 单元
    真实网格上至少有一个 WALL 面 owner 单元的 wall_distance 恰好卡在
    这个 1e-8 下限（真实反推：观测到的 omega_wall 异常值 1.176e14 精确
    对应 d1=1e-8 代入公式的结果），算出 `omega_wall=60*nu/(beta1*d1^2)
    ~1e14`——比"远超任何工程壁面 omega 值"的安全上限 `omega_max`
    (=1e6，sst.py::SSTModelFR.k_max/omega_max 文档) 还要大 8 个数量级。
    此前唯一的消费者（`compute_scalar_convection_residual` 的上风
    ghost）碰巧没有暴露这个问题——无滑移壁面上对流通量本身趋于零，
    ghost 值再大也乘的是接近零的质量通量，天然被掩盖；2026-09-04/05
    新增的两个消费者（`enforce_omega_wall_relaxation` 直接把这个值
    混合进 omega_field 本身、以及当天当场被证伪撤销的 SIPG 罚项）
    都没有这层"乘以近零对流通量"的天然保护，完全暴露了这个此前从未
    触发过的缺口——真实复现：`enforce_omega_wall_relaxation` 点隐式
    公式本身完全正确（bounded in [0,1) 已有专门单元测试钉住），但
    "正确地"把 omega 松弛向一个物理上荒谬的 1e14 目标值，5 步内就把
    全域 omega_mean 打到 1.08e12。修复：在这里、也就是唯一的真值来源，
    把 `omega_wall` 夹到 `solver.turb_model.omega_max`（没有该属性时
    退回 1e6 保守默认），让所有消费者（现在的和未来任何新增的）都
    自动受益，不需要各自重复防御。
    """
    flat = flat_face_override if flat_face_override is not None else get_flat_face_geometry(solver.mesh, solver.ops)
    n_faces = flat.n_faces
    n_fp = flat.n_fp
    beta1 = getattr(solver.turb_model, "beta1", 0.075)
    omega_max = getattr(solver.turb_model, "omega_max", 1e6)

    omega_wall_value_face = np.zeros((n_faces, n_fp), dtype=np.float64)
    wall_face_idx = np.nonzero(wall_mask)[0]
    if len(wall_face_idx) > 0:
        owner_cells = flat.owner_cell[wall_face_idx]
        # **长度尺度口径（2026-09-15 发现的系统性偏差，可切换）**
        #
        # Menter 的 omega 壁面处理 `omega_wall = 10*6*nu/(beta1*Δy1^2)`
        # （这里的 60 就是 10x6）是按**第一层单元中心**的壁距标定的经验
        # 公式。而这里原先取的是 `min`——单元内**全部解点**壁距的最小值。
        # 那既不是单元中心也不是单元高度，不对应任何标准口径，而且因为
        # Gauss-Legendre 解点在高阶时向单元边界聚集，它让目标值产生
        # **随阶数变化**的系统性高估（解点相对壁面的归一化位置实测）：
        #
        #   order=1: 最近解点 0.2113*h -> (0.5/0.2113)^2 =  5.60x 高估
        #   order=2: 最近解点 0.1127*h -> (0.5/0.1127)^2 = 19.68x 高估
        #   order=3: 最近解点 0.0694*h -> (0.5/0.0694)^2 = 51.86x 高估
        #
        # 一个经过标定的壁面函数绝不该有这种阶数依赖。这也解释了为什么
        # 本函数的目标值会顶到 `omega_max`、被下游文档称作"1e6 量级的
        # 应急上限"而不是"日常合理松弛目标"（见
        # `enforce_omega_wall_relaxation` 里两次被真实数据证伪的尝试
        # 记录）——它被喂了一个小 2.4~7.2 倍的长度尺度。
        #
        # `mean`（单元内解点壁距的均值，≈ 形心壁距）与 Menter 的口径
        # 一致，且对阶数是一阶无关的。**默认仍为 `min`**：这是湍流模型
        # 的物理改动，降低近壁 omega 会抬高 nu_t，必须用真实长程数据
        # 验证过才能改默认值——本项目在 omega 壁面处理上已经有两次
        # "数学上更对但真实数据证伪"的先例。
        _d1_mode = os.environ.get("AFCFD_OMEGA_WALL_D1", "min").lower()
        if _d1_mode not in ("min", "mean"):
            raise ValueError(
                f"AFCFD_OMEGA_WALL_D1={_d1_mode!r} 不是合法取值（min | mean）。"
                f"'min' 是既有行为（单元内解点壁距最小值），'mean' 是与 "
                f"Menter 标定口径一致的形心壁距。")
        wd_owner = solver.wall_distance[owner_cells]
        d1 = wd_owner.min(axis=1) if _d1_mode == "min" else wd_owner.mean(axis=1)
        d1 = np.maximum(d1, 1e-8)
        # 只统计真实自由度（2026-09-15 审计）：`rho[owner_cells]` 的行
        # 单元类型任意混合，所以用逐行掩码版。native 四面体的零填充槽位
        # 冻结在初值、会变馊，混进 nu = mu/rho 会带进几个百分点的偏差。
        # 阶数从**数组自身**的 SP 轴反解，不读 solver.current_order/order：
        # 填充划分由被归约数组的 n_sps 决定，从数组反解恒与它自洽（理由见
        # `order_from_n_sps` 文档）。
        from autoflowcfd.fr.native_padding import (
            order_from_n_sps, reduce_rows_over_real_sps,
        )
        _rows = rho[owner_cells]
        rho_owner = reduce_rows_over_real_sps(
            _rows, owner_cells < solver.mesh.n_prism_cells,
            order_from_n_sps(_rows.shape[1]), 'mean')
        nu_owner = mu / np.maximum(rho_owner, 1e-10)
        omega_wall = _omega_wall_formula(
            solver, owner_cells, nu_owner, d1, beta1)
        omega_wall = np.minimum(omega_wall, omega_max)
        omega_wall_value_face[wall_face_idx, :] = omega_wall[:, None]

    return omega_wall_value_face, wall_mask


def enforce_omega_wall_relaxation(solver, dt, relax: float = None,
                                   flat_face_override=None) -> None:
    """真实 bug 修复（2026-09-04，cube_demo 791,492 单元真实网格 Order
    Continuation P0->P1 跨阶后长程发散排查发现，grad_vel 修复之后仍持续
    发散的第二个独立根因）：`_compute_omega_wall_target` 按 Wilcox 解析式
    算出的壁面 omega 目标值（`omega_wall=60*nu/(beta1*d1^2)`，量级可达
    1e5~1e6）**只通过 `compute_scalar_convection_residual` 的上风 ghost
    生效**——`compute_scalar_diffusion_residual`（近壁 omega 动力学的
    主导机制，因为壁面无滑移使对流通量本身趋于零）文档明确写明这个
    解析值"当前对本函数的数值结果没有影响"，是已知、有意搁置的架构
    缺口（"diffusion 侧的解析壁面通量是更大的独立工作"）。

    真实后果（决定性验证，见 verify_gradfix_500steps.py 长程复现）：
    没有扩散侧的强约束，纯靠耗散项 D_omega=rho*beta*omega^2 的显式
    积分，边界层棱柱单元的 omega 会在数十~上百步内被压向下界
    （真实测得：166,980个边界层单元里 90,416 个、66%在150步内至少有
    一个解点 omega<1e-6，且这个比例逐步增长而非趋于稳定）——omega
    塌陷经 nu_t=a1*k/max(a1*omega,...) 的近零分母奇点反过来把湍流
    粘性比推到安全上限（真实测得 nu_t/nu_molecular~1e5，触及
    TURBULENT_VISCOSITY_RATIO_MAX），持续向平均流注入过量粘性应力，
    是 grad_vel 修复后仍能观测到的中长期（~100步后）持续增长的直接
    驱动源（而不是 grad_vel bug 本身遗留的影响——那个 bug 修复后已
    验证首个~90步完全无发散迹象，本机制独立起效于其后）。

    本函数用最低数值风险的方式补上这个缺口：不改动扩散残差/DG通量
    的稳定性特征，而是在 update_fields+positivity limiter 之后，
    直接对 WALL 面 owner 单元的 omega_field 做一次向解析壁面目标值的
    松弛（标准壁面函数做法，等价于 OpenFOAM omegaWallFunction 对
    近壁单元值的直接赋值/松弛处理，不是发明新方案）。`relax` 是
    固定松弛系数（每步只走向目标值的这个比例，不是硬性 hard-set，
    避免单步冲击过大引入新的震荡）。

    2026-09-05 曾尝试把这里改成"点隐式"推导的动态松弛系数
    （`relax_eff = dt*c_wall/(1+dt*c_wall)`，c_wall 正比于 1/d1^2）
    ——数学上确实排除了显式罚项的刚性超调（另一次已撤销的 SIPG
    尝试），但真实网格验证**再次证伪**：动态 relax_eff 对细网格近壁
    单元（d1 小）天然趋近 1（几乎每步都把 omega 直接怼到 target），
    而 target 本身（哪怕已经被下面 `_compute_omega_wall_target` 的
    `omega_max` 上限保护，不再是失控的 1e14）仍然是 1e6 这个量级的
    "应急上限"，不是"日常合理松弛目标"——把大量边界层单元在几步内
    强行拉到这个量级，会让 D_k=rho*beta_star*k*omega 这个耗散项跟着
    暴涨，2 步内就把全域 k_mean 从 38 打到 0.17（真实数值，不是
    NaN/Inf，但同样是不可接受的物理扰动）。而固定的 `relax=0.5`
    对*所有*单元一视同仁地只走一半路程，天然更温和、给耦合系统留出
    调整时间——这版已用真实生产续算验证 900+ 步保持平均流场零漂移
    （见项目记忆），比"数学上更精确"但经验证更具破坏性的点隐式版本
    更适合作为当前的工程选择。教训：这类近壁松弛的"正确性"不能只看
    单个 ODE 是否无条件稳定，还要看它对耦合场（k 反过来依赖 omega）
    造成的扰动幅度是否温和——本函数改回固定 relax，`dt` 参数保留
    只是为了不破坏调用方签名，不再参与计算。

    Args:
        solver: FRSolver 实例
        dt: 未使用（保留参数位置以兼容调用方签名，见上面"教训"一节）。
        relax: 松弛系数，每步 omega_field[wall_owner] 更新为
            `(1-relax)*old + relax*omega_wall_target`
        flat_face_override: 分布式路径复用同一约定，见
            `compute_turbulence_transport_residual` 同名参数文档
    """
    if relax is None:
        relax = 0.5
    hit_cells, avg_target = omega_wall_cell_targets(solver, flat_face_override)
    if hit_cells.size == 0:
        return
    turb = solver.turb_model
    turb.omega_field[hit_cells, :] = (
        (1.0 - relax) * turb.omega_field[hit_cells, :] + relax * avg_target[:, None]
    )


def omega_wall_cell_targets(solver, flat_face_override=None):
    """壁面 owner 单元与各自的 Wilcox omega 目标值 `(hit_cells, avg_target)`。

    显式路径的每步松弛（`enforce_omega_wall_relaxation`）与隐式路径的残差
    内强约束（`fr_solver/turbulence/implicit.py`）共用这一份——同一个壁面
    条件只允许一个事实来源。角部单元是多个 WALL 面的 owner，取各面目标值
    的平均。没有壁面时返回两个空数组。
    """
    empty = (np.zeros(0, dtype=np.int64), np.zeros(0))
    wall_mask = _compute_wall_dirichlet_face_mask(solver)
    if not np.any(wall_mask):
        return empty

    Q = solver.state.Q
    rho = Q[:, :, 0]
    omega_wall_value_face, has_wall = _compute_omega_wall_target(
        solver, wall_mask, solver.mu_molecular, rho, flat_face_override=flat_face_override,
    )

    flat = flat_face_override if flat_face_override is not None else get_flat_face_geometry(solver.mesh, solver.ops)
    wall_face_idx = np.nonzero(has_wall)[0]
    if len(wall_face_idx) == 0:
        return empty
    owner_cells = flat.owner_cell[wall_face_idx]
    target = omega_wall_value_face[wall_face_idx, 0]  # 同一面上恒为同一常数，见函数文档

    # 同一个 owner 单元可能是多个 WALL 面的 owner（角部单元）——用
    # np.add.at 累加再除以命中次数取平均目标值，不能直接花式索引赋值
    # 覆盖（后写的面会覆盖先写的面，不是真正的平均）。
    sum_target = np.zeros(solver.state.n_cells)
    count = np.zeros(solver.state.n_cells)
    np.add.at(sum_target, owner_cells, target)
    np.add.at(count, owner_cells, 1.0)
    hit_cells = np.nonzero(count > 0)[0]
    avg_target = sum_target[hit_cells] / count[hit_cells]
    return hit_cells, avg_target
