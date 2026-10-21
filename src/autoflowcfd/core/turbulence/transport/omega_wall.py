"""AutoFlowCFD V2.0 - omega 壁面边界处理（Dirichlet 掩码、目标值）。

SST 的 omega 在壁面上有解析值（Wilcox `omega_wall = 60*nu/(beta1*d1^2)`），
它进对流的面外插 ghost 规则与扩散残差的面 Dirichlet（显式与隐式同一离散）。
此前显式路径还在每步之后对壁面单元全部解点做一次松弛，2026-10-01 删除：隐式 NK 收敛到 R≈1e-8 的
槽道定常解只做一次就被改动 2.13 倍（P1）/1.58 倍（P3），即正确的离散定常解不是它的不动点（显式槽道
40000 步残差停在 1e4）；隐式路径的同类整单元强约束 09-30 已因同一原因删除。
"""

import numpy as np

from typing import Tuple


from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
from autoflowcfd.core.turbulence.sst.bounds import (
    omega_wall_length_scale,
)



def boundary_groups(provider):
    """边界幽灵态提供者的逐面边界组编码与组配置 `(group_code, code_to_config, default_config)`。

    湍流量的壁面条件（k 的 Dirichlet 0、omega 壁面目标值）与来流条件都按边界组类型施加，所以提供者必须带分组
    信息：生产求解器的 `BoundaryGhostStateProvider`（`core/fr_solver/boundary/ghost.py::build_boundary_ghost_provider`）
    恒带。显式给出的零梯度外插 `DefaultGhostProvider` 表示"没有物理边界"（自由流守恒类测试），返回空分组。
    其它情况报错：2026-10-19（I-64）以前拿不到分组信息时静默返回全 False 掩码，壁面 k=0、omega 壁面值与来流
    条件全部失效而不报错（与 I-5 删除的幽灵态缺省同一类问题）。
    """
    from autoflowcfd.boundary.fr_ghost_state import DefaultGhostProvider

    if isinstance(provider, DefaultGhostProvider):
        return None, {}, None
    group_code = getattr(provider, "group_code", None)
    code_to_config = getattr(provider, "code_to_config", None)
    if group_code is None or code_to_config is None:
        raise ValueError(
            f"湍流量的边界条件需要按边界组给定条件的幽灵态提供者（带 group_code / code_to_config），"
            f"收到 {type(provider).__name__}；零梯度外插须显式传 DefaultGhostProvider()")
    return np.asarray(group_code), code_to_config, getattr(provider, "default_config", None)


def _compute_wall_dirichlet_face_mask(solver) -> np.ndarray:
    """**物理无滑移**壁面边界面掩码 `(n_faces,)`，供 k 的 Dirichlet 0 与 omega 壁面目标值使用。

    分组来自 `solver.boundary_ghost_provider`（要求见 `boundary_groups`）。只计入物理无滑移固壁
    （`boundary/wall_kind.py::is_physical_no_slip_wall`）：滑移壁（如风洞外壁，`is_no_slip=False`）物理上零剪切、
    不是边界层，k/omega 取零梯度（2026-09-12 cube_demo 真实网格上滑移壁被当成壁面、omega 被拉向上限的缺陷）。
    """
    return wall_dirichlet_face_mask(solver.boundary_ghost_provider, solver.mesh.face_connectivity.n_faces)


def wall_dirichlet_face_mask(provider, n_faces: int) -> np.ndarray:
    """`_compute_wall_dirichlet_face_mask` 的纯函数形式（CPU 与 GPU 共用）。"""
    group_code, code_to_config, _ = boundary_groups(provider)
    if group_code is None:
        return np.zeros(n_faces, dtype=np.bool_)

    # 物理无滑移固壁（`boundary/wall_kind.py`）：壁面模型（WMLES、RANS 自动壁面处理）下幽灵态的 is_no_slip 被
    # 关掉，但湍流量的壁面条件仍按物理事实施加
    from autoflowcfd.boundary.wall_kind import is_physical_no_slip_wall

    wall_codes = [code for code, cfg in code_to_config.items() if is_physical_no_slip_wall(cfg)]
    if not wall_codes:
        return np.zeros(n_faces, dtype=np.bool_)
    return np.isin(group_code, wall_codes)


def _compute_open_boundary_face_mask(solver) -> np.ndarray:
    """**开放边界**（流入/流出：FARFIELD、INLET、OUTLET 等）面掩码，供 k/omega 对流的来流条件使用
    （`compute_scalar_convection_residual` 的 `open_boundary_face` 参数），见 `open_boundary_code_mask`。"""
    return open_boundary_code_mask(solver.boundary_ghost_provider, solver.mesh.face_connectivity.n_faces)


def open_boundary_code_mask(provider, n_faces: int) -> np.ndarray:
    """按边界组类型逐面标记"开放边界"（未匹配任何组的面按 `default_config`）。

    分类读平均流热边界条件那一份唯一来源 `boundary/fr_ghost_state.py::ADIABATIC_THERMAL_BC_TYPES`（WALL/SYMMETRY）
    的补集，不另建 BC 类型表；分组要求见 `boundary_groups`。**不含**"是否边界点"这一条（内部面的 `group_code`
    也是 -1）：消费方按 `transport/faces.py::inflow_points` 与两侧坐标系的边界点（含混合拆分面的边界半区）求与。
    CPU 与 GPU 共用这一张。
    """
    from autoflowcfd.boundary.fr_ghost_state import ADIABATIC_THERMAL_BC_TYPES

    mask = np.zeros(n_faces, dtype=np.bool_)
    group_code, code_to_config, default_config = boundary_groups(provider)
    if group_code is None:
        return mask
    for code in np.unique(group_code):
        cfg = code_to_config.get(int(code), default_config)
        if cfg is not None and cfg.get("type") not in ADIABATIC_THERMAL_BC_TYPES:
            mask[group_code == code] = True
    return mask
    default_config = getattr(provider, "default_config", None)
    group_code = np.asarray(group_code)
    for code in np.unique(group_code):
        cfg = code_to_config.get(int(code), default_config)
        if cfg is not None and cfg.get("type") not in ADIABATIC_THERMAL_BC_TYPES:
            mask[group_code == code] = True
    return mask


def _omega_wall_formula(solver, owner_cells, nu_owner, d1, beta1):
    """omega 壁面目标值（未做 omega_max 钳制，由调用方统一钳）：公式随近壁处理方式
    （`turbulence/sst/bounds.py::omega_wall_value`，CPU 与 GPU 共用）。auto 用到贴壁单元 k 的**真实自由度**
    均值（native 四面体的零填充槽位冻结在初值，混进来会带偏；见 fr/native_padding.py）。"""
    from autoflowcfd.core.turbulence.sst.bounds import omega_wall_value

    turb = solver.turb_model
    k_owner = _owner_k_mean(solver, owner_cells) if turb.wall_treatment == "auto" else None
    return omega_wall_value(np, nu_owner, d1, k_owner, beta1, turb.wall_treatment)


def _owner_k_mean(solver, owner_cells):
    """贴壁单元 k 的真实自由度均值（auto 档壁面目标值的输入）。"""
    from autoflowcfd.fr.native_padding import order_from_n_sps, reduce_rows_over_real_sps

    k_rows = np.asarray(solver.turb_model.k_field)[owner_cells]
    return reduce_rows_over_real_sps(
        k_rows, owner_cells < solver.mesh.n_prism_cells, order_from_n_sps(k_rows.shape[1]), 'mean')


def _compute_omega_wall_target(
    solver, wall_mask: np.ndarray, mu: float, rho: np.ndarray, flat_face_override=None, with_k_slope: bool = False,
) -> Tuple[np.ndarray, ...]:
    """按 Wilcox 解析式计算 WALL 面各自的 omega 目标值（真实修复，
    V2.0 专家组盲审发现，2026-08-28）：

        omega_wall = 60*nu / (beta1 * d1^2)

    （Wilcox《Turbulence Modeling for CFD》标准公式，beta1=0.075 是
    SST 内层 beta 系数，core/turbulence/sst/model.py::SSTModelFR.beta1）。

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
            False）而没有被触发）

    Returns:
        (omega_wall_value_face, has_value_face)：
        - omega_wall_value_face: (n_faces, n_fp) float，WALL 面上恒为
          该面 owner 单元算出的标量（对该面所有 FP 广播同一个值，不是
          外插得到的逐 FP 不同值——d1 本身就是单元级别的代表量，不需要
          逐 FP 精细区分），非 WALL 面为 0（不会被使用，has_value_face
          对应位置为 False）
        - has_value_face: (n_faces,) bool，与 wall_mask 相同
        - （`with_k_slope` 时）k_slope_face: (n_faces,) 目标值的对数对 owner 单元 k 真实自由度均值的导数
          （`sst/bounds.py::log_omega_wall_k_slope`；resolved 档恒为 0），湍流解析块 Jacobian 用

    真实 bug 修复（2026-09-05，真实网格验证决定性发现）：`d1 = np.maximum(
    d1, 1e-8)` 只防止除零，不防止结果本身失控——cube_demo 791,492 单元
    真实网格上至少有一个 WALL 面 owner 单元的 wall_distance 恰好卡在
    这个 1e-8 下限（真实反推：观测到的 omega_wall 异常值 1.176e14 精确
    对应 d1=1e-8 代入公式的结果），算出 `omega_wall=60*nu/(beta1*d1^2)
    ~1e14`——比"远超任何工程壁面 omega 值"的安全上限 `omega_max`
    (=1e6，sst/model.py::SSTModelFR.k_max/omega_max 文档) 还要大 8 个数量级。
    此前唯一的消费者（`compute_scalar_convection_residual` 的上风
    ghost）碰巧没有暴露这个问题——无滑移壁面上对流通量本身趋于零，
    ghost 值再大也乘的是接近零的质量通量，天然被掩盖；2026-09-04/05
    新增的两个消费者（当时的显式步后松弛直接把这个值混合进 omega_field
    本身——已于 2026-10-01 删除，以及当天当场被证伪撤销的 SIPG 罚项）
    都没有这层"乘以近零对流通量"的天然保护，完全暴露了这个此前从未
    触发过的缺口——真实复现：当时的点隐式松弛公式本身正确，但
    "正确地"把 omega 松弛向一个物理上荒谬的 1e14 目标值，5 步内就把
    全域 omega_mean 打到 1.08e12。修复：在这里、也就是唯一的真值来源，
    把 `omega_wall` 夹到 `solver.turb_model.omega_max`，让所有消费者（现在的和
    未来任何新增的）都自动受益，不需要各自重复防御。上界随最近壁面解点给定
    （`turbulence/sst/bounds.py` 模块文档，2026-10-03：此前写死 1e6，壁面解析网格上
    比物理壁面值低两个量级，壁面条件被压小、钳位处残差不可微）。
    """
    flat = flat_face_override if flat_face_override is not None else get_flat_face_geometry(solver.mesh, solver.ops)
    n_faces = flat.n_faces
    n_fp = flat.n_fp
    beta1 = solver.turb_model.beta1
    omega_max = solver.turb_model.omega_max

    omega_wall_value_face = np.zeros((n_faces, n_fp), dtype=np.float64)
    wall_face_idx = np.nonzero(wall_mask)[0]
    if len(wall_face_idx) > 0:
        owner_cells = flat.owner_cell[wall_face_idx]
        # **长度尺度口径（2026-09-15 发现的系统性偏差）**
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
        # 本函数的目标值会顶到 `omega_max`、被当时的显式步后松弛（2026-10-01
        # 已删除）文档称作"1e6 量级的应急上限"——它被喂了一个小 2.4~7.2 倍的
        # 长度尺度。
        #
        # `mean`（单元内解点壁距的均值，≈ 形心壁距）与 Menter 的口径
        # 一致，且对阶数是一阶无关的。口径随近壁处理方式：resolved 保持
        # `min`（放大式的既有标定，降低近壁 omega 会抬高 nu_t，未经真实
        # 长程数据验证不改），auto 用 `mean`（与壁面模型匹配高度一致）。
        # 口径判据、真实解点与壁面解点的处理见 `sst/bounds.py::omega_wall_length_scale`
        # （CPU/GPU 共用）。
        d1 = omega_wall_length_scale(solver.wall_distance[owner_cells],
                                     owner_cells < solver.mesh.n_prism_cells, np, solver.turb_model.wall_treatment)
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
    if not with_k_slope:
        return omega_wall_value_face, wall_mask
    k_slope_face = np.zeros(n_faces)
    if len(wall_face_idx) > 0 and solver.turb_model.wall_treatment == "auto":
        from autoflowcfd.core.turbulence.sst.bounds import log_omega_wall_k_slope

        k_slope_face[wall_face_idx] = log_omega_wall_k_slope(
            np, nu_owner, d1, _owner_k_mean(solver, owner_cells), beta1, omega_max)
    return omega_wall_value_face, wall_mask, k_slope_face
