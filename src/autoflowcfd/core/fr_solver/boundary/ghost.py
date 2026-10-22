"""AutoFlowCFD V2.0 - 边界幽灵态 provider（含 SEM 入口 FP 位置）
"""

from typing import Any, Dict, NamedTuple, Optional, Tuple

import numpy as np

from loguru import logger

from autoflowcfd.boundary.fr_ghost_state import BoundaryGhostStateProvider, InletSEMGhostState

from autoflowcfd.grid.connectivity.face_connectivity_boundary_tags import tag_boundary_groups_for_mesh, triangle_keys


class BoundaryPhysics(NamedTuple):
    """构造边界条件用到的物理设置。

    2026-10-21（I-106）以前构造函数接收整个求解器对象，完全分布式加载的 root 没有求解器，为此伪造一个
    `SimpleNamespace` 桩（`wall_model=object()` 只为让"是否为 None"的判断走对分支）；桩上缺哪个属性，就在哪条
    分支上崩溃或静默取错值（历史上缺过 turb_model_name、wall_model、ops、Tu / VR / 分子粘度 / SEM 涡数）。
    """
    freestream: dict               # rho_inf / vel_inf / p_inf 与攻角、侧滑角（`direction_from_freestream`）
    mu_molecular: float
    turb_model_name: str           # 规范化的大写模型名；决定速度入口是否注入合成湍流
    wall_modelled: bool            # 壁面模型激活（`build_wall_model` 非 None）：WALL 组幽灵态改为滑移镜像
    turbulence_intensity: float    # 合成湍流入口的雷诺应力与涡核尺度（与 RANS 来流同一组量）
    viscosity_ratio: float
    sem_num_eddies: Optional[int]  # None：按 Jarrin 密度判据自动取


def boundary_physics(solver) -> BoundaryPhysics:
    """求解器上的同名设置（各后端构造时设置）。"""
    return BoundaryPhysics(solver.freestream, float(solver.mu_molecular), solver.turb_model_name,
                           solver.wall_model is not None, solver._turbulence_intensity, solver._viscosity_ratio,
                           solver._sem_num_eddies)


def _compute_inlet_fp_positions(mesh, ops, is_target_face: np.ndarray) -> Dict[int, np.ndarray]:
    """一组边界面记录（全部记录，含拆分四边形的非主记录）各自通量点的物理坐标 `{面号: (n_fp, 3)}`。

    坐标取自唯一来源 `core/fr_operators/face_points.py::flux_point_positions`（P>=1 解点外插精确；P0 每面
    一个通量点，外插只能得到单元中心，取构造面几何时由网格节点算出的位置）。
    """
    from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
    from autoflowcfd.core.fr_operators.face_points import flux_point_positions

    flat = get_flat_face_geometry(mesh, ops)
    faces = np.flatnonzero(np.asarray(is_target_face))
    n_fp = int(flat.n_fp)
    xyz = flux_point_positions(mesh, flat, np.repeat(faces, n_fp), np.tile(np.arange(n_fp), faces.size))
    xyz = xyz.reshape(faces.size, n_fp, 3)
    return {int(f): xyz[k] for k, f in enumerate(faces)}


def _build_sem_inlet(mesh, ops, physics: BoundaryPhysics, name: str, is_group_face: np.ndarray,
                     flow_direction: np.ndarray, Q_free) -> Optional[InletSEMGhostState]:
    """一个速度入口组的合成湍流入口（`boundary/synthetic_inlet.py`）：涡核尺度、雷诺应力与来流湍流同一组量。"""
    from autoflowcfd.boundary.synthetic_inlet import SyntheticEddyMethod, sem_length_scale, synthetic_inlet_seed
    from autoflowcfd.core.fr_solver.turbulence.init import freestream_k_omega
    from autoflowcfd.core.turbulence.sgs import les_filter_width
    from autoflowcfd.core.turbulence.sst.bounds import BETA_STAR

    positions_by_face = _compute_inlet_fp_positions(mesh, ops, is_group_face)
    if not positions_by_face:
        return None
    faces = np.fromiter(positions_by_face, dtype=np.int64)
    n_fp = next(iter(positions_by_face.values())).shape[0]
    # 涡核尺度 sigma = max(k^(3/2)/epsilon, Delta)：Delta 取入口面所属单元的 LES 滤波宽度（计入阶数）
    n_cells = int(mesh.n_cells)
    delta = les_filter_width(mesh.cell_volumes, np.arange(n_cells) < int(mesh.n_prism_cells), int(mesh.order))
    k_inf, omega_inf = freestream_k_omega(physics.freestream, physics.mu_molecular, physics.turbulence_intensity,
                                          physics.viscosity_ratio)
    sigma_face = sem_length_scale(k_inf, omega_inf, BETA_STAR, delta[mesh.face_connectivity.owner_cell[faces]])
    face_sigma = {int(f): np.full(n_fp, sigma_face[i]) for i, f in enumerate(faces)}

    all_positions = np.concatenate([positions_by_face[int(f)] for f in faces])
    sem = SyntheticEddyMethod(all_positions, np.repeat(sigma_face, n_fp), flow_direction,
                              num_eddies=physics.sem_num_eddies, seed=synthetic_inlet_seed(name))
    # 各向同性雷诺应力 R = (2/3) k I（k 由湍流强度给定，与 RANS 来流同一个量）
    reynolds_stress = (2.0 / 3.0) * k_inf * np.eye(3)
    logger.info(
        f"合成湍流入口（SEM）'{name}'：{faces.size} 个面，涡核 {sem.num_eddies} 个"
        f"（密度判据 {sem.recommended_num_eddies}），涡核尺度 {sigma_face.min():.4g}~{sigma_face.max():.4g} m"
        f"（k^1.5/epsilon = {np.sqrt(k_inf) / (BETA_STAR * omega_inf):.4g} m），u' = {np.sqrt(2.0 * k_inf / 3.0):.4g} m/s")
    return InletSEMGhostState(sem, positions_by_face, face_sigma, Q_mean=np.array(Q_free),
                              reynolds_stress=reynolds_stress)


def build_boundary_ghost_provider(mesh, ops, physics: BoundaryPhysics,
                                  bc_overrides: Dict[str, Dict[str, Any]]) -> BoundaryGhostStateProvider:
    """构建边界幽灵态提供者 (BD-01)：`mesh` 的边界组按 `mesh.boundary_bc_types`（或 `bc_overrides`）给出条件；
    `ops` 只用于合成湍流入口的通量点定位。求解器调用方传 `boundary_physics(self)`。

    Raises:
        ValueError: 网格没有面连接关系（尚未 `load_from_volume_mesh(build_faces=True)`），或边界面 / 边界组的
            条件不全（见下）。2026-10-19 以前没有面连接关系时只打警告、返回 None，残差核心函数再把 None 静默换成
            零梯度外插——边界条件对残差不可见（I-5）。
    """
    if mesh.face_connectivity is None:
        raise ValueError("网格没有面连接关系（load_from_volume_mesh(build_faces=True)），无法施加边界条件")

    freestream = physics.freestream
    rho_inf, vel_inf, p_inf = freestream["rho_inf"], freestream["vel_inf"], freestream["p_inf"]
    # 来流方向由攻角/侧滑角决定（此前这里把它硬编码成 +x，见
    # core/utils/flow_direction.py 模块文档列出的五处硬编码）。
    from autoflowcfd.core.utils.flow_direction import direction_from_freestream
    _dir = direction_from_freestream(freestream)
    _v_free = vel_inf * _dir
    Q_free = [rho_inf, float(_v_free[0]), float(_v_free[1]), float(_v_free[2]), p_inf]

    bc_types = mesh.boundary_bc_types or {}

    face_conn = mesh.face_connectivity
    boundary_idx = face_conn.get_boundary_face_indices()
    # 声明为周期的组必须真的配对过：配对在网格加载时完成（`face_connectivity_periodic.py`），配对后的周期面是内部
    # 面。这里防的是加载之后才把类型改成 PERIODIC——那些面仍在边界上，却没有任何边界条件
    for name in (n for n, raw_type in bc_types.items() if raw_type == "PERIODIC"):
        n_left = int(np.count_nonzero(np.isin(triangle_keys(face_conn.face_node_ids[boundary_idx]),
                                              triangle_keys(mesh.boundary_faces[name]))))
        if n_left:
            raise ValueError(f"周期边界组 '{name}' 有 {n_left} 个面未配对（仍是边界面）：周期类型必须在网格加载之前"
                             "给出，并带 paired_with/translation 参数")
    group_code, name_to_code = tag_boundary_groups_for_mesh(mesh, face_conn)
    # 每个边界面都必须属于一个边界组（2026-10-09 以前没有组的边界面静默当成远场：网格内部缺口的面因此在流场
    # 内部施加来流条件——cube_demo 生成网格的 86 个缺口面里有 20 个就是这样）
    untagged = boundary_idx[group_code[boundary_idx] < 0]
    if len(untagged):
        centers = face_conn.center[untagged]
        raise ValueError(
            f"{len(untagged)}/{len(boundary_idx)} 个边界面不属于任何边界组（位于 {centers.min(axis=0)} ~ "
            f"{centers.max(axis=0)}），无法给出边界条件：体网格不封闭（内部缺口），或体网格与面网格不对应")

    # 真实 bug 修复（#9，V2.0 专家组盲审第4轮，2026-08-28，WMLES 假滑移
    # 边界）：WMLES 激活时，WALL 组此前恒用 is_no_slip=True 构造 ghost
    # state——`wall_ghost_state` 把 ghost 切向速度镜像为与内部值相反
    # （Q_ghost=2*v_wall-Q_int），逼出一个基于（本项目近壁网格通常无法
    # 真正分辨粘性底层的）解析速度梯度的虚假壁面剪应力，然后
    # 壁面模型（当时的 `solver_helpers.compute_wmles_wall_stress_correction`）又把壁面模型
    # 算出的 tau_w 作为**额外**动量源项叠加在这个虚假剪应力之上——两者
    # 同时存在，造成双重计权。
    #
    # 按 WMLES 壁面模型文献的标准做法修正（Kawai & Larsson 团队官方页面
    # https://wmles.umd.edu/wall-stress-models/coupling-les-to-a-wall-stress-model/
    # 明确用词"replace"；Kang et al. 2024, arXiv:2405.15899，与本项目同为
    # DG/FR 类弱式框架，给出具体公式并证明这样构造"不会双重计权"）：
    # wall model 给出的 tau_w 应该**取代**、而不是叠加在解析梯度算出的
    # 切向剪应力上。做法是让 WMLES 激活时的 WALL ghost state 退化成
    # is_no_slip=False（与下面 SLIP_WALL 用的完全同一套构造：法向速度
    # 镜像翻转以保证不可穿透 u_n=0，切向速度与内部值完全相同、无跳跃）——
    # 这让该面在 BR1/LDG 粘性通量组装里对切向方向的梯度贡献退化为零
    # （没有虚假的解析剪应力），tau_w 因此成为该面切向应力的唯一来源，
    # 不再与之共存。法向速度仍然是不可穿透边界，物理上不受影响。
    #
    # 这把"粘性 WALL + WMLES"与"SLIP_WALL"底层复用同一个 ghost state
    # 构造。**措辞更正（2026-09-14）**：此前这里写"共享同一个数值机制是
    # 刻意的工程简化"——"简化"不准确。两者的**正确** ghost 态恰好就是
    # 同一个（滑移/对称壁的物理条件 u_n=0+切向自由滑移，与 WMLES 要求的
    # "切向无跳跃以免双重计权、法向仍不可穿透"，都精确对应"法向镜像反号
    # + 切向保持"这一个构造），是同一个精确构造被两个不同物理条件同时
    # 要求，不是用一个近似去凑合两种情形，复用它没有引入任何误差。
    # 完整论证与热边界条件（两个分支都是绝热壁；本项目目前没有等温壁
    # 这种 BC 类型）见 wall_ghost_state 文档字符串。
    # **`is_no_slip` 与 `physical_no_slip` 是两件事（2026-09-18 拆开）**：
    #   `is_no_slip`       —— **幽灵态构造开关**。WMLES 激活时它被关掉，
    #                         为的是不与壁面模型的应力修正双重计权
    #                         （完整论证见上方与 wall_ghost_state 文档）。
    #   `physical_no_slip` —— **物理事实**：这面墙是不是静止的不可穿透
    #                         固壁。WMLES 并不改变这一点。
    #
    # 为什么必须拆：此前两者共用一个字段，于是 `WALL`（物理无滑移）与
    # `SLIP_WALL`（物理滑移）在 WMLES 下**都**变成 `is_no_slip=False`，
    # 下游无从区分（壁面距离、壁面模型、气动力积分、湍流壁面条件都只认物理
    # 无滑移固壁，唯一判据 `boundary/wall_kind.py`）。
    wall_is_no_slip = not physics.wall_modelled
    type_map: Dict[str, Tuple[str, Dict[str, Any]]] = {
        "WALL": ("WALL", {"is_no_slip": wall_is_no_slip,
                          "physical_no_slip": True}),
        "SLIP_WALL": ("WALL", {"is_no_slip": False,
                               "physical_no_slip": False}),
        "VELOCITY_INLET": ("INLET", {"Q_inlet": Q_free}),
        "PRESSURE_OUTLET": ("OUTLET", {"p_outlet": p_inf}),
        "SYMMETRY": ("SYMMETRY", {}),
        # 远场（来流特征边界）。2026-10-09 以前没有组的边界面一律静默当成它，现在必须显式声明
        "FARFIELD": ("FARFIELD", {"Q_free": Q_free}),
    }

    # BD-02：LES/DDES 模式下给 VELOCITY_INLET 组接入合成湍流入口 (SEM)，
    # 取代常量 Q_inlet——算法本身（boundary/synthetic_inlet.py）已在
    # V2.0 二次评审时修复正确（真实入口几何/雷诺应力 Cholesky 分解/
    # 涡核时间演化），但此前从未被任何边界路径调用。涡结构的时间推进由求解器每个物理步调用
    # `advance_synthetic_inlets`，不在这里（构造阶段）推进。
    #
    # 解析湍流脉动的模型（LES、WMLES、DDES、IDDES）在速度入口注入合成湍流。壁面模型只给出近壁应力，不提供
    # 入口的湍流内容：2026-10-19（I-75）以前 WMLES 以"壁面模型已承担近壁湍流校正"为由被排除，发展流的入口是层流。
    from autoflowcfd.core.turbulence.registry import uses_synthetic_inlet

    use_sem = uses_synthetic_inlet(physics.turb_model_name)

    code_to_config: Dict[int, Dict[str, Any]] = {}
    for name, code in name_to_code.items():
        override = bc_overrides.get(name)
        if override is not None:
            code_to_config[code] = override
            continue
        # 类型必须明确（2026-10-09 以前缺类型、或类型不认识时一律静默当成远场：壁面被当成远场时结果完全错误、
        # 没有任何提示）
        if name not in bc_types:
            raise ValueError(
                f"边界组 '{name}' 没有边界条件类型（mesh.boundary_bc_types 缺这个组，也没有 bc_overrides）")
        raw_type = bc_types[name]          # 周期组不在这里（打标签不含周期组，配对后它们的面是内部面）
        if raw_type not in type_map:
            raise ValueError(f"边界组 '{name}' 的类型 {raw_type!r} 求解器不支持（支持 {sorted(type_map)}）")
        mapped_type, default_params = type_map[raw_type]
        config: Dict[str, Any] = {"type": mapped_type, **default_params}

        if mapped_type == "INLET" and use_sem:
            config["sem"] = _build_sem_inlet(mesh, ops, physics, name, group_code == code, _dir, Q_free)

        code_to_config[code] = config

    logger.info(
        f"Boundary conditions configured for {len(name_to_code)} group(s): "
        f"{[(name, code_to_config[code]['type']) for name, code in name_to_code.items() if code in code_to_config]}"
    )

    return BoundaryGhostStateProvider(group_code, code_to_config,
                                      code_to_name={code: name for name, code in name_to_code.items()})
