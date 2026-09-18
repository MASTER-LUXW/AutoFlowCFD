"""
AutoFlowCFD V2.0 - FRSolver 边界条件配置构建 (从 fr_solver.py 拆分)

把网格边界组信息与用户/默认 BC 参数接到 boundary/fr_ghost_state.py 的
BoundaryGhostStateProvider 接口上，供 core/fr_residual_inviscid.py 使用。
"""

from typing import Any, Dict, Optional

import numpy as np
from loguru import logger

from autoflowcfd.boundary.fr_ghost_state import BoundaryGhostStateProvider, InletSEMGhostState
from autoflowcfd.grid.connectivity.face_connectivity import tag_boundary_groups_for_mesh

# LES/DDES 入口合成湍流默认参数（BD-02）。真实修复（V2.0 专家组盲审
# 发现，2026-08-28）：此前这两个量恒为硬编码常量，没有任何 CLI/配置
# 途径覆盖，跑真实工程案例（不同来流湍流度/涡核密度）只能改源码。现在：
# - 目标雷诺应力改为复用已有的 `--turbulence-intensity`（solver._turbulence_
#   intensity，同一个量本来就用于 RANS 自由来流 k/omega 初值），不再单独
#   维护一个 SEM 专用湍流度——一个旋钮统一表达"来流湍流强度"，
#   见 build_boundary_ghost_provider 内 u_fluct 的计算。这个下面的常量
#   现在只是 solver 没有设置 _turbulence_intensity 属性时（理论上不会
#   发生，FRSolver/GPUFRSolver 构造时恒会设置）的兜底默认值。
# - 涡核数量新增 `--sem-num-eddies` CLI 选项（solver._sem_num_eddies），
#   默认值沿用原来的 200（未指定时保持向后兼容的行为）。
_SEM_DEFAULT_TURBULENCE_INTENSITY = 0.01
_SEM_DEFAULT_NUM_EDDIES = 200


def _compute_inlet_fp_positions(solver, face_conn, is_target_face: np.ndarray) -> Dict[int, np.ndarray]:
    """预计算一组边界面各自 Flux Points 的物理坐标。

    Flux Points 几何（fr/face_flux_points.py::FaceFluxPointGeometry）本身
    不存储物理坐标（只存插值矩阵/法向/面积权重，见 G-01 数值审计发现），
    这里用同一个外插矩阵直接作用在 `mesh.sps_coords`（SPs 的物理坐标场）
    上——外插算子是线性的，对坐标分量和对流场分量做外插是同一个矩阵
    运算，不需要另外实现一套"参考坐标 -> 物理坐标"映射。
    """
    mesh = solver.mesh
    ops = solver.ops
    positions: Dict[int, np.ndarray] = {}

    for f in np.nonzero(is_target_face)[0]:
        ffp = mesh.face_flux_points[f]
        if not ffp.owner_is_primary:
            continue
        owner_cell = int(face_conn.owner_cell[f])
        axis, side = ffp.owner_axis, ffp.owner_side
        oc_code = int(face_conn.owner_cube_face[f])
        # 真实 bug 修复（2026-09-03，同一处见 postprocess/fr_coefficients.py::
        # extrap_to_face 文档）：四面体坍缩坐标基已删除，`axis`
        # （`ffp.owner_axis`）对 native 四面体面存的是复用槽位的
        # excluded_vertex（可达 3），不能无条件拿去索引占位全零的
        # `ops.boundary_extrap_tet` 字典——按 `oc_code>=6` 分派到
        # `ops.boundary_extrap_native_tet[excluded_vertex]`（形状
        # (n_fp,n_native)，只对 `mesh.sps_coords` 的真实自由度切片
        # `[:n_native]` 求值）。这条路径只在 LES/DDES 的 INLET SEM 合成
        # 湍流入口用到——若某个 INLET 面恰好被四面体单元拥有会真实触发。
        if oc_code >= 6:
            excluded_vertex = oc_code - 6
            E = ops.boundary_extrap_native_tet[excluded_vertex]  # (n_fp, n_native)
            positions[f] = E @ mesh.sps_coords[owner_cell][:E.shape[1]]
        else:
            E = ops.boundary_extrap_prism[(axis, side)]
            positions[f] = E @ mesh.sps_coords[owner_cell]  # (n_fp, 3)

    return positions


def build_boundary_ghost_provider(solver, bc_overrides: Dict[str, Dict[str, Any]]) -> Optional[BoundaryGhostStateProvider]:
    """构建边界幽灵态提供者 (BD-01)。

    Returns:
        BoundaryGhostStateProvider，若网格没有面连接关系（尚未
        load_from_volume_mesh(build_faces=True)）则返回 None
    """
    if solver.mesh.face_connectivity is None:
        logger.warning(
            "Mesh has no face_connectivity - boundary conditions will NOT be enforced "
            "in the residual. This is only acceptable for isolated unit testing, never "
            "for a real solve."
        )
        return None

    rho_inf, vel_inf, p_inf = solver.freestream["rho_inf"], solver.freestream["vel_inf"], solver.freestream["p_inf"]
    # 来流方向由攻角/侧滑角决定（此前这里把它硬编码成 +x，见
    # core/utils/flow_direction.py 模块文档列出的五处硬编码）。
    from autoflowcfd.core.utils.flow_direction import direction_from_freestream
    _dir = direction_from_freestream(solver.freestream)
    _v_free = vel_inf * _dir
    Q_free = [rho_inf, float(_v_free[0]), float(_v_free[1]), float(_v_free[2]), p_inf]

    boundary_groups = solver.mesh.boundary_groups or {}
    bc_types = solver.mesh.boundary_bc_types or {}

    face_conn = solver.mesh.face_connectivity
    group_code, name_to_code = tag_boundary_groups_for_mesh(solver.mesh, face_conn)

    # 真实 bug 修复（#9，V2.0 专家组盲审第4轮，2026-08-28，WMLES 假滑移
    # 边界）：WMLES 激活时，WALL 组此前恒用 is_no_slip=True 构造 ghost
    # state——`wall_ghost_state` 把 ghost 切向速度镜像为与内部值相反
    # （Q_ghost=2*v_wall-Q_int），逼出一个基于（本项目近壁网格通常无法
    # 真正分辨粘性底层的）解析速度梯度的虚假壁面剪应力，然后
    # `solver_helpers.compute_wmles_wall_stress_correction` 又把壁面模型
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
    wall_is_no_slip = getattr(solver, "wmles_model", None) is None
    type_map = {
        "WALL": ("WALL", {"is_no_slip": wall_is_no_slip}),
        "SLIP_WALL": ("WALL", {"is_no_slip": False}),
        "VELOCITY_INLET": ("INLET", {"Q_inlet": Q_free}),
        "PRESSURE_OUTLET": ("OUTLET", {"p_outlet": p_inf}),
        "SYMMETRY": ("SYMMETRY", {}),
    }

    # BD-02：LES/DDES 模式下给 VELOCITY_INLET 组接入合成湍流入口 (SEM)，
    # 取代常量 Q_inlet——算法本身（boundary/synthetic_inlet.py）已在
    # V2.0 二次评审时修复正确（真实入口几何/雷诺应力 Cholesky 分解/
    # 涡核时间演化），但此前从未被任何边界路径调用。solver._sem_instances
    # 供 FRSolver.step() 每个物理步调用一次 advance()（见该列表的
    # 消费点），不在这里（构造阶段）就调用，因为这里只跑一次。
    #
    # WMLES 显式排除（V2.0 专家组盲审复核，2026-08-28，行为本身不变，
    # 仅补充说明）：WMLES 依赖壁面模型（对数律/Spalding 律）而非入口湍流
    # 结构本身来正确预测近壁应力——`wmles_model` 非 None 时即使
    # `turb_model_name` 恰好也是 "LES"/"DDES"（WMLES 是叠加在其上的近壁
    # 处理，不是独立的第三种 turb_model_name 取值），也不注入 SEM 入口
    # 脉动，理由是 WMLES 本身的壁面应力建模已经承担了近壁湍流校正的角色，
    # 额外叠加 SEM 入口脉动不是这个方案设计要解决的问题。CLI
    # `--turbulence-model` 帮助文本（见 solve_transient_command.py）已
    # 补充说明这条排除规则，避免用户误以为 wmles 模式也会获得 SEM 入口
    # 湍流。
    use_sem = solver.turb_model_name in ("LES", "DDES", "IDDES") and getattr(solver, "wmles_model", None) is None
    solver._sem_instances = []

    code_to_config: Dict[int, Dict[str, Any]] = {}
    for name, code in name_to_code.items():
        override = bc_overrides.get(name)
        if override is not None:
            code_to_config[code] = override
            continue
        raw_type = bc_types.get(name, "FARFIELD")
        mapped_type, default_params = type_map.get(raw_type, ("FARFIELD", {"Q_free": Q_free}))
        config = {"type": mapped_type, **default_params}

        if mapped_type == "INLET" and use_sem:
            from autoflowcfd.boundary.synthetic_inlet import SyntheticEddyMethod

            is_this_group_face = group_code == code
            positions_by_face = _compute_inlet_fp_positions(solver, face_conn, is_this_group_face)
            if positions_by_face:
                all_positions = np.concatenate(list(positions_by_face.values()), axis=0)
                # SEM 入口的平动方向同样按真实来流方向（此前硬编码 +x）
                flow_direction = _v_free.copy()
                # length_scale：入口面法向尺度的量级（用坐标散布估计），
                # 太小涡核影响区退化、太大失去局部湍流结构，取入口面
                # 特征尺度的 1/10 是标准 SEM 实践的经验起点。
                span = np.max(all_positions, axis=0) - np.min(all_positions, axis=0)
                length_scale = max(float(np.max(span)) / 10.0, 1e-3)

                num_eddies = getattr(solver, "_sem_num_eddies", _SEM_DEFAULT_NUM_EDDIES)
                sem = SyntheticEddyMethod(
                    num_eddies=num_eddies, length_scale=length_scale
                )
                sem.configure_inlet_box(all_positions, flow_direction=flow_direction)

                turbulence_intensity = getattr(
                    solver, "_turbulence_intensity", _SEM_DEFAULT_TURBULENCE_INTENSITY
                )
                u_fluct = turbulence_intensity * vel_inf
                reynolds_stress = np.diag([u_fluct**2, u_fluct**2, u_fluct**2])

                sem_ghost = InletSEMGhostState(
                    sem, positions_by_face, Q_mean=np.array(Q_free), reynolds_stress=reynolds_stress
                )
                config["sem"] = sem_ghost
                solver._sem_instances.append(sem)
                logger.info(
                    f"BD-02: Synthetic Eddy Method inlet turbulence enabled for group '{name}' "
                    f"({len(positions_by_face)} faces, {num_eddies} eddies, "
                    f"length_scale={length_scale:.4g}, u'={u_fluct:.3g} m/s, "
                    f"Tu={turbulence_intensity:.1%})"
                )

        code_to_config[code] = config

    default_config = {"type": "FARFIELD", "Q_free": Q_free}

    logger.info(
        f"Boundary conditions configured for {len(name_to_code)} group(s): "
        f"{[(name, code_to_config[code]['type']) for name, code in name_to_code.items()]}"
    )

    return BoundaryGhostStateProvider(group_code, code_to_config, default_config)


#: BJ 越界判据里"没有 Dirichlet 值"的标记。用 NaN 而不是哨兵数值：任何
#: 有限哨兵都可能与真实边界值撞车，而 NaN 在 `xp.isfinite` 下是无歧义的。
_NO_DIRICHLET = float("nan")


def build_boundary_dirichlet_table(provider, n_faces: int, n_var: int = 5):
    """给 BJ 越界判据构造逐边界面的**物理边界值**表，(n_faces, n_var)。

    为什么需要它（完整推导与实测见
    `fr_operators/bounds_sensor.py::compute_bounds_violation_mask` 的
    `bnd_dirichlet` 参数文档）：边界面不提供"邻居单元均值"，单纯排除会让
    该单元在那个方向上的包络变成**单侧**的，从而破坏判据的设计不变量
    "线性场恒不触发"。后果是贴壁单元被结构性误判——实测贴壁层命中率
    14.393%（全域只有 0.400%），壁面剪应力被压掉 14 倍
    （Blasius 平板 du/dy 93.0 vs `off` 的 1312.7，cf -93.37% vs -6.33%）。

    单邻居情形下仅凭单元均值**无法**区分"陡峭单调"与"本单元是异常值"
    （两条替代方案已被实测否掉，见那边文档），必须引入边界条件这一份
    外部信息。本函数就是那份信息。

    ## 各边界类型给什么

    * **无滑移壁面（WALL 且 `is_no_slip`）**：动量三列给 `0.0`。静止壁
      上 `rho*u_wall = 0`，与密度无关，所以这一项是精确的、不需要知道
      壁面密度。密度与总能没有 Dirichlet 值（本项目的壁面都是绝热壁，
      没有等温壁这种 BC 类型），留 NaN。
      这一档正是上面那个缺陷的全部来源：边界层的动量剖面单调终止于壁面。
    * **其余全部类型留 NaN**（退回排除，与修复前逐位一致）。逐项理由：
        - `SYMMETRY` / 滑移壁：镜像邻居的均值对标量与切向分量**恰好
          等于**本单元均值，而"排除"对包络初值 `cell_mean` 是无操作
          ——所以排除在这里就是精确的；法向分量的真实镜像均值是 `-m_n`，
          而对称面上 `m_n` 物理上为零，两者一致。
        - `INLET` / `FARFIELD`：来流场在边界附近是均匀的，没有终止于
          边界的强梯度，单侧包络不构成约束。SEM 合成湍流入口逐面逐步
          变化，不存在可预先制表的定值。
        - `OUTLET`：只给定静压，动量无 Dirichlet；且出口处剖面的法向
          邻居都在内部，不存在单侧问题。
    * **移动壁面**（`wall_velocity` 存在且**有非零分量**；静止壁在配置里
      写作 `[0,0,0]`，按静止处理）：留 NaN 并打一次警告。
      `rho*u_wall` 需要壁面密度，而这里拿不到逐面密度；给个错的值比
      退回排除更糟。本项目目前没有移动壁算例。

    Args:
        provider: `build_boundary_ghost_provider` 的返回值。读它的
            `group_code`（(n_faces,) 每面的边界组编码，内部面为 -1）与
            `code_to_config`/`default_config`。**分布式路径无需特殊处理**：
            那几条路径已经把 `provider.group_code` 重切到
            `partition.local_faces`（见 `core/mpi/distributed_mesh_loader.py`
            与 `distributed_order_continuation.py`），与 `dist_flat_face`
            同一索引空间。
        n_faces: 面数，必须与判据里 `owner_cell` 的长度一致
        n_var: 变量数（守恒变量，通常 5）

    Returns:
        (n_faces, n_var) float64；`provider` 为 None 或没有 `group_code`
        时返回 None（调用方据此退回"排除"行为）。

    Raises:
        ValueError: `provider.group_code` 长度与 `n_faces` 不符 —— 索引
            空间对不上时静默继续会让整张表错位，那是一个看不出来的错误。
    """
    if provider is None:
        return None
    group_code = getattr(provider, "group_code", None)
    if group_code is None:
        return None
    gc = np.asarray(group_code)
    if gc.size != n_faces:
        raise ValueError(
            f"boundary_ghost_provider.group_code 长度 {gc.size} 与 n_faces "
            f"{n_faces} 不符——两者必须处在同一个面索引空间，否则整张"
            f"Dirichlet 表会错位"
        )
    if n_var < 4:
        raise ValueError(f"n_var={n_var} 至少要覆盖 3 个动量分量")

    table = np.full((n_faces, n_var), _NO_DIRICHLET, dtype=np.float64)
    code_to_config = getattr(provider, "code_to_config", {}) or {}
    default_config = getattr(provider, "default_config", None) or {}

    moving_wall_seen = False
    codes = np.unique(gc)
    for code in codes:
        cfg = code_to_config.get(int(code), default_config)
        if not cfg or cfg.get("type") != "WALL":
            continue
        if not cfg.get("is_no_slip", True):
            # 滑移壁：与 SYMMETRY 同一条理由，排除即精确。
            continue
        # 静止壁在配置里写作 `wall_velocity=[0,0,0]` 而不是 None
        # （见 build_boundary_ghost_provider 的 type_map），所以判据是
        # "全零即静止"，不能写成 `is not None`——那会把所有真实算例的
        # 壁面都当成移动壁跳过（2026-09-18 实测踩到：表里 5 列全是 NaN，
        # 贴壁层命中率毫无变化）。
        wv = cfg.get("wall_velocity")
        if wv is not None and np.any(np.asarray(wv, dtype=np.float64) != 0.0):
            moving_wall_seen = True
            continue
        table[gc == code, 1:4] = 0.0

    if moving_wall_seen:
        logger.warning(
            "[BJ 判据] 检测到给定了 wall_velocity 的移动壁面。构造动量的"
            "Dirichlet 值需要逐面壁面密度（rho*u_wall），这里拿不到，"
            "因此这些面退回'排除'——那会让贴壁单元的邻域包络在壁面方向上"
            "单侧收窄、可能被误判（静止壁的定量后果是壁面剪应力被压掉 "
            "14 倍）。若本算例确实有移动壁且用到 sensor+bounds 门控，"
            "需要先把逐面壁面密度接进来。"
        )
    return table
