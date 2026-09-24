"""AutoFlowCFD V2.0 - 边界幽灵态 provider（含 SEM 入口 FP 位置）

从 `src/autoflowcfd/core/fr_solver/boundary.py`(原 506 行)拆出(2026-09-24, 项目"单文件不超 500 行"规范)。**纯搬家, 逻辑未改**。
"""

from typing import Any, Dict, Optional

import numpy as np

from loguru import logger

from autoflowcfd.boundary.fr_ghost_state import BoundaryGhostStateProvider, InletSEMGhostState

from autoflowcfd.grid.connectivity.face_connectivity import tag_boundary_groups_for_mesh
from .constants import _SEM_DEFAULT_NUM_EDDIES, _SEM_DEFAULT_TURBULENCE_INTENSITY


def _compute_inlet_fp_positions(solver, face_conn, is_target_face: np.ndarray) -> Dict[int, np.ndarray]:
    """预计算一组边界面各自 Flux Points 的物理坐标。

    Flux Points 几何（fr/face_flux_points/geometry.py::FaceFluxPointGeometry）本身
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
        oc_code = int(face_conn.owner_cube_face[f])
        # 原生面（四面体 [6,10) / 棱柱 [10,15)）统一走
        # `ops.native_face_extrap`，不在这里自己按 code-6 取表 ——
        # 两类单元的键与 n_native 都不同，配错不会报错、只会静默用错
        # 矩阵（见 FROperators 里那段说明）。只对 `mesh.sps_coords` 的
        # 真实自由度切片 `[:n_native]` 求值。这条路径只在 LES/DDES 的
        # INLET SEM 合成湍流入口用到。
        E = ops.native_face_extrap(oc_code)  # (n_fp, n_native)
        positions[f] = E @ mesh.sps_coords[owner_cell][:E.shape[1]]

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
    # **`is_no_slip` 与 `physical_no_slip` 是两件事（2026-09-18 拆开）**：
    #   `is_no_slip`       —— **幽灵态构造开关**。WMLES 激活时它被关掉，
    #                         为的是不与壁面模型的应力修正双重计权
    #                         （完整论证见上方与 wall_ghost_state 文档）。
    #   `physical_no_slip` —— **物理事实**：这面墙是不是静止的不可穿透
    #                         固壁。WMLES 并不改变这一点。
    #
    # 为什么必须拆：此前两者共用一个字段，于是 `WALL`（物理无滑移）与
    # `SLIP_WALL`（物理滑移）在 WMLES 下**都**变成 `is_no_slip=False`，
    # 下游无从区分。`build_boundary_dirichlet_table` 正是按它判断"壁面
    # 动量是否为零"，结果是 **WMLES 算例整张 Dirichlet 表全是 NaN** ——
    # BJ 越界判据在贴壁层的结构性误判原封不动回来，而 WMLES 恰恰是最
    # 关心壁面剪应力的那一类算例（该误判的实测代价是 cf -93%）。
    wall_is_no_slip = getattr(solver, "wmles_model", None) is None
    type_map = {
        "WALL": ("WALL", {"is_no_slip": wall_is_no_slip,
                          "physical_no_slip": True}),
        "SLIP_WALL": ("WALL", {"is_no_slip": False,
                               "physical_no_slip": False}),
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
