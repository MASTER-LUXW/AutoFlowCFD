"""
AutoFlowCFD V2.0 - FR 边界幽灵态构造 (Tier-0 重建版, 对应 BD-01)

规范文档（3_系统实现方式-算法流程.md §2.2）明确要求边界处理"构造幽灵状态
(Ghost State) 参与黎曼求解"，而不是简单罚项。本模块给出与
core/fr_residual_inviscid.py 的 Riemann 求解流程直接对接的幽灵态构造
函数——所有函数统一在**原始变量** Q=(rho,u,v,w,p) 上工作（不是守恒变量），
输出的幽灵态直接喂给 AUSM+up 黎曼求解器，取代旧版本 fr_solver.py 里从未
被调用的 penalty-only FRWeakBC 路径（BD-01 此前完全没有接入求解主循环，
是本次评审报告里最高优先级的问题之一）。

修复记录：boundary/fr_weak_bc.py 的 compute_wall_bc_flux 在同一个函数内
对 u_int 的含义前后不一致——无滑移分支把 u_int[:,1:4] 当速度直接置零，
但紧接着的滑移分支却用 u_int[:,1:4]/u_int[:,0:1] 当动量除密度处理——
两个分支对"u_int 是原始变量还是守恒变量"的假设自相矛盾。本模块统一在
原始变量上实现，不存在这个问题。
"""

from typing import Optional

import numpy as np

GAMMA = 1.4


def wall_ghost_state(
    Q_int: np.ndarray, normal: np.ndarray, is_no_slip: bool = True, wall_velocity: Optional[np.ndarray] = None
) -> np.ndarray:
    """无滑移/滑移壁面幽灵态。

    Args:
        Q_int: (n_fp, 5) 内部外插原始变量 (rho,u,v,w,p)
        normal: (n_fp, 3) 单位外法向量
        is_no_slip: True=无滑移（速度=壁面速度，默认静止), False=滑移（法向速度为零，切向不变）
        wall_velocity: (3,) 壁面速度，None 时视为静止壁面

    Returns:
        Q_ghost: (n_fp, 5)

    Note（#9 修复，2026-08-28）：is_no_slip=False 这条分支被两种不同物理
    场景复用——真正的无粘/对称 SLIP_WALL 边界，以及 WMLES 激活时的粘性
    WALL 边界（见 fr_solver/boundary.py::build_boundary_ghost_provider）。
    后一种场景下，"切向速度保持不变"不是在建模真实滑移，而是刻意让这个
    面对 BR1/LDG 粘性通量的切向梯度贡献退化为零（ghost 切向速度与内部值
    无跳跃），使壁面模型另外算出的 tau_w
    （core/utils/solver_helpers.py::compute_wmles_wall_stress_correction）
    成为该面切向应力的唯一来源，而不是叠加在一个由无滑移镜像逼出的虚假
    解析梯度剪应力之上——按 WMLES 壁面模型文献的标准做法（wall model 的
    tau_w 应该"取代"而非"叠加"解析梯度算出的剪应力，见
    build_boundary_ghost_provider 文档引用的 Kawai & Larsson 团队页面与
    Kang et al. 2024 arXiv:2405.15899）。

    **措辞更正（2026-09-14）**：此前这里与 `build_boundary_ghost_provider`
    都把两种场景共享同一构造称为"刻意的工程简化/复用"。"简化"不准确——
    这两种场景各自的**正确** ghost 态恰好就是同一个：
      * 滑移/对称壁：物理条件是 u_n=0 + 切向自由滑移 -> 法向镜像反号、
        切向保持，这是该条件的精确 ghost 构造；
      * WMLES 粘性壁：要求 tau_w 取代而非叠加解析剪应力 -> 切向必须无
        跳跃（梯度贡献为零）、法向仍需不可穿透 -> 同样是法向镜像反号、
        切向保持。
    两者是"同一个精确构造被两个不同物理条件同时要求"，不是用一个近似
    去凑合两种情形。复用它没有引入任何误差。

    **热边界条件（此前只隐含在代码里，2026-09-14 显式写出）**：本函数
    以 `Q_ghost = Q_int.copy()` 起手、只改速度分量，因此 rho 与 p（进而
    温度）都直接复制内部值——**两个分支都是零温度跳跃，即绝热壁**。
    对无滑移粘性壁，绝热是标准且自洽的选择（BR1/LDG 界面项因此不产生
    壁面热通量）；对滑移/对称壁，零热通量本来就是对称条件的要求。
    需要注意的是：本项目目前**没有等温壁**（指定壁温）这种边界类型，
    WMLES 也只给出 tau_w、没有壁面热通量模型——带传热的壁面工况需要
    另外新增 BC 类型与壁面热通量闭合，那是独立的功能项，不是本构造的
    近似。这里把它写明，避免读者误以为绝热是偶然结果。
    """
    Q_ghost = Q_int.copy()
    if is_no_slip:
        v_wall = np.zeros(3) if wall_velocity is None else np.asarray(wall_velocity)
        # 镜像构造：幽灵态速度 = 2*v_wall - 内部速度，使得黎曼求解器在界面上
        # （L、R 平均意义下）恢复出恰好等于 v_wall 的速度，是标准的无滑移
        # 幽灵单元构造方式（避免直接令幽灵态=v_wall 时界面平均速度实际上是
        # (v_int+v_wall)/2 而非 v_wall 本身的偏差）。
        Q_ghost[:, 1:4] = 2.0 * v_wall[np.newaxis, :] - Q_int[:, 1:4]
    else:
        vel = Q_int[:, 1:4]
        vel_n = np.sum(vel * normal, axis=1, keepdims=True)
        # 滑移壁面：法向速度镜像反号（不可穿透），切向速度保持
        Q_ghost[:, 1:4] = vel - 2.0 * vel_n * normal
    return Q_ghost


def farfield_ghost_state(Q_int: np.ndarray, Q_free: np.ndarray) -> np.ndarray:
    """远场边界幽灵态：直接取自由来流状态。

    与旧版本的罚项公式不同，这里把 Q_free 作为幽灵态送入 AUSM+up 黎曼
    求解器（而不是直接惩罚差值），黎曼求解器自身的特征分裂天然提供了
    亚声速远场所需的"扰动波可以传出、来流可以传入"的迎风行为，物理上
    比固定惩罚系数的强 Dirichlet 更合理。
    """
    return np.tile(np.asarray(Q_free), (Q_int.shape[0], 1))


def inlet_ghost_state(Q_int: np.ndarray, Q_inlet: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """速度入口幽灵态：法向流入时用指定入口状态，流出（回流）时用内部状态延拓。

    Args:
        Q_inlet: 形状 (5,)（全入口面统一常量，原有行为）或 (n_fp,5)
            （逐 Flux Point 不同——BD-02 合成湍流入口 SEM 用这个形式喂入
            随时间演化、逐点不同的脉动速度，见 boundary/synthetic_inlet.py
            与本文件 InletSEMGhostState）。
    """
    vel_n = np.sum(Q_int[:, 1:4] * normal, axis=1)
    inflow = vel_n < 0.0  # 法向量指向域外，流入时法向速度为负
    Q_inlet_arr = np.asarray(Q_inlet)
    Q_inlet_broadcast = Q_inlet_arr[np.newaxis, :] if Q_inlet_arr.ndim == 1 else Q_inlet_arr
    Q_ghost = np.where(inflow[:, np.newaxis], Q_inlet_broadcast, Q_int)
    return Q_ghost


class InletSEMGhostState:
    """给一组 INLET 边界面提供随时间演化、满足目标雷诺应力的合成湍流
    入口脉动幽灵态（BD-02）。

    与旧版本"SEM 算法本身正确但从未被任何边界路径调用"（V2.0 二次评审
    发现）不同，这个类是 SyntheticEddyMethod 与
    core/fr_residual_inviscid.py 期望的 boundary_ghost_provider 接口
    （(face_idx, Q_owner_fp, true_normal) -> Q_ghost）之间的真实粘合层，
    由 core/fr_solver_boundary.py::build_boundary_ghost_provider 在
    LES/DDES 模式下为 VELOCITY_INLET 组构造并接入。

    涡核的时间演化（advance）由 FRSolver.step() 每个物理步调用一次
    （不是每次残差求值都调用——RK 子迭代内多次调用 __call__ 复用同一批
    涡核位置，只有 generate_fluctuations 按位置重新求值，物理上等价于
    "本物理步内脉动场冻结、下一物理步涡核才对流"，是标准做法）。
    """

    def __init__(self, sem, face_positions: dict, Q_mean: np.ndarray, reynolds_stress: np.ndarray):
        self.sem = sem
        self.face_positions = face_positions  # face_idx -> (n_fp,3) 物理坐标
        self.Q_mean = np.asarray(Q_mean, dtype=float)
        self.reynolds_stress = reynolds_stress

    def __call__(self, face_idx: int, Q_owner_fp: np.ndarray, true_normal: np.ndarray) -> np.ndarray:
        positions = self.face_positions.get(face_idx)
        if positions is None:
            # 该面不在预先构造涡核影响区时枚举到的 INLET 面集合里（理论上
            # 不应发生，因为 face_positions 是按同一批 INLET 面构造的），
            # 兜底退回常量均值状态而不是崩溃。
            return np.tile(self.Q_mean, (Q_owner_fp.shape[0], 1))

        mean_u = self.Q_mean[1:4]
        u_inst = self.sem.generate_fluctuations(positions, mean_u, self.reynolds_stress)  # (n_fp,3)

        n_fp = positions.shape[0]
        Q_inlet_fp = np.tile(self.Q_mean, (n_fp, 1))
        Q_inlet_fp[:, 1:4] = u_inst

        return inlet_ghost_state(Q_owner_fp, Q_inlet_fp, true_normal)


def outlet_ghost_state(Q_int: np.ndarray, p_outlet: float, normal: np.ndarray) -> np.ndarray:
    """压力出口幽灵态：法向流出时固定静压、其余变量延拓；回流时用内部状态。"""
    vel_n = np.sum(Q_int[:, 1:4] * normal, axis=1)
    outflow = vel_n > 0.0

    Q_ghost = Q_int.copy()
    Q_ghost[:, 4] = np.where(outflow, p_outlet, Q_int[:, 4])
    return Q_ghost


def symmetry_ghost_state(Q_int: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """对称面幽灵态：法向速度反号镜像，切向速度、密度、压力延拓。"""
    Q_ghost = Q_int.copy()
    vel = Q_int[:, 1:4]
    vel_n = np.sum(vel * normal, axis=1, keepdims=True)
    Q_ghost[:, 1:4] = vel - 2.0 * vel_n * normal
    return Q_ghost


class BoundaryGhostStateProvider:
    """把「面 -> 边界组 -> BC 类型/参数」与幽灵态构造函数粘合起来，产出
    core/fr_residual_inviscid.compute_inviscid_residual_fr 需要的
    boundary_ghost_provider 可调用对象。

    Attributes:
        group_code: (n_faces,) 每个边界面所属边界组的整数编码
            （grid/face_connectivity.tag_boundary_groups 的输出），内部面/
            未匹配任何组的边界面为 -1
        code_to_config: Dict[int, dict]，组编码 -> {'type': 'WALL'/'FARFIELD'/
            'INLET'/'OUTLET'/'SYMMETRY', 以及该类型需要的参数}
        default_config: 未匹配到任何边界组的边界面使用的兜底配置
            （不允许静默地"什么都不做"——必须显式提供，通常设为 FARFIELD
            自由来流条件，工业外流场里未分类面多为远场边界）
    """

    def __init__(self, group_code: np.ndarray, code_to_config: dict, default_config: dict):
        self.group_code = group_code
        self.code_to_config = code_to_config
        self.default_config = default_config

    def __call__(self, face_idx: int, Q_owner_fp: np.ndarray, true_normal: np.ndarray) -> np.ndarray:
        code = int(self.group_code[face_idx])
        cfg = self.code_to_config.get(code, self.default_config)
        bc_type = cfg["type"]

        if bc_type == "WALL":
            return wall_ghost_state(
                Q_owner_fp,
                true_normal,
                is_no_slip=cfg.get("is_no_slip", True),
                wall_velocity=cfg.get("wall_velocity"),
            )
        elif bc_type == "FARFIELD":
            return farfield_ghost_state(Q_owner_fp, cfg["Q_free"])
        elif bc_type == "INLET":
            sem_ghost = cfg.get("sem")
            if sem_ghost is not None:
                return sem_ghost(face_idx, Q_owner_fp, true_normal)
            return inlet_ghost_state(Q_owner_fp, cfg["Q_inlet"], true_normal)
        elif bc_type == "OUTLET":
            return outlet_ghost_state(Q_owner_fp, cfg["p_outlet"], true_normal)
        elif bc_type == "SYMMETRY":
            return symmetry_ghost_state(Q_owner_fp, true_normal)
        else:
            raise ValueError(f"Unknown boundary condition type '{bc_type}' for face {face_idx}")
