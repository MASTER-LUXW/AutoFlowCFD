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
    """速度入口幽灵态：法向流入时用指定入口状态，流出（回流）时用内部状态延拓。"""
    vel_n = np.sum(Q_int[:, 1:4] * normal, axis=1)
    inflow = vel_n < 0.0  # 法向量指向域外，流入时法向速度为负
    Q_ghost = np.where(inflow[:, np.newaxis], np.asarray(Q_inlet)[np.newaxis, :], Q_int)
    return Q_ghost


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
            return inlet_ghost_state(Q_owner_fp, cfg["Q_inlet"], true_normal)
        elif bc_type == "OUTLET":
            return outlet_ghost_state(Q_owner_fp, cfg["p_outlet"], true_normal)
        elif bc_type == "SYMMETRY":
            return symmetry_ghost_state(Q_owner_fp, true_normal)
        else:
            raise ValueError(f"Unknown boundary condition type '{bc_type}' for face {face_idx}")
