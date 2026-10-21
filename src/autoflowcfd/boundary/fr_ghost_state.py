"""
AutoFlowCFD V2.0 - FR 边界幽灵态构造（BD-01）

边界处理按规范（3_系统实现方式-算法流程.md §2.2）"构造幽灵状态 (Ghost State) 参与黎曼求解"，而不是罚项。
本模块的幽灵态构造函数与 `core/fr_residual/inviscid.py` 的界面 Riemann 求解直接对接：所有函数统一在
**原始变量** Q=(rho,u,v,w,p) 上工作（不是守恒变量），输出的幽灵态直接喂给 AUSM+up 黎曼求解器。
"""

from typing import Optional

import numpy as np

from autoflowcfd.core.utils.array_module import array_module, to_host

# 幽灵态公式按 `array_module` 取数组模块：CPU 残差传 numpy、GPU 残差传 CuPy，同一份公式（2026-10-19，I-119：此前
# GPU 残差每次求值把整份原始变量拷回主机算幽灵态、再把全部面大小的数组传回设备）。


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
    WALL 边界（见 fr_solver/boundary/ghost.py::build_boundary_ghost_provider）。
    后一种场景下，"切向速度保持不变"不是在建模真实滑移，而是刻意让这个
    面对 BR1/LDG 粘性通量的切向梯度贡献退化为零（ghost 切向速度与内部值
    无跳跃），使壁面模型另外算出的 tau_w
    （core/turbulence/wall_model/，只作用于物理无滑移壁面）
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

    **2026-09-30 更正**：幽灵态切向无跳跃只让罚项为零；粘性公共通量里的
    切向牵引还取决于"另一侧"梯度。此前梯度一律取本侧，公共切向牵引就是
    解析梯度算出的那一份、与 `tau_w` 叠加 —— 上面"梯度贡献为零"并不成立。
    现在滑移分支落到镜像类粘性边界（`viscous_boundary_kind_for_config`），
    镜像场梯度使公共切向牵引恰为零，这句话才真正成立。

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
    xp = array_module(Q_int, normal)
    Q_ghost = Q_int.copy()
    if is_no_slip:
        v_wall = xp.zeros(3) if wall_velocity is None else xp.asarray(wall_velocity, dtype=Q_int.dtype)
        # 镜像构造：幽灵态速度 = 2*v_wall - 内部速度，使得黎曼求解器在界面上
        # （L、R 平均意义下）恢复出恰好等于 v_wall 的速度，是标准的无滑移
        # 幽灵单元构造方式（避免直接令幽灵态=v_wall 时界面平均速度实际上是
        # (v_int+v_wall)/2 而非 v_wall 本身的偏差）。
        Q_ghost[:, 1:4] = 2.0 * v_wall[None, :] - Q_int[:, 1:4]
    else:
        vel = Q_int[:, 1:4]
        vel_n = xp.sum(vel * normal, axis=1, keepdims=True)
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
    xp = array_module(Q_int)
    return xp.tile(xp.asarray(Q_free, dtype=Q_int.dtype), (Q_int.shape[0], 1))


def inlet_ghost_state(Q_int: np.ndarray, Q_inlet: np.ndarray) -> np.ndarray:
    """速度入口幽灵态：给定的入口状态，与本侧流向无关（哪些分量真正起作用由黎曼求解器的迎风决定，
    与远场同一机制）。

    2026-10-10 以前"本侧法向流出时改用内部状态延拓"。那是一个**间断**的切换：入口与无滑移壁相交处
    本侧法向速度在零附近，幽灵态在"入口状态（来流速度）"与"内部状态"之间来回跳，残差在该点不连续。
    实测（通道 4x4x2 棱柱 P1，均匀来流 30 m/s + 无滑移壁，BDF1 一个物理步 dt=1e-4，稠密差分 Jacobian
    的精确 Newton）：入口用切换时 6 次迭代 ||G|| 停在 3e-3~8e-3 来回跳，Jacobian 在迭代间时有时无地
    出现 -1.4e3 1/s 的特征值；不切换时 7.7e-2 -> 7.3e-7 -> 8e-11 二次收敛。回流点上延拓内部状态本身
    也不适定（没有给任何条件）。

    Args:
        Q_inlet: 形状 (5,)（全入口面统一常量）或 (n_fp,5)（逐通量点不同——合成湍流入口 SEM 喂入
            随时间演化的脉动速度，见 boundary/synthetic_inlet.py 与本文件 InletSEMGhostState）。
    """
    xp = array_module(Q_int)
    Q_inlet_arr = xp.asarray(Q_inlet, dtype=Q_int.dtype)
    if Q_inlet_arr.ndim == 1:
        return xp.tile(Q_inlet_arr, (Q_int.shape[0], 1))
    return Q_inlet_arr.copy()


class InletSEMGhostState:
    """一组 INLET 边界面的合成湍流入口幽灵态（`boundary/synthetic_inlet.py`）：平均来流叠加 SEM 速度脉动。

    由 `core/fr_solver/boundary/ghost.py::build_boundary_ghost_provider` 在 LES/DDES/IDDES 模式下为速度入口组构造。
    涡核每个物理步推进一次（`advance_synthetic_inlets`）；本物理步内（RK 子步、隐式内迭代）脉动场冻结。脉动在
    涡核推进后第一次被取用时对本对象的全部入口点一次性求值、缓存，按面切片取用。
    """

    def __init__(self, sem, face_positions: dict, face_sigma: dict, Q_mean: np.ndarray, reynolds_stress: np.ndarray):
        from autoflowcfd.boundary.synthetic_inlet import reynolds_stress_factor

        self.sem = sem
        self.face_positions = face_positions  # 面号 -> (n_fp, 3) 物理坐标
        self.face_sigma = face_sigma          # 面号 -> (n_fp,) 涡核尺度
        self.Q_mean = np.asarray(Q_mean, dtype=float)
        self.reynolds_stress = np.asarray(reynolds_stress, dtype=float)
        self._a_t = reynolds_stress_factor(self.reynolds_stress).T
        faces = list(face_positions)
        self._row = {f: k for k, f in enumerate(faces)}
        self._points = (np.stack([face_positions[f] for f in faces]) if faces else np.zeros((0, 1, 3)))
        self._sigma = (np.stack([face_sigma[f] for f in faces]) if faces else np.zeros((0, 1)))
        self._version: Optional[int] = None
        self._velocity: Optional[np.ndarray] = None   # (n_faces, n_fp, 3) 瞬时速度，按涡核版本缓存

    def restricted_to_faces(self, face_ids: np.ndarray) -> "InletSEMGhostState":
        """面号换成 `face_ids` 的下标（分布式：`face_ids` 是本 rank 面序对应的全局面号）；涡核对象共享
        （每个物理步推进一次，见 `synthetic_inlet.py::advance_synthetic_inlets`）。"""
        pairs = [(int(i), int(g)) for i, g in enumerate(np.asarray(face_ids)) if int(g) in self.face_positions]
        return InletSEMGhostState(self.sem, {i: self.face_positions[g] for i, g in pairs},
                                  {i: self.face_sigma[g] for i, g in pairs}, self.Q_mean, self.reynolds_stress)

    def _instantaneous_velocity(self) -> np.ndarray:
        velocity = self._velocity
        if velocity is None or self._version != self.sem.version:
            n_faces, n_fp = self._sigma.shape
            raw = self.sem.unit_fluctuations(self._points.reshape(-1, 3), self._sigma.reshape(-1))
            velocity = (self.Q_mean[1:4] + raw @ self._a_t).reshape(n_faces, n_fp, 3)
            self._velocity, self._version = velocity, self.sem.version
        return velocity

    def ghost_states(self, faces: np.ndarray, Q_owner_fp: np.ndarray) -> np.ndarray:
        """`(n, n_fp, 5)`：一批面（本对象面号空间）的入口幽灵态。"""
        faces = np.asarray(faces, dtype=np.int64)
        try:
            rows = np.fromiter((self._row[int(f)] for f in faces), dtype=np.int64, count=faces.size)
        except KeyError as e:
            # 坐标表按该组全部面记录构造（`fr_solver/boundary/ghost.py::_compute_inlet_fp_positions`），查不到
            # 说明面号空间错了（例如分布式用本 rank 面号查全局面号的表）——此前静默返回无脉动的均值态
            raise KeyError(f"SEM 入口面 {e.args[0]} 不在坐标表里（面号空间与构造时不一致）") from None
        Q_inlet = np.broadcast_to(self.Q_mean, Q_owner_fp.shape).copy()
        Q_inlet[..., 1:4] = self._instantaneous_velocity()[rows]
        return inlet_ghost_state(Q_owner_fp.reshape(-1, 5), Q_inlet.reshape(-1, 5)).reshape(Q_owner_fp.shape)

    def __call__(self, face_idx: int, Q_owner_fp: np.ndarray, true_normal: np.ndarray) -> np.ndarray:
        return self.ghost_states(np.array([face_idx]), Q_owner_fp[None])[0]


def outlet_ghost_state(Q_int: np.ndarray, p_outlet: float, normal: np.ndarray) -> np.ndarray:
    """压力出口幽灵态：密度与速度延拓，压力给定——流出时是静压 `p_outlet`，回流时把 `p_outlet` 当总压
    （静压取 `p_outlet - rho u_n^2 / 2`，`u_n` 为本侧法向速度）。

    回流时扣掉动压是工业软件压力出口的通行做法（给定值按总压解释）：回流带进域内的动能由压差提供，
    回流不会自我加速。两支在 `u_n = 0` 处连续且一阶导数连续。

    2026-10-10 以前"回流时幽灵态取内部状态"（压力也延拓），在 `u_n = 0` 处幽灵态压力从 `p_outlet`
    跳到内部压力。出口与无滑移壁相交处本侧法向速度在零附近，残差不连续，Newton 不收敛（同
    `inlet_ghost_state` 的实测：只把出口换成切换写法，6 次精确 Newton 后 ||G|| 仍在 1e-3 量级，
    Jacobian 最小特征值实部在 -8e2 ~ -9e3 1/s 之间跳）。
    """
    xp = array_module(Q_int, normal)
    rho = Q_int[:, 0]
    vel_n = xp.sum(Q_int[:, 1:4] * normal, axis=1)
    backflow = xp.minimum(vel_n, 0.0)

    Q_ghost = Q_int.copy()
    Q_ghost[:, 4] = p_outlet - 0.5 * rho * backflow ** 2
    return Q_ghost


def symmetry_ghost_state(Q_int: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """对称面幽灵态：法向速度反号镜像，切向速度、密度、压力延拓。"""
    xp = array_module(Q_int, normal)
    Q_ghost = Q_int.copy()
    vel = Q_int[:, 1:4]
    vel_n = xp.sum(vel * normal, axis=1, keepdims=True)
    Q_ghost[:, 1:4] = vel - 2.0 * vel_n * normal
    return Q_ghost


def ghost_state_for_config(cfg: dict, Q_int: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """一组边界配置下的幽灵态（不含 SEM 入口，那一支逐面依赖涡核位置，由调用方处理）。

    Args:
        cfg: 边界配置 {'type': 'WALL'/'FARFIELD'/'INLET'/'OUTLET'/'SYMMETRY', 及该类型的参数}
        Q_int: (n, 5) 本侧外插原始变量
        normal: (n, 3) 单位外法向

    这是边界类型 -> 幽灵态构造的唯一分派处（单面与批量调用都经过这里）。
    """
    bc_type = cfg["type"]
    if bc_type == "WALL":
        return wall_ghost_state(Q_int, normal, is_no_slip=cfg.get("is_no_slip", True),
                                wall_velocity=cfg.get("wall_velocity"))
    if bc_type == "FARFIELD":
        return farfield_ghost_state(Q_int, cfg["Q_free"])
    if bc_type == "INLET":
        return inlet_ghost_state(Q_int, cfg["Q_inlet"])
    if bc_type == "OUTLET":
        return outlet_ghost_state(Q_int, cfg["p_outlet"], normal)
    if bc_type == "SYMMETRY":
        return symmetry_ghost_state(Q_int, normal)
    raise ValueError(f"未知的边界条件类型 '{bc_type}'")


class GhostStateProvider:
    """边界幽灵态提供者基类。

    子类实现单面接口 `__call__(face_idx, Q_owner_fp (n_fp,5), true_normal (n_fp,3)) -> (n_fp,5)`；
    残差组装统一调用批量接口 `ghost_states`，默认逐面调用 `__call__`，按边界组批量化的实现覆盖它。
    """

    def __call__(self, face_idx: int, Q_owner_fp: np.ndarray, true_normal: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def ghost_states(self, faces: np.ndarray, Q_owner: np.ndarray, normals: np.ndarray) -> np.ndarray:
        """一批边界面的幽灵态。

        Args:
            faces: (m,) 面号
            Q_owner: (m, n_fp, 5) 各面通量点上的本侧原始变量
            normals: (m, n_fp, 3) 单位外法向
        Returns:
            (m, n_fp, 5)，与 `Q_owner` 同一数组模块（逐面 `__call__` 在主机上执行，设备数组经主机往返——只有
            没有按组批量实现的提供者走这里）
        """
        xp = array_module(Q_owner, normals)
        if xp is not np:
            return xp.asarray(self.ghost_states(to_host(faces), to_host(Q_owner), to_host(normals)))
        out = np.empty_like(Q_owner)
        for k, f in enumerate(faces):
            out[k] = self(int(f), Q_owner[k], normals[k])
        return out


def require_ghost_provider(provider, who: str):
    """残差 / 梯度核心函数的边界幽灵态提供者必须由调用方显式给出，返回它本身（I-5，2026-10-19）。

    此前这些函数收到 None 时静默换成 `DefaultGhostProvider`（零梯度外插）：边界条件对残差不可见，结果看不出
    异常。单 GPU 与多 GPU 都因此出过"全部边界退化成镜像内部值"的真实缺陷（2026-08-27 / 08-28）。零梯度外插
    是合法的选择（自由流守恒测试），但必须由调用方显式传 `DefaultGhostProvider()`。
    """
    if provider is None:
        raise ValueError(f"{who}: 必须显式给出边界幽灵态提供者（生产求解器用 build_boundary_ghost_provider 构建的"
                         f"按边界组给定条件的提供者；零梯度外插须显式传 DefaultGhostProvider()）")
    return provider


class DefaultGhostProvider(GhostStateProvider):
    """零梯度外插（幽灵态 = 本侧外插态）：对均匀流场恰好零跳跃。

    只用于自由流守恒类测试与没有物理边界的单元测试；生产求解器一律用按边界组给定条件的
    `BoundaryGhostStateProvider`（`core/fr_solver/boundary/ghost.py::build_boundary_ghost_provider`）。
    """

    def __call__(self, face_idx: int, Q_owner_fp: np.ndarray, true_normal: np.ndarray) -> np.ndarray:
        return Q_owner_fp.copy()

    def ghost_states(self, faces: np.ndarray, Q_owner: np.ndarray, normals: np.ndarray) -> np.ndarray:
        return Q_owner.copy()


class BoundaryGhostStateProvider(GhostStateProvider):
    """把「面 -> 边界组 -> BC 类型/参数」与幽灵态构造函数粘合起来，供残差组装取边界幽灵态。

    Attributes:
        group_code: (n_faces,) 每个边界面所属边界组的整数编码
            （grid/face_connectivity.tag_boundary_groups 的输出），内部面为 -1；每个边界面都必须属于一个组
            （`core/fr_solver/boundary/ghost.py::build_boundary_ghost_provider` 构造期检查）
        code_to_config: Dict[int, dict]，组编码 -> {'type': 'WALL'/'FARFIELD'/
            'INLET'/'OUTLET'/'SYMMETRY', 以及该类型需要的参数}
        code_to_name: Dict[int, str]，组编码 -> 组名（后处理按组给出分量用；可缺省）

    2026-10-21（I-73）以前另有 `default_config`（未匹配任何组的面取远场）：未分组边界面早在构造期就报错，它只剩
    死兜底，却仍让读 BC 语义的地方（粘性边界种类、湍流开放边界）把内部面也按远场分类。
    """

    def __init__(self, group_code: np.ndarray, code_to_config: dict, code_to_name: Optional[dict] = None):
        self.group_code = group_code
        self.code_to_config = code_to_config
        self.code_to_name = dict(code_to_name or {})

    def config_of_code(self, code: int) -> dict:
        cfg = self.code_to_config.get(int(code))
        if cfg is None:
            raise KeyError(f"边界组编码 {int(code)} 没有配置：边界面都必须属于一个已配置的边界组")
        return cfg

    def restricted_to_faces(self, face_ids: np.ndarray) -> "BoundaryGhostStateProvider":
        """只含 `face_ids` 这些面、面号换成其下标的 provider（分布式：全局 provider -> 本 rank 面序）。

        组编码与按面号查表的 SEM 入口坐标一起换（此前四处分布式构造点只重切 `group_code`，SEM 坐标表仍以
        全局面号为键：本 rank 面号要么查不到、要么查到别处的坐标）。
        """
        face_ids = np.asarray(face_ids, dtype=np.int64)
        configs = {}
        for code, cfg in self.code_to_config.items():
            if cfg.get("sem") is not None:
                cfg = dict(cfg, sem=cfg["sem"].restricted_to_faces(face_ids))
            configs[code] = cfg
        return BoundaryGhostStateProvider(np.asarray(self.group_code)[face_ids], configs,
                                          code_to_name=self.code_to_name)

    def __call__(self, face_idx: int, Q_owner_fp: np.ndarray, true_normal: np.ndarray) -> np.ndarray:
        return self.ghost_states(np.array([face_idx]), Q_owner_fp[np.newaxis], true_normal[np.newaxis])[0]

    def ghost_states(self, faces: np.ndarray, Q_owner: np.ndarray, normals: np.ndarray) -> np.ndarray:
        """按边界组批量构造：同一组共享一份配置，底层幽灵态函数本身按任意长度向量化（SEM 入口按面查缓存的
        瞬时速度，同样整组一次）。
        `faces` 是主机面号；`Q_owner` / `normals` 可在主机或设备上（按 `array_module`），常规边界条件整组在
        同一数组模块上算，SEM 入口（涡核在主机上）只把该组的面经主机往返。
        """
        xp = array_module(Q_owner, normals)
        faces = np.asarray(to_host(faces), dtype=np.int64)
        out = xp.empty_like(Q_owner)
        if faces.size == 0:
            return out
        n_fp = Q_owner.shape[1]
        codes = np.asarray(self.group_code)[faces]
        for code in np.unique(codes):
            sel_host = np.flatnonzero(codes == code)
            sel = sel_host if xp is np else xp.asarray(sel_host)
            cfg = self.config_of_code(code)
            sem_ghost = cfg.get("sem") if cfg["type"] == "INLET" else None
            if sem_ghost is not None:
                out[sel] = xp.asarray(sem_ghost.ghost_states(faces[sel_host], to_host(Q_owner[sel])))
                continue
            out[sel] = ghost_state_for_config(
                cfg, Q_owner[sel].reshape(-1, 5), normals[sel].reshape(-1, 3)).reshape(len(sel_host), n_fp, 5)
        return out


# ---------------------------------------------------------------------------
# 粘性通量的逐面边界种类（2026-09-30，取代 2026-09-15 的"绝热面掩码"）
# ---------------------------------------------------------------------------

#: 封闭边界（壁面/对称面）：没有质量穿越，法向热通量为零（本项目没有等温壁，
#: 见 `wall_ghost_state` 的"热边界条件"一节）。湍流 k/omega 的"开放边界"判据
#: 取它的补集（`core/turbulence/transport/omega_wall.py`）。
ADIABATIC_THERMAL_BC_TYPES = frozenset({"WALL", "SYMMETRY"})


def viscous_boundary_kind_for_config(cfg: dict) -> int:
    """一组边界配置对应的粘性边界种类（`core/fr_operators/flux_kernels/viscous_bc.py`）。

    与 `BoundaryGhostStateProvider.__call__` 读同一套配置键：幽灵态**给定**
    的分量按 Dirichlet 施加，**延拓**内部值的分量取零法向粘性通量，镜像构造
    的边界（对称面、滑移壁含 WMLES 壁面）取镜像场的梯度。逐类理由见
    `viscous_bc.py` 模块文档。

    WMLES 壁面（`is_no_slip=False`）落到镜像类：切向牵引恰为零，壁面模型给的
    `tau_w` 才是该面切向应力的唯一来源。此前"切向无跳跃 ⇒ 梯度贡献为零"的
    说法（见 `wall_ghost_state` 文档）并不成立 —— 梯度取本侧时公共切向牵引
    就是解析梯度算出的那一份，与 `tau_w` 叠加。
    """
    from autoflowcfd.core.fr_operators.flux_kernels import (
        VBC_DIRICHLET, VBC_MIRROR, VBC_NEUMANN, VBC_NOSLIP_WALL,
    )

    bc_type = cfg["type"]
    if bc_type == "WALL":
        return VBC_NOSLIP_WALL if cfg.get("is_no_slip", True) else VBC_MIRROR
    if bc_type == "SYMMETRY":
        return VBC_MIRROR
    if bc_type in ("FARFIELD", "INLET"):
        return VBC_DIRICHLET
    if bc_type == "OUTLET":
        return VBC_NEUMANN
    raise ValueError(f"Unknown boundary condition type '{bc_type}'")


def build_viscous_boundary_kind(n_faces: int, is_boundary: np.ndarray, ghost_provider) -> np.ndarray:
    """逐面粘性边界种类，`(n_faces,)` int8；内部面为 `VBC_INTERIOR`。

    BC 语义按**鸭子类型**读取：provider 只要带 ``group_code``/``code_to_config``
    （``BoundaryGhostStateProvider`` 以及委托给它的包装层，如验证算例的逐点剖面
    入口）就逐组分类，边界面的组没有配置时报错。
    ``DefaultGhostProvider`` 的幽灵态是本侧延拓，按延拓分量的规则取
    ``VBC_NEUMANN``（零法向粘性通量）。其余 provider 没有可读的 BC 语义，
    **直接报错**。

    为什么不能按 ``isinstance`` 判、也不能对未知 provider 兜底：2026-09-30 实测，
    Blasius 算例的 provider 被逐点剖面入口包了一层，``isinstance`` 判不中，全部
    边界面（含无滑移壁）落进兜底分支 —— 兜底取 Neumann 时壁面剪应力整个消失，
    P1/P2 都收敛到 cf 只有精确值 12%~19% 的错解；此前的"绝热面掩码"用同一个
    ``isinstance``，兜底是"全部透射"，于是那个算例的壁面热条件从未真正被施加过，
    只是没有暴露。边界分类错了是静默的物理错误，必须在构造期失败。

    结果按 provider 实例缓存（配置在构造后不再变化），避免每次残差求值重复
    扫描边界面。
    """
    from autoflowcfd.core.fr_operators.flux_kernels import VBC_INTERIOR, VBC_NEUMANN

    cached = getattr(ghost_provider, "_afcfd_vbc_kind", None)
    if cached is not None and cached.shape[0] == n_faces:
        return cached

    kind = np.full(n_faces, VBC_INTERIOR, dtype=np.int8)
    bnd = np.where(is_boundary)[0]
    group_code = getattr(ghost_provider, "group_code", None)
    code_to_config = getattr(ghost_provider, "code_to_config", None)
    if group_code is not None and code_to_config is not None:
        codes = np.asarray(group_code)[bnd]
        for code in np.unique(codes):
            cfg = code_to_config.get(int(code))
            if cfg is None:
                raise ValueError(f"边界组 {int(code)} 不在 code_to_config 里 —— 无法确定该组面的粘性边界条件")
            kind[bnd[codes == code]] = viscous_boundary_kind_for_config(cfg)
    elif isinstance(ghost_provider, DefaultGhostProvider):
        kind[bnd] = VBC_NEUMANN
    elif bnd.size > 0:
        raise TypeError(
            f"{type(ghost_provider).__name__} 没有可读的边界条件语义"
            f"（group_code / code_to_config），无法确定粘性通量的边界处理；"
            f"请提供 BoundaryGhostStateProvider 或委托给它、并共享这两个属性的包装层")

    try:
        ghost_provider._afcfd_vbc_kind = kind
    except AttributeError:  # 只读/带 __slots__ 的 provider，放弃缓存即可
        pass
    return kind


def restrict_ghost_provider(provider, face_ids: np.ndarray):
    """把幽灵态提供者限制到 `face_ids`（分布式本 rank 面序）：`BoundaryGhostStateProvider` 用其
    `restricted_to_faces`；None 原样返回。"""
    if provider is None:
        return None
    return provider.restricted_to_faces(face_ids)
