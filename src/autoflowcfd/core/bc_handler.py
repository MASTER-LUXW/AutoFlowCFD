"""FVM 求解器的边界条件处理器。

本模块负责为有限体积法求解器的不同边界类型应用边界条件。

Key Components:
    - BoundaryConditionHandler: 把边界条件应用到 ghost 单元
"""

import numpy as np
from typing import Dict
from loguru import logger


class BoundaryConditionHandler:
    """FVM 求解器的边界条件应用器。"""

    def __init__(self, grid_data, face_extractor, rho_inf: float = 1.225, p_inf: float = 101325.0):
        self.grid_data = grid_data
        self.face_extractor = face_extractor

        # 热力学常数
        self.gamma = 1.4  # 空气比热比

        # 自由来流参考条件（与 FRSolver._initialize_solution 以及
        # AeroCoefficientCalculator 共用，均来自 SteadyConfig——单一数据源）。
        self.rho_inf = rho_inf
        self.p_inf = p_inf

        # 平滑速度过渡的爬升（ramp）机制
        self.ramp_factor = 0.0  # 从 0 开始，逐步增加到 1.0
        self.base_inlet_velocity = 30.0  # 基准入口速度 (m/s)；会被 FRSolver 用 config 里的值覆盖
        self.base_farfield_velocity = 30.0  # 基准远场速度 (m/s)；会被 FRSolver 用 config 里的值覆盖
        self.ramp_iterations = 0  # 求解过程中设置
        # 缓存的 边界面 -> 类型 映射（惰性构建），以及与之对应、按
        # np.where(boundary_flags)[0] 顺序排列的 numpy 数组版本——见
        # _precompute_face_types。
        self._face_types = None
        self._btypes_array = None

    def update_ramp_factor(self, iteration: int, max_iter: int):
        """更新爬升系数，用于平滑过渡速度。

        在前 20% 的迭代内，把入口/远场速度从 0 线性增加到完整值，避免
        数值不稳定。

        Args:
            iteration: 当前迭代数
            max_iter: 最大迭代数
        """
        self.ramp_iterations = max(10, int(max_iter * 0.2))  # 至少 10 次迭代，或总数的 20%

        if iteration <= self.ramp_iterations:
            # 从 0 到 1 线性爬升
            self.ramp_factor = iteration / self.ramp_iterations
        else:
            self.ramp_factor = 1.0

        return self.ramp_factor

    def get_current_inlet_velocity(self) -> float:
        """获取施加了爬升系数后的当前入口速度。"""
        return self.base_inlet_velocity * self.ramp_factor

    def get_current_farfield_velocity(self) -> float:
        """获取施加了爬升系数后的当前远场速度。"""
        return self.base_farfield_velocity * self.ramp_factor

    def _inlet_bc(self) -> np.ndarray:
        """带爬升系数的定速入口边界条件。

        与 FARFIELD 不同，INLET 是用户明确命名的强制入流面，因此这里
        直接给定硬 Dirichlet 自由来流状态就是物理上正确的做法——不存在
        需要判断流动方向的歧义。

        自由来流湍流量（守恒形式）：一个较小的 k 和适中的 omega，给出
        较小的自由来流涡粘性。
        """
        gamma = 1.4
        rho_inf = self.rho_inf

        # 施加爬升系数到速度
        u_inf = self.get_current_inlet_velocity()
        p_inf = self.p_inf

        rhou_inf = rho_inf * u_inf
        E_inf = p_inf / (gamma - 1.0) + 0.5 * rho_inf * u_inf**2

        # k_inf、omega_inf 以原始变量形式给出 -> 转换为守恒形式。
        k_inf = 1.5 * (0.01 * max(u_inf, 1.0))**2   # 1% 湍流强度
        omega_inf = 5.0 * max(u_inf, 1.0) / 0.1     # 长度尺度 ~0.1 m
        return np.array([rho_inf, rhou_inf, 0.0, 0.0, E_inf,
                        rho_inf * k_inf, rho_inf * omega_inf])

    @staticmethod
    def _classify(name_upper: str) -> str:
        if "BODY" in name_upper or "CAR" in name_upper:
            return "WALL"
        elif "GROUND" in name_upper:
            return "GROUND"
        elif "INLET" in name_upper or "INFLOW" in name_upper:
            return "INLET"
        elif "OUTLET" in name_upper:
            return "OUTLET"
        elif "SYMMETRY" in name_upper:
            return "SYMMETRY"
        elif "TUNNEL" in name_upper:
            # 命名为 "tunnel" 的边界是一个物理（即便无摩擦）风洞壁面——
            # 零穿透、自由滑移——而不是开放的域边界。这里复用 SYMMETRY 的
            # ghost 状态（镜像法向速度分量，不施加粘性剪切），对无粘壁面
            # 处理而言与滑移壁面在数学上是等价的。以前这个分类被归到
            # FARFIELD（一个开放的、特征边界，允许质量自由穿越），这对
            # 真实风洞壁面而言是错误的物理，会让流动从本该是固壁的地方
            # 泄漏出去。
            return "SYMMETRY"
        elif "FARFIELD" in name_upper:
            return "FARFIELD"
        return "WALL"

    def _precompute_face_types(self) -> None:
        """一次性构建 face_idx -> 边界类型 的映射（以前是每次调用 O(N^2)）。

        先从边界组构建 单元 -> 类型 的查找表，再通过每个边界面的 owner
        单元套用到该面上。这样就不用在每次迭代、每个面上都对所有边界组
        做线性扫描。

        同时缓存 `self._btypes_array`，这是 `build_boundary_states` 每次
        调用都需要的同一份逐边界面类型数组，按固定的
        `np.where(boundary_flags)[0]` 顺序排列——网格的 boundary_flags 在
        求解过程中不会变，所以这个数组只需要算一次，而不必像以前那样
        每次调用 build_boundary_states（每个外层迭代 2-3 次，每个 RK
        阶段一次）都靠 Python `.get()` 循环重建。
        """
        cell_type: Dict[int, str] = {}
        for boundary_name in self.grid_data.boundaries.boundary_names:
            btype = self._classify(boundary_name.upper())
            for c in self.grid_data.boundaries.get_cell_indices(boundary_name):
                cell_type[int(c)] = btype

        flags = self.face_extractor.boundary_flags
        conn = self.face_extractor.face_connectivity
        bfaces = np.where(flags)[0]
        self._face_types = {}
        btypes = np.empty(len(bfaces), dtype=object)
        for i, face_idx in enumerate(bfaces):
            owner = int(conn[face_idx, 0])
            btype = cell_type.get(owner, "WALL")
            self._face_types[int(face_idx)] = btype
            btypes[i] = btype
        self._btypes_array = btypes

    def build_boundary_states(self, solution: np.ndarray) -> np.ndarray:
        """返回每个面的 ghost 守恒状态（n_faces, 7）。

        内部面对应的行保持为零（残差不会用到）；边界面的行是
        :meth:`apply_boundary_condition` 给出的 ghost 状态。

        已优化：用向量化实现取代了原来遍历边界面的 Python 循环。按类型
        对所有边界面分组处理，追求最高性能。
        """
        n_faces = len(self.face_extractor.boundary_flags)
        states = np.zeros((n_faces, 7), dtype=np.float64)

        # 一次性取出所有边界面索引
        bface_mask = self.face_extractor.boundary_flags
        if not np.any(bface_mask):
            return states

        bfaces = np.where(bface_mask)[0]
        n_bfaces = len(bfaces)

        # 批量提取所有边界面的 owner 单元索引
        owner_indices = self.face_extractor.face_connectivity[bfaces, 0].astype(np.int32)

        # 批量提取 owner 单元的内部守恒状态
        U_interior = solution[owner_indices]  # (n_bfaces, 7)

        # 一次性把所有边界面分解为原始变量
        rho = np.maximum(U_interior[:, 0], 1e-9)
        vel = U_interior[:, 1:4] / rho[:, None]  # (n_bfaces, 3)
        u, v, w = vel[:, 0], vel[:, 1], vel[:, 2]
        ke = 0.5 * rho * np.sum(vel**2, axis=1)
        p = np.maximum((self.gamma - 1.0) * (U_interior[:, 4] - ke), 100.0)
        k = np.maximum(U_interior[:, 5] / rho, 0.0)
        omega = np.maximum(U_interior[:, 6] / rho, 1e-6)

        # 取出所有边界面的法向量
        normals = self.face_extractor.face_normals[bfaces]  # (n_bfaces, 3)

        # 每个面的边界类型，已在 _precompute_face_types 里预计算好（网格
        # 的 boundary_flags/owner 在求解过程中不变，所以与上面其它量不同，
        # 这个数组不需要每次调用都重建）。
        if self._face_types is None:
            self._precompute_face_types()
        btypes = self._btypes_array

        # 逐个边界类型分组处理（每组内部是向量化的）
        unique_types = np.unique(btypes)

        for btype in unique_types:
            type_mask = (btypes == btype)
            if not np.any(type_mask):
                continue

            # 该类型在 bfaces 数组内的索引
            type_indices_in_bfaces = np.where(type_mask)[0]
            # 实际的面索引
            type_face_indices = bfaces[type_indices_in_bfaces]

            # 取出该边界类型的数据
            rho_t = rho[type_mask]
            u_t, v_t, w_t = u[type_mask], v[type_mask], w[type_mask]
            p_t = p[type_mask]
            k_t = k[type_mask]
            omega_t = omega[type_mask]
            normals_t = normals[type_mask]
            U_int_t = U_interior[type_indices_in_bfaces]

            # 按类型应用边界条件
            if btype in ["WALL", "GROUND"]:
                ghost_states = self._wall_bc_vectorized(
                    rho_t, u_t, v_t, w_t, p_t, k_t, omega_t,
                    normals_t, btype
                )
            elif btype == "INLET":
                # 强制入流：固定的自由来流 Dirichlet 状态。
                n_type = np.sum(type_mask)
                ghost_states = np.tile(self._inlet_bc(), (n_type, 1))
            elif btype == "FARFIELD":
                ghost_states = self._farfield_bc_vectorized(
                    rho_t, u_t, v_t, w_t, p_t, k_t, omega_t, normals_t
                )
            elif btype == "OUTLET":
                ghost_states = self._outlet_bc_vectorized(
                    rho_t, u_t, v_t, w_t, p_t, k_t, omega_t, normals_t
                )
            elif btype == "SYMMETRY":
                # 对称边界条件需要总能量 E
                E_t = U_int_t[:, 4]
                ghost_states = self._symmetry_bc_vectorized(
                    rho_t, u_t, v_t, w_t, E_t, k_t, omega_t, normals_t
                )
            else:
                # 默认：直接复制内部状态
                ghost_states = U_int_t.copy()

            # 把 ghost 状态写回输出数组
            states[type_face_indices] = ghost_states

        return states

    def _wall_bc_vectorized(self, rho: np.ndarray, u: np.ndarray, v: np.ndarray,
                           w: np.ndarray, p: np.ndarray, k: np.ndarray,
                           omega: np.ndarray, normals: np.ndarray,
                           wall_type: str = "WALL") -> np.ndarray:
        """向量化的壁面边界条件，带数值稳定性保护。"""
        gamma = self.gamma

        # === 数值稳定性：限幅速度以防止爆炸 ===
        MAX_VELOCITY = 1e4  # 10 km/s 物理上界
        vel_mag = np.sqrt(u**2 + v**2 + w**2)
        clip_factor = np.minimum(1.0, MAX_VELOCITY / np.maximum(vel_mag, 1e-12))

        u = u * clip_factor
        v = v * clip_factor
        w = w * clip_factor

        # 用限幅后的速度重新计算动能
        ke = 0.5 * rho * (u**2 + v**2 + w**2)

        # 保证压力为正且在合理范围内
        p = np.maximum(p, 100.0)
        p = np.minimum(p, 1e8)  # 防止极端压力

        # 目标壁面速度
        u_wall, v_wall, w_wall = 0.0, 0.0, 0.0
        if wall_type == "GROUND":
            u_wall = self.get_current_farfield_velocity()

        # Ghost = 2*wall - interior（镜像反射）
        u_ghost = 2.0 * u_wall - u
        v_ghost = 2.0 * v_wall - v
        w_ghost = 2.0 * w_wall - w

        # 同样限幅 ghost 速度
        vel_ghost_mag = np.sqrt(u_ghost**2 + v_ghost**2 + w_ghost**2)
        ghost_clip = np.minimum(1.0, MAX_VELOCITY / np.maximum(vel_ghost_mag, 1e-12))
        u_ghost *= ghost_clip
        v_ghost *= ghost_clip
        w_ghost *= ghost_clip

        rho_ghost = rho
        rhou_ghost = rho_ghost * u_ghost
        rhov_ghost = rho_ghost * v_ghost
        rhow_ghost = rho_ghost * w_ghost

        # 用限幅后的值计算 ghost 能量
        E_ghost = p / (gamma - 1.0) + 0.5 * rho_ghost * (u_ghost**2 + v_ghost**2 + w_ghost**2)

        # 湍流量：壁面处 k -> 0（直接给 Dirichlet ghost 值——若用
        # -rho*k 做镜像，也活不过 to_primitive 里 k = max(rho_k/rho, 0)
        # 的截断，所以干脆直接写），omega 从内部值外推。
        rhok_ghost = np.zeros_like(rho_ghost)
        rhow_ghost_sst = rho_ghost * omega

        return np.column_stack([
            rho_ghost, rhou_ghost, rhov_ghost, rhow_ghost,
            E_ghost, rhok_ghost, rhow_ghost_sst
        ])

    def _farfield_bc_vectorized(self, rho: np.ndarray, u: np.ndarray, v: np.ndarray,
                               w: np.ndarray, p: np.ndarray, k: np.ndarray,
                               omega: np.ndarray, normals: np.ndarray) -> np.ndarray:
        """特征（Riemann 不变量）亚声速远场边界条件。

        一个箱形远场/风洞边界，前面/侧面是局部入流，后面/顶面是局部
        出流——在所有面上都强加完整的自由来流 Dirichlet 状态（以前的
        做法）会把质量强行压过本该是出流的那些面，使那里的压力场产生
        偏差，进而反馈进 Cd/Cl 并拖慢/削弱收敛。这里改用标准的一维
        Riemann 不变量沿面法向外推：出行的不变量 R+ 取自内部，入行的
        不变量 R- 取自固定的自由来流状态，算出的法向速度指示哪一侧
        (入流还是出流) 决定切向速度和熵 (rho, p) 由谁提供，与 SU2/OpenFOAM
        等常见的亚声速特征远场边界条件做法一致。假设自由来流方向为 +x，
        与 `_inlet_bc`/`_precompute_face_types` 的约定一致
        （v_inf = w_inf = 0）。
        """
        gamma = self.gamma
        rho_inf = self.rho_inf
        p_inf = self.p_inf
        u_inf = self.get_current_farfield_velocity()

        rho_safe = np.maximum(rho, 1e-10)
        p_safe = np.maximum(p, 100.0)
        c = np.sqrt(gamma * p_safe / rho_safe)
        c_inf = np.sqrt(gamma * p_inf / max(rho_inf, 1e-10))

        nx, ny, nz = normals[:, 0], normals[:, 1], normals[:, 2]
        un = u * nx + v * ny + w * nz
        un_inf = u_inf * nx  # 自由来流方向为 +x：v_inf = w_inf = 0

        R_plus = un + 2.0 * c / (gamma - 1.0)
        R_minus = un_inf - 2.0 * c_inf / (gamma - 1.0)

        un_b = 0.5 * (R_plus + R_minus)
        c_b = np.maximum((gamma - 1.0) / 4.0 * (R_plus - R_minus), 1e-6)

        inflow = un_b < 0.0

        u_tang = u - un * nx
        v_tang = v - un * ny
        w_tang = w - un * nz
        u_tang_inf = u_inf - un_inf * nx
        v_tang_inf = -un_inf * ny
        w_tang_inf = -un_inf * nz

        u_tang_b = np.where(inflow, u_tang_inf, u_tang)
        v_tang_b = np.where(inflow, v_tang_inf, v_tang)
        w_tang_b = np.where(inflow, w_tang_inf, w_tang)

        rho_side = np.where(inflow, rho_inf, rho_safe)
        p_side = np.where(inflow, p_inf, p_safe)
        k_inf = 1.5 * (0.01 * max(u_inf, 1.0))**2
        omega_inf = 5.0 * max(u_inf, 1.0) / 0.1
        k_b = np.where(inflow, k_inf, k)
        omega_b = np.where(inflow, omega_inf, omega)

        s = p_side / rho_side ** gamma
        rho_b = np.maximum(c_b**2 / (gamma * s), 1e-10) ** (1.0 / (gamma - 1.0))
        p_b = rho_b * c_b**2 / gamma

        u_b = u_tang_b + un_b * nx
        v_b = v_tang_b + un_b * ny
        w_b = w_tang_b + un_b * nz

        rhou_b = rho_b * u_b
        rhov_b = rho_b * v_b
        rhow_b = rho_b * w_b
        E_b = p_b / (gamma - 1.0) + 0.5 * rho_b * (u_b**2 + v_b**2 + w_b**2)

        return np.column_stack([rho_b, rhou_b, rhov_b, rhow_b, E_b, rho_b * k_b, rho_b * omega_b])

    def _outlet_bc_vectorized(self, rho: np.ndarray, u: np.ndarray, v: np.ndarray,
                             w: np.ndarray, p: np.ndarray, k: np.ndarray,
                             omega: np.ndarray, normals: np.ndarray) -> np.ndarray:
        """出口边界条件（给定静压，其余外推）。

        对回流安全：如果局部内部速度实际上指向域内、穿过这个"只应出流"
        的面（un < 0——每当分离/涡脱落尾迹（例如钝体的尾迹）在到达出口
        面之前还没稳定成附着轴向流动时，这种情况很常见），单纯的零梯度
        外推密度和速度会把这个反向的、被尾迹扰动过的状态原样、无界地
        重新注入域内——这是一种自我强化的不稳定性（曾直接观测到：一个
        方块算例里，尾迹到出口平面只有约 7 个车身宽度远，密度就在出口
        平面处堆积到自由来流的 10 倍以上）。遇到回流时，退回到自由来流
        密度和零速度——一种安全、有界的"静止储槽"假设——而不是外推那个
        被扰动的内部状态。
        """
        gamma = self.gamma
        p_outlet = self.p_inf

        un = u * normals[:, 0] + v * normals[:, 1] + w * normals[:, 2]
        backflow = un < 0.0

        rho_g = np.where(backflow, self.rho_inf, rho)
        u_g = np.where(backflow, 0.0, u)
        v_g = np.where(backflow, 0.0, v)
        w_g = np.where(backflow, 0.0, w)

        # 上面的"静止储槽"假设只重置了密度/速度/压力——k/omega 也必须
        # 一起重置，否则回流会通过唯一没被这个重置机制堵住的通道，悄悄
        # 把尾迹自己的湍流特征重新灌回域内。储槽应该带有自由来流湍流，
        # 而不是尾迹湍流——用的是与 _inlet_bc 相同的 1% 强度 / 0.1 m
        # 长度尺度公式。
        u_ref = max(self.base_inlet_velocity, 1.0)
        k_inf = 1.5 * (0.01 * u_ref) ** 2
        omega_inf = 5.0 * u_ref / 0.1
        k_g = np.where(backflow, k_inf, k)
        omega_g = np.where(backflow, omega_inf, omega)

        rhou = rho_g * u_g
        rhov = rho_g * v_g
        rhow = rho_g * w_g
        E = p_outlet / (gamma - 1.0) + 0.5 * rho_g * (u_g**2 + v_g**2 + w_g**2)

        return np.column_stack([rho_g, rhou, rhov, rhow, E, rho_g * k_g, rho_g * omega_g])

    def _symmetry_bc_vectorized(self, rho: np.ndarray, u: np.ndarray, v: np.ndarray,
                               w: np.ndarray, E: np.ndarray, k: np.ndarray,
                               omega: np.ndarray, normal: np.ndarray) -> np.ndarray:
        """向量化的对称边界条件。"""
        # 法向速度分量
        u_n = u * normal[:, 0] + v * normal[:, 1] + w * normal[:, 2]

        # 镜像法向速度：u_ghost = u - 2*u_n*n
        u_ghost = u - 2.0 * u_n * normal[:, 0]
        v_ghost = v - 2.0 * u_n * normal[:, 1]
        w_ghost = w - 2.0 * u_n * normal[:, 2]

        rhou_ghost = rho * u_ghost
        rhov_ghost = rho * v_ghost
        rhow_ghost = rho * w_ghost

        return np.column_stack([
            rho, rhou_ghost, rhov_ghost, rhow_ghost, E,
            rho * k, rho * omega
        ])
