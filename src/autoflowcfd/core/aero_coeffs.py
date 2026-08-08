"""FVM 求解器的气动系数计算器。

本模块通过对车身表面积分压力来计算气动系数（Cd、Cl 等）。

Key Components:
    - AeroCoefficientCalculator: 计算阻力和升力系数
"""

import numpy as np
from typing import Optional, Tuple
from loguru import logger

from .aero_reference_area import ReferenceAreaMixin


class AeroCoefficientCalculator(ReferenceAreaMixin):
    """从解场计算气动系数。

    参考面积（`_compute_reference_area` 及其两种实现）的计算逻辑在
    `aero_reference_area.ReferenceAreaMixin` 里，纯粹是为了控制单文件
    行数拆出去的，不是独立的概念层——本类的公开接口不变。
    """

    def __init__(self, grid_data, face_extractor, rho_inf: float = 1.225, vel_inf: float = 30.0):
        """初始化气动系数计算器。

        Args:
            grid_data: 体网格数据 (VolumeMeshData)
            face_extractor: 体网格的面提取器
            rho_inf: 自由来流密度 (kg/m^3)，与求解器的初始条件、
                入口/远场边界条件共用同一份 SteadyConfig——必须一致，
                Cd/Cl 才能相对实际自由来流归一化，而不是各算各的猜测值。
            vel_inf: 自由来流速度大小 (m/s)，作用同 rho_inf。
        """
        self.grid_data = grid_data
        self.face_extractor = face_extractor
        self.rho_inf = rho_inf
        self.vel_inf = vel_inf

        # 参考面积缓存，避免重复计算
        self._cached_ref_area = None
        self._ref_area_computed = False

        # 关键优化：缓存车身面索引（网格生成后固定不变）
        self._cached_body_faces = None
        self._body_faces_cached = False

    def compute_coefficients(
        self,
        solution: np.ndarray,
        iteration: int = 0,
        viscous_residual: Optional[object] = None,
        grad_vel: Optional[np.ndarray] = None,
        mu_t: Optional[np.ndarray] = None,
        boundary_states: Optional[np.ndarray] = None,
    ) -> Tuple[float, float, float, float]:
        """计算阻力和升力系数。

        Args:
            solution: 解数组，形状 (n_cells, 7)
            iteration: 当前迭代数（用于调试）
            viscous_residual: 可选的 ViscousRANSResidual 实例，用它的
                wall_shear_stress() 方法计算车身表面的摩擦（粘性剪切）
                阻力/升力——与动量残差实际平衡的是同一份应力，因此 Cd/Cl
                和求解器实际求解的结果保持一致。若为 None（或
                grad_vel/mu_t/boundary_states 为 None），则只返回压差
                （形状）阻力——跳过摩擦阻力。
            grad_vel, mu_t, boundary_states: 求解循环每次迭代已经算好的
                量，这里复用以避免重复计算速度梯度/涡粘性。

        Returns:
            (Cd, Cl, Cd_pressure, Cd_friction) 元组
        """
        try:
            # 提取原始变量
            rho = solution[:, 0]
            rhou = solution[:, 1]
            rhov = solution[:, 2]
            rhow = solution[:, 3]
            E = solution[:, 4]

            gamma = 1.4
            velocity_x = rhou / np.maximum(rho, 1e-10)
            velocity_y = rhov / np.maximum(rho, 1e-10)
            velocity_z = rhow / np.maximum(rho, 1e-10)

            V_squared = velocity_x**2 + velocity_y**2 + velocity_z**2
            pressure = (gamma - 1.0) * (E - 0.5 * rho * V_squared)

            # 自由来流条件（来自 SteadyConfig，与求解器的初始条件、边界
            # 条件共用）。
            rho_inf = self.rho_inf
            vel_inf = self.vel_inf
            q_inf = 0.5 * rho_inf * vel_inf**2

            if q_inf < 1e-6:
                logger.warning("Dynamic pressure too small")
                return 0.0, 0.0, 0.0, 0.0

            # 识别车身面
            body_face_indices = self._identify_body_faces()

            if len(body_face_indices) == 0:
                logger.warning(f"[Iter {iteration}] No body faces found - returning Cd=0, Cl=0")
                return 0.0, 0.0, 0.0, 0.0

            # 取面数据
            face_normals = self.face_extractor.face_normals[body_face_indices]
            face_areas = self.face_extractor.face_areas[body_face_indices]

            # 校验形状
            if len(body_face_indices) != len(face_normals):
                logger.error(
                    f"[Iter {iteration}] CRITICAL: body_face_indices length ({len(body_face_indices)}) "
                    f"!= face_normals length ({len(face_normals)})"
                )
                return 0.0, 0.0, 0.0, 0.0

            if len(body_face_indices) != len(face_areas):
                logger.error(
                    f"[Iter {iteration}] CRITICAL: body_face_indices length ({len(body_face_indices)}) "
                    f"!= face_areas length ({len(face_areas)})"
                )
                return 0.0, 0.0, 0.0, 0.0

            # 取车身表面压力
            body_cell_indices = self.face_extractor.face_connectivity[body_face_indices, 0]
            p_body = pressure[body_cell_indices]
            p_ref = 101325.0

            dp = p_body - p_ref

            # 校验 dp 形状
            if len(dp) != len(face_areas):
                logger.error(
                    f"[Iter {iteration}] CRITICAL: dp length ({len(dp)}) "
                    f"!= face_areas length ({len(face_areas)})"
                )
                return 0.0, 0.0, 0.0, 0.0

            # 压差（形状）阻力/升力。
            Fx_p = -np.sum(dp * face_normals[:, 0] * face_areas)
            Fz_p = -np.sum(dp * face_normals[:, 2] * face_areas)

            # 摩擦（粘性剪切）贡献，前提是调用方提供了计算所需的量。
            # tau_n = tau.n（n 从流体指向外，即从 owner 单元指向车身）
            # 是壁面施加在流体上的牵引力（Cauchy 约定）；根据牛顿第三
            # 定律，流体施加在车身上的力是 -tau_n。这个符号在接入之前
            # 已经用一个简单的 Couette 流类比数值验证过（流体以 U>0 掠过
            # 静止壁面，必须产生一个正的、顺流向的摩擦阻力）——朴素地
            # 直接写 '+=' 会得到符号错误的力。
            Fx_f = 0.0
            Fz_f = 0.0
            if (viscous_residual is not None and grad_vel is not None
                    and mu_t is not None and boundary_states is not None):
                vel = np.column_stack([velocity_x, velocity_y, velocity_z])
                tau_n_all = viscous_residual.wall_shear_stress(rho, vel, mu_t, grad_vel, boundary_states)

                # tau_n_all 形状：(n_boundary_faces, 3)
                # 把 body_face_indices 映射到边界面的位置

                # 取所有边界面的全局索引
                boundary_face_ids = np.where(self.face_extractor.boundary_flags)[0]

                # 关键修复：确保 body_face_indices 确实都是边界面
                # 用 np.isin 找出有效索引，再映射到边界面的位置
                mask_valid = np.isin(body_face_indices, boundary_face_ids)

                if np.any(mask_valid):
                    valid_body_faces = body_face_indices[mask_valid]

                    # 把全局面索引映射到它在 tau_n_all 里的位置（即在
                    # boundary_face_ids 里的位置）。np.where 自身的输出
                    # 已经按升序排列，而 mask_valid（对 boundary_face_ids
                    # 做 np.isin）已经保证 valid_body_faces 里的每个值都
                    # 在其中，因此 np.searchsorted 可以直接给出每个面的
                    # 精确位置——这是每次求解器迭代都要跑的路径，不需要
                    # 在 Python 层构建/查询字典。
                    body_pos_in_boundary = np.searchsorted(boundary_face_ids, valid_body_faces)

                    # 安全检查：确保索引在范围内
                    if len(body_pos_in_boundary) > 0 and np.max(body_pos_in_boundary) < len(tau_n_all):
                        tau_n_body = tau_n_all[body_pos_in_boundary]

                        # 关键修复：确保广播时形状匹配
                        # tau_n_body: (n_valid, 3)，需要取 x、z 分量
                        # face_areas 应与 valid_body_faces 长度相同
                        valid_face_areas = face_areas[mask_valid]

                        # 计算前先校验形状是否匹配
                        if tau_n_body.shape[0] != valid_face_areas.shape[0]:
                            logger.warning(
                                f"[Iter {iteration}] Shape mismatch: tau_n_body has "
                                f"{tau_n_body.shape[0]} faces but face_areas has "
                                f"{valid_face_areas.shape[0]} faces"
                            )
                            # 用较短的长度，避免报错
                            min_len = min(tau_n_body.shape[0], valid_face_areas.shape[0])
                            tau_n_body = tau_n_body[:min_len]
                            valid_face_areas = valid_face_areas[:min_len]

                        # 用匹配好的形状计算摩擦力
                        # tau_n_body[:, 0]：壁面剪切应力的 x 分量（形状 n_valid,）
                        # valid_face_areas：每个面的面积（形状 n_valid,）
                        # 逐元素相乘后求和
                        Fx_f = -np.sum(tau_n_body[:, 0] * valid_face_areas)
                        Fz_f = -np.sum(tau_n_body[:, 2] * valid_face_areas)
                    else:
                        logger.warning(f"[Iter {iteration}] Invalid boundary face indices detected")
                else:
                    logger.warning(f"[Iter {iteration}] No valid body boundary faces found for friction calculation")
            Fx = Fx_p + Fx_f
            Fz = Fz_p + Fz_f

            # 参考面积
            ref_area = self._compute_reference_area(body_face_indices)

            # 系数
            Cd = Fx / (q_inf * ref_area)
            Cl = Fz / (q_inf * ref_area)

            # 诊断分解：对于钝体（Ahmed Body 以压差阻力为主），摩擦阻力
            # 理应只占总阻力的一小部分。如果摩擦阻力占主导或量级明显
            # 失常，问题更可能出在边界粘性通量项，而不是（已经独立、
            # 更早验证过的）压力积分。
            Cd_p = Fx_p / (q_inf * ref_area)
            Cd_f = Fx_f / (q_inf * ref_area)

            # 记录力的量级，便于调试
            if iteration <= 10 or iteration % 50 == 0:
                logger.debug(
                    f"[Iter {iteration}] Force details:\n"
                    f"  Pressure force (Fx_p): {Fx_p:.4e} N\n"
                    f"  Friction force (Fx_f): {Fx_f:.4e} N\n"
                    f"  Total force (Fx):      {Fx:.4e} N\n"
                    f"  Dynamic pressure (q):  {q_inf:.2f} Pa\n"
                    f"  Reference area (A):    {ref_area:.4f} m^2\n"
                    f"  Friction/Pressure ratio: {abs(Fx_f/Fx_p)*100:.2f}%"
                )

            if abs(Fx_f) > abs(Fx_p) and abs(Fx_f) > 1e-9:
                logger.warning(
                    f"[Iter {iteration}] Skin-friction drag ({Fx_f:.4e} N) exceeds "
                    f"pressure drag ({Fx_p:.4e} N) in magnitude - unexpected for a "
                    f"bluff body, check wall_shear_stress/near-wall mesh scaling."
                )

            # 校验
            if not np.isfinite(Cd):
                logger.warning("Cd is not finite")
                Cd = 0.0
            if not np.isfinite(Cl):
                logger.warning("Cl is not finite")
                Cl = 0.0

            return float(Cd), float(Cl), float(Cd_p), float(Cd_f)

        except Exception:
            # 之前这里返回的是 6 元组，而正常路径返回 4 元组——调用方
            # solver_steady.py 按 4 元组 unpack，一旦触发这个分支就会变成
            # 一个更难排查的 ValueError（unpack 数量不对），把真正的异常
            # 原因掩盖掉。用 logger.exception 记录完整 traceback 后按正常
            # 路径同样的 4 元组形状返回兜底值。
            logger.exception("Failed to compute coefficients")
            return 0.0, 0.0, 0.0, 0.0

    def _identify_body_faces(self) -> np.ndarray:
        """从边界条件中识别车身表面面。

        关键优化：缓存结果，避免重复的昂贵计算。车身面在网格生成后
        固定不变，迭代过程中永远不会变化。

        Returns:
            属于车身边界的面索引数组
        """
        # 若有缓存则直接返回（99% 的调用都会命中缓存）
        if self._body_faces_cached and self._cached_body_faces is not None:
            return self._cached_body_faces

        # 找出所有匹配 bc_handler.py 自己 WALL/车身 分类规则的边界名
        # （BoundaryConditionHandler._classify 只要名字包含 "BODY" *或*
        # "CAR" 就认定为车身——以前只检查 "body"，导致一个把壁面边界
        # 命名为 "CAR" 的网格虽然拿到了正确的无滑移边界条件，这里却识别
        # 出零个车身面，每次迭代都命中提前退出的分支）。
        body_boundary_names = [
            name for name in self.grid_data.boundaries.boundary_names
            if 'BODY' in name.upper() or 'CAR' in name.upper()
        ]

        if not body_boundary_names:
            logger.warning("No body boundary found")
            return np.array([], dtype=np.int64)

        # 用集合并集收集所有车身单元索引（向量化）
        body_cell_set = set()
        for boundary_name in body_boundary_names:
            cells = self.grid_data.boundaries.get_cell_indices(boundary_name)
            body_cell_set.update(cells)

        if not body_cell_set:
            logger.warning(f"Body boundary found but no cells identified: {body_boundary_names}")
            return np.array([], dtype=np.int64)

        # 转成 numpy 数组以便快速查找
        body_cells_array = np.array(list(body_cell_set), dtype=np.int64)

        # 取所有边界面
        boundary_mask = self.face_extractor.boundary_flags

        # 取所有面的左侧单元索引
        left_cells = self.face_extractor.face_connectivity[:, 0]

        # 用 numpy isin 做向量化的成员测试（比 Python 循环快得多）
        is_body_face = np.isin(left_cells, body_cells_array) & boundary_mask

        # 取条件为真的索引
        body_face_indices = np.where(is_body_face)[0]

        # 缓存结果供后续调用使用
        self._cached_body_faces = body_face_indices.astype(np.int64)
        self._body_faces_cached = True

        return self._cached_body_faces
