"""
AutoFlowCFD - 高阶 FR 网格处理器 (V2.0 Foundation, 修复版)

本模块定义 HighOrderMesh 类，管理 Solution Points (SPs)、单元-面连接关系
以及相关的几何算子（Jacobian、GCL 验证等）。

核心功能：
1. 从 VolumeMeshData（棱柱+四面体混合体网格）构建高阶 SPs 场
2. 曲边映射（委托给 curved_mapping.py 中已数值验证的 Duffy 坍缩坐标实现）
3. 真实单元-面连接关系（委托给 face_connectivity.py，供 FR 残差组装的
   界面通量/校正项使用，取代旧版本中「全场平均态+硬编码法向量」的伪耦合）
4. 几何守恒律 (GCL) 验证（Kopriva 度量恒等式，而非旧版本错误的
   det(J) 均匀性判据——对坍缩坐标单元而言 det(J) 本就应当非均匀）

V2.0 修复记录（专家评审 Tier-0 #1,#2）：
- 原 _map_tet_to_physical / _map_prism_to_physical 的"重心坐标"形函数
  不满足单位分解（数值验证：四面体权重和在非零参数点处等于0.70而非1；
  棱柱形函数和恒为0.5），已移除，改用 curved_mapping.py 中基于 Duffy
  坍缩坐标、解析保证单位分解的正确实现。
- 原 compute_jacobian 抛出 ValueError（检测到负/零 Jacobian，即真正的
  网格畸变）后被 load_from_volume_mesh 静默捕获并替换为硬编码占位值
  （prism: 1e-6, tet: abs(线性体积)），掩盖网格畸变而非阻止其传播。
  已移除该 fallback：畸变单元现在会中止加载并报告具体单元 ID。
"""

from typing import Dict, Optional

import numpy as np
from loguru import logger

from autoflowcfd.fr.operators import generate_fr_operators
from autoflowcfd.grid.curved_mapping import (
    CurvedMapping,
    MeshDistortionError,
    fix_prism_orientation,
    fix_tet_orientation,
    map_prism_to_physical,
    map_tet_to_physical,
)
from autoflowcfd.grid.face_connectivity import FRFaceConnectivity, build_face_connectivity


class HighOrderMesh:
    """高阶 FR 网格数据结构。

    管理整个计算域的高阶网格信息，包括：
    - 所有单元的SPs物理坐标
    - 预计算的Jacobian矩阵
    - FR算子（微分矩阵、插值矩阵等）
    - 真实的单元-面连接关系（face_connectivity）

    Attributes:
        order: 多项式阶数 P
        n_points_1d: 每方向点数 (P+1)
        n_sps_per_cell: 每单元SPs数量
        n_cells: 单元总数
        n_prism_cells: 棱柱单元数量（棱柱占据全局索引 [0, n_prism_cells)）
        sps_coords: 所有单元SPs的物理坐标，形状 (n_cells, n_sps_per_cell, 3)
        jacobians: 预计算的Jacobian数据字典
        operators: FR算子集合
        face_connectivity: 真实单元-面连接关系（FRFaceConnectivity）
    """

    def __init__(self, order: int = 2):
        self.order = order
        self.n_points_1d = order + 1
        self.n_sps_per_cell = self.n_points_1d**3

        self.operators = generate_fr_operators(order)

        self.sps_coords: Optional[np.ndarray] = None
        self.jacobians: Optional[Dict[str, np.ndarray]] = None
        self.n_cells = 0
        self.n_prism_cells = 0
        self.face_connectivity: Optional[FRFaceConnectivity] = None
        self.face_flux_points: Optional[list] = None
        # 每个单元自身连接的面中，与相邻单元自己方向的最大失配量
        # 1-cos(夹角)，见 core/fr_troubled_cell.py::precompute_cell_face_misalignment；
        # 只在 load_from_volume_mesh(build_faces=True) 后才会被填充。
        self.cell_face_misalignment: Optional[np.ndarray] = None
        self.boundary_groups: Optional[Dict[str, np.ndarray]] = None
        self.boundary_bc_types: Optional[Dict[str, str]] = None

        # 供 fr/face_flux_points.py 按需重新映射任意计算立方体坐标点到
        # 物理坐标（Flux Points 与 SPs 是不同的点集，需要能对单元重新映射）
        self._fixed_prism_conn: Optional[np.ndarray] = None
        self._fixed_tet_conn: Optional[np.ndarray] = None
        self._node_coords: Optional[np.ndarray] = None

        # 真实几何单元体积，网格加载时用可信的目标阶数几何算一次并固定
        # 下来（与当前活动阶数无关）——P0（Order Continuation 最低阶）的
        # 有限体积残差路径（core/fr_residual_inviscid.py 的 n1d==1 分支）
        # 需要它：坍缩坐标下单点 Jacobian 做 1 点求积不能准确给出单元
        # 体积（同一奇异性问题，见 set_order 文档），必须用不依赖当前
        # 阶数、已在目标阶数下验证准确的这份体积。
        self.cell_volumes: Optional[np.ndarray] = None

        # Order Continuation（CL-02，P0->目标阶数平滑过渡）支持：SPs 坐标、
        # Jacobian、Flux Points 几何全部是阶数相关的（FR 方法要求解自由度
        # 与几何在同一组 SPs/FPs 上重合），不能只切换 FR 微分算子就假装
        # 换了阶数——按阶数缓存已构建过的完整几何，`set_order` 负责按需
        # 构建/切换（见该方法文档）。
        self._active_order: int = order
        self._order_geometry_cache: Dict[int, dict] = {}

    def load_from_volume_mesh(self, volume_mesh_data, build_faces: bool = True):
        """从 VolumeMeshData 对象加载并初始化高阶网格结构。

        Args:
            volume_mesh_data: VolumeMeshData 实例
            build_faces: 是否同时构建单元-面连接关系（默认 True；FR 求解器
                的无粘/粘性残差组装、边界条件施加都依赖它）

        Raises:
            MeshDistortionError: 任何单元在修正朝向后仍出现非正 Jacobian
                （真正的网格畸变，不做静默兜底）
        """
        logger.info("Initializing HighOrderMesh from VolumeMeshData...")

        nodes = volume_mesh_data.nodes.get_coordinates()

        prism_conn = volume_mesh_data.prism_cells.connectivity if volume_mesh_data.prism_cells else None
        tet_conn = volume_mesh_data.cells.connectivity
        n_prisms = len(prism_conn) if prism_conn is not None else 0
        self.n_prism_cells = n_prisms

        boundary_groups = None
        if getattr(volume_mesh_data, "boundaries", None) is not None:
            boundary_groups = volume_mesh_data.boundaries.groups

        self.n_cells = n_prisms + (len(tet_conn) if tet_conn is not None else 0)

        # --- 1/2. 棱柱/四面体单元：先修正朝向（保证正体积），存下修正后的
        # connectivity/节点坐标——这两者与阶数无关，是 _build_order_geometry
        # 在任意阶数下重算 SPs/Jacobian 的唯一输入。
        fixed_prism_conn = None
        if prism_conn is not None and n_prisms > 0:
            fixed_prism_conn = np.array(
                [fix_prism_orientation(prism_conn[i], nodes) for i in range(n_prisms)], dtype=prism_conn.dtype
            )

        fixed_tet_conn = None
        n_tets = len(tet_conn) if tet_conn is not None else 0
        if tet_conn is not None and n_tets > 0:
            fixed_tet_conn = np.array(
                [fix_tet_orientation(tet_conn[i], nodes) for i in range(n_tets)], dtype=tet_conn.dtype
            )

        self._fixed_prism_conn = fixed_prism_conn
        self._fixed_tet_conn = fixed_tet_conn
        self._node_coords = nodes

        geom = self._build_order_geometry(self.order)
        self.sps_coords = geom["sps_coords"]
        self.jacobians = geom["jacobians"]
        self._ref_cube_sps = geom["ref_cube_sps"]

        # 用当前（目标）阶数的高阶 Gauss-Legendre 求积算一次真实体积并固定
        # 下来——见 __init__ 里 cell_volumes 属性的文档：P0 阶段 1 点求积
        # 不能准确积分坍缩坐标下变化剧烈的 det(J)，必须复用这份已在目标
        # 阶数下验证过的体积，不能依赖当前活动阶数重新计算。
        self.cell_volumes = self.get_all_cell_volumes()

        logger.info(f"HighOrderMesh initialized: {self.n_cells} cells (all Jacobians verified positive)")

        if getattr(volume_mesh_data, "boundaries", None) is not None:
            self.boundary_groups = boundary_groups
            self.boundary_bc_types = volume_mesh_data.boundaries.bc_types

        if build_faces:
            self.face_connectivity = build_face_connectivity(
                fixed_prism_conn, fixed_tet_conn, nodes
            )

            # 周期边界配对：必须在这里、build_face_flux_points 之前完成——
            # 配对把周期面从 is_boundary=True 翻转成内部面，需要在
            # build_face_flux_points 的 owner/neighbor 分组判据（按
            # face_connectivity.is_boundary 分流）生效之前就已经翻转好，
            # 否则周期面会被当成普通边界面处理（不会有跨单元插值），
            # 见 grid/face_connectivity.py::apply_periodic_pairing_from_boundary_map
            # 文档。
            boundary_map = getattr(volume_mesh_data, "boundaries", None)
            if boundary_map is not None and "PERIODIC" in getattr(boundary_map, "bc_types", {}).values():
                from autoflowcfd.grid.face_connectivity import apply_periodic_pairing_from_boundary_map

                self.face_connectivity = apply_periodic_pairing_from_boundary_map(
                    self.face_connectivity, boundary_map
                )

            from autoflowcfd.fr.face_flux_points_merge import build_face_flux_points

            logger.info("Building Flux Points geometry (owner/neighbor matching)...")
            self.face_flux_points = build_face_flux_points(self.face_connectivity, self)
            logger.info(f"Flux Points geometry built for {len(self.face_flux_points)} faces")

            # 把修正后的朝向写回 volume_mesh_data，保证后续任何直接使用
            # connectivity 的代码（边界组匹配、可视化等）看到一致的朝向。
            if fixed_prism_conn is not None:
                volume_mesh_data.prism_cells.connectivity[:] = fixed_prism_conn
            if fixed_tet_conn is not None:
                volume_mesh_data.cells.connectivity[:] = fixed_tet_conn

            if self.jacobians is not None:
                from autoflowcfd.core.fr_troubled_cell import (
                    log_degenerate_cell_report,
                    precompute_cell_face_misalignment,
                )

                self.cell_face_misalignment = precompute_cell_face_misalignment(self)
                log_degenerate_cell_report(
                    self.jacobians["det_jacs"].reshape(self.n_cells, self.n_sps_per_cell),
                    self.cell_face_misalignment,
                    self.jacobians["scaled_quality"].reshape(self.n_cells, self.n_sps_per_cell),
                )
        elif self.jacobians is not None:
            from autoflowcfd.core.fr_troubled_cell import log_degenerate_cell_report

            log_degenerate_cell_report(
                self.jacobians["det_jacs"].reshape(self.n_cells, self.n_sps_per_cell),
                scaled_quality=self.jacobians["scaled_quality"].reshape(self.n_cells, self.n_sps_per_cell),
            )

        # 把刚构建好的这个阶数的完整几何缓存起来——Order Continuation 结束时
        # 切回目标阶数应直接复用这份（本来就是目标阶数），不应该重新触发一次
        # 昂贵的 Flux Points 重建（见 set_order 文档）。
        self._order_geometry_cache[self.order] = {
            "n_points_1d": self.n_points_1d,
            "n_sps_per_cell": self.n_sps_per_cell,
            "sps_coords": self.sps_coords,
            "jacobians": self.jacobians,
            "ref_cube_sps": self._ref_cube_sps,
            "operators": self.operators,
            "face_flux_points": self.face_flux_points,
            "cell_face_misalignment": self.cell_face_misalignment,
        }
        self._active_order = self.order

    def _generate_reference_cube_sps(self, order: Optional[int] = None) -> np.ndarray:
        """生成计算立方体 [-1,1]^3 内的张量积 Gauss-Legendre SPs 坐标。

        四面体、棱柱共用同一套计算立方体坐标（Duffy 坍缩坐标的计算域）；
        单元类型差异完全体现在 curved_mapping 的物理映射函数中，这里不再
        像旧版本那样对四面体/棱柱分别生成不同的"参考点"。

        Args:
            order: 目标阶数；None 时使用 self.order（当前活动阶数）。
                Order Continuation 需要在切换到某个阶数*之前*为该阶数生成
                SPs，此时 self.order 还是旧阶数，必须显式传入。
        """
        from autoflowcfd.fr.operators import gauss_legendre

        n_points_1d = (order + 1) if order is not None else self.n_points_1d
        sps_1d, _ = gauss_legendre(n_points_1d)
        xx, yy, zz = np.meshgrid(sps_1d, sps_1d, sps_1d, indexing="ij")
        return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    def _build_order_geometry(self, order: int) -> Dict[str, np.ndarray]:
        """在给定阶数下，从已修正朝向的 connectivity/节点坐标重新推导
        SPs 物理坐标与 Jacobian（不依赖 self.order/self.n_points_1d 的当前值，
        可在切换阶数*之前*安全调用）。

        Order Continuation（CL-02）的核心前提：FR 方法的解自由度与几何量
        必须共享同一组 SPs——只换 FR 微分算子（self.operators）而不重新
        推导这里的量，P0/P1 阶段的梯度/残差计算会直接用错误维度的
        Jacobian（真实网格已复现：reshape 到 27 SPs/cell 的 Jacobian 硬套
        1 SP/cell 的状态场，直接崩溃）。

        Returns:
            {"sps_coords", "jacobians", "ref_cube_sps"}
        """
        from autoflowcfd.core.fr_troubled_cell import compute_scaled_jacobian_quality

        n_points_1d = order + 1
        n_sps_per_cell = n_points_1d**3
        sps_coords = np.zeros((self.n_cells, n_sps_per_cell, 3))
        mapper = CurvedMapping(order)
        all_dets, all_inv_jacs, all_scaled_quality = [], [], []

        ref_cube_sps = self._generate_reference_cube_sps(order)
        n_prisms = self.n_prism_cells

        if self._fixed_prism_conn is not None and n_prisms > 0:
            for i in range(n_prisms):
                cell_nodes = self._node_coords[self._fixed_prism_conn[i]]
                phys_sps = map_prism_to_physical(ref_cube_sps, cell_nodes)
                sps_coords[i] = phys_sps

                jac_data = mapper.compute_jacobian(
                    phys_sps, cell_id=i, cell_type="prism", cell_nodes=cell_nodes, ref_cube_sps=ref_cube_sps
                )
                all_dets.append(jac_data["det_jacs"])
                all_inv_jacs.append(jac_data["inv_jacs"])
                all_scaled_quality.append(
                    compute_scaled_jacobian_quality(jac_data["jacobians"], jac_data["det_jacs"])
                )

        if self._fixed_tet_conn is not None and len(self._fixed_tet_conn) > 0:
            n_tets = len(self._fixed_tet_conn)
            for i in range(n_tets):
                cell_idx = n_prisms + i
                cell_nodes = self._node_coords[self._fixed_tet_conn[i]]
                phys_sps = map_tet_to_physical(ref_cube_sps, cell_nodes)
                sps_coords[cell_idx] = phys_sps

                jac_data = mapper.compute_jacobian(
                    phys_sps, cell_id=cell_idx, cell_type="tet", cell_nodes=cell_nodes, ref_cube_sps=ref_cube_sps
                )
                all_dets.append(jac_data["det_jacs"])
                all_inv_jacs.append(jac_data["inv_jacs"])
                all_scaled_quality.append(
                    compute_scaled_jacobian_quality(jac_data["jacobians"], jac_data["det_jacs"])
                )

        jacobians = None
        if all_dets:
            jacobians = {
                "det_jacs": np.concatenate(all_dets, axis=0),
                "inv_jacs": np.concatenate(all_inv_jacs, axis=0),
                "scaled_quality": np.concatenate(all_scaled_quality, axis=0),
            }

        return {"sps_coords": sps_coords, "jacobians": jacobians, "ref_cube_sps": ref_cube_sps}

    def set_order(self, order: int) -> None:
        """切换网格当前活动的多项式阶数（Order Continuation 专用）。

        SPs 坐标、Jacobian、Flux Points 几何（含 Newton 面点位定位）全部
        随阶数重新推导——这些量不是"复用同一套再插值"就够的，FR 方法要求
        解自由度与几何在同一组 SPs/FPs 上重合。按阶数缓存：目标阶数（网格
        加载时已经构建过）与之前访问过的阶数直接复用缓存，不重复触发昂贵
        的逐面 Newton 点位定位重建。

        Args:
            order: 目标阶数
        """
        if order == self._active_order:
            return

        if order not in self._order_geometry_cache:
            geom = self._build_order_geometry(order)

            # 临时切到新阶数的基础几何量：build_face_flux_points /
            # precompute_cell_face_misalignment 直接读取 self.n_points_1d /
            # self.jacobians / self.operators，必须先落地才能调用。
            self.order = order
            self.n_points_1d = order + 1
            self.n_sps_per_cell = self.n_points_1d**3
            self.sps_coords = geom["sps_coords"]
            self.jacobians = geom["jacobians"]
            self._ref_cube_sps = geom["ref_cube_sps"]
            self.operators = generate_fr_operators(order)

            face_flux_points = None
            cell_face_misalignment = None
            if self.face_connectivity is not None:
                from autoflowcfd.fr.face_flux_points_merge import build_face_flux_points

                logger.info(f"Order continuation: building Flux Points geometry for P{order}...")
                face_flux_points = build_face_flux_points(self.face_connectivity, self)
                self.face_flux_points = face_flux_points
                logger.info(f"Order continuation: Flux Points geometry built for P{order}")

                if self.jacobians is not None:
                    from autoflowcfd.core.fr_troubled_cell import precompute_cell_face_misalignment

                    cell_face_misalignment = precompute_cell_face_misalignment(self)
                    self.cell_face_misalignment = cell_face_misalignment

            self._order_geometry_cache[order] = {
                "n_points_1d": self.n_points_1d,
                "n_sps_per_cell": self.n_sps_per_cell,
                "sps_coords": self.sps_coords,
                "jacobians": self.jacobians,
                "ref_cube_sps": self._ref_cube_sps,
                "operators": self.operators,
                "face_flux_points": face_flux_points,
                "cell_face_misalignment": cell_face_misalignment,
            }

        cached = self._order_geometry_cache[order]
        self.order = order
        self.n_points_1d = cached["n_points_1d"]
        self.n_sps_per_cell = cached["n_sps_per_cell"]
        self.sps_coords = cached["sps_coords"]
        self.jacobians = cached["jacobians"]
        self._ref_cube_sps = cached["ref_cube_sps"]
        self.operators = cached["operators"]
        self.face_flux_points = cached["face_flux_points"]
        self.cell_face_misalignment = cached["cell_face_misalignment"]
        self._active_order = order

    def verify_gcl(self, tolerance: float = 1e-8) -> bool:
        """验证几何守恒律 (GCL)：对每个单元做 Kopriva 度量恒等式检验。

        Args:
            tolerance: 度量恒等式残差容差（P>=2 时应能达到机器精度量级，
                见 curved_mapping.CurvedMapping.compute_metric_identity_residual
                的文档说明；P0/P1 阶段存在坍缩坐标固有的混叠误差，不适用
                本严格判据，应在目标求解阶数下调用）

        Returns:
            bool: 全部单元 GCL 是否通过
        """
        if self.sps_coords is None:
            return False

        mapper = CurvedMapping(self.order)
        max_residual = 0.0
        n_failed = 0
        for i in range(self.n_cells):
            cell_type = "prism" if i < self.n_prism_cells else "tet"
            if cell_type == "prism":
                cell_nodes = self._node_coords[self._fixed_prism_conn[i]]
            else:
                cell_nodes = self._node_coords[self._fixed_tet_conn[i - self.n_prism_cells]]
            residual = mapper.compute_metric_identity_residual(
                self.sps_coords[i], cell_type=cell_type, cell_nodes=cell_nodes, ref_cube_sps=self._ref_cube_sps
            )
            cell_max = float(np.max(np.abs(residual)))
            max_residual = max(max_residual, cell_max)
            if cell_max >= tolerance:
                n_failed += 1

        logger.info(f"GCL check: max metric-identity residual = {max_residual:.6e} (tolerance={tolerance:.1e})")
        if n_failed > 0:
            logger.warning(f"GCL check failed for {n_failed}/{self.n_cells} cells")
        return n_failed == 0

    def get_cell_volume(self, cell_id: int) -> float:
        """计算指定单元的体积（Jacobian 行列式在参考单元上的加权积分）。

        使用 Gauss-Legendre 张量积求积权重做精确积分（∫∫∫ det(J) da db dc），
        而不是旧版本"假设行列式在单元内近似常数"的简化平均——对坍缩坐标
        映射而言 det(J) 本身就强烈非均匀，简单平均会引入明显误差。
        """
        if self.jacobians is None or cell_id >= self.n_cells:
            return 0.0

        from autoflowcfd.fr.operators import gauss_legendre

        _, w_1d = gauss_legendre(self.n_points_1d)
        wx, wy, wz = np.meshgrid(w_1d, w_1d, w_1d, indexing="ij")
        weights_3d = (wx * wy * wz).ravel()

        start = cell_id * self.n_sps_per_cell
        det_jacs = self.jacobians["det_jacs"][start : start + self.n_sps_per_cell]
        return float(np.sum(det_jacs * weights_3d))

    def get_all_cell_volumes(self) -> np.ndarray:
        """向量化版本的 get_cell_volume，一次性算出所有单元的精确体积
        （Gauss-Legendre 张量积求积，而不是"det(J)均值*8"的简化近似——
        后者被 core/fr_solver.py 的 CFL/网格尺度估计沿用了很久，已在此
        统一替换为正确的加权积分）。

        Returns:
            volumes: 形状 (n_cells,)
        """
        if self.jacobians is None:
            return np.zeros(self.n_cells)

        from autoflowcfd.fr.operators import gauss_legendre

        _, w_1d = gauss_legendre(self.n_points_1d)
        wx, wy, wz = np.meshgrid(w_1d, w_1d, w_1d, indexing="ij")
        weights_3d = (wx * wy * wz).ravel()

        det_jacs = self.jacobians["det_jacs"].reshape(self.n_cells, self.n_sps_per_cell)
        return np.sum(det_jacs * weights_3d[np.newaxis, :], axis=1)
