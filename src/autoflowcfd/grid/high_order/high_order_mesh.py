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
from ..curved_mapping.curved_mapping import CurvedMapping
from ..curved_mapping.curved_mapping_orientation import (
    fix_prism_orientation,
    fix_tet_orientation,
)
from ..connectivity.face_connectivity import FRFaceConnectivity, build_face_connectivity


class HighOrderMesh:
    """高阶 FR 网格数据结构。

    管理整个计算域的高阶网格信息，包括：
    - 所有单元的SPs物理坐标
    - 预计算的Jacobian矩阵
    - FR算子（微分矩阵、插值矩阵等）
    - 真实的单元-面连接关系（face_connectivity）

    属性:
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

    def __init__(self, order: int = 2, tet_basis_mode: str = "collapsed"):
        self.order = order
        self.n_points_1d = order + 1
        self.n_sps_per_cell = self.n_points_1d**3
        # 四面体体积基函数选择（Part6/7/8 文档，路径C/native 分支）：
        # 默认 "collapsed" 与此前完全一致；"native" 时 `_build_order_
        # geometry`/`set_order` 改用 native 直边常数 Jacobian 构造几何、
        # `self.operators` 携带 `D_native_tet_padded`/`lift_native_tet_
        # padded` 等 native 字段（见 Part8 文档"三、本次会话实现范围"）。
        self.tet_basis_mode = tet_basis_mode

        self.operators = generate_fr_operators(order, tet_basis_mode=tet_basis_mode)

        self.sps_coords: Optional[np.ndarray] = None
        self.jacobians: Optional[Dict[str, np.ndarray]] = None
        # 体积项去混叠（超过-积分，V2.0 二次评审 Tier 0 #2）用的
        # 细网格几何：过积分阶数固定为 2*order（二次非线性去混叠的标准
        # 经验法则，见 fr/collapsed_basis.py::build_overintegration_operators
        # 文档），"det_jacs"/"inv_jacs" 形状与 self.jacobians 同构但按
        # n_sps_per_cell_fine 展开；order==0 时为 None（P0 走独立的有限
        # 体积残差路径，不需要）。
        self.jacobians_fine: Optional[Dict[str, np.ndarray]] = None
        self.n_sps_per_cell_fine: int = 0
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
        self.boundary_surface_mesh: Optional[Dict[str, object]] = None

        # 供 fr/face_flux_points.py 按需重新映射任意计算立方体坐标点到
        # 物理坐标（Flux 点 与 SPs 是不同的点集，需要能对单元重新映射）
        self._fixed_prism_conn: Optional[np.ndarray] = None
        self._fixed_tet_conn: Optional[np.ndarray] = None
        self._node_coords: Optional[np.ndarray] = None

        # 真实几何单元体积，网格加载时用可信的目标阶数几何算一次并固定
        # 下来（与当前活动阶数无关）——P0（顺序 Continuation 最低阶）的
        # 有限体积残差路径（core/fr_residual_inviscid.py 的 n1d==1 分支）
        # 需要它：坍缩坐标下单点 Jacobian 做 1 点求积不能准确给出单元
        # 体积（同一奇异性问题，见 set_order 文档），必须用不依赖当前
        # 阶数、已在目标阶数下验证准确的这份体积。
        self.cell_volumes: Optional[np.ndarray] = None

        # 顺序 Continuation（CL-02，P0->目标阶数平滑过渡）支持：SPs 坐标、
        # Jacobian、Flux 点 几何全部是阶数相关的（FR 方法要求解自由度
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

        抛出异常:
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
        self.jacobians_fine = geom["jacobians_fine"]
        self.n_sps_per_cell_fine = geom["n_sps_per_cell_fine"]

        # 用当前（目标）阶数的高阶 Gauss-Legendre 求积算一次真实体积并固定
        # 下来——见 __init__ 里 cell_volumes 属性的文档：P0 阶段 1 点求积
        # 不能准确积分坍缩坐标下变化剧烈的 det(J)，必须复用这份已在目标
        # 阶数下验证过的体积，不能依赖当前活动阶数重新计算。
        self.cell_volumes = self.get_all_cell_volumes()

        logger.info(f"HighOrderMesh initialized: {self.n_cells} cells (all Jacobians verified positive)")

        if getattr(volume_mesh_data, "boundaries", None) is not None:
            self.boundary_groups = boundary_groups
            self.boundary_bc_types = volume_mesh_data.boundaries.bc_types

        # 供 tag_boundary_groups_for_mesh 做逐面几何匹配用（见该函数文档：
        # 修复 tag_boundary_groups 的单元级别聚合缺陷需要原始表面网格的
        # 边界组三角面片数据，不是 self.boundary_groups 这份已经聚合到
        # 单元粒度的派生结果）——只有 input_file 是 .nas 体网格 +
        # --surface-mesh 反推边界这条路径才有（import_external_volume_mesh
        # 写入 volume_mesh.surface_mesh），.pkl 路径没有，属性缺失时
        # tag_boundary_groups_for_mesh 自动回退到单元级别匹配。
        if getattr(volume_mesh_data, "surface_mesh", None) is not None:
            self.boundary_surface_mesh = volume_mesh_data.surface_mesh

        if build_faces:
            self.face_connectivity = build_face_connectivity(
                fixed_prism_conn, fixed_tet_conn, nodes
            )

            # native 四面体（路径C，Part6/7/8 文档）：build_face_connectivity
            # 无论 tet_basis_mode 是什么都只产出坍缩坐标编码（0~5，见该函数
            # 及 face_connectivity.py 模块文档"`build_face_connectivity`
            # 因此保持完全不变、只产出坍缩坐标编码"一节）——这里翻译成
            # native 编码（6~9），下游 `build_face_flux_points`
            # （`face_flux_points_merge.py`）按 `code>=6` 自动探测启用
            # numba native 分支（Part7 文档"执行状态更新"节），不需要另外
            # 传参。只翻译四面体侧记录，棱柱不受影响（`with_native_tet_
            # faces` 文档）。
            if self.tet_basis_mode == "native":
                self.face_connectivity = self.face_connectivity.with_native_tet_faces(n_prisms)

            # 周期边界配对：必须在这里、build_face_flux_points 之前完成——
            # 配对把周期面从 is_boundary=True 翻转成内部面，需要在
            # build_face_flux_points 的 owner/neighbor 分组判据（按
            # face_connectivity.is_boundary 分流）生效之前就已经翻转好，
            # 否则周期面会被当成普通边界面处理（不会有跨单元插值），
            # 见 grid/face_connectivity.py::apply_periodic_pairing_from_boundary_map
            # 文档。
            boundary_map = getattr(volume_mesh_data, "boundaries", None)
            if boundary_map is not None and "PERIODIC" in getattr(boundary_map, "bc_types", {}).values():
                from autoflowcfd.grid.connectivity.face_connectivity import apply_periodic_pairing_from_boundary_map

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
                from autoflowcfd.core.fr_operators.troubled_cell import (
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
            from autoflowcfd.core.fr_operators.troubled_cell import log_degenerate_cell_report

            log_degenerate_cell_report(
                self.jacobians["det_jacs"].reshape(self.n_cells, self.n_sps_per_cell),
                scaled_quality=self.jacobians["scaled_quality"].reshape(self.n_cells, self.n_sps_per_cell),
            )

        # 把刚构建好的这个阶数的完整几何缓存起来——顺序 Continuation 结束时
        # 切回目标阶数应直接复用这份（本来就是目标阶数），不应该重新触发一次
        # 昂贵的 Flux 点 重建（见 set_order 文档）。
        self._order_geometry_cache[self.order] = {
            "n_points_1d": self.n_points_1d,
            "n_sps_per_cell": self.n_sps_per_cell,
            "sps_coords": self.sps_coords,
            "jacobians": self.jacobians,
            "ref_cube_sps": self._ref_cube_sps,
            "operators": self.operators,
            "face_flux_points": self.face_flux_points,
            "cell_face_misalignment": self.cell_face_misalignment,
            "jacobians_fine": self.jacobians_fine,
            "n_sps_per_cell_fine": self.n_sps_per_cell_fine,
        }
        self._active_order = self.order

    def _generate_reference_cube_sps(self, order: Optional[int] = None) -> np.ndarray:
        """生成计算立方体参考 SPs 坐标。实现见
        high_order_mesh_order.py::generate_reference_cube_sps（从本文件
        拆出，控制单文件行数），文档字符串也在那里。"""
        from .high_order_mesh_order import generate_reference_cube_sps

        return generate_reference_cube_sps(self, order)

    def _compute_jacobians_at_ref_points(
        self, mapper: "CurvedMapping", ref_pts: np.ndarray, want_scaled_quality: bool
    ) -> Optional[Dict[str, np.ndarray]]:
        """在给定参考点集上批量计算精确 Jacobian。实现见
        high_order_mesh_order.py::compute_jacobians_at_ref_points。"""
        from .high_order_mesh_order import compute_jacobians_at_ref_points

        return compute_jacobians_at_ref_points(self, mapper, ref_pts, want_scaled_quality)

    def _build_order_geometry(self, order: int) -> Dict[str, np.ndarray]:
        """在给定阶数下重新推导 SPs 物理坐标与 Jacobian。实现见
        high_order_mesh_order.py::build_order_geometry。"""
        from .high_order_mesh_order import build_order_geometry

        return build_order_geometry(self, order)

    def set_order(self, order: int) -> None:
        """切换网格当前活动的多项式阶数（顺序 Continuation 专用）。
        实现见 high_order_mesh_order.py::set_order。"""
        from .high_order_mesh_order import set_order as _set_order

        _set_order(self, order)

    def verify_gcl(self, tolerance: float = 1e-8) -> bool:
        """验证几何守恒律 (GCL)：对每个单元做 Kopriva 度量恒等式检验。

        Args:
            tolerance: 度量恒等式残差容差（P>=2 原生检验即可达到机器精度
                量级；P1 用下面的过积分检验后同样能达到，见该分支文档）

        Returns:
            bool: 全部单元 GCL 是否通过

        度量阶数与解阶数解耦（2026-08-23，此前"P1 GCL=0.105"被记录为
        已知、搁置的诊断局限，用户明确要求修复）：原实现对 adj(J) 精确
        求值后，用与当前求解阶数*相同*的坍缩坐标微分矩阵 D_3d_tet/prism
        求散度——但 adj(J) 在 Duffy 坍缩坐标下是有理函数、不是多项式，
        P1（degree=1）的微分矩阵次数不足以精确微分它，给出一个纯属
        诊断函数自身局限的 0.105（P2=1.6e-13、P3=1.2e-8 因为微分矩阵
        次数够高，问题不明显），不代表真实求解器残差有问题（真实求解器
        路径 compute_inviscid_residual_fr 的 P1 均匀自由流场残差实测
        1.25e-10，是 P1/P2/P3 三者里最好的，见 order_continuation.py
        模块文档）。

        修复思路：不再让"用来微分 adj(J)"的算子阶数与"求解阶数"绑死，
        复用真实求解器体积项本来就已经在用的过积分（over-integration）
        基础设施本身（`self.jacobians_fine`/`self.operators.
        overint_D_fine_tet/prism`，over_order=min(2*order,
        OVERINTEGRATION_MAX_ORDER)，见 high_order_mesh_order.py::
        build_order_geometry 文档）：adj(J) 在过积分细网格上精确求值
        （tet_exact_jacobian/prism_exact_jacobian 本身与阶数无关，多少
        个点都能精确求值），散度改用细网格自己更高次、更准确的微分矩阵，
        与体积项组装用的是完全同一套算子。

        真实验证结果（同一份合成混合网格）：P1 从 0.105 降到 1.64e-13
        （12 个数量级），确认原判据是纯粹的诊断局限。但同一次验证也
        发现：P2/P3 切到过积分反而从原生的 1.6e-13/1.2e-8 变差到
        1.17e-8（两者的 over_order 都被 OVERINTEGRATION_MAX_ORDER=3
        封顶，退化到同一个网格）——不是过积分本身有 bug，是本项目已经
        记录过的坍缩坐标高阶模态基条件数问题（Vandermonde 矩阵条件数
        随阶数增长，见 fr/collapsed_basis.py 相关文档）：degree-3 微分
        矩阵的截断误差改善不足以抵消它更差的舍入误差特性，P2 原生
        degree-2 矩阵在这个光滑合成网格上恰好已经落在"截断/舍入都小"
        的甜点区。因此只对 P1（唯一有真实、大幅度诊断局限的阶数）切换
        到过积分路径，P2/P3 保留已经很好的原生检验，不用一个"理论上
        更精确"但实测更差的路径替换一个已经工作良好的路径。
        """
        if self.sps_coords is None:
            return False

        if self.order == 1 and self.jacobians_fine is not None:
            return self._verify_gcl_overintegrated(tolerance)

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

    def _verify_gcl_overintegrated(self, tolerance: float) -> bool:
        """`verify_gcl` 的过积分实现，见该方法文档"度量阶数与解阶数
        解耦"一节。散度算子与真实体积项组装（fr_residual_inviscid.py）
        用的是同一个 `contract_shared_operator_2axis`，不是重新实现一遍
        同一个数学操作的第二份独立代码。
        """
        from autoflowcfd.core.fr_operators.volume_contract import contract_shared_operator_2axis

        n_prism = self.n_prism_cells
        n_fine = self.n_sps_per_cell_fine
        det_jacs_fine = self.jacobians_fine["det_jacs"].reshape(self.n_cells, n_fine)
        inv_jacs_fine = self.jacobians_fine["inv_jacs"].reshape(self.n_cells, n_fine, 3, 3)
        adj_fine = det_jacs_fine[..., None, None] * inv_jacs_fine  # adj[c,j,m,i] = adj(J)_{m,i}

        residual = np.zeros((self.n_cells, n_fine, 3))
        if n_prism > 0:
            residual[:n_prism] = contract_shared_operator_2axis(
                self.operators.overint_D_fine_prism, adj_fine[:n_prism]
            )
        if self.n_cells > n_prism:
            residual[n_prism:] = contract_shared_operator_2axis(
                self.operators.overint_D_fine_tet, adj_fine[n_prism:]
            )

        cell_max = np.max(np.abs(residual), axis=(1, 2))
        max_residual = float(np.max(cell_max))
        n_failed = int(np.sum(cell_max >= tolerance))

        logger.info(
            f"GCL check (over-integrated, order={self.order}): max metric-identity "
            f"residual = {max_residual:.6e} (tolerance={tolerance:.1e})"
        )
        if n_failed > 0:
            logger.warning(f"GCL check failed for {n_failed}/{self.n_cells} cells")
        return n_failed == 0

    def get_cell_volume(self, cell_id: int) -> float:
        """计算指定单元的体积（Jacobian 行列式在参考单元上的加权积分）。

        使用 Gauss-Legendre 张量积求积权重做精确积分（∫∫∫ det(J) da db dc），
        而不是旧版本"假设行列式在单元内近似常数"的简化平均——对坍缩坐标
        映射而言 det(J) 本身就强烈非均匀，简单平均会引入明显误差。

        native 四面体（`tet_basis_mode=="native"`）不走这套张量积求积
        权重——理由见 `get_all_cell_volumes` 文档，这里用同一个常数
        Jacobian*参考体积的精确公式。
        """
        if self.jacobians is None or cell_id >= self.n_cells:
            return 0.0

        if getattr(self, "tet_basis_mode", "collapsed") == "native" and cell_id >= self.n_prism_cells:
            _NATIVE_REF_TET_VOLUME = 4.0 / 3.0
            det_j = self.jacobians["det_jacs"][cell_id * self.n_sps_per_cell]
            return float(det_j * _NATIVE_REF_TET_VOLUME)

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

        `tet_basis_mode=="native"` 时四面体部分**不能**沿用这套张量积
        求积权重（Part8 文档"三、本次会话实现范围"新发现的一个真实
        坑）：`weights_3d` 的和是 8（[-1,1]^3 立方体体积），坍缩坐标下
        `det_jacs` 逐点变化、乘积分本来就是把"立方体计算域"积分变换到
        "物理四面体"，这个 8 已经隐含在坍缩映射的度量变化里；但 native
        直边四面体的 `det_jacs` 是**常数**（不依赖计算域参数化），直接
        乘 `weights_3d` 会把体积算大 8/(4/3)=6 倍（4/3 是标准参考四面体
        `(-1,-1,-1),(1,-1,-1),(-1,1,-1),(-1,-1,1)` 的体积，见
        `native_simplex_basis.py::compute_native_tet_jacobian` 文档）。
        native 四面体单元直接用 `det_j*4/3`，不需要任何求积（常数
        Jacobian 乘参考体积就是精确物理体积，见该函数文档）。

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
        volumes = np.sum(det_jacs * weights_3d[np.newaxis, :], axis=1)

        if getattr(self, "tet_basis_mode", "collapsed") == "native" and self.n_prism_cells < self.n_cells:
            _NATIVE_REF_TET_VOLUME = 4.0 / 3.0
            volumes[self.n_prism_cells:] = det_jacs[self.n_prism_cells:, 0] * _NATIVE_REF_TET_VOLUME

        return volumes
