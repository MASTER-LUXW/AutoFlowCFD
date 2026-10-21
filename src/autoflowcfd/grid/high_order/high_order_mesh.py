"""
AutoFlowCFD - 高阶 FR 网格（HighOrderMesh）。

管理解点（SPs）、单元-面连接关系以及相关的几何量：
1. 从 VolumeMeshData（棱柱 + 四面体混合体网格）构建原生基解点与解析精确的度量（四面体常数 Jacobian、
   棱柱 `fr/native_prism/basis.py::native_prism_exact_jacobian`）；负 / 零 Jacobian 中止加载并报告单元 ID
2. 单元-面连接关系（`grid/connectivity/face_connectivity.py`，供 FR 残差组装的界面通量使用）
3. 几何守恒律（GCL）验证（Kopriva 度量恒等式，`gcl_check.py`）
"""

from typing import TYPE_CHECKING, Dict, Optional

import numpy as np
from loguru import logger

from autoflowcfd.fr.operators import generate_fr_operators
from ..curved_mapping.curved_mapping_orientation import (
    fix_prism_orientation,
    fix_tet_orientation,
)
from ..connectivity.face_connectivity import FRFaceConnectivity, build_face_connectivity

if TYPE_CHECKING:
    from autoflowcfd.fr.face_flux_points.data import _KernelFaceData

#: 标准参考四面体 `(-1,-1,-1),(1,-1,-1),(-1,1,-1),(-1,-1,1)` 的体积
#: （见 `fr/native_tet/basis.py::compute_native_tet_jacobian`）。
_NATIVE_REF_TET_VOLUME = 4.0 / 3.0


def _prism_sp_volume_weights(order: int, n_sps_per_cell: int) -> np.ndarray:
    """棱柱逐解点的参考体积求积权重，形状 `(n_sps_per_cell,)`。

    解点是三角形 Warp&Blend ⊗ 挤出方向 Gauss 的 `(p+1)^2(p+2)/2` 个点，权重由
    `fr/native_prism/quadrature.build_native_prism_sp_weights` 给出（和为参考棱柱体积 4），**零填充槽位
    权重为 0**。`get_cell_volume` 与 `get_all_cell_volumes` 共用这一份。
    """
    from autoflowcfd.fr.native_prism.quadrature import (
        build_native_prism_sp_weights,
    )

    w_native = build_native_prism_sp_weights(order)
    w = np.zeros(n_sps_per_cell, dtype=np.float64)
    w[: w_native.shape[0]] = w_native
    return w


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

    def __init__(self, order: int = 2):
        self.order = order
        self.n_points_1d = order + 1
        self.operators = generate_fr_operators(order)
        self.n_sps_per_cell = self.operators.n_sps

        self.sps_coords: Optional[np.ndarray] = None
        self.jacobians: Optional[Dict[str, np.ndarray]] = None
        # 体积项去混叠（超过-积分，V2.0 二次评审 Tier 0 #2）用的
        # 细网格几何：过积分阶数见 `fr/overintegration_order.py`，"det_jacs"/"inv_jacs" 形状与
        # self.jacobians 同构但按
        # n_sps_per_cell_fine 展开；order==0 时为 None（P0 走独立的有限
        # 体积残差路径，不需要）。
        self.jacobians_fine: Optional[Dict[str, np.ndarray]] = None
        self.n_sps_per_cell_fine: int = 0
        self.n_cells = 0
        self.n_prism_cells = 0
        self.face_connectivity: Optional[FRFaceConnectivity] = None
        self.face_flux_points: Optional["_KernelFaceData"] = None
        self.boundary_groups: Optional[Dict[str, np.ndarray]] = None
        self.boundary_bc_types: Optional[Dict[str, str]] = None
        self.boundary_surface_mesh: Optional[Dict[str, object]] = None

        # 供 fr/face_flux_points/geometry.py 按需重新映射任意计算立方体坐标点到
        # 物理坐标（Flux 点 与 SPs 是不同的点集，需要能对单元重新映射）
        self._fixed_prism_conn: Optional[np.ndarray] = None
        self._fixed_tet_conn: Optional[np.ndarray] = None
        self._node_coords: Optional[np.ndarray] = None

        # 真实几何单元体积，网格加载时用可信的目标阶数几何算一次并固定
        # 下来（与当前活动阶数无关）——P0（顺序 Continuation 最低阶）的
        # 有限体积残差路径（core/fr_residual/inviscid_p0.py）
        # 需要它：P0 只有 1 个解点，单点求积不能准确给出棱柱（双线性侧面时 det(J) 非常数）的体积，
        # 必须用不依赖当前阶数的这份精确体积。
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
        # connectivity/节点坐标——这两者与阶数无关，是 build_order_geometry
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

        from .high_order_mesh_order import build_order_geometry

        geom = build_order_geometry(self, self.order)
        self.sps_coords = geom.sps_coords
        self.jacobians = geom.jacobians
        self.jacobians_fine = geom.jacobians_fine
        self.n_sps_per_cell_fine = geom.n_sps_per_cell_fine

        # 用当前（目标）阶数的高阶 Gauss-Legendre 求积算一次真实体积并固定
        # 下来——见 __init__ 里 cell_volumes 属性的文档。
        self.cell_volumes = self.get_all_cell_volumes()

        logger.info(f"HighOrderMesh initialized: {self.n_cells} cells (all Jacobians verified positive)")

        if getattr(volume_mesh_data, "boundaries", None) is not None:
            self.boundary_groups = boundary_groups
            self.boundary_bc_types = volume_mesh_data.boundaries.bc_types

        # 供 tag_boundary_groups_for_mesh 做逐面几何匹配用（单元粒度的 self.boundary_groups 在角/棱单元上
        # 无法判定每个面属于哪个组）。体网格生成、导入与 .nas 体网格求解路径都写入 volume_mesh.surface_mesh，
        # 随 .pkl 一起保存；缺失时（合成网格、旧版 .pkl）回退到单元级别匹配，有歧义的单元报错。
        if getattr(volume_mesh_data, "surface_mesh", None) is not None:
            self.boundary_surface_mesh = volume_mesh_data.surface_mesh

        if build_faces:
            self.face_connectivity = build_face_connectivity(
                fixed_prism_conn, fixed_tet_conn, nodes
            )

            # 面编码直接是原生编码（四面体 [6,10)、棱柱 [10,15)，`build_face_connectivity`）

            # 周期边界配对：必须在这里、build_face_flux_points 之前完成——
            # 配对把周期面从 is_boundary=True 翻转成内部面，需要在
            # build_face_flux_points 的 owner/neighbor 分组判据（按
            # face_connectivity.is_boundary 分流）生效之前就已经翻转好，
            # 否则周期面会被当成普通边界面处理（不会有跨单元插值），
            # 见 grid/connectivity/ 的 apply_periodic_pairing_from_boundary_map
            # 文档。
            boundary_map = getattr(volume_mesh_data, "boundaries", None)
            if boundary_map is not None and "PERIODIC" in getattr(boundary_map, "bc_types", {}).values():
                from autoflowcfd.grid.connectivity.face_connectivity_periodic import (
                    apply_periodic_pairing_from_boundary_map,
                )

                self.face_connectivity = apply_periodic_pairing_from_boundary_map(
                    self.face_connectivity, boundary_map
                )

            from autoflowcfd.fr.face_flux_points.merge import build_face_flux_points

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
                from autoflowcfd.core.fr_operators.metric_diagnostics import (
                    log_mesh_metric_report,
                    precompute_cell_face_misalignment,
                )

                log_mesh_metric_report(
                    self.jacobians["scaled_quality"].reshape(self.n_cells, self.n_sps_per_cell),
                    precompute_cell_face_misalignment(self),
                )
        elif self.jacobians is not None:
            from autoflowcfd.core.fr_operators.metric_diagnostics import log_mesh_metric_report

            log_mesh_metric_report(self.jacobians["scaled_quality"].reshape(self.n_cells, self.n_sps_per_cell))

        # 把刚构建好的这个阶数的完整几何缓存起来——顺序 Continuation 结束时
        # 切回目标阶数应直接复用这份（本来就是目标阶数），不应该重新触发一次
        # 昂贵的 Flux 点 重建（见 set_order 文档）。
        from .high_order_mesh_order import cache_active_order

        cache_active_order(self)

    def loaded_node_coords(self) -> np.ndarray:
        """修正单元朝向时存下的节点坐标（与阶数无关的几何输入）；`load_from_volume_mesh` 之前调用即报错。"""
        if self._node_coords is None:
            raise RuntimeError("HighOrderMesh 尚未加载网格（先调用 load_from_volume_mesh）")
        return self._node_coords

    def set_order(self, order: int) -> None:
        """切换网格当前活动的多项式阶数（顺序 Continuation 专用）。
        实现见 high_order_mesh_order.py::set_order。"""
        from .high_order_mesh_order import set_order as _set_order

        _set_order(self, order)

    def verify_gcl(self, tolerance: float = 1e-8) -> bool:
        """几何守恒律自检——薄委托到 `gcl_check.verify_gcl`
        （2026-09-19 拆出，见该模块文档）。"""
        from .gcl_check import verify_gcl as _impl

        return _impl(self, tolerance)

    def get_cell_volume(self, cell_id: int) -> float:
        """计算指定单元的体积（Jacobian 行列式在参考单元上的加权积分）。

        四面体：常数 Jacobian × 参考四面体体积 4/3（精确）；棱柱：解点上 det(J) 乘原生求积权重
        （`_prism_sp_volume_weights`，对 det(J) 的多项式次数精确）。
        """
        if self.jacobians is None or cell_id >= self.n_cells:
            return 0.0

        if cell_id >= self.n_prism_cells:
            det_j = self.jacobians["det_jacs"][cell_id * self.n_sps_per_cell]
            return float(det_j * _NATIVE_REF_TET_VOLUME)

        weights_3d = _prism_sp_volume_weights(self.order, self.n_sps_per_cell)
        start = cell_id * self.n_sps_per_cell
        det_jacs = self.jacobians["det_jacs"][start : start + self.n_sps_per_cell]
        return float(np.sum(det_jacs * weights_3d))

    def get_all_cell_volumes(self) -> np.ndarray:
        """向量化版本的 get_cell_volume，一次性算出所有单元的精确体积。

        四面体直接用 `det_j*4/3`（直边四面体的 det(J) 是常数，4/3 是标准参考四面体
        `(-1,-1,-1),(1,-1,-1),(-1,1,-1),(-1,-1,1)` 的体积，见 `native_tet/basis.py::compute_native_tet_jacobian`）；
        棱柱用原生求积权重（和为参考棱柱体积 4，见 `_prism_sp_volume_weights` 与 `fr/native_prism/quadrature.py`）。
        CFL / 网格尺度、IP 罚项长度与 LES 滤波宽度都消费这份体积（不能用 `mean(det_jacs)` 代替）。

        Returns:
            volumes: 形状 (n_cells,)
        """
        if self.jacobians is None:
            return np.zeros(self.n_cells)

        weights_3d = _prism_sp_volume_weights(self.order, self.n_sps_per_cell)
        det_jacs = self.jacobians["det_jacs"].reshape(self.n_cells, self.n_sps_per_cell)
        volumes = np.sum(det_jacs * weights_3d[np.newaxis, :], axis=1)

        if self.n_prism_cells < self.n_cells:
            volumes[self.n_prism_cells:] = (
                det_jacs[self.n_prism_cells:, 0] * _NATIVE_REF_TET_VOLUME)

        return volumes
