"""CFD 用体网格生成器。

通过 BL 挤出 + tetgen 核心填充的混合装配方式，从面三角化网格生成三维
四面体体网格。

本模块是一个协调者，把实际工作委托给专门的子模块：
- mesh_background：混合装配编排
- mesh_utils：校验与辅助函数
"""

import numpy as np
from typing import Dict, Optional, TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from ...schema.grid_boundaries import BoundaryMap
    from ...schema.grid_data import VolumeMeshData
from ..utils.mesh_utils import validate_surface_mesh, validate_bounding_box


class VolumeMeshGenerator:
    """从表面几何生成 3D 体积网格。

    将表面三角化（来自 NAS 文件）转换为适合 FVM 求解器的体积网格，
    通过 BL 挤出 + tetgen 核心填充（mesh_background.generate_hybrid_mesh）。

    Attributes:
        growth_rate: 边界层网格增长率
        min_cell_size: 最小单元尺寸约束
        target_cells: 目标体积单元数
    """

    def __init__(
        self,
        growth_rate: float = 1.2,
        min_cell_size: float = 0.01,
        target_cells: int = 400000,
        max_cell_size: Optional[float] = None,
        bl_layers: Optional[int] = None,
        bl_only: bool = False,
        bl_only_output: Optional[str] = None,
        core_only: bool = False,
    ):
        """初始化体积网格生成器。

        Args:
            growth_rate: 层厚度的几何增长率（1.2 为典型值）
            min_cell_size: 最小允许单元尺寸（米），默认 1cm
            target_cells: 目标总单元数
            max_cell_size: 可选的 core 区域单元尺寸硬上限（米），
                从 BL 的近壁大小向外分级（mesh_background.
                generate_hybrid_mesh）。None 使 core 填充的单元
                尺寸无界（仅应用 tetgen 自身的形状质量界限，
                单元可以长到粗糙远场输入面允许的那么大）。
            bl_layers: BL 阶段在剩余体积由 tetgen 直接从 BL 自身
                外表面填充之前挤出多少层（见 mesh_extrusion.
                extrude_layers 自身的 bl_layers 文档）。None（默认）
                使用 8。
            bl_only: 若为 True，只生成并导出 BL 棱柱层网格。
            bl_only_output: bl_only 为 True 时使用的输出 .nas 路径。
                与 bl_only 一起必须提供。
            core_only: 若为 True，在 core 区域 tetgen 填充后立即导出
                （仅 core 四面体，不与 BL 拼接）并停止——与
                bl_only_output 相同的输出路径复用。
        """
        self.growth_rate = growth_rate
        self.min_cell_size = min_cell_size
        self.target_cells = target_cells
        self.max_cell_size = max_cell_size
        self.bl_layers = bl_layers
        self.bl_only = bl_only
        self.bl_only_output = bl_only_output
        self.core_only = core_only

        logger.info(
            f"VolumeMeshGenerator initialized: growth_rate={growth_rate}, "
            f"min_cell_size={min_cell_size}m, "
            f"target_cells={target_cells}, max_cell_size={max_cell_size}, "
            f"bl_layers={bl_layers}, bl_only={bl_only}"
        )

    def generate_from_surface(
        self,
        surface_nodes: np.ndarray,
        surface_faces: np.ndarray,
        bounding_box: Dict[str, np.ndarray],
        surface_boundaries: Optional['BoundaryMap'] = None,
    ) -> 'VolumeMeshData':
        """从表面几何生成体积网格（BL + tetgen 核心填充混合）。

        Args:
            surface_nodes: 表面节点坐标，shape=(n_nodes, 3)
            surface_faces: 表面三角连接关系，shape=(n_faces, 3)
            bounding_box: 计算域边界 {min: [x,y,z], max: [x,y,z]}
            surface_boundaries: 可选的表面网格边界映射

        Returns:
            VolumeMeshData: 包含节点、单元和边界的完整体积网格

        Raises:
            ValueError: 输入几何无效
            RuntimeError: 网格生成失败
        """
        # 验证输入
        validate_surface_mesh(surface_nodes, surface_faces)
        validate_bounding_box(bounding_box)

        logger.info(
            f"Generating volume mesh from {len(surface_nodes)} nodes, "
            f"{len(surface_faces)} faces..."
        )

        return self._generate_hybrid_with_backoff(
            surface_nodes, surface_faces, bounding_box, surface_boundaries
        )

    def _generate_hybrid_with_backoff(
        self,
        surface_nodes: np.ndarray,
        surface_faces: np.ndarray,
        bounding_box: Dict[str, np.ndarray],
        surface_boundaries: Optional['BoundaryMap'],
        max_backoff_attempts: int = 1,
    ) -> 'VolumeMeshData':
        """网格质量修复循环的 Stage C：如果 generate_hybrid_mesh
        （内部已经运行 Stage A 光顺和一次 Stage B 定向重试——见
        mesh_gen/mesh_repair.py）仍然无法通过 MeshQualityValidator，
        用回退的全局参数（更大的 min_cell_size、更少的层）重试
        *整个* 生成——更粗糙的网格给尖锐特征按比例更多空间，
        在遇到相同的退化四面体失败模式之前，代价是分辨率。
        这是当 Stage A/B 的更有针对性的修复不够时的粗粒、无定向
        回退；每次尝试是完整的（可能多分钟的）重新生成，所以尝试
        次数有上限——与 generate_hybrid_mesh 内部可能每次尝试运行
        的一次 Stage B 重试组合，理论最坏情况是
        (max_backoff_attempts + 1) * 2 次完整生成过程（默认：2 个
        Stage C 级别 * 2 = 4，从早期的 3 * 2 = 6 降低——直接在
        困难的真实案例（90 度锐角物体）上测量，该案例实际上无论
        允许多少次尝试都不会收敛，每个额外级别纯增壁钟时间而
        无质量收益）。
        """
        from ..background.mesh_background import generate_hybrid_mesh
        from ...validation.quality_validator import MeshQualityValidator

        growth_rate = self.growth_rate
        min_cell_size = self.min_cell_size

        validator = MeshQualityValidator()
        # 跟踪目前为止产生的最佳网格，即使后续尝试抛出——
        # 后续回退级别失败不应该丢弃先前尝试仍可用的（虽然质量
        # 未通过的）网格。
        best_mesh = None
        best_report = None
        last_error: Optional[Exception] = None

        for attempt in range(max_backoff_attempts + 1):
            if attempt > 0:
                min_cell_size *= 1.5
                logger.warning(
                    f"Stage C: retrying generation (attempt "
                    f"{attempt}/{max_backoff_attempts}) with backed-off "
                    f"parameters: min_cell_size={min_cell_size:.6f}m - "
                    + (
                        f"previous attempt raised: {last_error}"
                        if last_error is not None
                        else "mesh quality gate still failing after Stage A/B"
                    )
                )

            try:
                volume_mesh = generate_hybrid_mesh(
                    surface_nodes, surface_faces, bounding_box,
                    growth_rate=growth_rate,
                    min_cell_size=min_cell_size,
                    target_cells=self.target_cells,
                    surface_boundaries=surface_boundaries,
                    max_cell_size=self.max_cell_size,
                    bl_layers=self.bl_layers,
                    export_bl_only=self.bl_only,
                    export_bl_only_path=self.bl_only_output,
                    export_core_only=self.core_only,
                    export_core_only_path=self.bl_only_output,
                )
            except RuntimeError as e:
                # fill_core_volume（mesh_tetgen_core.py）在 BL 表面
                # 自交或 tetgen 鲁棒性失败时抛出 RuntimeError——
                # 正是回退参数（更少/更薄的层）要修复的失败模式。
                # 以前这里没有捕获这个异常，所以它直接穿透所有剩余
                # 的回退尝试，完全中止生成，使 Stage C 在最严重的
                # 失败类别上失效，同时对下面较温和的"已生成但未
                # 通过质量门"情况仍然有效。
                last_error = e
                logger.warning(f"Stage C: attempt {attempt} raised during generation: {e}")
                continue

            report = validator.validate_volume_mesh(volume_mesh)
            # 真正保留最佳尝试，不仅是最近一次——尽管变量名如此，
            # 但以前每次迭代都无条件覆盖 best_mesh/best_report，
            # 所以无论新尝试是否实际更好，最后运行的那次都静默胜出。
            # 在真实案例上直接确认：尝试 0 产生 37 个重叠单元，
            # 尝试 1（回退参数，因为尝试 0 在其他标准上仍失败质量门
            # 而触发）产生 127 个——在本项目自身防重叠工作关注的那
            # 个指标上更差——但返回的是尝试 1。n_overlapping_cells 用作
            # 排名键（不是整体通过/失败，每次尝试按构造都已缺少它，
            # 也不是多标准分数），因为它是质量报告自身的 CRITICAL
            # 严重度字段——其他警告（偏斜度、非正交性、长宽比）是
            # HIGH/MEDIUM。
            if best_report is None or report.n_overlapping_cells < best_report.n_overlapping_cells:
                best_mesh, best_report = volume_mesh, report
            if report.passed:
                if attempt > 0:
                    logger.success(f"Stage C: attempt {attempt} passed the quality gate")
                break
        else:
            if best_mesh is None:
                # 每次尝试，包括最后一次，都抛出——没有网格可回退，
                # 所以将最后一次失败浮现出来，而不是返回 None 给调用方。
                raise RuntimeError(
                    f"Stage C: mesh generation failed on all "
                    f"{max_backoff_attempts + 1} attempt(s) (including backed-off "
                    f"parameters); last error: {last_error}"
                ) from last_error
            logger.error(
                f"Stage C: mesh quality gate still failing after "
                f"{max_backoff_attempts} backoff attempt(s) - returning the best "
                f"attempt's mesh anyway (best-effort); see the quality report above "
                f"for which cells/regions are still implicated. The solve-time "
                f"quality gate (autoflowcfd solve steady) will catch this before any "
                f"iterations run, unless --skip-quality-check is passed."
            )

        return best_mesh
