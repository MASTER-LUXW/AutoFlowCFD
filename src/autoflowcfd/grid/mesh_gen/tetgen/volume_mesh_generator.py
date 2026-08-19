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

        # 曾经这里还有一个 Stage C（全局参数回退：质量门不过就把
        # min_cell_size 放大 1.5 倍重跑整个生成），已经移除——用户
        # 明确要求的理由是它的收益不稳定（V2.0 专项攻关记录：cube_demo
        # 上三次独立对照实验里有两次 Stage C 的回退尝试反而比原始参数
        # 更差，只有一次更好，且挑选标准只看 n_overlapping_cells 一项
        # CRITICAL 指标，不是整体质量），而代价是确定的——导出的网格
        # 用的 min_cell_size 会静默偏离用户在 CLI/API 里明确传入的值，
        # 这本身就违反了"网格生成器应该忠实执行用户请求的参数，而不是
        # 在背后换一份不同的网格"这个更基本的期望。现在只用一次
        # generate_hybrid_mesh 调用（内部仍有 Stage A 光顺 + 一次
        # Stage B 定向重试，两者都不改 min_cell_size，只是更精细地
        # 利用同一份参数下已有的信息——见 mesh_background.py 自身的
        # Stage B 重试文档），不再有隐藏的第二次全局重新生成。
        from ..background.mesh_background import generate_hybrid_mesh
        return generate_hybrid_mesh(
            surface_nodes, surface_faces, bounding_box,
            growth_rate=self.growth_rate,
            min_cell_size=self.min_cell_size,
            target_cells=self.target_cells,
            surface_boundaries=surface_boundaries,
            max_cell_size=self.max_cell_size,
            bl_layers=self.bl_layers,
            export_bl_only=self.bl_only,
            export_bl_only_path=self.bl_only_output,
            export_core_only=self.core_only,
            export_core_only_path=self.bl_only_output,
        )
