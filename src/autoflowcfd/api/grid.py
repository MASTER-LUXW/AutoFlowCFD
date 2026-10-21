"""AutoFlowCFDAPI 的网格入口：面网格加载与校验、体网格生成与求解前质量门。

校验与 CLI 同一套实现：面网格用 `GridValidator`（`grid validate` / `grid generate-volume` 第 2 步），体网格用求解前
质量门 `MeshQualityValidator`（`solve steady/transient`）。2026-10-19 以前这里另有两份"玩具"校验（面网格只看节点数与
有没有 WALL 组；体网格只看体积标准差/均值 > 2、`if volume_mesh.prism_cells` 对对象取真值恒真），与真实质量门不一致，
API 的 `run_*` 也不跑质量门。
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

from loguru import logger

from autoflowcfd.core.freestream_defaults import MU_MOLECULAR, RHO_INF
from autoflowcfd.grid.structures import GridData, VolumeMeshData


class _APIGridMixin:
    """网格加载、校验、体网格生成"""

    def load_grid(self, grid_file: Union[str, Path], units: str = "mm", bc_types: Optional[Dict[str, str]] = None,
                  encoding: str = "UTF-8", validate: bool = True) -> GridData:
        """解析 NAS 面网格；`validate` 时跑 `GridValidator`，不通过只记警告（与 `grid generate-volume` 一致）。

        Args:
            units: 文件坐标单位 mm / m / auto（同 CLI `--units`，内部一律换算为米）。2026-10-21 以前 API 没有这个参数，
                .nas 一律按毫米读入，米制网格被缩小 1000 倍
            bc_types: `{组名: 边界条件类型}`，覆盖按组名识别的结果（同 CLI `--bc-type`）
        """
        from autoflowcfd.grid.nas_io.parser import NASParser

        logger.info(f"Loading grid: {grid_file}")
        grid_data = NASParser(str(grid_file), encoding=encoding, units=units, bc_types=bc_types).parse()
        if validate:
            report = self.validate_grid(grid_data)
            if not report["passed"]:
                logger.warning(f"面网格质量校验未通过：{report.get('summary', '')}")
        logger.info(f"Grid loaded: {grid_data.node_count} nodes, {grid_data.cell_count} cells")
        self.grid_data = grid_data
        return grid_data

    def get_grid_info(self, grid_data: GridData) -> Dict[str, Any]:
        """面网格规模与边界组。"""
        info = {
            "node_count": grid_data.node_count,
            "cell_count": grid_data.cell_count,
            "boundary_count": len(grid_data.boundaries) if hasattr(grid_data, "boundaries") else 0,
        }
        if hasattr(grid_data, "boundary_groups"):
            info["boundary_groups"] = grid_data.boundary_groups
        return info

    def validate_grid(self, grid_data: GridData) -> Dict[str, Any]:
        """面网格质量校验（`GridValidator`，与 CLI `grid validate` 同一实现）。"""
        from autoflowcfd.grid.validation.validator import GridValidator

        return GridValidator(grid_data).validate()

    def generate_volume_mesh(self, grid_data: GridData, *, first_layer_height: Optional[float] = None,
                             target_yplus: Optional[float] = None, ref_velocity: Optional[float] = None,
                             ref_length: Optional[float] = None, rho_inf: float = RHO_INF,
                             mu_molecular: float = MU_MOLECULAR, **kwargs) -> VolumeMeshData:
        """由面网格生成体网格，与 CLI `grid generate-volume` 同一个入口
        （`grid/nas_io/parser_volume.py::generate_volume_mesh_from_surface`）。

        第一层高度直接给出（`first_layer_height`，米）或由目标 y+ 推出（`target_yplus` + `ref_velocity` + `ref_length`，
        `rho_inf` / `mu_molecular` 为估计用的物性），二选一，与 CLI 同一个函数
        （`grid/mesh_gen/extrusion/first_layer.py::resolve_first_layer_height`）。kwargs：growth_rate / min_cell_size /
        max_cell_size / bl_layers / bl_only / core_only，含义与 CLI 同名选项一致；未知参数报错。

        Raises:
            ValueError: 第一层高度的参数组合不对
            TypeError: 未知参数
        """
        from autoflowcfd.grid.mesh_gen.extrusion.first_layer import resolve_first_layer_height
        from autoflowcfd.grid.nas_io.parser_volume import generate_volume_mesh_from_surface

        height = resolve_first_layer_height(first_layer_height, target_yplus, ref_velocity, ref_length, rho_inf,
                                            mu_molecular)
        params = dict(kwargs)
        if height is not None:
            params["first_layer_height"] = height
        volume_mesh = generate_volume_mesh_from_surface(grid_data, params)
        self.volume_mesh = volume_mesh
        return volume_mesh

    def get_volume_mesh_info(self, volume_mesh: VolumeMeshData) -> Dict[str, Any]:
        """体网格规模。"""
        return {
            "node_count": volume_mesh.node_count,
            "cell_count": volume_mesh.cell_count,
            "cell_type": volume_mesh.cell_type,
        }

    def validate_volume_mesh(self, volume_mesh: VolumeMeshData):
        """体网格质量报告（`MeshQualityValidator`，即求解前质量门用的同一份判据；`.passed`、`.summary()`）。"""
        from autoflowcfd.grid.validation.quality_validator import MeshQualityValidator

        return MeshQualityValidator().validate_volume_mesh(volume_mesh)
