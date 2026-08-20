"""AutoFlowCFDAPI 的网格操作辅助方法。

从 api.py 拆出，控制单文件行数。包含面网格加载/验证、体网格生成/验证等方法。
这些方法在 AutoFlowCFDAPI 类中以委托方式调用（传 self 作为第一个参数）。
"""

import numpy as np
from typing import Any, Dict, Union
from pathlib import Path
from loguru import logger

from autoflowcfd.grid.nas_io.parser import NASParser
from autoflowcfd.grid.mesh_gen.tetgen.volume_mesh_generator import VolumeMeshGenerator
from autoflowcfd.grid.structures import GridData, VolumeMeshData
from autoflowcfd.grid.validation.validator import GridValidator


def api_load_grid(self, grid_file: Union[str, Path],
                  encoding: str = "UTF-8", validate: bool = True) -> GridData:
    """加载并解析网格文件。"""
    logger.info(f"Loading grid: {grid_file}")

    parser = NASParser(str(grid_file), encoding=encoding)
    grid_data = parser.parse()

    if validate:
        logger.info("Validating grid quality...")
        validation_result = api_validate_surface_grid(self, grid_data)
        if not validation_result['valid']:
            logger.warning(f"Grid validation warnings: {validation_result['warnings']}")
        else:
            logger.info("Grid validation passed")

    logger.info(
        f"Grid loaded: {grid_data.node_count} nodes, "
        f"{grid_data.cell_count} cells"
    )

    self.grid_data = grid_data
    return grid_data


def api_get_grid_info(self, grid_data: GridData) -> Dict[str, Any]:
    """获取网格信息。"""
    info = {
        'node_count': grid_data.node_count,
        'cell_count': grid_data.cell_count,
        'boundary_count': len(grid_data.boundaries) if hasattr(grid_data, 'boundaries') else 0,
    }
    if hasattr(grid_data, 'boundary_groups'):
        info['boundary_groups'] = grid_data.boundary_groups
    return info


def api_validate_grid(self, grid_data: GridData) -> Dict[str, Any]:
    """验证网格质量。"""
    validator = GridValidator(grid_data)
    result = validator.validate()
    if 'error_count' not in result:
        result['error_count'] = len(result.get('errors', []))
    if 'warning_count' not in result:
        result['warning_count'] = len(result.get('warnings', []))
    return result


def api_validate_surface_grid(self, grid_data: GridData) -> Dict[str, Any]:
    """验证表面网格质量的内部方法。"""
    warnings = []
    errors = []

    if grid_data.node_count < 3:
        errors.append("Insufficient nodes for surface mesh")

    if grid_data.cell_count == 0:
        errors.append("No surface elements found")

    if hasattr(grid_data, 'boundaries'):
        bm = grid_data.boundaries
        if not bm.bc_types:
            warnings.append("No boundary conditions defined")
        else:
            has_wall = any(bc_type == 'WALL' for bc_type in bm.bc_types.values())
            if not has_wall:
                warnings.append("No WALL boundary condition found")

    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings
    }


def api_generate_volume_mesh(self, grid_data: GridData, **kwargs) -> VolumeMeshData:
    """从面网格生成体网格。

    此前这里调用 `VolumeMeshGenerator(grid_data, method=method, **kwargs)`
    再调用 `.generate()`——但 VolumeMeshGenerator 真实的构造函数完全不
    接受 grid_data/method 这两个位置/关键字参数（只接受 growth_rate/
    min_cell_size/target_cells/max_cell_size/bl_layers 等网格生成参数，
    见 grid/mesh_gen/tetgen/volume_mesh_generator.py），也没有 `.generate()`
    方法（真实方法是 `.generate_from_surface(surface_nodes, surface_faces,
    bounding_box, surface_boundaries=None)`）——任何调用都必然
    TypeError，是从未被真正跑通过的死代码（V2.0 专家组评审逐行核实）。
    改为镜像 CLI `grid generate-volume` 命令（cli/grid_volume_commands.py）
    的真实调用方式：从 grid_data 取出 nodes/cells/boundaries，自己算
    bounding box，传给 generate_from_surface。

    kwargs 透传给 VolumeMeshGenerator 构造函数（growth_rate/
    min_cell_size/target_cells/max_cell_size/bl_layers 等，与 CLI
    `grid generate-volume` 的同名选项含义一致）。
    """
    logger.info("Generating volume mesh...")

    nodes = grid_data.nodes
    surface_nodes = np.column_stack([nodes.x, nodes.y, nodes.z])
    surface_faces = grid_data.cells.connectivity
    bounding_box = {
        'min': surface_nodes.min(axis=0),
        'max': surface_nodes.max(axis=0),
    }

    mesh_gen = VolumeMeshGenerator(**kwargs)
    volume_mesh = mesh_gen.generate_from_surface(
        surface_nodes=surface_nodes,
        surface_faces=surface_faces,
        bounding_box=bounding_box,
        surface_boundaries=getattr(grid_data, 'boundaries', None),
    )

    logger.info(
        f"Volume mesh generated: {volume_mesh.node_count} nodes, "
        f"{volume_mesh.cell_count} cells"
    )

    self.volume_mesh = volume_mesh
    return volume_mesh


def api_get_volume_mesh_info(self, volume_mesh: VolumeMeshData) -> Dict[str, Any]:
    """获取体网格信息。"""
    return {
        'node_count': volume_mesh.node_count,
        'cell_count': volume_mesh.cell_count,
        'cell_type': volume_mesh.cell_type,
    }


def api_validate_volume_mesh(self, volume_mesh: VolumeMeshData) -> Dict[str, Any]:
    """验证体网格质量。"""
    errors = []
    warnings = []

    if volume_mesh.node_count < 4:
        errors.append("Insufficient nodes for volume mesh")

    if volume_mesh.cell_count == 0:
        errors.append("No volume elements found")

    if hasattr(volume_mesh, 'cell_volumes') and volume_mesh.cell_volumes is not None:
        negative_volumes = np.sum(volume_mesh.cell_volumes <= 0)
        if negative_volumes > 0:
            errors.append(f"Found {negative_volumes} cells with non-positive volume")

        vol_mean = np.mean(volume_mesh.cell_volumes)
        vol_std = np.std(volume_mesh.cell_volumes)
        if vol_std / vol_mean > 2.0:
            warnings.append("High variation in cell volumes")

    if hasattr(volume_mesh, 'prism_cells') and volume_mesh.prism_cells:
        n_prisms = len(volume_mesh.prism_cells.connectivity)
        logger.info(f"Found {n_prisms} prism layers")

    if hasattr(volume_mesh, 'boundaries'):
        bm = volume_mesh.boundaries
        if not bm.bc_types:
            warnings.append("No boundary conditions defined on volume mesh")

    return {
        'passed': len(errors) == 0,
        'errors': errors,
        'warnings': warnings
    }
