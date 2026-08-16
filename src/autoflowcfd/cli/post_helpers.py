"""后处理 CLI 共用辅助函数 (从 post_commands.py 拆分)。

从 post_commands.py 拆出来（该文件原有 974 行，超过 400 行硬性拆分
阈值）：这一批"案例目录/checkpoint 定位与加载"辅助函数被
coefficients/export-vtk/report/convergence/transient-mean/
transient-rms/transient-psd 七个命令共用，与任何单个具体命令都不是
强绑定关系，独立成一个纯辅助模块最清晰——镜像
cli/solve_commands.py + cli/solve_helpers.py 已经用过的同一种拆分
方式（重量级命令主体保留在 *_commands.py，共用辅助函数搬到
*_helpers.py）。纯代码搬移，不改变任何行为。
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger


def _locate_grid_file(case_path: Path, grid: Optional[str]) -> Path:
    """定位网格文件（支持 .pkl 体网格缓存和 .nas 面网格）。"""
    if grid:
        grid_file = Path(grid)
        logger.info(f"Using specified grid file: {grid_file}")
        return grid_file

    # 优先级：volume_mesh.pkl（已保存的体网格）> grid/*.nas（面网格）> *.nas（面网格）
    grid_candidates = [
        case_path / "volume_mesh.pkl",
        case_path / "grid" / "*.nas",
        case_path / "*.nas",
    ]

    for pattern in grid_candidates:
        if pattern.exists():
            logger.info(f"Auto-detected grid file: {pattern}")
            return pattern
        if '*' in str(pattern):
            matches = list(pattern.parent.glob(pattern.name))
            if matches:
                logger.info(f"Auto-detected grid file: {matches[0]}")
                return matches[0]

    raise FileNotFoundError(
        f"Grid file not found in case directory: {case_path}\n"
        f"Please specify grid file with --grid option.\n"
        f"Expected locations:\n"
        f"  - {case_path}/volume_mesh.pkl (saved volume mesh)\n"
        f"  - {case_path}/grid/*.nas (surface mesh)\n"
        f"  - {case_path}/*.nas (surface mesh)"
    )


def _load_grid_data(grid_file: Path):
    """加载网格数据（.pkl 直接反序列化；.nas 解析并重新生成体网格）。"""
    logger.info("Loading grid data...")
    if grid_file.suffix.lower() == '.pkl':
        logger.info(f"Loading volume mesh from PKL: {grid_file}")
        try:
            with open(grid_file, 'rb') as f:
                grid_data = pickle.load(f)
            logger.success(f"✓ Volume mesh loaded: {grid_data.node_count} nodes, "
                         f"{grid_data.cell_count} cells")
        except Exception as e:
            raise ValueError(f"Failed to load volume mesh from {grid_file}: {e}")
    else:
        from autoflowcfd.grid import NASParser

        logger.warning(f"⚠ Parsing surface mesh file: {grid_file}")
        logger.warning("  This will RE-GENERATE the volume mesh!")
        logger.warning("  For best results, use volume_mesh.pkl if available.")

        parser = NASParser(str(grid_file))
        grid_data = parser.parse(generate_volume_mesh=True)
        logger.info(f"✓ Grid generated: {grid_data.node_count} nodes, "
                   f"{grid_data.cell_count} cells")
    return grid_data


def _locate_checkpoint(case_path: Path, checkpoint: Optional[str]) -> Path:
    """定位单个 checkpoint 文件（默认取最新的一个）。"""
    if checkpoint:
        ckpt_file = Path(checkpoint)
        logger.info(f"Using specified checkpoint: {ckpt_file}")
        return ckpt_file

    ckpt_dir = case_path / "checkpoints"
    latest_link = ckpt_dir / "latest"

    if latest_link.exists() and latest_link.is_symlink():
        ckpt_file = latest_link.resolve()
        logger.info(f"Auto-detected latest checkpoint: {ckpt_file}")
        return ckpt_file

    ckpt_files = _list_checkpoints(case_path)
    if ckpt_files:
        ckpt_file = ckpt_files[-1]
        logger.info(f"Auto-detected checkpoint: {ckpt_file}")
        return ckpt_file

    raise FileNotFoundError(
        f"No checkpoint files found in: {ckpt_dir}\n"
        f"Please specify checkpoint with --checkpoint option."
    )


def _list_checkpoints(case_path: Path) -> List[Path]:
    """列出案例目录下全部 checkpoint 文件，按迭代数排序（供
    transient-mean/transient-rms/transient-psd 遍历整条瞬态历史使用）。"""
    ckpt_dir = case_path / "checkpoints"
    if not ckpt_dir.exists():
        return []
    return sorted(
        ckpt_dir.glob("checkpoint_iter_*.h5"),
        key=lambda p: int(p.stem.split('_')[-1]),
    )


def _to_solution_vector(solution_data):
    """把 checkpoint 里读出的 numpy 数组包装成 SolutionVector。"""
    from autoflowcfd.core.backend.base import SolutionVector

    if isinstance(solution_data, np.ndarray):
        n_cells = solution_data.shape[0]
        n_variables = solution_data.shape[1] if len(solution_data.shape) > 1 else 5
        return SolutionVector(data=solution_data, n_cells=n_cells, n_variables=n_variables)
    return solution_data


def _load_case(case: str, grid: Optional[str] = None, checkpoint: Optional[str] = None) -> Tuple:
    """加载网格 + 单个 checkpoint 的解，供 coefficients/export-vtk 使用。

    Returns:
        (grid_data, solution, history, iteration, metadata)
    """
    from autoflowcfd.core.checkpoint import CheckpointManager

    case_path = Path(case)
    grid_file = _locate_grid_file(case_path, grid)
    grid_data = _load_grid_data(grid_file)

    ckpt_file = _locate_checkpoint(case_path, checkpoint)
    ckpt_manager = CheckpointManager(str(ckpt_file.parent))
    solution_data, history, iteration, metadata = ckpt_manager.load(ckpt_file, target_backend=None)
    logger.info(f"✓ Solution loaded from iteration {iteration}")

    solution = _to_solution_vector(solution_data)

    if grid_data.cell_count != solution.n_cells:
        raise ValueError(
            f"Grid-solution mismatch!\n"
            f"  Grid has {grid_data.cell_count} cells\n"
            f"  Solution expects {solution.n_cells} cells\n"
            f"  Please use the SAME grid file that was used in the original simulation."
        )

    return grid_data, solution, history, iteration, metadata


def _load_history_only(case: str, checkpoint: Optional[str] = None) -> Tuple[dict, int, dict]:
    """只加载一个 checkpoint 的收敛历史/元数据（report/convergence 用，不需要网格）。"""
    from autoflowcfd.core.checkpoint import CheckpointManager

    case_path = Path(case)
    ckpt_file = _locate_checkpoint(case_path, checkpoint)
    ckpt_manager = CheckpointManager(str(ckpt_file.parent))
    _solution, history, iteration, metadata = ckpt_manager.load(ckpt_file, target_backend=None)
    return history, iteration, metadata


def _replay_history(history: dict):
    """把 checkpoint 里的收敛历史（每方程残差 + 系数的并行数组）重放进
    一个新的 ConvergenceAnalyzer，供 report/convergence 复用
    ConvergenceAnalyzer/SimulationReport 已有的分析/导出逻辑，而不是
    重新实现一遍。"""
    from autoflowcfd.postprocess import AerodynamicCoefficients, ConvergenceAnalyzer

    analyzer = ConvergenceAnalyzer()
    iterations = history.get('iterations', [])
    residuals_by_eq = history.get('residuals', {})
    coeffs_by_name = history.get('coefficients', {})
    cfl_history = history.get('cfl_history', [])

    for idx, it in enumerate(iterations):
        residuals = {eq: values[idx] for eq, values in residuals_by_eq.items() if idx < len(values)}
        cfl = cfl_history[idx] if idx < len(cfl_history) else 0.0
        coefficients = None
        if 'Cd' in coeffs_by_name and idx < len(coeffs_by_name['Cd']):
            coefficients = AerodynamicCoefficients(
                Cd=coeffs_by_name.get('Cd', [0.0] * (idx + 1))[idx],
                Cl=coeffs_by_name.get('Cl', [0.0] * (idx + 1))[idx],
            )
        analyzer.add_iteration(iteration=it, residuals=residuals, cfl=cfl, coefficients=coefficients)

    return analyzer


def _cell_centroids(grid_data) -> np.ndarray:
    """计算逐单元中心点坐标，形状 (n_cells, 3)。

    与 core/solver_steady_setup.py、core/transient_solver_loop.py 里
    的中心点计算完全一致：三棱柱单元（若存在）占据全局单元索引空间的
    前段，四面体在后。
    """
    nodes_array = np.column_stack([grid_data.nodes.x, grid_data.nodes.y, grid_data.nodes.z])
    tet_connectivity = grid_data.cells.connectivity.astype(np.int64)
    tet_centroids = nodes_array[tet_connectivity].mean(axis=1)
    prism_cells_obj = getattr(grid_data, 'prism_cells', None)
    if prism_cells_obj is not None:
        prism_connectivity = prism_cells_obj.connectivity.astype(np.int64)
        prism_centroids = nodes_array[prism_connectivity].mean(axis=1)
        return np.vstack([prism_centroids, tet_centroids])
    return tet_centroids


def _export_point_fields_vtk(
    output_path: Path,
    grid_data,
    vector_fields: Dict[str, np.ndarray],
    scalar_fields: Dict[str, np.ndarray],
    binary: bool = False,
) -> None:
    """把已经是节点分辨率的场（TransientStatistics 算出的 mean/RMS 场）
    写成只含 POINT_DATA 的 legacy VTK 文件。

    复用 VTKExporter 里已经验证过的网格写入逻辑（_write_points/
    _write_cells，处理三棱柱+四面体混合网格）和标量/矢量场写入逻辑
    （_write_scalar/_write_vector），而不是重新实现一遍 VTK 格式细节——
    只是这里的场数据来源（已经在节点分辨率上）和 VTKExporter.export()
    的主路径（从单元中心的求解器数据出发，逐单元/逐节点各写一份）不同，
    所以没有直接复用 export()/export_boundaries()。
    """
    from autoflowcfd.postprocess import VTKExporter

    exporter = VTKExporter(grid_data, solution=None)
    n_points = grid_data.node_count

    mode = 'wb' if binary else 'w'
    with open(output_path, mode) as f:
        exporter._wl(f, "# vtk DataFile Version 3.0\n", binary)
        exporter._wl(f, f"AutoFlowCFD Export - {output_path.name}\n", binary)
        exporter._wl(f, ("BINARY\n" if binary else "ASCII\n"), binary)
        exporter._wl(f, "\n", binary)
        exporter._wl(f, "DATASET UNSTRUCTURED_GRID\n", binary)
        exporter._wl(f, "\n", binary)

        exporter._write_points(f, binary)
        exporter._write_cells(f, binary)

        exporter._wl(f, f"POINT_DATA {n_points}\n", binary)
        for name, values in vector_fields.items():
            exporter._write_vector(f, name, values, binary)
        for name, values in scalar_fields.items():
            exporter._write_scalar(f, name, values, binary)
