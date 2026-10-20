"""后处理 CLI 共用辅助函数 (从 post_commands.py 拆分)。

从 post_commands.py 拆出来（该文件原有 974 行，超过 400 行硬性拆分
阈值）：这一批"案例目录/checkpoint 定位与加载"辅助函数被
coefficients/export-vtk/report/convergence/transient-mean/
transient-rms/transient-psd 七个命令共用，与任何单个具体命令都不是
强绑定关系，独立成一个纯辅助模块最清晰（重量级命令主体保留在 *_commands.py）。
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger


def _locate_grid_file(case_path: Path, grid: Optional[str], metadata: Optional[dict] = None) -> Path:
    """后处理用的体网格文件：`--grid` > 算例目录里的 `volume_mesh.pkl` > checkpoint 记录的求解输入文件。

    解属于求解时的那张体网格，后处理必须用同一张。2026-10-09 以前这里把任何 .nas 都当成面网格、现场重新
    生成体网格（单元数与顺序都与解对不上；传入体网格 .nas 时 tetgen 直接失败），找不到时还会去算例目录里
    搜面网格。
    """
    if grid:
        logger.info(f"Using specified grid file: {grid}")
        return Path(grid)
    cached = case_path / "volume_mesh.pkl"
    if cached.exists():
        logger.info(f"Auto-detected grid file: {cached}")
        return cached
    if metadata is None:       # 调用方还没读 checkpoint（瞬态统计类命令）：取最新 checkpoint 的元数据
        metadata = _load_history_only(str(case_path))[2]
    recorded = metadata.get("input_file")
    if recorded and Path(str(recorded)).exists():
        logger.info(f"Using the solver input mesh recorded in the checkpoint: {recorded}")
        return Path(str(recorded))
    raise FileNotFoundError(
        f"找不到体网格：算例目录 {case_path} 里没有 volume_mesh.pkl，checkpoint 记录的求解输入文件 "
        f"{recorded!r} 也不存在。请用 --grid 给出求解时用的体网格（.pkl 或含 CTETRA/CPENTA 的 .nas）。")


def _load_grid_data(grid_file: Path):
    """加载求解时用的体网格：.pkl 直接反序列化；.nas 必须是体网格（`grid/conversion/source.py::
    load_volume_mesh_nas`，带边界三角形时一并读入供边界面片定组）。面网格报错——后处理不重新生成网格。"""
    logger.info("Loading grid data...")
    if grid_file.suffix.lower() == '.pkl':
        logger.info(f"Loading volume mesh from PKL: {grid_file}")
        try:
            with open(grid_file, 'rb') as f:
                grid_data = pickle.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load volume mesh from {grid_file}: {e}")
    else:
        from autoflowcfd.grid.conversion.source import load_volume_mesh_nas

        logger.info(f"Loading volume mesh from NAS: {grid_file}")
        try:
            grid_data = load_volume_mesh_nas(str(grid_file))
        except ValueError as e:
            raise ValueError(f"{e}。后处理需要求解时用的体网格（解的单元与它一一对应），不能由面网格重新生成。")
    logger.success(f"✓ Volume mesh loaded: {grid_data.node_count} nodes, {grid_data.cell_count} cells")
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


def _physical_time(metadata: dict, ckpt_file) -> float:
    """checkpoint 记录的物理时间（时间精确推进累计，`core/utils/checkpoint_time.py`）。

    Raises:
        ValueError: 没有记录——不是时间精确（瞬态）运行写出的 checkpoint，或早于 2026-10-20 写出（当时物理时间
            不推进也不记录，此前这里拿迭代数当物理时间，--warmup-time 按迭代数比较）
    """
    from autoflowcfd.core.utils.checkpoint_time import PHYSICAL_TIME_KEY

    if PHYSICAL_TIME_KEY not in metadata:
        raise ValueError(f"{ckpt_file} 没有记录物理时间：不是瞬态（时间精确）运行写出的 checkpoint，或早于 2026-10-20 "
                         "写出，无法给出时间轴")
    return float(metadata[PHYSICAL_TIME_KEY])


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
    from autoflowcfd.core.utils.checkpoint import CheckpointManager

    case_path = Path(case)
    ckpt_file = _locate_checkpoint(case_path, checkpoint)
    ckpt_manager = CheckpointManager()
    solution_data, history, iteration, metadata = ckpt_manager.load(ckpt_file)
    logger.info(f"✓ Solution loaded from iteration {iteration}")

    grid_data = _load_grid_data(_locate_grid_file(case_path, grid, metadata))

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
    from autoflowcfd.core.utils.checkpoint import CheckpointManager

    case_path = Path(case)
    ckpt_file = _locate_checkpoint(case_path, checkpoint)
    ckpt_manager = CheckpointManager()
    _solution, history, iteration, metadata = ckpt_manager.load(ckpt_file)
    return history, iteration, metadata


def _replay_history(history: dict):
    """把 checkpoint 里的收敛历史（每方程残差 + 系数的并行数组，`core/fr_solver/convergence_history.py`）重放进
    一个新的 ConvergenceAnalyzer，供 report/convergence 复用 ConvergenceAnalyzer/SimulationReport 已有的分析/导出逻辑。

    逐步力系数是压力积分部分（Cd_p / Cl_p / Cs_p，求解时设置了参考面积才有），按记录的名称原样填进分析器
    （2026-10-20 以前当作完整的 Cd / Cl 填入、Cs_p 丢掉、其余系数写 0.0）；没算的步为 nan、不填。
    """
    import math

    from autoflowcfd.postprocess import ConvergenceAnalyzer

    analyzer = ConvergenceAnalyzer()
    iterations = history.get('iterations', [])
    residuals_by_eq = history.get('residuals', {})
    coeffs_by_name = history.get('coefficients', {})
    cfl_history = history.get('cfl_history', [])

    for idx, it in enumerate(iterations):
        residuals = {eq: values[idx] for eq, values in residuals_by_eq.items() if idx < len(values)}
        cfl = cfl_history[idx] if idx < len(cfl_history) else 0.0
        coefficients = {name: float(values[idx]) for name, values in coeffs_by_name.items()
                        if idx < len(values) and not math.isnan(values[idx])}
        analyzer.add_iteration(iteration=it, residuals=residuals, cfl=cfl, coefficients=coefficients)

    return analyzer


def _export_point_fields_vtk(
    output_path: Path,
    grid_data,
    vector_fields: Dict[str, np.ndarray],
    scalar_fields: Dict[str, np.ndarray],
    binary: bool = False,
) -> None:
    """把节点分辨率的场（瞬态统计的时均/RMS）写成只含 POINT_DATA 的 legacy VTK（与体网格导出同一写入器）。"""
    from autoflowcfd.postprocess.vtk_export_legacy import export_volume

    export_volume(Path(output_path), grid_data, {}, {**vector_fields, **scalar_fields}, binary)
