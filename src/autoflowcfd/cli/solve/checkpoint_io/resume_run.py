"""单机续算的"求解 + 落盘"（CLI `solve resume` 单机分支与 Python API `resume_simulation` 共用的唯一实现）。

输出写在 checkpoint 所在算例目录（`<output_dir>/checkpoints/xxx.h5` 的 `<output_dir>`）；中间 checkpoint 的文件名用续算
起点的绝对迭代数作偏移，不会覆盖续算之前已有的同名文件。2026-10-19 以前 API 续算完全不落盘。
"""

from pathlib import Path
from typing import Optional

from .write import periodic_checkpoint_callback, write_single_node_outputs


def continue_single_node(solver, iteration: int, metadata: dict, checkpoint_file: str, *, max_iter: int, dt: float,
                         tol: float, checkpoint_interval: int, phase_max_iter: Optional[int],
                         residual_drop_threshold: float):
    """已从 checkpoint 重建的单机求解器继续求解 `max_iter` 步并落盘；返回 `SolverResult`。

    Args:
        iteration: checkpoint 记录的迭代数（续算起点）
        metadata: 重建所用的 checkpoint 元数据（input_file/turbulence_model/backend/surface_mesh）
        dt: 已解析的时间步长（`core/utils/checkpoint_time.py::resume_time_step`）
        tol: 相对收敛容差；时间精确的续算按 `run_tolerance` 跑满全部步
    """
    from autoflowcfd.core.utils.checkpoint_time import run_tolerance

    output_dir = str(Path(checkpoint_file).parent.parent)
    names = (metadata["input_file"], metadata["turbulence_model"], metadata["backend"])
    surface_mesh = metadata.get("surface_mesh")
    result = solver.solve(
        max_iter=max_iter, dt=dt, tol=run_tolerance(tol, solver),
        checkpoint_callback=periodic_checkpoint_callback(checkpoint_interval, output_dir, *names,
                                                         surface_mesh=surface_mesh, iteration_offset=iteration),
        phase_max_iter=phase_max_iter, residual_drop_threshold=residual_drop_threshold)
    write_single_node_outputs(solver, output_dir, iteration + result.iterations, *names, surface_mesh=surface_mesh)
    return result
