"""AutoFlowCFDAPI 的配置和恢复辅助方法。

从 api.py 拆出，控制单文件行数。包含稳态/瞬态配置创建、配置文件加载
和从检查点恢复仿真等方法。
"""

from typing import Any, Dict
from pathlib import Path
from loguru import logger

from autoflowcfd.config.solver_config import (
    SteadyConfig, TransientConfig, BackendType, TurbulenceModel,
)


def api_create_steady_config(
    self,
    backend: str = "cpu",
    order: int = 2,
    max_iter: int = 1000,
    turbulence: str = "sst_kw",
    **kwargs
) -> SteadyConfig:
    """创建稳态仿真配置（委托函数）。"""
    turb_model_map = {
        'none': TurbulenceModel.NONE,
        'sst_kw': TurbulenceModel.SST_KW,
        'sa': TurbulenceModel.SA,
        'des': TurbulenceModel.DES,
        'ddes': TurbulenceModel.DDES,
        'les': TurbulenceModel.LES,
    }
    turb_model = turb_model_map.get(turbulence.lower(), TurbulenceModel.SST_KW)

    return SteadyConfig(
        backend=BackendType(backend),
        order=order,
        turbulence=turb_model,
        max_iter=max_iter,
        **kwargs
    )


def api_create_transient_config(
    self,
    backend: str = "cpu",
    order: int = 2,
    time_method: str = "dual-time",
    turbulence_model: str = None,
    mode: str = None,
    max_iter: int = 1000,
    dt: float = 1e-4,
    total_time: float = None,
    **kwargs
) -> TransientConfig:
    """创建瞬态仿真配置（委托函数）。"""
    from autoflowcfd.config.solver_config import TimeIntegrationScheme

    time_scheme_map = {
        'backward_euler': TimeIntegrationScheme.BACKWARD_EULER,
        'rk2': TimeIntegrationScheme.RK2,
        'rk3': TimeIntegrationScheme.RK3,
        'ab3': TimeIntegrationScheme.AB3,
        'dual-time': TimeIntegrationScheme.BACKWARD_EULER,
        'imex': TimeIntegrationScheme.RK3,
    }
    time_scheme = time_scheme_map.get(time_method, TimeIntegrationScheme.RK3)

    if mode is not None and turbulence_model is None:
        turbulence_model = mode
    if turbulence_model is None:
        turbulence_model = "sst"

    return TransientConfig(
        backend=BackendType(backend),
        order=order,
        turbulence=TurbulenceModel(turbulence_model),
        time_scheme=time_scheme,
        dt=dt,
        total_time=total_time if total_time is not None else dt * max_iter,
        **kwargs
    )


def api_load_config(self, config_file: str) -> Dict[str, Any]:
    """加载配置文件（委托函数）。支持 JSON 和 YAML。"""
    file_path = Path(config_file)

    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    if file_path.suffix in ['.json']:
        import json
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif file_path.suffix in ['.yml', '.yaml']:
        try:
            import yaml
            with open(config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except ImportError:
            logger.warning("PyYAML not installed, cannot load YAML config")
            return {}
    else:
        logger.warning(f"Unsupported config file format: {file_path.suffix}")
        return {}


def api_resume_simulation(
    self,
    checkpoint_file: str,
    max_iter: int = 0,
    dt: float = 1e-3,
    tol: float = 1e-6,
    backend: str = None,
    surface_mesh: str = None,
    threads: int = -1,
) -> Any:
    """从检查点恢复仿真（委托函数）。

    此前这里把 `self.grid_data`（表面网格，且往往在 resume 场景下根本
    没设置过——恢复仿真通常是一个独立的新会话，不会先调用 load_grid）
    直接传给 `FRSolver(mesh=self.grid_data, ...)`，随后调用
    `solver.solve(initial_solution=solution, start_iteration=iteration,
    **kwargs)`——但 FRSolver.solve() 真实签名是
    `solve(max_iter, dt, tol, checkpoint_callback)`，根本不接受
    initial_solution/start_iteration，必然 TypeError（V2.0 专家组评审
    逐行核实）。且 `self.grid_data is None` 分支返回一个只含拍扁体积
    平均解的占位 SolverResult，从不真正恢复求解器。

    改为直接复用 CLI `solve resume`（cli/solve_commands.py）已验证
    正确的重建逻辑（cli/solve_checkpoint_io.py::rebuild_solver_from_
    checkpoint）——checkpoint 的 metadata 自带重建 HighOrderMesh +
    FRSolver 所需的全部参数（input_file/order/turbulence_model/backend/
    自由来流条件），不依赖调用方是否设置过 self.grid_data。

    Args:
        checkpoint_file: checkpoint 文件路径
        max_iter: 恢复后继续迭代的次数；0（默认）表示只恢复状态、不
            继续求解，直接返回恢复后的（未收敛）结果
        dt, tol: 继续迭代时使用的时间步长/收敛容差
        backend: 后端覆盖，None 时沿用 checkpoint 记录的原始后端
        surface_mesh: checkpoint 记录的 input_file 若是 .nas 体网格，
            需要提供原始面网格来反推边界分组
        threads: CPU 后端 numba 并行线程数

    Returns:
        SolverResult
    """
    if not Path(checkpoint_file).exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")

    logger.info(f"Resuming simulation from checkpoint: {checkpoint_file}")

    from autoflowcfd.cli.solve_checkpoint_io import rebuild_solver_from_checkpoint

    solver, iteration, metadata = rebuild_solver_from_checkpoint(
        checkpoint_file, backend=backend, surface_mesh=surface_mesh, threads=threads,
    )
    self.solver = solver

    logger.info(f"State restored from checkpoint (iter={iteration})")

    if max_iter <= 0:
        # SolverResult 真实字段只有 converged/iterations/final_residual
        # （core/fr_solver/state.py）——不带 solution/residuals，恢复后
        # 的解场请从 self.solver.state.U 读取（已在上面设好 self.solver）。
        from autoflowcfd.core.fr_solver.state import SolverResult
        return SolverResult(
            converged=False,
            iterations=iteration,
            final_residual=float('nan'),
        )

    result = solver.solve(max_iter=max_iter, dt=dt, tol=tol)
    return result
