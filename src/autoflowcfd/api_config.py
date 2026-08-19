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


def api_resume_simulation(self, checkpoint_file: str, **kwargs) -> Any:
    """从检查点恢复仿真（委托函数）。"""
    from autoflowcfd.core.utils.checkpoint import CheckpointManager
    from autoflowcfd.core import FRSolver
    from autoflowcfd.core.time_integration.base import TimeIntegrationScheme

    if not Path(checkpoint_file).exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")

    logger.info(f"Resuming simulation from checkpoint: {checkpoint_file}")

    config = SteadyConfig()
    manager = CheckpointManager(config)
    solution, history, iteration, metadata = manager.load(checkpoint_file)

    logger.info(
        f"Checkpoint loaded: iteration={iteration}, "
        f"solution shape={solution.shape}"
    )

    if self.grid_data is not None:
        solver = FRSolver(
            mesh=self.grid_data,
            order=getattr(config, 'order', 2),
            turb_model_name=getattr(config, 'turbulence', 'sst_kw'),
            time_scheme=TimeIntegrationScheme.SSP_RK3,
        )
        result = solver.solve(
            initial_solution=solution,
            start_iteration=iteration,
            **kwargs,
        )
        return result
    else:
        logger.warning("grid_data 未设置，无法重建求解器")
        from autoflowcfd.core.fr_solver.state import SolverResult
        return SolverResult(
            iterations=iteration,
            converged=False,
            solution=solution,
            residuals=history.get('residuals', {}),
        )
