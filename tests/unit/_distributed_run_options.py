"""完全分布式加载入口（`build_fully_distributed_rank_package` / `distributed_mesh_load_v2`）必传的运行配置：测试共用的
一组取值（CLI 从命令行选项给出），个别测试按需覆盖。"""

from autoflowcfd.core.time_integration.base import DEFAULT_DUAL_TIME_STEPS, TimeIntegrationScheme
from autoflowcfd.core.turbulence.wall_model.treatment import DEFAULT_WALL_TREATMENT


def distributed_run_options(**overrides) -> dict:
    options = dict(turb_model_name="NONE", turbulence_intensity=0.01, viscosity_ratio=5.0,
                   time_scheme=TimeIntegrationScheme.SSP_RK3, dual_time_inner_iter=DEFAULT_DUAL_TIME_STEPS,
                   cfl_start=None, cfl_max=None, cfl_min=None, time_accurate=False, adaptive_cfl=True,
                   artificial_viscosity_enabled=False, artificial_viscosity_alpha=1.0,
                   wall_treatment=DEFAULT_WALL_TREATMENT, low_mach_precond=True, sem_num_eddies=None)
    unknown = set(overrides) - set(options)
    if unknown:
        raise TypeError(f"不是运行配置项: {sorted(unknown)}")
    options.update(overrides)
    return options
