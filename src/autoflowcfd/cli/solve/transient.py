"""`solve transient` 命令（URANS / DES / LES）：选项与公共前段，按后端执行交给 `cli/solve/run.py::run_solve`
（与 `solve steady` 同一份）；命令组定义见同目录 `commands.py`。
"""

from typing import Optional

import click


from autoflowcfd.cli.solve.solver_factory import validate_backend_options
from autoflowcfd.cli.solve.config_file import check_physical_ranges, config_file_overrides
from autoflowcfd.cli.solve.commands import solve
from autoflowcfd.cli.solve.shared_options import config_default, shared_solve_options
from autoflowcfd.config.solver_config import TransientConfig
from autoflowcfd.core.time_integration.base import DEFAULT_DUAL_TIME_STEPS

_TIME_METHOD_CHOICES = ["dual-time", "rk3"]


@solve.command(name='transient')
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--time-method", "-t",
              type=click.Choice(_TIME_METHOD_CHOICES),
              default=config_default(TransientConfig, "time_scheme", _TIME_METHOD_CHOICES), show_default=True,
              help="时间推进方法，两档都是时间精确的。dual-time（默认）：每个物理步 --dt 是若干个带 BDF 物理时间项"
                   "的隐式 Newton 步，步长不受显式稳定条件限制。rk3：显式 SSP-RK3、全场同一物理步长——每个 --dt 切成"
                   "若干子步，子步长取全场显式稳定步长（--cfl-start）的最小值、最后一个子步恰好落在 t+dt；CFL 固定、"
                   "低马赫预处理关闭（core/time_integration/time_accurate.py）。")
@click.option("--turbulence-model", "-m",
              type=click.Choice(["sst", "sa", "ddes", "iddes", "wmles", "les"]),
              default=config_default(TransientConfig, "turbulence"), show_default=True,
              help="湍流模型。sa 为 SA-neg URANS（core/turbulence/sa）；ddes/iddes 为 SST-DDES/IDDES（Gritskevich et al. "
                   "2012，core/turbulence/des/）。ddes/iddes/les/wmles 在 VELOCITY_INLET 边界启用合成湍流入口")
@click.option("--max-iter", "-n", type=int, default=config_default(TransientConfig, "max_iter"), show_default=True,
              help="物理时间步数（给了 --physical-time 时按 physical_time/dt 换算，忽略本选项）")
@click.option("--dt", type=float, default=config_default(TransientConfig, "dt"), show_default=True,
              help="物理时间步长 (秒)")
@click.option('--cfl-start', type=float, default=config_default(TransientConfig, 'cfl_init'),
              help='CFL 数。--time-method rk3：显式子步的固定 CFL（全场显式稳定步长按它取，默认 0.03，'
                   'adaptive_cfl/policy.py 唯一来源）；dual-time：内迭代伪时间 CFL 的起点（SER 律，默认 1e4）。')
@click.option('--cfl-max', type=float, default=config_default(TransientConfig, 'cfl_max'),
              help='dual-time 内迭代伪时间 CFL 的上限（SER 律）。rk3 档 CFL 固定为 --cfl-start，不用它。')
@click.option('--cfl-min', type=float, default=config_default(TransientConfig, 'cfl_min'),
              help='dual-time 内迭代伪时间 CFL 的下限（SER 律）。rk3 档 CFL 固定为 --cfl-start，不用它。')
@click.option("--physical-time", type=float, default=config_default(TransientConfig, "total_time"),
              help="总物理时间（秒）；不给时按 --max-iter 个物理步推进")
@click.option('--dual-time-inner-iter', type=int, default=DEFAULT_DUAL_TIME_STEPS,
              help=f'--time-method dual-time 时每个物理步的内迭代（带 BDF 物理时间项的 Newton 步）次数上限，'
                   f'残差降到该物理步起点的 1e-3 即提前结束，通常 2~7 次（默认上限 {DEFAULT_DUAL_TIME_STEPS}）')
@click.option("--probe", "probes", nargs=3, type=float, multiple=True,
              help="探针位置 X Y Z（米，可重复给出）：每个物理步在该点求 FR 解多项式的值，压力与速度写进输出目录的 "
                   "probes.csv（布局在 probes.json），供 post transient-psd 使用；续算接着写")
@click.option('--init-from', 'init_checkpoint', type=click.Path(exists=True), default=None,
              help='从稳态 checkpoint 文件初始化瞬态求解器（典型工作流：先稳态 SST 收敛，'
                   '再从该流场启动 DES/LES 瞬态计算，避免从均匀流场直接启动需要极长的瞬态发展时间）。'
                   'checkpoint 阶数可以低于 --order（平均流与湍流场精确延拓到目标阶数），不能高于；'
                   '从 checkpoint 起步时直接在 --order 上推进，不做阶数爬坡')
@shared_solve_options(TransientConfig)
@config_file_overrides
def transient(input_file: str, backend: str, order: int, time_method: str,
              turbulence_model: str, max_iter: int, phase_max_iter: Optional[int], residual_drop_threshold: float,
              dt: float, cfl_start: Optional[float], cfl_max: Optional[float], cfl_min: Optional[float],
              aoa_deg: float, aos_deg: float,
              physical_time: float,
              output_dir: str, surface_mesh: Optional[str],
              skip_quality_check: bool, reference_area: Optional[float],
              dual_time_inner_iter: int, threads: int, probes: tuple, init_checkpoint: Optional[str],
              turbulence_intensity: float, viscosity_ratio: float, sem_num_eddies: Optional[int],
              mu_molecular: float, rho_inf: float, vel_inf: float, p_inf: float,
              config_path: Optional[str], n_ranks: int, multi_gpu: bool,
              fully_distributed: bool, gpu_device: Optional[int],
              artificial_viscosity_enabled: bool, artificial_viscosity_alpha: float, wall_treatment: str,
              checkpoint_interval: int, units: str = "mm", bc_type: tuple = ()) -> None:
    """运行瞬态 FR 仿真 (DES/LES)。

    \f

    Args:
        input_file: 输入体网格文件 - .pkl 或 .nas 体网格（需要配合
            --surface-mesh）- 先用 'grid generate-volume' 或 'grid
            import-volume' 从面网格生成/导入体网格
        backend: 计算后端
        order: FR 阶数
        time_method: 时间推进方法
        turbulence_model: 湍流模型 (推荐 DDES 或 LES)
        max_iter: 最大迭代次数
        phase_max_iter: Order Continuation(--order>=1 时触发)非最终阶段各自的
            最大迭代步数上限，None 时按阶段数均分（目标阶数总是吃掉剩余步数）
        residual_drop_threshold: Order Continuation 单阶段提前升阶所需的残差
            下降倍数，默认100
        dt: 时间步长
        physical_time: 总物理时间（秒）
        output_dir: 输出目录
        surface_mesh: 原始面网格路径，input_file 是 .nas 体网格时必填
        units: .nas 体网格与面网格的坐标单位（入口处转存成输出目录下的米制 volume_mesh.pkl）
        skip_quality_check: 跳过求解前的网格质量门检查
        init_checkpoint: 从稳态 checkpoint 初始化（可选）
        probes: 探针位置（米），每个物理步采样（`core/probes`）
    """
    from autoflowcfd.core.mpi import is_root

    if is_root():
        print("=== Starting Transient FR Simulation ===")
    from autoflowcfd.cli.solve.mesh_loader import prepare_solver_input
    from autoflowcfd.grid.nas_io.boundary_types import parse_bc_type_options
    input_file, surface_mesh = prepare_solver_input(input_file, surface_mesh, units, output_dir,
                                                    parse_bc_type_options(bc_type))

    # `--config` 已由 `config_file_overrides` 在调用前应用；CLI 路径不经过配置类的构造校验，物理量范围在这里拦截
    check_physical_ranges(turbulence_intensity, viscosity_ratio, mu_molecular, rho_inf, vel_inf, p_inf)
    if physical_time:
        from autoflowcfd.core.time_integration.base import physical_time_steps
        max_iter = physical_time_steps(float(physical_time), dt)

    from autoflowcfd.cli.solve.run import SolveRun, physics_from_options, run_solve
    from autoflowcfd.core.time_integration.base import scheme_from_name

    validate_backend_options(backend, n_ranks, multi_gpu)
    time_scheme = scheme_from_name(time_method)
    run_solve(SolveRun(
        input_file=input_file, surface_mesh=surface_mesh, skip_quality_check=skip_quality_check,
        output_dir=output_dir, checkpoint_interval=checkpoint_interval,
        backend=backend, n_ranks=n_ranks, multi_gpu=multi_gpu, fully_distributed=fully_distributed,
        gpu_device=gpu_device, threads=threads,
        order=order, turbulence_model=turbulence_model, time_scheme=time_scheme,
        # 显式档做时间精确推进（全场同一物理步长的子步，time_integration/time_accurate.py）；dual-time 本身时间精确
        time_accurate=time_scheme != scheme_from_name("dual-time"), dual_time_inner_iter=dual_time_inner_iter,
        # CFL：rk3 档为显式子步的固定 CFL（cfl_start），dual-time 档为内迭代伪时间 CFL 的 SER 律参数
        cfl_start=cfl_start, cfl_max=cfl_max, cfl_min=cfl_min,
        # 低马赫伪时间预处理的请求值：dual-time 不用它，显式时间精确推进强制关闭（configure_time_stepping）
        low_mach_precond=True,
        physics=physics_from_options(rho_inf, vel_inf, p_inf, aoa_deg, aos_deg, mu_molecular, turbulence_intensity,
                                     viscosity_ratio, artificial_viscosity_enabled, artificial_viscosity_alpha,
                                     wall_treatment),
        sem_num_eddies=sem_num_eddies, reference_area=reference_area, init_checkpoint=init_checkpoint,
        probes=tuple(probes),
        # 瞬态不以残差为停止条件，跑满指定步数（tol=0）
        max_iter=max_iter, dt=dt, tol=0.0, phase_max_iter=phase_max_iter,
        residual_drop_threshold=residual_drop_threshold))
