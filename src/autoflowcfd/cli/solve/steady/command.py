"""`solve steady` 命令：click 选项、公共前段（`--config` 覆盖与物理量范围校验）与后端分派。

四个后端分支在同目录的 `multi_gpu`/`single_gpu`/`cpu_mpi`/`cpu_single` 模块
（2026-09-25 从单文件 `cli/solve/steady.py` 拆出）；命令组定义见 `cli/solve/commands.py`。
"""

import click


from autoflowcfd.cli.solve.solver_factory import validate_backend_options
from autoflowcfd.cli.solve.config_file import check_physical_ranges, config_file_overrides
from autoflowcfd.cli.solve.commands import solve
from autoflowcfd.cli.solve.shared_options import config_default, shared_solve_options
from autoflowcfd.config.solver_config import SteadyConfig

_TURBULENCE_CHOICES = ['none', 'sst', 'sa', 'ddes', 'iddes', 'wmles', 'les']
_TIME_SCHEME_CHOICES = ['rk3', 'newton-krylov']

from .multi_gpu import _run_multi_gpu
from .cpu_mpi import _run_cpu_mpi
from .single_node import _run_single_node


@solve.command(name='steady')
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--turbulence-model', '-m', type=click.Choice(_TURBULENCE_CHOICES),
              default=config_default(SteadyConfig, 'turbulence'), show_default=True,
              help='湍流模型。sa 为 SA-neg（Allmaras, Johnson & Spalart 2012，为高阶离散设计的负值鲁棒 '
                   'Spalart-Allmaras；高阶 P2/P3 隐式稳态推荐，SST 的 C0 折点使其在高阶下收敛缓慢，见 '
                   'core/turbulence/sa 模块文档）。ddes/iddes/les/wmles 在 VELOCITY_INLET 边界启用合成湍流入口')
@click.option('--max-iter', '-n', type=int, default=config_default(SteadyConfig, 'max_iter'), show_default=True,
              help='最大迭代步数')
@click.option('--time-scheme', '-t', type=click.Choice(_TIME_SCHEME_CHOICES),
              default=config_default(SteadyConfig, 'time_scheme', _TIME_SCHEME_CHOICES), show_default=True,
              help='稳态伪时间推进格式。rk3：显式 SSP-RK3（CFL 受显式稳定极限约束，'
                   '真实网格上走完一个绕体特征时间要上万步）；newton-krylov：矩阵自由 '
                   'Newton-Krylov + 伪瞬态延拓（SER CFL 律 + 单元块 Jacobi 预处理，见 '
                   'core/time_integration/implicit/），收敛步数由非线性程度而不是最小'
                   '单元决定。全部后端（单机 CPU/GPU、--n-ranks>1 的 CPU 分布式与'
                   '--multi-gpu，含 --fully-distributed）同一份算法：分布式下内积与'
                   '线搜索跨 rank 归约、块 Jacobi 用全局一致着色。')
@click.option('--cfl-start', type=float, default=config_default(SteadyConfig, 'cfl_init'),
              help='自适应 CFL 初始值，缺省按 --time-scheme 取该格式 CFL 律的默认值（rk3 0.03；newton-krylov 5，adaptive_cfl/policy.py）。残差'
                   '不下降时 CFL 停在这个值；复杂网格上起步就发散可调低。默认值的实测依据见 core/time_integration/adaptive_cfl 模块文档')
@click.option('--cfl-max', type=float, default=config_default(SteadyConfig, 'cfl_max'),
              help='自适应 CFL 上限，缺省按 --time-scheme 取：rk3 0.06（低于实测失效点并留裕度——越界一次之后收缩救不回来）；newton-krylov 1e8（隐式没有线性稳定极限）'
                   '。依据见 core/time_integration/adaptive_cfl 模块文档')
@click.option('--cfl-min', type=float, default=config_default(SteadyConfig, 'cfl_min'),
              help='自适应 CFL 下限，缺省按 --time-scheme 取：rk3 0.01；newton-krylov 0.5（更小的伪时间步由 Newton 步自身的 dtau 缩档负责）。下限只限制收'
                   '缩，不会抬高 CFL')
@click.option('--init-from', 'init_checkpoint', type=click.Path(exists=True), default=None,
              help='以已有 checkpoint 的场为初场（例如低阶解收敛后直接在更高阶上继续，不重走低阶的爬坡）。'
                   'checkpoint 阶数可以低于 --order（平均流与湍流场精确延拓），不能高于；起步后直接在 '
                   '--order 上推进，--tol 相对于该阶数上的第一个残差。全部后端可用')
@click.option('--tol', type=float, default=config_default(SteadyConfig, 'convergence_tol'), show_default=True,
              help='相对收敛容差（目标阶数的残差下降到初值的 tol 倍即收敛；Order Continuation 低一阶放宽 10 倍）')
@click.option('--low-mach-precond/--no-low-mach-precond', 'low_mach_precond',
              default=config_default(SteadyConfig, 'low_mach_precond'), show_default=True,
              help='低马赫数伪时间预处理（rk3/newton-krylov；见 core/utils/preconditioning.py）：不改变定常解，'
                   '低马赫数下把声学刚性从伪时间推进里去掉。设置写入 checkpoint，续算沿用')
@shared_solve_options(SteadyConfig)
@config_file_overrides
def solve_steady(input_file, backend, order, turbulence_model, max_iter, time_scheme, cfl_start, cfl_max, cfl_min,
                 aoa_deg, aos_deg, phase_max_iter, residual_drop_threshold, output_dir, checkpoint_interval, surface_mesh, skip_quality_check, reference_area, threads, n_ranks, fully_distributed, gpu_device, multi_gpu, turbulence_intensity, viscosity_ratio, sem_num_eddies, mu_molecular, rho_inf, vel_inf, p_inf, config_path, tol, artificial_viscosity_enabled, artificial_viscosity_alpha, init_checkpoint, wall_treatment, low_mach_precond, units):
    """执行稳态 FR 求解（P0~P3，全部湍流模型）。

    输入文件必须是体网格 - .pkl（`grid generate-volume`/`grid
    import-volume` 的输出）或 .nas 体网格（需要配合 --surface-mesh 反推边界
    分组）。求解前会强制检查网格质量门，除非传了 --skip-quality-check。
    """
    print("=== Starting Steady FR Simulation ===")
    # .nas 体网格按 --units 导入并转存成输出目录下的米制 volume_mesh.pkl，之后全程用它
    from autoflowcfd.cli.solve.mesh_loader import prepare_solver_input
    input_file, surface_mesh = prepare_solver_input(input_file, surface_mesh, units, output_dir)

    # `--config` 已由 `config_file_overrides` 在调用前应用；CLI 路径不经过配置类的构造校验，物理量范围在这里拦截
    check_physical_ranges(turbulence_intensity, viscosity_ratio, mu_molecular, rho_inf, vel_inf, p_inf)
    print(f"\nInput Grid : {input_file}")
    print(f"Backend    : {backend} | Order: P{order} | Method: {time_scheme}")
    print(f"Turbulence : {turbulence_model} | Max Iter: {max_iter}")
    if n_ranks > 1:
        print(f"MPI Ranks  : {n_ranks} (domain decomposition)")
    if init_checkpoint:
        print(f"Init From  : {init_checkpoint}")
    print("Wall Dist : exact point-to-wall-surface distance (AABB tree)\n")

    # 1. 按后端分派（各分支的参数由 AST 分析得出，逐名传入）。单机（n_ranks == 1）的 CPU 与单 GPU 同一个分支
    validate_backend_options(backend, n_ranks, multi_gpu)
    if backend == 'gpu' and multi_gpu:
        _run_multi_gpu(
            aoa_deg=aoa_deg,
            artificial_viscosity_alpha=artificial_viscosity_alpha,
            wall_treatment=wall_treatment,
            low_mach_precond=low_mach_precond,
            artificial_viscosity_enabled=artificial_viscosity_enabled,
            aos_deg=aos_deg,
            cfl_max=cfl_max,
            cfl_min=cfl_min,
            cfl_start=cfl_start,
            checkpoint_interval=checkpoint_interval,
            fully_distributed=fully_distributed,
            gpu_device=gpu_device,
            init_checkpoint=init_checkpoint,
            input_file=input_file,
            max_iter=max_iter,
            mu_molecular=mu_molecular,
            n_ranks=n_ranks,
            order=order,
            output_dir=output_dir,
            p_inf=p_inf,
            phase_max_iter=phase_max_iter,
            reference_area=reference_area,
            residual_drop_threshold=residual_drop_threshold,
            rho_inf=rho_inf,
            skip_quality_check=skip_quality_check,
            surface_mesh=surface_mesh,
            time_scheme=time_scheme,
            tol=tol,
            turbulence_intensity=turbulence_intensity,
            turbulence_model=turbulence_model,
            vel_inf=vel_inf,
            viscosity_ratio=viscosity_ratio,
            sem_num_eddies=sem_num_eddies,
        )
    elif n_ranks > 1:
        _run_cpu_mpi(
            aoa_deg=aoa_deg,
            artificial_viscosity_alpha=artificial_viscosity_alpha,
            wall_treatment=wall_treatment,
            low_mach_precond=low_mach_precond,
            artificial_viscosity_enabled=artificial_viscosity_enabled,
            aos_deg=aos_deg,
            cfl_max=cfl_max,
            cfl_min=cfl_min,
            cfl_start=cfl_start,
            checkpoint_interval=checkpoint_interval,
            fully_distributed=fully_distributed,
            init_checkpoint=init_checkpoint,
            input_file=input_file,
            max_iter=max_iter,
            mu_molecular=mu_molecular,
            n_ranks=n_ranks,
            order=order,
            output_dir=output_dir,
            p_inf=p_inf,
            phase_max_iter=phase_max_iter,
            reference_area=reference_area,
            residual_drop_threshold=residual_drop_threshold,
            rho_inf=rho_inf,
            skip_quality_check=skip_quality_check,
            surface_mesh=surface_mesh,
            threads=threads,
            time_scheme=time_scheme,
            tol=tol,
            turbulence_intensity=turbulence_intensity,
            turbulence_model=turbulence_model,
            vel_inf=vel_inf,
            viscosity_ratio=viscosity_ratio,
            sem_num_eddies=sem_num_eddies,
        )
    else:
        _run_single_node(
            aoa_deg=aoa_deg,
            aos_deg=aos_deg,
            artificial_viscosity_alpha=artificial_viscosity_alpha,
            wall_treatment=wall_treatment,
            low_mach_precond=low_mach_precond,
            artificial_viscosity_enabled=artificial_viscosity_enabled,
            backend=backend,
            cfl_max=cfl_max,
            cfl_min=cfl_min,
            cfl_start=cfl_start,
            checkpoint_interval=checkpoint_interval,
            gpu_device=gpu_device,
            init_checkpoint=init_checkpoint,
            input_file=input_file,
            max_iter=max_iter,
            mu_molecular=mu_molecular,
            order=order,
            output_dir=output_dir,
            p_inf=p_inf,
            phase_max_iter=phase_max_iter,
            reference_area=reference_area,
            residual_drop_threshold=residual_drop_threshold,
            rho_inf=rho_inf,
            sem_num_eddies=sem_num_eddies,
            skip_quality_check=skip_quality_check,
            surface_mesh=surface_mesh,
            threads=threads,
            time_scheme=time_scheme,
            tol=tol,
            turbulence_intensity=turbulence_intensity,
            turbulence_model=turbulence_model,
            vel_inf=vel_inf,
            viscosity_ratio=viscosity_ratio,
        )
