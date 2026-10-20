"""`solve transient` 命令 (DES/LES)

命令组定义见同目录 `commands.py`。
"""

from typing import Optional

import click


from autoflowcfd.cli.solve.solver_factory import validate_backend_options
from autoflowcfd.cli.solve.checkpoint_io import periodic_checkpoint_callback, write_single_node_outputs
from autoflowcfd.cli.solve.mesh_loader import load_mesh_for_solver
from autoflowcfd.cli.solve.checkpoint_io import start_single_node_from_checkpoint
from autoflowcfd.cli.solve.config_file import check_physical_ranges, config_file_overrides
from autoflowcfd.cli.solve.aero_coefficients import _report_aerodynamic_coefficients, resolve_reference_area
from autoflowcfd.cli.solve.commands import solve
from autoflowcfd.cli.solve.shared_options import config_default, shared_solve_options
from autoflowcfd.config.solver_config import TransientConfig

_TIME_METHOD_CHOICES = ["dual-time", "rk3"]
from autoflowcfd.cli.solve.solver_factory import build_single_node_solver
from autoflowcfd.cli.solve.transient_distributed import _solve_transient_distributed
from autoflowcfd.core.time_integration.base import DEFAULT_DUAL_TIME_STEPS


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
              checkpoint_interval: int, units: str = "mm") -> None:
    """运行瞬态 FR 仿真 (DES/LES)。

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
    print("=== Starting Transient FR Simulation (DES/LES) ===")
    from autoflowcfd.cli.solve.mesh_loader import prepare_solver_input
    input_file, surface_mesh = prepare_solver_input(input_file, surface_mesh, units, output_dir)

    # `--config` 已由 `config_file_overrides` 在调用前应用；CLI 路径不经过配置类的构造校验，物理量范围在这里拦截
    check_physical_ranges(turbulence_intensity, viscosity_ratio, mu_molecular, rho_inf, vel_inf, p_inf)
    print(f"\nInput Grid : {input_file}")
    print(f"Backend    : {backend} | Order: P{order} | Method: {time_method}")
    print(f"Turbulence : {turbulence_model} | dt: {dt:.2e}")
    if init_checkpoint:
        print(f"Init From  : {init_checkpoint}")
    if physical_time:
        from autoflowcfd.core.time_integration.base import physical_time_steps
        max_iter = physical_time_steps(float(physical_time), dt)
        print(f"Physical Time: {physical_time}s | Iterations: {max_iter}\n")
    else:
        print(f"Iterations : {max_iter}\n")

    # 词汇->枚举唯一事实来源（见 core/time_integration/base.py）。
    from autoflowcfd.core.time_integration.base import scheme_from_name

    # 后端组合校验（与 `solve steady/resume` 同一个函数）
    validate_backend_options(backend, n_ranks, multi_gpu)
    # 显式档做时间精确推进（全场同一物理步长的子步，time_integration/time_accurate.py）；dual-time 本身时间精确
    time_accurate = scheme_from_name(time_method) != scheme_from_name("dual-time")

    if n_ranks > 1:
        # 分布式瞬态求解路径（2026-09-02 补齐——此前本命令完全没有
        # 分布式支持，DUAL_TIME/DES/LES 瞬态仿真只能单机跑，与
        # solve_steady_command.py 已有的分布式覆盖不一致；同一批还
        # 补齐了 DistributedFRSolver/MultiGPUDistributedSolver 对
        # DUAL_TIME 本身的支持，见 core/mpi/distributed_solver.py 与
        # core/gpu/distributed/gpu_distributed.py 对应说明——没有这里
        # 的 CLI 入口，那两处修复也无法被真正用到）。
        # --init-from（2026-09-02 续接）：三条分布式路径已真正接入
        # （`restore_distributed_state_from_checkpoint`，见 core/mpi/
        # distributed_checkpoint.py 模块文档——复用 `gather_global_
        # state`/`scatter_local_state` 这套既有基础设施，"没有实现"
        # 从一开始就不是设计上的限制，只是没人接上），不再拒绝。
        # --phase-max-iter/--residual-drop-threshold（2026-09-02 续接）：
        # 三条分布式路径已真正接入 Order Continuation（见
        # solve_steady_command.py 同一处修复文档），不再拒绝，直接
        # 透传给 `_solve_transient_distributed`。
        _solve_transient_distributed(
            input_file, order, surface_mesh, skip_quality_check,
            scheme_from_name(time_method), dual_time_inner_iter,
            turbulence_model, max_iter, dt, output_dir,
            reference_area, threads, turbulence_intensity, viscosity_ratio, sem_num_eddies,
            mu_molecular, rho_inf, vel_inf, p_inf,
            n_ranks, multi_gpu, fully_distributed, gpu_device,
            checkpoint_interval, phase_max_iter, residual_drop_threshold,
            init_checkpoint,
            cfl_start=cfl_start, cfl_max=cfl_max, cfl_min=cfl_min, time_accurate=time_accurate,
            artificial_viscosity_enabled=artificial_viscosity_enabled,
            artificial_viscosity_alpha=artificial_viscosity_alpha,
            wall_treatment=wall_treatment,
            aoa_deg=aoa_deg, aos_deg=aos_deg, probes=probes,
        )
        return

    # 1. 网格加载与处理（含求解前质量门检查）
    mesh, volume_data = load_mesh_for_solver(
        input_file, order, surface_mesh=surface_mesh, skip_quality_check=skip_quality_check,
    )

    # 2. 映射时间推进方法
    time_scheme = scheme_from_name(time_method)

    # 3. 初始化求解器（CPU / 单 GPU；输运湍流模型（含 DES）带上壁距，LES/WMLES 不需要）
    solver = build_single_node_solver(
        backend, mesh, volume_data,
        gpu_device=gpu_device,
        order=order,
        turb_model_name=turbulence_model.upper(),
        time_scheme=time_scheme,
        dual_time_inner_iter=dual_time_inner_iter,
        n_threads=threads,
        turbulence_intensity=turbulence_intensity,
        viscosity_ratio=viscosity_ratio,
        sem_num_eddies=sem_num_eddies,
        mu_molecular=mu_molecular,
        rho_inf=rho_inf, vel_inf=vel_inf, p_inf=p_inf,
        aoa_deg=aoa_deg, aos_deg=aos_deg,
        # CFL：rk3 档为显式子步的固定 CFL（cfl_start），dual-time 档为内迭代伪时间 CFL 的 SER 律参数
        cfl_start=cfl_start, cfl_max=cfl_max, cfl_min=cfl_min, time_accurate=time_accurate,
        artificial_viscosity_enabled=artificial_viscosity_enabled,
        artificial_viscosity_alpha=artificial_viscosity_alpha,
        wall_treatment=wall_treatment,
    )

    # 4.5. 从 checkpoint 初始化（可选：以稳态结果为初场启动瞬态计算）
    if init_checkpoint:
        start_single_node_from_checkpoint(solver, init_checkpoint)

    # 参考面积（未给时沿来流方向自动估算，与 solve steady 同一个函数）
    reference_area = resolve_reference_area(solver, volume_data, reference_area)
    from autoflowcfd.core.probes import attach_probes

    attach_probes(solver, probes, output_dir)

    # 5. 执行瞬态求解（瞬态不用收敛判据 tol，跑满指定步数；--checkpoint-interval 定期落盘）
    try:
        result = solver.solve(
            max_iter=max_iter, dt=dt, tol=0.0,
            checkpoint_callback=periodic_checkpoint_callback(
                checkpoint_interval, output_dir, input_file, turbulence_model, backend,
                surface_mesh=surface_mesh),
            phase_max_iter=phase_max_iter,
            residual_drop_threshold=residual_drop_threshold)
        print(f"\n✅ Transient Simulation Finished: Steps={result.iterations}, Final Residual={result.final_residual:.6e}")

        # 6. 结果（.pkl 全量状态 + HDF5 checkpoint，后者供 solve resume 使用）与气动系数
        write_single_node_outputs(solver, output_dir, result.iterations, input_file, turbulence_model, backend,
                                  surface_mesh=surface_mesh)
        _report_aerodynamic_coefficients(solver.host_view(), reference_area)

    except Exception as e:
        print(f"\n❌ Transient Simulation Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise click.Abort()
