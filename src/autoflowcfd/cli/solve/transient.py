"""`solve transient` 命令 (DES/LES)

命令组定义见同目录 `commands.py`。
"""

from typing import Optional

import click
from loguru import logger

from autoflowcfd.core import FRSolver
from autoflowcfd.cli.solve.helpers import (
    compute_wall_distance_for_solver,
    load_mesh_for_solver,
    restore_state_from_checkpoint,
    save_results,
    write_checkpoint,
    load_physical_config_if_given,
    resolve_physical_constants,
    resolve_turbulence_model,
)
from autoflowcfd.cli.solve.aero_coefficients import _report_aerodynamic_coefficients
from autoflowcfd.cli.solve.commands import solve
from autoflowcfd.cli.solve.transient_distributed import _solve_transient_distributed


@solve.command(name='transient')
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--backend", "-b", type=click.Choice(["cpu", "gpu"]),
              default="cpu", help="计算后端")
@click.option("--order", "-p", type=click.IntRange(1, 3), default=2,
              help="FR 离散阶数")
@click.option("--time-method", "-t",
              type=click.Choice(["rk3", "imex", "dual-time"]),
              default="rk3",
              help="时间推进方法。**只有 dual-time 是时间精确的**："
                   "rk3/imex 下 --dt 参数被忽略，平均流与湍流场都按逐单元"
                   "局部 CFL 步长推进（稳态收敛加速手段，见 "
                   "core/fr_solver/step.py::step 的 dt 语义一节），各单元"
                   "推进的物理时间并不相同。DES/LES/WMLES 这类以时间解析"
                   "湍流结构为目的的模型必须用 dual-time，否则结果不能当作"
                   "非稳态数据解读——组合不当时本命令会显式警告。")
@click.option("--turbulence-model", "-m",
              type=click.Choice(["sst", "ddes", "iddes", "wmles", "les"]),
              default="ddes",
              help="湍流模型。iddes 按 Shur et al. (2008)/Gritskevich et al. (2012) "
                   "SST-IDDES 重新实现（取代此前的死代码版本），部分常数（f_e2 的 "
                   "c_t/c_l）本会话未能独立复核原始文献数值，见 core/turbulence/des.py "
                   "::IDDESModel 文档。注意：ddes/iddes/les 会在 VELOCITY_INLET 边界"
                   "自动启用 BD-02 合成湍流入口 (SEM)；wmles 不会（WMLES 依赖壁面模型本身 "
                   "正确预测近壁应力，不需要额外的入口湍流结构，见 "
                   "core/fr_solver/boundary.py 文档）")
@click.option("--max-iter", "-n", default=100, help="最大迭代次数")
@click.option('--phase-max-iter', type=int, default=None,
              help='Order Continuation（--order>=2 时触发）非最终阶段(P0/P1/...，不含目标'
                   '阶数)各自的最大迭代步数上限。默认(不传)时保留旧行为——总步数按阶段数'
                   '机械均分。传具体值后目标阶数改为吃掉这次求解剩余的全部步数，不再随'
                   '阶段数被稀释，见 core/utils/order_continuation.py 文档。仅 CPU 后端支持')
@click.option('--residual-drop-threshold', type=float, default=100.0,
              help='Order Continuation 单个非最终阶段判定"可以提前升阶"的残差下降倍数，'
                   '默认100(降2个数量级)。仅 CPU 后端支持')
@click.option('--aoa', 'aoa_deg', type=float, default=0.0,
              help='攻角 alpha（度，绕 y 轴、抬头为正，默认 0）。**2026-09-17 新增**：'
                   '此前来流方向在全代码库被硬编码成 +x，没有任何攻角选项——而攻角'
                   '扫掠是最常见的外流气动研究。开启后 Q_free（边界自由来流态）、'
                   '初场、SEM 入口方向、气动力的风轴系分解（Cd 沿来流、Cl 垂直于来流）、'
                   '参考面积的迎风投影五处统一按它构造。0 时与此前行为逐位相同。'
                   '力矩 Cm/Cy/Cr 仍报在体轴系，不随攻角旋转（气动数据标准呈现方式）。'
                   '约定与风轴系公式见 core/utils/flow_direction.py')
@click.option('--aos', 'aos_deg', type=float, default=0.0,
              help='侧滑角 beta（度，绕 z 轴，默认 0）。语义与 --aoa 同，见其说明')
@click.option("--dt", default=1e-5, help="时间步长 (秒)")
@click.option('--cfl-start', type=float, default=0.03,
              help='自适应 CFL 初始值（默认 0.03）。**只对 --time-method '
                   'rk3/imex 生效**：那两档下 step() 忽略 --dt、按逐单元'
                   '局部 CFL 步长推进，自适应控制器是激活的；dual-time 档'
                   '不构造这个控制器（内层伪时间有自己的逻辑）。'
                   '**2026-09-17 补齐**：此前 solve transient 一个 CFL 选项'
                   '都没有，rk3/imex 瞬态只能吃 FRSolver 构造默认值，配置'
                   '不出本项目在两张真实网格上实测稳定的 ~0.03。')
@click.option('--cfl-max', type=float, default=0.06,
              help='自适应 CFL 上限（默认 0.06，2026-09-17 从 0.5 下调）。'
                   '语义与 --cfl-start 同，只对 rk3/imex 生效。下调依据见 `solve steady --cfl-max` 的帮助：2026-09-17 按直接谱测量 + 两张真实网格的失效点重定，线性极限约 0.117、实测失效点 plate 0.30 / 平板边界层 0.10，默认值留 1.7 倍以上裕度。')
@click.option('--cfl-min', type=float, default=0.01,
              help='自适应 CFL 下限（默认 0.01，与 solve steady/resume 对齐）。'
                   '注意下限会把低于它的 --cfl-start 钳上去，做低 CFL 工况'
                   '时三个都要一起调（见 core/time_integration/'
                   'adaptive_cfl.py 模块文档第 11 条）。')
@click.option("--physical-time", default=None, help="总物理时间（秒）")
@click.option("--output", "-o", "output_dir", default="./transient_results", help="输出目录")
@click.option("--use-eikonal", is_flag=True, help='使用 Eikonal 方程求解壁面距离')
@click.option("--surface-mesh", "-s", type=click.Path(exists=True), default=None,
              help='原始面网格路径 - input_file 是 .nas 体网格时必填，用于反推边界分组；input_file 是 .pkl 时不需要')
@click.option("--skip-quality-check", is_flag=True, help='跳过求解前的网格质量门检查（不建议，仅用于临时诊断）')
@click.option('--reference-area', type=float, default=None, help='气动系数参考面积 (m^2)，提供时求解结束后打印 Cd/Cl')
@click.option('--dual-time-inner-iter', type=int, default=20,
              help='--time-method dual-time 时每个物理步的伪时间内迭代次数（此前恒为硬编码3，'
                   '真实测得默认保守CFL策略下通常不足以收敛到物理时间精度，见 TimeIntegrator 文档）')
@click.option('--threads', '-j', type=int, default=-1, help='CPU 后端 numba 并行 kernel 使用的线程数，默认 -1 = 4（本机真实网格实测扩展性甜点，不是核数）')
@click.option('--init-from', 'init_checkpoint', type=click.Path(exists=True), default=None,
              help='从稳态 checkpoint 文件初始化瞬态求解器（典型工作流：先稳态 SST 收敛，'
                   '再从该流场启动 DES/LES 瞬态计算，避免从均匀流场直接启动需要极长的瞬态发展时间）')
@click.option('--turbulence-intensity', type=float, default=0.01,
              help='来流湍流强度 Tu（默认 0.01=1%%）。也驱动 ddes/iddes/les 模式下 '
                   'BD-02 SEM 入口的目标雷诺应力（2026-08-28 起复用同一个值，'
                   '此前 SEM 用独立硬编码 5%%，见 core/fr_solver/boundary.py 文档）')
@click.option('--sem-num-eddies', type=int, default=200,
              help='ddes/iddes/les 模式下 BD-02 合成湍流入口 (SEM) 的涡核数量（默认 200）')
@click.option('--viscosity-ratio', type=float, default=5.0, help='来流粘性比 VR=nu_t/nu（默认 5.0）')
@click.option('--mu-molecular', type=float, default=1.8e-5, help='分子动力粘度 (Pa*s)，默认 1.8e-5（标准状态下空气），非标准工况请覆盖')
@click.option('--rho-inf', type=float, default=1.225, help='自由流密度 (kg/m^3)，默认 1.225（标准海平面空气）')
@click.option('--vel-inf', type=float, default=33.33, help='自由流速度大小 (m/s)，默认 33.33')
@click.option('--p-inf', type=float, default=101325.0, help='自由流静压 (Pa)，默认 101325.0（标准大气压）')
@click.option('--config', 'config_path', type=click.Path(exists=True), default=None,
              help='从 YAML 文件读取物理常量默认值（mu_molecular/rho_inf/vel_inf/p_inf/'
                   'turbulence_intensity/viscosity_ratio）；显式传入的同名 --xxx 选项优先于此文件')
@click.option('--n-ranks', type=int, default=1,
              help='MPI rank 总数（>1 时走分布式求解器——CPU MPI"传统模式"，或配合 '
                   '--multi-gpu/--fully-distributed 走对应的分布式构造入口）。'
                   '2026-09-02 补齐：此前本命令完全没有分布式支持，DUAL_TIME/DES/LES '
                   '瞬态仿真只能单机跑，与 solve steady 已有的分布式覆盖不一致')
@click.option('--multi-gpu', is_flag=True, help='启用多 GPU + MPI 分布式求解（每个 rank 使用一块 GPU）')
@click.option('--fully-distributed', is_flag=True,
              help='"完全分布式加载"（只有 root rank 加载完整网格），需要 --n-ranks>1，'
                   '与 --multi-gpu 互斥；已支持 --time-method dual-time（2026-09-02 起，'
                   'package 已接入 time_scheme 字段）')
@click.option('--gpu-device', type=int, default=None, help='--multi-gpu 时的 GPU 设备号')
@click.option('--checkpoint-interval', type=int, default=100,
              help='分布式路径中间 checkpoint 保存间隔（单机路径瞬态求解不做中间保存，'
                   '只在结束后写一次，与 solve steady 的分布式分支同一个约定）')
def transient(input_file: str, backend: str, order: int, time_method: str,
              turbulence_model: str, max_iter: int, phase_max_iter: Optional[int], residual_drop_threshold: float,
              dt: float, cfl_start: float, cfl_max: float, cfl_min: float,
        aoa_deg: float, aos_deg: float,
        physical_time: float,
              output_dir: str, use_eikonal: bool, surface_mesh: Optional[str],
              skip_quality_check: bool, reference_area: Optional[float],
              dual_time_inner_iter: int, threads: int, init_checkpoint: Optional[str],
              turbulence_intensity: float, viscosity_ratio: float, sem_num_eddies: int,
              mu_molecular: float, rho_inf: float, vel_inf: float, p_inf: float,
              config_path: Optional[str], n_ranks: int, multi_gpu: bool,
              fully_distributed: bool, gpu_device: Optional[int],
              checkpoint_interval: int) -> None:
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
        phase_max_iter: Order Continuation(--order>=2 时触发)非最终阶段各自的
            最大迭代步数上限，None(默认)时保留旧行为(按阶段数均分)；仅 CPU 后端支持
        residual_drop_threshold: Order Continuation 单阶段提前升阶所需的残差
            下降倍数，默认100
        dt: 时间步长
        physical_time: 总物理时间（秒）
        output_dir: 输出目录
        use_eikonal: 是否使用 Eikonal 方程
        surface_mesh: 原始面网格路径，input_file 是 .nas 体网格时必填
        skip_quality_check: 跳过求解前的网格质量门检查
        init_checkpoint: 从稳态 checkpoint 初始化（可选）
    """
    print(f"=== Starting Transient FR Simulation (DES/LES) ===")

    # 物理常量解析：显式 CLI 选项 > --config YAML > 上面 click 声明的内建默认值。
    # 不能硬编码——见 solve_physical_constants.py 文档。
    _phys_cfg = load_physical_config_if_given(config_path)
    _resolved = resolve_physical_constants(
        click.get_current_context(),
        {
            'turbulence_intensity': turbulence_intensity, 'viscosity_ratio': viscosity_ratio,
            'mu_molecular': mu_molecular, 'rho_inf': rho_inf, 'vel_inf': vel_inf, 'p_inf': p_inf,
            'order': order, 'dt': dt,
            'phase_max_iter': phase_max_iter, 'residual_drop_threshold': residual_drop_threshold,
        },
        _phys_cfg,
    )
    turbulence_intensity = _resolved['turbulence_intensity']
    viscosity_ratio = _resolved['viscosity_ratio']
    mu_molecular = _resolved['mu_molecular']
    rho_inf = _resolved['rho_inf']
    vel_inf = _resolved['vel_inf']
    p_inf = _resolved['p_inf']
    order = _resolved['order']
    dt = _resolved['dt']
    phase_max_iter = _resolved['phase_max_iter']
    residual_drop_threshold = _resolved['residual_drop_threshold']
    # turbulence_model：见 solve_steady_command.py 同一处的说明。
    turbulence_model = resolve_turbulence_model(click.get_current_context(), turbulence_model, _phys_cfg)
    # physical_time ← config.total_time：字段名不同（CLI 用 physical_time，
    # TransientConfig 用 total_time），resolve_physical_constants 的同名
    # getattr 不适用；physical_time 的 click 默认值就是 None，不需要
    # get_parameter_source 也能安全判断"用户是否显式传过"。
    if physical_time is None and _phys_cfg is not None and hasattr(_phys_cfg, 'total_time'):
        physical_time = _phys_cfg.total_time

    # Tu/VR/mu_molecular/rho_inf/vel_inf/p_inf 范围校验：与
    # solve_steady_command.py 入口同一规则（CLI 路径不经过
    # SolverConfig.__post_init__ 的校验，2026-08-25 代码审查）。
    if not (0.0 < turbulence_intensity <= 1.0):
        raise click.BadParameter("湍流强度 Tu 必须在 (0, 1] 区间", param_hint="--turbulence-intensity")
    if viscosity_ratio <= 0.0:
        raise click.BadParameter("粘性比 VR 必须 > 0", param_hint="--viscosity-ratio")
    if mu_molecular <= 0.0:
        raise click.BadParameter("分子动力粘度必须 > 0", param_hint="--mu-molecular")
    if rho_inf <= 0.0:
        raise click.BadParameter("自由流密度必须 > 0", param_hint="--rho-inf")
    if vel_inf <= 0.0:
        raise click.BadParameter("自由流速度必须 > 0", param_hint="--vel-inf")
    if p_inf <= 0.0:
        raise click.BadParameter("自由流静压必须 > 0", param_hint="--p-inf")
    # --phase-max-iter/--residual-drop-threshold（2026-09-02 续接）：
    # 全部四种后端（单机 CPU/单 GPU/多GPU/MPI 分布式）现在都真正接入了
    # Order Continuation，不再需要任何"某后端不支持"的拒绝，见
    # solve_steady_command.py 同一处修复文档。
    print(f"\nInput Grid : {input_file}")
    print(f"Backend    : {backend} | Order: P{order} | Method: {time_method}")
    print(f"Turbulence : {turbulence_model} | dt: {dt:.2e}")
    if init_checkpoint:
        print(f"Init From  : {init_checkpoint}")
    if physical_time:
        max_iter = int(float(physical_time) / dt)
        print(f"Physical Time: {physical_time}s | Iterations: {max_iter}\n")
    else:
        print(f"Iterations : {max_iter}\n")

    # 词汇->枚举唯一事实来源（见 core/time_integration/base.py）。
    from autoflowcfd.core.time_integration.base import scheme_from_name

    if n_ranks > 1 or multi_gpu:
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
            turbulence_model, max_iter, dt, use_eikonal, output_dir,
            reference_area, threads, turbulence_intensity, viscosity_ratio,
            mu_molecular, rho_inf, vel_inf, p_inf,
            n_ranks, multi_gpu, fully_distributed, gpu_device, backend,
            checkpoint_interval, phase_max_iter, residual_drop_threshold,
            init_checkpoint,
            cfl_start=cfl_start, cfl_max=cfl_max, cfl_min=cfl_min,
            aoa_deg=aoa_deg, aos_deg=aos_deg,
        )
        return

    # 1. 网格加载与处理（含求解前质量门检查）
    mesh, volume_data = load_mesh_for_solver(
        input_file, order, surface_mesh=surface_mesh, skip_quality_check=skip_quality_check,
    )

    # 2. 映射时间推进方法
    time_scheme = scheme_from_name(time_method)

    # 时间精度与湍流模型的组合校验（2026-09-15）：本命令的默认组合是
    # `--time-method rk3` + `--turbulence-model ddes`，而 rk3/imex 下
    # `step()` **忽略** --dt、按逐单元局部 CFL 步长推进（见
    # core/fr_solver/step.py::step 文档"dt 参数的语义按 time_scheme 分两种
    # 情况"一节）。各单元因此推进的物理时间并不相同，结果不是时间精确解。
    # DES/LES/WMLES 的全部意义就在于时间上解析湍流结构，用非时间精确的
    # 推进跑出来的场不能当作非稳态数据解读——而此前这个组合没有任何提示。
    #
    # 这里**不直接拒绝**：rk3 + DES 作为"先把流场大致吹起来"的快速冒烟
    # 是有用的（本项目既有的 DDES/LES CLI 端到端验证就是这么跑的），
    # 拒绝会破坏一条已验证可用的路径。但必须显式、醒目地说清它不是什么。
    _TIME_RESOLVED_MODELS = ("ddes", "iddes", "les", "wmles")
    if (time_method in ("rk3", "imex")
            and turbulence_model.lower() in _TIME_RESOLVED_MODELS):
        click.secho(
            "\n⚠️  时间精度警告：--turbulence-model "
            + turbulence_model
            + " 是以**时间解析**湍流结构为目的的模型，但 --time-method "
            + time_method + " **不是时间精确的**。",
            fg="yellow", bold=True)
        click.secho(
            "   rk3/imex 下 --dt 被忽略，平均流与湍流场按逐单元局部 CFL "
            "步长推进（稳态收敛加速手段），各单元推进的物理时间并不相同。",
            fg="yellow")
        click.secho(
            "   本次结果可用于观察流场大致形态，但**不能当作非稳态/频谱"
            "数据解读**（涡脱落频率、TKE 谱、相位等一概无效）。",
            fg="yellow")
        click.secho(
            "   要做真正的非稳态仿真请用：--time-method dual-time"
            "（配合 --dt 与 --dual-time-inner-iter）。\n",
            fg="yellow")

    # 3. 初始化求解器
    solver = FRSolver(
        mesh=mesh,
        backend=backend,
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
        # CFL 三元组（2026-09-17 补齐）：rk3/imex 档走逐单元局部 CFL
        # 推进、自适应控制器是激活的，此前这里一个都不传。
        cfl_start=cfl_start, cfl_max=cfl_max, cfl_min=cfl_min,
    )

    # 4. 计算壁面距离场（DES/LES/WMLES 必须）
    compute_wall_distance_for_solver(solver, volume_data, use_eikonal=use_eikonal)

    # 4.5. 从 checkpoint 初始化（可选：以稳态结果为初场启动瞬态计算）
    if init_checkpoint:
        from types import SimpleNamespace
        from autoflowcfd.core.utils.checkpoint import CheckpointManager

        print(f"\n🔄 从 checkpoint 加载稳态结果作为瞬态初场...")
        solution, history, ckpt_iter, ckpt_meta = CheckpointManager(
            config=SimpleNamespace(), output_dir="."
        ).load(init_checkpoint)

        restore_state_from_checkpoint(init_checkpoint, solver, ckpt_meta)
        print(f"   源 checkpoint 迭代数: {ckpt_iter}\n")

    # 5. 执行瞬态求解
    try:
        # 瞬态求解通常不需要 tol，而是跑满指定的时间步
        result = solver.solve(max_iter=max_iter, dt=dt, tol=0.0,
                               phase_max_iter=phase_max_iter,
                               residual_drop_threshold=residual_drop_threshold)
        print(f"\n✅ Transient Simulation Finished: Steps={result.iterations}, Final Residual={result.final_residual:.6e}")

        # 6. 保存结果（.pkl 全量状态 + HDF5 checkpoint，后者供 solve resume 使用）
        save_results(solver, output_dir)
        # solver.current_order 而非 order：瞬态求解本身不做 Order
        # Continuation 爬升，但 input_file 若是从 steady 阶段的 checkpoint
        # resume 而来，order 这个闭包变量可能仍是 steady 侧的目标阶数，
        # 与 resume 时实际重建出的 solver.current_order 不一定相等——见
        # solve_steady_command.py 里同名参数的说明，避免同一类 checkpoint
        # 形状不匹配 bug。
        write_checkpoint(
            solver, output_dir, result.iterations, input_file, solver.current_order,
            turbulence_model, backend,
            history={"iterations": [result.iterations]}, surface_mesh=surface_mesh,
            target_order=solver.order,
        )

        # 7. 气动系数（提供 --reference-area 时）
        _report_aerodynamic_coefficients(solver, reference_area)

    except Exception as e:
        print(f"\n❌ Transient Simulation Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise click.Abort()
