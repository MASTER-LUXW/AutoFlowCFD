"""`solve steady` 命令 —— 从 solve_steady_commands.py 拆出，控制单文件行数。

见 solve_steady_commands.py 文档说明整体拆分结构。
"""

import click
from loguru import logger

from autoflowcfd.core import FRSolver
from autoflowcfd.core.time_integration.base import TimeIntegrationScheme
from autoflowcfd.cli.solve_helpers import (
    compute_wall_distance_for_solver,
    load_mesh_for_solver,
    save_results,
    write_checkpoint,
    load_physical_config_if_given,
    resolve_physical_constants,
    resolve_turbulence_model,
)
from autoflowcfd.cli.solve_aero_coefficients import (
    _compute_reference_area_auto,
    _report_aerodynamic_coefficients,
)
from autoflowcfd.cli.solve_commands import solve


@solve.command(name='steady')
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--backend', type=click.Choice(['cpu', 'gpu']), default='cpu', help='计算后端 (CPU/GPU)')
@click.option('--order', type=int, default=2, help='FR 多项式阶数 (P1/P2/P3)')
@click.option('--flux-type', type=click.Choice(['radau', 'gauss']), default='radau',
              help="FR 修正函数族（#14）：'radau'（默认，此前唯一使用过的方案，"
                   "Huynh 记法 g_DG）；'gauss' 是与 Spectral Difference 等价的新方案"
                   "（见 fr/matrix_operators.py 文档）。目前仅单机 CPU 路径支持，"
                   "GPU/多 GPU/MPI 分布式路径传非默认值会报错而不是静默忽略")
@click.option('--turbulence-model', type=click.Choice(['none', 'sst', 'ddes', 'iddes', 'wmles', 'les']), default='sst',
              help='湍流模型。真实bug修复（2026-09-02，排查多GPU分布式DDES/IDDES时发现）：此前这里的'
                   'Choice列表缺 iddes/les 两项——底层单机/CPU MPI/多GPU分布式路径均已支持这两个模型'
                   '（solve transient命令的Choice列表本来就包含它们），steady命令这里一直没有同步，'
                   '导致 --turbulence-model iddes/les 在steady命令下无法使用（会被click直接拒绝），'
                   '与transient命令行为不一致。另外注意：ddes/iddes/les 会在 VELOCITY_INLET 边界'
                   '自动启用 BD-02 合成湍流入口 (SEM)；wmles 不会（WMLES 依赖壁面模型本身正确'
                   '预测近壁应力，不需要额外的入口湍流结构，见 core/fr_solver/boundary.py 文档）')
@click.option('--max-iter', type=int, default=1000, help='最大迭代次数')
@click.option('--cfl-start', type=float, default=0.1,
              help='自适应 CFL 初始值（稳态伪时间迭代，默认 0.1）。残差不下降时 CFL '
                   '会一直停在这个值——复杂网格上如果起步就发散可调低。下限是独立的 '
                   '--cfl-min（2026-09-15 起；此前本文案把两者混为一谈，而下限默认 '
                   '0.05 会把低于它的 --cfl-start 钳上去）。')
@click.option('--cfl-max', type=float, default=0.5,
              help='自适应 CFL 上限（稳态，默认 0.5，2026-09-07 从 0.3 上调）。'
                   'SSP-RK3 线性稳定极限 ~1.0，残差稳定下降的算例可以试 0.8；'
                   'AUSM+up 低马赫预处理激活的算例真实可用上限更低，发散时回调到 0.3')
@click.option('--cfl-min', type=float, default=0.05,
              help='自适应 CFL 下限（稳态，默认 0.05）。**注意这个默认值高于'
                   '真 P1（AFCFD_FILTER_MODE=off，零阶数损失）在 79 万单元 '
                   'cube_demo 上实测稳定的 0.03**——做低 CFL 工况时必须显式'
                   '调低，否则 --cfl-start 会被这个下限钳上去（见 '
                   'core/time_integration/adaptive_cfl.py 模块文档第 11 条）。')
@click.option('--phase-max-iter', type=int, default=None,
              help='Order Continuation（--order>=2 时触发）非最终阶段(P0/P1/...，不含目标'
                   '阶数)各自的最大迭代步数上限。默认(不传)时保留旧行为——总步数按阶段数'
                   '机械均分(max_iter // 阶段数)，目标阶数与非最终阶段拿到同一份额。传具体值'
                   '后非最终阶段各自最多跑这么多步(提前满足--residual-drop-threshold仍可'
                   '提前升阶)，目标阶数改为吃掉这次求解剩余的全部步数，不再随阶段数被稀释。'
                   '目前仅单机 CPU 路径支持，GPU/多GPU/MPI 分布式路径传非默认值会报错')
@click.option('--residual-drop-threshold', type=float, default=100.0,
              help='Order Continuation 单个非最终阶段判定"可以提前升阶"的残差下降倍数，'
                   '默认100(降2个数量级)，原来硬编码，现在可配置。目前仅单机 CPU 路径支持，'
                   'GPU/多GPU/MPI 分布式路径传非默认值会报错')
@click.option('--output', '-o', 'output_dir', type=click.Path(), default='./results', help='结果输出目录')
@click.option('--checkpoint-interval', type=int, default=100, help='检查点保存间隔')
@click.option('--use-eikonal', is_flag=True, help='使用 Eikonal 方程求解壁面距离（更精确但较慢）')
@click.option('--surface-mesh', '-s', type=click.Path(exists=True), default=None,
              help='原始面网格路径 - input_file 是 .nas 体网格时必填，用于反推边界分组；input_file 是 .pkl 时不需要')
@click.option('--skip-quality-check', is_flag=True, help='跳过求解前的网格质量门检查（不建议，仅用于临时诊断）')
@click.option('--reference-area', type=float, default=None, help='气动系数参考面积 (m^2)，提供时求解结束后打印 Cd/Cl')
@click.option('--threads', '-j', type=int, default=-1, help='CPU 后端 numba 并行 kernel 使用的线程数，默认 -1 = 4（本机真实网格实测扩展性甜点，不是核数）')
@click.option('--n-ranks', '--np', type=int, default=1, help='MPI 并行 rank 数（域分解并行，需配合 mpirun 使用。默认 1 = 单机模式）')
@click.option('--fully-distributed', is_flag=True,
              help='真正的完全分布式网格加载（2026-09-02 新增，同日/次日续接补齐 SST/DDES/'
                   'IDDES/WMLES/LES 全部湍流模型 + Order Continuation + checkpoint resume + '
                   '多 GPU）：只有 root rank 加载完整网格，其余 rank 只接收 root 预先切好的'
                   '紧凑数据，不需要各自持有完整网格（--n-ranks>1 默认走的"传统模式"是每个'
                   ' rank 独立加载完整网格，内存上不是最优）。可单独用（CPU MPI），也可配合'
                   '--multi-gpu 使用（多 GPU 完全分布式加载，见 '
                   'gpu_distributed_fully_distributed.py 模块文档）。需要 --n-ranks>1 才生效。'
                   '此前这段帮助文本声称"只支持 turbulence_model none 且不支持 Order '
                   'Continuation/resume"是过时信息（写于该功能刚实现、后续批次未同步更新），'
                   '已更正。')
@click.option('--gpu-device', type=int, default=0, help='GPU 设备 ID（默认 0，多 GPU 时每个 rank 自动分配）')
@click.option('--multi-gpu', is_flag=True, help='启用多 GPU + MPI 分布式求解（每个 rank 使用一块 GPU）')
@click.option('--turbulence-intensity', type=float, default=0.01,
              help='来流湍流强度 Tu（默认 0.01=1%%），外部气动 ≤1%%，城市道路 3-5%%。'
                   '也驱动 --turbulence-model ddes 时 BD-02 SEM 入口的目标雷诺应力')
@click.option('--viscosity-ratio', type=float, default=5.0, help='来流粘性比 VR=nu_t/nu（默认 5.0），外部气动推荐 2-10')
@click.option('--sem-num-eddies', type=int, default=200,
              help='--turbulence-model ddes 时 BD-02 合成湍流入口 (SEM) 的涡核数量（默认 200）')
@click.option('--mu-molecular', type=float, default=1.8e-5, help='分子动力粘度 (Pa*s)，默认 1.8e-5（标准状态下空气），非标准工况请覆盖')
@click.option('--rho-inf', type=float, default=1.225, help='自由流密度 (kg/m^3)，默认 1.225（标准海平面空气）')
@click.option('--vel-inf', type=float, default=33.33, help='自由流速度大小 (m/s)，默认 33.33')
@click.option('--p-inf', type=float, default=101325.0, help='自由流静压 (Pa)，默认 101325.0（标准大气压）')
@click.option('--config', 'config_path', type=click.Path(exists=True), default=None,
              help='从 YAML 文件读取物理常量默认值（mu_molecular/rho_inf/vel_inf/p_inf/'
                   'turbulence_intensity/viscosity_ratio）；显式传入的同名 --xxx 选项优先于此文件')
@click.option('--artificial-viscosity', 'artificial_viscosity_enabled', is_flag=True,
              help='启用 Persson-Peraire 模态传感器 + 局部人工粘性（2026-08-29 新增，默认关闭，'
                   '见 core/fr_operators/artificial_viscosity.py 与 ProjectFiles/V2.0/'
                   '7_重大问题修复-求解稳定性.md）。用解本身的模态谱衰减速率（而非残差量级）'
                   '判断单元是否欠分辨率，只对触发传感器的单元叠加局部人工粘性')
@click.option('--av-alpha', 'artificial_viscosity_alpha', type=float, default=1.0,
              help='人工粘性强度标定常数（无量纲，默认1.0），只在 --artificial-viscosity 时有意义')
@click.option('--entropy-stable-volume', 'entropy_stable_volume_enabled', is_flag=True,
              help='体积项过积分分支启用 Chandrashekar (2013) 熵守恒两点通量替代逐点通量代入'
                   '（2026-08-30 新增，默认关闭，见 core/fr_operators/flux_kernels.py::'
                   'entropy_stable_volume_divergence_batch 与 ProjectFiles/V2.0/'
                   '8_算法重构-Entropy-Stable_Split-Form通量重构-Part1/2.md）。真实测试确认在'
                   '已启用过积分的基础上再改善约2~4倍，代价是体积项计算量从O(n_fine)升到'
                   'O(n_fine^2)，仅 CPU 后端实现')
def solve_steady(input_file, backend, order, flux_type, turbulence_model, max_iter, cfl_start, cfl_max, cfl_min, phase_max_iter, residual_drop_threshold, output_dir, checkpoint_interval, use_eikonal, surface_mesh, skip_quality_check, reference_area, threads, n_ranks, fully_distributed, gpu_device, multi_gpu, turbulence_intensity, viscosity_ratio, sem_num_eddies, mu_molecular, rho_inf, vel_inf, p_inf, config_path, artificial_viscosity_enabled, artificial_viscosity_alpha, entropy_stable_volume_enabled):
    """执行稳态 FR 求解。

    支持高阶精度 (P1-P4) 和多种湍流模型 (SST, DDES, WMLES)。
    输入文件必须是体网格 - .pkl（`grid generate-volume`/`grid
    import-volume` 的输出）或 .nas 体网格（需要配合 --surface-mesh 反推边界
    分组）。求解前会强制检查网格质量门，除非传了 --skip-quality-check。
    """
    print(f"=== Starting Steady FR Simulation ===")

    # 物理常量解析：显式 CLI 选项 > --config YAML > 上面 click 声明的内建默认值。
    # 不能硬编码——mu_molecular/rho_inf/vel_inf/p_inf 等基础物理量必须能被
    # 用户为非标准工况（不同流体/温度/高度）覆盖，见 solve_physical_constants.py 文档。
    _phys_cfg = load_physical_config_if_given(config_path)
    _resolved = resolve_physical_constants(
        click.get_current_context(),
        {
            'turbulence_intensity': turbulence_intensity, 'viscosity_ratio': viscosity_ratio,
            'mu_molecular': mu_molecular, 'rho_inf': rho_inf, 'vel_inf': vel_inf, 'p_inf': p_inf,
            'order': order, 'max_iter': max_iter,
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
    max_iter = _resolved['max_iter']
    phase_max_iter = _resolved['phase_max_iter']
    residual_drop_threshold = _resolved['residual_drop_threshold']
    # turbulence_model：CLI 字符串词汇与 SteadyConfig.turbulence 的枚举
    # 命名不完全一致，需要专门的映射，不能靠 resolve_physical_constants
    # 的同名 getattr（见 resolve_turbulence_model 文档）。
    turbulence_model = resolve_turbulence_model(click.get_current_context(), turbulence_model, _phys_cfg)

    # Tu/VR/mu_molecular/rho_inf/vel_inf/p_inf 范围校验：CLI 路径不构造
    # SolverConfig，其 __post_init__ 的校验到不了这里，必须在入口拦截
    # （2026-08-25 代码审查；mu_molecular/rho_inf/vel_inf/p_inf 是本轮新增）。
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
    # #14：GPU/多GPU/MPI 分布式路径的 ops 构造未接入 flux_type（那些路径
    # 直接调用 generate_fr_operators(order) 不带 flux_point_type，见下方
    # 各分支），非默认值在这些路径上会被静默忽略——宁可显式报错，不静默
    # 退回默认方案（与本项目其余地方"不允许静默降级"的一贯原则一致）。
    if flux_type != 'radau' and (backend == 'gpu' or n_ranks > 1):
        raise click.BadParameter(
            "--flux-type gauss 目前只有单机 CPU 路径支持（GPU/多GPU/MPI 分布式"
            "路径的算子构造尚未接入这个选择）。请去掉 --backend gpu/--multi-gpu/"
            "--n-ranks，或使用默认的 --flux-type radau。",
            param_hint="--flux-type",
        )
    # --phase-max-iter/--residual-drop-threshold（2026-09-02 续接）：
    # 全部四种后端（单机 CPU/单 GPU/多GPU/MPI 分布式，含"传统模式"与
    # "完全分布式加载"）现在都真正接入了 Order Continuation（`solve()`
    # 在 `self.order>=2` 时自动分派到 `run_distributed_order_
    # continuation`，见 core/mpi/distributed_order_continuation.py/
    # core/gpu/solver/gpu_solver_order_continuation.py 模块文档），
    # 不再需要任何"某后端不支持"的拒绝。
    print(f"\nInput Grid : {input_file}")
    print(f"Backend    : {backend} | Order: P{order} | Method: rk3")
    print(f"Turbulence : {turbulence_model} | Max Iter: {max_iter}")
    if n_ranks > 1:
        print(f"MPI Ranks  : {n_ranks} (domain decomposition)")
    if use_eikonal:
        print(f"Wall Dist : Eikonal (graph-Dijkstra approx)\n")
    else:
        print(f"Wall Dist : KD-Tree (Geometric)\n")

    # 1. 网格加载与处理
    if backend == 'gpu' and multi_gpu and n_ranks > 1:
        # 多 GPU + MPI 分布式路径
        from autoflowcfd.core.gpu import gpu_available
        if not gpu_available:
            print("\n❌ CuPy not available. Install with: pip install cupy-cuda12x")
            raise click.Abort()

        from autoflowcfd.core.mpi import mpi_available, is_root
        if not mpi_available:
            print("\n❌ MPI not available. Install mpi4py and run with mpirun.")
            raise click.Abort()

        from autoflowcfd.core.gpu.distributed.gpu_distributed import MultiGPUDistributedSolver
        from autoflowcfd.fr.operators import generate_fr_operators

        if fully_distributed:
            # 多 GPU"完全分布式加载"（#1，2026-09-02 实现——此前只有
            # "传统模式"，见 MultiGPUDistributedSolver.from_fully_
            # distributed_package/gpu_distributed_fully_distributed.py
            # 模块文档）：只有 root rank 加载完整网格，root 预先按每个
            # rank 的 compact 索引空间切好紧凑包再分发，与 CPU
            # `--fully-distributed`（不加 --multi-gpu）同一套
            # `distributed_mesh_load_v2`/`build_fully_distributed_rank_
            # package`，只是构造出的是 GPU 常驻状态的求解器。
            import math
            from autoflowcfd.core.mpi.distributed_mesh_loader import distributed_mesh_load_v2
            from autoflowcfd.core.fr_solver.solver import _MACH_REF_FLOOR

            freestream = {"rho_inf": rho_inf, "vel_inf": vel_inf, "p_inf": p_inf}
            mach_ref = max(
                vel_inf / math.sqrt(max(1.4 * p_inf / max(rho_inf, 1e-10), 1e-10)),
                _MACH_REF_FLOOR,
            )
            package, root_context = distributed_mesh_load_v2(
                input_file, order, surface_mesh, n_ranks,
                freestream=freestream, mu_molecular=mu_molecular, mach_ref=mach_ref,
                enable_viscous=True, skip_quality_check=skip_quality_check,
                turb_model_name=turbulence_model.upper(),
                turbulence_intensity=turbulence_intensity, viscosity_ratio=viscosity_ratio,
                cfl_start=cfl_start, cfl_max=cfl_max, cfl_min=cfl_min,
            )
            solver = MultiGPUDistributedSolver.from_fully_distributed_package(
                package, n_ranks=n_ranks, device_id=gpu_device, root_context=root_context,
            )
        else:
            mesh, volume_data = load_mesh_for_solver(
                input_file, order, surface_mesh=surface_mesh, skip_quality_check=skip_quality_check
            )
            ops = generate_fr_operators(order)

            solver = MultiGPUDistributedSolver(
                mesh=mesh, ops=ops, n_ranks=n_ranks,
                device_id=gpu_device,
                mu_molecular=mu_molecular,
                rho_inf=rho_inf, vel_inf=vel_inf, p_inf=p_inf,
                # 真实 bug 修复（V2.0 专家组盲审发现）：此前从不传 turb_model，
                # --turbulence-model 无论填什么都被静默丢弃、恒定跑层流，
                # 终端打印的 Turbulence 行却仍显示用户输入的模型名。
                turb_model=turbulence_model.upper(),
                # SST 分布式湍流真正接入后（2026-09-02）才需要这两个值算
                # k_inf/omega_inf——此前 turb_model 恒被拒绝，这两个 CLI
                # 选项从未真正传到这里过，与单机 GPU 路径（上面
                # GPUFRSolver 构造处）保持一致。
                turbulence_intensity=turbulence_intensity,
                viscosity_ratio=viscosity_ratio,
                # 真实缺口修复（2026-09-15）：`--cfl-start/--cfl-max/--cfl-min`
                # 此前在**全部分布式路径**上被静默丢弃（这四处构造点都不传），
                # 与本文件上方注释记录过的 turb_model/turbulence_intensity
                # 同类。多 GPU 传统模式有自适应控制器（见 gpu_distributed.py
                # 里 _cfl_controller 构造处），必须一并透传。
                cfl_start=cfl_start, cfl_max=cfl_max, cfl_min=cfl_min,
            )

        # 中间 checkpoint 保存回调（2026-09-02 补齐——此前 output_interval
        # 只控制进度打印，跑到一半崩溃/被杀会丢失全部进度，与单机路径
        # `_checkpoint_cb` 同一个设计，见 solver.solve/save_checkpoint_
        # distributed 文档"完成度"一节说明）。
        def _multi_gpu_checkpoint_cb(solver_ref, iteration):
            if iteration % checkpoint_interval != 0:
                return
            try:
                # order/target_order 分离（2026-09-02，Order Continuation
                # 接入多GPU分布式路径后补齐——与 CPU 分布式
                # _distributed_checkpoint_cb 同一处修复同一个理由）：
                # 爬坡阶段中途的 checkpoint 必须记录
                # solver_ref.current_order（U_sps 实际形状），不是固定
                # 的目标 order。
                saved_path = solver_ref.save_checkpoint_distributed(
                    output_dir, iteration, input_file,
                    solver_ref.current_order, turbulence_model, backend="gpu",
                    target_order=solver_ref.order,
                )
                if saved_path and is_root():
                    print(f"   [Checkpoint] iter {iteration} saved: {saved_path}")
            except Exception as e:
                if is_root():
                    print(f"   [Checkpoint] Warning: save failed at iter {iteration}: {e}")

        try:
            result = solver.solve(
                max_iter=max_iter, dt=1e-3, tol=1e-6,
                checkpoint_callback=_multi_gpu_checkpoint_cb,
                phase_max_iter=phase_max_iter, residual_drop_threshold=residual_drop_threshold,
            )
            print(f"\n✅ Multi-GPU Simulation Finished")
            # #4（2026-08-28）：此前这里从不保存结果——分布式 checkpoint
            # save/load 依赖的 self.U_gpu 本地尺寸缺陷（#1）修复之前，
            # 保存也没有意义，见 MultiGPUDistributedSolver.
            # save_checkpoint_distributed 文档。
            saved_path = solver.save_checkpoint_distributed(
                output_dir, max_iter, input_file,
                solver.current_order, turbulence_model, backend="gpu",
                target_order=solver.order,
            )
            if saved_path and is_root():
                print(f"   Checkpoint saved: {saved_path}")
            solver.cleanup()
        except Exception as e:
            print(f"\n❌ Multi-GPU Simulation Failed: {e}")
            raise click.Abort()

    elif backend == 'gpu' and not multi_gpu:
        # 单 GPU 路径
        from autoflowcfd.core.gpu import gpu_available
        if not gpu_available:
            print("\n❌ CuPy not available. Install with: pip install cupy-cuda12x")
            raise click.Abort()

        from autoflowcfd.core.gpu.solver.gpu_solver import GPUFRSolver
        from autoflowcfd.fr.operators import generate_fr_operators

        mesh, volume_data = load_mesh_for_solver(
            input_file, order, surface_mesh=surface_mesh, skip_quality_check=skip_quality_check
        )
        ops = generate_fr_operators(order)

        solver = GPUFRSolver(
            mesh=mesh, ops=ops, order=order,
            device_id=gpu_device,
            turbulence_intensity=turbulence_intensity,
            viscosity_ratio=viscosity_ratio,
            mu_molecular=mu_molecular,
            rho_inf=rho_inf, vel_inf=vel_inf, p_inf=p_inf,
            # 真实 bug 修复（V2.0 专家组盲审发现）：此前从不传 turb_model，
            # --turbulence-model 无论填什么都被静默丢弃、恒定跑层流，
            # 终端打印的 Turbulence 行却仍显示用户输入的模型名。
            turb_model=turbulence_model.upper(),
            # 真实缺口修复（2026-09-14）：`--cfl-start/--cfl-max` 此前只
            # 到得了 CPU 的 FRSolver，GPU 路径连自适应 CFL 控制器都没有、
            # 恒用固定 CFL。GPUFRSolver 现在有了控制器（见
            # core/gpu/solver/gpu_solver.py 里 _cfl_controller 的注释），
            # 这两个 CLI 选项必须一并透传，否则又是一个"选项在 GPU 下被
            # 静默丢弃"的陷阱（与上面 turb_model 那处同类）。
            cfl_start=cfl_start, cfl_max=cfl_max, cfl_min=cfl_min,
        )

        try:
            result = solver.solve(
                max_iter=max_iter, dt=1e-3, tol=1e-6,
                phase_max_iter=phase_max_iter, residual_drop_threshold=residual_drop_threshold,
            )
            print(f"\n✅ GPU Simulation Finished: Iterations={result['iterations']}")

            # 保存结果
            state_cpu = solver.get_state_cpu()
            import pickle, os
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, 'final_state.pkl'), 'wb') as f:
                pickle.dump(state_cpu, f)
            solver.cleanup()

        except Exception as e:
            print(f"\n❌ GPU Simulation Failed: {e}")
            raise click.Abort()

    elif n_ranks > 1:
        # 分布式求解器路径。
        #
        # #2（V2.0 专家组盲审第4轮，2026-08-28）：此前这里唯一的路径是
        # "完全分布式网格加载"（只有 root 加载完整网格，通过
        # distributed_mesh_load 分发局部数据）——但 distributed_mesh_load
        # 产出的 local_mesh 缺少 face_connectivity/face_flux_points（见
        # distributed_mesh_loader.py::build_local_mesh_from_data 文档），
        # DistributedFRSolver.__init__ 内部 build_distributed_flat_face
        # 需要真实的全局面几何才能构建分布式面几何，这条路径构造期必然
        # 失败——这是一个更深的架构缺口（真正做到"只有 root 持有完整
        # 网格"需要 root 逐 rank 预构建+分发压缩几何数据，工作量与 GPU
        # 侧 #1 修复相当，未在本次修复范围内）。
        #
        # 改为"传统模式"（与 --multi-gpu 已经在用的模式一致，见
        # solve_steady_command.py 的 --multi-gpu 分支）：每个 rank 独立
        # 加载完整网格，进程内分区——不是内存最优，但 mesh.face_
        # connectivity 是真实、完整的，build_distributed_flat_face 能
        # 正确工作，DistributedMeshAdapter/distributed_compute_*_residual
        # 的 local+halo 压缩索引空间重排（#2 修复）才有意义。
        from autoflowcfd.core.mpi import mpi_available, is_root
        if not mpi_available:
            print("\n❌ MPI not available. Please install mpi4py and run with mpirun.")
            print("   pip install mpi4py")
            print("   mpirun -np {n_ranks} autoflowcfd solve steady ...")
            raise click.Abort()

        from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver
        from autoflowcfd.fr.operators import generate_fr_operators

        if fully_distributed:
            # 真正的完全分布式网格加载（2026-09-02 实现，同日续接补齐
            # SST/DDES/IDDES/WMLES/LES——见 distributed_mesh_loader.py::
            # distributed_mesh_load_v2/DistributedFRSolver.from_fully_
            # distributed_package 文档"范围边界"一节）。此前这里的
            # 硬编码 `!= 'NONE'` 拒绝早于 SST 支持接入、且从未跟随后续
            # 批次同步更新，是过时的信息源——真实 bug：既拒绝了后端已经
            # 支持的全部湍流模型，也从未把 `turbulence_model` 传给
            # `distributed_mesh_load_v2`（该函数签名里 `turb_model_name`
            # 参数一直被忽略，恒用默认值 'NONE'）。
            import math
            from autoflowcfd.core.mpi.distributed_mesh_loader import distributed_mesh_load_v2
            from autoflowcfd.core.fr_solver.solver import _MACH_REF_FLOOR

            freestream = {"rho_inf": rho_inf, "vel_inf": vel_inf, "p_inf": p_inf}
            # 与 FRSolver.__init__ 同一个公式（见该文件 mach_ref 计算处），
            # 不是本处新发明的近似——AUSM+up Weiss-Smith 预处理要求分区
            # 两侧用同一个真实值，公式本身也必须与单机路径逐字一致。
            mach_ref = max(
                vel_inf / math.sqrt(max(1.4 * p_inf / max(rho_inf, 1e-10), 1e-10)),
                _MACH_REF_FLOOR,
            )
            # root_context（2026-09-02，Order Continuation 支持）：只有
            # root rank 非 None，持有完整全局网格供后续阶数切换重新
            # 分发用，见 distributed_mesh_load_v2/redistribute_fully_
            # distributed_for_new_order 文档。
            package, root_context = distributed_mesh_load_v2(
                input_file, order, surface_mesh, n_ranks,
                freestream=freestream, mu_molecular=mu_molecular,
                mach_ref=mach_ref,
                enable_viscous=True, skip_quality_check=skip_quality_check,
                turb_model_name=turbulence_model.upper(),
                turbulence_intensity=turbulence_intensity, viscosity_ratio=viscosity_ratio,
                cfl_start=cfl_start, cfl_max=cfl_max, cfl_min=cfl_min,
            )
            solver = DistributedFRSolver.from_fully_distributed_package(
                package, n_ranks=n_ranks, root_context=root_context,
            )

            print(f"[Distributed] Initialized with {n_ranks} ranks (fully-distributed mode: "
                  f"only root rank loaded the full mesh)")
            print(f"[Distributed] {solver.partition.n_local_cells} local cells, "
                  f"{solver.partition.n_halo} halo cells")
        else:
            # 传统模式：每个 rank 独立加载完整网格（与单机/--multi-gpu 路径
            # 同一个 load_mesh_for_solver 入口）。
            mesh, volume_data = load_mesh_for_solver(
                input_file, order, surface_mesh=surface_mesh, skip_quality_check=skip_quality_check
            )
            ops = generate_fr_operators(order)

            # 创建分布式求解器（传入 face_connectivity 触发"传统模式"分区，
            # 见 DistributedFRSolver.__init__ 的"兼容旧接口"分支——本次修复
            # 之前这条分支就存在，只是 CLI 从未真正用过它，见上方说明）。
            solver = DistributedFRSolver(
                mesh=mesh,
                ops=ops,
                face_connectivity=mesh.face_connectivity,
                n_ranks=n_ranks,
                backend=backend,
                order=order,
                turb_model_name=turbulence_model,
                time_scheme=TimeIntegrationScheme.SSP_RK3,
                n_threads=threads,
                turbulence_intensity=turbulence_intensity,
                viscosity_ratio=viscosity_ratio,
                mu_molecular=mu_molecular,
                rho_inf=rho_inf, vel_inf=vel_inf, p_inf=p_inf,
                # 见多 GPU 传统模式同一处注释：CFL 边界参数此前在分布式
                # 路径上被静默丢弃。DistributedFRSolver 从 solver_kwargs
                # 读这三个键（见其 _cfl_controller 构造处）。
                cfl_start=cfl_start, cfl_max=cfl_max, cfl_min=cfl_min,
            )

            # 初始化状态
            print(f"[Distributed] Initialized with {n_ranks} ranks")
            print(f"[Distributed] {solver.partition.n_local_cells} local cells, "
                  f"{solver.partition.n_halo} halo cells")
            print(f"[Distributed] Traditional mode: every rank loaded the full mesh "
                  f"(not memory-optimal, see #2 fix notes; use --fully-distributed "
                  f"for the memory-optimal path)")

        # 中间 checkpoint 保存回调（2026-09-02 补齐——此前只在 solve()
        # 返回之后保存一次最终 checkpoint，跑到一半崩溃/被杀会丢失全部
        # 进度，且没有任何"从分布式 checkpoint 继续跑"的机制，与单机
        # 路径 `_checkpoint_cb` 同一个设计，见 `DistributedFRSolver.
        # solve`/`solve_commands.py::resume` 的 `--n-ranks`/`--multi-gpu`
        # 分支说明）。
        from autoflowcfd.core.mpi.distributed_checkpoint import (
            distributed_save_results,
            distributed_save_checkpoint,
        )

        def _distributed_checkpoint_cb(solver_ref, iteration):
            if iteration % checkpoint_interval != 0:
                return
            try:
                # order 传 solver_ref.current_order（不是固定的目标
                # order）：Order Continuation 接入分布式路径后
                # （2026-09-02），爬坡阶段中途保存的 checkpoint 的
                # `U_sps` 形状对应的是当时的 current_order，不是最终
                # 目标阶数，见 distributed_save_checkpoint 同名参数
                # 文档。target_order 记录真正的目标，供 resume 时继续
                # 爬坡。
                saved_path = distributed_save_checkpoint(
                    solver_ref, output_dir, iteration,
                    input_file, solver_ref.current_order, turbulence_model, backend,
                    target_order=solver_ref.order,
                )
                if saved_path and is_root():
                    print(f"   [Checkpoint] iter {iteration} saved: {saved_path}")
            except Exception as e:
                if is_root():
                    print(f"   [Checkpoint] Warning: save failed at iter {iteration}: {e}")

        # 执行分布式求解
        try:
            solver.solve(
                n_steps=max_iter, dt=1e-3, output_interval=checkpoint_interval,
                checkpoint_callback=_distributed_checkpoint_cb,
                phase_max_iter=phase_max_iter, residual_drop_threshold=residual_drop_threshold,
            )
            print(f"\n✅ Distributed Simulation Finished")

            # 保存结果（分布式版本：root 收集全局数据后保存）
            distributed_save_results(solver, output_dir)
            distributed_save_checkpoint(
                solver, output_dir, max_iter,
                input_file, solver.current_order, turbulence_model, backend,
                target_order=solver.order,
            )

        except Exception as e:
            print(f"\n❌ Distributed Simulation Failed: {str(e)}")
            raise click.Abort()

    else:
        # 单机求解器路径：所有 rank 加载完整网格
        mesh, volume_data = load_mesh_for_solver(
            input_file, order, surface_mesh=surface_mesh, skip_quality_check=skip_quality_check,
        )
        # 单机求解器路径（默认）
        solver = FRSolver(
            mesh=mesh,
            backend=backend,
            order=order,
            turb_model_name=turbulence_model,
            time_scheme=TimeIntegrationScheme.SSP_RK3,
            n_threads=threads,
            cfl_start=cfl_start,
            cfl_max=cfl_max,
            cfl_min=cfl_min,
            turbulence_intensity=turbulence_intensity,
            viscosity_ratio=viscosity_ratio,
            sem_num_eddies=sem_num_eddies,
            mu_molecular=mu_molecular,
            rho_inf=rho_inf, vel_inf=vel_inf, p_inf=p_inf,
            flux_type=flux_type,
            artificial_viscosity_enabled=artificial_viscosity_enabled,
            artificial_viscosity_alpha=artificial_viscosity_alpha,
            entropy_stable_volume_enabled=entropy_stable_volume_enabled,
        )

        # 2.5. 计算壁面距离场（如果湍流模型需要）
        compute_wall_distance_for_solver(solver, volume_data, use_eikonal=use_eikonal)

        # 传递参考面积到求解器，供迭代中输出气动力系数
        # 如果未指定 --reference-area，尝试从面网格自动计算投影面积
        if reference_area is None:
            auto_ref_area = _compute_reference_area_auto(volume_data)
            if auto_ref_area is not None:
                reference_area = auto_ref_area
        solver._reference_area = reference_area

        # 构建中间 checkpoint 保存回调（每 checkpoint_interval 步保存一次）
        def _checkpoint_cb(solver_ref, iteration):
            if iteration % checkpoint_interval != 0:
                return
            try:
                save_results(solver_ref, output_dir, quiet=True)
                # 必须用 solver_ref.current_order（这一步实际求解用的阶数），
                # 不能用外层闭包捕获的 order（CLI --order，Order Continuation
                # 的最终目标阶数）——真实复现：Order Continuation 还没爬升到
                # 目标阶数时（例如 P0 阶段的中间 checkpoint）两者不相等，用
                # 目标阶数重建 mesh/FRSolver 会得到与 checkpoint 里存的
                # U_sps 形状不匹配的 n_sps，resume 直接报错拒绝恢复。
                write_checkpoint(
                    solver_ref, output_dir, iteration,
                    input_file, solver_ref.current_order, turbulence_model, backend,
                    quiet=True, surface_mesh=surface_mesh, target_order=solver_ref.order,
                )
                print(f"   [Checkpoint] iter {iteration} saved")
            except Exception as e:
                print(f"   [Checkpoint] Warning: save failed at iter {iteration}: {e}")

        # 3. 执行求解
        try:
            result = solver.solve(max_iter=max_iter, dt=1e-3, tol=1e-6,
                                  checkpoint_callback=_checkpoint_cb,
                                  phase_max_iter=phase_max_iter,
                                  residual_drop_threshold=residual_drop_threshold)
            print(f"\n✅ Simulation Finished: Iterations={result.iterations}, Residual={result.final_residual:.6e}")

            # 4. 保存结果（.pkl 全量状态 + HDF5 checkpoint，后者供 solve resume 使用）
            save_results(solver, output_dir)
            # solver.current_order 而非 order：理由同上方 _checkpoint_cb 里的
            # 说明。正常跑完的情况下 Order Continuation 应该已经爬升到目标
            # 阶数、二者相等，但读活的值而不是假设闭包变量仍然成立更稳妥。
            write_checkpoint(
                solver, output_dir, result.iterations, input_file, solver.current_order,
                turbulence_model, backend, surface_mesh=surface_mesh, target_order=solver.order,
            )

            # 5. 气动系数（提供 --reference-area 时）
            _report_aerodynamic_coefficients(solver, reference_area)

        except Exception as e:
            print(f"\n❌ Simulation Failed: {str(e)}")
            raise click.Abort()
