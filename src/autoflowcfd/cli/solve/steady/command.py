"""`solve steady` 命令：click 选项、公共前段（物理常数解析与范围校验）与后端分派。

四个后端分支在同目录的 `multi_gpu`/`single_gpu`/`cpu_mpi`/`cpu_single` 模块
（2026-09-25 从单文件 `cli/solve/steady.py` 拆出）；命令组定义见 `cli/solve/commands.py`。
"""

import click

from autoflowcfd.cli.solve.helpers import (
    load_physical_config_if_given,
    resolve_physical_constants,
    resolve_turbulence_model,
)
from autoflowcfd.cli.solve.commands import solve

from .multi_gpu import _run_multi_gpu
from .single_gpu import _run_single_gpu
from .cpu_mpi import _run_cpu_mpi
from .cpu_single import _run_cpu_single


@solve.command(name='steady')
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--backend', type=click.Choice(['cpu', 'gpu']), default='cpu', help='计算后端 (CPU/GPU)')
@click.option('--order', type=int, default=2, help='FR 多项式阶数 (P1/P2/P3)')
@click.option('--turbulence-model', type=click.Choice(['none', 'sst', 'ddes', 'iddes', 'wmles', 'les']), default='sst',
              help='湍流模型。真实bug修复（2026-09-02，排查多GPU分布式DDES/IDDES时发现）：此前这里的'
                   'Choice列表缺 iddes/les 两项——底层单机/CPU MPI/多GPU分布式路径均已支持这两个模型'
                   '（solve transient命令的Choice列表本来就包含它们），steady命令这里一直没有同步，'
                   '导致 --turbulence-model iddes/les 在steady命令下无法使用（会被click直接拒绝），'
                   '与transient命令行为不一致。另外注意：ddes/iddes/les 会在 VELOCITY_INLET 边界'
                   '自动启用 BD-02 合成湍流入口 (SEM)；wmles 不会（WMLES 依赖壁面模型本身正确'
                   '预测近壁应力，不需要额外的入口湍流结构，见 core/fr_solver/boundary.py 文档）')
@click.option('--max-iter', type=int, default=1000, help='最大迭代次数')
@click.option('--time-scheme', type=click.Choice(['rk3', 'newton-krylov']), default='rk3',
              help='稳态伪时间推进格式。rk3：显式 SSP-RK3（CFL 受显式稳定极限约束，'
                   '真实网格上走完一个绕体特征时间要上万步）；newton-krylov：矩阵自由 '
                   'Newton-Krylov + 伪瞬态延拓（SER CFL 律 + 单元块 Jacobi 预处理，见 '
                   'core/time_integration/implicit/），收敛步数由非线性程度而不是最小'
                   '单元决定。newton-krylov 支持单机 CPU 与单机 GPU；分布式（--n-ranks>1、'
                   '--multi-gpu）尚未实现，会明确报错。')
@click.option('--cfl-start', type=float, default=None,
              help='自适应 CFL 初始值。**默认按 --time-scheme 取该格式 CFL 律的签名默认值**'
                   '（rk3 为显式控制器的 0.03；newton-krylov 为 SER 律的 5）——两者差两个'
                   '数量级，CLI 不再写死任何一个（adaptive_cfl/policy.py）。以下为 rk3 默认值'
                   '的标定历史：初始值（稳态伪时间迭代，默认 0.03）。残差不下降时 '
                   'CFL 会一直停在这个值——复杂网格上如果起步就发散可调低。下限是'
                   '独立的 --cfl-min。**2026-09-17 从 0.1 下调**：0.1 是 AUSM+up '
                   'P5± 饱和缺陷（提交 837cd95）修复之前定的，而那个缺陷本身让通量'
                   '的谱半径大 5.2 倍；修复后重新定界（见 --cfl-max 帮助）。0.03 是'
                   '唯一在真实网格上跑过 350+ 步单调下降的起步值，控制器从它往 '
                   '--cfl-max 爬。')
@click.option('--cfl-max', type=float, default=None,
              help='自适应 CFL 上限，默认按 --time-scheme 取（rk3 为 0.06；newton-krylov 为 '
                   'SER 的 1e4——隐式没有线性稳定极限，上限只防止 dtau 失去伪瞬态阻尼）。'
                   'rk3 默认值标定历史：上限（稳态，默认 0.06）。**2026-09-17 从 0.5 '
                   '下调**，依据是三类实测：(1) 直接谱测量——预处理后算子 '
                   'Gamma^-1 R 在干净通道网格上 max|dt*lambda| = 0.444 @CFL 0.03、'
                   '裕度 3.9 倍，对应线性极限 CFL 约 0.117（同一测量还发现旧的 '
                   'AUSM+up legacy 档在 CFL 0.03 就已越界 1.33 倍，这解释了为什么'
                   '此前记录的稳定边界只有 0.036）；(2) 真实网格 plate_demo'
                   '（363k 单元）0.03 稳、0.10 稳（峰后回落 8.5%）、0.30 第 13 步'
                   '发散；(3) 平板边界层通道在 0.10 第 3187 步发散——所以 0.10 '
                   '不是普遍安全值。0.06 低于每一个实测失效点且留 1.7 倍以上裕度。'
                   '为什么必须留这么多：越界一次之后收缩救不回来（见 '
                   'core/time_integration/adaptive_cfl.py 模块文档第 12 条）。'
                   '旧文案里那句"SSP-RK3 线性稳定极限 ~1.0"是标量对流的教科书值，'
                   '与本项目 CFL 参数的定义（面基谱半径 + 低马赫预处理波速）不是'
                   '同一个量纲，已删除。')
@click.option('--cfl-min', type=float, default=None,
              help='自适应 CFL 下限，默认按 --time-scheme 取（rk3 为 0.01；newton-krylov 为 '
                   'SER 的 0.5，更小的伪时间步由 Newton 步自身的 dtau 缩档负责）。'
                   'rk3 默认值标定历史：下限（稳态，默认 0.01）。**2026-09-17 从 0.05 '
                   '改为 0.01**：0.05 高于真 P1（AFCFD_FILTER_MODE=off，零阶数'
                   '损失）在 79 万单元 cube_demo 与 plate_demo 两张真实网格上'
                   '实测稳定的 ~0.03，等于一个已验证可用的工作点通过 CLI 根本'
                   '到不了（--cfl-start 会被这个下限钳上去，见 '
                   'core/time_integration/adaptive_cfl.py 模块文档第 11 条）。'
                   '配置层 `SteadyConfig.cfl_min` 早在 2026-09-15 就是 0.01，'
                   '这里没跟上——那正是提交 5e4e18a"配置层与 CLI 默认值相差 '
                   '20 倍"没关完的另一半。改动方向是安全的：下限只允许控制器'
                   '收缩得更多，绝不会抬高 CFL，因此不可能把原本稳定的运行变'
                   '成不稳定。')
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
def solve_steady(input_file, backend, order, turbulence_model, max_iter, time_scheme, cfl_start, cfl_max, cfl_min,
                 aoa_deg, aos_deg, phase_max_iter, residual_drop_threshold, output_dir, checkpoint_interval, use_eikonal, surface_mesh, skip_quality_check, reference_area, threads, n_ranks, fully_distributed, gpu_device, multi_gpu, turbulence_intensity, viscosity_ratio, sem_num_eddies, mu_molecular, rho_inf, vel_inf, p_inf, config_path, artificial_viscosity_enabled, artificial_viscosity_alpha, entropy_stable_volume_enabled):
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
    # --phase-max-iter/--residual-drop-threshold（2026-09-02 续接）：
    # 全部四种后端（单机 CPU/单 GPU/多GPU/MPI 分布式，含"传统模式"与
    # "完全分布式加载"）现在都真正接入了 Order Continuation（`solve()`
    # 在 `self.order>=2` 时自动分派到 `run_distributed_order_
    # continuation`，见 core/mpi/distributed_order_continuation.py/
    # core/gpu/solver/gpu_solver_order_continuation.py 模块文档），
    # 不再需要任何"某后端不支持"的拒绝。
    print(f"\nInput Grid : {input_file}")
    print(f"Backend    : {backend} | Order: P{order} | Method: {time_scheme}")
    if time_scheme == 'newton-krylov' and n_ranks > 1:
        raise click.UsageError(
            "--time-scheme newton-krylov 的分布式版本（--n-ranks>1 / --multi-gpu）尚未"
            "实现：GMRES 与块 Jacobi 需要跨 rank 的归约与全局着色。请用单机（CPU 或"
            "单 GPU），或改用 --time-scheme rk3。")
    print(f"Turbulence : {turbulence_model} | Max Iter: {max_iter}")
    if n_ranks > 1:
        print(f"MPI Ranks  : {n_ranks} (domain decomposition)")
    if use_eikonal:
        print(f"Wall Dist : Eikonal (graph-Dijkstra approx)\n")
    else:
        print(f"Wall Dist : KD-Tree (Geometric)\n")

    # 1. 按后端分派（各分支的参数由 AST 分析得出，逐名传入）
    if backend == 'gpu' and multi_gpu and (n_ranks > 1):
        _run_multi_gpu(
            use_eikonal=use_eikonal,
            aoa_deg=aoa_deg,
            aos_deg=aos_deg,
            cfl_max=cfl_max,
            cfl_min=cfl_min,
            cfl_start=cfl_start,
            checkpoint_interval=checkpoint_interval,
            fully_distributed=fully_distributed,
            gpu_device=gpu_device,
            input_file=input_file,
            max_iter=max_iter,
            mu_molecular=mu_molecular,
            n_ranks=n_ranks,
            order=order,
            output_dir=output_dir,
            p_inf=p_inf,
            phase_max_iter=phase_max_iter,
            residual_drop_threshold=residual_drop_threshold,
            rho_inf=rho_inf,
            skip_quality_check=skip_quality_check,
            surface_mesh=surface_mesh,
            turbulence_intensity=turbulence_intensity,
            turbulence_model=turbulence_model,
            vel_inf=vel_inf,
            viscosity_ratio=viscosity_ratio,
        )
    elif backend == 'gpu' and (not multi_gpu):
        _run_single_gpu(
            use_eikonal=use_eikonal,
            aoa_deg=aoa_deg,
            aos_deg=aos_deg,
            cfl_max=cfl_max,
            cfl_min=cfl_min,
            cfl_start=cfl_start,
            gpu_device=gpu_device,
            input_file=input_file,
            max_iter=max_iter,
            mu_molecular=mu_molecular,
            order=order,
            output_dir=output_dir,
            p_inf=p_inf,
            phase_max_iter=phase_max_iter,
            residual_drop_threshold=residual_drop_threshold,
            rho_inf=rho_inf,
            skip_quality_check=skip_quality_check,
            surface_mesh=surface_mesh,
            time_scheme=time_scheme,
            turbulence_intensity=turbulence_intensity,
            turbulence_model=turbulence_model,
            vel_inf=vel_inf,
            viscosity_ratio=viscosity_ratio,
        )
    elif n_ranks > 1:
        _run_cpu_mpi(
            use_eikonal=use_eikonal,
            aoa_deg=aoa_deg,
            aos_deg=aos_deg,
            backend=backend,
            cfl_max=cfl_max,
            cfl_min=cfl_min,
            cfl_start=cfl_start,
            checkpoint_interval=checkpoint_interval,
            fully_distributed=fully_distributed,
            input_file=input_file,
            max_iter=max_iter,
            mu_molecular=mu_molecular,
            n_ranks=n_ranks,
            order=order,
            output_dir=output_dir,
            p_inf=p_inf,
            phase_max_iter=phase_max_iter,
            residual_drop_threshold=residual_drop_threshold,
            rho_inf=rho_inf,
            skip_quality_check=skip_quality_check,
            surface_mesh=surface_mesh,
            threads=threads,
            turbulence_intensity=turbulence_intensity,
            turbulence_model=turbulence_model,
            vel_inf=vel_inf,
            viscosity_ratio=viscosity_ratio,
        )
    else:
        _run_cpu_single(
            aoa_deg=aoa_deg,
            aos_deg=aos_deg,
            artificial_viscosity_alpha=artificial_viscosity_alpha,
            artificial_viscosity_enabled=artificial_viscosity_enabled,
            backend=backend,
            cfl_max=cfl_max,
            cfl_min=cfl_min,
            cfl_start=cfl_start,
            checkpoint_interval=checkpoint_interval,
            entropy_stable_volume_enabled=entropy_stable_volume_enabled,
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
            turbulence_intensity=turbulence_intensity,
            turbulence_model=turbulence_model,
            use_eikonal=use_eikonal,
            vel_inf=vel_inf,
            viscosity_ratio=viscosity_ratio,
        )
