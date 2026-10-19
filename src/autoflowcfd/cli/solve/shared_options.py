"""`solve steady` / `solve transient` 共用的命令行选项（唯一定义）与选项默认值的唯一来源。

**默认值从配置类读**（`config/solver_config.py` 的 `SteadyConfig` / `TransientConfig` 字段默认值）：配置文件、
Python API 与命令行三条入口的默认值是同一份。2026-10-19 以前 CLI 选项各自写死默认值，与配置类不一致——稳态
最大步数 1000 vs 50、瞬态湍流模型 ddes vs sst、瞬态时间步长 1e-5 vs 1e-4、瞬态输出目录、`--gpu-device`
（稳态 CLI 默认 0：`--multi-gpu` 时每个 rank 都用 0 号 GPU，而帮助文字说"每个 rank 自动分配"；多 GPU 求解器
只有收到 None 时才按 rank % n_gpus 分配）。

**两条命令共用的选项只定义一次**：此前约 25 个选项在两个命令文件里各写一份，帮助文字已经各自漂移（同一个
`--aoa` 一边是一句约定说明、一边是修改日志；`--turbulence-intensity` 两边描述的作用范围不同）。
"""

import click

from autoflowcfd.cli.solve.option_help import PHASE_MAX_ITER_HELP, RESIDUAL_DROP_THRESHOLD_HELP, THREADS_HELP
from autoflowcfd.core.fr_solver.boundary.constants import _SEM_DEFAULT_NUM_EDDIES
from autoflowcfd.core.turbulence.wall_model.treatment import WALL_TREATMENTS
from autoflowcfd.core.utils.order_continuation.policy import MAX_ORDER


def config_default(config_cls, field: str, choices=None):
    """配置类字段的默认值，换成命令行词汇（湍流模型 `sst_kw` -> `sst`；时间格式取 `choices` 里对应的那个词）。"""
    from autoflowcfd.config.solver_config import field_default

    value = field_default(config_cls, field)
    if field == "turbulence":
        from autoflowcfd.config.turbulence_names import turbulence_solver_name

        return turbulence_solver_name(value)
    if field == "time_scheme":
        from autoflowcfd.core.time_integration.base import scheme_from_name

        return next(c for c in choices if scheme_from_name(c) == scheme_from_name(value))
    return getattr(value, "value", value)


def _stack(*decorators):
    def apply(func):
        for dec in reversed(decorators):
            func = dec(func)
        return func
    return apply


def shared_solve_options(config_cls):
    """两条求解命令共用的选项（默认值取自 `config_cls`）。"""
    d = lambda field: config_default(config_cls, field)  # noqa: E731
    lo = config_cls._min_order
    return _stack(
        click.option('--backend', '-b', type=click.Choice(['cpu', 'gpu']), default=d('backend'), show_default=True,
                     help='计算后端'),
        click.option('--order', '-p', type=click.IntRange(lo, MAX_ORDER), default=d('order'), show_default=True,
                     help=f'FR 目标阶数（{lo}~{MAX_ORDER}）；>=1 时从 P0 逐阶爬坡（Order Continuation），'
                          '--init-from 起步时直接在该阶数上推进'),
        click.option('--phase-max-iter', type=int, default=d('phase_max_iter'), help=PHASE_MAX_ITER_HELP),
        click.option('--residual-drop-threshold', type=float, default=d('residual_drop_threshold'),
                     show_default=True, help=RESIDUAL_DROP_THRESHOLD_HELP),
        click.option('--output', '-o', 'output_dir', type=click.Path(), default=d('output_dir'), show_default=True,
                     help='结果输出目录'),
        click.option('--checkpoint-interval', type=int, default=d('checkpoint_interval'), show_default=True,
                     help='中间 checkpoint 保存间隔（步，单机与分布式都生效），结束时另写一次'),
        click.option('--surface-mesh', '-s', type=click.Path(exists=True), default=None,
                     help='原始面网格路径：输入是 .nas 体网格时必填，用于反推边界分组；.pkl 不需要'),
        click.option('--skip-quality-check', is_flag=True, help='跳过求解前的网格质量门检查（不建议，仅用于临时诊断）'),
        click.option('--reference-area', type=float, default=None,
                     help='气动系数参考面积 (m^2)；不给时沿来流方向的迎风投影自动估算'),
        click.option('--threads', '-j', type=int, default=d('n_threads'), show_default=True, help=THREADS_HELP),
        click.option('--n-ranks', '--np', type=int, default=1,
                     help='MPI rank 数（>1 时域分解并行，需配合 mpirun；默认 1 = 单机）'),
        click.option('--fully-distributed', is_flag=True,
                     help='完全分布式网格加载：只有 root rank 加载完整网格，其余 rank 只接收切好的紧凑数据；'
                          '可与 --multi-gpu 组合，需要 --n-ranks>1'),
        click.option('--multi-gpu', is_flag=True, help='多 GPU + MPI 分布式求解（每个 rank 一块 GPU），需要 --backend gpu'),
        click.option('--gpu-device', type=int, default=d('gpu_device'),
                     help='GPU 设备号。不给时：单 GPU 用 0，--multi-gpu 按 rank % GPU 数自动分配；'
                          '给出时全部 rank 用该设备'),
        click.option('--turbulence-intensity', type=float, default=d('turbulence_intensity'), show_default=True,
                     help='来流湍流强度 Tu（0.01 = 1%）：RANS 来流 k 与 ddes/iddes/les/wmles 合成湍流入口的目标雷诺应力'),
        click.option('--viscosity-ratio', type=float, default=d('viscosity_ratio'), show_default=True,
                     help='来流粘性比 nu_t/nu（外部气动常用 2~10）'),
        click.option('--sem-num-eddies', type=int, default=_SEM_DEFAULT_NUM_EDDIES,
                     help='ddes/iddes/les/wmles 合成湍流入口的涡核数（不给时按 Jarrin 密度判据 V_B/sigma^3 取）'),
        click.option('--mu-molecular', type=float, default=d('mu_molecular'), show_default=True,
                     help='分子动力粘度 (Pa*s)，非标准状态空气请覆盖'),
        click.option('--rho-inf', type=float, default=d('rho_inf'), show_default=True, help='自由流密度 (kg/m^3)'),
        click.option('--vel-inf', type=float, default=d('vel_inf'), show_default=True, help='自由流速度大小 (m/s)'),
        click.option('--p-inf', type=float, default=d('p_inf'), show_default=True, help='自由流静压 (Pa)'),
        click.option('--aoa', 'aoa_deg', type=float, default=0.0, show_default=True,
                     help='攻角 alpha（度，绕 y 轴、抬头为正）。来流边界、初场、合成湍流入口方向、气动力风轴系分解与'
                          '参考面积的迎风投影都按它构造；力矩报在体轴系。约定见 core/utils/flow_direction.py'),
        click.option('--aos', 'aos_deg', type=float, default=0.0, show_default=True,
                     help='侧滑角 beta（度，绕 z 轴），语义同 --aoa'),
        click.option('--config', 'config_path', type=click.Path(exists=True), default=None,
                     help='YAML 配置文件（`config init` 生成模板，字段见 docs/CONFIGURATION_GUIDE.md）：文件里写出的'
                          '字段覆盖对应选项的默认值，命令行显式给出的选项优先于文件'),
        click.option('--artificial-viscosity', 'artificial_viscosity_enabled', is_flag=True,
                     default=d('artificial_viscosity_enabled'),
                     help='问题单元人工粘性（全部后端）：熵残差判据 (h/p)·max|û·∇s| 定位数值熵产生的单元（P1 起有效），'
                          '只对它们叠加全部守恒变量的 div(ν∇U)（core/fr_operators/artificial_viscosity/）。设置写入 '
                          'checkpoint，续算沿用'),
        click.option('--av-alpha', 'artificial_viscosity_alpha', type=float, default=d('artificial_viscosity_alpha'),
                     show_default=True,
                     help='人工粘性强度标定常数（无量纲，ν = 斜坡·alpha·|u|·h/p），只在 --artificial-viscosity 时有意义'),
        click.option('--wall-treatment', type=click.Choice(list(WALL_TREATMENTS)), default=d('wall_treatment'),
                     show_default=True,
                     help='RANS（sa/sst/ddes/iddes）的近壁处理：resolved = 低雷诺数壁面处理（要求第一层 y+~1）；'
                          'auto = 自动壁面处理（Spalding 平衡壁面模型给壁面剪应力，第一层 y+ 从 1 到对数层都可用），'
                          '见 core/turbulence/wall_model/treatment.py。设置写入 checkpoint，续算沿用'),
    )
