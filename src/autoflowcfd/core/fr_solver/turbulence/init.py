"""AutoFlowCFD V2.0 - 湍流模型的构造与自由来流/边界值设定。"""

from typing import Any, NamedTuple, Optional

import numpy as np
from loguru import logger

from autoflowcfd.core.turbulence.sa import SAModel
from autoflowcfd.core.turbulence.sst import SSTModelFR
from autoflowcfd.core.turbulence.des import DDESModel, IDDESModel, compute_h_max_and_h_wn
from autoflowcfd.core.turbulence.sgs import WALEModel

from .wall_distance import apply_wall_distance_to_model


def _freestream_turbulence_parameters(solver) -> tuple:
    """来流湍流参数 `(Tu, VR)`：湍流强度与涡粘比 `mu_t/mu`（SST 由两者定 k/omega，SA-neg 由 VR
    定 nu_tilde，两种模型按同一组物理量给定来流）。外部气动默认值参考 Fluent 手册
    （Tu <= 1%，VR = 2~10）。全部求解器与完全分布式加载的根桩都设置这两个属性，直接取（此前 getattr 兜底让
    根桩上缺的值静默变成默认值）。"""
    return solver._turbulence_intensity, solver._viscosity_ratio


def _set_freestream_turbulence(solver) -> tuple:
    """求解器的来流 `(k_inf, omega_inf)`（`freestream_k_omega`，取求解器上的来流、分子粘度与 Tu / VR）。"""
    return freestream_k_omega(solver.freestream, solver.mu_molecular, *_freestream_turbulence_parameters(solver))


def freestream_k_omega(freestream: dict, mu_molecular: float, turbulence_intensity: float,
                       viscosity_ratio: float) -> tuple:
    """根据来流条件从 Tu/VR 推导物理自洽的 k/omega 初值。

    工业 RANS 标准做法（Fluent 用户手册 Section 7.3.2、OpenFOAM 通用实践）：
    不直接指定 k 和 omega，而是从湍流强度 Tu 和粘性比 VR 推导，
    确保 k 和 omega 通过 nu_t 物理耦合，避免拍脑袋组合导致源项失衡。

    公式：
        k_inf   = 1.5 * (U_inf * Tu)^2
        nu_t    = VR * nu（nu = mu/rho 运动粘度）
        omega_inf = k_inf / nu_t

    Returns:
        (k_inf, omega_inf): 来流湍动能和比耗散率
    """
    # 直接取键，不给兜底（2026-09-24）：`freestream` 对全部求解器类无条件
    # 设置，兜底永远不会生效；一旦某条路径真的丢了字段，`.get(k, 33.33)`
    # 会把缺陷伪装成"用了一个合理的来流"。真实例子：
    # 一个（随模态滤波已删除的）测试一直传着键名全错的
    # `{"rho": 1.0, "u": 1.0, ...}`，作者想要单位量级，实际用的是兜底的
    # p_inf=101325 —— 没人发现，因为什么都没报错。
    vel_inf = freestream["vel_inf"]
    rho_inf = freestream["rho_inf"]
    nu = mu_molecular / max(rho_inf, 1e-10)

    Tu, VR = turbulence_intensity, viscosity_ratio

    k_inf = 1.5 * (vel_inf * Tu) ** 2
    nu_t_inf = VR * nu
    omega_inf = k_inf / max(nu_t_inf, 1e-30)

    logger.debug(
        f"Freestream turbulence: Tu={Tu}, VR={VR}, "
        f"k_inf={k_inf:.4e}, omega_inf={omega_inf:.4e}, "
        f"tau={1.0/(0.09*omega_inf):.6e}s"
    )
    return k_inf, omega_inf


def create_sa_model(owner, n_cells: int, n_sps: int, xp=np) -> SAModel:
    """SA-neg 模型（CPU 求解器、GPU 求解器共用的构造）：来流由涡粘比 `VR` 给定（与 SST 定
    omega_inf 用的同一个参数，`_freestream_turbulence_parameters`），参考运动粘度 `mu / rho_inf`。"""
    _, VR = _freestream_turbulence_parameters(owner)
    return SAModel(n_cells, n_sps, nu_ref=owner.mu_molecular / owner.freestream["rho_inf"],
                   viscosity_ratio=VR, xp=xp)


#: 显式格式的湍流产生项渐变步数：前这么多步内 production_factor 从 0 线性增加到 1，防止初始流场未发展时
#: P_k >> D_k 导致 k/omega 指数爆炸（配合物理上界 k_max/omega_max）。
TURB_PRODUCTION_RAMP_STEPS = 50


def production_ramp_steps(time_scheme) -> int:
    """按时间格式给出产生项渐变步数（唯一来源）。

    隐式稳态（Newton-Krylov）不渐变：渐变按**步数**计，显式 50 步只是很短的一段伪时间，隐式 50 个 Newton 步
    却几乎是整个 P0 阶段，期间每步要解的方程都在变（产生项每步增加 2%），Newton 追着移动目标走——残差停在
    这 2% 变化的量级上，CFL 被反复回退。隐式的全局化已由 PTC（起步 CFL 5，强阻尼）与正性限制器提供，SA 用的
    是允许负值的 SA-neg。湍流平板 P0->P3 同一快照 A/B（2026-10-05，渐变 50 vs 0）：SA P0 55->30 步、GMRES
    501->158，P1~P3 步数相同，5 个站位 cf 逐位相同；SST P0 119->55 步、GMRES 5154->742，P1 32->24 步
    （SST 两臂都卡 P2，是已知的 SST 折点问题，与渐变无关）。
    """
    from autoflowcfd.core.time_integration.base import TimeIntegrationScheme, scheme_from_name

    if scheme_from_name(time_scheme) == TimeIntegrationScheme.NEWTON_KRYLOV:
        return 0
    return TURB_PRODUCTION_RAMP_STEPS


def init_production_ramp(owner, time_scheme) -> None:
    """置渐变计数器初值（五个求解器构造点共用，在时间格式确定之后调用）。

    构造时不推进计数器，每一步由 `advance_production_ramp` 推进一次：2026-10-05 以前 CPU 单机与 CPU MPI 传统
    模式在构造时就推进了一次（渐变期间产生项因子比单 GPU / CPU MPI 完全分布式 / 多 GPU 多 1/N），多 GPU 则靠
    `advance_production_ramp` 里的懒默认值。checkpoint 恢复湍流场后由 `restore_production_ramp` 续接计数。
    """
    owner._turb_ramp_step = 0
    owner._turb_production_ramp_steps = production_ramp_steps(time_scheme)
    owner._turb_production_ramp_complete = False


def advance_production_ramp(owner, model) -> None:
    """推进一步湍流产生项渐变：按 `owner` 上的计数器设置 `model.production_factor`。

    全部后端共用这一份（CPU 单机/分布式视图、单机 GPU、多 GPU）。`owner` 持有
    `init_production_ramp` 置好的 `_turb_ramp_step` / `_turb_production_ramp_steps` /
    `_turb_production_ramp_complete`（渐变完成时一次性置 True，供 Order Continuation 重置残差基准）。2026-09-25 以前 CPU 与
    单机 GPU 各写一份，多 GPU 分布式则**从未推进过**（`production_factor` 恒为 1，
    与其余后端前 50 步的物理不同）。
    """
    if model is None or not hasattr(model, 'production_factor'):
        return
    ramp_steps = owner._turb_production_ramp_steps
    current_step = owner._turb_ramp_step
    if ramp_steps <= 0 or current_step >= ramp_steps:
        model.production_factor = 1.0
        if not owner._turb_production_ramp_complete:
            owner._turb_production_ramp_complete = True
            logger.info(
                f"[ProductionRamp] Ramp complete after {ramp_steps} steps, "
                f"production_factor = 1.0"
            )
    else:
        model.production_factor = current_step / ramp_steps
    owner._turb_ramp_step = current_step + 1


RAMP_STEP_KEY = "turb_ramp_step"
RAMP_TOTAL_KEY = "turb_ramp_steps"
RAMP_COMPLETE_KEY = "turb_ramp_complete"


def production_ramp_metadata(owner) -> dict:
    """要写进 checkpoint 元数据的渐变进度（单机与分布式写入端共用）。"""
    return {RAMP_STEP_KEY: int(owner._turb_ramp_step), RAMP_TOTAL_KEY: int(owner._turb_production_ramp_steps),
            RAMP_COMPLETE_KEY: bool(owner._turb_production_ramp_complete)}


def restore_production_ramp(owner, model, metadata: dict) -> None:
    """checkpoint 恢复出湍流场之后续接产生项渐变（单机、单 GPU 主机视图与分布式恢复端共用）。

    2026-10-09 以前三处恢复路径一律把计数器推到终点（"湍流场已恢复 = 已充分发展"）：在渐变期间写出的
    checkpoint 续算时产生项因子从例如 0.02 跳到 1，续算与连续计算不是同一个算例。现在按 checkpoint 记录的
    进度续接：

    * 写出时渐变已走完（含不渐变的隐式稳态，总步数 0）：直接完成；
    * 写出时还在渐变中：从记录的步数继续（各显式格式的总步数同为 `TURB_PRODUCTION_RAMP_STEPS`）；
    * 早于本记录的 checkpoint：进度不可知，沿用此前的做法视为已完成。

    `_ramp_baseline_reset_done` 随完成标记置位：渐变已完成时 Order Continuation 不再重置残差基准（否则会丢掉
    刚恢复的阶段起始残差），未完成时留给渐变结束那一步重置。
    """
    if model is None or not hasattr(model, "production_factor"):
        return
    total = int(owner._turb_production_ramp_steps)
    if RAMP_STEP_KEY not in metadata:
        step, complete = total, True
    else:
        ckpt_step, ckpt_total = int(metadata[RAMP_STEP_KEY]), int(metadata[RAMP_TOTAL_KEY])
        if ckpt_step >= ckpt_total:
            step, complete = total, bool(metadata[RAMP_COMPLETE_KEY])
        else:
            step, complete = min(ckpt_step, total), False
    owner._turb_ramp_step = step
    owner._turb_production_ramp_complete = complete
    owner._ramp_baseline_reset_done = complete
    # 因子取上一次 `advance_production_ramp` 留下的值（下一步推进时会重设；主机视图回写设备时带着它）
    if complete or total <= 0:
        model.production_factor = 1.0
    elif step > 0:
        model.production_factor = (step - 1) / total


def _update_production_ramp(solver) -> None:
    """CPU 求解器（及其分布式视图适配器）的产生项渐变，见 `advance_production_ramp`。"""
    advance_production_ramp(solver, getattr(solver, 'turb_model', None))


class TurbulenceModels(NamedTuple):
    """`build_turbulence_models` 的结果：输运模型（SST 族 / SA-neg）、DES 长度尺度模型（DDES / IDDES）、
    代数亚格子模型（LES / WMLES 的 WALE）；不适用的为 None。"""
    transport: Any
    des: Any
    sgs: Any


def build_turbulence_models(owner, name: str, n_cells: int, n_sps: int, cp: Any = None,
                            device_id: Optional[int] = None) -> TurbulenceModels:
    """按模型名构造湍流模型对象——CPU 单机、CPU 分布式两种加载、单 GPU、多 GPU 两种加载唯一的构造处。

    SST 族：来流 k/omega 由 Tu/VR 推出（`_set_freestream_turbulence`），物理上界与近壁处理方式见
    `turbulence/sst/bounds.py::configure_sst_model`（omega 上界在壁距设定后按最近壁面解点重定）；SA-neg 的来流由
    涡粘比给定（`create_sa_model`）。网格尺度（DES 的 h_max / h_wn）与壁距由各后端按自己的单元编号给出。

    Args:
        owner: 求解器或其根桩（带 `freestream`、`mu_molecular`、`_turbulence_intensity`、`_viscosity_ratio`、
            `wall_treatment`）
        name: 湍流模型名（`turbulence/registry.py::SUPPORTED_MODELS`）
        cp, device_id: GPU 后端给出自己的 cupy 模块与设备号，输运模型的场在该 GPU 上（`GPUTurbulenceSST` / cupy 版
            SA-neg）；CPU 后端都不给

    2026-10-21（I-82）以前五个后端各写一条按名字构造的 if/elif 链。
    """
    from autoflowcfd.core.turbulence.registry import (
        is_sa_family, is_sst_family, normalized, require_supported, uses_algebraic_sgs,
    )

    require_supported(name)
    name = normalized(name)
    if (cp is None) != (device_id is None):
        raise ValueError("GPU 构造须同时给出 cupy 模块与设备号")
    transport: Any = None
    des: Any = None
    sgs: Any = None
    if is_sa_family(name):
        if cp is None:
            transport = create_sa_model(owner, n_cells, n_sps)
        else:
            with cp.cuda.Device(device_id):
                transport = create_sa_model(owner, n_cells, n_sps, xp=cp)
    elif is_sst_family(name):
        from autoflowcfd.core.turbulence.sst.bounds import configure_sst_model

        k_inf, omega_inf = _set_freestream_turbulence(owner)
        if device_id is None:
            transport = SSTModelFR(n_cells, n_sps, k_inf=k_inf, omega_inf=omega_inf)
        else:
            from autoflowcfd.core.gpu.turbulence.gpu_turbulence_sst import GPUTurbulenceSST

            transport = GPUTurbulenceSST(n_cells, n_sps, device_id, k_inf=k_inf, omega_inf=omega_inf)
        configure_sst_model(transport, owner.freestream["vel_inf"], owner.wall_treatment)
        if name == "DDES":
            des = DDESModel()
        elif name == "IDDES":
            des = IDDESModel()
    elif uses_algebraic_sgs(name):
        sgs = WALEModel()
    return TurbulenceModels(transport, des, sgs)


def describe_turbulence_models(name: str, models: TurbulenceModels) -> str:
    """启动日志里的一行湍流模型说明（各后端共用）。"""
    m = models.transport
    if isinstance(m, SAModel):
        return (f"SA-neg（nu_tilde_inf={m.nu_tilde_inf:.4e}, chi_inf={m.nu_tilde_inf / m.nu_ref:.4g}, "
                f"粘性比 {m.viscosity_ratio:g}）")
    if isinstance(m, SSTModelFR):
        base = f"k_inf={m.k_inf:.4e}, omega_inf={m.omega_inf:.4e}"
        return f"{name}（基于 SST k-omega，{base}）" if models.des is not None else f"SST k-omega（{base}）"
    if models.sgs is not None:
        return f"{name}（WALE 亚格子模型）"
    return "层流（无湍流模型）"


def init_turbulence_models(solver, n_cells: int, n_sps: int) -> None:
    """CPU 求解器（单机与传统模式分布式）的湍流模型初始化（对应 FRSolver._init_turbulence_models）：对象由
    `build_turbulence_models` 构造，这里补上 CPU 一侧按全局网格算的 DES 网格尺度与已设定的壁距。产生项渐变
    计数器由构造点在时间格式确定后经 `init_production_ramp` 设置。"""
    name = solver.turb_model_name
    # WMLES 的壁面模型由 `FRSolver.__init__` 先于边界幽灵态构造（幽灵态据它取滑移镜像），这里不补建
    if name == "WMLES" and getattr(solver, "wall_model", None) is None:
        raise RuntimeError("WMLES 的壁面模型必须先于边界幽灵态构造（FRSolver.__init__），这里不补建")
    models = build_turbulence_models(solver, name, n_cells, n_sps)
    solver.turb_model, solver.ddes_model, solver.sgs_model = models
    if models.des is not None:
        # 逐单元网格尺度只依赖几何，算一次缓存：DDES 的 LES 长度用最大边长 h_max（棱柱边界层网格上 cube_root(V)
        # 会系统性低估扁平单元的面内尺度），IDDES 另用壁法向尺度 h_wn
        solver._iddes_h_max, solver._iddes_h_wn = compute_h_max_and_h_wn(solver.mesh)
    if models.transport is not None and getattr(solver, "wall_distance", None) is not None:
        apply_wall_distance_to_model(solver)
    print(f"   [OK] {describe_turbulence_models(name, models)}")
