"""AutoFlowCFD V2.0 - 湍流模型的构造与自由来流/边界值设定。

从 `core/fr_solver/turbulence.py`（原 795 行）拆出（2026-09-24，项目
"单文件不超 500 行"规范）。**纯搬家，逻辑未改** —— 判据是 SST 黄金轨迹
逐位相同（见 `ProjectFiles/V2.0/29`）。
"""


import numpy as np
from loguru import logger

from autoflowcfd.core.turbulence.sa import SAModel
from autoflowcfd.core.turbulence.sst import SSTModelFR
from autoflowcfd.core.turbulence.des import DDESModel, IDDESModel, compute_h_max_and_h_wn
from autoflowcfd.core.turbulence.wmles import WMLESModel
from autoflowcfd.core.turbulence.sgs import WALEModel
from autoflowcfd.core.turbulence.sst.bounds import OMEGA_MAX_FLOOR

from .wall_distance import apply_wall_distance_to_model


def _filter_matrices_are_identity(ops) -> bool:
    """两个模态滤波矩阵是否都是单位阵。

    `AFCFD_FILTER_MODE=off` 下（以及 mild 档取 sigma_top=1.0 这种等价
    配置）矩阵就是单位阵，继续对 k/omega 各乘一遍纯属浪费内存带宽
    （79 万单元 P1 下每个标量场 (791492,8) ≈ 50MB，k 与 omega 各一遍，
    每步一次）。判据直接看矩阵内容而不是环境变量，不依赖"环境变量与
    算子构造保持同步"这个隐含假设——与 fr_solver/filter.py::
    build_filter_func 里的短路同一个理由、同一套判据。
    """
    for M in (getattr(ops, 'filter_prism', None), getattr(ops, 'filter_tet', None)):
        if M is None:
            continue
        A = np.asarray(M)
        if not (A.ndim == 2 and A.shape[0] == A.shape[1]):
            return False
        # 容差而不是逐位相等：`off` 档矩阵是 np.eye 直接返回、逐位相等，
        # 但 `mild` 档取 sigma_top=1.0 时矩阵是数值算出的
        # `V @ diag(1) @ inv(V)`——数学上是单位阵、浮点上偏差 ~1e-16。
        # 那种配置同样是无操作，同样应当被短路。容差 1e-12：元素是 O(1)、
        # Vandermonde 条件数在本项目工作阶数下只有个位数（order=3 才 56），
        # 1e-12 远高于舍入噪声又远低于任何有意义的滤波强度。
        if not np.allclose(A, np.eye(A.shape[0]), rtol=0.0, atol=1e-12):
            return False
    return True


def _freestream_turbulence_parameters(solver) -> tuple:
    """来流湍流参数 `(Tu, VR)`：湍流强度与涡粘比 `mu_t/mu`（SST 由两者定 k/omega，SA-neg 由 VR
    定 nu_tilde，两种模型按同一组物理量给定来流）。外部气动默认值参考 Fluent 手册
    （Tu <= 1%，VR = 2~10）。"""
    return getattr(solver, '_turbulence_intensity', 0.01), getattr(solver, '_viscosity_ratio', 5.0)


def _set_freestream_turbulence(solver) -> tuple:
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
    # `test_bounds_sensor_mirror` 一直传着键名全错的
    # `{"rho": 1.0, "u": 1.0, ...}`，作者想要单位量级，实际用的是兜底的
    # p_inf=101325 —— 没人发现，因为什么都没报错。
    vel_inf = solver.freestream["vel_inf"]
    rho_inf = solver.freestream["rho_inf"]
    mu = getattr(solver, 'mu_molecular', 1.8e-5)
    nu = mu / max(rho_inf, 1e-10)

    Tu, VR = _freestream_turbulence_parameters(solver)

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


def _set_turbulence_bounds(solver) -> None:
    """根据来流条件设置 k/omega 物理上界。

    k_max = 0.5 * vel_inf^2：湍动能不可能超过平均流动能（湍流强度 100% 的极限）。
    omega_max：随最近壁面解点给定（`turbulence/sst/bounds.py` 模块文档"omega 上界随最近
    壁面解点给定"）；壁距尚未设定时取 `OMEGA_MAX_FLOOR`，设定壁距时再按它重定。

    不设上界时，SST 输运方程的源项+输运项正反馈会导致 k/omega 指数增长到
    1e260+ 量级（实测 cube_demo 100 步内即达到），而平均流完全不受影响
    （nu_t 被 SST a1 限幅保持合理），形成隐蔽的发散失效模式。
    """
    if not hasattr(solver, 'turb_model') or solver.turb_model is None:
        return
    if not hasattr(solver.turb_model, 'k_max'):
        return  # 不是 SST 模型，无上界属性
    # 直接取键，理由见 `_set_freestream_turbulence` 同名注释
    vel_inf = solver.freestream["vel_inf"]
    solver.turb_model.k_max = 0.5 * vel_inf ** 2  # 湍动能 ≤ 平均流动能
    solver.turb_model.omega_max = OMEGA_MAX_FLOOR
    if getattr(solver, "wall_distance", None) is not None:
        apply_wall_distance_to_model(solver)
    logger.debug(
        f"Turbulence bounds set: k_max={solver.turb_model.k_max:.2f}, "
        f"omega_max={solver.turb_model.omega_max:.0e} "
        f"(vel_inf={vel_inf:.2f})"
    )


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
    `advance_production_ramp` 里的懒默认值。checkpoint 恢复湍流场后由恢复路径把计数器推到终点。
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


def _update_production_ramp(solver) -> None:
    """CPU 求解器（及其分布式视图适配器）的产生项渐变，见 `advance_production_ramp`。"""
    advance_production_ramp(solver, getattr(solver, 'turb_model', None))


def init_turbulence_models(solver, n_cells: int, n_sps: int) -> None:
    """初始化湍流模型（对应 FRSolver._init_turbulence_models）。产生项渐变计数器由构造点在时间格式确定后经
    `init_production_ramp` 设置。"""

    # 从 Tu/VR 推导物理自洽的 k/omega 初值（工业标准）
    k_inf, omega_inf = _set_freestream_turbulence(solver)

    if solver.turb_model_name == "SST":
        solver.turb_model = SSTModelFR(n_cells, n_sps, k_inf=k_inf, omega_inf=omega_inf)
        _set_turbulence_bounds(solver)
        print(f"   [OK] SST k-omega model initialized "
              f"(k_inf={k_inf:.4e}, omega_inf={omega_inf:.4e})")

    elif solver.turb_model_name == "DDES":
        solver.turb_model = SSTModelFR(n_cells, n_sps, k_inf=k_inf, omega_inf=omega_inf)
        _set_turbulence_bounds(solver)
        solver.ddes_model = DDESModel()
        # h_max（2026-09-02 补齐，与下面 IDDES 分支同一处几何量、同一个
        # 一次性缓存策略）：`apply_to_sst_model` 现在优先用各向异性感知
        # 的 max_edge 网格尺度而不是 cube_root(V)，见该方法文档——本项目
        # 高度依赖棱柱边界层网格，cube_root 会系统性低估扁平单元的 Δ。
        # 只需要 h_max（第一个返回值），h_wn 是 IDDES 专属几何量，DDES
        # 不用，但 compute_h_max_and_h_wn 只有一个返回两者的接口，丢弃
        # 用不到的 h_wn 即可。
        solver._iddes_h_max, _ = compute_h_max_and_h_wn(solver.mesh)
        print(f"   [OK] DDES model initialized (based on SST, "
              f"k_inf={k_inf:.4e}, omega_inf={omega_inf:.4e})")

    elif solver.turb_model_name == "IDDES":
        solver.turb_model = SSTModelFR(n_cells, n_sps, k_inf=k_inf, omega_inf=omega_inf)
        _set_turbulence_bounds(solver)
        solver.ddes_model = IDDESModel()
        # h_max/h_wn 只依赖网格几何（边长），与流场状态无关——mesh 在
        # 整个求解过程中不变，初始化时算一次并缓存在 solver 上，避免
        # 每步都重新调用 quality_metrics 的边长几何计算（见
        # compute_turbulence_source 里 solver._iddes_h_max/_iddes_h_wn
        # 的消费点）。
        solver._iddes_h_max, solver._iddes_h_wn = compute_h_max_and_h_wn(solver.mesh)
        print(f"   [OK] IDDES model initialized (based on SST, "
              f"k_inf={k_inf:.4e}, omega_inf={omega_inf:.4e})")

    elif solver.turb_model_name == "SA":
        solver.turb_model = m = create_sa_model(solver, n_cells, n_sps)
        print(f"   [OK] SA-neg model initialized (nu_tilde_inf={m.nu_tilde_inf:.4e}, "
              f"chi_inf={m.nu_tilde_inf / m.nu_ref:.4g}, viscosity ratio={m.viscosity_ratio:g})")

    elif solver.turb_model_name == "WMLES":
        # `solver.wmles_model` 已在 `FRSolver.__init__` 第 3 步提前构造
        # （必须先于 boundary_ghost_provider 构造，见该处说明——2026-
        # 09-02 修复的构造顺序 bug）；这里不再重复构造，只在它意外为
        # None 时（例如某个不经过 FRSolver.__init__ 第 3 步、直接调用
        # 本函数的测试/脚本场景）按 GPU 版同一个公式补建，避免真正生产
        # 路径下出现两个物理等价但对象不同的 WMLESModel 实例。
        if getattr(solver, "wmles_model", None) is None:
            rho_inf = solver.freestream.get("rho_inf", 1.225)
            solver.wmles_model = WMLESModel(nu=solver.mu_molecular / max(rho_inf, 1e-10))
        solver.sgs_model = WALEModel()
        print("   [OK] WMLES model initialized")

    elif solver.turb_model_name == "LES":
        solver.sgs_model = WALEModel()
        print("   [OK] LES with WALE SGS model initialized")

    elif solver.turb_model_name == "NONE":
        print("   [OK] Laminar flow (no turbulence model)")

    else:
        raise ValueError(f"Unknown turbulence model: {solver.turb_model_name}")
