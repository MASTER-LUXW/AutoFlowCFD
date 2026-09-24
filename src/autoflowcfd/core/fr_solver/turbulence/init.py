"""AutoFlowCFD V2.0 - 湍流模型的构造与自由来流/边界值设定。

从 `core/fr_solver/turbulence.py`（原 795 行）拆出（2026-09-24，项目
"单文件不超 500 行"规范）。**纯搬家，逻辑未改** —— 判据是 SST 黄金轨迹
逐位相同（见 `ProjectFiles/V2.0/29`）。
"""


import numpy as np
from loguru import logger

from autoflowcfd.core.turbulence.sst import SSTModelFR
from autoflowcfd.core.turbulence.des import DDESModel, IDDESModel, compute_h_max_and_h_wn
from autoflowcfd.core.turbulence.wmles import WMLESModel
from autoflowcfd.core.turbulence.sgs import WALEModel


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
    vel_inf = solver.freestream.get("vel_inf", 33.33)
    rho_inf = solver.freestream.get("rho_inf", 1.225)
    mu = getattr(solver, 'mu_molecular', 1.8e-5)
    nu = mu / max(rho_inf, 1e-10)

    # 外部气动默认值（参考 Fluent 手册：Tu ≤ 1%, VR = 2-10）
    Tu = getattr(solver, '_turbulence_intensity', 0.01)
    VR = getattr(solver, '_viscosity_ratio', 5.0)

    k_inf = 1.5 * (vel_inf * Tu) ** 2
    nu_t_inf = VR * nu
    omega_inf = k_inf / max(nu_t_inf, 1e-30)

    logger.debug(
        f"Freestream turbulence: Tu={Tu}, VR={VR}, "
        f"k_inf={k_inf:.4e}, omega_inf={omega_inf:.4e}, "
        f"tau={1.0/(0.09*omega_inf):.6e}s"
    )
    return k_inf, omega_inf


def _set_turbulence_bounds(solver) -> None:
    """根据来流条件设置 k/omega 物理上界。

    k_max = 0.5 * vel_inf^2：湍动能不可能超过平均流动能（湍流强度 100% 的极限）。
    omega_max = 1e6：远大于任何工程壁面 omega 值（壁面 omega ~ U_tau^2/nu ~ 1e4
    量级，1e6 留 100 倍裕度）。

    不设上界时，SST 输运方程的源项+输运项正反馈会导致 k/omega 指数增长到
    1e260+ 量级（实测 cube_demo 100 步内即达到），而平均流完全不受影响
    （nu_t 被 SST a1 限幅保持合理），形成隐蔽的发散失效模式。
    """
    if not hasattr(solver, 'turb_model') or solver.turb_model is None:
        return
    if not hasattr(solver.turb_model, 'k_max'):
        return  # 不是 SST 模型，无上界属性
    vel_inf = solver.freestream.get("vel_inf", 33.33)
    solver.turb_model.k_max = 0.5 * vel_inf ** 2  # 湍动能 ≤ 平均流动能
    solver.turb_model.omega_max = 1e6  # 保守上界
    logger.debug(
        f"Turbulence bounds set: k_max={solver.turb_model.k_max:.2f}, "
        f"omega_max={solver.turb_model.omega_max:.0e} "
        f"(vel_inf={vel_inf:.2f})"
    )


def _update_production_ramp(solver) -> None:
    """更新湍流产项渐变因子。

    前 N 步内 production_factor 从 0 线性增加到 1，防止初始流场未发展时
    P_k >> D_k（产生项超过耗散项 8 个量级）导致 k/omega 指数爆炸。
    工业 RANS 求解器（Fluent、OpenFOAM）的标准做法。

    渐变完成时设置 _turb_production_ramp_complete = True（一次性标记），
    供 Order Continuation 等上层逻辑检测并重置残差基准值。
    """
    if not hasattr(solver, 'turb_model') or solver.turb_model is None:
        return
    if not hasattr(solver.turb_model, 'production_factor'):
        return
    ramp_steps = getattr(solver, '_turb_production_ramp_steps', 0)
    current_step = getattr(solver, '_turb_ramp_step', 0)
    if ramp_steps <= 0 or current_step >= ramp_steps:
        solver.turb_model.production_factor = 1.0
        # 渐变完成：一次性标记（之前未完成且现在已完成）
        if not getattr(solver, '_turb_production_ramp_complete', False):
            solver._turb_production_ramp_complete = True
            logger.info(
                f"[ProductionRamp] Ramp complete after {ramp_steps} steps, "
                f"production_factor = 1.0"
            )
    else:
        solver.turb_model.production_factor = current_step / ramp_steps
    # 递增计数器（每调用一次代表一个迭代步）
    solver._turb_ramp_step = current_step + 1


def init_turbulence_models(solver, n_cells: int, n_sps: int) -> None:
    """初始化湍流模型（对应 FRSolver._init_turbulence_models）。"""
    # 湍流产项渐变计数器（与迭代步数同步，控制 production_factor 从 0 渐增到 1）
    solver._turb_ramp_step = 0
    # 渐变完成标记（_update_production_ramp 在渐变完成时设为 True）
    solver._turb_production_ramp_complete = False
    # 渐变步数：前 turb_production_ramp_steps 步内，产生项从 0 线性增加到全量。
    # 工业 RANS 标准做法：防止初始流场未发展时 P_k >> D_k 导致 k/omega 指数爆炸。
    # 50 步足够：配合物理上界限制（k_max, omega_max），k/omega 在此步数内达到准平衡。
    # Fluent 默认 ~50 步，OpenFOAM ~100 步；过长的 ramp 浪费收敛机会。
    solver._turb_production_ramp_steps = 50

    # 从 Tu/VR 推导物理自洽的 k/omega 初值（工业标准）
    k_inf, omega_inf = _set_freestream_turbulence(solver)

    if solver.turb_model_name == "SST":
        solver.turb_model = SSTModelFR(n_cells, n_sps, k_inf=k_inf, omega_inf=omega_inf)
        _set_turbulence_bounds(solver)
        _update_production_ramp(solver)
        print(f"   [OK] SST k-omega model initialized "
              f"(k_inf={k_inf:.4e}, omega_inf={omega_inf:.4e}, "
              f"production ramp: {solver._turb_production_ramp_steps} steps)")

    elif solver.turb_model_name == "DDES":
        solver.turb_model = SSTModelFR(n_cells, n_sps, k_inf=k_inf, omega_inf=omega_inf)
        _set_turbulence_bounds(solver)
        _update_production_ramp(solver)
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
              f"k_inf={k_inf:.4e}, omega_inf={omega_inf:.4e}, "
              f"production ramp: {solver._turb_production_ramp_steps} steps)")

    elif solver.turb_model_name == "IDDES":
        solver.turb_model = SSTModelFR(n_cells, n_sps, k_inf=k_inf, omega_inf=omega_inf)
        _set_turbulence_bounds(solver)
        _update_production_ramp(solver)
        solver.ddes_model = IDDESModel()
        # h_max/h_wn 只依赖网格几何（边长），与流场状态无关——mesh 在
        # 整个求解过程中不变，初始化时算一次并缓存在 solver 上，避免
        # 每步都重新调用 quality_metrics 的边长几何计算（见
        # compute_turbulence_source 里 solver._iddes_h_max/_iddes_h_wn
        # 的消费点）。
        solver._iddes_h_max, solver._iddes_h_wn = compute_h_max_and_h_wn(solver.mesh)
        print(f"   [OK] IDDES model initialized (based on SST, "
              f"k_inf={k_inf:.4e}, omega_inf={omega_inf:.4e}, "
              f"production ramp: {solver._turb_production_ramp_steps} steps)")

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
        print(f"   [OK] WMLES model initialized")

    elif solver.turb_model_name == "LES":
        solver.sgs_model = WALEModel()
        print(f"   [OK] LES with WALE SGS model initialized")

    elif solver.turb_model_name == "NONE":
        print(f"   [OK] Laminar flow (no turbulence model)")

    else:
        raise ValueError(f"Unknown turbulence model: {solver.turb_model_name}")
