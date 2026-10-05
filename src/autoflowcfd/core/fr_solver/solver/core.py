"""AutoFlowCFD V2.0 - FR 求解器主类 (Final Integration)

本模块整合 FRState, HighOrderMesh, FR Kernels, Turbulence Models 及 Weak BCs。
它是 V2.0 求解器的总控中心，负责协调各模块完成 N-S 方程的高阶离散与求解。

核心功能:
1. 支持多种湍流模型（SST/DDES/IDDES/WMLES/LES）
2. 自动切换RANS/LES模式
3. 完整的时间推进循环
4. 残差监控和收敛判断

2026-09-25 按职责拆成同目录的 mixin（`setup` 装配阶段、`residuals` 残差与
委托、`solve_loop` 求解循环）与 `threads`（BLAS/numba 线程）；本文件只留
构造签名、参数文档与装配顺序。
"""

from autoflowcfd.core.time_integration.base import DEFAULT_DUAL_TIME_STEPS
from typing import Any, Dict, Optional

from autoflowcfd.core.fr_solver.state import FRState
from autoflowcfd.core.time_integration.base import TimeIntegrationScheme
from autoflowcfd.grid.high_order.high_order_mesh import HighOrderMesh

from ..solver_geometry import _SolverGeometryMixin
from .residuals import _SolverResidualMixin
from .setup import _SolverSetupMixin
from .solve_loop import _SolverSolveMixin
from .threads import configure_numba_threads


from autoflowcfd.core.fr_solver.boundary.constants import _SEM_DEFAULT_NUM_EDDIES

class FRSolver(_SolverSetupMixin, _SolverSolveMixin, _SolverResidualMixin,
               _SolverGeometryMixin):
    """
    基于通量重构 (FR) 方法的 N-S 方程求解器。
    
    Attributes:
        mesh: 高阶网格对象
        state: 求解器状态容器
        ops: 预计算的 FR 算子
        bc_handler: 弱边界条件处理器
        turb_model: 湍流模型处理器
    """
    def __init__(self, mesh: 'HighOrderMesh', order: int = 2,
                 turb_model_name: str = "SST",
                 time_scheme: TimeIntegrationScheme = TimeIntegrationScheme.SSP_RK3,
                 initial_state: Optional[FRState] = None,
                 rho_inf: float = 1.225, vel_inf: float = 33.33, p_inf: float = 101325.0,
                 # 攻角/侧滑角（度）。0/0 时来流严格沿 +x，与此前把方向
                 # 硬编码成 +x 的行为**逐位相同**，默认路径数值不变。
                 # 约定与风轴系公式见 core/utils/flow_direction.py。
                 aoa_deg: float = 0.0, aos_deg: float = 0.0,
                 bc_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
                 mu_molecular: float = 1.8e-5,
                 dual_time_inner_iter: int = DEFAULT_DUAL_TIME_STEPS,
                 n_threads: int = -1,
                 adaptive_cfl: bool = True,
                 # 三个 CFL 参数默认 None（2026-09-25）：由
                 # `adaptive_cfl/policy.py::build_cfl_policy` 按时间格式取对应
                 # CFL 律的签名默认值（显式控制器 0.03/0.06/0.01、隐式 SER
                 # 5/1e4/0.5，差两个数量级）。此前这里写死显式那一组，隐式
                 # 格式不传 CFL 时会静默拿到 0.03——等于退化成显式步长。
                 cfl_start: Optional[float] = None,
                 cfl_max: Optional[float] = None,
                 cfl_min: Optional[float] = None,
                 turbulence_intensity: float = 0.01,
                 viscosity_ratio: float = 5.0,
                 sem_num_eddies: int = _SEM_DEFAULT_NUM_EDDIES,
                 artificial_viscosity_enabled: bool = False,
                 artificial_viscosity_alpha: float = 1.0,
                 low_mach_precond: bool = True):
        """
        初始化 FRSolver。

        Args:
            mesh: HighOrderMesh 类型的高阶网格对象
            order: 多项式阶数
            turb_model_name: 湍流模型名称 ("SST"/"DDES"/"IDDES"/"WMLES"/"LES"/"NONE")
            mu_molecular: 分子动力粘度（默认 1.8e-5 Pa*s，标准状态下空气）。
                此前粘性残差（fr_residual_viscous.py 默认参数）与粘性 CFL
                步长（_compute_local_time_step）各自独立硬编码这个值，没有
                任何受支持的方式一致地设置自定义粘度——两处必须同步改，
                否则 CFL 步长会与真正参与残差组装的粘度脱节（同类问题见
                记忆条目 hardcoded_molecular_viscosity_mismatch）。现在两处
                都从这个构造参数读取。
            time_scheme: 时间推进方案
            initial_state: 初始状态（用于 Order Continuation）
            aoa_deg, aos_deg: 攻角/侧滑角（度）。此前来流方向在全代码库被
                硬编码成 +x、没有任何攻角选项；现在 `Q_free`（边界自由来流
                态）、初场、SEM 入口方向、气动力的风轴系分解、参考面积的
                迎风投影五处统一按这两个角度构造。0/0 时逐位退化为原行为。
            rho_inf, vel_inf, p_inf: 自由来流条件（密度/速度**大小**/静压；
                方向由 aoa_deg/aos_deg 决定，不再固定 +x），
                用作 FARFIELD 边界的幽灵态、未匹配到边界组的默认边界条件，
                以及 INLET 组未显式覆盖时的默认入口状态
            bc_overrides: 按边界组名称覆盖 BC 类型/参数，例如
                {"inlet": {"type": "INLET", "Q_inlet": [...]}, "car_body": {"type": "WALL"}}；
                未提供的组按 mesh.boundary_bc_types 自动映射
                （WALL/SLIP_WALL->WALL, VELOCITY_INLET->INLET,
                PRESSURE_OUTLET->OUTLET, SYMMETRY->SYMMETRY, 其余->FARFIELD）
            dual_time_inner_iter: DUAL_TIME 方案每个物理步的伪时间内迭代
                次数（仅 time_scheme=DUAL_TIME 时有意义）。此前完全没有
                途径设置，恒为硬编码 3；真实测得默认保守 CFL 起点下 3 次
                内迭代通常远不足以让伪残差收敛到物理时间精度要求的水平
                （见 TimeIntegrator.__init__ 与 step_dual_time 文档）。
            n_threads: numba 并行 kernel（无粘/粘性残差界面项，见
                core/fr_residual_inviscid_kernel.py、
                core/fr_viscous_flux_kernel.py 模块文档"多核并行"一节）
                使用的 CPU 线程数。默认 -1 **不是** `os.cpu_count()`，
                而是解析成 **8**——见下方 `_DEFAULT_N_THREADS` 处的完整
                实测记录（2026-09-13 重新测量：体积项/梯度链路改成 numba
                kernel 且 BLAS 线程限制为 1 之后，79 万单元真实网格 P1
                的甜点从此前的 4 上移到 8，nt=12/16 仍然净倒退，根因是
                界面项按图着色逐色串行调用导致的屏障开销 + 本机 P/E 混合
                核在静态均分调度下的不均衡）。调用方仍可显式传更大或更小
                的 n_threads 覆盖（CLI 见 `--threads`/`-j`）。
                只在这里调用一次 `numba.set_num_threads`——两个界面
                kernel 的 `n_threads` 参数要求调用方紧邻调用前取
                `numba.get_num_threads()`，如果这个全局状态在其他地方
                被并发修改，会破坏该约束（见两个 kernel 模块文档"多核
                并行"一节的坑E）。
            artificial_viscosity_enabled: 是否启用问题单元人工粘性（默认 False，
                可选能力）：熵残差判据 `(h/p) max|u_hat.grad s|` 定位数值熵产生
                的单元，对它们叠加全部守恒变量的拉普拉斯 `div(nu grad U)`，
                见 `core/fr_operators/artificial_viscosity/entropy_viscosity.py`。
            artificial_viscosity_alpha: 人工粘性强度标定常数（无量纲，默认
                1.0），`nu = ramp * alpha * |u| * h / p`。
            turbulence_intensity: 来流湍流强度 Tu（默认 0.01 = 1%），用于从
                物理自洽的公式推导 k/omega 初值（工业 RANS 标准做法）。
                外部气动默认 ≤1%，城市道路 3-5%，风洞对标 0.5-2%。
            viscosity_ratio: 来流粘性比 VR = nu_t/nu（默认 5.0），与 Tu 共同
                决定 omega 初值。外部气动推荐 2-10。
            sem_num_eddies: LES/DDES/IDDES 模式下 BD-02 合成湍流入口 (SEM) 的
                涡核数量（默认 200，此前恒为硬编码常量，见
                fr_solver/boundary.py 模块文档 2026-08-28 的修复说明）。
                只在 turb_model_name 为 LES/DDES/IDDES 且未激活 WMLES 时生效。
        """
        resolved_n_threads = configure_numba_threads(n_threads, mesh)

        self._setup_options(mesh, order, artificial_viscosity_enabled, artificial_viscosity_alpha)
        # 安全地获取网格信息
        n_cells = getattr(mesh, 'n_cells', 0)
        n_sps = getattr(mesh, 'n_sps_per_cell', 8)
        self._setup_initial_state(initial_state, n_cells, n_sps, turb_model_name,
                                  rho_inf, vel_inf, p_inf, aoa_deg, aos_deg)
        self._setup_operators(order, time_scheme, low_mach_precond)
        self._setup_physics_and_boundary(
            turb_model_name, mu_molecular, rho_inf, vel_inf, p_inf, aoa_deg, aos_deg,
            turbulence_intensity, viscosity_ratio, sem_num_eddies, bc_overrides)
        self._setup_turbulence_time_and_runtime(
            n_cells, n_sps, order, time_scheme, dual_time_inner_iter, adaptive_cfl,
            cfl_start, cfl_max, cfl_min)

        print("[OK] FRSolver Ready:")
        print(f"   Cells: {n_cells}, Order: P{order}")
        print(f"   Turbulence: {turb_model_name}")
        print(f"   Time Scheme: {time_scheme.value}")
        print(f"   Threads: {resolved_n_threads}")
