"""
AutoFlowCFD V2.0 - FR 求解器主类 (Final Integration)

本模块整合 FRState, HighOrderMesh, FR Kernels, Turbulence Models 及 Weak BCs。
它是 V2.0 求解器的总控中心，负责协调各模块完成 N-S 方程的高阶离散与求解。

核心功能:
1. 支持多种湍流模型（SST/DDES/WMLES/LES）
2. 自动切换RANS/LES模式
3. 完整的时间推进循环
4. 残差监控和收敛判断
"""

import os
import numpy as np
import numba
import logging
from typing import Dict, Any, Optional, Tuple
from autoflowcfd.core.fr_state import FRState, SolverResult
from autoflowcfd.grid.high_order_mesh import HighOrderMesh
from autoflowcfd.fr.operators import generate_fr_operators, FROperators
from autoflowcfd.core.fr_residual_inviscid import compute_inviscid_residual_fr
from autoflowcfd.core.time_integration import TimeIntegrator, TimeIntegrationScheme
from autoflowcfd.core.fr_residual_viscous import compute_viscous_residual as compute_viscous_residual_ldg

# 导入辅助模块
from . import solver_helpers
from . import order_continuation
from . import fr_solver_turbulence
from . import fr_solver_boundary
from . import fr_solver_step

# 配置日志
logger = logging.getLogger(__name__)


class FRSolver:
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
                 turb_model_name: str = "SST", n_vars: int = 5,
                 time_scheme: TimeIntegrationScheme = TimeIntegrationScheme.SSP_RK3,
                 initial_state: Optional[FRState] = None,
                 backend: str = "cpu",
                 rho_inf: float = 1.225, vel_inf: float = 30.0, p_inf: float = 101325.0,
                 bc_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
                 mu_molecular: float = 1.8e-5,
                 dual_time_inner_iter: int = 20,
                 n_threads: int = -1):
        """
        初始化 FRSolver。

        Args:
            mesh: HighOrderMesh 类型的高阶网格对象
            order: 多项式阶数
            turb_model_name: 湍流模型名称 ("SST"/"DDES"/"WMLES"/"LES"/"NONE")
            mu_molecular: 分子动力粘度（默认 1.8e-5 Pa*s，标准状态下空气）。
                此前粘性残差（fr_residual_viscous.py 默认参数）与粘性 CFL
                步长（_compute_local_time_step）各自独立硬编码这个值，没有
                任何受支持的方式一致地设置自定义粘度——两处必须同步改，
                否则 CFL 步长会与真正参与残差组装的粘度脱节（同类问题见
                记忆条目 hardcoded_molecular_viscosity_mismatch）。现在两处
                都从这个构造参数读取。
            n_vars: 守恒变量数量（默认5：rho, rho_u, rho_v, rho_w, rho_e）
            time_scheme: 时间推进方案
            initial_state: 初始状态（用于 Order Continuation）
            backend: 计算后端 ("cpu" 或 "gpu")
            rho_inf, vel_inf, p_inf: 自由来流条件（密度/速度大小[沿+x]/静压），
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
                使用的 CPU 线程数。默认 -1 **不是** `os.cpu_count()`——真实
                545,597 单元生产网格上实测的线程数扩展曲线（P2，16 物理核
                机器）：nt=1 87.63s、nt=2 61.25s、nt=4 57.35s（峰值）、
                nt=8 61.46s、nt=16 67.19s，超过 4 线程后不是收益递减而是
                净倒退（16 线程比 4 线程还慢），根因见下方与 kernel 模块
                文档"多核并行的扩展性上限"一节。因此 -1 解析成 **4**（本机
                实测的甜点，不是理论值），不是自动检测到的物理核数——这是
                目前唯一有真实网格实测数据支撑的默认值；调用方仍可显式传
                更大或更小的 n_threads 覆盖（CLI 见 `--threads`/`-j`），但
                默认不应该让用户在毫无预警的情况下用一个实测更差的核数。
                只在这里调用一次 `numba.set_num_threads`——两个界面
                kernel 的 `n_threads` 参数要求调用方紧邻调用前取
                `numba.get_num_threads()`，如果这个全局状态在其他地方
                被并发修改，会破坏该约束（见两个 kernel 模块文档"多核
                并行"一节的坑E）。
        """
        # numba 全局线程数只在这里设置一次（求解器生命周期内不再修改），
        # 理由见本方法 n_threads 参数文档。必须在任何残差 kernel 被调用
        # 之前设置。-1 解析成 4（本机真实网格实测的扩展性甜点），不是
        # os.cpu_count()——见 n_threads 参数文档，超过 4 线程实测是净
        # 倒退，盲目用满全部核数在这个 kernel 的当前实现下是有害默认值。
        _DEFAULT_N_THREADS = 4
        resolved_n_threads = n_threads if n_threads > 0 else _DEFAULT_N_THREADS
        numba.set_num_threads(resolved_n_threads)

        # 防御性内存检查：两个界面 kernel 各自的私有累加缓冲区峰值约
        # n_threads * n_cells * n_sps * 5 vars * 8 bytes（无粘/粘性两次
        # 调用不会同时存活，见 fr_residual_inviscid_kernel.py 模块文档
        # "多核并行"一节），超过系统总内存一半就提醒用户，不静默跑到
        # OOM。取不到总内存（非 Windows 平台没有对应 ctypes 调用）时
        # 直接跳过，不影响求解——这只是个提醒，不是硬性门禁。
        n_cells_est = getattr(mesh, 'n_cells', 0)
        n_sps_est = getattr(mesh, 'n_sps_per_cell', 8)
        buf_bytes = resolved_n_threads * n_cells_est * n_sps_est * 5 * 8
        try:
            import ctypes

            class _MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            total_mem = stat.ullTotalPhys
            if total_mem > 0 and buf_bytes > 0.5 * total_mem:
                print(
                    f"⚠️  警告：n_threads={resolved_n_threads} 下界面 kernel 私有累加缓冲区峰值约 "
                    f"{buf_bytes / 1e9:.1f}GB，超过系统总内存（{total_mem / 1e9:.1f}GB）的一半，"
                    f"叠加其他计算环节的内存占用可能导致 OOM。建议用更小的 n_threads。"
                )
        except Exception:
            pass

        self.mesh = mesh
        self.order = order
        self.backend_type = backend.lower()
        
        # 安全地获取网格信息
        n_cells = getattr(mesh, 'n_cells', 0)
        n_sps = getattr(mesh, 'n_sps_per_cell', 8)
        
        # 1. 初始化状态 (S-01)
        if initial_state is not None:
            # 使用提供的初始状态（Order Continuation）
            self.state = initial_state
            print(f"✅ Using provided initial state from lower order")
        else:
            # 根据湍流模型确定变量数。必须先 upper()：self.turb_model_name
            # 要到第 5 步才被规范化为大写，这里若直接用构造参数原始大小写
            # 比较，会导致 `turbulence-model sst`（小写，steady CLI 路径）
            # 与 `SST`（大写，transient CLI 路径）判出不同的 n_vars——已用
            # 两条 CLI 路径实测复现（steady 得到 n_vars=5，transient 得到 7）。
            default_n_vars = 7 if turb_model_name.upper() in ["SST", "DDES"] else 5
            self.state = FRState(n_cells, n_sps, default_n_vars)
            # 初场必须与自由来流边界条件一致（都用同一套 rho_inf/vel_inf/p_inf），
            # 否则显式伪时间推进第一步就要吸收一个几个数量级的压力跳跃
            # （旧版本硬编码 rho=1,p=1 的"单位"初场，与真实边界条件的
            # rho_inf~1.2, p_inf~1e5 相差 5 个数量级，显式格式在这种冲击下
            # 数值发散——这不是残差组装的 bug，是初始条件与边界条件不一致
            # 导致的可预见的数值不稳定，工业代码从来不会这样初始化）。
            self.state.initialize_uniform(rho=rho_inf, u=vel_inf, v=0.0, w=0.0, p=p_inf)
        
        # 2. 预计算算子 (G-04)
        self.ops = generate_fr_operators(order)
        
        # 3. 初始化边界条件 (BD-01) —— 真正参与残差组装的幽灵态边界条件
        # （不再持有未被使用的 FRWeakBC 罚项处理器实例——那是旧版本从未被
        # 求解主循环调用过的死代码路径，真正生效的是下面的
        # boundary_ghost_provider，见 boundary/fr_ghost_state.py）
        #
        # self.turb_model_name 必须先于 boundary_ghost_provider 构造
        # （BD-02：LES/DDES 模式下要给 VELOCITY_INLET 组接入合成湍流
        # 入口，需要在构造 ghost provider 时就知道湍流模型），其余湍流
        # 模型对象（turb_model/wmles_model/sgs_model）留到下面第 5 步
        # 再真正初始化——ghost provider 构造时只用 getattr(...,None) 安全
        # 读取 wmles_model，不依赖它已经存在。
        self.turb_model_name = turb_model_name.upper()
        self.freestream = {"rho_inf": rho_inf, "vel_inf": vel_inf, "p_inf": p_inf}
        self.boundary_ghost_provider = self._build_boundary_ghost_provider(bc_overrides or {})
        self.mu_molecular = mu_molecular

        # 4. 初始化计算后端 (B-01)——见 solver_helpers.py::resolve_backend_type 文档。
        self.backend = None
        self.backend_type = solver_helpers.resolve_backend_type(self.backend_type)

        # 5. 初始化湍流模型（self.turb_model_name 已在第 3 步设置）
        self.turb_model = None
        self.ddes_model = None
        self.wmles_model = None
        self.sgs_model = None
        
        self._init_turbulence_models(n_cells, n_sps)
        
        # 6. 初始化时间积分器 (S-05)
        self.time_integrator = TimeIntegrator(scheme=time_scheme, dual_time_steps=dual_time_inner_iter)

        # DUAL_TIME 专用：物理时间层 n-1 的解（BDF2 时间导数项需要），
        # None 表示还没有跑过物理步（下一步会退化为 BDF1），见 step()
        # 与 order_continuation.interpolate_to_new_order_checked（阶数
        # 变化后 SPs 布局改变，必须让这份历史失效，否则形状不匹配/物理
        # 上不连续的历史层会被静默用于 BDF2）。
        self._dual_time_U_prev: Optional[np.ndarray] = None
        
        # 7. 壁面距离场（用于DDES/WMLES）
        self.wall_distance = None

        # 8. Order Continuation 状态
        self.current_order = order
        self.order_continuation_enabled = True
        
        print(f"✅ FRSolver Ready:")
        print(f"   Cells: {n_cells}, Order: P{order}")
        print(f"   Turbulence: {turb_model_name}")
        print(f"   Backend: {self.backend_type.upper()}")
        print(f"   Time Scheme: {time_scheme.value}")
        print(f"   Threads: {resolved_n_threads}")

    def _init_turbulence_models(self, n_cells: int, n_sps: int):
        """初始化湍流模型（委托给 fr_solver_turbulence）。"""
        fr_solver_turbulence.init_turbulence_models(self, n_cells, n_sps)
    
    def _build_boundary_ghost_provider(self, bc_overrides: Dict[str, Dict[str, Any]]):
        """构建边界幽灵态提供者 (BD-01)，委托给 fr_solver_boundary。"""
        return fr_solver_boundary.build_boundary_ghost_provider(self, bc_overrides)

    def compute_wall_distance_field(self, mesh_nodes: np.ndarray,
                                   wall_indices: np.ndarray,
                                   connectivity: Optional[np.ndarray] = None,
                                   use_eikonal: bool = False):
        """计算壁面距离场（用于 DDES/WMLES/SST），委托给 fr_solver_turbulence。

        Args:
            mesh_nodes: 全部网格节点坐标
            wall_indices: WALL 边界节点索引
            connectivity: 节点邻接表，use_eikonal=True 时必须提供 - 见
                fr_solver_turbulence.compute_wall_distance_field 自己的文档
            use_eikonal: 是否用 Eikonal 方程（而不是纯欧氏 KD-Tree）求解
        """
        fr_solver_turbulence.compute_wall_distance_field(
            self, mesh_nodes, wall_indices, connectivity=connectivity, use_eikonal=use_eikonal
        )

    def solve(self, max_iter: int = 1000, dt: float = 1e-4, tol: float = 1e-6) -> SolverResult:
        """
        执行稳态/瞬态求解循环。
        
        Args:
            max_iter: 最大迭代次数
            dt: 时间步长
            tol: 收敛容差
            
        Returns:
            SolverResult: 包含收敛状态、最终残差和迭代次数的结果对象
        """
        logger_msg = f"Starting solve loop with {self.time_integrator.scheme.value}"
        if self.turb_model_name != "NONE":
            logger_msg += f", turbulence={self.turb_model_name}"
        print(logger_msg)
        
        # Order Continuation: 从低阶开始逐步提升精度
        if self.order_continuation_enabled and self.order >= 2:
            return self._solve_with_order_continuation(max_iter, dt, tol)
        
        import time
        converged = False
        final_residual = 1e10
        
        for i in range(max_iter):
            t_start = time.time()
            res = self.step(dt)
            t_end = time.time()
            final_residual = res
            
            # 每 10 步或第 1 步打印详细信息
            if i == 0 or (i + 1) % 10 == 0:
                print(f"Iteration {i+1}: Residual = {res:.6e} | Time/step: {t_end - t_start:.2f}s")
                
            if res < tol:
                converged = True
                print(f"✅ Converged at iteration {i+1} with residual {res:.6e}")
                break
        
        return SolverResult(converged=converged, iterations=i+1, final_residual=final_residual)
    
    def _solve_with_order_continuation(self, max_iter: int, dt: float, tol: float) -> SolverResult:
        """实现 Order Continuation 策略：从P0逐步提升到目标阶数（委托给 order_continuation）。"""
        return order_continuation.run_order_continuation(self, max_iter, dt, tol)

    def _interpolate_to_new_order(self, new_order: int):
        """将解从当前阶数插值到新的阶数（委托给 order_continuation）。"""
        order_continuation.interpolate_to_new_order_checked(self, new_order)

    def step(self, dt: float) -> float:
        """执行一个时间步长 (S-05)。见 fr_solver_step.py::step 文档。"""
        return fr_solver_step.step(self, dt)

    def compute_turbulence_source(self, dt: float) -> Optional[tuple]:
        """计算湍流模型源项（委托给 fr_solver_turbulence）。"""
        return fr_solver_turbulence.compute_turbulence_source(self, dt)

    def apply_turbulence_corrections(self):
        """应用湍流模型的修正（SGS 涡粘系数），委托给 fr_solver_turbulence。

        WMLES 壁面剪应力**不**在这里施加——它是一个真正的残差贡献项，
        必须在时间积分*之前*参与残差组装才能生效，见
        compute_viscous_residual() 里的调用与该方法文档（T-05 修复：
        此前在这里调用，而这里在 step() 中排在状态更新*之后*，对本步
        毫无影响，架构上不可能生效）。
        """
        fr_solver_turbulence.apply_turbulence_corrections(self)

    def compute_inviscid_residual(self):
        """
        计算无粘残差 (S-02/S-04)。

        真实的曲边/坍缩坐标 FR 离散：体积项用逆变通量 (contravariant flux)
        散度实现（度量项一致，满足自由流场保持性/离散GCL），界面项用基于
        真实单元-面连接关系的 AUSM+up 黎曼求解 + Radau/VCJH 校正函数投影，
        边界面通过 boundary_ghost_provider (BD-01) 构造物理正确的幽灵态。

        取代旧版本"用全场平均态+硬编码法向量冒充相邻单元"的伪校正项——
        详见 core/fr_residual_inviscid.py 模块文档与
        tests/unit/test_fr_residual_inviscid.py 的自由流场保持性验证。
        """
        self.state._update_primitives()

        if not self.state.U.flags['C_CONTIGUOUS']:
            self.state.U = np.ascontiguousarray(self.state.U)

        # GPU 分发 (B-01)：请求 GPU 后端时走 CuPy 加速路径。
        # P0（阶数延续热身阶段）使用 CuPy RawKernel（core/gpu/gpu_p0_inviscid.py）；
        # P>=1 高阶 FR 使用 CuPy 向量化实现（core/gpu/gpu_inviscid.py）。
        # GPU 不可用时自动回退 CPU。
        if self.backend_type == "gpu":
            if self.mesh.n_points_1d == 1:
                from .gpu.gpu_p0_inviscid import compute_inviscid_residual_p0_cupy
                res_euler = compute_inviscid_residual_p0_cupy(
                    self.state.U, self.mesh,
                    boundary_ghost_provider=self.boundary_ghost_provider,
                )
            else:
                # P>=1 高阶 FR GPU 路径
                from .gpu.gpu_inviscid import compute_inviscid_residual_fr_gpu
                res_euler = compute_inviscid_residual_fr_gpu(
                    self.state.U, self.mesh, self.ops,
                    boundary_ghost_provider=self.boundary_ghost_provider,
                )
        else:
            res_euler = compute_inviscid_residual_fr(
                self.state.U, self.mesh, self.ops,
                boundary_ghost_provider=self.boundary_ghost_provider,
            )

        if self.state.n_vars > 5:
            # 湍流量 (k, omega) 的对流输运项当前仍由 compute_turbulence_source
            # 单独处理（局部源项积分，不含对流通量），此处只补零占位维度，
            # 不在这里静默引入未经验证的湍流对流项。
            res_full = np.zeros((res_euler.shape[0], res_euler.shape[1], self.state.n_vars))
            res_full[:, :, :5] = res_euler
            return res_full
        return res_euler

    def compute_viscous_residual(self):
        """
        计算粘性残差 (S-03)。

        真实的 BR1 面耦合粘性离散（core/fr_viscous_flux.py），并把湍流模型
        算出的涡粘系数真正耦合进应力张量/热传导（T-01/T-04/T-06 修复：
        此前调用处从不传湍流粘度，粘性通量永远只用分子粘度 1.8e-5，
        SST/DDES/WALE 算出的 nu_t 场只在自身模型内部自用，从未进入
        动量/能量方程的扩散项）。

        Returns:
            viscous_res: 粘性残差（WMLES 激活时已叠加壁面剪应力修正，
                见 solver_helpers.compute_wmles_wall_stress_correction
                文档 T-05 修复说明——必须在这里（残差组装、时间积分之前）
                施加才能真正影响本步的解，而不是像此前那样在状态更新
                之后才计算）
        """
        mu_t_field = self._get_turbulent_viscosity_field()
        res = compute_viscous_residual_ldg(
            self.state.U, self.state.Q, self.ops, self.mesh,
            mu=self.mu_molecular,
            mu_t_field=mu_t_field,
            boundary_ghost_provider=self.boundary_ghost_provider,
        )

        if self.wmles_model is not None:
            wall_stress_correction = solver_helpers.compute_wmles_wall_stress_correction(self)
            if wall_stress_correction is not None:
                res = res + wall_stress_correction[..., : res.shape[-1]]

        return res

    def _get_turbulent_viscosity_field(self) -> Optional[np.ndarray]:
        """汇总当前激活的湍流模型给出的动力涡粘度场 mu_t = rho * nu_t（委托给 fr_solver_turbulence）。"""
        return fr_solver_turbulence.get_turbulent_viscosity_field(self)

    def _compute_gradients(self) -> np.ndarray:
        """
        计算守恒变量的梯度。

        Returns:
            grad_U: 梯度，形状 (n_cells, n_sps, n_vars, 3)
        """
        from autoflowcfd.core.fr_residual_viscous import compute_gradients
        return compute_gradients(self.state.U, self.ops, self.mesh)
    
    def _get_metric_flux_scale(self) -> np.ndarray:
        """逐 SP 的度量"通量面积"标度 sum_m ||adj(J)[:,m,:]||，只依赖网格
        几何（与流场状态无关），缓存后避免每个时间步重复计算——供
        _compute_local_time_step 的几何/度量 CFL 限制使用，见该方法文档。
        """
        cached = getattr(self, "_metric_flux_scale_cache", None)
        if cached is not None and cached.shape[0] == self.state.U.shape[0]:
            return cached
        det_jacs = self.mesh.jacobians["det_jacs"].reshape(self.mesh.n_cells, self.mesh.n_sps_per_cell)
        inv_jacs = self.mesh.jacobians["inv_jacs"].reshape(self.mesh.n_cells, self.mesh.n_sps_per_cell, 3, 3)
        adj_j = det_jacs[..., None, None] * inv_jacs  # (n_cells,n_sps,3,3), adj_j[...,m,i]
        adj_row_norms = np.linalg.norm(adj_j, axis=-1)  # (n_cells,n_sps,3): 每个参考方向 m 的 |adj(J)[:,m,:]|
        metric_flux_scale = np.sum(adj_row_norms, axis=-1)  # (n_cells,n_sps)
        self._metric_flux_scale_cache = metric_flux_scale
        return metric_flux_scale

    def _compute_local_time_step(self) -> np.ndarray:
        """计算局部时间步长（基于CFL条件）。实现见
        fr_solver_cfl.py::compute_local_time_step（从本文件拆出，控制
        单文件行数），文档字符串也在那里。"""
        from .fr_solver_cfl import compute_local_time_step

        return compute_local_time_step(self)

    def _get_cell_volumes(self) -> np.ndarray:
        """
        获取单元体积（精确求积，见 HighOrderMesh.get_all_cell_volumes）。

        Returns:
            volumes: 单元体积，形状 (n_cells,)
        """
        return self.mesh.get_all_cell_volumes()

    def _get_grid_scale(self) -> np.ndarray:
        """
        获取网格尺度（用于LES/SGS模型）。

        Returns:
            delta: 网格尺度，形状 (n_cells, n_sps)
        """
        n_cells, n_sps = self.state.U.shape[:2]

        volumes = self.mesh.get_all_cell_volumes()
        delta = np.power(np.abs(volumes), 1.0 / 3.0)
        return np.tile(delta[:, np.newaxis], (1, n_sps))
