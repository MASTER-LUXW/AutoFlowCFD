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

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
from autoflowcfd.core.fr_state import FRState, SolverResult
from autoflowcfd.grid.high_order_mesh import HighOrderMesh
from autoflowcfd.fr.operators import generate_fr_operators, FROperators
from autoflowcfd.core.fr_residual_inviscid import compute_inviscid_residual_fr
from autoflowcfd.core.time_integration import TimeIntegrator, TimeIntegrationScheme
from autoflowcfd.core.fr_residual_viscous import compute_viscous_residual as compute_viscous_residual_ldg
from autoflowcfd.core.fr_solver_filter import build_filter_func

# 导入辅助模块
from . import solver_helpers
from . import order_continuation
from . import fr_solver_turbulence
from . import fr_solver_boundary

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
                 bc_overrides: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        初始化 FRSolver。

        Args:
            mesh: HighOrderMesh 类型的高阶网格对象
            order: 多项式阶数
            turb_model_name: 湍流模型名称 ("SST"/"DDES"/"WMLES"/"LES"/"NONE")
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
        """
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
            # 根据湍流模型确定变量数
            default_n_vars = 7 if turb_model_name in ["SST", "DDES"] else 5
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
        self.freestream = {"rho_inf": rho_inf, "vel_inf": vel_inf, "p_inf": p_inf}
        self.boundary_ghost_provider = self._build_boundary_ghost_provider(bc_overrides or {})

        # 4. 初始化计算后端 (B-01)
        #
        # 此前这里构造一个 CUDABackend 实例、调用一次 .initialize()，
        # 之后 self.backend 在全文件里再也不会被引用——backend="gpu"
        # 对实际计算路径没有任何影响，只改变构造时打印哪几行日志（V2.0
        # 专家评审报告 B-01 项指出的问题）。真正的 GPU 加速路径见
        # core/backend/fr_gpu_p0.py：目前只对 P0（Order Continuation
        # 最低阶/有限体积）无粘残差实现了真实 CUDA kernel（忠实移植
        # 已验证正确的 CPU 版 _compute_inviscid_residual_fv_p0，见该模块
        # 文档），P>=1（真正的高阶 FR，坍缩坐标度量张量外插 + 逐面记录
        # 字典键控分发）尚未实现 GPU 版本，如实回退 CPU、如实记录日志，
        # 不再静默构造一个从未被使用的对象。是否真正走 GPU 由
        # compute_inviscid_residual() 在每次调用时按 self.backend_type
        # 与当前网格阶数共同判断（阶数延续期间会在多个阶数之间切换）。
        self.backend = None
        if self.backend_type == "gpu":
            from .backend.fr_gpu_p0 import gpu_p0_available
            if gpu_p0_available():
                logger.info(
                    "GPU (CUDA) backend available - will accelerate the P0 finite-volume inviscid "
                    "residual only (order continuation warm-up stage); P>=1 orders still run on CPU "
                    "(see core/backend/fr_gpu_p0.py for scope)."
                )
            else:
                logger.warning(
                    "GPU backend requested but no CUDA device found (and NUMBA_ENABLE_CUDASIM not set) "
                    "- falling back to CPU entirely"
                )
                self.backend_type = "cpu"

        if self.backend_type == "cpu":
            logger.info(f"CPU Backend (Numba) initialized")
        
        # 5. 初始化湍流模型
        self.turb_model_name = turb_model_name.upper()
        self.turb_model = None
        self.ddes_model = None
        self.wmles_model = None
        self.sgs_model = None
        
        self._init_turbulence_models(n_cells, n_sps)
        
        # 6. 初始化时间积分器 (S-05)
        self.time_integrator = TimeIntegrator(scheme=time_scheme)

        # DUAL_TIME 专用：物理时间层 n-1 的解（BDF2 时间导数项需要），
        # None 表示还没有跑过物理步（下一步会退化为 BDF1），见 step()
        # 与 order_continuation.interpolate_to_new_order_checked（阶数
        # 变化后 SPs 布局改变，必须让这份历史失效，否则形状不匹配/物理
        # 上不连续的历史层会被静默用于 BDF2）。
        self._dual_time_U_prev: Optional[np.ndarray] = None
        
        # 7. 壁面距离场（用于DDES/WMLES）
        self.wall_distance = None
        
        # 8. 壁面边界信息（用于WMLES）
        self._wall_sp_info = None
        
        # 9. Order Continuation 状态
        self.current_order = order
        self.order_continuation_enabled = True
        
        print(f"✅ FRSolver Ready:")
        print(f"   Cells: {n_cells}, Order: P{order}")
        print(f"   Turbulence: {turb_model_name}")
        print(f"   Backend: {self.backend_type.upper()}")
        print(f"   Time Scheme: {time_scheme.value}")

    def _init_turbulence_models(self, n_cells: int, n_sps: int):
        """初始化湍流模型（委托给 fr_solver_turbulence）。"""
        fr_solver_turbulence.init_turbulence_models(self, n_cells, n_sps)
    
    def _build_boundary_ghost_provider(self, bc_overrides: Dict[str, Dict[str, Any]]):
        """构建边界幽灵态提供者 (BD-01)，委托给 fr_solver_boundary。"""
        return fr_solver_boundary.build_boundary_ghost_provider(self, bc_overrides)

    def compute_wall_distance_field(self, mesh_nodes: np.ndarray,
                                   wall_indices: np.ndarray):
        """计算壁面距离场（用于 DDES/WMLES/SST），委托给 fr_solver_turbulence。"""
        fr_solver_turbulence.compute_wall_distance_field(self, mesh_nodes, wall_indices)

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
        """
        执行一个时间步长 (S-05)。

        平均流（5个欧拉变量）真正通过 self.time_integrator 推进
        （SSP-RK2/RK3/IMEX/Dual-Time，由构造时的 time_scheme 决定），
        取代旧版本里恒定不变的单级前向欧拉——此前不管 CLI 传
        --time-method rk3/imex/dual-time 哪一个，step() 内部都硬编码执行
        `U = U + dt_local*residual`，`self.time_integrator` 被构造出来后
        从未被调用过。

        湍流量 (k,omega) 的输运方程仍用独立的单步显式更新（算子分裂：
        平均流走高阶 RK 子迭代，湍流方程走更简单、专门做过刚性限制
        的更新，是工业 RANS/DES 求解器常见做法，避免把湍流源项的强
        非线性刚性直接卷入平均流的多级残差重新求值）。

        dt 参数的语义按 time_scheme 分两种情况（此前不管哪种 scheme，
        dt 参数都被完全忽略，实际步长恒由 self._compute_local_time_step()
        的逐单元局部 CFL 步长决定——这对稳态 RANS 收敛加速是对的，但
        意味着 CLI `solve transient`（DES/LES）传入的 `--dt`/`--physical-time`
        从未真正生效，瞬态仿真没有时间精度，这是本次修复的问题）：
        - SSP-RK2/RK3/IMEX（稳态收敛加速模式）：dt 参数确实被忽略，
          步长仍由局部 CFL 决定——用于收敛到定常解，不要求时间精度，
          局部时间步是标准且正确的加速手段。
        - DUAL_TIME（DES/LES 等真正非稳态仿真应使用的模式）：dt 现在
          是真正的物理时间步长，通过 BDF1/BDF2 时间导数项耦合进伪残差
          （见 TimeIntegrator.step_dual_time），伪时间迭代收敛后得到的
          解在物理时间上精确前进了 dt；局部 CFL 步长只用作内层伪时间
          迭代的加速手段，不影响物理时间精度。

        Args:
            dt: 见上——SSP-RK/IMEX 模式下被忽略，DUAL_TIME 模式下是真正
                生效的物理时间步长

        Returns:
            residual_norm: 残差范数
        """
        try:
            self.state._update_primitives()

            # 湍流源项在当前状态下求值一次（沿用旧有的单步显式-半隐式
            # 阻尼更新，见 turbulence_sst.py::update_fields）
            turb_source = self.compute_turbulence_source(dt)

            n_cells, n_sps, n_vars = self.state.U.shape
            dt_local = self._compute_local_time_step()  # (n_cells, n_sps)

            U_flat = self.state.U.reshape(n_cells * n_sps, n_vars)
            dt_local_flat = dt_local.reshape(n_cells * n_sps)

            def mean_flow_residual(U_flat_trial: np.ndarray) -> np.ndarray:
                """TimeIntegrator 约定：dU/dt = -residual_func(U)。"""
                U_trial = U_flat_trial.reshape(n_cells, n_sps, n_vars)
                saved_U = self.state.U
                self.state.U = U_trial
                try:
                    inv_res = self.compute_inviscid_residual()
                    visc_res = self.compute_viscous_residual()
                finally:
                    self.state.U = saved_U
                total = inv_res + visc_res  # 已是 dU/dt，形状 (n_cells,n_sps,n_vars)
                return -total.reshape(n_cells * n_sps, n_vars)

            # residual0 是 TimeIntegrator 自身的 R(U) 约定（dU/dt=-R），
            # 复用它既避免重复计算 Stage 0 残差，也用来更新
            # self.state.dU_dt——收敛监控 (get_residual_norm) 依赖这个量，
            # 重构 step() 时若遗漏这一步，会让残差历史恒为 0（表面上"已收敛"，
            # 实际只是从未被更新过），已用非均匀扰动初场验证发现并修复。
            residual0 = mean_flow_residual(U_flat)
            self.state.dU_dt = (-residual0).reshape(n_cells, n_sps, n_vars)

            # 模态滤波回调（S-05 补充修复）：见 fr_solver_filter.py 文档——
            # 必须传给 TimeIntegrator，由它在*每个* RK stage 的正定性投影
            # 之后立即施加，抑制坍缩坐标节点配置法固有的混叠噪声放大；
            # 只在最终组合结果上滤波一次不够，真实复现噪声在中间 stage
            # 就已放大到 NaN。
            filter_func = build_filter_func(self)

            if self.time_integrator.scheme == TimeIntegrationScheme.DUAL_TIME:
                # 真正时间精度的物理时间推进：dt 是物理时间步长（不再被
                # 忽略），dt_local 只用作内层伪时间迭代的局部加速步长，
                # 两者不能混用——见 TimeIntegrator.step_dual_time 文档。
                U_new_flat = self.time_integrator.step_dual_time(
                    U_flat,
                    mean_flow_residual,
                    dt_local_flat,
                    dt_physical=dt,
                    solution_prev=self._dual_time_U_prev,
                    max_inner_iter=self.time_integrator.dual_time_steps,
                    filter_func=filter_func,
                )
                self._dual_time_U_prev = U_flat.copy()
            else:
                U_new_flat = self.time_integrator.step(
                    U_flat, mean_flow_residual, dt_local_flat, p_floor=1.0, residual0=residual0,
                    filter_func=filter_func,
                )
            self.state.U = U_new_flat.reshape(n_cells, n_sps, n_vars)

            # 湍流量单步显式更新（与平均流的 RK 子迭代解耦）
            if turb_source is not None and n_vars > 5:
                self.state.U[:, :, 5] += dt_local * turb_source[0]
                self.state.U[:, :, 6] += dt_local * turb_source[1]
                self.state.U[:, :, 5] = np.maximum(self.state.U[:, :, 5], 0.0)
                self.state.U[:, :, 6] = np.maximum(self.state.U[:, :, 6], 1e-8)

            self.apply_turbulence_corrections()
            self.state._update_primitives()

            residual_norm = self.state.get_residual_norm()
            return residual_norm

        except Exception as e:
            logger.error(f"Step failed with error: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def compute_turbulence_source(self, dt: float) -> Optional[tuple]:
        """计算湍流模型源项（委托给 fr_solver_turbulence）。"""
        return fr_solver_turbulence.compute_turbulence_source(self, dt)


    def apply_turbulence_corrections(self):
        """应用湍流模型的修正（WMLES 壁面应力、SGS 涡粘系数），委托给 fr_solver_turbulence。"""
        fr_solver_turbulence.apply_turbulence_corrections(self)

    def _apply_wmles_wall_stress(self):
        """
        应用WMLES壁面剪应力到动量方程残差（委托给 solver_helpers）。
        """
        solver_helpers.apply_wmles_wall_stress(self)

    def _extract_wall_boundary_info(self):
        """
        从网格或边界管理器中提取壁面边界信息（委托给 solver_helpers）。
        """
        self._wall_sp_info = solver_helpers.extract_wall_boundary_info(self)
    
    def _auto_detect_wall_boundaries(self):
        """
        基于几何特征自动检测壁面边界（委托给 solver_helpers）。
        """
        self._wall_sp_info = solver_helpers.auto_detect_wall_boundaries(self)
    
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

        # GPU 分发 (B-01)：只有当前网格是 P0（阶数延续热身阶段，
        # mesh.n_points_1d==1）且请求了 GPU 后端时才真正走 GPU kernel
        # （core/backend/fr_gpu_p0.py）；P>=1 无论 backend_type 是什么都
        # 走 CPU 的 compute_inviscid_residual_fr（其内部对 P0 也有一条
        # CPU 参考实现分支，两者数值上已用 tests/unit/test_fr_gpu_p0.py
        # 交叉验证一致，这里选哪一条纯粹是性能路径的选择，不影响物理）。
        if self.backend_type == "gpu" and self.mesh.n_points_1d == 1:
            from .backend.fr_gpu_p0 import compute_inviscid_residual_p0_gpu
            res_euler = compute_inviscid_residual_p0_gpu(
                self.state.U, self.mesh,
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
            viscous_res: 粘性残差
        """
        mu_t_field = self._get_turbulent_viscosity_field()
        return compute_viscous_residual_ldg(
            self.state.U, self.state.Q, self.ops, self.mesh,
            mu_t_field=mu_t_field,
        )

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
        """
        计算局部时间步长（基于CFL条件）。

        真正的稳定性限制取三个独立机制中更严格的一个：
        0. 【已撤销】低马赫数预处理——2026-08-14 Couette 合成算例定量验证
           过程中真实复现并确认：这里曾经引入的 Weiss-Smith 预处理
           （`preconditioned_acoustic_eigs`）只用来放松 CFL *步长估计*，
           但实际参与残差计算的 AUSM+up 通量（core/fr_kernels.py::
           compute_ausm_up_flux）自身完全没有做任何 Weiss-Smith 预处理——
           它内部用的始终是*真实物理*声速 aL/aR（只有 Liou 2001 式的
           界面声速插值修正，调整耗散强度，不改变特征波速本身）。这两者
           不一致：CFL 步长按"预处理后、人为缩小的"波速估计出一个偏大
           的 dt，但真正被显式积分的却是未预处理、用真实声速主导刚性的
           AUSM+up 通量——真实复现（棱柱/四面体网格均可复现）：自由参考
           马赫数 mach_ref 越小（越贴近 Couette/Poiseuille 这类低速层流
           算例的真实工况），这个 dt 相对真实稳定性极限就越大，扫描
           参考速度 1~30 m/s 精确复现了这个失稳阈值（<~15 m/s 对应
           M<~0.044 必然在数步内 NaN，>=20 m/s 稳定）——不是"要更保守
           CFL"就能绕开的问题，是步长估计与实际被积分的物理不一致这一
           结构性缺陷。真正一致的做法需要连 AUSM+up 通量本身也做
           Weiss-Smith 预处理（改动数值通量本身，属于更大的算法工作，
           已记录待后续评估），在此之前 CFL 步长必须如实按*真实*声速
           估计，不能假装用了一套实际并未生效的预处理来"合法"放宽步长。
           wave_speed 现在恒为真实的 |u|+a（未预处理），与 AUSM+up 通量
           实际使用的特征波速一致。
        1. 对流 CFL（原有逻辑）：dt = CFL * h / wave_speed，h 用单元的
           精确求积体积——这是标准有限体积式估计，按"单元平均"尺度衡量。
        2. 粘性稳定性限制（新增，同样是修复真实存在的失稳）：显式格式
           对粘性（分子+湍流）扩散项的稳定性时间步长是 dt<=C*rho*V^(2/3)
           /mu_eff（抛物型稳定性条件），与上面的对流限制是完全独立的
           机制——粘性主导流动（低速层流、边界层内部）下这个限制可能
           严格得多，此前完全没有被施加过，真实复现：Couette 层流验证
           算例里这正是导致发散的根本原因之一（另一个是上面 0 提到的
           低马赫数刚性）。公式与 TimeIntegrator.local_time_step 一致。
        3. 几何/度量 CFL（此前已修复的失稳）：坍缩坐标下同一个
           四面体/棱柱单元内，不同 SP 的 det(J) 天然可以相差几百倍——
           已用完美正四面体数值验证，这是 Duffy 坍缩变换在 P=2 时的
           固有性质，与单元形状/网格质量无关，不是可以"修好"的缺陷。
           无粘残差公式 residual = -div_comp/det(J) 对*非均匀*流场（自由
           流场因离散GCL恒等式精确抵消是例外）在 det(J) 很小的 SP 处，
           把一个本身有界的参考空间通量散度 div_comp（真实网格实测量级
           ~0.01~0.3，不随 det(J) 一起等比例缩小——这是把 P 阶多项式
           微分矩阵套在"度量项(有理)×非常数流场"这个不再是低阶多项式的
           乘积上的固有混叠截断误差）放大到失稳量级——真实网格上单元
           509974/525292 等（det(J) 低至 ~2e-14）在仅 1% 幅度的温和非
           均匀扰动下，无粘残差被放大到 1e10~1e11 量级，用原有"单元
           平均体积"CFL 算出的步长完全无法感知、更谈不上限制这种
           SP 级别的刚性，几步之内必然发散为 NaN——已数值复现验证。
           标准有限体积 CFL 公式 dt=CFL*V/Σ(A_f*(|u·n|+a)) 在这里的
           直接类比：用该 SP 自己的 det(J) 当作局部"体积"，
           sum_m ||adj(J)[SP,m,:]|| 当作局部"总通量面积"。

        Returns:
            dt_local: 局部时间步长，形状 (n_cells, n_sps)
        """
        n_cells, n_sps, n_vars = self.state.U.shape

        # 提取速度和声速
        rho = self.state.Q[:, :, 0]
        u = self.state.Q[:, :, 1] / np.maximum(rho, 1e-10)
        v = self.state.Q[:, :, 2] / np.maximum(rho, 1e-10)
        w = self.state.Q[:, :, 3] / np.maximum(rho, 1e-10)
        p = (1.4 - 1.0) * (self.state.Q[:, :, 4] - 0.5 * rho * (u**2 + v**2 + w**2))
        a = np.sqrt(np.maximum(1.4 * p / np.maximum(rho, 1e-10), 1e-10))

        vel_mag = np.sqrt(u**2 + v**2 + w**2)

        # 真实（未预处理）声学波速（见上方文档 0）：必须与 AUSM+up 通量
        # 实际使用的特征波速一致——那里从未做过 Weiss-Smith 预处理，CFL
        # 步长估计也不能假装做了。
        wave_speed = np.maximum(vel_mag + a, 1e-10)

        # 网格尺度：用 HighOrderMesh 的精确求积体积（不是"det(J)均值*8"近似），
        # Order Continuation 期间当前状态 n_sps 可能与网格 n_sps 不同，
        # 体积是逐单元量不受此影响，直接广播到当前 n_sps 即可。
        volumes = self.mesh.get_all_cell_volumes()
        h = np.power(np.abs(volumes), 1.0 / 3.0)
        h_expanded = np.tile(h[:, np.newaxis], (1, n_sps))

        CFL = 0.1  # 保守的CFL数
        dt_advective = CFL * h_expanded / wave_speed

        # 粘性稳定性限制（见上方文档 2）：分子粘度 + 当前湍流模型给出的
        # 涡粘（若有），与 TimeIntegrator.local_time_step 用同一公式
        # dt_visc = 0.25*CFL*rho*V^(2/3)/mu_eff。
        mu_t_field = self._get_turbulent_viscosity_field()  # None 或 (n_cells,n_sps)/(n_cells,mesh_n_sps)
        mu_molecular = 1.8e-5
        if mu_t_field is not None:
            if mu_t_field.shape[1] != n_sps:
                rep = int(np.ceil(n_sps / mu_t_field.shape[1]))
                mu_t_field = np.tile(mu_t_field, (1, rep))[:, :n_sps]
            mu_eff = mu_molecular + mu_t_field
        else:
            mu_eff = np.full_like(rho, mu_molecular)
        Lc2 = h_expanded ** 2  # V^(1/3) 的平方 = V^(2/3)
        dt_visc = 0.25 * CFL * rho * Lc2 / np.maximum(mu_eff, 1e-30)

        metric_flux_scale = self._get_metric_flux_scale()  # (n_cells,n_sps)
        det_jacs = self.mesh.jacobians["det_jacs"].reshape(n_cells, self.mesh.n_sps_per_cell)
        # Order Continuation 期间当前状态 n_sps 可能与网格 n_sps 不同——
        # 度量场是网格固有量，跟当前解阶数无关，按需重复/裁剪到当前
        # n_sps（与上面 h_expanded 对体积的处理是同一原则）。
        if det_jacs.shape[1] != n_sps:
            rep = int(np.ceil(n_sps / det_jacs.shape[1]))
            det_jacs = np.tile(det_jacs, (1, rep))[:, :n_sps]
            metric_flux_scale = np.tile(metric_flux_scale, (1, rep))[:, :n_sps]
        dt_geometric = CFL * np.abs(det_jacs) / np.maximum(metric_flux_scale * wave_speed, 1e-300)

        return np.minimum(np.minimum(dt_advective, dt_visc), dt_geometric)

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
