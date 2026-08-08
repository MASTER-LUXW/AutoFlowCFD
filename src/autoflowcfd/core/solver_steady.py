"""基于 Flux Reconstruction 格式的稳态求解器。

本模块实现稳态 RANS 求解器的协调器，编排 FVM 算法与求解循环。

Key Components:
    - SteadyResult: 稳态仿真结果容器
    - FRSolver: 稳态求解器主类（协调器）
"""

import time
import numpy as np
from typing import Dict, Optional, List, Union
from dataclasses import dataclass, field
from loguru import logger

from ..grid.structures import GridData, VolumeMeshData
from ..config.solver_config import SteadyConfig, TurbulenceModel, BackendType
from .time_integration import TimeIntegrator, TimeIntegrationScheme
from .backend import create_backend
from ..boundary.manager import BoundaryManager
from .fvm_core import FVMFaceExtractor
from .bc_handler import BoundaryConditionHandler
from .aero_coeffs import AeroCoefficientCalculator
from .checkpoint import CheckpointManager
from .solver_steady_setup import SteadySetupMixin
from .solver_steady_cfl import CFLTrendAdjustMixin


@dataclass
class SteadyResult:
    """稳态仿真结果容器。"""

    converged: bool
    iterations: int
    final_residual: float
    cd_history: List[float] = field(default_factory=list)
    cl_history: List[float] = field(default_factory=list)
    residuals_history: List[float] = field(default_factory=list)
    solution_final: Optional[np.ndarray] = None
    checkpoint_path: Optional[str] = None

    def get_mean_coefficients(self) -> Dict[str, float]:
        """计算平均气动系数。"""
        if len(self.cd_history) == 0:
            return {"Cd": 0.0, "Cl": 0.0}

        n_samples = max(1, len(self.cd_history) // 10)
        cd_mean = float(np.mean(self.cd_history[-n_samples:]))
        cl_mean = float(np.mean(self.cl_history[-n_samples:]))

        return {"Cd": cd_mean, "Cl": cl_mean}

    def get_convergence_rate(self) -> float:
        """计算平均收敛速率。"""
        if len(self.residuals_history) < 2:
            return 0.0

        initial_residual = self.residuals_history[0]
        final_residual = self.residuals_history[-1]

        if initial_residual <= 0:
            return 0.0

        total_reduction = np.log(initial_residual / max(final_residual, 1e-16))
        return total_reduction / self.iterations


class FRSolver(SteadySetupMixin, CFLTrendAdjustMixin):
    """Flux Reconstruction 稳态求解器协调器。

    编排 FVM 算法、边界条件和求解循环。

    `_prepare_geometry_and_residual`（solve() 迭代循环开始前的一次性
    准备工作：面几何、壁面距离、ViscousRANSResidual 构造）在
    `solver_steady_setup.SteadySetupMixin` 里；`_adjust_cfl_by_trend`
    （循环内按残差趋势调整 CFL）在 `solver_steady_cfl.CFLTrendAdjustMixin`
    里。两者都纯粹是为了控制单文件行数拆出去的，不是独立的概念层。
    """

    def __init__(self, grid_data: Union[GridData, VolumeMeshData], config: SteadyConfig):
        """初始化稳态求解器。"""
        self.grid_data = grid_data
        self.config = config

        logger.info(f"Initializing FRSolver")
        logger.info(f"  Grid: {grid_data.node_count} nodes, {grid_data.cell_count} cells")

        # Backend 选择。ViscousRANSResidual 的热点循环（AUSM+up 通量、
        # 粘性通量、SST 涡粘性/源项、Green-Gauss 梯度）会自动分发到 Numba
        # CPU kernel。--backend gpu 会让它额外尝试 CUDA kernel
        # （fvm_*_kernels_gpu.py）——这些 kernel 在本项目里**从未在真实
        # GPU 硬件上运行过**（开发时没有可用 GPU），若运行时实际没有
        # CUDA 设备会带警告退回 CPU 路径。在真实硬件上跑过并做数值校验
        # 之前，应把 --backend gpu 当作实验性/未验证功能对待。
        try:
            self.backend = create_backend(
                backend_type=config.backend.value,
                n_threads=config.n_threads,
                device_id=config.gpu_device,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize backend: {e}")

        self._use_gpu_residual = config.backend == BackendType.GPU
        if self._use_gpu_residual:
            logger.warning(
                "--backend gpu was requested: the RANS residual will attempt "
                "to dispatch to CUDA kernels (fvm_*_kernels_gpu.py). These "
                "kernels have not been validated on real GPU hardware in "
                "this project - if no CUDA device is available at runtime "
                "the solve transparently falls back to the CPU (Numba) path."
            )
        if config.order != 2:
            logger.warning(
                f"--order {config.order} was requested, but the residual's "
                "MUSCL reconstruction (Green-Gauss gradient + Barth-Jespersen "
                "limiter) is fixed at 2nd order - the FR order setting "
                "currently has no effect on the steady solve."
            )
        if config.turbulence not in (TurbulenceModel.SST_KW, TurbulenceModel.NONE):
            logger.warning(
                f"--turbulence {config.turbulence.value} was requested, but "
                "the residual's turbulence closure is a fixed SST k-omega "
                "model - only 'sst_kw' (on) or 'none' (off) currently apply."
            )

        # 时间积分器（伪时间上的显式 SSP-RK3）。
        self.time_integrator = TimeIntegrator(
            scheme=TimeIntegrationScheme.SSP_RK3,
            dt=1e-4,
            cfl_target=config.cfl_init,
        )

        # 边界管理器
        self.boundary_manager = BoundaryManager(grid_data.boundaries)

        # 解向量
        self.solution = None

        # 恢复求解时从 checkpoint 恢复的收敛历史（由 CLI 在调用 solve()
        # 之前设置——见 cli/solve_commands.py 的恢复路径）。solve() 用它
        # 来恢复自适应 CFL 状态（cfl_history），而不是每次恢复求解都
        # 重置回 config.cfl_init。
        self.convergence_history = None

        # FVM 面数据持有者（不使用 build_from_tetrahedra()——见 solve()：
        # 面数据来自 grid_data.ensure_faces_exist()，即 Numba 加速路径；
        # 这个实例只是作为共享数据容器，供 bc_handler/aero_calculator
        # 从中读取面数组）。
        self.face_extractor = FVMFaceExtractor()

        # 辅助模块
        self.bc_handler = BoundaryConditionHandler(
            grid_data, self.face_extractor,
            rho_inf=config.rho_inf, p_inf=config.p_inf,
        )
        self.aero_calculator = AeroCoefficientCalculator(
            grid_data, self.face_extractor,
            rho_inf=config.rho_inf, vel_inf=config.vel_inf,
        )

        # Checkpoint 管理器
        self.checkpoint_manager = CheckpointManager(
            config=config,
            output_dir=config.output_dir,
            checkpoint_interval=config.checkpoint_interval
        )

        logger.info("FRSolver initialization complete")

    def _get_cell_volumes(self) -> np.ndarray:
        """获取单元体积。"""
        if isinstance(self.grid_data, VolumeMeshData):
            return self.grid_data.get_cell_volumes()
        else:
            return self.cell_volumes

    def _initialize_solution(self):
        """用自由来流条件初始化解场。

        用自由来流速度作为初始条件，保证数值稳定性。从静止状态起步
        可能导致 HLLC 通量计算不稳定。

        解变量（守恒形式）：
            [rho, rhou, rhov, rhow, E, k, omega]
        """
        logger.info("Initializing solution field...")

        n_cells = self.grid_data.cell_count

        # 稳定初始化用的自由来流条件（单一数据源：self.config，与边界
        # 条件、Cd/Cl 归一化共用，保证三者始终一致）。
        rho_0 = self.config.rho_inf
        u_0 = self.config.vel_inf     # 用自由来流速度以保证稳定性
        v_0 = 0.0
        w_0 = 0.0
        p_0 = self.config.p_inf
        gamma = 1.4

        # 计算守恒变量
        rhou_0 = rho_0 * u_0
        rhov_0 = rho_0 * v_0
        rhow_0 = rho_0 * w_0
        E_0 = p_0 / (gamma - 1.0) + 0.5 * rho_0 * (u_0**2 + v_0**2 + w_0**2)

        # 湍流场：守恒形式 (rho*k, rho*omega)。
        # 自由来流：1% 湍流强度，长度尺度 ~0.1 m。
        u_ref = max(u_0, 1.0)
        k_0 = 1.5 * (0.01 * u_ref)**2
        omega_0 = 5.0 * u_ref / 0.1

        # 分配并初始化解数组
        self.solution = np.zeros((n_cells, 7), dtype=np.float64)
        self.solution[:, 0] = rho_0   # 密度
        self.solution[:, 1] = rhou_0  # x 方向动量
        self.solution[:, 2] = rhov_0  # y 方向动量
        self.solution[:, 3] = rhow_0  # z 方向动量
        self.solution[:, 4] = E_0     # 总能
        self.solution[:, 5] = rho_0 * k_0      # 守恒形式湍动能
        self.solution[:, 6] = rho_0 * omega_0  # 守恒形式比耗散率

        logger.info(f"Solution initialized: {n_cells} cells")
        logger.info(f"  Initial conditions: rho={rho_0:.3f} kg/m^3, u={u_0:.1f} m/s, p={p_0:.0f} Pa")
        logger.info(f"  Turbulence: k={k_0:.4e} m^2/s^2, omega={omega_0:.2f} 1/s")

    def _setup_boundary_conditions(self):
        """配置边界条件。"""
        logger.info("Setting up boundary conditions...")

        boundary_names = self.grid_data.boundaries.boundary_names
        vel_inf = self.config.vel_inf
        p_inf = self.config.p_inf

        for boundary_name in boundary_names:
            name_upper = boundary_name.upper()

            if "INLET" in name_upper or "INFLOW" in name_upper:
                # 存储基准速度供爬升机制使用（现在由 bc_handler 处理）
                self.bc_handler.base_inlet_velocity = vel_inf
                self.boundary_manager.add_bc(
                    boundary_name, bc_type="INLET",
                    velocity_x=vel_inf, pressure=p_inf, temperature=288.15,
                )
            elif "OUTLET" in name_upper:
                self.boundary_manager.add_bc(boundary_name, bc_type="OUTLET", pressure=p_inf)
            elif "BODY" in name_upper or "CAR" in name_upper:
                self.boundary_manager.add_bc(boundary_name, bc_type="WALL")
            elif "GROUND" in name_upper:
                self.bc_handler.base_farfield_velocity = vel_inf
                self.boundary_manager.add_bc(boundary_name, bc_type="GROUND", moving_wall_velocity=vel_inf)
            elif "TUNNEL" in name_upper:
                # 命名为 "tunnel" 的边界是一个物理（无摩擦）风洞壁面，
                # 不是开放的域边界——对应 bc_handler.py 的 _classify 里
                # 实际求解路径采用的同一种重新分类（SYMMETRY = 自由滑移、
                # 零穿透）。
                self.boundary_manager.add_bc(boundary_name, bc_type="SYMMETRY")
            elif "FARFIELD" in name_upper:
                self.bc_handler.base_farfield_velocity = vel_inf
                self.boundary_manager.add_bc(boundary_name, bc_type="FARFIELD", velocity_x=vel_inf, pressure=p_inf)
            elif "SYMMETRY" in name_upper:
                self.boundary_manager.add_bc(boundary_name, bc_type="SYMMETRY")
            else:
                self.bc_handler.base_farfield_velocity = vel_inf
                self.boundary_manager.add_bc(boundary_name, bc_type="FARFIELD", velocity_x=vel_inf, pressure=p_inf)

        logger.info(f"Boundary conditions setup: {len(boundary_names)} boundaries")

    def solve(self, max_iter: Optional[int] = None, start_iteration: int = 0):
        """执行稳态仿真。

        对二阶粘性 RANS 残差做显式 SSP-RK 伪时间推进，配合局部时间步长
        和归一化的多方程收敛判据。

        Args:
            max_iter: 总迭代数上限（若提供则覆盖 config 里的值）。这是
                一个绝对目标，而不是"再跑多少步"——从
                start_iteration=50 恢复求解、max_iter=2500 时，意思是
                "总共跑到第 2500 次迭代"（还剩 2450 步），与 CLI 文档
                里说明的语义一致。
            start_iteration: 已完成的迭代数（例如从 checkpoint 加载）。
                迭代计数、日志，以及入口/远场速度的爬升机制
                （BoundaryConditionHandler.update_ramp_factor）都从这里
                继续，而不是重新从 1 开始——否则恢复求解会把爬升系数
                猛地拉回 ~0，在恢复点重新引入一次边界条件的不连续。

        Returns:
            带解和历史记录的 SteadyResult
        """

        # 初始化
        if self.solution is None:
            self._initialize_solution()

        self._setup_boundary_conditions()

        (geom, wall_distance, wall_face_mask, mu_lam, turbulent, mach_ref,
         residual) = self._prepare_geometry_and_residual()

        actual_max_iter = max_iter if max_iter is not None else self.config.max_iter
        if start_iteration >= actual_max_iter:
            logger.warning(
                f"start_iteration ({start_iteration}) >= max_iter ({actual_max_iter}); "
                f"nothing to do."
            )
        logger.info(
            f"Starting steady RANS solve (start_iteration={start_iteration}, "
            f"max_iter={actual_max_iter}, turbulent={turbulent})"
        )

        # 恢复求解时恢复自适应 CFL 状态。没有这一步，每次从 checkpoint
        # 恢复时 CFL 都会重置回 config.cfl_init，丢弃自适应机制原本已经
        # 收敛到的值（例如为挺过一个刚性区域而降到远低于 cfl_init 的
        # CFL）——恢复后最初几步就会重新尝试原始的、已知有风险的
        # cfl_init，可能立刻再次发散。
        if self.convergence_history and self.convergence_history.get('cfl_history'):
            restored_cfl = float(self.convergence_history['cfl_history'][-1])
            if restored_cfl > 0:
                logger.info(
                    f"Restoring adaptive CFL from checkpoint history: "
                    f"{self.time_integrator.cfl_target:.4f} -> {restored_cfl:.4f}"
                )
                self.time_integrator.cfl_target = restored_cfl

        cd_history, cl_history, res_history = [], [], []
        initial_res_vec = None
        converged = False
        start = time.time()
        # 下面三种 CFL 调整机制（发散自动恢复、爆炸式增长防护、按残差
        # 趋势的自适应规则）共用的协调状态，避免它们互相打架——例如
        # 趋势规则在安全机制刚刚下调 CFL 之后的同一步或紧接着的下一步
        # 就把它调回去，还没让较低的 CFL 有机会真正起作用。
        last_cfl_cut_iteration = -10**9
        cfl_cut_cooldown = 20
        # 如果 start_iteration >= actual_max_iter，下面的循环体永远不会
        # 执行；保持 `iteration` 有定义（作为最后完成的迭代数），这样
        # 循环结束后的日志/checkpoint 代码就不会引用一个未绑定的局部
        # 变量。
        iteration = start_iteration

        def residual_func(U):
            bstates = self.bc_handler.build_boundary_states(U)
            return residual.compute(U, bstates)

        for step in range(1, actual_max_iter - start_iteration + 1):
            iteration = start_iteration + step
            # 重置本次迭代的标志：这一步 CFL 是否已经被某个安全机制
            # （发散恢复 / 爆炸式增长防护）下调过？如果是，下面基于趋势
            # 的规则会整个跳过，而不是有可能在同一步里把这次下调撤销。
            cfl_cut_this_iter = False
            if self.bc_handler is not None:
                self.bc_handler.update_ramp_factor(iteration, actual_max_iter)

            # 这次解对应的边界 ghost 状态——只算一次，下面（梯度、残差、
            # 气动系数）复用，而不是每个消费者都从头重建一遍。
            bstates = self.bc_handler.build_boundary_states(self.solution)

            # 粘性时间步限制用的有效粘性。
            rho_c, vel_c, p_c, T_c, k_c, w_c = residual.to_primitive(self.solution)

            # 诊断：梯度计算前记录形状
            if iteration <= 5 or iteration % 10 == 0:
                logger.debug(
                    f"[Iter {iteration}] Primitive variables shapes:\n"
                    f"  rho_c: {rho_c.shape}, vel_c: {vel_c.shape}\n"
                    f"  k_c: {k_c.shape}, w_c (omega): {w_c.shape}\n"
                    f"  wall_distance: {residual.wall_distance.shape}"
                )

            try:
                gvel = residual._velocity_gradient(vel_c, self.solution, bstates)

                if iteration <= 5 or iteration % 10 == 0:
                    logger.debug(f"[Iter {iteration}] Velocity gradient shape: {gvel.shape}")

                mu_t = residual._eddy_viscosity(rho_c, k_c, w_c, gvel) if turbulent \
                    else np.zeros(geom.n_cells)

                if iteration <= 5 or iteration % 10 == 0:
                    logger.debug(f"[Iter {iteration}] Eddy viscosity shape: {mu_t.shape}")
            except Exception as e:
                logger.error(f"[Iter {iteration}] Failed to compute viscosity: {e}")
                import traceback
                logger.error(traceback.format_exc())
                raise

            # 用当前 CFL 计算局部时间步长。omega=w_c 附加了 SST 源项
            # 刚性限制（见 local_time_step 文档字符串）——没有它，近壁
            # 单元的 omega 较大时，即便 CFL 对对流/粘性平均流场项来说
            # 相当安全，k/omega 方程仍可能不稳定。
            dt_local = self.time_integrator.local_time_step(
                self.solution, geom, mu_lam + mu_t, omega=(w_c if turbulent else None),
                mach_ref=mach_ref,
            )

            # 一次 SSP-RK 伪时间步。R 既是下面收敛监控要用的残差，也是
            # RK 格式自己的第 0 阶段残差（i=0 时 Ui=U0）——只算一次，
            # 通过 residual0= 复用，而不是让 step() 把这次（昂贵的：
            # MUSCL+HLLC+粘性+SST 源项）计算再算一遍。
            R = residual.compute(self.solution, bstates)
            self.solution = self.time_integrator.step(
                self.solution, residual_func, dt_local, residual0=R
            )

            # 更新后立即检查数值发散
            if not np.all(np.isfinite(self.solution)):
                logger.error(f"Solver diverged at iteration {iteration}: non-finite state detected")
                logger.error("  Possible causes:")
                logger.error("    1. CFL number too high for current grid/solution")
                logger.error("    2. Boundary condition inconsistency")
                logger.error("    3. Turbulence model stiffness (try reducing CFL)")
                logger.error(f"  Current CFL: {self.time_integrator.cfl_target:.4f}")

                # === 自动恢复尝试 ===
                if self.time_integrator.cfl_target > 0.01:
                    old_cfl = self.time_integrator.cfl_target
                    self.time_integrator.cfl_target = max(old_cfl * 0.1, 0.005)
                    last_cfl_cut_iteration = iteration
                    logger.warning(f"[AUTO-RECOVERY] Attempting automatic recovery by reducing CFL to {self.time_integrator.cfl_target:.4f}")

                    # 若有可用的上一步解，恢复它
                    if iteration > 1 and hasattr(self, '_last_stable_solution'):
                        self.solution = self._last_stable_solution.copy()
                        logger.info("[AUTO-RECOVERY] Restored solution from last stable state")

                    continue  # 用更低的 CFL 重试这次迭代
                else:
                    raise RuntimeError(f"Solver diverged at iteration {iteration}: non-finite state")

            # 保存稳定解，供潜在的恢复使用
            if iteration % 5 == 0:
                self._last_stable_solution = self.solution.copy()

            # 归一化多方程残差（质量/动量/能量的 RMS）。
            #
            # 按体积加权，而不是简单的逐单元平均：R 已经是单位体积量
            # （residual.compute() 已经除以 cell_volumes），但不加权的
            # 平均仍然会让每个单元的权重相等，无论它代表多大的一部分
            # 计算域——边界层可能有成千上万个局部残差本身就带噪声的
            # 微小单元，若不加权，它们会淹没来自（数量少得多、但体积
            # 大得多的）远场单元的信号。
            cell_volumes = geom.cell_volumes
            total_volume = float(np.sum(cell_volumes))
            res_vec = np.sqrt(np.sum(R[:, :5]**2 * cell_volumes[:, None], axis=0) / total_volume)

            # 5 个方程各自按自己首次迭代时的 RMS 归一化后，才合并成一个
            # 标量。没有这一步，能量方程的残差——量级本来就是
            # ~rho*vel_inf^3，比连续性/动量残差大好几个数量级——会主导
            # 一个简单合并的 L2 范数，使得"收敛"实际上只反映能量方程，
            # 而质量/动量可能还在变化。
            if initial_res_vec is None or iteration == 1:
                initial_res_vec = np.maximum(res_vec, 1e-30)
            rel_res = float(np.linalg.norm(res_vec / initial_res_vec)) / np.sqrt(len(res_vec))
            res_history.append(rel_res)

            # === 提前发散预警 ===
            if len(res_history) >= 3:
                recent_growth = res_history[-1] / max(res_history[-3], 1e-30)
                if recent_growth > 1e6:  # 检测到爆炸式增长
                    logger.warning(
                        f"[DIVERGENCE WARNING] Residual grew by factor {recent_growth:.2e} in 3 steps! "
                        f"Current CFL={self.time_integrator.cfl_target:.4f}. "
                        f"Consider manual intervention."
                    )
                    # 强制大幅下调 CFL
                    self.time_integrator.cfl_target = max(self.time_integrator.cfl_target * 0.2, 0.005)
                    last_cfl_cut_iteration = iteration
                    cfl_cut_this_iter = True
                    logger.warning(f"[AUTO-FIX] Aggressively reduced CFL to {self.time_integrator.cfl_target:.4f}")

            # 按残差趋势自适应调整 CFL——见
            # solver_steady_cfl.CFLTrendAdjustMixin._adjust_cfl_by_trend。
            last_cfl_cut_iteration = self._adjust_cfl_by_trend(
                res_history, iteration, cfl_cut_this_iter,
                last_cfl_cut_iteration, cfl_cut_cooldown,
            )

            # 每次迭代都算系数，保证监控的准确性。包含通过
            # wall_shear_stress() 算出的摩擦阻力/升力，复用本次迭代
            # 已经算好的 gvel/mu_t/bstates（在更新步之前算的），而不是
            # 针对更新后的解重新计算——用一次迭代的滞后换取不必每次
            # 迭代都为了监控输出而把梯度+涡粘性的开销翻倍，是划算的
            # 权衡。

            # 诊断：计算系数前记录形状
            if iteration <= 5 or iteration % 10 == 0:
                logger.debug(
                    f"[Iter {iteration}] Pre-compute diagnostics:\n"
                    f"  Solution shape: {self.solution.shape}\n"
                    f"  gvel shape: {gvel.shape if gvel is not None else 'None'}\n"
                    f"  mu_t shape: {mu_t.shape if mu_t is not None else 'None'}\n"
                    f"  bstates shape: {bstates.shape if bstates is not None else 'None'}\n"
                    f"  Face normals shape: {self.face_extractor.face_normals.shape}\n"
                    f"  Face areas shape: {self.face_extractor.face_areas.shape}"
                )

            try:
                Cd, Cl, Cd_p, Cd_f = self.aero_calculator.compute_coefficients(
                    self.solution, iteration,
                    viscous_residual=residual, grad_vel=gvel, mu_t=mu_t,
                    boundary_states=bstates,
                )
            except Exception as e:
                logger.error(f"[Iter {iteration}] Failed to compute aerodynamic coefficients: {e}")
                import traceback
                logger.error(traceback.format_exc())
                raise

            cd_history.append(Cd)
            cl_history.append(Cl)

            # 输出系数，格式更清晰
            # 主行：迭代信息
            logger.info(
                f"Iter {iteration:5d}/{actual_max_iter}  |  "
                f"Res(rel): {rel_res:.4e}  |  "
                f"Cd: {Cd:.4f}  |  "
                f"Cl: {Cl:.4f}"
            )

            # 第二行：Cd 分解（与第一个 | 对齐）
            # 计算缩进量，让 'Cd' 对齐到第一个 '|' 的位置
            # "Iter XXXX/XXXX  |  " 约 20 个字符，所以缩进到第一个 | 的位置
            prefix_len = len(f"Iter {iteration:5d}/{actual_max_iter}")
            logger.info(
                f"{'':>{prefix_len + 2}s}  "
                f"Cd breakdown: pressure={Cd_p:.4f}, friction={Cd_f:.4f}"
            )

            # 定期保存 checkpoint
            if self.checkpoint_manager.should_save(iteration):
                history_dict = {
                    'iterations': list(range(1, iteration + 1)),
                    'residuals': {'continuity': res_history.copy()},
                    'coefficients': {'Cd': cd_history.copy(), 'Cl': cl_history.copy()},
                    'cfl_history': [self.time_integrator.cfl_target] * len(res_history),
                }

                ckpt_path = self.checkpoint_manager.save(
                    solution=self.solution,
                    history=history_dict,
                    iteration=iteration,
                    extra_fields={'mu_t': mu_t},
                )

                if ckpt_path:
                    logger.info(f"Checkpoint saved at iteration {iteration}")

                    # 清理旧的 checkpoint
                    self.checkpoint_manager.cleanup_old_checkpoints(keep_last=3)

            # 收敛判据：归一化残差低于容差。
            if rel_res < self.config.convergence_tol:
                logger.success(f"Converged at iteration {iteration} (rel residual {rel_res:.3e})")
                converged = True
                break

        elapsed = time.time() - start
        logger.info(f"Solve finished: {len(res_history)} iters, {elapsed:.1f}s, converged={converged}")

        # 保存最终 checkpoint
        try:
            logger.debug("Preparing final checkpoint data...")
            final_history = {
                'iterations': list(range(1, iteration + 1)),
                'residuals': {'continuity': res_history.copy()},
                'coefficients': {'Cd': cd_history.copy(), 'Cl': cl_history.copy()},
                'cfl_history': [self.time_integrator.cfl_target] * len(res_history),
            }

            logger.debug(f"Final history keys: {final_history.keys()}")
            logger.debug(f"Cd history length: {len(cd_history)}, Cl history length: {len(cl_history)}")

            final_ckpt = self.checkpoint_manager.save(
                solution=self.solution,
                history=final_history,
                iteration=iteration,
                extra_fields={'mu_t': mu_t},
            )

            if final_ckpt:
                logger.info(f"Final checkpoint saved: {final_ckpt}")
        except Exception as e:
            logger.error(f"Failed to save final checkpoint: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

        return SteadyResult(
            converged=converged,
            iterations=len(res_history),
            final_residual=res_history[-1] if res_history else 0.0,
            cd_history=cd_history,
            cl_history=cl_history,
            residuals_history=res_history,
            solution_final=self.solution.copy(),
            checkpoint_path=final_ckpt,
        )
