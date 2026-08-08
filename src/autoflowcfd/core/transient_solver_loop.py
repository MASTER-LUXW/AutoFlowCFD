"""时间精确的非定常 RANS/DES 求解器。

推进的是与稳态求解器相同的粘性 RANS 残差
（:class:`~autoflowcfd.core.fvm_viscous_residual.ViscousRANSResidual`），
但用固定、均匀的时间步长在**物理**时间上前进——复用稳态求解器的面
几何、边界条件处理、气动力积分，而不是另外维护一套独立的
通量/边界条件/力 计算流程。

DES/DDES/LES 模式目前只实现了 Spalart DES97 基于网格尺度的长度尺度
限制器，作用在 SST k 耗散项上（见 :meth:`TransientSolver._apply_des_correction`）。
DDES 的边界层屏蔽函数和真正的可解析湍流 LES 亚格子模型都未实现；两者
目前都退化为 DES97，并在启动时给出警告。
"""

from __future__ import annotations

import time
import numpy as np
from typing import Optional
from loguru import logger

from ..grid.structures import VolumeMeshData
from ..config.solver_config import TransientConfig, TurbulenceModel, TimeIntegrationScheme as ConfigTimeScheme
from .fvm_core import FVMFaceExtractor
from .fvm_gradients import FaceGeometry
from .fvm_viscous_residual import ViscousRANSResidual, estimate_wall_distance
from .bc_handler import BoundaryConditionHandler
from .aero_coeffs import AeroCoefficientCalculator
from .checkpoint import CheckpointManager
from .time_integration import TimeIntegrator, TimeIntegrationScheme
from .transient_result import TransientResult

# TransientConfig/CLI 里的格式名早于稳态求解器的 SSP-RK 重写（关于
# "backward_euler"/"ab3" 为什么是显式格式的旧别名，见
# time_integration.py 自己的模块文档字符串）——这里把它们映射到本求解器
# 实际使用的真实积分器上。
_SCHEME_MAP = {
    ConfigTimeScheme.BACKWARD_EULER: TimeIntegrationScheme.FORWARD_EULER,
    ConfigTimeScheme.RK2: TimeIntegrationScheme.SSP_RK2,
    ConfigTimeScheme.RK3: TimeIntegrationScheme.SSP_RK3,
    ConfigTimeScheme.AB3: TimeIntegrationScheme.SSP_RK3,
}

MU_LAM = 1.7894e-5  # 288 K 下的 Sutherland 粘度, Pa s
SST_BETA_STAR = 0.09
C_DES = 0.61  # Spalart DES97 常数（SST 标定）


class TransientSolver:
    """时间精确的非定常 RANS/DES 求解器协调器。"""

    def __init__(self, grid_data: VolumeMeshData, config: TransientConfig):
        self.grid_data = grid_data
        self.config = config
        self.solution: Optional[np.ndarray] = None

        logger.info("Initializing TransientSolver")
        logger.info(f"  Grid: {grid_data.node_count} nodes, {grid_data.cell_count} cells")

        if config.turbulence in (TurbulenceModel.DDES, TurbulenceModel.LES):
            logger.warning(
                f"turbulence={config.turbulence.value}: this solver only implements "
                "the Spalart DES97 grid-length-scale limiter (see _apply_des_correction). "
                "DDES's boundary-layer shielding function is not implemented, so thin "
                "near-wall cells may switch to LES-mode dissipation where a real DDES "
                "would not; 'les' is not a resolved sub-grid-scale model here and falls "
                "back to the same DES97 treatment as 'des'."
            )
        self.use_des = config.turbulence in (
            TurbulenceModel.DES, TurbulenceModel.DDES, TurbulenceModel.LES
        )
        self.turbulent = config.turbulence != TurbulenceModel.NONE

        self.face_extractor = FVMFaceExtractor()
        self.bc_handler = BoundaryConditionHandler(
            grid_data, self.face_extractor,
            rho_inf=config.rho_inf, p_inf=config.p_inf,
        )
        self.aero_calculator = AeroCoefficientCalculator(
            grid_data, self.face_extractor,
            rho_inf=config.rho_inf, vel_inf=config.vel_inf,
        )
        self.checkpoint_manager = CheckpointManager(
            config, output_dir=config.output_dir,
            checkpoint_interval=config.checkpoint_interval,
        )

        scheme = _SCHEME_MAP.get(config.time_scheme)
        if scheme is None:
            logger.warning(f"Unknown time_scheme {config.time_scheme}, defaulting to SSP-RK3")
            scheme = TimeIntegrationScheme.SSP_RK3
        # cfl_target 这里不会用到（dt 是固定/均匀的，不做 CFL 自适应），
        # 但 TimeIntegrator.local_time_step 在本求解器里根本不会被调用。
        self.time_integrator = TimeIntegrator(scheme=scheme, dt=config.dt, cfl_target=1.0)

        self.current_time = 0.0
        self.n_steps = 0
        self.cd_history: list = []
        self.cl_history: list = []
        self.time_stamps: list = []
        self.checkpoint_path: Optional[str] = None

        self._geom: Optional[FaceGeometry] = None
        self._residual: Optional[ViscousRANSResidual] = None
        self._des_delta: Optional[np.ndarray] = None

        logger.info("TransientSolver initialization complete")

    # ------------------------------------------------------------------
    def _initialize_solution(self) -> None:
        """均匀自由来流初始条件（形式与 FRSolver 相同）。"""
        n_cells = self.grid_data.cell_count
        rho_0, u_0, p_0 = self.config.rho_inf, self.config.vel_inf, self.config.p_inf
        gamma = 1.4
        E_0 = p_0 / (gamma - 1.0) + 0.5 * rho_0 * u_0 ** 2
        u_ref = max(u_0, 1.0)
        k_0 = 1.5 * (0.01 * u_ref) ** 2
        omega_0 = 5.0 * u_ref / 0.1

        self.solution = np.zeros((n_cells, 7), dtype=np.float64)
        self.solution[:, 0] = rho_0
        self.solution[:, 1] = rho_0 * u_0
        self.solution[:, 4] = E_0
        self.solution[:, 5] = rho_0 * k_0
        self.solution[:, 6] = rho_0 * omega_0
        logger.info(f"Solution initialized: {n_cells} cells")

    def _setup_boundary_conditions(self) -> None:
        """把自由来流目标值接进 bc_handler（不做爬升：这是真正的时间
        精确推进，不是伪时间启动——在这里对边界条件做爬升只会引入一个
        不符合物理的、随时间变化的强迫项）。"""
        vel_inf = self.config.vel_inf
        for name in self.grid_data.boundaries.boundary_names:
            u = name.upper()
            if "INLET" in u or "INFLOW" in u:
                self.bc_handler.base_inlet_velocity = vel_inf
            elif "GROUND" in u or "FARFIELD" in u:
                self.bc_handler.base_farfield_velocity = vel_inf
            elif "BODY" in u or "CAR" in u or "OUTLET" in u or "SYMMETRY" in u or "TUNNEL" in u:
                # TUNNEL 被 bc_handler.py 的 _classify 分类为 SYMMETRY
                # （自由滑移风洞壁面）——与真正开放的 FARFIELD 边界不同，
                # 它完全不用 base_farfield_velocity。
                pass
            else:
                self.bc_handler.base_farfield_velocity = vel_inf
        self.bc_handler.ramp_factor = 1.0
        logger.info(f"Boundary conditions setup: {len(self.grid_data.boundaries.boundary_names)} boundaries")

    def _setup(self) -> None:
        """构建面几何/残差对象。幂等——可以安全地从 solve() 和
        load_checkpoint() 两处调用。"""
        if self._geom is not None:
            return

        self._setup_boundary_conditions()
        if self.solution is None:
            self._initialize_solution()

        logger.info("Building face geometry (optimized radix-sort)...")
        face_data_obj = self.grid_data.ensure_faces_exist()

        nodes_array = np.column_stack([
            self.grid_data.nodes.x, self.grid_data.nodes.y, self.grid_data.nodes.z,
        ])
        # 见 solver_steady.py 里同样修复的对应注释：三棱柱单元
        # （VolumeMeshData.prism_cells）占据全局单元索引空间的前段，
        # 四面体在后——中心点的计算顺序必须和下面
        # grid_data.get_cell_volumes() 已经采用的顺序一致，否则一旦用了
        # 三棱柱网格，每个边界层单元的中心点都会悄悄地和
        # cell_volumes/面的 owner-neighbour 索引对不上。
        tet_connectivity_int64 = self.grid_data.cells.connectivity.astype(np.int64)
        tet_centroids = nodes_array[tet_connectivity_int64].mean(axis=1)
        prism_cells_obj = getattr(self.grid_data, 'prism_cells', None)
        if prism_cells_obj is not None:
            prism_connectivity_int64 = prism_cells_obj.connectivity.astype(np.int64)
            prism_centroids = nodes_array[prism_connectivity_int64].mean(axis=1)
            cell_centroids = np.vstack([prism_centroids, tet_centroids])
        else:
            cell_centroids = tet_centroids
        boundary_flags = (face_data_obj.connectivity[:, 1] < 0).astype(np.int32)

        self.face_extractor.cell_centroids = cell_centroids
        self.face_extractor.face_connectivity = face_data_obj.connectivity
        self.face_extractor.face_normals = face_data_obj.normal
        self.face_extractor.face_areas = face_data_obj.area
        self.face_extractor.boundary_flags = boundary_flags

        cell_volumes = self.grid_data.get_cell_volumes()
        self._geom = FaceGeometry(
            connectivity=face_data_obj.connectivity,
            normals=face_data_obj.normal,
            areas=face_data_obj.area,
            centers=face_data_obj.center,
            boundary_flags=boundary_flags,
            cell_centroids=cell_centroids,
            cell_volumes=cell_volumes,
        )

        self.bc_handler._precompute_face_types()
        wall_face_mask = np.zeros(self._geom.n_faces, dtype=bool)
        for f, t in self.bc_handler._face_types.items():
            if t in ("WALL", "GROUND"):
                wall_face_mask[f] = True
        wall_distance = estimate_wall_distance(self._geom, wall_face_mask)

        # AUSM+up 低马赫数缩放函数 f_a 用的参考马赫数（见
        # fvm_viscous_residual.py 的 _ausm_up / solver_steady.py 里对应的
        # 注释）——这里的作用和稳态求解器里一样。
        gamma_air = 1.4
        a_inf = np.sqrt(gamma_air * self.config.p_inf / self.config.rho_inf)
        mach_ref = self.config.vel_inf / max(a_inf, 1e-30)

        self._residual = ViscousRANSResidual(
            self._geom, mu_lam=MU_LAM, wall_distance=wall_distance, turbulent=self.turbulent,
            mach_ref=mach_ref,
            wall_face_mask=wall_face_mask if self.config.use_wall_functions else None,
            use_gpu=self.config.is_gpu,
        )
        if self.config.use_wall_functions:
            logger.info(
                "Wall functions enabled (Menter scalable/automatic wall treatment) "
                f"on {np.sum(wall_face_mask)} WALL/GROUND faces - near-wall mesh no "
                "longer needs y+~1 to be accurate."
            )
        self._des_delta = np.cbrt(cell_volumes)

        self._warn_if_dt_unstable()

    def _warn_if_dt_unstable(self) -> None:
        """估算显式格式的稳定性上限（CFL=1，初始条件下的无粘 + SST 源项
        刚性），若配置的固定 dt 超过它则给出警告。与稳态求解器不同，
        本求解器从不根据 CFL 自适应 dt——一个不稳定的选择只会直接发散，
        所以提前给出这个警告是用户在那之前能得到的唯一提示。这只能是
        一个基于初始条件的估计：omega 会随时间演化，所以在 t=0 看起来
        安全的 dt，如果 omega 在壁面附近后续增大，仍可能变得不稳定。"""
        omega0 = None
        if self.turbulent:
            rho0 = np.maximum(self.solution[:, 0], 1e-9)
            omega0 = np.maximum(self.solution[:, 6] / rho0, 1e-6)
        dt_stable = self.time_integrator.local_time_step(self.solution, self._geom, None, omega=omega0)
        min_dt_stable = float(np.min(dt_stable))
        if self.config.dt > min_dt_stable:
            logger.warning(
                f"--dt {self.config.dt:.3e}s exceeds the estimated explicit stability "
                f"limit ({min_dt_stable:.3e}s at t=0, CFL=1, inviscid+SST-stiffness) "
                f"for the smallest/stiffest cell in this mesh. The scheme has no CFL "
                f"adaptation in transient mode (dt is fixed for time accuracy) - this "
                f"will likely diverge."
            )

    # ------------------------------------------------------------------
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """从 HDF5 checkpoint（稳态或瞬态）恢复。"""
        self._setup()
        solution, history, iteration, metadata = self.checkpoint_manager.load(checkpoint_path)
        if self.grid_data.cell_count != solution.shape[0]:
            raise ValueError(
                f"Grid cell count ({self.grid_data.cell_count}) does not match "
                f"checkpoint solution ({solution.shape[0]}). Provide the same "
                f"volume mesh used to produce this checkpoint."
            )
        self.solution = solution
        self.current_time = float(metadata.get('current_time', 0.0))
        self.n_steps = iteration
        if history:
            coeffs = history.get('coefficients', {})
            self.cd_history = list(coeffs.get('Cd', []))
            self.cl_history = list(coeffs.get('Cl', []))
            self.time_stamps = list(history.get('iterations', []))
        logger.info(f"Resumed transient state: t={self.current_time:.6f}s, step={self.n_steps}")

    def _apply_des_correction(self, U: np.ndarray, R: np.ndarray) -> None:
        """Spalart DES97 长度尺度限制器，作用在 SST k 耗散项上。

        用 F_DES = max(l_RANS / (C_des*Delta), 1) 缩放标准耗散项
        Dk = beta_star*rho*k*omega，其中 l_RANS 是模化的湍流长度尺度，
        Delta 是局部网格间距。一旦网格足够细、可以直接解析含能尺度
        （l_RANS > C_des*Delta），这会增大 k 的耗散，让模型让位、由
        可解析湍流接管——这是这里真正把 DES 模式和普通 RANS 区分开的
        唯一物理机制。
        """
        rho, vel, p, T, k, omega = self._residual.to_primitive(U)
        l_rans = np.sqrt(np.maximum(k, 0.0)) / np.maximum(SST_BETA_STAR * omega, 1e-12)
        l_les = C_DES * self._des_delta
        F_DES = np.maximum(l_rans / np.maximum(l_les, 1e-12), 1.0)
        Dk_std = SST_BETA_STAR * rho * k * omega
        # R 里已经包含了 _sst_sources 加上的 +Dk_std；这里只加增量部分
        # (F_DES - 1)，使总量等于 Dk_std * F_DES。
        R[:, 5] += Dk_std * (F_DES - 1.0)

    def _compute_residual(self, U: np.ndarray) -> np.ndarray:
        bstates = self.bc_handler.build_boundary_states(U)
        R = self._residual.compute(U, bstates)
        if self.use_des:
            self._apply_des_correction(U, R)
        return R

    # ------------------------------------------------------------------
    def solve(self) -> TransientResult:
        """把解从 ``current_time`` 推进到 ``config.total_time``。"""
        self._setup()
        geom = self._geom
        residual = self._residual

        dt = self.config.dt
        total_time = self.config.total_time
        dt_array = np.full(geom.n_cells, dt, dtype=np.float64)

        logger.info(
            f"Starting transient solve: t={self.current_time:.6f}s -> "
            f"{total_time:.6f}s, dt={dt:.3e}s, turbulent={self.turbulent}, "
            f"DES={self.use_des}"
        )
        start = time.time()

        while self.current_time < total_time - 1e-12:
            R0 = self._compute_residual(self.solution)
            self.solution = self.time_integrator.step(
                self.solution, self._compute_residual, dt_array, residual0=R0
            )

            self.current_time += dt
            self.n_steps += 1

            if not np.all(np.isfinite(self.solution)):
                raise RuntimeError(
                    f"Transient solve diverged at step {self.n_steps} "
                    f"(t={self.current_time:.6f}s): non-finite state detected"
                )

            bstates = self.bc_handler.build_boundary_states(self.solution)
            rho_c, vel_c, p_c, T_c, k_c, w_c = residual.to_primitive(self.solution)
            gvel = residual._velocity_gradient(vel_c, self.solution, bstates)
            mu_t = residual._eddy_viscosity(rho_c, k_c, w_c, gvel) if self.turbulent \
                else np.zeros(geom.n_cells)
            Cd, Cl, Cd_p, Cd_f = self.aero_calculator.compute_coefficients(
                self.solution, self.n_steps,
                viscous_residual=residual, grad_vel=gvel, mu_t=mu_t,
                boundary_states=bstates,
            )
            self.cd_history.append(Cd)
            self.cl_history.append(Cl)
            self.time_stamps.append(self.current_time)

            if self.n_steps % max(1, self.config.sample_interval) == 0 or self.n_steps == 1:
                logger.info(
                    f"Step {self.n_steps:6d} | t={self.current_time:.6f}s | "
                    f"Cd={Cd:.4f} | Cl={Cl:.4f}"
                )
                # 第二行：Cd 分解（与第一个 | 对齐）
                prefix_len = len(f"Step {self.n_steps:6d}")
                logger.info(
                    f"{'':>{prefix_len + 2}s}  "
                    f"Cd breakdown: pressure={Cd_p:.4f}, friction={Cd_f:.4f}"
                )

            if self.checkpoint_manager.should_save(self.n_steps):
                history_dict = {
                    'iterations': self.time_stamps.copy(),
                    'coefficients': {'Cd': self.cd_history.copy(), 'Cl': self.cl_history.copy()},
                }
                ckpt = self.checkpoint_manager.save(
                    solution=self.solution, history=history_dict, iteration=self.n_steps,
                    metadata={'current_time': self.current_time},
                    extra_fields={'mu_t': mu_t},
                )
                if ckpt:
                    self.checkpoint_path = ckpt

        elapsed = time.time() - start
        logger.info(
            f"Transient solve finished: {self.n_steps} steps, "
            f"t={self.current_time:.6f}s, {elapsed:.1f}s"
        )

        return TransientResult(
            solution_final=self.solution.copy(),
            total_time=self.current_time,
            n_steps=self.n_steps,
            cd_history=self.cd_history,
            cl_history=self.cl_history,
            time_stamps=self.time_stamps,
            checkpoint_path=self.checkpoint_path,
        )
