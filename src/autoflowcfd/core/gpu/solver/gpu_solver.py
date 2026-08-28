"""
AutoFlowCFD V2.0 - GPU FRSolver

完整的 GPU 版 FR 求解器，对应 core/fr_solver.py。
所有计算在 GPU 上完成，数据常驻显存，只在 I/O 时传输。

设计：
- 与 CPU 版 FRSolver 接口一致（solve/step/compute_*_residual）
- 内部使用 GPUArrayManager 管理 GPU 数据
- 残差计算全部走 CuPy（gpu_inviscid.py / gpu_viscous.py）
- 时间积分走 GPUTimeIntegrator（gpu_time_integration.py）
- 支持 P0 和 P>=1 两种路径
- 支持单 GPU 稳态/伪稳态求解

使用:
    solver = GPUFRSolver(mesh, ops, order=2, device_id=0)
    result = solver.solve(max_iter=1000, dt=1e-4, tol=1e-6)
"""

import time
import numpy as np
from typing import Optional, Dict, Any
from loguru import logger

from autoflowcfd.core.gpu import gpu_available, get_cupy
from autoflowcfd.core.gpu.array_manager import GPUArrayManager
from autoflowcfd.core.gpu.gpu_time_integration import (
    GPUTimeIntegrator,
    enforce_positivity_gpu,
    compute_local_cfl_step_gpu,
)
from autoflowcfd.core.gpu.solver.gpu_solver_init import _GPUSolverInitMixin
from autoflowcfd.core.gpu.solver.gpu_solver_io import _GPUSolverIOMixin


class GPUFRSolver(_GPUSolverInitMixin, _GPUSolverIOMixin):
    """GPU 版 FR 求解器。

    与 CPU 版 FRSolver 接口一致，内部全程使用 CuPy 数组。
    网格数据和求解状态常驻 GPU 显存。

    Attributes:
        mesh: HighOrderMesh（CPU 侧引用，用于几何查询）
        ops: FROperators
        array_mgr: GPUArrayManager 实例
        time_integrator: GPUTimeIntegrator 实例
        U_gpu: 当前守恒变量（CuPy 数组，常驻 GPU）
        Q_gpu: 当前原始变量（CuPy 数组，常驻 GPU）
    """

    def __init__(
        self,
        mesh,
        ops,
        order: int = 2,
        n_vars: int = 5,
        device_id: int = 0,
        time_scheme: str = "ssp_rk3",
        cfl: float = 1.0,
        rho_inf: float = 1.225,
        vel_inf: float = 33.33,
        p_inf: float = 101325.0,
        mu_molecular: float = 1.8e-5,
        boundary_ghost_provider=None,
        bc_overrides=None,
        turb_model: str = "NONE",
        turbulence_intensity: float = 0.01,
        viscosity_ratio: float = 5.0,
    ):
        """初始化 GPU FRSolver。

        Args:
            mesh: HighOrderMesh 实例
            ops: FROperators 实例
            order: 多项式阶数
            n_vars: 守恒变量数
            device_id: GPU 设备 ID
            time_scheme: 时间积分方案
            cfl: CFL 数
            rho_inf, vel_inf, p_inf: 自由来流条件
            mu_molecular: 分子动力粘度
            boundary_ghost_provider: 边界幽灵态提供者。None 时按
                CPU 版 FRSolver 同一套逻辑（fr_solver/boundary.py::
                build_boundary_ghost_provider）自行构建——真实 bug
                修复（V2.0 专家组盲审发现，2026-08-27）：此前这里
                恒为调用方传入的 None（CLI 从未构建过真实的），
                `compute_inviscid_residual_fr_gpu`/`compute_viscous_
                residual_fr_gpu` 拿到 None 后用 DefaultGhostProvider
                （镜像内部值）代替，等价于全部边界对残差不可见。
            bc_overrides: 按边界组名称覆盖 BC 类型/参数，透传给
                build_boundary_ghost_provider（与 CPU 版 FRSolver
                同名参数语义一致，见 fr_solver/solver.py 文档）
            turb_model: 湍流模型名称，支持 NONE/SST/DDES/IDDES/WMLES/LES
                （#7，V2.0 专家组盲审第四轮，2026-08-28，见本方法顶部
                文档"GPU 湍流模型支持范围"一节）。请求集合之外的值时
                显式拒绝，不会像此前那样静默退化成层流。

        Raises:
            NotImplementedError: turb_model 是本方法支持集合之外的值

        GPU 湍流模型支持范围（#7，V2.0 专家组盲审第四轮，2026-08-28）：
        NONE/SST/DDES/IDDES/WMLES/LES 均已实现（core/gpu/turbulence/
        gpu_turbulence_sst.py、gpu_turbulence_des.py、gpu_sgs.py、
        gpu_turbulence_wmles.py）。明确的、真实的既有限制（不是本次
        遗漏，是本次移植范围之外的独立大工作）：GPU 版 k/omega 输运
        （gpu_scalar_transport.py）虽已实现并接入，但没有
        `troubled_cell.py::suppress_residual_outliers` 那样的离群值抑制
        （只有 isfinite 归零这一道最后防线，见 gpu_scalar_transport.py
        模块文档）；GPU 侧整体仍未在真实 CUDA 硬件上执行验证过（本机
        无 CuPy），已用 numpy 替身对照 CPU 版逐位数值核对过所有新增
        公式，但真正的端到端 GPU 冒烟测试需要用户在有 GPU 的环境上补做。
        """
        if not gpu_available:
            raise RuntimeError(
                "CuPy is not available. Install with: pip install cupy-cuda12x"
            )
        _SUPPORTED_TURB_MODELS = ("NONE", "SST", "DDES", "IDDES", "WMLES", "LES")
        if turb_model is not None and str(turb_model).upper() not in _SUPPORTED_TURB_MODELS:
            raise NotImplementedError(
                f"GPUFRSolver（--backend gpu）目前支持 turbulence_model="
                f"{[m.lower() for m in _SUPPORTED_TURB_MODELS]}，收到的是 '{turb_model}'。"
            )
        turb_model_upper = str(turb_model).upper() if turb_model is not None else "NONE"

        self.mesh = mesh
        self.ops = ops
        self.order = order
        self.n_vars = n_vars
        self.device_id = device_id
        self.mu_molecular = mu_molecular
        # mach_ref：与 CPU 版 FRSolver.__init__（fr_solver/solver.py）
        # 同一套计算方式/同一个用途，见该文件对应注释。物理下限钳制同样与
        # CPU 版镜像同步（2026-08-26，P2 发散专项）：低于 0.1 的参考马赫数会让
        # AUSM+up Mp 压差扩散项的 1/mach_ref² 放大压倒显式推进稳定性，
        # 完整推导/实证标定记录见 fr_solver/solver.py::_MACH_REF_FLOOR。
        mach_ref = vel_inf / np.sqrt(max(1.4 * p_inf / max(rho_inf, 1e-10), 1e-10))
        mach_ref = max(mach_ref, 0.1)
        self.freestream = {"rho_inf": rho_inf, "vel_inf": vel_inf, "p_inf": p_inf, "mach_ref": mach_ref}
        self.turb_model_name = turb_model
        self.turb_model_gpu = None  # GPU SST 模型（SST/DDES/IDDES 共用，可选）
        self.ddes_model_gpu = None  # GPU DDES/IDDES 长度尺度计算器（可选）
        self.sgs_model_gpu = None  # GPU WALE 亚格子模型（WMLES/LES 共用，可选）
        # WMLES 激活时构造真实的 CPU 版 WMLESModel 实例（与 CPU 版
        # fr_solver_turbulence.py::init_turbulence_models 完全同一个类，
        # 摩擦速度迭代求解本身就是纯 numpy、不需要也不该有 GPU 版——见
        # gpu_turbulence_wmles.py 模块文档"为什么不重新实现"一节）。
        # 必须在下面 _build_boundary_ghost_provider 之前构造：
        # build_boundary_ghost_provider 用 getattr(solver,"wmles_model",
        # None) 判断 WALL 边界是否要切换成真滑移 ghost 态（is_no_slip=
        # False），见 fr_solver/boundary.py 文档。
        self.wmles_model = None
        if turb_model_upper == "WMLES":
            from autoflowcfd.core.turbulence.wmles import WMLESModel
            self.wmles_model = WMLESModel(nu=mu_molecular / max(rho_inf, 1e-10))
        self.bc_overrides = bc_overrides or {}
        self.boundary_ghost_provider = (
            boundary_ghost_provider if boundary_ghost_provider is not None
            else self._build_boundary_ghost_provider(self.bc_overrides)
        )

        # GPU 数组管理器
        self.array_mgr = GPUArrayManager(device_id=device_id)

        # 上传网格数据
        self.mesh_data = self.array_mgr.upload_mesh_data(mesh, ops)
        self.ops_data = {k: v for k, v in self.mesh_data.items()}

        # 上传面几何
        self.flat_face_gpu = None
        self._init_face_geometry()

        # 时间积分器
        self.time_integrator = GPUTimeIntegrator(scheme=time_scheme, cfl=cfl)

        # 初始化求解状态
        n_cells = mesh.n_cells
        n_sps = mesh.n_sps_per_cell

        cp = get_cupy()
        with cp.cuda.Device(device_id):
            # 均匀初场
            self.U_gpu = cp.zeros((n_cells, n_sps, n_vars), dtype=cp.float64)
            self.U_gpu[:, :, 0] = rho_inf
            self.U_gpu[:, :, 1] = rho_inf * vel_inf
            self.U_gpu[:, :, 4] = p_inf / (1.4 - 1.0) + 0.5 * rho_inf * vel_inf**2

            self.Q_gpu = cp.zeros((n_cells, n_sps, 5), dtype=cp.float64)
            self._update_primitives_gpu()

        # 初始化 GPU 湍流模型（#7：SST/DDES/IDDES 共用同一个 GPUTurbulenceSST
        # 源项 ODE，DDES/IDDES 只是额外提供一个 des_length_scale 替换掉
        # SST 内部的 RANS 耗散长度尺度，见 gpu_turbulence_des.py 模块文档；
        # WMLES/LES 不构造 SST，只构造 GPUWALEModel，与 CPU 版
        # fr_solver_turbulence.py::init_turbulence_models 分支结构一致）。
        if turb_model_upper in ("SST", "DDES", "IDDES"):
            from autoflowcfd.core.gpu.turbulence.gpu_turbulence_sst import GPUTurbulenceSST
            # 从 Tu/VR 推导物理自洽的 k/omega 初值（与 CPU 版一致）
            nu = mu_molecular / max(rho_inf, 1e-10)
            k_inf = 1.5 * (vel_inf * turbulence_intensity) ** 2
            nu_t_inf = viscosity_ratio * nu
            omega_inf = k_inf / max(nu_t_inf, 1e-30)
            self.turb_model_gpu = GPUTurbulenceSST(
                n_cells, n_sps, device_id, k_inf=k_inf, omega_inf=omega_inf
            )
            # 物理上界与 CPU 同一公式（fr_solver/turbulence.py::_set_turbulence_bounds）：
            # k_max = 0.5·vel_inf²（湍动能 ≤ 平均流动能）、omega_max = 1e6。
            # 2026-08-25 代码审查前 GPU 侧恒为 1e6，与 CPU 不一致。
            self.turb_model_gpu.k_max = 0.5 * vel_inf ** 2
            self.turb_model_gpu.omega_max = 1e6
            logger.info(f"GPU SST k-omega model initialized on device {device_id} "
                       f"(k_inf={k_inf:.4e}, omega_inf={omega_inf:.4e})")
            print(f"   [OK] GPU SST k-omega model initialized "
                  f"(k_inf={k_inf:.4e}, omega_inf={omega_inf:.4e})")

            if turb_model_upper == "DDES":
                from autoflowcfd.core.gpu.turbulence.gpu_turbulence_des import GPUDDESModel
                self.ddes_model_gpu = GPUDDESModel()
                print(f"   [OK] GPU DDES model initialized (based on SST)")
            elif turb_model_upper == "IDDES":
                from autoflowcfd.core.gpu.turbulence.gpu_turbulence_des import GPUIDDESModel
                from autoflowcfd.core.turbulence.des import compute_h_max_and_h_wn
                self.ddes_model_gpu = GPUIDDESModel()
                # h_max/h_wn 只依赖网格几何，与 CPU 版
                # fr_solver_turbulence.py 同一个一次性缓存策略：CPU 算一次
                # （复用已验证的 quality_metrics 边长几何函数），结果上传
                # 常驻显存，不是每步都重算。
                h_max_cpu, h_wn_cpu = compute_h_max_and_h_wn(mesh)
                with cp.cuda.Device(device_id):
                    self._iddes_h_max_gpu = cp.asarray(h_max_cpu)
                    self._iddes_h_wn_gpu = cp.asarray(h_wn_cpu)
                print(f"   [OK] GPU IDDES model initialized (based on SST)")

        elif turb_model_upper == "WMLES":
            from autoflowcfd.core.gpu.turbulence.gpu_sgs import GPUWALEModel
            self.sgs_model_gpu = GPUWALEModel()
            print(f"   [OK] GPU WMLES model initialized (wall stress correction + WALE SGS)")

        elif turb_model_upper == "LES":
            from autoflowcfd.core.gpu.turbulence.gpu_sgs import GPUWALEModel
            self.sgs_model_gpu = GPUWALEModel()
            print(f"   [OK] GPU LES with WALE SGS model initialized")

        # 网格尺度 Delta = V^(1/3)（WALE/Smagorinsky 用，与 CPU 版
        # fr_solver/solver_geometry.py::_get_grid_scale 同一个公式）：
        # 只依赖单元体积几何，与流场无关，一次性算好缓存。
        self._grid_scale_gpu = None
        if self.sgs_model_gpu is not None:
            volumes_cpu = mesh.get_all_cell_volumes()
            delta_cpu = np.power(np.abs(volumes_cpu), 1.0 / 3.0)
            delta_cpu = np.tile(delta_cpu[:, np.newaxis], (1, n_sps))
            with cp.cuda.Device(device_id):
                self._grid_scale_gpu = cp.asarray(delta_cpu)

        # 预计算壁面距离（用于湍流模型）——与 CPU 版
        # fr_solver_turbulence.py 的 ["SST","DDES","IDDES","WMLES","LES"]
        # 同一个判据集合（WMLES 壁面剪应力修正/LES 近壁行为都需要它，
        # 即便 WALE 本身的公式不直接读 d_wall）。
        self.wall_distance_gpu = None
        if self.turb_model_gpu is not None or self.sgs_model_gpu is not None:
            self._init_wall_distance_gpu()

        # WALL 边界面拓扑掩码（#7，k/omega 输运 Dirichlet BC 用）：纯几何/
        # 边界分组查询，只依赖 mesh + boundary_ghost_provider，与流场状态
        # 无关，一次性算好缓存，不是每步重算（见 gpu_scalar_transport.py::
        # compute_wall_dirichlet_mask_gpu 文档）。只有 SST/DDES/IDDES 才
        # 会真正用到（k/omega 输运的 Dirichlet 面）。
        self._wall_mask_k_gpu = None
        if self.turb_model_gpu is not None:
            from autoflowcfd.core.gpu.turbulence.gpu_scalar_transport import compute_wall_dirichlet_mask_gpu
            wall_mask_np = compute_wall_dirichlet_mask_gpu(mesh, self.boundary_ghost_provider)
            with cp.cuda.Device(device_id):
                self._wall_mask_k_gpu = cp.asarray(wall_mask_np)

        # 初始化 GPU 模态滤波（抑制混叠噪声）
        self.filter_func_gpu = None
        self._init_modal_filter_gpu()

        # DUAL_TIME 专用：物理时间层 n-1 的解（BDF2 时间导数项需要）
        self._dual_time_U_prev = None

        # 残差范数历史
        self.residual_history = []
        self.iteration = 0

        logger.info(
            f"GPUFRSolver initialized: {n_cells} cells, P{order}, "
            f"device {device_id}, scheme={time_scheme}, CFL={cfl}"
        )
        print(f"✅ GPUFRSolver Ready:")
        print(f"   Cells: {n_cells}, Order: P{order}")
        print(f"   Device: {device_id} ({self.array_mgr._device_name})")
        print(f"   Time Scheme: {time_scheme}, CFL: {cfl}")


    def _build_boundary_ghost_provider(self, bc_overrides):
        """构建边界幽灵态提供者 (BD-01)，与 CPU 版 FRSolver 复用同一套
        构建逻辑（fr_solver/boundary.py::build_boundary_ghost_provider
        只依赖 self.mesh/self.freestream/self.turb_model_name/
        self.wmles_model，在这个调用点之前均已设置好）。"""
        from autoflowcfd.core.fr_solver import boundary as fr_solver_boundary
        return fr_solver_boundary.build_boundary_ghost_provider(self, bc_overrides)

    def _update_primitives_gpu(self):
        """GPU 上更新原始变量。"""
        from autoflowcfd.core.gpu.residual.gpu_flux import conserved_to_primitive_gpu
        self.Q_gpu = conserved_to_primitive_gpu(self.U_gpu[..., :5])

    def compute_inviscid_residual_gpu(self, U_trial=None):
        """GPU 计算无粘残差。

        Args:
            U_trial: CuPy 数组 (n_cells, n_sps, n_vars)，试验解（可选）

        Returns:
            residual: CuPy 数组 (n_cells, n_sps, 5)
        """
        cp = get_cupy()
        U = U_trial if U_trial is not None else self.U_gpu

        if self.mesh.n_points_1d == 1:
            # P0 路径：使用 CuPy RawKernel
            from autoflowcfd.core.gpu.residual.gpu_p0_inviscid import (
                compute_inviscid_residual_p0_cupy_gpu_resident,
            )
            Q_flat = self.Q_gpu[:, 0, :5].copy()
            # 需要面连接关系数据
            fc = self.mesh.face_connectivity
            owner = cp.asarray(fc.owner_cell)
            neighbor = cp.asarray(
                np.where(fc.is_boundary, 0, fc.neighbor_cell)
            )
            is_bnd = cp.asarray(fc.is_boundary)
            # 面法向和面积（性能修复，真实复现：本函数是 GPU P0 无粘残差
            # 每步都要调用的热路径，此前这里逐面 `ffp_list[f]` 索引——自
            # face_flux_points_merge.py 的 flat array 重构后
            # `mesh.face_flux_points` 是 `_KernelFaceData`，`[f]` 会按需
            # *构造*一个完整 FaceFluxPointGeometry 对象，187 万面级别的
            # 网格上每步都这样做是灾难级开销，与
            # core/fr_operators/troubled_cell.py::precompute_cell_face_
            # misalignment 是同一类遗漏、同一次真实复现中一并发现。改用
            # inviscid_p0.py 里已经过测试、CPU P0 热路径同样在用的快速
            # 路径提取函数，数学上是同一批数据，只是不再逐面构造对象。
            n_faces = fc.n_faces
            from autoflowcfd.core.fr_residual.inviscid_p0 import _extract_p0_face_geometry
            normal, area_w = _extract_p0_face_geometry(self.mesh.face_flux_points, fc, n_faces)
            normal_gpu = cp.asarray(normal)
            area_w_gpu = cp.asarray(area_w)
            volumes_gpu = self.mesh_data.get('cell_volumes')
            if volumes_gpu is None:
                volumes_gpu = cp.asarray(self.mesh.get_all_cell_volumes())

            # 边界幽灵态（正确性修复，B-8 一并暴露）：此前这里恒传全零，
            # kernel 会把边界面外部状态当成零态解 AUSM 黎曼问题。改为与
            # gpu_p0_inviscid.py::compute_inviscid_residual_p0_cupy 同款的
            # 逐面 ghost_provider 预计算；范围含混合拆分面的边界子面记录（B-8）。
            from autoflowcfd.core.fr_residual.inviscid import DefaultGhostProvider
            ghost_provider = (
                self.boundary_ghost_provider
                if self.boundary_ghost_provider is not None
                else DefaultGhostProvider()
            )
            ffp_list = self.mesh.face_flux_points
            mixed_bnd_face = getattr(ffp_list, "mixed_bnd_face", None)
            if mixed_bnd_face is None:
                mixed_bnd_face = np.zeros(n_faces, dtype=np.bool_)
            mixed_bnd_frac = getattr(ffp_list, "mixed_p0_bnd_frac", None)
            if mixed_bnd_frac is None:
                mixed_bnd_frac = np.zeros(n_faces, dtype=np.float64)
            ghost_faces = np.nonzero(fc.is_boundary | mixed_bnd_face)[0]
            Q_flat_np = cp.asnumpy(Q_flat)
            Q_ghost_np = np.zeros((n_faces, 5), dtype=np.float64)
            for f in ghost_faces:
                oc = int(fc.owner_cell[f])
                Q_ghost_np[f, :] = ghost_provider(
                    f, Q_flat_np[oc: oc + 1], normal[f: f + 1]
                )[0]
            Q_ghost = cp.asarray(Q_ghost_np)
            mixed_bnd_frac_gpu = cp.asarray(mixed_bnd_frac)

            res = compute_inviscid_residual_p0_cupy_gpu_resident(
                Q_flat, owner, neighbor, is_bnd,
                normal_gpu, area_w_gpu, volumes_gpu,
                Q_ghost, mixed_bnd_frac_gpu, self.mesh.n_cells, n_faces,
                mach_ref=self.freestream["mach_ref"],
            )
            # 扩展到 (n_cells, n_sps, 5)
            return cp.broadcast_to(res, (self.mesh.n_cells, self.mesh.n_sps_per_cell, 5)).copy()
        else:
            # P>=1 高阶 FR GPU 路径
            from autoflowcfd.core.gpu.residual.gpu_inviscid import compute_inviscid_residual_fr_gpu
            return compute_inviscid_residual_fr_gpu(
                U, self.mesh, self.ops,
                boundary_ghost_provider=self.boundary_ghost_provider,
                mesh_data=self.mesh_data,
                ops_data=self.ops_data,
                flat_face_gpu=self.flat_face_gpu,
                device_id=self.device_id,
                mach_ref=self.freestream["mach_ref"],
            )

    def compute_viscous_residual_gpu(self, U_trial=None, mu_t_field=None):
        """GPU 计算粘性残差。

        Args:
            U_trial: CuPy 数组（可选）
            mu_t_field: 湍流涡粘度 rho*nu_t (n_cells, n_sps) CuPy 数组（可选）

        Returns:
            viscous_residual: CuPy 数组 (n_cells, n_sps, 5)
        """
        from autoflowcfd.core.gpu.residual.gpu_viscous import compute_viscous_residual_fr_gpu
        U = U_trial if U_trial is not None else self.U_gpu
        res = compute_viscous_residual_fr_gpu(
            U, self.mesh, self.ops,
            mu=self.mu_molecular,
            mu_t_field=mu_t_field,
            boundary_ghost_provider=self.boundary_ghost_provider,
            mesh_data=self.mesh_data,
            ops_data=self.ops_data,
            flat_face_gpu=self.flat_face_gpu,
            device_id=self.device_id,
        )

        # WMLES 壁面剪应力修正（#7）：与 CPU 版
        # `FRSolver.compute_viscous_residual` 同一个调用点——必须叠加在
        # 这里（残差组装阶段），不能等 step() 状态更新之后才生效，见
        # gpu_turbulence_wmles.py 模块文档。
        if self.wmles_model is not None:
            from autoflowcfd.core.gpu.turbulence.gpu_turbulence_wmles import (
                compute_wmles_wall_stress_correction_gpu,
            )
            wall_stress_correction = compute_wmles_wall_stress_correction_gpu(self, U=U)
            if wall_stress_correction is not None:
                res = res + wall_stress_correction[..., : res.shape[-1]]

        return res

    def _compute_local_time_step_gpu(self):
        """GPU 计算局部 CFL 时间步长（使用所有 SP 的谱半径）。"""
        cp = get_cupy()
        n_cells = self.mesh.n_cells
        n_sps = self.mesh.n_sps_per_cell

        fc = self.mesh.face_connectivity
        n_faces = fc.n_faces

        owner_cell = cp.asarray(fc.owner_cell)
        neighbor_cell = cp.asarray(
            np.where(fc.is_boundary, 0, fc.neighbor_cell)
        )
        is_boundary = cp.asarray(fc.is_boundary)

        # 面法向和面积：与 compute_inviscid_residual_gpu 同一类每步热路径
        # 性能修复，理由/验证方式相同（见该方法文档）——本方法是每步都要
        # 调用的局部 CFL 步长计算，逐面对象构造的开销在这里同样是每步
        # 复现，不是一次性成本。
        from autoflowcfd.core.fr_residual.inviscid_p0 import _extract_p0_face_geometry
        normal, area_w = _extract_p0_face_geometry(self.mesh.face_flux_points, fc, n_faces)
        normals_gpu = cp.asarray(normal)
        areas_gpu = cp.asarray(area_w)

        cell_volumes = self.mesh_data.get('cell_volumes')
        if cell_volumes is None:
            cell_volumes = cp.asarray(self.mesh.get_all_cell_volumes())

        # 几何/度量 CFL 限制所需数据（与 CPU 侧 cfl.py 的 dt_geometric
        # 同一机制，见 compute_local_cfl_step_gpu 参数文档）：det_jacs 已
        # 在 upload_mesh_data 里常驻显存，metric_flux_scale 只依赖网格
        # 几何（与流场状态无关），缓存后避免每步重复计算——与 CPU 侧
        # solver_geometry.py::_get_metric_flux_scale 同一缓存策略。
        det_jacs_gpu = self.mesh_data.get('det_jacs')
        adj_j_gpu = self.mesh_data.get('adj_j')
        metric_flux_scale_gpu = getattr(self, '_metric_flux_scale_gpu_cache', None)
        # 第四次评审第二轮复核发现：只用 `is None` 判断缓存是否有效，
        # 完全没有形状比较——CPU 侧 solver_geometry.py::_get_metric_flux_scale
        # 曾因"只比较 shape[0]、漏比 n_sps 维度"复现过跨阶数切换后返回
        # 陈旧形状缓存值的真实 bug（Order Continuation 切换阶数后 n_sps
        # 改变），这里连 shape[0] 都没比，是同一类问题的更宽松版本。
        # 当前 GPU 路径还没有接入 Order Continuation（CLI 直接以目标阶数
        # 一次性构造 GPUFRSolver），这个缺陷现在不会被触发，但保持与
        # CPU 侧同等的防御水位，不留一个"看起来复制了修复、实际没复制
        # 关键部分"的陷阱。
        if (metric_flux_scale_gpu is None or metric_flux_scale_gpu.shape != (n_cells, n_sps)) \
                and adj_j_gpu is not None:
            adj_row_norms = cp.linalg.norm(adj_j_gpu, axis=-1)  # (n_cells,n_sps,3)
            metric_flux_scale_gpu = cp.sum(adj_row_norms, axis=-1)  # (n_cells,n_sps)
            self._metric_flux_scale_gpu_cache = metric_flux_scale_gpu

        # 使用所有 SP 计算谱半径（取最大值），而非仅 SP0
        # 对每个 SP 独立计算 CFL 步长，然后取 cell 内最小值
        dt_all_sps = cp.zeros((n_cells, n_sps), dtype=cp.float64)
        for sp in range(n_sps):
            U_sp = self.U_gpu[:, sp:sp+1, :]  # (n_cells, 1, n_vars)
            det_jacs_sp = det_jacs_gpu[:, sp] if det_jacs_gpu is not None else None
            metric_flux_scale_sp = (
                metric_flux_scale_gpu[:, sp] if metric_flux_scale_gpu is not None else None
            )
            dt_sp = compute_local_cfl_step_gpu(
                U_sp, cell_volumes,
                owner_cell, neighbor_cell, is_boundary,
                normals_gpu, areas_gpu,
                None, None,
                cfl=self.time_integrator.cfl,
                poly_order=getattr(self, "order", 0),
                det_jacs_sp=det_jacs_sp,
                metric_flux_scale_sp=metric_flux_scale_sp,
            )
            dt_all_sps[:, sp] = dt_sp

        return cp.min(dt_all_sps, axis=1)  # (n_cells,)

    def step(self, dt: float = 0.0) -> float:
        """执行一个时间步。

        完整流程（与 CPU 版 step() 对应）：
        1. 更新原始变量
        2. 湍流源项求值（算子分裂：湍流走独立显式更新）
        3. 局部 CFL 步长
        4. 平均流残差计算（含湍流涡粘耦合）
        5. SSP-RK / IMEX / DUAL_TIME 时间推进
        6. 湍流场更新（k/ω 正性限制）

        Args:
            dt: 物理时间步长（稳态模式下被局部 CFL 步长覆盖，
                DUAL_TIME 模式下是真正的物理时间步长）

        Returns:
            residual_norm: 残差范数
        """
        cp = get_cupy()
        n_cells = self.mesh.n_cells
        n_sps = self.mesh.n_sps_per_cell

        self._update_primitives_gpu()

        # 湍流源项在当前状态下求值（算子分裂）
        mu_t_field = self.compute_turbulence_source_gpu()

        # 局部 CFL 步长
        dt_local = self._compute_local_time_step_gpu()
        dt_local_full = cp.broadcast_to(
            dt_local[:, None], (n_cells, n_sps)
        ).reshape(n_cells * n_sps)

        # 展平 U 用于时间积分器
        U_flat = self.U_gpu.reshape(n_cells * n_sps, self.n_vars)

        # 构建平均流残差函数（含湍流涡粘耦合）
        def mean_flow_residual(U_flat_trial):
            U_trial = U_flat_trial.reshape(n_cells, n_sps, self.n_vars)
            inv_res = self.compute_inviscid_residual_gpu(U_trial)
            visc_res = self.compute_viscous_residual_gpu(U_trial, mu_t_field=mu_t_field)
            total = inv_res + visc_res
            return -total.reshape(n_cells * n_sps, self.n_vars)

        # 初始残差
        residual0 = mean_flow_residual(U_flat)

        # 根据时间方案选择推进方式
        scheme = self.time_integrator.scheme

        if scheme == "dual_time":
            # DUAL_TIME: 真正时间精度的物理时间推进
            if self._dual_time_U_prev is None:
                # 第一个物理步：BDF1
                U_new_flat = self.time_integrator.step_dual_time(
                    U_flat, mean_flow_residual, dt_local_full,
                    dt_physical=dt,
                    solution_prev=None,
                    max_inner_iter=self.time_integrator.dual_time_steps if hasattr(self.time_integrator, 'dual_time_steps') else 5,
                    filter_func=self.filter_func_gpu,
                )
            else:
                # 后续物理步：BDF2
                U_new_flat = self.time_integrator.step_dual_time(
                    U_flat, mean_flow_residual, dt_local_full,
                    dt_physical=dt,
                    solution_prev=self._dual_time_U_prev,
                    max_inner_iter=self.time_integrator.dual_time_steps if hasattr(self.time_integrator, 'dual_time_steps') else 5,
                    filter_func=self.filter_func_gpu,
                )
            # 保存当前解作为下一步的 prev
            self._dual_time_U_prev = U_flat.copy()

        elif scheme == "imex_euler":
            # IMEX: 显式处理对流，隐式处理粘性
            def convective_residual_only(U_flat_trial):
                U_trial = U_flat_trial.reshape(n_cells, n_sps, self.n_vars)
                inv_res = self.compute_inviscid_residual_gpu(U_trial)
                return -inv_res.reshape(n_cells * n_sps, self.n_vars)

            def diffusive_residual_only(U_flat_trial):
                U_trial = U_flat_trial.reshape(n_cells, n_sps, self.n_vars)
                visc_res = self.compute_viscous_residual_gpu(U_trial, mu_t_field=mu_t_field)
                return -visc_res.reshape(n_cells * n_sps, self.n_vars)

            U_new_flat = self.time_integrator.step_imex(
                U_flat, convective_residual_only, diffusive_residual_only,
                dt_local_full, p_floor=1.0,
            )

        else:
            # SSP-RK2/RK3 or Forward Euler
            U_new_flat = self.time_integrator.step(
                U_flat, mean_flow_residual, dt_local_full,
                p_floor=1.0, residual0=residual0,
                filter_func=self.filter_func_gpu,
            )

        self.U_gpu = U_new_flat.reshape(n_cells, n_sps, self.n_vars)
        self._update_primitives_gpu()

        # SGS（WALE）涡粘系数更新（#7）：必须在状态更新之后调用，供
        # 下一步的粘性残差消费，与 CPU 版 apply_turbulence_corrections
        # 同一个操作分裂时序，见 gpu_solver_io.py::
        # _apply_turbulence_corrections_gpu 文档。
        self._apply_turbulence_corrections_gpu()

        # 残差范数
        residual_norm = float(cp.linalg.norm(residual0) / max(1, np.sqrt(residual0.size)))
        self.residual_history.append(residual_norm)
        self.iteration += 1

        return residual_norm

    def solve(
        self,
        max_iter: int = 1000,
        dt: float = 1e-4,
        tol: float = 1e-6,
        output_interval: int = 10,
    ) -> Dict[str, Any]:
        """执行稳态求解循环。

        Args:
            max_iter: 最大迭代次数
            dt: 时间步长（稳态模式下被 CFL 覆盖）
            tol: 收敛容差
            output_interval: 输出间隔

        Returns:
            结果字典
        """
        print(f"Starting GPU solve: max_iter={max_iter}, tol={tol}")
        converged = False
        final_residual = 1e10

        for i in range(max_iter):
            t_start = time.time()
            res = self.step(dt)
            t_end = time.time()
            final_residual = res

            if i == 0 or (i + 1) % output_interval == 0:
                mem = self.array_mgr.get_memory_usage()
                print(
                    f"GPU Iter {i+1}: Residual = {res:.6e} | "
                    f"Time/step: {t_end-t_start:.3f}s | "
                    f"GPU mem: {mem['used_mb']:.0f}/{mem['total_mb']:.0f} MB"
                )

            if res < tol:
                converged = True
                print(f"✅ GPU Converged at iteration {i+1} with residual {res:.6e}")
                break

            if not np.isfinite(res):
                print(f"❌ GPU Diverged at iteration {i+1} with residual {res}")
                break

        return {
            'converged': converged,
            'iterations': self.iteration,
            'final_residual': final_residual,
            'residual_history': self.residual_history,
        }

