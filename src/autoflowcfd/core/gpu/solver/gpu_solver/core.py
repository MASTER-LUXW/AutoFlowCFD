"""AutoFlowCFD V2.0 - GPUFRSolver 主类：构造与阶数切换

从 `src/autoflowcfd/core/gpu/solver/gpu_solver.py` 拆出（2026-09-24）。方法按职责分到同目录的 mixin 里，
这里只留构造与对外接口。
"""

import numpy as np
from typing import Optional
from loguru import logger
from autoflowcfd.core.gpu import gpu_available, get_cupy
from autoflowcfd.core.gpu.array_manager import GPUArrayManager
from autoflowcfd.core.gpu.gpu_time_integration import GPUTimeIntegrator
from autoflowcfd.core.gpu.solver.gpu_solver_init import _GPUSolverInitMixin
from autoflowcfd.core.gpu.solver.gpu_solver_io import _GPUSolverIOMixin
from .residual import _GPUSolverResidualMixin
from .timestep import _GPUSolverTimeStepMixin
from .step import _GPUSolverStepMixin


class GPUFRSolver(_GPUSolverResidualMixin, _GPUSolverTimeStepMixin, _GPUSolverStepMixin, _GPUSolverInitMixin, _GPUSolverIOMixin):
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
        rho_inf: float = 1.225,
        vel_inf: float = 33.33,
        p_inf: float = 101325.0,
        # 攻角/侧滑角（度）。0/0 时来流严格沿 +x，与此前把方向硬编码
        # 成 +x 的行为逐位相同。约定见 core/utils/flow_direction.py。
        aoa_deg: float = 0.0,
        aos_deg: float = 0.0,
        mu_molecular: float = 1.8e-5,
        boundary_ghost_provider=None,
        bc_overrides=None,
        turb_model: str = "NONE",
        turbulence_intensity: float = 0.01,
        viscosity_ratio: float = 5.0,
        low_mach_precond: bool = True,
        cfl_start: Optional[float] = None,
        cfl_max: Optional[float] = None,
        cfl_min: Optional[float] = None,
    ):
        """初始化 GPU FRSolver。

        Args:
            mesh: HighOrderMesh 实例
            ops: FROperators 实例
            order: 多项式阶数
            n_vars: 守恒变量数
            device_id: GPU 设备 ID
            time_scheme: 时间积分方案
            rho_inf, vel_inf, p_inf: 自由来流条件
            mu_molecular: 分子动力粘度
            low_mach_precond: 是否启用低马赫数伪时间预处理（默认 True，
                与 CPU 版一致）。环境变量 AFCFD_LOW_MACH_PRECOND=0/1
                优先于本参数。
            cfl_start, cfl_max, cfl_min: 自适应 CFL 参数（与 CPU 版 FRSolver
                同名参数、同一语义；None 取控制器默认值，规则见
                `adaptive_cfl/policy.py::build_cfl_policy`）。
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
        （gpu_scalar_transport.py）只有 isfinite 归零这一道防线 ——
        **这条此前记作"缺少 suppress_residual_outliers 那样的离群值
        抑制"，2026-09-19 起不再是缺口**：机制3 已整体删除（真实网格
        消融对照证明它无效，见 fr_residual/inviscid.py），CPU 侧现在
        同样只有 isfinite 归零，两侧对称。
        GPU 侧整体仍未在真实 CUDA 硬件上执行验证过（本机
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
        # Order Continuation 支持（2026-09-02，见 core/gpu/solver/
        # gpu_solver_order_continuation.py 模块文档）——与 CPU/GPU
        # 分布式版本同一约定：`self.order`（目标阶数）/
        # `self.current_order`（当前实际所在阶数）。
        self.current_order = order
        self.order_continuation_enabled = True
        self.n_vars = n_vars
        self.device_id = device_id
        self.mu_molecular = mu_molecular
        # mach_ref 的唯一来源（按 AUSM+up 预处理档钳下限）。此前这里手写一份、
        # 下限硬编码 0.1，而 CPU 自 2026-09-17 起在默认档用 0.05 —— 同一算例
        # 在 GPU 与 CPU 上拿到不同的 mach_ref（plate_demo：0.1 vs 0.0882）。
        from autoflowcfd.core.fr_solver.mach_ref import resolve_mach_ref
        mach_ref = resolve_mach_ref(rho_inf, vel_inf, p_inf)
        # aoa_deg/aos_deg（2026-09-17）：下游的 Q_free / SEM 入口方向 /
        # 气动力风轴系分解都从 freestream 字典读，GPU 这条路径同样要带上，
        # 否则同一组 CLI 参数在 --backend gpu 上会静默退回零攻角。
        self.freestream = {"rho_inf": rho_inf, "vel_inf": vel_inf, "p_inf": p_inf,
                           "mach_ref": mach_ref,
                           "aoa_deg": float(aoa_deg), "aos_deg": float(aos_deg)}

        # 低马赫数伪时间预处理（2026-09-14，与 CPU 版 FRSolver 同一机制/
        # 同一开关语义）：dt 按预处理波速放大**必须**与把 Gamma 作用到
        # 平均流残差上成对出现，缺一半就是 2026-08-25 在本文件
        # compute_local_cfl_step_gpu 里记录的那次失稳。完整推导见
        # core/utils/preconditioning.py 模块末尾；GPU 侧实现见
        # core/gpu/gpu_preconditioning.py。
        # 只在 SSP-RK2/RK3 下启用：DUAL_TIME 的物理时间导数项与 IMEX 的
        # 残差拆分都需要单独推导 Gamma 的分配方式，不套未经验证的近似
        # （与 CPU 侧同一判据）。
        from autoflowcfd.core.utils.preconditioning import resolve_low_mach_precond
        self.low_mach_precond_enabled = resolve_low_mach_precond(low_mach_precond, time_scheme)
        # 真实 bug 修复（2026-09-05，代码复审发现）：CPU 版
        # `DistributedFRSolver`/`MultiGPUDistributedSolver` 都把构造期
        # 传入的 `turbulence_intensity`/`viscosity_ratio` 存成
        # `self._turbulence_intensity`/`self._viscosity_ratio` 供后续
        # `_set_freestream_turbulence(solver)`（P0 降阶重置/resume 爆炸
        # 重置等场景都会调用）读取真实配置值；单机 `GPUFRSolver` 此前
        # 完全没有存这两个属性——虽然构造函数确实接收了这两个参数（见
        # 下面 k_inf/omega_inf 推导），但只用于构造期算一次初值，从不
        # 存到 self 上。后续任何需要重新推导来流湍流值的调用点（例如
        # `gpu_solver_order_continuation.py::gpu_solver_interpolate_to_
        # new_order` 的降阶重置分支、本次新增的 `distributed_order_
        # continuation.py::_reset_turbulence_if_resumed_field_exploded`
        # resume 安全重置）都会因为 `getattr(solver, '_turbulence_
        # intensity', 0.01)` 取不到真实值，静默退回默认值 Tu=0.01/
        # VR=5.0——用户配置了非默认湍流强度/粘性比的单机 GPU 算例，这些
        # 场景会重置成错误的来流湍流值而不报错。
        self._turbulence_intensity = turbulence_intensity
        self._viscosity_ratio = viscosity_ratio
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
        self.time_integrator = GPUTimeIntegrator(scheme=time_scheme)

        # CFL 策略（控制器 or 固定 CFL）的唯一事实来源：
        # `time_integration/adaptive_cfl/policy.py::build_cfl_policy`（六个后端
        # 构造点此前各写一份且已分叉，见该模块文档）。
        # 此前这里对 IMEX/DUAL_TIME 不建控制器、退回构造参数 `cfl`（默认 1.0，
        # 远超 P>=1 显式稳定极限），并对 cfl_start/cfl_max 留着"缺省退回 cfl"
        # 的硬编码兜底。
        from autoflowcfd.core.time_integration.adaptive_cfl.policy import build_cfl_policy
        self._cfl_controller, self.fixed_cfl_number = build_cfl_policy(
            time_scheme, cfl_start=cfl_start, cfl_max=cfl_max, cfl_min=cfl_min)

        # 初始化求解状态
        n_cells = mesh.n_cells
        n_sps = mesh.n_sps_per_cell

        cp = get_cupy()
        with cp.cuda.Device(device_id):
            # 均匀初场
            from autoflowcfd.core.utils.flow_direction import (
                freestream_conservative_state,
            )

            # 速度方向必须取自 aoa/aos（2026-09-24 修复）：此前这里写死 (vel_inf, 0, 0)，
            # 而边界 Q_free 用的是正确方向，`--aoa` 非零时初场与边界不一致。
            # 8 处同类写法已统一到 `freestream_conservative_state`（见其文档）。
            self.U_gpu = cp.empty((n_cells, n_sps, n_vars), dtype=cp.float64)
            self.U_gpu[:] = cp.asarray(
                freestream_conservative_state(self.freestream, n_vars))

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
                from autoflowcfd.core.turbulence.des import compute_h_max_and_h_wn
                self.ddes_model_gpu = GPUDDESModel()
                # h_max（2026-09-02 补齐，与下面 IDDES 分支同一处几何量、
                # 同一个一次性缓存策略）：`apply_to_sst_model_gpu` 现在
                # 优先用各向异性感知的 max_edge 网格尺度而不是
                # cube_root(V)，见 CPU 版 des.py::DDESModel.compute_
                # grid_scale 文档"Note"一节——本项目高度依赖棱柱边界层
                # 网格，cube_root 会系统性低估扁平单元的 Δ。只需要
                # h_max（第一个返回值），h_wn 是 IDDES 专属几何量。
                h_max_cpu, _ = compute_h_max_and_h_wn(mesh)
                with cp.cuda.Device(device_id):
                    self._iddes_h_max_gpu = cp.asarray(h_max_cpu)
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
        # compute_turbulence_face_masks_gpu 文档；同时缓存 k/omega 来流条件用的开放边界掩码）。只有 SST/DDES/IDDES 才
        # 会真正用到（k/omega 输运的 Dirichlet 面）。
        self._wall_mask_k_gpu = None
        self._open_mask_gpu = None
        if self.turb_model_gpu is not None:
            from autoflowcfd.core.gpu.turbulence.gpu_scalar_transport import compute_turbulence_face_masks_gpu
            wall_mask_np, open_mask_np = compute_turbulence_face_masks_gpu(mesh, self.boundary_ghost_provider)
            with cp.cuda.Device(device_id):
                self._wall_mask_k_gpu = cp.asarray(wall_mask_np)
                self._open_mask_gpu = cp.asarray(open_mask_np)

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
            f"device {device_id}, scheme={time_scheme}, CFL={self._current_cfl():g}"
        )
        print(f"✅ GPUFRSolver Ready:")
        print(f"   Cells: {n_cells}, Order: P{order}")
        print(f"   Device: {device_id} ({self.array_mgr._device_name})")
        print(f"   Time Scheme: {time_scheme}")
        from autoflowcfd.core.time_integration.adaptive_cfl.policy import describe_cfl_policy
        print("   " + describe_cfl_policy(self._cfl_controller, self.fixed_cfl_number))
        # 与 CPU 版 FRSolver 的启动日志对等（2026-09-16）：影响物理/数值
        # 的开关必须在日志里留痕，否则"这份日志是哪个配置跑出来的"只能
        # 靠翻进程命令行考古。GPU 侧此前完全没有这几行，而 CPU/GPU 交叉
        # 一致性对照恰恰要求两边确认取到同一档。
        from autoflowcfd.core.fr_operators.kernels import (
            ausm_precond_mode_label as _pm_label,
            resolve_ausm_precond_mode as _pm_resolve,
        )
        from autoflowcfd.fr.modal_filter import FILTER_MODE as _fm
        print(f"   Modal filter mode: {_fm}")
        print(f"   AUSM+up precond mode: {_pm_label(_pm_resolve())}")
        # troubled-cell 门控判据（只在 FILTER_MODE=sensor 档生效，
        # 两个维度刻意独立以便受控 A/B）。它决定 P>=1 时"对哪些
        # 单元施加模态滤波"，是与滤波档同等量级的物理开关。
        from autoflowcfd.core.fr_operators.bounds_sensor import (
            resolve_troubled_sensor as _ts_resolve,
        )
        print(f"   Troubled-cell sensor: {_ts_resolve()}")

    def _build_boundary_ghost_provider(self, bc_overrides):
        """构建边界幽灵态提供者 (BD-01)，与 CPU 版 FRSolver 复用同一套
        构建逻辑（fr_solver/boundary.py::build_boundary_ghost_provider
        只依赖 self.mesh/self.freestream/self.turb_model_name/
        self.wmles_model，在这个调用点之前均已设置好）。"""
        from autoflowcfd.core.fr_solver import boundary as fr_solver_boundary
        return fr_solver_boundary.build_boundary_ghost_provider(self, bc_overrides)

    def _interpolate_to_new_order(self, target_p: int) -> None:
        """阶数切换（2026-09-02，见 core/gpu/solver/gpu_solver_order_
        continuation.py 模块文档）——与 CPU/GPU 分布式版本同一个命名/
        调用约定。"""
        from autoflowcfd.core.gpu.solver.gpu_solver_order_continuation import (
            gpu_solver_interpolate_to_new_order,
        )
        gpu_solver_interpolate_to_new_order(self, target_p)

        # 自适应 CFL 控制器必须在阶数切换时复位（2026-09-14，与 CPU 侧
        # order_continuation.py 里 `_cfl_ctrl.reset()` 同一理由）：阶数变化
        # 会让残差发生一次跳变（插值误差），那不是"解在恶化"，不应触发
        # CFL 收缩。
        # **这条对单机 GPU 是必需的、不能照抄分布式路径的"不用管"**：
        # CPU MPI 分布式在阶数切换时把 `_local_solver` 置 None、下次访问
        # 重新构造一个全新 FRSolver（连带全新控制器），相当于免费拿到了
        # 复位；而这里 `gpu_solver_interpolate_to_new_order` 是**原地**改
        # mesh/ops/GPU 常驻数组，`self._cfl_controller` 会跨阶数存活下来。
        if getattr(self, "_cfl_controller", None) is not None:
            self._cfl_controller.reset()
