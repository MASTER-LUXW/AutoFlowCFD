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

import os
import time
import numpy as np
from typing import Optional, Dict, Any
from loguru import logger

from autoflowcfd.core.fr_solver.residual_diagnostics import check_residual_finite
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
            cfl: CFL 数
            rho_inf, vel_inf, p_inf: 自由来流条件
            mu_molecular: 分子动力粘度
            low_mach_precond: 是否启用低马赫数伪时间预处理（默认 True，
                与 CPU 版一致）。环境变量 AFCFD_LOW_MACH_PRECOND=0/1
                优先于本参数。
            cfl_start, cfl_max: 自适应 CFL 的初始值/上限（与 CPU 版
                FRSolver 同名参数、同一语义）。None 时 cfl_start 退回
                `cfl`、cfl_max 退回 max(cfl, 0.5)——这样不传这两个参数的
                既有调用方行为不变（起始 CFL 仍是它们传的 cfl）。
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
        # mach_ref：与 CPU 版 FRSolver.__init__（fr_solver/solver.py）
        # 同一套计算方式/同一个用途，见该文件对应注释。物理下限钳制同样与
        # CPU 版镜像同步（2026-08-26，P2 发散专项）：低于 0.1 的参考马赫数会让
        # AUSM+up Mp 压差扩散项的 1/mach_ref² 放大压倒显式推进稳定性，
        # 完整推导/实证标定记录见 fr_solver/solver.py::_MACH_REF_FLOOR。
        mach_ref = vel_inf / np.sqrt(max(1.4 * p_inf / max(rho_inf, 1e-10), 1e-10))
        mach_ref = max(mach_ref, 0.1)
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
        _env = os.environ.get("AFCFD_LOW_MACH_PRECOND")
        _req = bool(low_mach_precond) if _env is None else (_env == "1")
        self.low_mach_precond_enabled = _req and time_scheme in ("ssp_rk2", "ssp_rk3")
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
        self.time_integrator = GPUTimeIntegrator(scheme=time_scheme, cfl=cfl)

        # 自适应 CFL 控制器（2026-09-14 补齐）：GPU 路径此前**完全没有**
        # 接入它，`compute_local_cfl_step_gpu` 一直用构造时传入的固定
        # `time_integrator.cfl`——CPU 侧 2026-08-24 就有 AdaptiveCFLController
        # 并且 CLI 的 `--cfl-start/--cfl-max` 只对 CPU 生效，这是一处先于
        # 本轮存在的后端不对等缺口（不是本轮引入的）。稳态收敛步数强依赖
        # 这个机制（见 core/time_integration/adaptive_cfl.py 模块文档第
        # 4~7 条记录的四处真实缺陷与实测收益），GPU 不能只拿固定 CFL。
        # 与 CPU 同一原则：只在稳态收敛加速模式下启用；DUAL_TIME 的内层
        # 伪时间迭代有自己独立的自适应逻辑（time_integration/dual.py），
        # 两者面向不同的迭代结构、不能互相套用。
        self._cfl_controller = None
        if time_scheme in ("ssp_rk2", "ssp_rk3", "forward_euler"):
            from autoflowcfd.core.time_integration.adaptive_cfl import (
                AdaptiveCFLController,
            )
            self._cfl_controller = AdaptiveCFLController(
                cfl_start=cfl_start if cfl_start is not None else cfl,
                cfl_max=cfl_max if cfl_max is not None else max(cfl, 0.5),
                **({} if cfl_min is None else {'cfl_min': cfl_min}),
            )

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
            # face_flux_points/merge.py 的 flat array 重构后
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

    def _compute_local_time_step_gpu(self, return_physical_too: bool = False):
        """GPU 计算局部 CFL 时间步长（使用所有 SP 的谱半径）。

        `return_physical_too=True` 时额外返回"用物理波速算出的"那一份：
        启用低马赫数预处理时平均流的 dt 按预处理波速放大，而湍流标量
        （k/omega）必须继续用物理波速那一份（与 CPU 侧 cfl.py 同一处理，
        理由见那里的文档）。未启用预处理时两者是同一个数组对象。
        """
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
        precond = getattr(self, "low_mach_precond_enabled", False)
        dt_phys_all_sps = (cp.zeros((n_cells, n_sps), dtype=cp.float64)
                           if precond else None)
        for sp in range(n_sps):
            U_sp = self.U_gpu[:, sp:sp+1, :]  # (n_cells, 1, n_vars)
            det_jacs_sp = det_jacs_gpu[:, sp] if det_jacs_gpu is not None else None
            metric_flux_scale_sp = (
                metric_flux_scale_gpu[:, sp] if metric_flux_scale_gpu is not None else None
            )
            out_sp = compute_local_cfl_step_gpu(
                U_sp, cell_volumes,
                owner_cell, neighbor_cell, is_boundary,
                normals_gpu, areas_gpu,
                None, None,
                cfl=self._current_cfl(),
                poly_order=getattr(self, "order", 0),
                det_jacs_sp=det_jacs_sp,
                metric_flux_scale_sp=metric_flux_scale_sp,
                mach_ref=(self.freestream["mach_ref"] if precond else None),
                return_physical_too=precond,
            )
            if precond:
                dt_all_sps[:, sp], dt_phys_all_sps[:, sp] = out_sp
            else:
                dt_all_sps[:, sp] = out_sp

        # 单元内取最小值时**只看真实自由度**（2026-09-15 系统性审计）：
        # native 四面体的 n_sps 槽位里只有前 n_native=(p+1)(p+2)(p+3)/6 个
        # 是真实解点，其余是零填充槽位——它们在初始化时复制真实 SP #0、
        # 之后残差行被填零，于是**永远冻结在初始条件上**。CPU 侧
        # `cfl.py::compute_local_time_step` 返回的是逐 SP 的 (n_cells,n_sps)
        # 数组、填充槽位的 dt 只会乘到一个恒为零的残差上，所以那边不受
        # 影响；这里做了 `min(axis=1)` 把它归约成逐单元一个标量，冻结
        # 槽位就真的参与了竞争。
        # 具体污染路径：det_jacs/adj_j 对直边 native 四面体是**每单元一个
        # 常数**（见 high_order_mesh_order.py::compute_native_tet_jacobians，
        # 真实行与填充行同值），所以 dt_geometric 不受影响；受影响的是
        # 逐 SP 的波速 (|u|+a) 与 rho/mu_eff——填充槽位给的是初始条件的值。
        # min 取的是"最大波速/最大 mu_eff"那一侧，因此典型来流初始化下
        # （壁面附近流动减速）冻结槽位会把 dt 压得偏小：方向上偏保守、
        # 不会失稳，但它是用初始条件去限制当前时间步，而且会让 CPU-GPU
        # 交叉校验在四面体上无声地对不上。
        from autoflowcfd.fr.native_padding import (
            order_from_n_sps, reduce_per_cell_over_real_sps,
        )
        _np_cells = int(self.mesh.n_prism_cells)
        # 阶数从数组的 SP 轴反解（见 order_from_n_sps 文档）：填充划分由
        # 被归约数组自身决定，不读 solver 上可能短暂不同步的阶数属性。
        _p = order_from_n_sps(dt_all_sps.shape[1])
        dt_mean = reduce_per_cell_over_real_sps(
            dt_all_sps, _np_cells, _p, 'min', xp=cp)  # (n_cells,)
        if not return_physical_too:
            return dt_mean
        dt_phys = (reduce_per_cell_over_real_sps(
            dt_phys_all_sps, _np_cells, _p, 'min', xp=cp) if precond else dt_mean)
        return dt_mean, dt_phys

    def _current_cfl(self) -> float:
        """当前 CFL 数：有自适应控制器时用它，否则退回固定值。

        与 CPU 侧 cfl.py 里同一段逻辑对应（那里是
        `_cfl_controller.cfl_number if ... else 0.1`）。
        """
        c = getattr(self, "_cfl_controller", None)
        return c.cfl_number if c is not None else self.time_integrator.cfl

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

        # 局部 CFL 步长。启用低马赫数预处理时 dt_local 是**预处理后**的
        # 平均流步长（按 |un|+c_precond 取），dt_physical 是按物理波速那
        # 一份；两者的分工与 CPU 侧 step.py 完全一致。
        dt_local, dt_physical = self._compute_local_time_step_gpu(
            return_physical_too=True)
        dt_local_full = cp.broadcast_to(
            dt_local[:, None], (n_cells, n_sps)
        ).reshape(n_cells * n_sps)

        # 展平 U 用于时间积分器
        U_flat = self.U_gpu.reshape(n_cells * n_sps, self.n_vars)

        # 构建平均流残差函数（含湍流涡粘耦合）
        def mean_flow_residual_raw(U_flat_trial):
            """未经预处理的原始残差 R（约定 dU/dt = -R）。

            残差监控与自适应 CFL 都必须用这一份**物理**残差：Gamma 可逆、
            两者同时趋零，但量级不同，用预处理值会让打印的残差、收敛判据
            以及与历史算例的对比全部失去可比性（与 CPU 侧 step.py 里
            同名函数一致）。
            """
            U_trial = U_flat_trial.reshape(n_cells, n_sps, self.n_vars)
            inv_res = self.compute_inviscid_residual_gpu(U_trial)
            visc_res = self.compute_viscous_residual_gpu(U_trial, mu_t_field=mu_t_field)
            total = inv_res + visc_res
            return -total

        def mean_flow_residual(U_flat_trial):
            """供时间积分器推进用：启用预处理时返回 `Gamma R`，否则就是 R。

            Gamma 线性，作用在 R 上与作用在 dU/dtau 上等价；它**必须**与
            上面按预处理波速取的 dt 成对出现（见
            core/gpu/gpu_preconditioning.py 与 CPU 侧
            core/utils/preconditioning.py 模块文档）。
            Gamma 需要的原始变量由 `apply_low_mach_preconditioner_gpu`
            从 U_trial 自己推导——GPU 侧不依赖 `self.Q_gpu` 是否与试探态
            同步这条隐式契约（与 CPU 侧的刻意差异，见那份模块文档）。
            """
            res = mean_flow_residual_raw(U_flat_trial)
            if self.low_mach_precond_enabled:
                from autoflowcfd.core.gpu.gpu_preconditioning import (
                    apply_low_mach_preconditioner_gpu,
                )
                # `out=res` 就地写：`res` 是 `mean_flow_residual_raw` 刚
                # 算出来的新数组，stage 内施加完 Gamma 后原始值不再需要，
                # 省掉每个 RK stage 一份全场数组的显存（79 万单元 P2 约
                # 1.2GiB/stage）。`residual0` 那一处不能这样做——那里必须
                # 保留物理残差给残差范数与自适应 CFL 用。
                res = apply_low_mach_preconditioner_gpu(
                    res, U_flat_trial.reshape(n_cells, n_sps, self.n_vars),
                    self.freestream["mach_ref"], out=res,
                )
            return res.reshape(n_cells * n_sps, self.n_vars)

        # 初始残差：`residual0_raw` 供残差范数/自适应 CFL 使用（物理残差），
        # `residual0` 供积分器复用 Stage 0（必要时已施加 Gamma）。
        residual0_raw = mean_flow_residual_raw(U_flat)
        if self.low_mach_precond_enabled:
            from autoflowcfd.core.gpu.gpu_preconditioning import (
                apply_low_mach_preconditioner_gpu,
            )
            residual0 = apply_low_mach_preconditioner_gpu(
                residual0_raw, self.U_gpu, self.freestream["mach_ref"],
            ).reshape(n_cells * n_sps, self.n_vars)
        else:
            residual0 = residual0_raw.reshape(n_cells * n_sps, self.n_vars)

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

        # 残差范数：用**未预处理**的物理残差（见 mean_flow_residual_raw）
        # 显式 ravel：`residual0_raw` 现在是 3 维 (n_cells,n_sps,n_vars)，
        # `linalg.norm` 只在 ord=None 时才隐式对 >2 维做 ravel，依赖那条
        # 特殊语义没必要；ravel 后与改动前那份 2 维输入的范数逐位相同。
        _r = residual0_raw.ravel()
        residual_norm = float(cp.linalg.norm(_r) / max(1, np.sqrt(_r.size)))
        self.residual_history.append(residual_norm)
        self.iteration += 1

        # 自适应 CFL：按物理残差更新（与 CPU 侧 step.py 同一时序——在残差
        # 范数算出来之后、返回之前）
        if self._cfl_controller is not None:
            self._cfl_controller.update(residual_norm)

        return residual_norm

    def solve(
        self,
        max_iter: int = 1000,
        dt: float = 1e-4,
        tol: float = 1e-6,
        output_interval: int = 10,
        phase_max_iter: Optional[int] = None,
        residual_drop_threshold: float = 1e2,
    ) -> Dict[str, Any]:
        """执行稳态求解循环。

        Order Continuation 自动分派（2026-09-02，见 core/gpu/solver/
        gpu_solver_order_continuation.py 模块文档）：与 CPU `FRSolver.
        solve()`/GPU 分布式版本同一个判据——`self.order`（目标阶数）
        >= 2 时自动改用逐阶爬坡（`run_distributed_order_continuation`，
        尽管函数名带"distributed"，逻辑本身对 solver 只要求
        `step()`/`order`/`current_order`/`_interpolate_to_new_order`
        这几个鸭子类型接口，不依赖任何分布式概念，单机 GPU 复用同一份
        实现，不需要另写一份等价的迭代循环）。

        Args:
            max_iter: 最大迭代次数
            dt: 时间步长（稳态模式下被 CFL 覆盖）
            tol: 收敛容差
            output_interval: 输出间隔
            phase_max_iter, residual_drop_threshold: 仅在触发 Order
                Continuation（`self.order >= 2`）时生效，与单机 CPU
                `run_order_continuation` 同名参数同一含义。

        Returns:
            结果字典
        """
        if getattr(self, 'order_continuation_enabled', True) and self.order >= 2:
            from autoflowcfd.core.mpi.distributed_order_continuation import (
                run_distributed_order_continuation,
            )
            result = run_distributed_order_continuation(
                self, max_iter, dt, tol,
                phase_max_iter=phase_max_iter, residual_drop_threshold=residual_drop_threshold,
            )
            return {
                'converged': result.converged,
                'iterations': result.iterations,
                'final_residual': result.final_residual,
                'residual_history': self.residual_history,
            }

        print(f"Starting GPU solve: max_iter={max_iter}, tol={tol}")
        converged = False
        final_residual = 1e10
        _last_finite = None

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

            # 发散即中止（2026-09-16 统一）：此前这里只是 break，于是
            # 调用方拿到的是 converged=False，与"跑满预算仍未收敛"完全
            # 无法区分，收尾还会把 NaN 状态写成结果文件。改用与另外四条
            # 求解循环共享的 SolverDivergedError，让 CLI 非零退出。
            check_residual_finite(res, i + 1, last_finite=_last_finite)
            _last_finite = res

            if res < tol:
                converged = True
                print(f"✅ GPU Converged at iteration {i+1} with residual {res:.6e}")
                break

        return {
            'converged': converged,
            'iterations': self.iteration,
            'final_residual': final_residual,
            'residual_history': self.residual_history,
        }

