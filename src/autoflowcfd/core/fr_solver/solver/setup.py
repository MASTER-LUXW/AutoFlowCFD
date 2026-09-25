"""AutoFlowCFD V2.0 - `FRSolver.__init__` 的装配阶段（mixin，只含方法）。

从 `core/fr_solver/solver.py` 拆出（2026-09-25）：原 `__init__` 543 行，按
"选项 -> 初场 -> 算子 -> 物理与边界 -> 湍流与时间积分 -> 运行期状态"的
依赖顺序切成具名阶段，**顺序本身是契约**（各阶段注释里记录的几处真实
缺陷都出在顺序上，例如 WMLES 模型必须先于边界幽灵态构造）。
"""

from typing import Optional

import numpy as np

from autoflowcfd.core.fr_solver.mach_ref import resolve_mach_ref
from autoflowcfd.core.fr_solver.state import FRState
from autoflowcfd.core.time_integration.base import TimeIntegrator
from autoflowcfd.core.utils import solver_helpers
from autoflowcfd.core.utils.preconditioning import resolve_low_mach_precond
from autoflowcfd.fr.operators import generate_fr_operators


class _SolverSetupMixin:
    """`FRSolver.__init__` 按依赖顺序调用的装配阶段。"""

    def _setup_options(self, mesh, order, backend, artificial_viscosity_enabled,
                       artificial_viscosity_alpha, entropy_stable_volume_enabled):
        """网格/阶数/后端与两个可选数值开关（含"无操作/未移植"的显式警告）。"""
        self.mesh = mesh
        self.order = order
        self.artificial_viscosity_enabled = artificial_viscosity_enabled
        self.artificial_viscosity_alpha = artificial_viscosity_alpha
        # **order == 1 时人工粘性是精确的无操作**（2026-09-15 实测确认）。
        # Persson-Peraire 的 ramp 判据是 `s0 = -4*log10(order)`，order=1 时
        # s0 = 0，于是触发门限成了"顶模态能量占全胞 10% 以上"
        # （`S_e >= 10^(s0-kappa) = 0.1`）。而 P1 的"顶模态"**就是全部
        # 非常数模态**，即胞内变化本身——已解析的 P1 物理与混叠在这个
        # 指标下无法区分，传感器几乎不触发。
        #
        # 实测（plate_demo 363,392 单元，order=1，固定 CFL 0.03，其余参数
        # 逐项相同的 A/B）：开与不开 `--artificial-viscosity`，**前 51 步的
        # 残差与 Cd 逐字符完全相同**——人工粘性贡献精确为零。
        #
        # 这一点要紧，因为项目对退化单元残差放大的既定修复路线是
        # "网格质量门 + 耗散"（见 industry_practice_degenerate_cell_gcl
        # 项目记忆），而耗散那一半在**生产阶数**上不可用。order=2/3 的
        # 门限分别是 6.25e-3 / 1.24e-3，才是传感器的正常工作区间。
        #
        # 按本项目"不接受静默无操作"的约定，这里显式警告而不是让用户以为
        # 自己已经打开了一层保护。同源结论见 fr_solver/filter.py 里
        # `AFCFD_FILTER_MODE=sensor` 在 P1 上原理不适用的说明。
        if artificial_viscosity_enabled and order <= 1:
            import warnings
            warnings.warn(
                f"--artificial-viscosity 在 order={order} 上是**无操作**："
                f"Persson-Peraire 判据 s0=-4*log10(order)=0 使触发门限变成"
                f"顶模态能量占比 >= 10%，而 P1 的顶模态就是全部非常数模态，"
                f"传感器几乎不触发（真实网格 A/B 实测前 51 步残差/Cd 逐字符"
                f"相同）。要用人工粘性请在 order>=2 上使用；P1 的退化单元"
                f"问题目前只能靠网格质量门拦。",
                RuntimeWarning, stacklevel=2,
            )
        self.entropy_stable_volume_enabled = entropy_stable_volume_enabled
        if entropy_stable_volume_enabled and backend.lower() == "gpu":
            import warnings
            warnings.warn(
                "entropy_stable_volume_enabled=True 目前只在 CPU 路径实现"
                "（core/fr_residual/inviscid.py::compute_inviscid_residual_fr），"
                "GPU 路径（core/gpu/residual/gpu_inviscid.py）尚未移植，"
                "backend='gpu' 时这个开关不生效，体积项仍走 GPU 原有的"
                "逐点通量代入实现——不是被静默忽略导致错误结果，只是"
                "拿不到这个优化。",
                stacklevel=2,
            )
        self.backend_type = backend.lower()
        

    def _setup_initial_state(self, initial_state, n_cells, n_sps, turb_model_name,
                             rho_inf, vel_inf, p_inf, aoa_deg, aos_deg):
        """守恒量初场：Order Continuation 传入的低阶状态，或与来流一致的均匀场。"""
        # 1. 初始化状态 (S-01)
        if initial_state is not None:
            # 使用提供的初始状态（Order Continuation）
            self.state = initial_state
            print("   [OK] Using provided initial state from lower order")
        else:
            # 根据湍流模型确定变量数。必须先 upper()：self.turb_model_name
            # 要到第 5 步才被规范化为大写，这里若直接用构造参数原始大小写
            # 比较，会导致 `turbulence-model sst`（小写，steady CLI 路径）
            # 与 `SST`（大写，transient CLI 路径）判出不同的 n_vars——已用
            # 两条 CLI 路径实测复现（steady 得到 n_vars=5，transient 得到 7）。
            default_n_vars = 7 if turb_model_name.upper() in ["SST", "DDES", "IDDES"] else 5
            self.state = FRState(n_cells, n_sps, default_n_vars)
            # 初场必须与自由来流边界条件一致（都用同一套 rho_inf/vel_inf/p_inf），
            # 否则显式伪时间推进第一步就要吸收一个几个数量级的压力跳跃
            # （旧版本硬编码 rho=1,p=1 的"单位"初场，与真实边界条件的
            # rho_inf~1.2, p_inf~1e5 相差 5 个数量级，显式格式在这种冲击下
            # 数值发散——这不是残差组装的 bug，是初始条件与边界条件不一致
            # 导致的可预见的数值不稳定，工业代码从来不会这样初始化）。
            # 初场速度方向必须与边界 Q_free 用同一个来流方向，否则第一步
            # 就要吸收一个与攻角同量级的速度跳跃（与下面那段注释记录的
            # "初场与边界条件不一致必然发散"是同一类问题）。
            from autoflowcfd.core.utils.flow_direction import freestream_velocity
            _v0 = freestream_velocity(vel_inf, aoa_deg, aos_deg)
            self.state.initialize_uniform(
                rho=rho_inf, u=float(_v0[0]), v=float(_v0[1]), w=float(_v0[2]),
                p=p_inf)
        

    def _setup_operators(self, order, time_scheme, low_mach_precond):
        """FR 算子与低马赫预处理开关（两者都只依赖阶数/时间方案）。"""
        # 2. 预计算算子 (G-04)——四面体坍缩坐标基已删除（2026-09-03，
        # 见 fr/operators.py 模块文档），`generate_fr_operators` 不再
        # 接受 `tet_basis_mode` 参数，恒生成 native 四面体算子，与
        # `mesh`（同样恒为 native，见 HighOrderMesh 文档）天然一致，
        # 不再需要从 mesh 读取这个属性来保持两者同步。
        # 低马赫数伪时间预处理开关：规则与环境变量覆盖的唯一来源是
        # `core/utils/preconditioning.py::resolve_low_mach_precond`。
        self.low_mach_precond_enabled = resolve_low_mach_precond(
            low_mach_precond, time_scheme)

        self.ops = generate_fr_operators(order)

        # BLAS 线程数压到 1（性能优化 2026-09-13，见 `_limit_blas_threads`
        # 与 `autoflowcfd/__init__.py` 顶部的完整实测记录）。**位置很关键**：
        # 必须在网格几何（mesh.jacobians，调用方在构造 solver 之前就算好）
        # 与上面这行 FR 算子构造**之后**才限制——这两处的 LAPACK 结果会随
        # 线程数在最后一位上变化，而离散 GCL/自由流场保持性依赖度量量之间
        # 近乎精确的抵消（把限制提前会让棱柱 P2 保持性判据从 8.07e-7 退化到
        # 3.35e-6、真实测试失败）。在这之后限制：残差与全程多线程逐位相同，
        # 同时求解循环拿到 9~11% 的收益。
        # 注意：**不要**在这里调 `_limit_blas_threads()`——那样是进程级、
        # 粘性的，会污染同一进程里后续求解器的几何/算子构造（真实 bug，
        # 见 `blas_threads_limited` 文档记录的 Couette 连跑失败）。
        # 限制只在求解循环内生效，见 `solve()` 里的 `blas_threads_limited`。
        

    def _setup_physics_and_boundary(self, turb_model_name, mu_molecular, rho_inf, vel_inf,
                                    p_inf, aoa_deg, aos_deg, turbulence_intensity,
                                    viscosity_ratio, sem_num_eddies, bc_overrides):
        """来流字典、WMLES 模型、Tu/VR/SEM 与边界幽灵态（顺序是契约，见各段注释）。"""
        # 3. 初始化边界条件 (BD-01) —— 真正参与残差组装的幽灵态边界条件
        # （不再持有未被使用的 FRWeakBC 罚项处理器实例——那是旧版本从未被
        # 求解主循环调用过的死代码路径，真正生效的是下面的
        # boundary_ghost_provider，见 boundary/fr_ghost_state.py）
        #
        # self.turb_model_name 必须先于 boundary_ghost_provider 构造
        # （BD-02：LES/DDES 模式下要给 VELOCITY_INLET 组接入合成湍流
        # 入口，需要在构造 ghost provider 时就知道湍流模型），其余湍流
        # 模型对象（turb_model/sgs_model）留到下面第 5 步再真正初始化。
        #
        # 真实 bug 修复（2026-09-02，排查分布式 WMLES 支持时发现，与
        # 分布式本身无关，是本文件独立的一个真实回归）：上面这句"ghost
        # provider 构造时只用 getattr(...,None) 安全读取 wmles_model，
        # 不依赖它已经存在"是过时的错误理解——`build_boundary_ghost_
        # provider` 用 `getattr(solver,"wmles_model",None) is None` 判断
        # WALL 组是否要切换成 is_no_slip=False（见该函数文档 #9 修复
        # 说明），但 `self.wmles_model` 真正被赋值（`_init_turbulence_
        # models`，下面第 5 步）发生在 `boundary_ghost_provider` 构造
        # **之后**——`getattr` 在属性完全不存在时同样返回 None，不会
        # 报错，但这意味着 `wall_is_no_slip` 恒为 True，#9 修复的
        # "WMLES 假滑移边界"效果自那次修复写下起就从未真正生效过（只是
        # 不崩溃，掩盖了这个事实）。这里提前构造真正的 wmles_model（只
        # 需要 mu_molecular/rho_inf 两个已经是构造参数的标量，不依赖
        # 第 5 步其余状态），下面第 5 步改为不再重复构造（见该处新增的
        # guard）。
        self.turb_model_name = turb_model_name.upper()
        self.wmles_model = None
        if self.turb_model_name == "WMLES":
            from autoflowcfd.core.turbulence.wmles import WMLESModel
            self.wmles_model = WMLESModel(nu=mu_molecular / max(rho_inf, 1e-10))
        # mach_ref 的唯一来源（按 AUSM+up 预处理档钳下限，推导与标定见该函数）。
        mach_ref = resolve_mach_ref(rho_inf, vel_inf, p_inf)
        # aoa_deg/aos_deg 进 freestream 字典：下游的 Q_free 构造、SEM 入口
        # 方向、气动力风轴系分解都从这里读，不各自再传一遍参数（那样任何
        # 一条路径漏传都会静默退回 +x）。
        self.freestream = {"rho_inf": rho_inf, "vel_inf": vel_inf, "p_inf": p_inf,
                           "mach_ref": mach_ref,
                           "aoa_deg": float(aoa_deg), "aos_deg": float(aos_deg)}
        # Tu/VR/SEM 涡核数设置必须先于 boundary_ghost_provider 构造（与上面
        # turb_model_name 的顺序要求同理，2026-08-28 补充）：
        # build_boundary_ghost_provider 现在会读 solver._turbulence_intensity/
        # _sem_num_eddies 来驱动 BD-02 SEM 入口的目标雷诺应力/涡核数量
        # （见 fr_solver/boundary.py 模块文档），若这几个属性此时还没设置，
        # getattr 的兜底默认值会悄悄生效、用户传入的 --turbulence-intensity/
        # --sem-num-eddies 就会被忽略——必须在 _build_boundary_ghost_provider
        # 调用之前赋值。
        self._turbulence_intensity = turbulence_intensity
        self._viscosity_ratio = viscosity_ratio
        self._sem_num_eddies = sem_num_eddies
        # B-9：保存 bc_overrides 供 Order Continuation 切阶后重建边界幽灵态
        # provider（SEM 入口幽灵态持有构造阶数的 FP 坐标，见 order_continuation.py）。
        self.bc_overrides = bc_overrides or {}
        self.boundary_ghost_provider = self._build_boundary_ghost_provider(self.bc_overrides)
        self.mu_molecular = mu_molecular

    def _setup_turbulence_time_and_runtime(self, n_cells, n_sps, order, time_scheme,
                                           dual_time_inner_iter, adaptive_cfl,
                                           cfl_start, cfl_max, cfl_min):
        """后端、湍流模型、时间积分器、CFL 策略、启动日志开关与运行期状态。"""
        # 4. 初始化计算后端 (B-01)——见 solver_helpers.py::resolve_backend_type 文档。
        self.backend = None
        self.backend_type = solver_helpers.resolve_backend_type(self.backend_type)

        # 5. 初始化湍流模型（self.turb_model_name 已在第 3 步设置；
        # self.wmles_model 若适用也已在第 3 步提前构造好，供
        # boundary_ghost_provider 正确读取，这里不再重置，见第 3 步
        # 的说明与 _init_turbulence_models 里对应的 guard）
        self.turb_model = None
        self.ddes_model = None
        self.sgs_model = None

        self._init_turbulence_models(n_cells, n_sps)
        
        # 6. 初始化时间积分器 (S-05)
        self.time_integrator = TimeIntegrator(scheme=time_scheme, dual_time_steps=dual_time_inner_iter)

        # 6b. 自适应 CFL 控制器（2026-08-24）：
        # 稳态伪时间迭代中根据残差历史自动调节 CFL 数，替代此前硬编码 0.1。
        # 仅对稳态路径（SSP-RK2/RK3）生效；DUAL_TIME 有自己的内层自适应逻辑。
        # CFL 策略（控制器 or 固定 CFL）的唯一事实来源：
        # `time_integration/adaptive_cfl/policy.py::build_cfl_policy`（六个后端
        # 构造点此前各写一份且已分叉，见该模块文档）。
        # 固定 CFL（`adaptive_cfl=False` 或 DUAL_TIME）时记下请求值给 cfl.py 用
        # —— 2026-09-17 修过的"固定 CFL 请求被静默换成 0.1"缺陷就出在这里。
        from autoflowcfd.core.time_integration.adaptive_cfl.policy import (
            build_cfl_policy, describe_cfl_policy,
        )
        self._cfl_controller, self.fixed_cfl_number = build_cfl_policy(
            time_scheme, adaptive=adaptive_cfl,
            cfl_start=cfl_start, cfl_max=cfl_max, cfl_min=cfl_min)
        print("   " + describe_cfl_policy(self._cfl_controller, self.fixed_cfl_number))

        # 影响物理/数值的开关必须在启动日志里可见（2026-09-15 引入）：做
        # 人工粘性 A/B 对照时发现，`--artificial-viscosity` 生效与否在
        # 整份日志里**没有任何痕迹**，只能靠翻进程命令行确认——那让
        # "这份日志是哪个配置跑出来的"变成了事后考古。模态滤波档同理，
        # 它直接决定解的多项式阶数有没有被清掉一整阶。
        #
        # 2026-09-16：这三行原先写在上面 `if adaptive_cfl and ...` 的
        # **then 分支里**，所以关掉自适应 CFL（含全部 DUAL_TIME 运行）
        # 时它们一行都不打印——恰好复刻了它本该消除的那个问题。已移到
        # 分支外，与 CFL 档无关。同时补上 AUSM+up 预处理档：它决定壁面
        # 压力是否被推进 P5± 的饱和分支（legacy 档在固壁上有 7.3 倍虚假
        # 超压，见 fr_operators/kernels.py 模块级 PRECOND_* 常量上方的
        # 长注释），是比滤波档更强的物理开关，绝不能只能靠命令行考古。
        from autoflowcfd.core.fr_operators.kernels import (
            ausm_precond_mode_label as _pm_label,
            resolve_ausm_precond_mode as _pm_resolve,
        )
        from autoflowcfd.fr.modal_filter import FILTER_MODE as _fm
        _av = ("enabled (alpha=%g)" % self.artificial_viscosity_alpha
               if self.artificial_viscosity_enabled else "disabled")
        print(f"   Artificial viscosity: {_av}")
        print(f"   Modal filter mode: {_fm}")
        print(f"   AUSM+up precond mode: {_pm_label(_pm_resolve())}")
        # troubled-cell 门控判据（只在 FILTER_MODE=sensor 档生效，
        # 两个维度刻意独立以便受控 A/B）。它决定 P>=1 时"对哪些
        # 单元施加模态滤波"，是与滤波档同等量级的物理开关。
        from autoflowcfd.core.fr_operators.bounds_sensor import (
            resolve_troubled_sensor as _ts_resolve,
        )
        print(f"   Troubled-cell sensor: {_ts_resolve()}")

        # DUAL_TIME 专用：物理时间层 n-1 的解（BDF2 时间导数项需要），
        # None 表示还没有跑过物理步（下一步会退化为 BDF1），见 step()
        # 与 order_continuation.interpolate_to_new_order_checked（阶数
        # 变化后 SPs 布局改变，必须让这份历史失效，否则形状不匹配/物理
        # 上不连续的历史层会被静默用于 BDF2）。
        self._dual_time_U_prev: Optional[np.ndarray] = None

        # NEWTON_KRYLOV（隐式稳态）专用状态：
        #  * `_newton_forcing` 是 inexact-Newton 的 forcing term
        #    （Eisenstat-Walker），它**必须跨 Newton 步保持状态**——它用
        #    "上一步实际取得的残差下降"决定下一步该把线性系统解多准，
        #    每步新建一个等于永远走首步那档最保守的容差。
        #  * `_newton_last_info` 是上一步的诊断（eta / GMRES 迭代数 /
        #    theta / 残差求值次数 / dtau 缩放），供日志与测试读取。
        #  * `_newton_dtau_scale` 是 PTC 的 dtau 缩放因子（相对自适应
        #    CFL 给出的天花板），**必须跨步保持**：一步不被接受时它被
        #    缩小，用不完的档数由下一步继续（见
        #    `core/time_integration/implicit/dtau_control.py`）。
        # 阶数变化后这三个都要失效（见
        # `order_continuation.interpolate_to_new_order_checked`）：残差
        # 量级随阶数跳变，沿用旧的 forcing 状态会让升阶后的第一步用一个
        # 按旧量级算出的容差，而 dtau 缩放是按旧阶数的稳定性缩出来的。
        self._newton_forcing = None
        self._newton_last_info: Optional[dict] = None
        self._newton_dtau_scale: float = 1.0
        
        # 7. 壁面距离场（用于DDES/WMLES）
        self.wall_distance = None

        # 8. Order Continuation 状态
        self.current_order = order
        self.order_continuation_enabled = True

        # 9. 收敛历史（真实 bug 修复，V2.0 专家组盲审发现，2026-08-27）：
        # 此前 CPU FRSolver 从不记录逐迭代残差，api.py::get_convergence_history
        # 恒返回硬编码占位符 {"iterations": [], "residuals": []}——GPUFRSolver
        # 一直有这个属性且真正被 solve 循环填充，CPU 版本此前遗漏。solve()/
        # run_order_continuation() 每步迭代后 append，与 GPU 版同一约定。
        self.residual_history: list = []
