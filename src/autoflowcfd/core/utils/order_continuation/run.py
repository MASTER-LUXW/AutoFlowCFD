"""AutoFlowCFD V2.0 - 顶层编排：P0 -> P1 -> ... -> 目标阶数

从 `src/autoflowcfd/core/utils/order_continuation.py`(原 926 行)拆出(2026-09-24, 项目"单文件不超 500 行"规范)。**纯搬家, 逻辑未改**。
"""

import time as _time


from typing import Any, Optional


from autoflowcfd.core.fr_solver.residual_diagnostics import check_residual_finite

from .p0_reset import _reset_state_to_p0
from .turbulence_reset import _reset_turbulence_if_resumed_field_exploded


def run_order_continuation(solver: Any, max_iter: int, dt: float, tol: float,
                            checkpoint_callback=None,
                            phase_max_iter: Optional[int] = None,
                            residual_drop_threshold: float = 1e2):
    """实现 Order Continuation 策略：从 P0 逐步提升到目标阶数
    （从 fr_solver.py::FRSolver._solve_with_order_continuation 拆分）。

    Args:
        solver: FRSolver 实例
        max_iter: 总迭代次数
        dt: 时间步长
        tol: 收敛容差
        checkpoint_callback: 可选的中间 checkpoint 回调函数，
            签名为 callback(solver, iteration_number)，每步迭代后调用。
        phase_max_iter: 非最终阶段（P0/P1/...，不含目标阶数）各自的最大
            迭代步数上限。None（默认）时取 `max_iter // len(orders)`
            作为这个参数本身的默认值（按阶段数机械均分），传具体值则
            用显式值——**这只是 `phase_max_iter` 这一个数字的默认值来源，
            不是切换整套预算分配策略的开关**。不论 `phase_max_iter` 是
            默认算出来的还是显式传入的，**目标阶数永远不受这个上限约束，
            吃掉这次求解剩余的全部步数**（`max_iter - total_iter`）——
            这是本参数要解决的真实问题（2026-09-01，用户直接指出"不想
            机械地按 max_iter // len(orders) 判断"），旧行为下阶段数越
            多、目标阶数分到的步数占比越小，且与用户真正关心的目标阶数
            收敛程度毫无关系，纯粹是阶段计数的副作用。

            真实 bug 修复（2026-09-05，用户指出）：此前用
            `if phase_max_iter is not None` 分岔出两套完全不同的预算
            分配逻辑，等价于把"目标阶数不该被稀释"这条改进做成了必须
            显式传 `--phase-max-iter` 才能享受的 opt-in 特性——不传这个
            CLI 选项（生产环境最常见的用法）的用户，目标阶数依然被和
            非最终阶段一样按 `max_iter // len(orders)` 机械均分，本参数
            当初要解决的问题在默认路径上完全没有解决。现在改成：
            `phase_max_iter` 先按上述规则确定这一个数字本身（默认值或
            显式值），"目标阶数吃掉剩余全部步数"这条规则对两种来源
            一视同仁、无条件生效。
        residual_drop_threshold: 单个非最终阶段判定"可以提前升阶"的残差
            下降倍数，原来硬编码 `1e2`（降 2 个数量级），现在可配置。

    Returns:
        SolverResult: 求解结果
    """
    from autoflowcfd.core.fr_solver.state import SolverResult
    from autoflowcfd.fr.operators import generate_fr_operators

    print("\n=== Order Continuation Strategy ===")
    print(f"Starting from P0, targeting P{solver.order}")

    original_order = solver.order
    original_ops = solver.ops

    # resume 恢复出的 solver 状态是 checkpoint 里的真实解（可能已经在
    # P1/P2 阶段，见 solve_checkpoint_io.py::rebuild_solver_from_
    # checkpoint 打的 _resumed_from_checkpoint 标记），不是构造函数生成
    # 的占位均匀自由流场——下面"状态不在 P0 就重置回 P0 均匀流场"这条
    # 逻辑只对后者成立，对前者会把刚从 checkpoint 恢复的真实解直接
    # 丢弃，且不报错、不警告，静默从 P0 重新开始整个爬升（真实复现，
    # 2026-08-22：P1 checkpoint resume 后没有任何异常提示，但物理上
    # 完全从零开始）。resumed 时爬升范围也要从 solver.current_order
    # （checkpoint 实际所在阶数）开始，不是永远从 P0。
    resumed = getattr(solver, "_resumed_from_checkpoint", False)

    # Production ramp 基准重置标记（渐变完成后重置一次残差基准）
    # 必须在 resume 跳过代码之前初始化，否则 resume 跳过设的 True 会被覆盖
    solver._ramp_baseline_reset_done = False

    # Resume 时跳过 production ramp：checkpoint 里的流场已经充分发展，
    # k/omega 处于物理平衡态，重新跑 ramp 会人为抑制已有的湍流场。
    # 基准值处理：不清除 _phase_initial_residual——新代码存的 step 50 值
    # 正是我们想要的基准；旧代码存的 step 0 值由下面迭代循环里的
    # resume 检查分支处理（打印警告，第一步重新捕获）。
    if resumed:
        if hasattr(solver, 'turb_model') and solver.turb_model is not None:
            if hasattr(solver.turb_model, 'production_factor'):
                solver.turb_model.production_factor = 1.0
                solver._turb_production_ramp_complete = True
                solver._ramp_baseline_reset_done = True
                # 同步把渐变计数器推到完成值（真实修复，2026-08-25 代码审查）：
                # 重建的 solver 经 init_turbulence_models 后 _turb_ramp_step≈1，
                # 每步调用的 _update_production_ramp 在 current_step < ramp_steps
                # 时会用 current_step/ramp_steps 覆盖上面设的 1.0——不设这行，
                # "resume 跳过 ramp"会被架空，产生项仍从 ~2% 爬升 50 步。
                solver._turb_ramp_step = getattr(solver, '_turb_production_ramp_steps', 50)
                print(f"[INFO] Resume: skipping production ramp")

                # 湍流场是否被上界大面积钳制（旧 checkpoint 爆炸后被上界截断）：
                # 判据与完整推导见 `_reset_turbulence_if_resumed_field_exploded`
                # 文档 —— 那是全部后端共用的唯一实现（2026-09-24 合并）。
                _reset_turbulence_if_resumed_field_exploded(solver)

    current_state_n_sps = solver.state.U.shape[1]
    expected_p0_n_sps = 1

    if not resumed and current_state_n_sps != expected_p0_n_sps:
        _reset_state_to_p0(solver, expected_p0_n_sps)

    # 曾经在这里跳过 P=1（直接 P0->P2->...），理由是"P=1 下均匀自由流场
    # 残差达到 1563 倍来流压力"——这个说法已过时，被"解析精确雅可比"
    # （grid/curved_mapping.py::tet_exact_jacobian/prism_exact_jacobian）
    # 顺带修好，真实数值复核（_build_synthetic_mixed_mesh，与
    # TestFreeStreamPreservation/TestVolumeTermDealiasing 同一参考网格）：
    #   - 均匀自由流场残差（真实求解器路径，compute_inviscid_residual_fr）：
    #     P1=1.25e-10（相对p_inf），P2=1.37e-5，P3=8.2e-4——P1 现在反而是
    #     三者里最好的。
    #   - 纯几何 GCL 诊断（HighOrderMesh.verify_gcl，不经过体积项
    #     over-integration，只用 coarse 阶数自己的 D_3d_tet/prism 对
    #     adj(J) 求散度）：P1=0.105，P2=1.6e-13，P3=1.2e-8——这个数字
    #     依然真实存在，但是诊断函数自身的局限（adj(J) 是坍缩坐标下的
    #     有理函数不是多项式，P1 差分矩阵次数不够精确微分它），不代表
    #     真实求解器残差有问题（verify_gcl 文档已经写明"P0/P1 阶段
    #     ...不适用本严格判据"，本来就不该拿它当 P1 可用性的门禁）。
    #   - 非均匀（线性剪切）流场残差：P1 在退化四面体角点
    #     （计算立方体 (a,b,c)=(1,1,1)，Duffy 坍缩变换的奇异汇聚点）
    #     附近仍可能出现巨大局部残差（真实测得 2.85 vs P2 的 4.49e-6）
    #     ——但这是本项目已经记录、已经决定不在数值算法层面修的"四面体
    #     坍缩坐标各向异性"问题（见 tet_collapsed_coord_anisotropy 相关
    #     记录：应对方式是剪切区用棱柱而非四面体，不是改数值算法/加阈值），
    #     P2/P3 底层有同一个奇异性，只是点更多、数值上被摊薄——不是
    #     P1 专属、也不是跳过 P1 就能规避的风险。
    # 综上，均匀流场的顾虑已不成立，另外两点也不构成继续跳过 P1 的理由，
    # 恢复朴素的 P0->P1->...->目标阶数。
    #
    # resumed 时从 solver.current_order（checkpoint 实际所在阶数）开始，
    # 不是永远从 P0——理由见上面 resumed 变量的说明。非 resumed（正常
    # 新建求解器）时 solver.current_order 在上面未进入 if 分支的情况下
    # 仍等于构造时传入的 order，但那种情况下 current_state_n_sps 必然
    # 已经等于 expected_p0_n_sps（因为上面的重置分支已经处理过），所以
    # 这里统一用 solver.current_order 作为起点对两种场景都成立。
    starting_order = solver.current_order if resumed else 0
    orders = list(range(starting_order, original_order + 1))

    total_iter = 0
    # BLAS 线程数只在求解循环期间限制为 1（求解阶段实测快 9~11%）。
    # 作用域必须是「循环期间」而非「构造求解器时一次」——后者是进程级
    # 粘性状态，会让同进程里后续构造的求解器在 BLAS=1 下生成几何/算子，
    # 改变度量项末位并破坏离散 GCL（真实 bug，完整记录见
    # core/fr_solver/solver.py::blas_threads_limited 文档）。
    from autoflowcfd.core.fr_solver.solver import blas_threads_limited
    with blas_threads_limited(1):
        for target_p in orders:
            print(f"\n--- Phase: P{target_p} ---")

            if target_p > 0 and target_p != solver.current_order:
                solver._interpolate_to_new_order(target_p)

            solver.current_order = target_p
            solver.ops = generate_fr_operators(target_p)

            # 在构建新阶数几何*之前*先释放已经离开的阶段的完整几何缓存——
            # 原先这段清理放在下面 set_order 之后，导致新阶数几何构建期间旧阶数
            # 缓存仍完整驻留：79 万单元生产网格 P1→P2 切换时，P2 过积分几何构建峰值（预分配优化后仍有 ~4GB）
            # 叠加 P1 缓存 ~2.5GB 超出可用内存，分配失败崩溃（两次独立复现，
            # 2026-08-26）。set_order 构建新阶数几何时不读任何旧阶数缓存条目，
            # 提前清理语义不变。
            #
            # 背景（保留自首次修复，2026-08-21）：HighOrderMesh._order_geometry_
            # cache 的缓存语义是为“阶数可能被重新访问”的通用场景设计的，不知道
            # 本函数的调用模式是单调递增的，会让每个阶段完整的 Flux Points 几何
            # （逐面 Newton 插值算子，187 万面级别的网格上单阶数就有明显体量）
            # 无限期累积。首次修复时实测：79 万单元网格进入 P2 阶段第一次残差
            # 求值时，因同时驻留 P0+P1+P2 三份完整几何，一次 1.56 GiB 的过积分
            # 张量收缩分配失败崩溃。
            #
            # 只保留当前阶段 `target_p`，不对 `original_order` 破例（真实复现，
            # 2026-08-21，79 万单元/187 万面生产网格、本机 33GiB 物理内存：只破例保留
            # original_order 这一个改动版本，P0->P1 切换时依然 OOM——P1 阶段仍要同时驻留
            # P1 的完整 Flux Points 几何 + 被破例保留的 P2（original_order）几何两份，
            # 对 187 万面规模的网格，两份仍然超出可用内存，说 3->2 份不够，必须是
            # 3->1 份）。`orders = list(range(0, original_order+1))` 决定了循环最后一个
            # target_p 恰好就是 original_order，那次 `solver.mesh.set_order(target_p)`
            # 本来就会在缓存缺失时透明地触发重建（见 set_order 文档：
            # `if order not in mesh._order_geometry_cache: 重建`，不是异常路径）。
            stale_orders = [
                o for o in list(solver.mesh._order_geometry_cache)
                if o != target_p
            ]
            for o in stale_orders:
                del solver.mesh._order_geometry_cache[o]

            # mesh 的 SPs/Jacobian/Flux Points 几何是阶数相关的（见
            # HighOrderMesh.set_order 文档）——必须随 solver.ops 一起切换，
            # 否则梯度/残差计算会用错误维度的几何量崩溃。
            solver.mesh.set_order(target_p)

            # 真实 bug 修复（2026-09-06，cube_demo 真实网格 P0->P1 升阶后
            # k_mean 持续增长排查发现，见 fr_solver/turbulence.py::
            # recompute_wall_distance_for_current_order 文档完整推导）：
            # `interpolate_to_new_order`（上面 `solver._interpolate_to_new_
            # order(target_p)` 调用，发生在 `set_order` 之前）对 wall_distance
            # 做的是和 U/k_field 同一套 Lagrange 插值——但壁面距离不是解
            # 多项式场，P0 单 SP 的常数基函数会把"单元里离墙最近的 SP"这个
            # 空间分辨率信息广播抹平。`set_order(target_p)` 之后
            # `solver.mesh.sps_coords` 才反映新阶数真实的 SP 坐标，这里
            # 用缓存的纯几何量（WALL 节点坐标/Eikonal 节点距离场，与阶数
            # 无关）重新做一次精确查询，覆盖掉那个被插值污染的近似值；没有
            # 缓存时保留插值结果作为退化但形状正确的后备。
            from autoflowcfd.core.fr_solver.turbulence import recompute_wall_distance_for_current_order
            recompute_wall_distance_for_current_order(solver)

            # B-9 修复（2026-08-25，真实复现：solve transient ddes P0 阶段第一步
            # 幽灵态形状 (9,5) 无法广播进 (1,5)）：BD-02 的 SEM 入口幽灵态在
            # 构造时按当时阶数的 FP 几何预存了每面 FP 物理坐标（n_fp 阶数相关），
            # 切阶后与新阶幽灵数组形状失配；provider 其余部分（分组映射/
            # 幽灵态公式）无阶数相关状态。每次切阶后按当前阶数重建（开销仅
            # 边界面级循环，每个阶段切换只发生一次，可忽略）。
            if hasattr(solver, "_build_boundary_ghost_provider"):
                solver.boundary_ghost_provider = solver._build_boundary_ghost_provider(
                    getattr(solver, "bc_overrides", {})
                )

            expected_n_sps = solver.ops.D_3d.shape[0]
            actual_n_sps = solver.state.U.shape[1]
            if actual_n_sps != expected_n_sps:
                raise RuntimeError(
                    f"Order Continuation dimension mismatch after interpolation to P{target_p}: "
                    f"State has {actual_n_sps} SPs but operators expect {expected_n_sps} SPs"
                )

            # 自适应 CFL 重置（2026-08-24）：阶数切换导致残差跳变（插值误差），
            # 不应触发 CFL 缩小。重置后重新开始爬升阶段。
            _cfl_ctrl = getattr(solver, '_cfl_controller', None)
            if _cfl_ctrl is not None:
                _cfl_ctrl.reset()

            # 非最终阶段（P0/P1/...）vs 目标阶数的步数预算分派（2026-09-01，
            # 2026-09-05 修正，见函数文档 phase_max_iter 参数说明）：
            # `phase_max_iter` 未显式传入时只是取 `max_iter // len(orders)`
            # 作为这一个数字本身的默认值——不是切换整套预算分配策略的开关。
            # "目标阶数吃掉剩余全部步数、不再随阶段数量被稀释"这条规则对
            # 默认值和显式值一视同仁、无条件生效（真实 bug 修复：此前用
            # `is not None` 分岔，等价于把这条规则做成了必须显式传
            # `--phase-max-iter` 才能享受的 opt-in 特性，不传这个 CLI 选项
            # 的默认路径上，本参数当初要解决的问题完全没有解决）。
            is_final_stage = (target_p == original_order)
            effective_phase_max_iter = (
                phase_max_iter if phase_max_iter is not None else max_iter // len(orders)
            )
            stage_iter_budget = (max_iter - total_iter) if is_final_stage else effective_phase_max_iter
            phase_tol = tol * (10 ** (original_order - target_p))

            # CL-02 修复：阶数提升触发条件改为残差下降判据
            # 规范要求"残差降 2 个数量级后提升阶数"，而非固定迭代预算
            # 记录本阶数初始残差，用于判断相对下降量
            initial_residual_this_order = None
            min_iter_before_transition = 20  # 最少迭代次数，避免过早提升

            # resume 状态持久化修复（2026-08-23，真实 bug）：
            # `initial_residual_this_order` 是纯局部变量，每次调用
            # `run_order_continuation` 都从 None 重新记录——`solve steady`
            # 单次连续运行里 `run_order_continuation` 只调用一次，这个变量
            # 天然在每个阶数真正开始时被正确捕获一次；但 `solve resume`
            # 是全新进程、全新一次 `run_order_continuation` 调用，checkpoint
            # 恢复出来的状态通常已经在当前阶数收敛了一部分甚至大部分，
            # resume 后这里第一步测出来的残差会被错当成"这个阶数刚开始时
            # 的残差"，导致下面的残差下降判据（要求下降
            # residual_drop_threshold 倍）在还没有真实下降那么多的情况下
            # 被满足，过早升阶——真实复现：cube_demo 791k 网格从 P0
            # checkpoint resume，本该继续在 P0 收敛却在 resume 后几步内就
            # 满足了"下降 100 倍"判据升到 P1，插值到更高阶引入的截断误差
            # 精确对应之前长期排查的"P0->P1 残差暴涨"现象的一个独立成因
            # （与同一次调查里定位到的棱柱四边形侧面重复计数几何 bug是两个
            # 不同的问题，此前那次的具体案例最终由几何 bug 完全解释，但这个
            # resume 状态丢失的逻辑漏洞本身依然存在、换一个 checkpoint 就可能
            # 复现）。resume 恢复出来的第一个阶段（target_p == starting_order）
            # 如果 checkpoint 里带了上次持久化的阶段起始残差
            # （solver._phase_initial_residual，见 solve_checkpoint_io.py
            # write_checkpoint/rebuild_solver_from_checkpoint），直接用它做
            # 种子而不是等第一步重新捕获——这样"下降了多少倍"就是相对
            # *真正*的阶段起点算的，不是相对"这次 resume 调用第一步"算的。
            # 旧版本 checkpoint 没有这个字段时保留原有行为（第一步捕获），
            # 但打印警告，让用户知道这次 resume 的升阶判据可能提前触发
            # （与 k_field/omega_field 缺失时的向后兼容处理方式一致）。
            if resumed and target_p == starting_order:
                _persisted = getattr(solver, "_phase_initial_residual", None)
                if _persisted is not None:
                    initial_residual_this_order = _persisted
                    print(f"[INFO] P{target_p} resume：用 checkpoint 里保存的阶段起始残差 "
                          f"({_persisted:.6e}) 做种子，升阶判据按真实阶段起点计算")
                else:
                    print(f"[WARN] P{target_p} 从旧版本 checkpoint resume（缺少阶段起始残差记录）："
                          f"残差下降升阶判据将从这次 resume 的第一步重新开始计算，可能提前触发。")

            converged = False
            final_residual = 1e10
            _last_finite = None

            for i in range(stage_iter_budget):
                t_start = _time.time()
                res = solver.step(dt)
                t_end = _time.time()
                final_residual = res
                total_iter += 1
                # 收敛历史记录（V2.0 专家组盲审发现，2026-08-27，与
                # solver.py::solve() 的普通循环同一约定）：api.py::
                # get_convergence_history 读这个列表。
                if hasattr(solver, 'residual_history'):
                    solver.residual_history.append(res)

                if initial_residual_this_order is None:
                    initial_residual_this_order = res
                solver._phase_initial_residual = initial_residual_this_order

                # Production ramp 完成检测（2026-08-25）：
                # 渐变期间（前 50 步）湍流产生项被抑制，残差反映的是无湍流状态。
                # 渐变完成后湍流突然开启，残差可能跳升，导致相对 step 0 的“下降
                # 100x”判据永远无法满足（分母是 step 0 无湍流时的残差，分子是
                # 湍流开启后的残差，两者不在同一物理基准上）。
                # 修复：检测到渐变完成标记后，立即将基准残差重置为当前值，
                # 让 100x 判据从湍流完全开启后的第一个真实残差开始计算。
                # 发散即中止（2026-09-16）：这条是 Order Continuation 的
                # 逐阶循环，也就是全部 P2/P3 运行实际走的路径，此前同样
                # 没有任何有限性检查——项目记忆里那条“P2 第 4 步 inf”的
                # 运行就是在 inf 上继续迭代到预算耗尽的。
                check_residual_finite(res, i + 1, order=target_p,
                                      last_finite=_last_finite)
                _last_finite = res

                if getattr(solver, '_turb_production_ramp_complete', False):
                    if not getattr(solver, '_ramp_baseline_reset_done', False):
                        old_baseline = initial_residual_this_order
                        initial_residual_this_order = res
                        solver._phase_initial_residual = res
                        solver._ramp_baseline_reset_done = True
                        print(f"[INFO] P{target_p} Iter {i+1}: Production ramp complete, "
                              f"resetting residual baseline: {old_baseline:.6e} → {res:.6e}")

                if True:  # 每步都输出残差与气动力系数
                    drop_ratio = initial_residual_this_order / max(res, 1e-30)
                    msg = f"P{target_p} Iter {i+1}: Residual = {res:.6e} | Drop: {drop_ratio:.1f}x | Time: {t_end - t_start:.2f}s"
                    # 自适应 CFL 状态
                    _cfl_ctrl = getattr(solver, '_cfl_controller', None)
                    if _cfl_ctrl is not None:
                        msg += f" | CFL={_cfl_ctrl.cfl_number:.3f}"
                    # 每步输出气动力系数（轻量级压力积分，不含粘性力梯度）
                    ref_area = getattr(solver, '_reference_area', None)
                    if ref_area is not None and ref_area > 0:
                        from autoflowcfd.postprocess.fr_coefficients import compute_forces_pressure_only
                        aero = compute_forces_pressure_only(solver, ref_area)
                        msg += f" | Cd={aero['Cd']:.4f} Cl={aero['Cl']:.4f} Cs={aero['Cs']:.4f}"
                    # 按方程分别归一化残差 + 最大残差定位（参照 Fluent scaled
                    # residuals / STAR-CCM+ Max 监视器，2026-09-12 新增，见
                    # residual_diagnostics.py 模块文档"背景"一节完整推导）：
                    # 合并 RMS（上面的 `res`）在少数单元残差幅值远超全场时会
                    # 被这几个单元主导、掩盖其余方程真实的收敛/发散趋势——
                    # 这次真实排查里方块前驻点单元的能量方程残差比全局RMS
                    # 还大，只用一个合并数字完全看不出来。这里只新增打印，
                    # 不改变 `initial_residual_this_order`/`drop_ratio` 这条
                    # 现有升阶判据的任何行为。
                    #
                    # 打印频率（2026-09-13 用户反馈修复）：此前每一步都打印这行
                    # 扩展诊断，正常收敛过程中绝大多数步的信息量重复（长期跟踪
                    # 用不需要逐步都看），把终端刷成大量看似"无用"的长行——只在
                    # 第 1 步（立即看到基线）和其后每 10 步打印一次，兼顾"排查
                    # 问题时能及时看到"和"正常运行时不刷屏"两者。
                    freestream = getattr(solver, 'freestream', None)
                    if freestream is not None and hasattr(solver.state, 'dU_dt') and (i == 0 or (i + 1) % 10 == 0):
                        from autoflowcfd.core.fr_solver.residual_diagnostics import (
                            compute_scaled_residuals, format_scaled_residual_line,
                        )
                        diag = compute_scaled_residuals(solver.state.dU_dt, freestream)
                        msg += " | " + format_scaled_residual_line(
                        diag,
                        # 最大残差单元的体积分位：残差被 det(J) 除，体积
                        # 极小的退化单元天然把任何通量不平衡放大若干个
                        # 量级，所以这个数字是区分"退化单元机制"与"壁面
                        # 处理机制"最直接的单个指标（2026-09-16 真实排查
                        # 驱动，见 residual_diagnostics.py::
                        # cell_volume_percentile 文档）。
                        cell_volumes=getattr(
                            getattr(solver, "mesh", None), "cell_volumes", None),
                    )
                        # 累计伪时间 / 物体尺度对流时标（2026-09-17）：与
                        # 单阶数路径（solver.py::solve 的常规循环）逐字段
                        # 对齐。残差是否收敛与物理场是否建立是两件事，只报
                        # 前者会让人拿启动暂态的气动力系数去和文献值比；
                        # 完整记录见 `fr_solver/pseudotime_budget.py`。
                        # **n_steps 用 total_iter**（跨阶段累计），因为
                        # `solver.tau_accum` 也是跨阶段累加的。
                        _ptb_fn = getattr(
                            solver, "_pseudo_time_budget", None)
                        _ptb = (_ptb_fn(n_steps=total_iter)
                                if _ptb_fn is not None else None)
                        if _ptb is not None:
                            from autoflowcfd.core.fr_solver.pseudotime_budget import (
                                format_pseudo_time_budget,
                            )
                            msg += " | " + format_pseudo_time_budget(
                                _ptb, compact=True)
                    print(msg)

                # 中间 checkpoint 保存（按 --checkpoint-interval 间隔）
                if checkpoint_callback is not None:
                    checkpoint_callback(solver, total_iter)

                # 收敛判据：相对容差（残差相对本阶段初始值下降 1/tol 倍）
                # tol=1e-6 配合 phase_tol 的阶数缩放，实际含义：
                #   P0: 下降 4 个量级 (1/(tol*100) = 1e4)
                #   P1: 下降 5 个量级 (1/(tol*10)  = 1e5)
                #   P2: 下降 6 个量级 (1/(tol*1)   = 1e6)
                # 替代此前的绝对判据 res < phase_tol（要求 RMS 残差低于 1e-4~1e-6，
                # 对 Mach 0.1~0.3 流动初始残差 ~1e8 需下降 12~14 个量级，永远不可达）。
                drop_for_convergence = initial_residual_this_order / max(res, 1e-30)
                required_drop = 1.0 / max(phase_tol, 1e-30)
                if i >= 1 and drop_for_convergence >= required_drop:
                    converged = True
                    print(f"[OK] P{target_p} converged at iter {i+1} "
                          f"(residual dropped {drop_for_convergence:.1e}x >= {required_drop:.1e}x)")
                    break

                # 阶数提升判据（CL-02）：残差相对初始值下降足够多
                # 非最高阶时，满足下降条件即可提前进入下一阶
                if (target_p < original_order
                        and i >= min_iter_before_transition
                        and initial_residual_this_order > 0
                        and initial_residual_this_order / max(res, 1e-30) >= residual_drop_threshold):
                    print(f"[OK] P{target_p} residual dropped {initial_residual_this_order/res:.1f}x "
                          f"(>= {residual_drop_threshold:.0e}x), advancing to next order at iter {i+1}")
                    break

            if target_p == original_order and converged:
                print(f"\n[OK] Order Continuation completed: Final P{original_order} converged")
                _print_pseudo_time_summary(solver, total_iter)
                return SolverResult(converged=True, iterations=total_iter, final_residual=final_residual)

    solver.order = original_order
    solver.ops = original_ops

    _print_pseudo_time_summary(solver, total_iter)
    return SolverResult(converged=False, iterations=total_iter, final_residual=final_residual)


def _print_pseudo_time_summary(solver, total_iter: int) -> None:
    """收尾时把"物理场到底走了多远"完整报一次（两个 return 点共用）。

    为什么这不是可选的锦上添花：残差范数只说"离散方程的不平衡量在变小"，
    完全不说"物理场走了多远"。两者可以同时成立且互不矛盾——plate_demo 上
    残差单调下降 350 步而物理场只走完一个绕板特征时间的 1.8%，导致启动
    暂态的压力分布被当成壁面处理缺陷追了好几天。见
    `core/fr_solver/pseudotime_budget.py` 模块文档。
    """
    fn = getattr(solver, "_pseudo_time_budget", None)
    if fn is None:
        # 只有 FRSolver 定义了这个方法。本函数被写成对求解器类型宽容，
        # 是因为 `run_order_continuation` 此后可能被别的求解器类复用，
        # 那时缺一行诊断不该让求解失败。
        return
    b = fn(n_steps=total_iter)
    if b is None:
        return
    from autoflowcfd.core.fr_solver.pseudotime_budget import (
        format_pseudo_time_budget,
    )
    print(format_pseudo_time_budget(b))
