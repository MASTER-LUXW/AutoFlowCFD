"""
AutoFlowCFD V2.0 - Order Continuation Utilities

本模块包含 Order Continuation 方法所需的插值工具。
"""

import time as _time

import numpy as np
from typing import Any
from loguru import logger


def _build_linear_interp_matrix_3d(old_sps_1d: np.ndarray, new_sps_1d: np.ndarray) -> np.ndarray:
    """构造把 old_sps_1d 张量积网格上的节点值线性插值/外插到 new_sps_1d
    张量积网格上的算子矩阵 W，形状 (new_n_sps, old_n_sps)，满足
    new_values = W @ old_values。

    这个矩阵只依赖两组 SPs 的参考坐标位置，与场在哪个单元、取哪个变量
    完全无关——用 scipy 自身的 `RegularGridInterpolator(method='linear',
    fill_value=None)` 对 old 网格的每个标准基向量探测求值来精确提取
    该线性算子的每一列（而不是手推线性外插公式），保证与本函数替换前
    的逐单元逐变量循环实现逐位数值一致，包括 fill_value=None 的线性
    外插行为、以及 P0 阶段单点"网格"的退化情形（已用真实单位基探测
    验证：scipy 对单点网格的处理是常数广播，不会报错）——用随机场数据
    在 P0->P1/P1->P2/P2->P3 三组真实会用到的阶数转换上做过逐位对比，
    最大误差为浮点舍入级（<=1.8e-15），见开发过程记录的验证脚本。

    只需要对 old_n_sps 个基向量各构造一次插值器（P0->P1 时 1 次，
    P1->P2 时 8 次，P2->P3 时 27 次），在阶数切换时只算一次、供全部
    单元和全部变量共用——取代了原实现里"每个单元、每个变量各自构造
    一次 RegularGridInterpolator 对象"的纯 Python 循环（真实网格上
    79 万单元 x 5~7 个变量意味着几百万次 Python 级对象构造，是 Order
    Continuation 阶数切换时的一个真实、可测量的性能瓶颈）。
    """
    from scipy.interpolate import RegularGridInterpolator

    old_n1d = len(old_sps_1d)
    new_n1d = len(new_sps_1d)
    old_n_sps = old_n1d ** 3
    new_n_sps = new_n1d ** 3

    new_xx, new_yy, new_zz = np.meshgrid(new_sps_1d, new_sps_1d, new_sps_1d, indexing='ij')
    new_pts = np.column_stack([new_xx.ravel(), new_yy.ravel(), new_zz.ravel()])

    W = np.zeros((new_n_sps, old_n_sps))
    basis = np.zeros((old_n1d, old_n1d, old_n1d))
    basis_flat = basis.reshape(-1)
    for k in range(old_n_sps):
        basis_flat[k] = 1.0
        interp = RegularGridInterpolator(
            (old_sps_1d, old_sps_1d, old_sps_1d), basis,
            method='linear', bounds_error=False, fill_value=None
        )
        W[:, k] = interp(new_pts)
        basis_flat[k] = 0.0
    return W


def interpolate_to_new_order(solver: Any, new_order: int):
    """
    将解从当前阶数插值到新的阶数（Order Continuation核心逻辑）。

    文档更正：这里做的是逐变量的张量积网格线性插值
    （`scipy.interpolate.RegularGridInterpolator(method='linear')`，
    在旧 SPs 构成的规则网格上对每个守恒变量独立插值到新 SPs 位置），
    不是 L2 投影——真正的 L2 投影需要用求积权重把旧解在新的多项式空间上
    做最佳逼近（保证单元积分量守恒），当前实现没有做任何这样的求积/
    投影计算，也没有任何守恒性校验。此前文档字符串声称"L2投影...保持
    积分守恒"与实际实现不符，先如实改正说明；真正实现守恒的 L2 投影是
    独立的后续工作。

    性能说明：插值算子矩阵（`_build_linear_interp_matrix_3d`）只依赖
    新旧 SPs 的参考坐标、与单元/变量无关，本函数只构造一次、向量化
    应用到全部单元和全部场（U、k/omega、壁面距离），取代了此前"每个
    单元每个变量各自构造一次 RegularGridInterpolator"的纯 Python 循环
    ——数值结果逐位不变（见 `_build_linear_interp_matrix_3d` 文档字符串
    的验证说明），只是实现方式从循环换成矩阵乘法。

    Args:
        solver: FRSolver 实例
        new_order: 目标多项式阶数
    """
    old_order = solver.current_order
    print(f"  Interpolating solution from P{old_order} to P{new_order}...")

    # 获取新旧SPs数量 - 关键修复：直接计算，不依赖solver.state.n_sps
    old_n_points_1d = old_order + 1
    old_n_sps = old_n_points_1d ** 3

    new_n_points_1d = new_order + 1
    new_n_sps = new_n_points_1d ** 3

    print(f"    Old SPs/cell: {old_n_sps}, New SPs/cell: {new_n_sps}")

    # 如果阶数相同，无需插值
    if old_n_sps == new_n_sps:
        print(f"    Same order, skipping interpolation")
        return

    # 获取参考单元内的SPs坐标
    from autoflowcfd.fr.quadrature_points import gauss_legendre

    # 旧阶数的SPs（参考单元）
    old_sps_1d, _ = gauss_legendre(old_order + 1)
    # 新阶数的SPs（参考单元）
    new_sps_1d, _ = gauss_legendre(new_order + 1)

    W = _build_linear_interp_matrix_3d(old_sps_1d, new_sps_1d)

    # 更新状态——(n_cells, old_n_sps, n_vars) -> (n_cells, new_n_sps, n_vars)
    # 的向量化应用，取代原来的逐单元逐变量循环。
    new_U = np.einsum('ab,cbv->cav', W, solver.state.U)
    solver.state.U = new_U
    solver.state.n_sps = new_n_sps
    solver.state.Q = np.zeros_like(solver.state.U)
    solver.state._update_primitives()

    # 更新湍流场（如果有）——(n_cells, old_n_sps) -> (n_cells, new_n_sps)
    if hasattr(solver.turb_model, 'k_field'):
        solver.turb_model.k_field = np.einsum('ab,cb->ca', W, solver.turb_model.k_field)
        solver.turb_model.omega_field = np.einsum('ab,cb->ca', W, solver.turb_model.omega_field)

    # nu_t（湍流涡粘系数）同样按每单元 SPs 存储，但不会随 k_field/
    # omega_field 自动变形——它只在 compute_turbulence_source 被调用时
    # 才按当时的 k/omega 重新算出。此前假设"任何读取 nu_t 的代码之前，
    # compute_turbulence_source 总会先跑一遍把它刷新成当前阶数的正确
    # 形状"，所以这里从未插值它；但 `_compute_local_time_step` 的粘性
    # CFL 项如果在 nu_t 刷新之前就先被调用（阶数切换后的第一步），会读到
    # 上一阶数形状的陈旧 nu_t——真实复现：P1->P2 切换后 rho 已是
    # (n_cells,27)、nu_t 还留着 P1 的 (n_cells,8)，两者形状既不相等也
    # 没有一方是 1，相乘直接 ValueError 广播失败。用同一个 W 矩阵一并
    # 插值，与 k_field/omega_field 一致处理。
    if getattr(solver.turb_model, "nu_t", None) is not None and solver.turb_model.nu_t.shape[1] == old_n_sps:
        solver.turb_model.nu_t = np.einsum('ab,cb->ca', W, solver.turb_model.nu_t)

    # 壁面距离场同样按每单元 SPs 存储（core/fr_solver_turbulence.py 的湍流
    # 源项计算直接按 SP 索引取值），阶数变化后形状同样必须一起插值——
    # 此前遗漏这一步，P0 阶段用均值压缩过的 (n_cells,1) 场会在阶数提升到
    # P1/P2 后与新的 SPs 数量不匹配，下一次湍流源项计算会形状不符崩溃
    # （真实网格已复现：与 mesh Jacobian 缺少按阶数重建是同一类"阶数变化
    # 后遗漏同步派生量"问题的另一处）。
    if getattr(solver, "wall_distance", None) is not None:
        solver.wall_distance = np.einsum('ab,cb->ca', W, solver.wall_distance)

    # DDES 的有效长度尺度按上一个阶数的 SPs 维度算出，阶数变化后与刚插值
    # 完的 k_field 形状不再匹配——不能像 k_field/omega_field/wall_distance
    # 那样直接插值（它依赖 nu_t，而 nu_t 要到这一阶数第一次
    # compute_source_terms 调用后才会被重新算出，插值一个维度对但物理上
    # 过期的值没有意义），直接清空即可：下一步 compute_source_terms 会
    # 因为 des_length_scale is None 自动退回标准 RANS 耗散项（物理上是
    # 合理的边界处理，见 fr_solver_turbulence.py 的文档），再下一步
    # apply_to_sst_model 就能用这一阶数正确维度的 nu_t 重新算出它（真实
    # 网格已复现：不清空会在 P1->P2 等跨阶数切换时因形状不匹配崩溃）。
    if getattr(solver, "turb_model", None) is not None and hasattr(solver.turb_model, "des_length_scale"):
        solver.turb_model.des_length_scale = None

    # 同一类"残差计算之后才更新的缓存量，跨阶数切换后维度过期"问题
    # （见上面 des_length_scale 的处理）：LES/WMLES 的 SGS 涡粘
    # (sgs_model.nu_t) 由 apply_turbulence_corrections 在 step() 末尾算出，
    # 但 compute_viscous_residual（同一步更早）就要读取它——跨阶数切换后
    # 直接清空，get_turbulent_viscosity_field 已经对 None 做了判断（这一
    # 步退化为纯分子粘度，物理上合理的边界处理），下一步 SGS 涡粘会用新
    # 维度重新算出（真实网格已复现：不清空会在 P1->P2 等切换时因形状
    # 不匹配崩溃）。
    if getattr(solver, "sgs_model", None) is not None and hasattr(solver.sgs_model, "nu_t"):
        solver.sgs_model.nu_t = None

    # 更新SPs数量
    solver.state.n_sps = new_n_sps  # 关键修复：确保n_sps属性被正确更新
    solver.current_order = new_order

    # 注意：不在这里更新solver.ops，由调用者负责

    print(f"  ✅ Solution interpolated to P{new_order}")


def interpolate_to_new_order_checked(solver: Any, new_order: int) -> None:
    """interpolate_to_new_order 的带维度校验版本，从 fr_solver.py 拆分
    （对应旧版本 FRSolver._interpolate_to_new_order 方法体）。"""
    interpolate_to_new_order(solver, new_order)

    # 阶数变化后 SPs 每单元数量改变，DUAL_TIME 保存的上一物理时间层历史
    # （若存在）形状不再匹配，且严格来说也不再是同一离散空间下的解，
    # 必须让它失效——否则下一步 BDF2 会静默用一份形状不匹配/物理上不
    # 连续的历史层，而不是干净地退化回 BDF1。
    if hasattr(solver, "_dual_time_U_prev"):
        solver._dual_time_U_prev = None

    n_points_1d = new_order + 1
    new_n_sps = n_points_1d ** 3

    actual_n_sps = solver.state.U.shape[1]
    if actual_n_sps != new_n_sps:
        logger.error(
            f"After interpolation: expected {new_n_sps} SPs but got {actual_n_sps}. "
            f"This indicates a bug in the interpolation routine."
        )
        raise RuntimeError(
            f"State dimension mismatch after Order Continuation: "
            f"expected {new_n_sps} SPs/cell, got {actual_n_sps}"
        )

    logger.info(f"Order Continuation: Successfully interpolated to P{new_order} ({new_n_sps} SPs/cell)")


def run_order_continuation(solver: Any, max_iter: int, dt: float, tol: float,
                            checkpoint_callback=None):
    """实现 Order Continuation 策略：从 P0 逐步提升到目标阶数
    （从 fr_solver.py::FRSolver._solve_with_order_continuation 拆分）。

    Args:
        solver: FRSolver 实例
        max_iter: 总迭代次数
        dt: 时间步长
        tol: 收敛容差
        checkpoint_callback: 可选的中间 checkpoint 回调函数，
            签名为 callback(solver, iteration_number)，每步迭代后调用。

    Returns:
        SolverResult: 求解结果
    """
    from autoflowcfd.core.fr_solver.state import FRState, SolverResult
    from autoflowcfd.fr.operators import generate_fr_operators

    print("\n=== Order Continuation Strategy ===")
    print(f"Starting from P0, targeting P{solver.order}")

    original_order = solver.order
    original_ops = solver.ops

    current_state_n_sps = solver.state.U.shape[1]
    expected_p0_n_sps = 1

    if current_state_n_sps != expected_p0_n_sps:
        print(f"[INFO] Current state has {current_state_n_sps} SPs/cell, reinitializing from P0...")

        p0_state = FRState(solver.state.n_cells, expected_p0_n_sps, solver.state.n_vars)
        p0_state.initialize_uniform(
            rho=solver.freestream["rho_inf"], u=solver.freestream["vel_inf"],
            v=0.0, w=0.0, p=solver.freestream["p_inf"],
        )
        solver.state = p0_state

        if getattr(solver, "turb_model", None) is not None and hasattr(solver.turb_model, "k_field"):
            # omega 初值必须与 SSTModelFR.__init__ 的默认值（1.0，见
            # turbulence_sst.py）一致——此前这里用 1e-2，相差 100 倍，
            # 意味着从 P0 重新初始化和求解器首次构造走的是两套不同的
            # 湍流初场惯例。
            solver.turb_model.k_field = np.ones((solver.state.n_cells, expected_p0_n_sps)) * 1e-6
            solver.turb_model.omega_field = np.ones((solver.state.n_cells, expected_p0_n_sps)) * 1.0
            # nu_t 同样必须重置到 P0 维度——理由同 interpolate_to_new_order
            # 里的 nu_t 插值处理：它不会自动跟着 k_field/omega_field 变形，
            # 只在 compute_source_terms 被调用时才按当时的 k/omega 重新
            # 算出，遗漏会让它保留重置前的形状，被 _compute_local_time_step
            # 在 compute_turbulence_source 刷新它之前读取时引发同一类形状
            # 不匹配问题。用 SSTModelFR.__init__ 同样的初值约定（零）。
            if hasattr(solver.turb_model, "nu_t"):
                solver.turb_model.nu_t = np.zeros((solver.state.n_cells, expected_p0_n_sps))
            print(f"[INFO] Turbulence fields reset to P0 dimensions")

        if getattr(solver, "turb_model", None) is not None and hasattr(solver.turb_model, "des_length_scale"):
            # 同 interpolate_to_new_order 里的处理：清空而不是插值，理由见
            # 该函数文档。
            solver.turb_model.des_length_scale = None

        if getattr(solver, "sgs_model", None) is not None and hasattr(solver.sgs_model, "nu_t"):
            solver.sgs_model.nu_t = None

        if solver.wall_distance is not None:
            old_wall_dist = solver.wall_distance
            if old_wall_dist.ndim == 2 and old_wall_dist.shape[1] > 1:
                mean_wall_dist = np.mean(old_wall_dist, axis=1, keepdims=True)
                solver.wall_distance = np.tile(mean_wall_dist, (1, expected_p0_n_sps))
                print(f"[INFO] Wall distance field reset to P0 dimensions")

        solver.current_order = 0
        solver.ops = generate_fr_operators(0)
        solver.mesh.set_order(0)

        print(f"[INFO] Reinitialized to P0 ({expected_p0_n_sps} SP/cell)")

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
    orders = list(range(0, original_order + 1))

    total_iter = 0
    for target_p in orders:
        print(f"\n--- Phase: P{target_p} ---")

        if target_p > 0:
            solver._interpolate_to_new_order(target_p)

        solver.current_order = target_p
        solver.ops = generate_fr_operators(target_p)
        # mesh 的 SPs/Jacobian/Flux Points 几何是阶数相关的（见
        # HighOrderMesh.set_order 文档）——必须随 solver.ops 一起切换，
        # 否则梯度/残差计算会用错误维度的几何量崩溃。
        solver.mesh.set_order(target_p)

        expected_n_sps = solver.ops.D_3d.shape[0]
        actual_n_sps = solver.state.U.shape[1]
        if actual_n_sps != expected_n_sps:
            raise RuntimeError(
                f"Order Continuation dimension mismatch after interpolation to P{target_p}: "
                f"State has {actual_n_sps} SPs but operators expect {expected_n_sps} SPs"
            )

        # 释放已经离开的阶段的完整几何缓存（HighOrderMesh._order_geometry_
        # cache，见 set_order/high_order_mesh_order.py 文档）——本函数是
        # 该缓存唯一的调用方（`grep .set_order(` 全仓库确认），且单调递增
        # 遍历 P0->目标阶数、一旦离开某个阶段就再也不会回来；但 set_order
        # 自身的缓存语义是为"阶数可能被重新访问"的通用场景设计的，不知道
        # 这个调用模式是单调的，会让每个阶段完整的 Flux Points 几何（逐面
        # Newton 插值算子，187 万面级别的网格上单阶数就有明显体量）无限期
        # 累积在内存里从未释放。真实网格已复现：79 万单元网格进入 P2 阶段
        # 第一次残差求值时，因为同时驻留 P0+P1+P2 三份完整几何，一次 1.56
        # GiB 的过积分张量收缩分配失败崩溃。
        #
        # 只保留当前阶段 `target_p`，不再对 `original_order` 破例（真实
        # 复现，2026-08-21，79 万单元/187 万面生产网格、本机 33GiB
        # 物理内存：只破例保留 original_order 这一个改动版本，P0->P1
        # 切换时依然 OOM——P1 阶段仍要同时驻留 P1 的完整 Flux Points
        # 几何 + 被破例保留的 P2（original_order）几何两份，对 187 万面
        # 规模的网格，两份仍然超出可用内存，说 3->2 份不够，必须是
        # 3->1 份）。`original_order` 破例保留的唯一目的是省下"本函数
        # 结束时切回目标阶数"那一次 Flux Points 重建；但 `orders =
        # list(range(0, original_order+1))` 决定了循环最后一个 target_p
        # 恰好就是 original_order，那一次 `solver.mesh.set_order(target_p)`
        # 本来就会在缓存缺失时透明地触发重建（见 set_order 文档：
        # `if order not in mesh._order_geometry_cache: 重建`，不是异常
        # 路径）——破例保留换来的只是省掉这一次重建，用峰值内存翻倍
        # 换一次性能优化，真实网格上不划算，去掉这个特殊情况。
        stale_orders = [
            o for o in list(solver.mesh._order_geometry_cache)
            if o != target_p
        ]
        for o in stale_orders:
            del solver.mesh._order_geometry_cache[o]

        phase_max_iter = max_iter // len(orders)
        phase_tol = tol * (10 ** (original_order - target_p))

        # CL-02 修复：阶数提升触发条件改为残差下降判据
        # 规范要求"残差降 2 个数量级后提升阶数"，而非固定迭代预算
        # 记录本阶数初始残差，用于判断相对下降量
        initial_residual_this_order = None
        residual_drop_threshold = 1e2  # 残差下降 2 个数量级
        min_iter_before_transition = 20  # 最少迭代次数，避免过早提升

        converged = False
        final_residual = 1e10

        for i in range(phase_max_iter):
            t_start = _time.time()
            res = solver.step(dt)
            t_end = _time.time()
            final_residual = res
            total_iter += 1

            if initial_residual_this_order is None:
                initial_residual_this_order = res

            if True:  # 每步都输出残差与气动力系数
                drop_ratio = initial_residual_this_order / max(res, 1e-30)
                msg = f"P{target_p} Iter {i+1}: Residual = {res:.6e} | Drop: {drop_ratio:.1f}x | Time: {t_end - t_start:.2f}s"
                # 每步输出气动力系数（轻量级压力积分，不含粘性力梯度）
                ref_area = getattr(solver, '_reference_area', None)
                if ref_area is not None and ref_area > 0:
                    from autoflowcfd.postprocess.fr_coefficients import compute_forces_pressure_only
                    aero = compute_forces_pressure_only(solver, ref_area)
                    msg += f" | Cd={aero['Cd']:.4f} Cl={aero['Cl']:.4f} Cs={aero['Cs']:.4f}"
                print(msg)

            # 中间 checkpoint 保存（按 --checkpoint-interval 间隔）
            if checkpoint_callback is not None:
                checkpoint_callback(solver, total_iter)

            # 收敛判据：绝对容差
            if res < phase_tol:
                converged = True
                print(f"[OK] P{target_p} converged at iter {i+1}")
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
            return SolverResult(converged=True, iterations=total_iter, final_residual=final_residual)

    solver.order = original_order
    solver.ops = original_ops

    return SolverResult(converged=False, iterations=total_iter, final_residual=final_residual)
