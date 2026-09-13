"""
AutoFlowCFD V2.0 - Order Continuation Utilities

本模块包含 Order Continuation 方法所需的插值工具。
"""

import time as _time

import numpy as np
from typing import Any, Optional
from loguru import logger

from autoflowcfd.core.fr_solver.turbulence import _set_freestream_turbulence


def _lagrange_basis_matrix_1d(old_nodes: np.ndarray, new_nodes: np.ndarray) -> np.ndarray:
    """构造 1D Lagrange 基函数求值矩阵 L，形状 (len(new_nodes), len(old_nodes))，
    满足 L[i, j] = l_j(new_nodes[i])，其中 l_j 是穿过 old_nodes 的唯一
    次数为 len(old_nodes)-1 的 Lagrange 基多项式（在 old_nodes[j] 处取值
    1，在其余 old_nodes[k]（k!=j）处取值 0）。

    用 barycentric Lagrange 插值公式（Berrut & Trefethen 2004）而不是
    构造/求解 Vandermonde 矩阵，数值稳定性更好且是标准做法：
        w_j = 1 / prod_{k!=j} (x_j - x_k)
        l_j(y) = (w_j / (y - x_j)) / sum_k (w_k / (y - x_k))
    当 y 精确等于某个 x_m 时用 Kronecker delta 直接短路（避免除以零，
    也保证节点自身处的取值精确为 0/1 而不是浮点噪声）。

    old_nodes 只有 1 个点（P0 阶段）时退化为常数基函数 l_0(y)=1，
    与 barycentric 公式本身在 n=1 时的极限行为一致（分子分母是同一个
    非零标量的比值，恒为 1），不需要特殊分支。
    """
    n_old = len(old_nodes)
    n_new = len(new_nodes)
    L = np.zeros((n_new, n_old))

    # barycentric 权重 w_j = 1 / prod_{k!=j}(x_j - x_k)
    diffs = old_nodes[:, None] - old_nodes[None, :]  # (n_old, n_old)
    np.fill_diagonal(diffs, 1.0)  # 避免自身对自身除以 0，对角本来就不参与连乘
    w = 1.0 / np.prod(diffs, axis=1)  # (n_old,)

    for i, y in enumerate(new_nodes):
        exact = np.isclose(y, old_nodes, rtol=0.0, atol=1e-13)
        if np.any(exact):
            L[i, :] = 0.0
            L[i, np.argmax(exact)] = 1.0
            continue
        terms = w / (y - old_nodes)  # (n_old,)
        L[i, :] = terms / np.sum(terms)
    return L


def _build_linear_interp_matrix_3d(old_sps_1d: np.ndarray, new_sps_1d: np.ndarray) -> np.ndarray:
    """构造把 old_sps_1d 张量积网格上的节点值精确 Lagrange 多项式延拓
    （prolongation）到 new_sps_1d 张量积网格上的算子矩阵 W，形状
    (new_n_sps, old_n_sps)，满足 new_values = W @ old_values。

    真实 bug 修复（2026-09-02）：此前这里用 `scipy.interpolate.
    RegularGridInterpolator(method='linear')` 在 old SPs 之间做分段
    多线性插值——这不是对 old SPs 所隐含的那个次数为 old_order 的
    Lagrange 多项式的精确求值，而是一个更低阶（分段线性）的近似。
    对 old_order<=1（P0->P1、P1->P2 转换）old 方向每维只有 1~2 个点，
    分段线性恰好与真正的 Lagrange 多项式（常数/线性）重合，误差不可见；
    但对 old_order>=2（P2->P3 等本项目实际会用到的转换，见
    `interpolate_to_new_order` 文档）old 方向每维有 3+ 个点，分段线性
    插值与真正的二次/更高次 Lagrange 多项式在非节点位置系统性不同——
    引入了真实的、此前未被识别的插值误差，而不只是"不是 L2 投影但足够
    好"。

    Order Continuation 只会从低阶向高阶单调推进（`run_order_continuation`
    里 `orders = list(range(starting_order, original_order+1))`，
    `target_p` 严格递增，从未反向调用），意味着 new 方向的张量积多项式
    空间（次数 new_order）严格包含 old 方向的多项式空间（次数
    old_order <= new_order）——old SPs 上的节点值所唯一确定的那个
    次数为 old_order 的多项式，可以在 new 多项式空间里被精确表示、
    不需要近似。因此这里改为精确的 Lagrange 基函数求值矩阵
    （`_lagrange_basis_matrix_1d`）而不是数值插值——新节点上的值就是
    旧多项式的精确解析求值，不引入任何近似误差（浮点舍入级之外）。
    这比"L2 投影"更强：L2 投影是"在新空间里找最接近旧函数的多项式"，
    但当旧函数本来就精确落在新空间里时，L2 投影的解就是旧函数本身——
    这里直接算的正是这个精确解，不需要通过求积/投影方程迂回得到，
    同时自动保证任意阶矩（含单元积分/质量）精确守恒，不只是"近似
    守恒"。

    参考空间是张量积（Gauss-Legendre 节点各方向独立），且几何映射
    （参考单元->物理单元的 Jacobian）不随解的多项式阶数变化，只依赖
    单元几何——因此参考空间的精确多项式延拓等价于物理空间的精确延拓，
    不需要额外的物理空间求积权重修正。

    只需要对每个 1D 方向构造一次 Lagrange 基矩阵（与本函数替换前一样，
    在阶数切换时只算一次、供全部单元和全部变量共用），然后用 Kronecker
    积把 3 个独立方向的 1D 矩阵组装成完整的 3D 张量积矩阵——3D 张量积
    基函数 phi_{a,b,c}(x,y,z)=l_a(x)*l_b(y)*l_c(z) 在新节点
    (new_x_i,new_y_j,new_z_k) 处的取值就是 L[i,a]*L[j,b]*L[k,c]，与
    `new_pts`/`basis_flat` 沿用的 C-order（meshgrid(indexing='ij') 后
    ravel）嵌套索引约定完全一致，用 `np.kron` 三次组装即可，不需要
    重新推导索引映射。
    """
    L = _lagrange_basis_matrix_1d(old_sps_1d, new_sps_1d)
    return np.kron(np.kron(L, L), L)


def interpolate_to_new_order(solver: Any, new_order: int):
    """
    将解从当前阶数插值到新的阶数（Order Continuation核心逻辑）。

    真正实现（2026-09-02 修复）：对每个守恒变量做逐张量积方向的精确
    Lagrange 多项式延拓（`_build_linear_interp_matrix_3d`，內部现在是
    `_lagrange_basis_matrix_1d` 的 3D Kronecker 积，不再是分段线性
    近似插值，见该函数文档"真实 bug 修复"一节）。由于 Order
    Continuation 只单调升阶（`run_order_continuation` 里 `target_p`
    严格递增），旧 SPs 隐含的次数为 old_order 的多项式精确落在新的
    （次数 new_order>=old_order）张量积多项式空间里——这里做的是这个
    多项式在新节点上的精确解析求值，不是近似插值，也不需要通过求积/
    投影方程迂回：任意阶矩（含单元积分量）在浮点舍入误差范围内精确
    守恒，比"L2 投影"这个目标更强（L2 投影解的是"新空间里最接近旧
    函数的多项式"这个更弱的问题，只有当旧函数本来就精确落在新空间时
    两者才重合——这里正是这个精确重合的情形，直接算解析解而不必假装
    只能近似）。此前文档声称"L2投影...保持积分守恒"但实现只是分段线性
    插值、二者不符；此前更正版文档反过来又声称"不是 L2 投影、不保证
    守恒、真正的 L2 投影是独立后续工作"——这个更正本身也只对了一半：
    实现确实此前不是 L2 投影，但"真正的 L2 投影是独立后续工作"这个
    结论是不必要的，本次直接用精确多项式延拓一次性解决，不需要另开
    一个"实现 L2 投影"的后续任务。

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
    from autoflowcfd.core.fr_solver.state import FRState, SolverResult
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

                # 检测湍流场是否被上界大面积钳制（旧 checkpoint k/omega 爆炸后
                # resume 被上界截断）。如果超过 10% 的单元 k/omega 接近上界，
                # 说明湍流场从未真正恢复，必须重置到来流初值让 SST 源项重新
                # 建立平衡。
                #
                # 真实 bug 修复（2026-09-05，cube_demo 791,492 单元真实网格
                # `solve resume` 长程验证决定性发现）：本节注释一直描述的
                # 判据是"统计有多大比例的单元被钳制在上界附近"，但下面这段
                # 代码此前从未真正这样算过——只拿 k_mean 和一个
                # max(1%*k_max, 10*k_inf) 公式比较，是对注释意图的错误实现。
                # 真实复现：cube_demo 这类强分离钝体绕流（尾流/剪切层湍流度
                # 远高于来流），健康、充分发展的 k 场 k_mean=38.11（是
                # k_inf=0.167 的 228 倍），但 reset_threshold=max(5.55,1.67)
                # =5.55——k_mean 远超这个阈值，被误判成"爆炸残留"，
                # resume 第一次调用 solver.solve() 就把整个 k_field/
                # omega_field 直接清零重置回自由流初值，销毁了几千步真实
                # 演化出的湍流场（同一批 solve resume 直接调用
                # rebuild_solver_from_checkpoint+手动 solver.step() 不经过
                # 这段 resumed 分支时完全正常，交叉验证坐实了问题就在这里）。
                # 用真实数据核实：这份健康 checkpoint 上，真正被钳制在
                # k_max/omega_max 90%以上的单元占比分别只有 4.36%/0.07%，
                # 远低于注释一直声称的 10% 判据——现在改成真正按这个比例
                # 判断，而不是看均值。
                if hasattr(solver.turb_model, 'k_max') and hasattr(solver.turb_model, 'k_field'):
                    k = solver.turb_model.k_field
                    omega = getattr(solver.turb_model, 'omega_field', None)
                    k_max_limit = solver.turb_model.k_max
                    omega_max_limit = getattr(solver.turb_model, 'omega_max', None)

                    k_near_ceiling_frac = float(np.mean(k >= 0.9 * k_max_limit))
                    omega_near_ceiling_frac = (
                        float(np.mean(omega >= 0.9 * omega_max_limit))
                        if omega is not None and omega_max_limit is not None else 0.0
                    )
                    ceiling_frac_threshold = 0.10
                    if k_near_ceiling_frac > ceiling_frac_threshold or omega_near_ceiling_frac > ceiling_frac_threshold:
                        k_inf, omega_inf = _set_freestream_turbulence(solver)
                        print(f"[WARN] Resume: {100*k_near_ceiling_frac:.2f}% of k / "
                              f"{100*omega_near_ceiling_frac:.2f}% of omega values are clamped "
                              f"near their ceiling (k_max={k_max_limit:.2f}, "
                              f"omega_max={omega_max_limit}) — exceeds {100*ceiling_frac_threshold:.0f}% "
                              f"threshold, turbulence field not recovered from previous explosion. "
                              f"Resetting to freestream values.")
                        # 用 Tu/VR 推导的物理自洽值重置（与初始化一致）
                        solver.turb_model.k_field[:] = k_inf
                        solver.turb_model.omega_field[:] = omega_inf
                        if hasattr(solver.turb_model, 'nu_t'):
                            solver.turb_model.nu_t[:] = 0.0
                        # Resume 算例不重新跑 ramp：平均流场已充分发展，
                        # Tu/VR 推导的 k/ω 物理自洽（P_k/D_k ≈ 1），
                        # 不需要 ramp 抑制初始瞬态。保持 production_factor=1.0。
                        # 残差基线保持 checkpoint 保存的值，不重置。

    current_state_n_sps = solver.state.U.shape[1]
    expected_p0_n_sps = 1

    if not resumed and current_state_n_sps != expected_p0_n_sps:
        print(f"[INFO] Current state has {current_state_n_sps} SPs/cell, reinitializing from P0...")

        p0_state = FRState(solver.state.n_cells, expected_p0_n_sps, solver.state.n_vars)
        p0_state.initialize_uniform(
            rho=solver.freestream["rho_inf"], u=solver.freestream["vel_inf"],
            v=0.0, w=0.0, p=solver.freestream["p_inf"],
        )
        solver.state = p0_state

        if getattr(solver, "turb_model", None) is not None and hasattr(solver.turb_model, "k_field"):
            # 用 Tu/VR 推导的物理自洽值重置（与 init_turbulence_models 一致）
            k_inf, omega_inf = _set_freestream_turbulence(solver)
            solver.turb_model.k_field = np.ones((solver.state.n_cells, expected_p0_n_sps)) * k_inf
            solver.turb_model.omega_field = np.ones((solver.state.n_cells, expected_p0_n_sps)) * omega_inf
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
        # flux_point_type 显式透传 solver.flux_type（#14，2026-08-28）：
        # 此前恒用默认值重建算子，跨阶数切换后会静默丢失用户显式选择的
        # 'gauss' 修正函数方案，退回默认的 'radau'。
        solver.ops = generate_fr_operators(0, flux_point_type=getattr(solver, 'flux_type', 'radau'))
        solver.mesh.set_order(0)

        # 真实 bug 修复（2026-09-06）：上面第 434-439 行的 `np.mean` 压缩
        # 只是权宜的形状占位，不是壁面距离在 P0 下的正确值——见
        # `recompute_wall_distance_for_current_order` 文档完整推导。
        # `set_order(0)` 之后 `solver.mesh.sps_coords` 才反映 P0 真实的
        # 单 SP 坐标，这里重新查询覆盖掉那个被压缩、失真的值；如果没有
        # 缓存（没调用过 `compute_wall_distance_field`，或本来就没有
        # wall_distance），保留上面的均值压缩结果作为退化但形状正确的
        # 后备。
        from autoflowcfd.core.fr_solver.turbulence import recompute_wall_distance_for_current_order
        recompute_wall_distance_for_current_order(solver)

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
    for target_p in orders:
        print(f"\n--- Phase: P{target_p} ---")

        if target_p > 0 and target_p != solver.current_order:
            solver._interpolate_to_new_order(target_p)

        solver.current_order = target_p
        # flux_point_type 显式透传，见上面 P0 重置分支同一处修复的说明。
        solver.ops = generate_fr_operators(target_p, flux_point_type=getattr(solver, 'flux_type', 'radau'))

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
                    msg += " | " + format_scaled_residual_line(diag)
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
            return SolverResult(converged=True, iterations=total_iter, final_residual=final_residual)

    solver.order = original_order
    solver.ops = original_ops

    return SolverResult(converged=False, iterations=total_iter, final_residual=final_residual)
