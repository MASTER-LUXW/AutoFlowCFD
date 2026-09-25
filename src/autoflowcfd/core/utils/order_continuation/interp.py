"""AutoFlowCFD V2.0 - 阶数切换时解场/湍流场的插值（延拓）算子

从 `src/autoflowcfd/core/utils/order_continuation.py`(原 926 行)拆出(2026-09-24, 项目"单文件不超 500 行"规范)。**纯搬家, 逻辑未改**。
"""


import numpy as np

from typing import Any

from loguru import logger




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

    # 延拓算子**按基分派**（2026-09-20 修复的真实生产缺陷）：一维 Gauss
    # 点的张量积 Lagrange 插值只对**坍缩棱柱基**成立；native 四面体
    # （自 2026-09-03 起是四面体唯一实现）与 native 棱柱（2026-09-20 起
    # 是默认）的解点都不在那个张量积网格上。用线性场做判据实测 P1->P2
    # 的相对误差：坍缩棱柱 4.3e-16（对）、native 四面体 7.0e-01、
    # native 棱柱 1.4e-01。完整推导与零填充槽位的处理见
    # `fr/order_interp.py` 模块文档。
    #
    # P0->P1 不受这条缺陷影响（P0 是常数场，任何插值都给出同一个常数），
    # 所以走 `P0 -> P1 -> P2` 的生产运行只在最后那一跳被污染。
    from autoflowcfd.fr.order_interp import apply_order_interp

    n_prism_cells = int(solver.mesh.n_prism_cells)

    def _lift(field):
        return apply_order_interp(field, n_prism_cells, old_order, new_order)

    # 更新状态——(n_cells, old_n_sps, n_vars) -> (n_cells, new_n_sps, n_vars)
    solver.state.U = _lift(solver.state.U)
    solver.state.n_sps = new_n_sps
    solver.state.Q = np.zeros_like(solver.state.U)
    solver.state._update_primitives()

    # 更新湍流场（如果有）——(n_cells, old_n_sps) -> (n_cells, new_n_sps)
    if hasattr(solver.turb_model, 'k_field'):
        solver.turb_model.k_field = _lift(solver.turb_model.k_field)
        solver.turb_model.omega_field = _lift(solver.turb_model.omega_field)

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
        solver.turb_model.nu_t = _lift(solver.turb_model.nu_t)

    # 壁面距离场同样按每单元 SPs 存储（core/fr_solver_turbulence.py 的湍流
    # 源项计算直接按 SP 索引取值），阶数变化后形状同样必须一起插值——
    # 此前遗漏这一步，P0 阶段用均值压缩过的 (n_cells,1) 场会在阶数提升到
    # P1/P2 后与新的 SPs 数量不匹配，下一次湍流源项计算会形状不符崩溃
    # （真实网格已复现：与 mesh Jacobian 缺少按阶数重建是同一类"阶数变化
    # 后遗漏同步派生量"问题的另一处）。
    if getattr(solver, "wall_distance", None) is not None:
        # 这一步只是让形状先对上；真正的壁距在下面
        # `recompute_wall_distance_for_current_order` 里按新解点重新做
        # KD-Tree 查询（壁距是**纯几何量**，不是解多项式场，见该函数文档）。
        solver.wall_distance = _lift(solver.wall_distance)

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

    # 同理，NEWTON_KRYLOV 的 inexact-Newton forcing term 状态也必须失效：
    # 它记的是"上一步的残差范数"，而残差量级随阶数跳变（升阶会重新引入
    # 高阶内容、残差通常抬升一个量级以上）。沿用旧状态会让升阶后的第一步
    # 用一个按旧量级算出的线性容差 —— 要么过严（白花残差求值）、要么
    # 过松（Newton 方向不成方向）。干净地退回首步那档保守容差才是对的。
    if hasattr(solver, "_newton_forcing"):
        solver._newton_forcing = None
        solver._newton_last_info = None
        solver._newton_dtau_scale = 1.0
        # 块尺寸（真实解点数 x 变量数）随阶数变化，冻结的 `J_cc` 属于旧的
        # 离散空间，必须整体丢弃，由下一步按新阶数重建。
        solver._newton_block_precond = None
        # 隐式 k-omega 步的同类状态（forcing / dtau 缩放 / 块 Jacobi）同理
        solver._newton_turb_state = None

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
