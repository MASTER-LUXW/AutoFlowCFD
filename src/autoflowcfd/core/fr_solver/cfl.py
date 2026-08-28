"""
AutoFlowCFD V2.0 - FRSolver 局部时间步长计算 (从 fr_solver.py 拆分)

从 fr_solver.py 拆出来（控制单文件行数，>400 行需拆分的项目规范）。
签名以 `solver: FRSolver` 为第一参数，FRSolver 上保留同名薄委托方法，
调用方式不变。
"""

import numpy as np


def compute_local_time_step(solver) -> np.ndarray:
    """
    计算局部时间步长（基于CFL条件）。

    真正的稳定性限制取三个独立机制中更严格的一个：

    （历史记录，第四次评审核实并更正：本节曾经短暂记录过一次
    "CFL 改用 Weiss-Smith 预处理声速 c_precond 替代物理声速 a" 的尝试，
    但这个改动在 2026-08-24 就已撤销——见下方 `wave_speed` 计算处
    "物理声速用于 CFL 估计（2026-08-24 修复）" 的完整说明：显式 SSP-RK3
    积分的是物理通量，稳定性由物理通量谱半径 |u_n|+a 决定，用预处理声速
    反而会把 dt 高估约 10 倍、导致失稳。此前这里的文档没有同步更新，
    仍描述着已经被撤销的设计，容易误导后来者"重新接入" c_precond 而
    复现同一个真实发生过的失稳——`low_mach_cfl_ausm_inconsistency` 项目
    记忆记录的正是这次事故。Weiss-Smith 预处理本身仍然在用，但只用于
    AUSM+up 通量本身的低马赫数耗散修正，不影响这里的 CFL 步长估计。）

    1. 对流 CFL（已升级为基于面的谱半径）：dt = CFL * V / sum_f(wave_speed_f * A_f)，
       替代此前的 dt = CFL * V^(1/3) / wave_speed。旧公式假设各向同性单元，
       对边界层薄棱柱单元高估稳定步长 ~100 倍（V^(1/3) ~ 1e-3 m vs 实际
       最小维度 ~1e-5 m），导致有效 CFL ~10 远超 SSP-RK3 稳定极限 ~1。
       新公式自然捕捉各向异性：薄面面积小 → 谱半径小 → dt 小。
    2. 粘性稳定性限制（新增，同样是修复真实存在的失稳）：显式格式
       对粘性（分子+湍流）扩散项的稳定性时间步长是 dt<=C*rho*V^(2/3)
       /mu_eff（抛物型稳定性条件），与上面的对流限制是完全独立的
       机制——粘性主导流动（低速层流、边界层内部）下这个限制可能
       严格得多，此前完全没有被施加过，真实复现：Couette 层流验证
       算例里这正是导致发散的根本原因之一（另一个是上面 0 提到的
       低马赫数刚性）。公式与 TimeIntegrator.local_time_step 一致。
    3. 几何/度量 CFL（此前已修复的失稳）：坍缩坐标下同一个
       四面体/棱柱单元内，不同 SP 的 det(J) 天然可以相差几百倍——
       已用完美正四面体数值验证，这是 Duffy 坍缩变换在 P=2 时的
       固有性质，与单元形状/网格质量无关，不是可以"修好"的缺陷。
       无粘残差公式 residual = -div_comp/det(J) 对*非均匀*流场（自由
       流场因离散GCL恒等式精确抵消是例外）在 det(J) 很小的 SP 处，
       把一个本身有界的参考空间通量散度 div_comp（真实网格实测量级
       ~0.01~0.3，不随 det(J) 一起等比例缩小——这是把 P 阶多项式
       微分矩阵套在"度量项(有理)×非常数流场"这个不再是低阶多项式的
       乘积上的固有混叠截断误差）放大到失稳量级——真实网格上单元
       509974/525292 等（det(J) 低至 ~2e-14）在仅 1% 幅度的温和非
       均匀扰动下，无粘残差被放大到 1e10~1e11 量级，用原有"单元
       平均体积"CFL 算出的步长完全无法感知、更谈不上限制这种
       SP 级别的刚性，几步之内必然发散为 NaN——已数值复现验证。
       标准有限体积 CFL 公式 dt=CFL*V/Σ(A_f*(|u·n|+a)) 在这里的
       直接类比：用该 SP 自己的 det(J) 当作局部"体积"，
       sum_m ||adj(J)[SP,m,:]|| 当作局部"总通量面积"。

    Returns:
        dt_local: 局部时间步长，形状 (n_cells, n_sps)
    """
    n_cells, n_sps, n_vars = solver.state.U.shape

    # 提取速度和声速。solver.state.Q 存的是原始变量 (rho, u, v, w, p)
    # （见 fr_state.py::_update_primitives），不是守恒变量——此前这里
    # 把已经是 u/p 的 Q[1]/Q[4] 当 rho*u/rho*E 又转换了一次，实测使
    # 声速被系统性低估（214.8 应为 340.3，来流 rho=1.225,u=30,p=101325
    # 时），对流/几何 CFL 步长被高估约 1.55 倍。直接读取即可。
    rho = solver.state.Q[:, :, 0]
    u = solver.state.Q[:, :, 1]
    v = solver.state.Q[:, :, 2]
    w = solver.state.Q[:, :, 3]
    p = solver.state.Q[:, :, 4]
    a = np.sqrt(np.maximum(1.4 * p / np.maximum(rho, 1e-10), 1e-10))

    vel_mag = np.sqrt(u**2 + v**2 + w**2)

    # 物理声速用于 CFL 估计（2026-08-24 修复）：
    # 此前用 Weiss-Smith 预处理后的 c_precond 替代物理声速 a 计算 CFL，
    # 导致 dt 被高估 ~10 倍（Mach 0.1 下 c_precond≈36 m/s vs a≈340 m/s），
    # 有效 CFL 从目标的 0.1 飙到 ~0.54，远超 SSP-RK3 稳定极限。
    # 诊断复现：CFL=0.1 在 Step 2 发散，CFL=0.05 在 Step 3 发散，
    # CFL=0.01（有效 CFL≈0.054）才稳定——与"有效 CFL ~0.5 触发失稳"
    # 精确吻合。
    #
    # 原理：显式 SSP-RK3 积分的是物理通量（AUSM+up 用真实声速 a 计算
    # 数值通量），其稳定性由物理通量的谱半径决定（|u_n|+a），不由
    # 预处理后的谱半径（|u_n|+c_precond）决定。预处理改善的是伪时间
    # 系统的条件数（加速收敛），不改变显式积分的稳定性限制。
    #
    # Weiss-Smith 预处理仍用于 AUSM+up 通量本身（改善低 Mach 数值
    # 耗散特性），但 CFL 估计必须用物理波速。
    wave_speed = np.maximum(vel_mag + a, 1e-10)

    # 基于面的谱半径（face-based spectral radius），
    # 替代此前的 V^(1/3) 各向同性假设。对于边界层薄棱柱单元，V^(1/3)
    # 比实际最小维度大约 100 倍，导致有效 CFL 远超 SSP-RK3 稳定极限。
    # 基于面的公式 dt = CFL * V / sum_f(wave_speed_f * A_f) 自然捕捉
    # 各向异性：薄面的面积小 → 谱半径小 → dt 小，物理正确。
    # 与 TimeIntegrator.local_time_step (base.py) 用同一公式。
    # 自适应 CFL（2026-08-24）：从 solver 上的 AdaptiveCFLController 读取
    # 当前 CFL 数，替代此前硬编码 0.1。控制器根据残差历史自动调节 CFL，
    # 收敛好时逐步放大（加速收敛），恶化时缩小（保证稳定）。无控制器时
    # 回退到固定 0.1（向后兼容，也用于诊断脚本的 monkey-patch 场景）。
    _cfl_controller = getattr(solver, '_cfl_controller', None)
    CFL = _cfl_controller.cfl_number if _cfl_controller is not None else 0.1

    # 阶数相关的 CFL 收紧：基于面的谱半径已经考虑了单元几何，
    # 但显式 FR/DG 格式的稳定性极限仍随阶数增长（微分矩阵谱半径随 p 增大），
    # 标准结果：对流项 CFL ~ 1/(2p+1)，粘性项 CFL ~ 1/(2p+1)^2。
    poly_order = getattr(solver, "current_order", 0)
    order_factor_advective = 1.0 / (2 * poly_order + 1)
    order_factor_viscous = 1.0 / (2 * poly_order + 1) ** 2

    volumes = solver.mesh.get_all_cell_volumes()

    fc = solver.mesh.face_connectivity
    face_areas = fc.area  # (n_faces,) 物理面面积
    face_normals = fc.normal  # (n_faces, 3) 面法向量（模=面积，单位化后得法向）
    # 归一化得到单位法向
    face_norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    face_unit_normals = face_normals / np.maximum(face_norms, 1e-30)

    # 每个面的波速（owner 侧）
    owner_cells = fc.owner_cell  # (n_faces,)
    is_bnd = fc.is_boundary  # (n_faces,)

    # wave_speed shape: (n_cells, n_sps) → 取 SP0 用于面级 CFL（P0 只有一个 SP）
    # 面谱半径 = (|u·n| + a) * A_f —— 标准有限体积 CFL 公式，
    # 使用物理声速（不用预处理声速，见上方文档）。
    a_o = a[owner_cells, 0]  # (n_faces,)
    vel_owner_x = u[owner_cells, 0]
    vel_owner_y = v[owner_cells, 0]
    vel_owner_z = w[owner_cells, 0]
    un_owner = (vel_owner_x * face_unit_normals[:, 0] +
                vel_owner_y * face_unit_normals[:, 1] +
                vel_owner_z * face_unit_normals[:, 2])
    # 谱半径贡献 = (|un| + a) * A_f
    spectral_per_face = (np.abs(un_owner) + a_o) * face_areas

    # 每个单元的谱半径 = sum of face contributions
    spectral = np.zeros(n_cells, dtype=np.float64)
    np.add.at(spectral, owner_cells, spectral_per_face)
    # 内部面：neighbor 侧也贡献
    neighbor_cells = fc.neighbor_cell  # (n_faces,) 边界面为 -1
    internal = ~is_bnd
    if np.any(internal):
        nc = neighbor_cells[internal]
        a_n = a[nc, 0]
        vel_neigh_x = u[nc, 0]
        vel_neigh_y = v[nc, 0]
        vel_neigh_z = w[nc, 0]
        un_neigh = (vel_neigh_x * face_unit_normals[internal, 0] +
                    vel_neigh_y * face_unit_normals[internal, 1] +
                    vel_neigh_z * face_unit_normals[internal, 2])
        spectral_per_face_neigh = (np.abs(un_neigh) + a_n) * face_areas[internal]
        np.add.at(spectral, nc, spectral_per_face_neigh)

    spectral = np.maximum(spectral, 1e-30)
    # 基于面的 CFL: dt = CFL * V / spectral
    dt_face = CFL * order_factor_advective * volumes / spectral
    # 广播到 (n_cells, n_sps)
    dt_advective = np.tile(dt_face[:, np.newaxis], (1, n_sps))

    # 粘性稳定性限制（见上方文档 2）：分子粘度 + 当前湍流模型给出的
    # 涡粘（若有），与 TimeIntegrator.local_time_step 用同一公式
    # dt_visc = 0.25*CFL*rho*V^(2/3)/mu_eff。
    mu_t_field = solver._get_turbulent_viscosity_field()  # None 或 (n_cells,n_sps)/(n_cells,mesh_n_sps)
    mu_molecular = solver.mu_molecular
    if mu_t_field is not None:
        if mu_t_field.shape[1] != n_sps:
            rep = int(np.ceil(n_sps / mu_t_field.shape[1]))
            mu_t_field = np.tile(mu_t_field, (1, rep))[:, :n_sps]
        mu_eff = mu_molecular + mu_t_field
    else:
        mu_eff = np.full_like(rho, mu_molecular)
    Lc2 = np.power(np.abs(volumes), 2.0 / 3.0)  # V^(2/3)
    Lc2_expanded = np.tile(Lc2[:, np.newaxis], (1, n_sps))
    dt_visc = 0.25 * CFL * order_factor_viscous * rho * Lc2_expanded / np.maximum(mu_eff, 1e-30)

    metric_flux_scale = solver._get_metric_flux_scale()  # (n_cells,n_sps)
    det_jacs = solver.mesh.jacobians["det_jacs"].reshape(n_cells, solver.mesh.n_sps_per_cell)
    # Order Continuation 期间当前状态 n_sps 可能与网格 n_sps 不同——
    # 度量场是网格固有量，跟当前解阶数无关，按需重复/裁剪到当前
    # n_sps（与上面 h_expanded 对体积的处理是同一原则）。
    if det_jacs.shape[1] != n_sps:
        rep = int(np.ceil(n_sps / det_jacs.shape[1]))
        det_jacs = np.tile(det_jacs, (1, rep))[:, :n_sps]
        metric_flux_scale = np.tile(metric_flux_scale, (1, rep))[:, :n_sps]
    dt_geometric = CFL * np.abs(det_jacs) / np.maximum(metric_flux_scale * wave_speed, 1e-300)

    return np.minimum(np.minimum(dt_advective, dt_visc), dt_geometric)
