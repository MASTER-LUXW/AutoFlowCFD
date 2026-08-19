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
    0. 【已撤销】低马赫数预处理——2026-08-14 Couette 合成算例定量验证
       过程中真实复现并确认：这里曾经引入的 Weiss-Smith 预处理
       （`preconditioned_acoustic_eigs`）只用来放松 CFL *步长估计*，
       但实际参与残差计算的 AUSM+up 通量（core/fr_kernels.py::
       compute_ausm_up_flux）自身完全没有做任何 Weiss-Smith 预处理——
       它内部用的始终是*真实物理*声速 aL/aR（只有 Liou 2001 式的
       界面声速插值修正，调整耗散强度，不改变特征波速本身）。这两者
       不一致：CFL 步长按"预处理后、人为缩小的"波速估计出一个偏大
       的 dt，但真正被显式积分的却是未预处理、用真实声速主导刚性的
       AUSM+up 通量——真实复现（棱柱/四面体网格均可复现）：自由参考
       马赫数 mach_ref 越小（越贴近 Couette/Poiseuille 这类低速层流
       算例的真实工况），这个 dt 相对真实稳定性极限就越大，扫描
       参考速度 1~30 m/s 精确复现了这个失稳阈值（<~15 m/s 对应
       M<~0.044 必然在数步内 NaN，>=20 m/s 稳定）——不是"要更保守
       CFL"就能绕开的问题，是步长估计与实际被积分的物理不一致这一
       结构性缺陷。真正一致的做法需要连 AUSM+up 通量本身也做
       Weiss-Smith 预处理（改动数值通量本身，属于更大的算法工作，
       已记录待后续评估），在此之前 CFL 步长必须如实按*真实*声速
       估计，不能假装用了一套实际并未生效的预处理来"合法"放宽步长。
       wave_speed 现在恒为真实的 |u|+a（未预处理），与 AUSM+up 通量
       实际使用的特征波速一致。
    1. 对流 CFL（原有逻辑）：dt = CFL * h / wave_speed，h 用单元的
       精确求积体积——这是标准有限体积式估计，按"单元平均"尺度衡量。
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

    # 真实（未预处理）声学波速（见上方文档 0）：必须与 AUSM+up 通量
    # 实际使用的特征波速一致——那里从未做过 Weiss-Smith 预处理，CFL
    # 步长估计也不能假装做了。
    wave_speed = np.maximum(vel_mag + a, 1e-10)

    # 网格尺度：用 HighOrderMesh 的精确求积体积（不是"det(J)均值*8"近似），
    # Order Continuation 期间当前状态 n_sps 可能与网格 n_sps 不同，
    # 体积是逐单元量不受此影响，直接广播到当前 n_sps 即可。
    volumes = solver.mesh.get_all_cell_volumes()
    h = np.power(np.abs(volumes), 1.0 / 3.0)
    h_expanded = np.tile(h[:, np.newaxis], (1, n_sps))

    CFL = 0.1  # 保守的CFL数（P0 基准值）

    # 阶数相关的 CFL 收紧（此前完全缺失——全代码库搜索确认没有任何地方
    # 按 solver.current_order 收紧过这个常量）：h 用的是整个宏单元体积
    # V^(1/3)，与该单元内有多少个 solution point 无关，所以 P0（1 个 SP/
    # 单元）和 P1（8 个）、P2（27 个）用的是完全相同的 dt 公式——但显式
    # FR/DG 格式的稳定性极限本身随阶数增长（微分矩阵谱半径随 p 增大），
    # 标准结果是对流项稳定 CFL ~ 1/(2p+1)，扩散/粘性项因算子等效于二阶
    # 微分，谱半径按 p 的平方增长，稳定 CFL ~ 1/(2p+1)^2（Kopriva,
    # "Implementing Spectral Methods for PDEs" 对 DGSEM 稳定性的标准
    # 推导；FR 用同一套配置点，结论同样适用）。真实复现的失稳模式与此
    # 精确吻合：P0 阶段用同一个 CFL=0.1 大致还在稳定域内（收敛很慢，
    # 说明其实已经接近边界），Order Continuation 一旦插值到 P1，
    # 残差立刻在第 1 步跳增且不再下降，湍流交叉扩散项（数值上最刚性
    # 的子系统）最先溢出——这正是"同一个步长，稳定域已经收紧"的典型
    # 特征，不是插值或湍流模型本身的 bug。
    # 命名为 poly_order 而非 p——本函数前面已经用 p 表示压力
    # （line 79，solver.state.Q[:,:,4]），同名会遮蔽它，虽然当前压力变量
    # 用完即弃、不会产生真实 bug，但对以后维护是隐患。
    poly_order = getattr(solver, "current_order", 0)
    order_factor_advective = 1.0 / (2 * poly_order + 1)
    order_factor_viscous = 1.0 / (2 * poly_order + 1) ** 2

    dt_advective = CFL * order_factor_advective * h_expanded / wave_speed

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
    Lc2 = h_expanded ** 2  # V^(1/3) 的平方 = V^(2/3)
    dt_visc = 0.25 * CFL * order_factor_viscous * rho * Lc2 / np.maximum(mu_eff, 1e-30)

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
