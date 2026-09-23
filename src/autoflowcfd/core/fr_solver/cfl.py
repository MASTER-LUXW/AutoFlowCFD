"""
AutoFlowCFD V2.0 - FRSolver 局部时间步长计算 (从 fr_solver.py 拆分)

从 fr_solver.py 拆出来（控制单文件行数，>400 行需拆分的项目规范）。
签名以 `solver: FRSolver` 为第一参数，FRSolver 上保留同名薄委托方法，
调用方式不变。
"""

import numpy as np


def compute_local_time_step(solver, return_physical_too: bool = False):
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
       对粘性（分子+湍流）扩散项的稳定性时间步长是
       dt<=C*rho*6*V^2/(mu_eff*sum_f A_f^2)（各向异性正确形式，2026-09-23
       从 `V^(2/3)` 改；立方体上两者逐位相同，见下方 `Lc2` 处的完整推导）
       /mu_eff（抛物型稳定性条件），与上面的对流限制是完全独立的
       机制——粘性主导流动（低速层流、边界层内部）下这个限制可能
       严格得多，此前完全没有被施加过，真实复现：Couette 层流验证
       算例里这正是导致发散的根本原因之一（另一个是上面 0 提到的
       低马赫数刚性）。
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

    低马赫数伪时间预处理（2026-09-14 新增，见 `core/utils/preconditioning.py`
    模块末尾"伪时间预处理矩阵 Gamma"一节完整推导）：`solver.
    low_mach_precond_enabled` 为真时，平均流的 dt 改用**预处理后**的波速
    (|un| + c_precond) 而不是物理波速 (|un| + a)。这与上面第 0 条记录的
    2026-08-24 事故**不是同一件事**：那次只把 c_precond 塞进 CFL、没有改
    被积分的方程，必然失稳；现在 `step.py` 会把同一套 beta^2 定义下的
    预处理矩阵 Gamma 作用到平均流残差上（`dU/dtau = -Gamma R`），
    Jacobian 谱半径本身就变成了 (|un| + c_precond)，两者成对出现才成立。
    M=0.1 的外流场下这一对改动让平均流 dt 放大约 5 倍（(33+340)/(33+36)）。

    湍流标量（k/omega）拿到的仍然是**物理波速**算出的那份 dt（调用方传
    `return_physical_too=True` 取第二个返回值）：湍流输运是被动标量的
    对流+扩散，不含声学模态，本来就不该被声速限制——但它的显式更新里
    `transport_k/transport_omega` 刻意没有做 point-implicit 阻尼（见
    `turbulence/sst.py::update_fields` 文档），所以这里保持保守、不跟着
    放大，把预处理的收益严格限制在平均流上，避免把一个已经验证稳定的
    湍流更新推到未经验证的步长上。

    Returns:
        return_physical_too=False（默认）: dt_local，形状 (n_cells, n_sps)
        return_physical_too=True: (dt_mean_flow, dt_physical)——前者可能是
            预处理后的（未启用预处理时两者是同一个数组对象）
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
    # 自适应 CFL（2026-08-24）：从 solver 上的 AdaptiveCFLController 读取
    # 当前 CFL 数，替代此前硬编码 0.1。控制器根据残差历史自动调节 CFL，
    # 收敛好时逐步放大（加速收敛），恶化时缩小（保证稳定）。
    #
    # ===== 真实缺陷修复（2026-09-17）=====
    #
    # 原来这一行是 `... if _cfl_controller is not None else 0.1`，即
    # **没有控制器时静默用硬编码 0.1**，把调用方请求的 CFL 整个丢掉。
    # 于是 `adaptive_cfl=False` 的每一次运行都跑在 0.1 上：本次排查里两条
    # "CFL 0.10" 与 "CFL 0.05" 的平板边界层运行给出逐位相同的残差轨迹、
    # 在同一步（3187）发散，就是这个 bug 的直接证据；
    # `tests/validation/test_couette.py` 等全部 `adaptive_cfl=False`
    # 的用法同样一直静默跑在 0.1。
    #
    # 现在按优先级取：控制器（自适应开启）> `solver.fixed_cfl_number`
    # （自适应关闭时由构造函数记下的请求值）> 0.1（两者都没有的**替身
    # 对象**，例如诊断脚本/测试 stub；这一档会打一次警告，不再静默）。
    _cfl_controller = getattr(solver, '_cfl_controller', None)
    if _cfl_controller is not None:
        CFL = _cfl_controller.cfl_number
    else:
        _fixed = getattr(solver, 'fixed_cfl_number', None)
        if _fixed is not None:
            CFL = float(_fixed)
        else:
            CFL = 0.1
            if not getattr(solver, '_afcfd_cfl_fallback_warned', False):
                from loguru import logger

                logger.warning(
                    "[CFL] 求解器既没有自适应控制器也没有 "
                    "fixed_cfl_number，回退到 0.1。真实求解器不会走到这一"
                    "档（构造函数两条分支都会设其中之一），走到这里说明"
                    "调用方是个替身对象——若它本意是固定 CFL，请显式设 "
                    "`solver.fixed_cfl_number`，否则这个 0.1 与你请求的值"
                    "无关。"
                )
                try:
                    solver._afcfd_cfl_fallback_warned = True
                except Exception:
                    pass

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
    # 涡粘（若有）
    # dt_visc = 0.25*CFL*rho*Lc2/mu_eff，Lc2 见下方（各向异性正确的
    # `6*V^2/sum_f A_f^2`，立方体上等于 V^(2/3)）。
    mu_t_field = solver._get_turbulent_viscosity_field()  # None 或 (n_cells,n_sps)/(n_cells,mesh_n_sps)
    mu_molecular = solver.mu_molecular
    if mu_t_field is not None:
        if mu_t_field.shape[1] != n_sps:
            rep = int(np.ceil(n_sps / mu_t_field.shape[1]))
            mu_t_field = np.tile(mu_t_field, (1, rep))[:, :n_sps]
        mu_eff = mu_molecular + mu_t_field
    else:
        mu_eff = np.full_like(rho, mu_molecular)
    # 粘性长度尺度平方：**各向异性正确**的形式（2026-09-23 修复）。
    #
    # ## 曾经错在哪
    #
    # 原来用 `V^(2/3)`，那是**几何平均**边长的平方。对各向异性单元（边界层
    # 网格的贴壁棱柱是典型：壁法向 dy 远小于流向/展向）它远大于真正限制
    # 稳定性的那个方向的尺度，于是 `dt_visc` 被**高估**、粘性稳定性限制
    # 实际上没有被施加。
    #
    # ## 正确形式
    #
    # DG 扩散算子的谱半径按 `nu * sum_f (A_f/V) / h_f`、而 `h_f = V/A_f`，
    # 所以 `lambda_max ~ nu * sum_f A_f^2 / V^2`，即
    #
    #     dt_visc <= C * rho * V^2 / (mu_eff * sum_f A_f^2)
    #
    # 这与上面对流项那条 `dt = CFL*V/sum_f (|u_n|+a)*A_f` 是**同一套
    # 面基形式**，不是新引入的长度尺度定义。
    #
    # ## 归一化：立方体上**逐位**复现旧值，所以这是纯各向异性修正
    #
    # 边长 h 的立方体：`V = h^3`、`sum_f A_f^2 = 6h^4`，于是
    # `6*V^2/sum_f A_f^2 = h^2 = V^(2/3)`。乘那个 6 之后，各向同性网格上
    # 本项与修改前**完全相同** —— 因此 `0.25` 这个系数与
    # `order_factor_viscous` 的既有标定不需要重新标定，改动只在各向异性
    # 单元上生效。
    #
    # ## 实测量级
    #
    # 平板算例贴壁棱柱（dx=6.25e-2、dz=7.36e-2、dy=1.228e-2）：
    #   V = 2.824e-5、sum_f A_f^2 = 1.339e-5
    #   旧 V^(2/3)        = 9.24e-4
    #   新 6V^2/sum A_f^2 = 3.58e-4      -> 旧值高估 2.58 倍
    # 真实边界层网格的各向异性远超本算例：固定 dx/dz、令 dy -> 0 时
    # `sum_f A_f^2` 被两个封盖主导、趋于常数，于是新值 ~ 3*dy^2 而旧值
    # ~ (A_cap*dy)^(2/3)，比值按 `dy^(-4/3)` 增长 —— dy=1e-6 量级时高估
    # 三个数量级以上。这就是"本算例不咬人、真实网格会咬"的根据。
    sum_face_area_sq = np.zeros(n_cells, dtype=np.float64)
    np.add.at(sum_face_area_sq, owner_cells, face_areas * face_areas)
    if np.any(internal):
        np.add.at(sum_face_area_sq, neighbor_cells[internal],
                  face_areas[internal] * face_areas[internal])
    Lc2 = (6.0 * volumes * volumes
           / np.maximum(sum_face_area_sq, 1e-300))
    Lc2_expanded = np.tile(Lc2[:, np.newaxis], (1, n_sps))
    # **IP 罚项的刚性必须折进来**（2026-09-23，与内部面罚项同批）：
    # 罚项对粘性算子谱半径的贡献是 `c_ip*mu/h * A_f/V ~ c_ip*mu*A_f^2/V^2`，
    # 与基础扩散项 `mu/h^2` 同阶、只差一个 `c_ip` 量级的因子。实测拟合
    # （均匀基态、16 单元、原生 P1、纯粘性算子数值雅可比谱）：
    #
    #     c_eff:      0      10.67    21.33    42.67    85.33   170.67   341.33
    #     |lambda|max 22.6   730.8   1465.8   2935.8   5875.8  11756    23516
    #     拟合  |lambda|max ~= 22.6 + 68.6*c_eff  =>  放大 (1 + 3.03*c_eff) 倍
    #
    # 所以除以 `1 + 3*c_ip`。不折进来就是"算子变刚了但控制器不知道"，
    # 显式积分在粘性成为约束方的网格上会失稳 —— 项目 2026-09-05 给 omega
    # 壁面加显式 SIPG 罚项那次翻车（79 万单元真实网格 20 步内 omega_mean
    # 2.8e4 -> 5.4e11）就是同一类刚性没被 dt 反映。
    #
    # **本机真实网格实测代价为零**：plate_demo_les（179,237 单元）上粘性
    # 在折进罚项刚性之后仍然**一个单元都不是约束方**（最坏单元 dt_visc
    # 1.466e-07 vs 最坏 dt_adv 1.612e-09，余量 91 倍），全场 dt 由声学
    # 对流限制定在 4.9e-08。折进去是纯安全margin，不牺牲收敛速度。
    from autoflowcfd.core.fr_operators.flux_kernels import (
        resolve_viscous_ip_constant,
    )
    _ip_stiffness = 1.0 + 3.0 * resolve_viscous_ip_constant(poly_order)
    dt_visc = (0.25 * CFL * order_factor_viscous * rho * Lc2_expanded
               / np.maximum(mu_eff * _ip_stiffness, 1e-30))

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

    dt_physical = np.minimum(np.minimum(dt_advective, dt_visc), dt_geometric)

    if not getattr(solver, "low_mach_precond_enabled", False):
        return (dt_physical, dt_physical) if return_physical_too else dt_physical

    # === 预处理后的平均流 dt（见本函数文档"低马赫数伪时间预处理"一节）===
    # 只有对流项与几何/度量项里的波速需要换成预处理声速；粘性限制
    # （dt_visc）与声速无关，原样复用。
    # beta^2 必须按**速度模**取，不能按面法向速度 un 取——完整论证见
    # `preconditioning.py::preconditioned_sound_speed` 文档：Gamma 里的
    # beta^2 是速度模定义的，而 |un| <= |u| 使 beta^2(un) <= beta^2(|u|)，
    # 误用前者会让有效声速偏小、dt 被系统性高估（流动与面法向越斜越严重），
    # 与 2026-08-24 那次"步长与算子不成对"的事故同类。
    # （`preconditioned_acoustic_eigs` 从它的第一个参数自算 beta^2，服务的
    #  是逐面通量特征值，不能直接拿来定伪时间步长。）
    from autoflowcfd.core.utils.preconditioning import preconditioned_sound_speed
    mach_ref = solver.freestream["mach_ref"]
    vel_mag_owner = vel_mag[owner_cells, 0]
    c_pre_face = preconditioned_sound_speed(vel_mag_owner, a_o, mach_ref)
    spectral_p = np.zeros(n_cells, dtype=np.float64)
    np.add.at(spectral_p, owner_cells, (np.abs(un_owner) + c_pre_face) * face_areas)
    if np.any(internal):
        c_pre_nb = preconditioned_sound_speed(vel_mag[nc, 0], a_n, mach_ref)
        np.add.at(spectral_p, nc, (np.abs(un_neigh) + c_pre_nb) * face_areas[internal])
    spectral_p = np.maximum(spectral_p, 1e-30)
    dt_adv_p = np.tile((CFL * order_factor_advective * volumes / spectral_p)[:, np.newaxis],
                       (1, n_sps))

    # 几何/度量项：逐 SP 的波速同样换成预处理值，beta^2 同样按速度模取
    c_pre_sp = preconditioned_sound_speed(vel_mag, a, mach_ref)
    wave_speed_p = np.maximum(vel_mag + c_pre_sp, 1e-10)
    dt_geo_p = CFL * np.abs(det_jacs) / np.maximum(metric_flux_scale * wave_speed_p, 1e-300)

    dt_mean = np.minimum(np.minimum(dt_adv_p, dt_visc), dt_geo_p)
    return (dt_mean, dt_physical) if return_physical_too else dt_mean
