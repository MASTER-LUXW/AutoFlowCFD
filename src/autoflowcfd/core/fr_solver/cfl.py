"""
AutoFlowCFD V2.0 - FRSolver 局部时间步长计算

签名以 `solver: FRSolver` 为第一参数，FRSolver 上保留同名薄委托方法，
调用方式不变。
"""

import numpy as np


def compute_local_time_step(solver, return_physical_too: bool = False):
    """
    计算局部时间步长（基于CFL条件）。

    真正的稳定性限制取三个独立机制中更严格的一个。波速用物理声速 |u_n|+a：显式积分的是物理通量，稳定性由
    物理通量的谱半径决定（只把 Weiss-Smith 预处理声速塞进 CFL、不改被积方程会把 dt 高估约 10 倍并失稳；
    AUSM+up 里的 Weiss-Smith 只修正低马赫数耗散，见 `wave_speed` 处说明）。启用伪时间低马赫预处理时的情形见下文。

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
       严格得多（Couette 层流验证算例曾因缺这一项而发散）。
    3. 几何/度量 CFL：逐解点用该 SP 自己的 det(J) 当作局部"体积"、sum_m ||adj(J)[SP,m,:]|| 当作局部"总通量
       面积"——标准有限体积 CFL `dt=CFL*V/Σ(A_f*(|u·n|+a))` 的逐点类比。按面的对流 CFL 用的是单元平均量，
       单元内度量逐点变化（棱柱有双线性侧面、曲边几何）时由这一项取单元内最严者；无粘残差
       `-div_comp/det(J)` 在 det(J) 小的点上被放大，单元平均步长感知不到。

    低马赫数伪时间预处理（见 `core/utils/preconditioning.py`
    模块末尾"伪时间预处理矩阵 Gamma"一节完整推导）：`solver.
    low_mach_precond_enabled` 为真时，平均流的 dt 改用**预处理后**的波速
    (|un| + c_precond) 而不是物理波速 (|un| + a)。这与上面"只改 CFL 不改方程"
    的失稳情形不同：`step.py` 会把同一套 beta^2 定义下的
    预处理矩阵 Gamma 作用到平均流残差上（`dU/dtau = -Gamma R`），
    Jacobian 谱半径本身就变成了 (|un| + c_precond)，两者成对出现才成立。
    M=0.1 的外流场下这一对改动让平均流 dt 放大约 5 倍（(33+340)/(33+36)）。

    湍流标量（k/omega）拿到的仍然是**物理波速**算出的那份 dt（调用方传
    `return_physical_too=True` 取第二个返回值）：湍流输运是被动标量的
    对流+扩散，不含声学模态，本来就不该被声速限制——但它的显式更新里
    `transport_k/transport_omega` 刻意没有做 point-implicit 阻尼（见
    `turbulence/sst/update.py::update_fields` 文档），所以这里保持保守、不跟着
    放大，把预处理的收益严格限制在平均流上，避免把一个已经验证稳定的
    湍流更新推到未经验证的步长上。

    Returns:
        return_physical_too=False（默认）: dt_local，形状 (n_cells, n_sps)，同一单元的各槽位取同一个值
            （对流/粘性/几何三项限制逐解点求出后在真实解点上取最小，与 GPU 后端同一定义）
        return_physical_too=True: (dt_mean_flow, dt_physical)——前者可能是
            预处理后的（未启用预处理时两者是同一个数组对象）
    """
    from .cfl_limits import local_time_step_limits
    from .cfl_viscous import viscous_length_scale_sq

    Q = solver.state.Q            # 原始变量 (rho, u, v, w, p)（`core/fr_solver/state.py::_update_primitives`）
    n_cells, n_sps = Q.shape[:2]

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
    from autoflowcfd.core.time_integration.adaptive_cfl.policy import current_cfl_number
    CFL = current_cfl_number(solver)
    poly_order = getattr(solver, "current_order", 0)
    volumes = solver.mesh.get_all_cell_volumes()
    fc = solver.mesh.face_connectivity
    face_normals = np.asarray(fc.normal)     # (n_faces, 3) 面法向量（模=面积，单位化后得法向）
    unit_normals = face_normals / np.maximum(np.linalg.norm(face_normals, axis=1, keepdims=True), 1e-30)

    # 有效粘度：分子粘度 + 湍流涡粘 + 人工粘性（rho*nu_av），见 `_get_cfl_viscosity_field` 文档
    mu_t_field = solver._get_cfl_viscosity_field()  # None 或 (n_cells, n_sps)
    mu_eff = (np.full((n_cells, n_sps), solver.mu_molecular) if mu_t_field is None
              else solver.mu_molecular + mu_t_field)
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
    # 公式本身在 `cfl_viscous.py`（CPU 与 GPU 后端共用一份）。
    Lc2 = viscous_length_scale_sq(volumes, fc.owner_cell, fc.neighbor_cell, fc.is_boundary, fc.area)
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
    # 度量与状态必须在同一组解点上（换阶时 `mesh.set_order` 与状态同步切换）。2026-10-21（I-76）以前不一致时
    # 把度量平铺 / 裁剪到状态宽度：不同阶数的解点下标互不对应，那样拼出的度量没有意义
    precond = getattr(solver, "low_mach_precond_enabled", False)
    dt_physical_sp, dt_mean_sp = local_time_step_limits(
        Q, volumes, fc.owner_cell, fc.neighbor_cell, fc.is_boundary, unit_normals, fc.area,
        solver.mesh.jacobians["det_jacs"].reshape(n_cells, solver.mesh.n_sps_per_cell),
        solver._get_metric_flux_scale(), mu_eff, Lc2, CFL, poly_order,
        mach_ref=solver.freestream["mach_ref"] if precond else None)

    # 一个单元一个步长：三项限制逐解点求出后，在单元的**真实**解点上取最小值再广播回全部槽位（原生四面体的零
    # 填充槽位冻结在初值，不参与）。2026-10-09 以前返回逐解点各不相同的步长：同一个单元的自由度由同一个多项式
    # 耦合，各走各的伪时间步与两个 GPU 后端（一直是单元内取最小）不是同一个离散。
    from autoflowcfd.fr.native_padding import order_from_n_sps, reduce_per_cell_over_real_sps

    n_prism = int(solver.mesh.n_prism_cells)
    sp_order = order_from_n_sps(n_sps)

    def _per_cell(dt_sp):
        cell_min = reduce_per_cell_over_real_sps(dt_sp, n_prism, sp_order, 'min')
        return np.repeat(cell_min[:, np.newaxis], n_sps, axis=1)

    dt_physical = _per_cell(dt_physical_sp)
    dt_mean = dt_physical if dt_mean_sp is dt_physical_sp else _per_cell(dt_mean_sp)
    return (dt_mean, dt_physical) if return_physical_too else dt_mean
