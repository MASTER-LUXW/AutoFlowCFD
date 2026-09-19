"""Taylor-Green 涡（TGV）粘性动能衰减定量验证：真正三维、三方向周期、
真正粘性耦合的非定常流动，验证 FR 离散 + 三方向周期边界条件 + LDG 粘性
残差组合后，能否复现涡动能随粘性耗散单调衰减这一 TGV 最核心的物理
特征。

背景与设计取舍：
1. 三方向周期网格必须用四面体（不能像 Couette/等熵涡那样用棱柱）——
   棱柱只能沿单一挤出轴给出天然全等的封盖面，另外两个方向的边界会是
   `FaceExtractor` 按自身规则拆分的四边形侧面，两端拆分不保证互为
   平移镜像（本项目周期边界条件开发时在单方向棱柱网格上真实复现过
   这个失败模式）。四面体所有面天生是三角形，用与 (i,j,k) 无关的
   固定局部拆分模板（见 `_tgv_mesh.py`，复用已验证的
   `_channel_mesh.py::build_channel_mesh` 同一套模板）可以同时在
   x/y/z 三个方向给出天然全等的边界三角化，代价是要接受项目记忆
   `tet_collapsed_coord_anisotropy` 里记录的坍缩坐标各向异性风险——
   已通过独立的自由流场保持性 + 非均匀周期一致场无异常离群残差
   两项检查确认这套三方向周期四面体网格在本算例参数下没有触发
   该风险（见 test_tgv_freestream_preservation）。
2. 分子粘度：`FRSolver` 当前没有暴露配置 mu 的参数，
   `compute_viscous_residual`/`_compute_local_time_step` 各自独立硬编码
   mu=1.8e-5（详见项目记忆 hardcoded_molecular_viscosity_mismatch）。
   TGV 要在粗网格、有限步数预算内看到有意义的粘性衰减，需要一个远大于
   空气分子粘度的等效粘度（对应一个能被这套粗网格分辨的低雷诺数层流
   TGV，不是文献里 Re=1600 那种需要精细网格才能分辨的准湍流衰减曲线）
   ——本文件对两处硬编码统一 monkeypatch 成同一个真实 mu，避免重蹈
   声学 CFL/AUSM+up 不一致的覆辙。
3. 时间推进：与等熵涡一致，用全局（非逐单元局部）步长的显式 SSP-RK3，
   理由同样是局部时间步长是稳态收敛加速技术、会破坏时间精度。
4. 计算预算：实测（Re=20, n=4^3 网格, 150 步）动能比 KE/KE0 从初始
   短暂的数值适应小波动（第 0 步 1.069，正常的初场到离散 SPs 插值
   适应瞬态，不是发散迹象）后单调下降到 0.678（约 32% 净耗散），
   耗时 359s——用这组已验证的真实数据标定判据阈值，不追求复现文献
   Re=1600 那条需要 32^3+ 网格、上千个涡转时间积分预算的标准曲线。
"""
import numpy as np

from autoflowcfd.core.fr_solver import FRSolver
from autoflowcfd.core.fr_residual.viscous import compute_viscous_residual as _compute_visc_res
from autoflowcfd.core.time_integration import TimeIntegrationScheme

from ._tgv_mesh import build_triply_periodic_tet_mesh

ORDER = 2
N = 4
L = 2.0 * np.pi
LC = 1.0
RHO_INF, P_INF, U0 = 1.225, 101325.0, 30.0
GAMMA = 1.4
RE = 20.0
MU = RHO_INF * U0 * LC / RE
N_STEPS = 150


def _build_tgv_solver():
    mesh = build_triply_periodic_tet_mesh(order=ORDER, n=N, L=L)
    solver = FRSolver(
        mesh=mesh, order=ORDER, turb_model_name="NONE", n_vars=5,
        time_scheme=TimeIntegrationScheme.SSP_RK3,
        rho_inf=RHO_INF, vel_inf=U0, p_inf=P_INF,
    )
    solver.order_continuation_enabled = False

    # 见模块文档第 2 条：两处硬编码 mu 必须同步 monkeypatch。
    def _visc_res_with_mu():
        mu_t_field = solver._get_turbulent_viscosity_field()
        return _compute_visc_res(solver.state.U, solver.state.Q, solver.ops, solver.mesh,
                                  mu=MU, mu_t_field=mu_t_field)
    solver.compute_viscous_residual = _visc_res_with_mu

    orig_local_dt = solver._compute_local_time_step

    def _global_dt(return_physical_too: bool = False):
        """全局最小 dt（TGV 要求所有单元同步推进，见本函数下方说明）。

        `return_physical_too` 是生产 `step()` 的契约（2026-09-14 低马赫数
        伪时间预处理引入）：平均流用 `dt_local`、湍流标量更新用
        `dt_physical`。本算例是层流、且这里不做预处理放大，所以两者是
        同一个数组 —— 与生产实现在"未启用预处理时两者是同一个数组"这条
        约定一致（见 `core/fr_solver/cfl.py::compute_local_time_step`）。

        **这个参数此前缺失导致本测试整体跑不起来**（TypeError），于是
        四面体网格上唯一的稳定性/物理量回归算例长期处于"失败但原因是
        测试桩腐化"的状态 —— 2026-09-18 修复。
        """
        n_cells, n_sps, _ = solver.state.U.shape
        Q = solver.state.Q
        rho = Q[:, :, 0]
        u = Q[:, :, 1] / np.maximum(rho, 1e-10)
        v = Q[:, :, 2] / np.maximum(rho, 1e-10)
        w = Q[:, :, 3] / np.maximum(rho, 1e-10)
        p = (GAMMA - 1.0) * (Q[:, :, 4] - 0.5 * rho * (u**2 + v**2 + w**2))
        a = np.sqrt(np.maximum(GAMMA * p / np.maximum(rho, 1e-10), 1e-10))
        wave_speed = np.maximum(np.sqrt(u**2 + v**2 + w**2) + a, 1e-10)
        volumes = solver.mesh.get_all_cell_volumes()
        h = np.power(np.abs(volumes), 1.0 / 3.0)
        h_exp = np.tile(h[:, None], (1, n_sps))
        CFL = 0.1
        dt_adv = CFL * h_exp / wave_speed
        dt_visc = 0.25 * CFL * rho * h_exp**2 / MU
        dt = np.minimum(dt_adv, dt_visc)
        # 叠加原函数的几何/度量 CFL 限制（原函数同样接受这个 kwarg，
        # 取它的平均流那一份）
        dt = np.minimum(dt, orig_local_dt())
        dt_global = np.full_like(dt, dt.min())
        if return_physical_too:
            return dt_global, dt_global
        return dt_global

    solver._compute_local_time_step = _global_dt
    return solver, mesh


def _set_tgv_ic(solver, mesh):
    x = mesh.sps_coords[:, :, 0]
    y = mesh.sps_coords[:, :, 1]
    z = mesh.sps_coords[:, :, 2]
    u = U0 * np.sin(x / LC) * np.cos(y / LC) * np.cos(z / LC)
    v = -U0 * np.cos(x / LC) * np.sin(y / LC) * np.cos(z / LC)
    w = np.zeros_like(u)
    p = P_INF + (RHO_INF * U0**2 / 16.0) * (np.cos(2 * x / LC) + np.cos(2 * y / LC)) * (np.cos(2 * z / LC) + 2.0)
    rho = np.full_like(u, RHO_INF)
    E = p / (GAMMA - 1.0) + 0.5 * rho * (u**2 + v**2 + w**2)

    solver.state.U[:, :, 0] = rho
    solver.state.U[:, :, 1] = rho * u
    solver.state.U[:, :, 2] = rho * v
    solver.state.U[:, :, 3] = rho * w
    solver.state.U[:, :, 4] = E
    solver.state._update_primitives()


def _kinetic_energy(solver) -> float:
    Q = solver.state.Q
    rho = Q[:, :, 0]
    ke_density = 0.5 * rho * (Q[:, :, 1]**2 + Q[:, :, 2]**2 + Q[:, :, 3]**2)
    return float(ke_density.mean())


def _analytic_tgv_dissipation(n: int = 128):
    """TGV 初场的 `(eps, K)` 闭式/解析解，用解析导数在细网格上积分。

    完全不依赖被测代码（导数是手写的解析式），所以可以当独立判据。

        u =  U0 sin(x/LC) cos(y/LC) cos(z/LC)
        v = -U0 cos(x/LC) sin(y/LC) cos(z/LC)
        w =  0
        S_ij = 0.5 (d_i u_j + d_j u_i),   eps = 2 nu <S_ij S_ij>
        K = <rho/2 (u^2+v^2+w^2)>

    Returns:
        `(eps, K)`
    """
    nu = MU / RHO_INF
    g = (np.arange(n) + 0.5) * L / n
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    sx, cx = np.sin(X / LC), np.cos(X / LC)
    sy, cy = np.sin(Y / LC), np.cos(Y / LC)
    sz, cz = np.sin(Z / LC), np.cos(Z / LC)
    u = U0 * sx * cy * cz
    v = -U0 * cx * sy * cz
    ux = U0 * cx * cy * cz / LC
    uy = -U0 * sx * sy * cz / LC
    uz = -U0 * sx * cy * sz / LC
    vx = U0 * sx * sy * cz / LC
    vy = -U0 * cx * cy * cz / LC
    vz = U0 * cx * sy * sz / LC
    s12 = 0.5 * (uy + vx)
    s13 = 0.5 * uz
    s23 = 0.5 * vz
    ss = (ux ** 2 + vy ** 2
          + 2.0 * (s12 ** 2 + s13 ** 2 + s23 ** 2))
    eps = 2.0 * nu * float(ss.mean())
    K = 0.5 * RHO_INF * float((u ** 2 + v ** 2).mean())
    return eps, K


def test_tgv_kinetic_energy_decays_monotonically():
    """从标准 TGV 解析初场出发推进 150 个全局步，验证：(a) 全程数值
    稳定；(b) 排除第 0 步的初场-离散适应瞬态后，动能单调不增（真正的
    粘性耗散签名）；(c) 净衰减幅度落在实测校准范围内。
    """
    solver, mesh = _build_tgv_solver()
    _set_tgv_ic(solver, mesh)

    ke0 = _kinetic_energy(solver)
    ke_history = [ke0]
    dt_history = []
    for i in range(N_STEPS):
        dt_this = float(solver._compute_local_time_step()[0, 0])
        dt_history.append(dt_this)
        solver.step(dt_this)
        assert np.all(np.isfinite(solver.state.U)), f"solution diverged (NaN/Inf) at global step {i}"
        ke_history.append(_kinetic_energy(solver))

    # 跳过 step 0 的初场插值适应瞬态（真实观测：KE/KE0 短暂升到 ~1.07
    # 又回落，是初场在 SPs 上离散表示引入的一次性数值适应，不是物理
    # 现象），从 step 1 起要求动能单调不增。
    tail = ke_history[1:]
    increases = [tail[i] for i in range(1, len(tail)) if tail[i] > tail[i - 1] * 1.001]
    assert len(increases) == 0, f"kinetic energy increased after the initial transient: {increases}"

    # ===== 净衰减：用**解析耗散率**判，不用校准区间 =====
    #
    # 此前这里是 `0.5 < KE/KE0 < 0.85`，一个**校准值**。2026-09-18 查明
    # 它在两个前提下才成立，而两个前提都已不再成立：
    #   (a) `FILTER_MODE=legacy`（每个 RK stage 清掉一整阶），默认值已于
    #       2026-09-17 改成 `sensor`；
    #   (b) P2 四面体残差被机制3 **整体清零**（零填充槽位把中位数拖到 0，
    #       见 `core/fr_operators/troubled_cell.py::_outlier_ref_and_flag_
    #       kernel` 文档），也就是说当时根本没有物理演化 —— 那 15~50% 的
    #       "粘性衰减"记录的是滤波器的人工耗散。
    # 而本测试因为一处测试桩腐化（`_global_dt` 没跟上 `return_physical_too`
    # 契约）长期 TypeError、跑都跑不起来，所以这个过时一直没暴露。
    #
    # 换成解析判据：TGV 初场的动能耗散率有闭式解
    #     K   = <rho/2 (u^2+v^2+w^2)>
    #     eps = 2 nu <S_ij S_ij>,   dK/dt|_0 = -rho * eps
    # 本算例实测（150 步）走完的物理时间只有衰减时标 K/(rho*eps) 的
    # **1~3%**，所以净衰减本来就应当是**百分之几**，不可能是 15~50%。
    #
    # 三档实测（同一算例、同一初场）：
    #     off      K/K0=0.9892   dK/dt/解析 = +0.31~+0.54   单调、始终为负
    #     sensor   K/K0=1.0513   dK/dt/解析 = -3.1~-5.7     **能量增长**
    #     legacy   K/K0=0.8492   早期 dK/dt/解析 = +327      过耗散 300 倍
    # `off` 是唯一物理自洽的那一档（欠耗散可预期：4^3 个 P2 四面体对 TGV
    # 严重欠分辨，且这里的 K 是 SP 算术平均、与体积平均口径不同）。
    #
    # 判据取两条，都是**物理必须**、不是校准：
    #   1. 动能不得增长（滤波/离散只可能耗散，不可能产能）；
    #   2. 净衰减量级要与解析耗散率同量级（0.1~3 倍），既排除"几乎不耗散"
    #      也排除"过耗散一个数量级"。
    # 基准取 **step 1**（与上面的单调性判据同一口径）：step 0 那次是初场
    # 在 SPs 上的离散表示适应，一次性、与物理耗散无关（`project` 档实测
    # 跳到 1.0756 之后才开始正常衰减）。
    eps_ana, k_ana = _analytic_tgv_dissipation()
    ke_base = ke_history[1]
    t_total = sum(dt_history[1:])
    expect_drop = (RHO_INF * eps_ana / k_ana) * t_total     # 线性估计的相对衰减
    actual_drop = 1.0 - ke_history[-1] / ke_base
    assert actual_drop > 0.0, (
        f"动能净增长 {-actual_drop * 100:+.2f}%（相对 step 1；"
        f"KE/KE0={ke_history[-1] / ke0:.4f}）—— 滤波与离散耗散只可能"
        f"减少动能。"
        f"实测 `FILTER_MODE=sensor`（默认，BJ 判据在这个欠分辨光滑场上"
        f"100% 标记、等价于全局施加 mild 非幂等衰减）会出现这个现象，"
        f"`off` 与 `project` 都不会。")
    ratio = actual_drop / expect_drop
    assert 0.1 < ratio < 3.0, (
        f"净衰减 {actual_drop * 100:.3f}% 与解析耗散率给出的 "
        f"{expect_drop * 100:.3f}% 相差 {ratio:.2f} 倍（允许 0.1~3 倍）。"
        f"总物理时间 {t_total:.5e} s，占衰减时标 "
        f"{t_total / (k_ana / (RHO_INF * eps_ana)) * 100:.2f}%")


def test_tgv_freestream_preservation():
    """均匀自由流场（无粘/粘性残差理论上处处严格为零）保持性——独立
    验证三方向周期四面体网格本身没有引入虚假源项，也没有触发项目记忆
    tet_collapsed_coord_anisotropy 记录的坍缩坐标各向异性放大问题。

    无粘残差判据用相对量（/p_inf），阈值与
    tests/unit/test_fr_residual_inviscid.py::TestFreeStreamPreservation
    P=2 情形取同一个 3e-5——同一个 G-04（跨单元插值统一到坍缩坐标模态基
    +lu_solve）+S-02（体积项 over-integration）修复组合是这里舍入误差
    的共同来源，两处理应共享同一条已审查过的精度基线，不应各自定一套
    不可比的绝对阈值。实测 rel=1.30e-5（max|inv_res|=1.315，p_inf=101325），
    与另一测试文件实测的 1.06e-5 同一数量级，二者互相印证：这是该修复
    组合已知、有界的舍入噪声下限，不是本文件三方向周期配对（G-05）引入
    的新缺陷——已用诊断脚本核实：384 个单元中残差 > 1e-2（绝对）的既包含
    全部 168 个接触周期面的单元，也包含全部 216 个不接触周期面的内部
    单元，比例上没有随"是否接触周期面"系统性区分，说明放大源自坍缩坐标
    模态基本身、与周期配对逻辑无关。
    """
    solver, mesh = _build_tgv_solver()
    solver.state.U[:, :, 0] = RHO_INF
    solver.state.U[:, :, 1] = RHO_INF * U0
    solver.state.U[:, :, 4] = P_INF / (GAMMA - 1.0) + 0.5 * RHO_INF * U0**2
    solver.state._update_primitives()

    inv_res = solver.compute_inviscid_residual()
    visc_res = solver.compute_viscous_residual()
    rel_inv_res = np.max(np.abs(inv_res)) / P_INF
    assert rel_inv_res < 3e-5, f"rel_inv_res={rel_inv_res:.3e}"
    # 粘性残差的浮点噪声下限正比于 mu（应力张量本身是 mu 的线性函数）；
    # 本文件用的 mu=1.8375 Pa·s 比 Couette/等熵涡测试用的默认分子粘度
    # 1.8e-5 大约 1.0208e5 倍，直接沿用那两个测试的 1e-6 阈值不合理。
    # 用诊断脚本在同一套三方向周期四面体网格上把 mu 换回默认 1.8e-5
    # 单独测得 max|visc_res|=1.04e-7，乘以上述 mu 比值得 1.062e-2——与
    # 这里 mu=1.8375 时实测的 1.066e-2 只差 <1%，证实了"正比于 mu"这条
    # 线性关系，即这个量级是同一个 G-04/S-02 舍入噪声下限按 mu 线性缩放
    # 的结果，不是新缺陷。阈值取该缩放值的约 3 倍安全余量（与上面
    # rel_inv_res 判据、以及 test_fr_residual_inviscid.py 里同一原则
    # 一致），而不是照抄一个为不同 mu 标定的绝对阈值。
    assert np.max(np.abs(visc_res)) < 3e-2
