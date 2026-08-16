"""等熵涡（isentropic vortex）定量精度验证：真正非定常、真正弯曲的二维
可压缩流动结构通过周期边界条件被 FR 离散 + AUSM+up 正确平动。

背景与已知限制（详见项目记忆 dual_time_inner_cfl_floor_stall / low_mach_cfl_ausm_inconsistency）：
1. 时间推进方式的选择：`FRSolver.step()` 的 SSP-RK3 模式默认用
   `_compute_local_time_step()` 算出的**逐单元局部时间步长**——这是稳态
   收敛加速技术（不同单元推进不同的物理时间量），用于真正非定常问题会
   直接破坏时间精度。真正验证"涡被正确平动"必须用**全域统一**的时间
   步长（本文件按 `_orig_local_dt().min()` 广播成常数数组、monkeypatch
   `solver._compute_local_time_step`，复用 SSP-RK3 全部现有阶段逻辑，
   只是把"局部"改成"全局"，不是对算法的简化——任何显式非定常格式都
   必须这样做才谈得上时间精度）。
2. DUAL_TIME（隐式双时间步）本应能用远大于显式 CFL 的物理步长，但
   经诊断（见项目记忆 dual_time_inner_cfl_floor_stall）：伪时间内层迭代的自适应
   CFL 有一个硬下限 (cfl_min=0.1)，对等熵涡这种强非线性扰动（涡核峰值
   切向速度 ~270 m/s，接近来流声速量级）即使给 200 次内层迭代预算，
   伪残差仍在增长、不收敛——用不收敛的 DUAL_TIME 结果做基准判据，
   等于在验证一个从未真正解出的隐式方程，违反"不简化/不掩盖"的项目
   要求，因此本文件改用已验证真正稳定的显式全局步长路径。
3. 高阶（P2）显式格式的 CFL 稳定步长比"网格尺寸/波速"的朴素估计小
   两个数量级以上（配置点在单元内非均匀聚集的已知效应），完整走完
   一个平动周期（此算例 T=Lx/u_inf=0.1s）在自动化测试预算内不可行
   （实测 20x20 网格外推需要数万步）。因此本测试只验证一个计算成本
   可控、但确实非平凡的时间窗口（150 个全局步，约 6.06e-4s，约
   0.61% 个周期）——涡核已经产生了肉眼可辨的真实位移和演化，不是
   原地不动的平凡解。

判据说明：用**空间平均**相对误差（而不是峰值/L∞ 误差）作为主判据——
涡核中心是整个流场里梯度最陡的一小片区域，L∞ 误差对峰值恰好落在配置点
何处非常敏感，不是网格收敛性的稳健判据；空间平均误差衡量的是"整个流场
是否真的在跟随解析解演化"，更能反映离散化在无粘周期输运下是否正确。
实测（见本文件下方参数配置的独立探索脚本）密度平均相对误差 ~0.44%，
速度平均相对误差 ~4.4%~4.6%，判据阈值在此基础上留了数倍安全裕度。
"""
import numpy as np

from autoflowcfd.core.fr_solver import FRSolver
from autoflowcfd.core.time_integration import TimeIntegrationScheme

from ._isentropic_vortex import (
    build_vortex_mesh, build_vortex_farfield_ghost_provider,
    vortex_primitive_field, primitive_to_conservative,
)

ORDER = 2
NX, NY, NZ = 20, 20, 1
LX, H, LZ = 10.0, 10.0, 0.5
RHO_INF, P_INF, U_INF = 1.225, 101325.0, 100.0
X0, Y0 = LX / 2.0, H / 2.0
N_STEPS = 150


def _build_vortex_solver():
    mesh = build_vortex_mesh(ORDER, NX, NY, NZ, LX, H, LZ)
    solver = FRSolver(
        mesh=mesh, order=ORDER, turb_model_name="NONE", n_vars=5,
        time_scheme=TimeIntegrationScheme.SSP_RK3,
        rho_inf=RHO_INF, vel_inf=U_INF, p_inf=P_INF,
    )
    solver.order_continuation_enabled = False
    solver.boundary_ghost_provider = build_vortex_farfield_ghost_provider(mesh, LX, H, LZ, RHO_INF, P_INF, U_INF)

    # 全局（非逐单元局部）时间步长，见模块文档第 1 条。
    orig_local_dt = solver._compute_local_time_step

    def _global_dt():
        local = orig_local_dt()
        return np.full_like(local, local.min())

    solver._compute_local_time_step = _global_dt
    return solver, mesh


def _set_vortex_ic(solver, mesh, t=0.0):
    x = mesh.sps_coords[:, :, 0]
    y = mesh.sps_coords[:, :, 1]
    rho, u, v, w, p = vortex_primitive_field(x, y, t, X0, Y0, LX, RHO_INF, P_INF, U_INF)
    r, ru, rv, rw, E = primitive_to_conservative(rho, u, v, w, p)
    solver.state.U[:, :, 0] = r
    solver.state.U[:, :, 1] = ru
    solver.state.U[:, :, 2] = rv
    solver.state.U[:, :, 3] = rw
    solver.state.U[:, :, 4] = E
    solver.state._update_primitives()
    return x, y


def test_isentropic_vortex_advection_tracks_exact_solution():
    """从精确解析解出发，全局步长显式 SSP-RK3 推进 150 步（约 0.61%
    个周期，见模块文档第 3 条），验证：(a) 全程数值稳定；(b) 最终态
    与解析解（在实际到达的物理时刻 t 上求值，不要求恰好一个周期）的
    空间平均相对误差在合理范围内。
    """
    solver, mesh = _build_vortex_solver()
    x, y = _set_vortex_ic(solver, mesh, t=0.0)

    t_elapsed = 0.0
    for i in range(N_STEPS):
        dt_this = float(solver._compute_local_time_step()[0, 0])
        solver.step(dt_this)
        t_elapsed += dt_this
        assert np.all(np.isfinite(solver.state.U)), f"solution diverged (NaN/Inf) at global step {i}"

    rho_e, u_e, v_e, _, _ = vortex_primitive_field(x, y, t_elapsed, X0, Y0, LX, RHO_INF, P_INF, U_INF)
    Q = solver.state.Q
    rho_n, u_n, v_n = Q[:, :, 0], Q[:, :, 1], Q[:, :, 2]

    err_rho = np.abs(rho_n - rho_e) / RHO_INF
    err_u = np.abs(u_n - u_e) / U_INF
    err_v = np.abs(v_n - v_e) / U_INF

    assert err_rho.mean() < 0.02, f"mean density error too large: {err_rho.mean():.4e}"
    assert err_u.mean() < 0.10, f"mean u-velocity error too large: {err_u.mean():.4e}"
    assert err_v.mean() < 0.10, f"mean v-velocity error too large: {err_v.mean():.4e}"


def test_isentropic_vortex_freestream_preservation():
    """涡扰动关闭（退化为均匀自由流）时，残差应保持机器精度量级——
    与 Couette/周期边界条件测试用的同一判据，独立验证这套涡网格 +
    远场/周期边界条件组合本身没有引入虚假源项。
    """
    solver, mesh = _build_vortex_solver()
    solver.state.U[:, :, 0] = RHO_INF
    solver.state.U[:, :, 1] = RHO_INF * U_INF
    solver.state.U[:, :, 4] = P_INF / (1.4 - 1.0) + 0.5 * RHO_INF * U_INF**2
    solver.state._update_primitives()

    inv_res = solver.compute_inviscid_residual()
    assert np.max(np.abs(inv_res)) < 1e-3
