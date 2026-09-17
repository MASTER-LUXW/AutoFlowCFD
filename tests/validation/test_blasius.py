"""Blasius 平板层流边界层 —— 本项目第一个"能真正收敛 + 有精确解"的算例。

## 缺口

在这个文件之前，本项目**没有任何算例能同时满足"能收敛"与"有精确解可定量
核对"**：

  * `test_couette.py` 的稳态由**粘性扩散时标** `H^2/nu` 支配（比对流时标大
    三个数量级），显式格式需约 27 万步——该文件自己的文档就承认"无法在合理
    预算内定量收敛"，只验证了误差方向正确与不发散。
  * 三张真实 ANSA 网格（plate_demo / plate_demo_les / cube_demo）都是钝体
    正对来流，**本来就没有稳态解**；plate_demo 走完一个绕板特征时间还需约
    2 万步（见 `core/fr_solver/pseudotime_budget.py` 模块文档记录的那次
    误判）。
  * `test_isentropic_vortex.py` 验证空间精度阶数，但是无粘周期问题，不碰
    壁面、粘性、收敛。

平板边界层补的正是这一块：稳态由**对流时标** `L/U` 决定（信息只往下游传），
且有经典精确解 `cf = 0.664/sqrt(Re_x)`。

## 本文件的两层判据

**第一层（本文件里跑，进 CI）**：短程运行 + 与分辨率无关的**精确**判据。
这些判据不依赖是否已收敛：

  1. 参考解实现本身对文献值（`f''(0) = 0.33206`、`f'(eta)` 逐点）——现场
     RK4 打靶积分 Blasius 方程，不查表；
  2. `|w|` 必须**有界**（不是机器零——原因见下方"已知的、已量化的离散
     伪迹：展向速度"一节：网格的三角化不是 z 对称的）；
  3. 压力/密度全程正；
  4. 不发散；
  5. 进出口质量流量不平衡有界（未收敛时不为零，但必须小且在收敛）。

**第二层（长程运行，不进 CI）**：`cf(x)` 对 Blasius 的定量误差。需要约
3 个流过时间（约 5.5 万步、2 小时），脚本是
`scratchpad/run_blasius.py`，启动器 `scratchpad/launch_blasius.ps1`。
实测结果记录在下方"长程实测"一节，本文件只钉住量级不重跑。

## 长程实测（2026-09-17，nx=32, delta 内 6 层, P1, 2304 单元, Re_L=1e4）

### 这个算例意外地成了默认滤波档的决定性证据

`AFCFD_FILTER_MODE=off`（无任何滤波）下它**必然发散**，而且发散点由**伪
时间**而不是步数决定：

    CFL 0.10   第 3187 步发散   发散前 tau/T_flow = 0.164
    CFL 0.05   第 6381 步发散   发散前 tau/T_flow = 0.164

6381/3187 = 2.002 恰好是 CFL 之比，两条在**同一伪时间**崩溃，匹配伪时间处
的残差（2.6243e8 vs 2.6239e8）与压力范围（[73798,127582] vs [73512,127687]）
几乎相同。**所以那不是 CFL 失稳**——降 CFL 只是用两倍步数走到同一个物理
状态再炸。压力摆幅 ±26%（来流 101.3 kPa）在 Mach 0.09 的边界层流动里完全
非物理。

换成 `sensor` + `bounds`（2026-09-17 起的默认）：

    it=  3000  tau/T=0.164  res 1.415e6  p=[101316, 101538]
    it=  6000  tau/T=0.328  res 1.677e6  p=[101287, 101572]
    it=  9000  tau/T=0.492  res 1.707e6  p=[101298, 101579]
    it= 12000  tau/T=0.655  res 1.770e6  p=[101323, 101587]

跑到 `off` 发散点的 **4 倍伪时间**仍然干净，压力带全程约 ±135 Pa，残差已
基本平台化（每 3000 步 +2%）。这是 `sensor+bounds` 默认值的第二份真实算例
证据（第一份是 plate_demo_volume_les 上 legacy iter 112 发散 -> 216 步残差
单调降 3.5 倍）。

### 已知的、已量化的离散伪迹：展向速度

`_channel_mesh.build_channel_mesh_prism` 把每个 (x,z) 矩形按**固定对角线**
切成两个三角形，所以网格**不是 z 对称的**——离散解本来就有 z 相关分量，
`w = 0` 不是离散解的精确性质（连续问题的 w=0 解在 z 对称网格上才会被精确
保持）。实测它是**有界的**，不是失稳：

    步数      1      10      50     100     300     600    1000    1500    2000    3000
    |w|max/U  ~0  0.0010%  0.048%  0.170%  0.683%  1.154%  1.504%  1.760%  1.922%  2.103%

增量逐段变小，收敛到约 2.2%。另外已用隔离实验确认**两个残差算子都严格
保持 w=0**：给一个 z 无关、w=0 的精确可表示场，无粘 `rho_w` 残差 4.79e-6
（对比 `rho_u` 的 3.66e5，相对机器零）、粘性 `rho_w` 残差 4.71e-13；模态
滤波与低马赫预处理矩阵同样严格保持（`rho_w` 恒为 0）。所以 w 的来源只能是
网格自身的 z 不对称，不是算子缺陷。

因此本文件的判据用"有界"而不是"机器零"（见
`TestExactCriteriaOnAShortRun::test_spanwise_velocity_is_machine_zero` 的
容差与说明）。
"""

import numpy as np
import pytest

from ._blasius_case import (
    L_PLATE,
    P_INF,
    RHO_INF,
    U_INF,
    blasius_cf,
    blasius_delta99,
    blasius_fpp0,
    blasius_profile,
    blasius_thicknesses,
    build_blasius_solver,
    inlet_outlet_mass_flux,
    nu_for,
    wall_shear_profile,
)


class TestReferenceSolutionItself:
    """参考解的实现必须先自证正确，否则后面的对比毫无意义。"""

    def test_shooting_recovers_literature_fpp0(self):
        """`f''(0)` 打靶结果对文献值 0.33206。

        这个数**不是**硬编码进来的：`blasius_fpp0` 用 RK4 积分
        `f''' + 0.5 f f'' = 0` 并二分打靶把 `f'(inf)` 打到 1，所以它同时
        校验了积分器、边界条件和打靶逻辑。
        """
        assert blasius_fpp0() == pytest.approx(0.33206, abs=2e-5)

    def test_profile_matches_literature_table(self):
        """`f'(eta)` 对经典表值（Schlichting / White 教科书）。"""
        eta = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ref = np.array([0.3298, 0.6298, 0.8461, 0.9555, 0.9916])
        got = blasius_profile(eta)
        assert np.abs(got - ref).max() < 2e-4, dict(zip(eta.tolist(),
                                                        got.tolist()))

    def test_profile_is_monotone_and_saturates(self):
        """`f'` 单调不减、在 eta 大处趋 1。

        eta=8 处的容差按 Blasius 解自身的渐近性给：打靶把 `f'` 打到 1 是在
        `eta_max = 10` 处，所以 eta=8 处应当**略小于** 1（实测 0.9999963，
        差 3.7e-6）。第一版用 abs=1e-6 断言"等于 1"，把解的真实渐近行为
        当成了误差。
        """
        eta = np.linspace(0.0, 8.0, 81)
        fp = blasius_profile(eta)
        assert fp[0] == pytest.approx(0.0, abs=1e-12)
        assert np.all(np.diff(fp) >= -1e-12)
        assert 1.0 - 1e-4 < fp[-1] <= 1.0 + 1e-9, fp[-1]
        # eta=10（打靶点）处才应当精确是 1
        assert blasius_profile(np.array([10.0]))[0] == pytest.approx(
            1.0, abs=1e-8)

    def test_cf_is_the_standard_correlation(self):
        nu = nu_for(1.0e4)
        x = np.array([0.1, 0.5, 1.0])
        re_x = U_INF * x / nu
        assert np.allclose(blasius_cf(x, nu), 0.664 / np.sqrt(re_x))

    def test_thicknesses_have_the_standard_ratios(self):
        """delta99 : delta* : theta = 4.91 : 1.7208 : 0.6641（Blasius）。"""
        t = blasius_thicknesses(0.5, nu_for(1.0e4))
        assert t["delta_star"] / t["theta"] == pytest.approx(2.591, rel=1e-3)
        assert t["delta99"] / t["theta"] == pytest.approx(7.394, rel=1e-3)

    def test_delta_grows_as_sqrt_x(self):
        nu = nu_for(1.0e4)
        d = blasius_delta99(np.array([0.25, 1.0]), nu)
        assert d[1] / d[0] == pytest.approx(2.0, rel=1e-12)


class TestCaseConstruction:
    """算例的几何/分辨率是按 Blasius 反推的，不是拍的。"""

    def test_resolution_targets_are_met(self):
        solver, meta = build_blasius_solver(nx=16, cells_in_delta=6.0)
        assert meta["cells_in_delta_at_L"] == pytest.approx(6.0, rel=1e-9)
        # 通道高度必须远于边界层，否则零压梯度前提不成立
        assert meta["H"] / meta["delta99_at_L"] == pytest.approx(6.0, rel=1e-9)

    def test_reynolds_number_is_laminar(self):
        """Re_L 必须低于平板转捩的约 5e5，否则 Blasius 不适用。"""
        _, meta = build_blasius_solver(nx=8)
        assert meta["Re_L"] < 5.0e5

    def test_viscosity_is_derived_not_hardcoded(self):
        """mu 由目标 Re_L 反推（刻意不是空气真实值，见 `_blasius_case`
        模块文档），这里核对反推关系本身。"""
        _, meta = build_blasius_solver(nx=8, re_l=2.5e4)
        assert meta["nu"] == pytest.approx(U_INF * L_PLATE / 2.5e4)
        assert meta["mu"] == pytest.approx(RHO_INF * meta["nu"])


class TestExactCriteriaOnAShortRun:
    """与"是否已收敛"无关的精确判据 —— 短程运行即可断言。"""

    N_STEPS = 400

    @pytest.fixture(scope="class")
    def run(self):
        solver, meta = build_blasius_solver(nx=16, cells_in_delta=4.0,
                                            cfl=0.10)
        res = []
        for _ in range(self.N_STEPS):
            r = solver.step(1e-4)
            res.append(r)
            if not np.isfinite(r):
                break
        return solver, meta, np.asarray(res)

    def test_does_not_diverge(self, run):
        solver, meta, res = run
        assert len(res) == self.N_STEPS, f"第 {len(res)} 步残差非有限"
        assert np.all(np.isfinite(res))

    def test_spanwise_velocity_stays_bounded(self, run):
        """`|w|` 必须**有界**——但不是机器零，原因是网格本身。

        判据校准（如实记录）：第一版断言 `|w| < 1e-10*U`，实测 400 步后是
        0.283 m/s（0.94% U）而失败。追查结论是**判据不对，不是求解器坏**：

          * `_channel_mesh.build_channel_mesh_prism` 把每个 (x,z) 矩形按
            **固定对角线**切成两个三角形，所以网格**不是 z 对称的**——
            连续问题的 `w=0` 解只在 z 对称网格上才被离散精确保持；
          * 两个残差算子都已用隔离实验确认**严格保持 w=0**：给一个 z 无关、
            w=0 的精确可表示场，无粘 `rho_w` 残差 4.79e-6（对比 `rho_u` 的
            3.66e5，相对机器零）、粘性 `rho_w` 残差 4.71e-13；模态滤波与
            低马赫预处理矩阵同样严格保持（`rho_w` 恒为 0）；
          * 它是**饱和**的而不是失稳的：|w|max/U 在 100/300/600/1500/3000
            步分别是 0.170% / 0.683% / 1.154% / 1.760% / 2.103%，增量逐段
            变小，收敛到约 2.2%。

        所以这里断言"有界且远低于会掩盖真实缺陷的量级"。5% 这个上界取在
        实测饱和值（2.2%）的两倍多一点：真实的方向性缺陷（例如度量项写错
        方向、边界分组把 z 面误标）会给出 O(1) 的 w，挡得住。
        """
        solver, meta, res = run
        w = np.abs(np.asarray(solver.state.Q[..., 3]))
        assert np.all(np.isfinite(w))
        assert w.max() < 0.05 * U_INF, (
            f"|w|max = {w.max():.4e} ({100*w.max()/U_INF:.2f}% U) 超过 5% U"
            f"——实测的网格伪迹饱和在约 2.2%，超过这个量级说明是别的机制")

    def test_spanwise_velocity_is_a_minority_of_cells(self, run):
        """展向伪迹只应出现在少数单元里（实测 100 步时 392/1536 约 26%，
        但 |w| 的量级极小）。这里查它没有变成全场现象。"""
        solver, meta, res = run
        w = np.abs(np.asarray(solver.state.Q[..., 3]))
        frac_large = float(np.mean(w.max(axis=1) > 0.02 * U_INF))
        assert frac_large < 0.3, (
            f"{100*frac_large:.1f}% 的单元 |w| 超过 2% U —— 伪迹变成全场现象了")

    def test_pressure_and_density_stay_positive(self, run):
        solver, meta, res = run
        Q = np.asarray(solver.state.Q)
        assert Q[..., 0].min() > 0.0
        assert Q[..., 4].min() > 0.0

    def test_pressure_stays_within_physical_envelope(self, run):
        """压力不该超出 [p_inf - q_inf, 滞止压力 + 余量] 太多。

        容差按启动暂态给（非定常时局部压力**可以**超过滞止压力，见
        `pseudotime_budget.py`——定常判据不适用于暂态），所以这里只挡
        明显非物理的量级，不做定常判据。
        """
        solver, meta, res = run
        p = np.asarray(solver.state.Q[..., 4])
        q_inf = 0.5 * RHO_INF * U_INF ** 2
        assert p.min() > P_INF - 20.0 * q_inf
        assert p.max() < P_INF + 20.0 * q_inf

    def test_inlet_outlet_mass_flux_imbalance_is_small(self, run):
        """进出口质量流量不平衡：未收敛时不为零，但必须小。

        稳态下它必须趋零（长程运行的判据）；短程只断言"有界且量级合理"，
        不假装它是精确守恒检验。
        """
        solver, meta, res = run
        m_in, m_out, imb = inlet_outlet_mass_flux(
            solver, solver.mesh, L_PLATE, meta["H"], meta["Lz"])
        assert m_in > 0.0 and m_out > 0.0
        assert imb < 5.0e-2, f"in={m_in:.6e} out={m_out:.6e} imb={imb:.3e}"

    def test_wall_shear_is_positive_and_decays_downstream(self, run):
        """壁面剪应力必须为正，且沿流向单调减（Blasius 的 x^{-1/2}）。

        这条不依赖分辨率，只查**趋势**——定量误差要长程运行，见模块文档。
        """
        solver, meta, res = run
        x, cf, tau, y1 = wall_shear_profile(solver, solver.mesh)
        m = x > 0.3 * L_PLATE          # 上游启动区先排除
        assert m.sum() >= 3
        assert np.all(tau[m] > 0.0)
        # 允许离散噪声，用首尾对比而不是逐点单调
        assert cf[m][-1] < cf[m][0], f"cf 沿流向没有衰减：{cf[m]}"

    def test_wall_normal_velocity_is_small_at_the_wall(self, run):
        """贴壁第一层的 v 应远小于 U（边界层里 v/U ~ 1/sqrt(Re_x)）。"""
        solver, meta, res = run
        xyz = np.asarray(solver.mesh.sps_coords)
        y = xyz[..., 1]
        first = np.abs(y - y.min()) <= 1e-6 * (y.max() - y.min())
        v = np.asarray(solver.state.Q[..., 2])[first]
        assert np.abs(v).max() < 0.1 * U_INF, np.abs(v).max()
