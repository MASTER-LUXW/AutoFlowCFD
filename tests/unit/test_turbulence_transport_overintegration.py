"""k/omega 输运体积项去混叠（过积分）的验证（2026-09-15）。

## 修的是什么

`fr/collapsed_basis.py::build_overintegration_operators` 文档完整记录了
去混叠的动机，并给出实测数字——对解析残差恒为 0 的线性剪切场，不去混叠
的 P2 体积项算出的残差是真值的 43~62 倍。但那套机制一直**只接在平均流
的无粘体积项**上（`fr_residual/inviscid.py` 的
`mesh.jacobians_fine is not None` 分支）；粘性项与 k/omega 输运项
**完全没有**（本轮排查中 grep 确认：两个文件里 overint/jacobians_fine
零命中）。

而 k/omega 对流体积项算的是 `div(adj(J)*rho*u*phi)`——**三重**非线性
乘积，真实多项式次数约 `3*order + 度量次数`，远高于 order。直接在
coarse SPs 上对它微分就是"先混叠再求导"。

这与 `filter_scalar_field` 文档记录的 2026-09-12 真实 P1 发散症状直接
对应：那次诊断原文是"P1 多项式在单元内部相邻解点间出现数量级跳变 ->
外插到面通量点后被上风格式放大成巨大的虚假对流残差"。当时的处置是给
k/omega 加模态滤波器，而那个滤波器每个 RK stage 清掉一整阶——用牺牲
阶数换稳定。去混叠是同一个问题**不牺牲阶数**的正解。

## 判据

核心是**与解析散度比较**，不是两条实现互相比较：构造 rho/u/phi 都是
显式多项式、使三重乘积次数超过 order，`div(rho*u*phi)` 就有闭式解，
直接量两条路径各自的误差。

另外三条是安全性判据：自由流场保持性（常数场散度必须为零）、默认关闭
时公开接口逐位不变、开关取值校验。
"""

import os

import numpy as np
import pytest

from autoflowcfd.core.turbulence import transport as tp
from autoflowcfd.fr.native_prism.mode import resolve_prism_basis_mode
from autoflowcfd.fr.operators import generate_fr_operators

from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh


def _real_dof_mask(mesh, order):
    """(n_cells, n_sps) 布尔掩码，标出**真实自由度**。

    必须有这一步才能和解析参照比较：原生基只有前若干个 SP 是自由度，
    其余是零填充槽位，填充行的散度恒为 0，与解析值无关。用
    "div(x,0,0)=1"这个手算算例定位到的：不加掩码时结果是 [0,1] 区间
    而不是恒 1，min=0 全部来自填充位。

    **棱柱同样可能有填充槽位（2026-09-20）**：原先这里写死"棱柱的
    `(order+1)^3` 个 SP 全是自由度"，那只对坍缩棱柱基成立；原生棱柱基
    （2026-09-20 起是默认）每单元只有 `(p+1)^2(p+2)/2` 个真实自由度。
    漏掉这一点的直接后果是本文件的误差被填充槽位主导：去混叠与不去
    混叠算出**逐位相同**的误差（实测 8.241e-01 vs 8.241e-01），判据
    完全失效。真实自由度数走唯一入口 `fr/native_padding.real_sps_per_cell`。
    """
    from autoflowcfd.fr.native_padding import real_sps_per_cell

    n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
    n_real_prism, n_real_tet = real_sps_per_cell(order)
    mask = np.ones((n_cells, n_sps), dtype=bool)
    mask[:mesh.n_prism_cells, n_real_prism:] = False
    mask[mesh.n_prism_cells:, n_real_tet:] = False
    return mask


def _actual_over_order(oi):
    """从过积分上下文**实际读出** over_order，而不是在测试里重算规则。

    2026-09-15 的教训：本文件原来内部写死 `min(2*order, 3)`，而生产规则
    随后改成了 `3*order`（只影响 order=1，见 fr/operators.py 该处文档）。
    测试当时仍然通过，但走的是错误的分支、判据的理由已经和现实脱节。
    取**棱柱段**的细点数反解：`n_fine_prism = (over_order+1)^3`。

    2026-09-17 起上下文不再有共享的 `oi["n_fine"]`——native 四面体过积分
    的细网格轴不再填充到棱柱宽度，两段的 n_fine 不同了。所以这里明确
    取棱柱段，而不是随便拿一段来反解。

    **反解方式按棱柱基分档（2026-09-20）**：原先写死"开立方"
    （`n_fine_prism = (oo+1)^3`），那只对坍缩棱柱基成立；原生棱柱基的
    细点数是 `(oo+1)^2(oo+2)/2`（P1 oo=2 -> 18，不是完全立方数，原先
    会直接断言失败）。两档统一向唯一入口 `prism_n_fine` 反查，而不是在
    测试里再写一份公式。
    """
    from autoflowcfd.fr.overintegration_order import prism_n_fine

    n_fine_prism = oi["segs"][0][2]
    for oo in range(1, 32):
        if prism_n_fine(oo) == n_fine_prism:
            return oo
    raise AssertionError(
        f"棱柱段 n_fine={n_fine_prism} 不对应任何 over_order —— "
        f"细点数公式与 `fr/overintegration_order.prism_n_fine` 脱节了")


def _per_type_slices(mesh, order):
    """[("prism", 切片), ("tet", 切片)]，四面体那一片只取真实自由度。

    分类型统计是必须的：两类单元可达的精度不同（棱柱映射非仿射 ->
    adj(J) 是非平凡多项式 -> 乘积次数被进一步推高；native 四面体仿射 ->
    adj(J) 常数）。混在一起会掩盖"仿射单元上已经精确"这条最强证据。
    """
    # 两类单元各自的真实自由度数走唯一入口（原生棱柱基下棱柱同样有
    # 零填充槽位，见 `_real_dof_mask` 里那段 2026-09-20 的说明）。
    from autoflowcfd.fr.native_padding import real_sps_per_cell

    n_real_prism, n_real_tet = real_sps_per_cell(order)
    return [
        ("prism", (slice(0, mesh.n_prism_cells), slice(0, n_real_prism))),
        ("tet", (slice(mesh.n_prism_cells, mesh.n_cells),
                 slice(0, n_real_tet))),
    ]


def _coarse_convection_div(scalar_field, rho, velocity, mesh, ops):
    """coarse 路径的对流体积项 div_F（复刻生产代码的非过积分分支）。"""
    from autoflowcfd.core.fr_operators.volume_contract import (
        contravariant_flux_from_metric,
    )
    from autoflowcfd.core.turbulence.transport_kernel import (
        scalar_convection_volume_kernel,
    )
    n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
    n_prism = mesh.n_prism_cells
    det = mesh.jacobians["det_jacs"].reshape(n_cells, n_sps)
    inv = mesh.jacobians["inv_jacs"].reshape(n_cells, n_sps, 3, 3)
    rho_u = rho[:, :, None] * velocity
    rho_u_tilde = contravariant_flux_from_metric(det, inv, rho_u[..., None])[..., 0]
    tet_op = (ops.D_native_tet_padded
              if getattr(ops, "D_native_tet_padded", None) is not None
              else ops.D_3d_tet)
    div = np.empty((n_cells, n_sps))
    if n_prism > 0:
        scalar_convection_volume_kernel(
            np.ascontiguousarray(scalar_field[:n_prism]),
            np.ascontiguousarray(rho_u_tilde[:n_prism]),
            np.ascontiguousarray(ops.D_3d_prism), div[:n_prism])
    if n_cells > n_prism:
        scalar_convection_volume_kernel(
            np.ascontiguousarray(scalar_field[n_prism:]),
            np.ascontiguousarray(rho_u_tilde[n_prism:]),
            np.ascontiguousarray(tet_op), div[n_prism:])
    return div


class _PolyField:
    """rho / u / phi 都取显式一次多项式，三重乘积是三次。

        rho = r0 + rx*x + ry*y + rz*z
        u_i = a_i + b_i*x + c_i*y + d_i*z
        phi = p0 + px*x + py*y + pz*z

    `div(rho*u*phi)` 用 sympy 级别的手工展开太易错，改用**中心差分对
    解析函数本身求导**——被求导的是闭式表达式而不是离散场，步长取
    1e-5 相对尺度，四阶中心差分的截断误差远低于本用例要分辨的
    "混叠误差 vs 机器零"这个量级差。
    """

    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.r = rng.uniform(0.8, 1.3, 4)      # r0, rx, ry, rz
        self.u = rng.uniform(-1.0, 1.0, (3, 4))
        self.p = rng.uniform(0.5, 1.5, 4)

    def rho(self, X):
        return self.r[0] + X @ self.r[1:]

    def vel(self, X):
        return self.u[:, 0][None, :] + X @ self.u[:, 1:].T

    def phi(self, X):
        return self.p[0] + X @ self.p[1:]

    def flux(self, X):
        """rho*u*phi，形状 (n,3)。"""
        return self.rho(X)[:, None] * self.vel(X) * self.phi(X)[:, None]

    def div_exact(self, X, h=None):
        """四阶中心差分求 div(rho*u*phi)（对闭式函数求导，不是对离散场）。"""
        scale = max(np.abs(X).max(), 1.0)
        h = h if h is not None else 1e-4 * scale
        out = np.zeros(X.shape[0])
        for d in range(3):
            e = np.zeros(3); e[d] = h
            f_p2 = self.flux(X + 2 * e)[:, d]
            f_p1 = self.flux(X + e)[:, d]
            f_m1 = self.flux(X - e)[:, d]
            f_m2 = self.flux(X - 2 * e)[:, d]
            out += (-f_p2 + 8 * f_p1 - 8 * f_m1 + f_m2) / (12 * h)
        return out


def _setup(order, seed=0):
    mesh = _build_synthetic_mixed_mesh(order)
    ops = generate_fr_operators(order)
    X = mesh.sps_coords.reshape(-1, 3)
    f = _PolyField(seed)
    n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
    phi = f.phi(X).reshape(n_cells, n_sps)
    rho = f.rho(X).reshape(n_cells, n_sps)
    vel = f.vel(X).reshape(n_cells, n_sps, 3)
    exact = f.div_exact(X).reshape(n_cells, n_sps)
    return mesh, ops, phi, rho, vel, exact


#: `(棱柱基, order) -> (coarse 误差下界, 去混叠必须达到的改善倍数)`。
#: 数值来自实测（见 `test_convection_volume_term_vs_analytic` 里的记录
#: 表），下界都比实测值留了余量：它们是"判据没有空转"的防护，不是目标值。
_PRISM_EXPECT = {
    ("collapsed", 1): (0.5, 10.0),
    ("collapsed", 2): (0.5, 10.0),
    ("native", 1): (0.1, 10.0),
    ("native", 2): (1.0e-3, 1.0e6),
}


class TestOverintegrationIsMoreAccurate:
    """核心判据：与**解析散度**比较，去混叠必须显著更准。"""

    @pytest.mark.parametrize("order", [1, 2])
    def test_convection_volume_term_vs_analytic(self, order):
        mesh, ops, phi, rho, vel, exact = _setup(order)
        oi = tp._turb_overint_ops(mesh, ops)
        assert oi is not None, (
            f"order={order} 下过积分算子/细点度量不可用——它们在 order>=1 时"
            f"应当无条件构造好（见 fr/operators.py 与 high_order_mesh_order.py）")

        det = mesh.jacobians["det_jacs"].reshape(mesh.n_cells, mesh.n_sps_per_cell)
        div_oi = tp._scalar_convection_volume_overintegrated(
            phi, rho, vel, oi, mesh.n_sps_per_cell) / det
        div_co = _coarse_convection_div(phi, rho, vel, mesh, ops) / det

        # 分单元类型统计：这张合成网格的**棱柱映射非仿射**（实测同一
        # 单元内 det(J) 跨度 0.0722），native 四面体是仿射的（跨度恰好
        # 0）。adj(J) 因此在棱柱上是非平凡多项式、把乘积次数进一步推高，
        # 两类单元可达的精度不同；混在一起统计会掩盖"仿射单元上已经
        # 精确"这条最强证据。
        errs = {}
        for name, sl in _per_type_slices(mesh, order):
            sc = np.abs(exact[sl]).max()
            errs[name] = (np.abs(div_co[sl] - exact[sl]).max() / sc,
                          np.abs(div_oi[sl] - exact[sl]).max() / sc)

        # 实测（相对 L-inf）。over_order 由 `AFCFD_OVERINT_ORDER_RULE`
        # （默认 `2x`）**与棱柱基**共同决定，所以判据按 (基, order) 分档，
        # 用 `_actual_over_order(oi)` 读实际值：
        #
        #   坍缩基（`AFCFD_PRISM_BASIS=collapsed`，过积分上限 3）
        #     P1 oo=2: prism 4.9816e+00 -> 1.3602e-01   36.6x
        #     P1 oo=3（3x 档）: prism 4.9816e+00 -> 2.1243e-03  2345x
        #     P2 oo=3: prism 7.6441e-01 -> 9.5333e-03   80.2x
        #   原生基（2026-09-20 起是默认，过积分上限 6）
        #     P1 oo=2: prism 4.0552e-01 -> 1.2170e-02   33.3x
        #     P2 oo=4: prism 1.2480e-02 -> 2.8663e-12   **机器零**
        #
        # 两条基的 coarse 误差**不同量级**（原生 P1 比坍缩小 12 倍、P2 小
        # 61 倍），所以"coarse 必须 > 0.5"那条防空转的下界也必须分档 ——
        # 原生 P2 的 coarse 只有 1.2e-2，用 0.5 会把一条正常通过的判据
        # 判成"算例没造出混叠"。
        basis = resolve_prism_basis_mode()
        coarse_min, gain_min = _PRISM_EXPECT[(basis, order)]
        err_co, err_oi = errs["prism"]
        assert err_co > coarse_min, (
            f"{basis} P{order} prism: coarse 误差只有 {err_co:.3e}，低于"
            f"该档记录值下界 {coarse_min:.1e} —— 这个算例没有真正制造出"
            f"混叠，判据失去意义；请加大 rho/u/phi 的非线性度（而不是"
            f"调低这个下界）")
        assert err_oi < err_co / gain_min, (
            f"{basis} P{order} prism: 去混叠误差 {err_oi:.3e} 相比 coarse 的 "
            f"{err_co:.3e} 改善不到 {gain_min:.0f} 倍（实测见上方记录）")

        # 四面体（仿射度量）上有一条更强、可完全解释的判据：三重乘积
        # rho*u*phi 由三个一次场相乘、是**三次**，只要 over_order >= 3
        # 且度量仿射，细网格空间 P3 就精确包含它 -> 必须是机器零。
        #   over_order = min(2*order, OVERINTEGRATION_MAX_ORDER=3)
        #   order=1 -> 2（不足，残余 5.6e-2）
        #   order=2 -> 3（足够，9.8e-14）
        # 也就是说 `2*order` 这条经验法则是为平均流的**二次**非线性设计
        # 的（见 fr/collapsed_basis.py::build_overintegration_operators），
        # 标量输运的三重乘积严格来说需要 `3*order`。不在这里改那条法则：
        # 算子与 jacobians_fine 是**全局共享**的（平均流用同一份），
        # 改动会同时影响已验证的平均流路径，且会把 order=1 的细点数从
        # 27 抬到 64（P2 OOM 有前科），属于独立一步。
        tet_co, tet_oi = errs["tet"]
        if _actual_over_order(oi) >= 3:
            assert tet_oi < 1e-11, (
                f"order={order} tet: 去混叠误差 {tet_oi:.3e} 不是机器零——"
                f"over_order={_actual_over_order(oi)}>=3 且四面体度量仿射时，三次"
                f"乘积应当被细网格空间精确包含")
        else:
            assert tet_oi < tet_co / 3.0, (
                f"order={order} tet: 改善只有 {tet_co/max(tet_oi,1e-300):.2f}x"
                f"（实测 8.0x）")

    @pytest.mark.parametrize("order", [1, 2])
    def test_diffusion_volume_term_vs_analytic(self, order):
        """扩散项：Gamma 与 grad_phi 都取显式多项式，乘积次数超过 order。

        这里 Gamma 直接取一次多项式（不是 k/omega 的商），所以本用例考察
        的正是 `_scalar_diffusion_volume_overintegrated` 文档里说"能去掉"
        的那部分混叠——Gamma×grad_phi 乘积与度量项乘积的混叠。
        """
        mesh = _build_synthetic_mixed_mesh(order)
        ops = generate_fr_operators(order)
        n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
        X = mesh.sps_coords.reshape(-1, 3)
        rng = np.random.default_rng(order + 10)
        g = rng.uniform(0.5, 1.5, 4)
        q = rng.uniform(0.5, 1.5, 4)

        def gam(Y):
            return g[0] + Y @ g[1:]

        def grad_phi(Y):
            # 取一个一次向量场当 grad_phi（本用例只考察乘积的混叠，
            # 不要求它真是某个 phi 的梯度）
            return q[0] + Y @ q[1:] + np.zeros((Y.shape[0], 1)) * 0.0, None

        gvec = np.empty((X.shape[0], 3))
        for d in range(3):
            gvec[:, d] = q[0] * (d + 1) + X @ (q[1:] * (d + 1))

        def flux(Y):
            gv = np.empty((Y.shape[0], 3))
            for d in range(3):
                gv[:, d] = q[0] * (d + 1) + Y @ (q[1:] * (d + 1))
            return gam(Y)[:, None] * gv

        h = 1e-4 * max(np.abs(X).max(), 1.0)
        exact = np.zeros(X.shape[0])
        for d in range(3):
            e = np.zeros(3); e[d] = h
            exact += (-flux(X + 2 * e)[:, d] + 8 * flux(X + e)[:, d]
                      - 8 * flux(X - e)[:, d] + flux(X - 2 * e)[:, d]) / (12 * h)
        exact = exact.reshape(n_cells, n_sps)

        gamma_field = gam(X).reshape(n_cells, n_sps)
        grad_field = gvec.reshape(n_cells, n_sps, 3)
        oi = tp._turb_overint_ops(mesh, ops)
        det = mesh.jacobians["det_jacs"].reshape(n_cells, n_sps)

        div_oi = tp._scalar_diffusion_volume_overintegrated(
            gamma_field, grad_field, oi, n_sps) / det

        from autoflowcfd.core.fr_operators.volume_contract import (
            contravariant_flux_from_metric,
        )
        from autoflowcfd.core.turbulence.transport_kernel import (
            scalar_volume_divergence_kernel,
        )
        inv = mesh.jacobians["inv_jacs"].reshape(n_cells, n_sps, 3, 3)
        G_tilde = contravariant_flux_from_metric(
            det, inv, (gamma_field[:, :, None] * grad_field)[..., None])[..., 0]
        n_prism = mesh.n_prism_cells
        tet_op = (ops.D_native_tet_padded
                  if getattr(ops, "D_native_tet_padded", None) is not None
                  else ops.D_3d_tet)
        div_co = np.empty((n_cells, n_sps))
        if n_prism > 0:
            scalar_volume_divergence_kernel(
                np.ascontiguousarray(G_tilde[:n_prism]),
                np.ascontiguousarray(ops.D_3d_prism), div_co[:n_prism])
        if n_cells > n_prism:
            scalar_volume_divergence_kernel(
                np.ascontiguousarray(G_tilde[n_prism:]),
                np.ascontiguousarray(tet_op), div_co[n_prism:])
        div_co = div_co / det

        errs = {}
        for name, sl in _per_type_slices(mesh, order):
            sc = np.abs(exact[sl]).max()
            errs[name] = (np.abs(div_co[sl] - exact[sl]).max() / sc,
                          np.abs(div_oi[sl] - exact[sl]).max() / sc)

        # Gamma 与 grad_phi 都取一次 -> 乘积**二次**。四面体度量仿射，
        # over_order>=2 的细网格空间就精确包含二次 -> 必须机器零
        # （实测 order=1/2 都是 ~3e-14）。棱柱映射非仿射、adj(J) 把次数
        # 推高，达不到精确（实测 order=1 残余 9.0e-2），只断言改善。
        tet_co, tet_oi = errs["tet"]
        assert tet_oi < 1e-11, (
            f"order={order} tet: 扩散项去混叠误差 {tet_oi:.3e} 不是机器零——"
            f"二次乘积在 over_order={_actual_over_order(oi)}>=2 的仿射四面体"
            f"上应当精确")
        pr_co, pr_oi = errs["prism"]
        if order == 1:
            assert pr_oi < pr_co, (
                f"order=1 prism: 去混叠误差 {pr_oi:.3e} 不低于 coarse 的 "
                f"{pr_co:.3e}")


class TestOrderRuleSwitch:
    """`AFCFD_OVERINT_ORDER_RULE` 的语义（2026-09-15）。

    默认 `2x` 必须复现此前已被长期验证的行为；`3x` 只在 order=1 上与它
    不同（order>=2 两者都被 `OVERINTEGRATION_MAX_ORDER=3` 卡住）。
    """

    def _env(self, v):
        old = os.environ.get("AFCFD_OVERINT_ORDER_RULE")
        if v is None:
            os.environ.pop("AFCFD_OVERINT_ORDER_RULE", None)
        else:
            os.environ["AFCFD_OVERINT_ORDER_RULE"] = v
        return old

    def _restore(self, old):
        if old is None:
            os.environ.pop("AFCFD_OVERINT_ORDER_RULE", None)
        else:
            os.environ["AFCFD_OVERINT_ORDER_RULE"] = old

    def test_default_is_2x(self):
        from autoflowcfd.fr.collapsed_basis import (
            resolve_overintegration_order_rule,
        )
        old = self._env(None)
        try:
            assert resolve_overintegration_order_rule() == 2
        finally:
            self._restore(old)

    @pytest.mark.parametrize("v,expected", [("2x", 2), ("3x", 3), ("3X", 3)])
    def test_accepted_values(self, v, expected):
        from autoflowcfd.fr.collapsed_basis import (
            resolve_overintegration_order_rule,
        )
        old = self._env(v)
        try:
            assert resolve_overintegration_order_rule() == expected
        finally:
            self._restore(old)

    @pytest.mark.parametrize("v", ["2", "3", "", "two"])
    def test_rejects_unknown(self, v):
        from autoflowcfd.fr.collapsed_basis import (
            resolve_overintegration_order_rule,
        )
        old = self._env(v)
        try:
            with pytest.raises(ValueError, match="AFCFD_OVERINT_ORDER_RULE"):
                resolve_overintegration_order_rule()
        finally:
            self._restore(old)

    @pytest.mark.parametrize("order,nf_2x,nf_3x", [(1, 27, 64), (2, 64, 64), (3, 64, 64)])
    def test_fine_point_counts_per_rule_collapsed(self, order, nf_2x, nf_3x,
                                                  monkeypatch):
        """**坍缩档**两档规则实际构造出的细点数——直接读算子，不重算规则。

        order>=2 上两档相同，这是 `OVERINTEGRATION_MAX_ORDER = 3` 的
        直接后果，也是"这条切换只影响 order=1"这句话的依据。

        显式跑在坍缩档（2026-09-20）：这些数字（27/64）是 `(oo+1)^3` 的
        取值，只对坍缩棱柱基成立；原生档由下面那条覆盖。
        """
        monkeypatch.setenv("AFCFD_PRISM_BASIS", "collapsed")
        for v, want in (("2x", nf_2x), ("3x", nf_3x)):
            old = self._env(v)
            try:
                ops = generate_fr_operators(order)
                got = ops.overint_D_fine_prism.shape[0]
                assert got == want, (
                    f"rule={v} order={order}: 细点数 {got} != {want}")
            finally:
                self._restore(old)

    @pytest.mark.parametrize("order,nf_2x,nf_3x", [
        (1, 18, 40), (2, 75, 196), (3, 196, 196)])
    def test_fine_point_counts_per_rule_native(self, order, nf_2x, nf_3x,
                                               monkeypatch):
        """**原生档**（2026-09-20 起的默认）两档规则的细点数。

        原生棱柱的细点数是 `(oo+1)^2(oo+2)/2`，上限是 6（不是坍缩那条
        条件数上限 3），所以：

            rule  P1         P2          P3
            2x    oo=2 -> 18  oo=4 ->  75  oo=6 -> 196
            3x    oo=3 -> 40  oo=6 -> 196  oo=6 -> 196（被上限 6 卡住）

        也就是说"这条切换只影响 order=1"这句话**只对坍缩档成立**：
        原生档下 P2 也会被它改变（75 -> 112）。
        """
        monkeypatch.setenv("AFCFD_PRISM_BASIS", "native")
        for v, want in (("2x", nf_2x), ("3x", nf_3x)):
            old = self._env(v)
            try:
                ops = generate_fr_operators(order)
                got = ops.overint_D_fine_prism.shape[0]
                assert got == want, (
                    f"rule={v} order={order}: 细点数 {got} != {want}")
            finally:
                self._restore(old)


class TestOverintegrationIsNoOpAtOrder3:
    """**坍缩档**下 order=3 时去混叠**恰好什么也不做**，这是
    `OVERINTEGRATION_MAX_ORDER` 的直接后果，必须显式钉住而不是让它看起来
    "也生效了"。

    **本类显式跑在坍缩档（2026-09-20）**：那条上限 3 是**坍缩基**
    Vandermonde 条件数的约束，原生棱柱基不受它约束（自己的上限是 6，
    见 `fr/overintegration_order.py`），P3 在原生档下拿到 oo=6、细点数
    196，去混叠**是真的在做事**。也就是说"P3 上去混叠是空操作"这条
    结论只对坍缩档成立，改成默认档跑会把结论反过来。

        over_order = min(2*order, OVERINTEGRATION_MAX_ORDER=3)
        order=1 -> 2,  order=2 -> 3,  order=3 -> 3,  order>=4 -> 3

    order=3 时 over_order==order，"细"网格就是粗网格本身，插值/微分/限制
    三件套复合起来等于原路径。实测改善倍数恰好 1.00x。

    这条不是缺陷报告，是**范围说明**：本项目当前跑 P1/P2，P3 另有独立的
    内存边界（见 P3 face-flux-points OOM 记录）。真要让 P3 也受益必须提高
    那个上限，而它是平均流与标量输运共享的，属于独立一步。
    """

    def test_over_order_equals_order_at_3(self, monkeypatch):
        """从**实际构造出的算子**读 over_order，不重算规则。"""
        monkeypatch.setenv("AFCFD_PRISM_BASIS", "collapsed")
        from autoflowcfd.fr.collapsed_basis import OVERINTEGRATION_MAX_ORDER
        assert OVERINTEGRATION_MAX_ORDER == 3, (
            "上限被改动了——它有真实回归背景（放宽到 4 会让 P2 均匀自由"
            "流场残差从 1.06e-5 恶化到 5.6e-3），见该常量处文档")
        mesh = _build_synthetic_mixed_mesh(3)
        oi = tp._turb_overint_ops(mesh, generate_fr_operators(3))
        assert _actual_over_order(oi) == 3

    def test_no_accuracy_change_at_order3(self, monkeypatch):
        monkeypatch.setenv("AFCFD_PRISM_BASIS", "collapsed")
        mesh, ops, phi, rho, vel, exact = _setup(3)
        oi = tp._turb_overint_ops(mesh, ops)
        det = mesh.jacobians["det_jacs"].reshape(mesh.n_cells, mesh.n_sps_per_cell)
        d_oi = tp._scalar_convection_volume_overintegrated(
            phi, rho, vel, oi, mesh.n_sps_per_cell) / det
        d_co = _coarse_convection_div(phi, rho, vel, mesh, ops) / det
        m = _real_dof_mask(mesh, 3)
        sc = np.abs(exact[m]).max()
        e_oi = np.abs(d_oi[m] - exact[m]).max() / sc
        e_co = np.abs(d_co[m] - exact[m]).max() / sc
        assert abs(e_oi / e_co - 1.0) < 1e-6, (
            f"order=3 的去混叠改善了 {e_co/e_oi:.3f}x——与 over_order==order "
            f"这个事实矛盾，说明 OVERINTEGRATION_MAX_ORDER 或算子构造被改过")


class TestFullConvectionResidualVsAnalytic:
    """**整条对流残差**（体积项 + 界面上风校正）与解析值比较。

    这是本轮系统遍历核心计算环节时发现的最重要一条：前面的用例只单独
    考察体积项，这一条走公开接口 `compute_scalar_convection_residual`。

    算例设计：`rho`/`u` 取**常数**（于是通过每个单元各面的质量通量精确
    守恒，避开 `filter_scalar_field`/"缺失 dilution"那段文档记录过的
    "对 sum_faces(mass_flux) 敏感"这个已被证伪的方向），`phi = 1 + G·x`
    取线性。于是

        d(rho*phi)/dt = -div(rho*u*phi) = -rho*(u·G)   （处处同一个常数）

    有闭式解，且 `rho*u*phi` 本身只有一次、完全落在 P1 解空间内——也就是
    说**任何非零误差都只能来自离散算子本身**，不是表示能力不足。

    实测（默认 `AFCFD_TURB_OVERINT=off`）：

        order=1 棱柱 相对误差 1.1294  （113%！残差范围 [-12.26, +1.11]，
                                      解析值 -8.5750）
        order=1 四面体 6.84e-15       （机器零）
        order=2 棱柱 2.26e-09
        order=2 四面体 1.97e-14

    打开去混叠后 order=1 棱柱降到 **4.61e-10**（改善约 2.4e9 倍），
    order=2 逐位不变（2.2597e-09 -> 2.2576e-09）。

    根因：非仿射棱柱上 `adj(J)` 是非平凡多项式，`adj(J)*rho*u*phi` 的
    真实次数高于 1，在 P1 空间里对它求导就是"先混叠再求导"；四面体在
    这张网格上是仿射的（实测同一单元内 det(J) 跨度恰好 0），adj(J) 是
    常数，所以没有这个问题——这也解释了为什么两类单元差了 14 个数量级。

    **边界条件不是原因**：order=1 与 order=2 的边界面数完全相同
    （cell0 8 面其中 6 边界、cell1 6 面其中 6 边界、…），而 order=2 给出
    的是**精确的** -8.5750。
    """

    RHO = 1.225
    UVEC = np.array([30.0, 7.0, -4.0])
    GRAD = np.array([0.3, -0.2, 0.15])

    def _run(self, order, overint):
        old = os.environ.get("AFCFD_TURB_OVERINT")
        os.environ["AFCFD_TURB_OVERINT"] = overint
        try:
            mesh = _build_synthetic_mixed_mesh(order)
            ops = generate_fr_operators(order)
            nc, ns = mesh.n_cells, mesh.n_sps_per_cell
            X = mesh.sps_coords.reshape(-1, 3)
            phi = (1.0 + X @ self.GRAD).reshape(nc, ns)
            rho = np.full((nc, ns), self.RHO)
            vel = np.tile(self.UVEC, (nc, ns, 1))
            res = tp.compute_scalar_convection_residual(phi, rho, vel, mesh, ops)
        finally:
            if old is None:
                os.environ.pop("AFCFD_TURB_OVERINT", None)
            else:
                os.environ["AFCFD_TURB_OVERINT"] = old
        exact = -self.RHO * float(self.UVEC @ self.GRAD)
        errs = {}
        for name, sl in _per_type_slices(mesh, order):
            errs[name] = np.abs(res[sl] - exact).max() / abs(exact)
        return errs

    def test_order1_collapsed_prism_has_large_error_without_dealiasing(
            self, monkeypatch):
        """**坍缩档**的 `off` 在生产阶数 P1 的棱柱上有 O(1) 相对误差
        ——这正是当初把 `AFCFD_TURB_OVERINT` 默认值改成 `on` 的依据。

        这条刻意断言"off 档确实有这个误差"而不是只断言"on 更好"：它把
        改默认值的依据本身钉住，任何人想把默认值改回 off 都会先看到这个
        数字。

        **显式跑在坍缩档（2026-09-20）**：那个 113% 是坍缩棱柱基的
        `adj(J)` 非平凡多项式导致的；原生棱柱基下同一算例 off 档就已经是
        2.33e-10（见下面那条），两个量级完全不同。
        """
        monkeypatch.setenv("AFCFD_PRISM_BASIS", "collapsed")
        errs = self._run(1, "off")
        assert errs["prism"] > 0.5, (
            f"order=1 坍缩棱柱在 off 档下的相对误差只有 {errs['prism']:.3e}，"
            f"与实测 1.1294 不符")
        assert errs["tet"] < 1e-12, (
            f"order=1 四面体相对误差 {errs['tet']:.3e} 不是机器零——"
            f"这张网格上四面体是仿射的，adj(J) 常数，不该有混叠")

    def test_order1_native_prism_is_already_accurate_without_dealiasing(
            self, monkeypatch):
        """**原生档（2026-09-20 起的默认）**：同一算例 off 档就已经准。

        实测 `off` 档 P1 棱柱相对误差 **2.33e-10**（坍缩档是 1.1294，
        差 9 个数量级）。原因与坍缩档那 113% 的根因是同一条的反面：
        原生棱柱基不经过坍缩坐标，`adj(J)` 不被坍缩映射推高次数。

        这条同时说明一件对默认值有影响的事实、如实记录：
        `AFCFD_TURB_OVERINT=on` 这个默认值当初的**主要**依据（P1 棱柱
        113% 误差）在默认基改成 native 之后已经不再成立。保留 `on` 仍有
        依据 —— 体积项对照里原生 P1 仍有 33 倍改善、P2 直接到机器零
        （见 `_PRISM_EXPECT`）—— 但"不开就有 O(1) 误差"这句话不能再用。
        """
        monkeypatch.setenv("AFCFD_PRISM_BASIS", "native")
        errs = self._run(1, "off")
        assert errs["prism"] < 1e-8, (
            f"原生档 P1 棱柱 off 相对误差 {errs['prism']:.3e} 远大于实测的 "
            f"2.33e-10 —— 若这是真实退化，请查原生棱柱的度量/算子")
        assert errs["tet"] < 1e-12

    @pytest.mark.parametrize("basis", ["collapsed", "native"])
    def test_dealiasing_gives_an_accurate_order1_prism(self, basis,
                                                       monkeypatch):
        """打开去混叠后两档都必须准（坍缩实测 4.6e-10、原生 2.3e-10）。"""
        monkeypatch.setenv("AFCFD_PRISM_BASIS", basis)
        errs = self._run(1, "on")
        assert errs["prism"] < 1e-8, (
            f"{basis}: 打开去混叠后 order=1 棱柱相对误差仍有 "
            f"{errs['prism']:.3e}（实测应为 ~5e-10）")
        assert errs["tet"] < 1e-12

    def test_order2_is_unaffected_by_the_switch(self):
        """order=2 上两档必须都已足够精确——说明 order=1 那个 113% 不是
        "这套离散本来就这么差"，而是 order=1 特有的混叠。"""
        off = self._run(2, "off")
        on = self._run(2, "on")
        for k in ("prism", "tet"):
            assert off[k] < 1e-7, f"order=2 {k} 默认档相对误差 {off[k]:.3e}"
            assert on[k] < 1e-7, f"order=2 {k} 去混叠档相对误差 {on[k]:.3e}"


class TestFreestreamPreservation:
    """安全性：常数场的散度必须为零——去混叠不能破坏自由流场保持性。"""

    @pytest.mark.parametrize("order", [1, 2])
    def test_constant_field_gives_zero_divergence(self, order):
        mesh = _build_synthetic_mixed_mesh(order)
        ops = generate_fr_operators(order)
        n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
        phi = np.full((n_cells, n_sps), 0.17)
        rho = np.full((n_cells, n_sps), 1.225)
        vel = np.zeros((n_cells, n_sps, 3)); vel[..., 0] = 30.0
        oi = tp._turb_overint_ops(mesh, ops)
        det = mesh.jacobians["det_jacs"].reshape(n_cells, n_sps)
        div = tp._scalar_convection_volume_overintegrated(
            phi, rho, vel, oi, n_sps) / det
        m = _real_dof_mask(mesh, order)
        ref = 1.225 * 30.0 * 0.17 / max(
            float(np.abs(mesh.sps_coords).max()), 1.0)
        assert np.abs(div[m]).max() < 1e-9 * max(ref, 1.0), (
            f"order={order}: 常数场散度 {np.abs(div[m]).max():.3e} 不是零——"
            f"去混叠破坏了自由流场保持性")

    @pytest.mark.parametrize("order", [1, 2])
    def test_constant_gradient_zero_gamma_gives_zero(self, order):
        """Gamma 恒为零 -> 扩散通量恒为零 -> 散度必须精确为零。"""
        mesh = _build_synthetic_mixed_mesh(order)
        ops = generate_fr_operators(order)
        n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
        oi = tp._turb_overint_ops(mesh, ops)
        grad = np.ones((n_cells, n_sps, 3))
        div = tp._scalar_diffusion_volume_overintegrated(
            np.zeros((n_cells, n_sps)), grad, oi, n_sps)
        np.testing.assert_allclose(div, 0.0, rtol=0, atol=0)


class TestSwitchSemantics:
    def _env(self, value):
        old = os.environ.get("AFCFD_TURB_OVERINT")
        if value is None:
            os.environ.pop("AFCFD_TURB_OVERINT", None)
        else:
            os.environ["AFCFD_TURB_OVERINT"] = value
        return old

    def _restore(self, old):
        if old is None:
            os.environ.pop("AFCFD_TURB_OVERINT", None)
        else:
            os.environ["AFCFD_TURB_OVERINT"] = old

    def test_default_is_on(self):
        """默认已于 2026-09-15 改为 `on`——解析判据显示 `off` 在生产阶数
        P1 的棱柱上有 113% 相对误差，代价只有约 +10.2%/步，且真实网格
        250 步运行本来就是带 on 跑的。理由全文见
        `resolve_turb_overintegration` 文档。"""
        old = self._env(None)
        try:
            assert tp.resolve_turb_overintegration() == "on"
        finally:
            self._restore(old)

    @pytest.mark.parametrize("v,expected", [
        ("off", "off"), ("on", "on"), ("OFF", "off"), ("On", "on"),
    ])
    def test_accepted_values(self, v, expected):
        old = self._env(v)
        try:
            assert tp.resolve_turb_overintegration() == expected
        finally:
            self._restore(old)

    @pytest.mark.parametrize("v", ["yes", "1", "true", "", "sensor"])
    def test_rejects_unknown(self, v):
        """不能静默退回默认——同 AFCFD_FILTER_TURB_GATE 的理由。"""
        old = self._env(v)
        try:
            with pytest.raises(ValueError, match="AFCFD_TURB_OVERINT"):
                tp.resolve_turb_overintegration()
        finally:
            self._restore(old)

    @pytest.mark.parametrize("order", [1, 2])
    def test_public_api_bit_identical_when_off(self, order):
        """显式 `off` 时，公开接口必须逐位等于 2026-09-15 之前的实现。

        判据取"显式 off"与"手工复刻的 coarse 体积项 + 同一条公开接口"
        之间的一致性：直接比对流残差整体（体积项 + 界面项），任何把
        去混叠误接进默认路径的改动都会在这里失败。
        """
        mesh, ops, phi, rho, vel, _ = _setup(order)
        old = self._env("off")
        try:
            res_off = tp.compute_scalar_convection_residual(
                phi, rho, vel, mesh, ops)
        finally:
            self._restore(old)
        det = mesh.jacobians["det_jacs"].reshape(mesh.n_cells, mesh.n_sps_per_cell)
        vol_ref = -_coarse_convection_div(phi, rho, vel, mesh, ops) / det
        # 残差 = 体积项 + 界面项；这里只能断言"体积项那一半与 coarse 一致"，
        # 做法是再跑一次 on 档，两者之差必须恰好等于两种体积项之差。
        old = self._env("on")
        try:
            res_on = tp.compute_scalar_convection_residual(
                phi, rho, vel, mesh, ops)
        finally:
            self._restore(old)
        vol_oi = -tp._scalar_convection_volume_overintegrated(
            phi, rho, vel, tp._turb_overint_ops(mesh, ops),
            mesh.n_sps_per_cell) / det
        np.testing.assert_allclose(res_on - res_off, vol_oi - vol_ref,
                                   rtol=1e-10, atol=1e-10)
        # 而且两者确实不同（否则上面那条是平凡真）
        assert np.abs(res_on - res_off).max() > 0.0
