"""k/omega 输运体积项去混叠（过积分）的验证（2026-09-15）。

## 修的是什么

`fr/native_prism/overintegration.py` 与 `fr/native_tet/overintegration.py`
的模块文档完整记录了
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

from autoflowcfd.core.fr_operators.volume_contract import get_overintegration_context
from autoflowcfd.core.turbulence.transport.convection import _scalar_convection_volume_overintegrated
from autoflowcfd.core.turbulence.transport.diffusion import _scalar_diffusion_volume_overintegrated
from autoflowcfd.fr.native_prism.mode import resolve_prism_basis_mode
from autoflowcfd.fr.operators import generate_fr_operators

from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh
from tests.unit._turbulence_transport_overintegration_common import (
    _PRISM_EXPECT,
    _actual_over_order,
    _coarse_convection_div,
    _per_type_slices,
    _setup,
)


class TestOverintegrationIsMoreAccurate:
    """核心判据：与**解析散度**比较，去混叠必须显著更准。"""

    @pytest.mark.parametrize("order", [1, 2])
    def test_convection_volume_term_vs_analytic(self, order):
        mesh, ops, phi, rho, vel, exact = _setup(order)
        oi = get_overintegration_context(mesh, ops)
        assert oi is not None, (
            f"order={order} 下过积分算子/细点度量不可用——它们在 order>=1 时"
            f"应当无条件构造好（见 fr/operators.py 与 high_order_mesh_order.py）")

        det = mesh.jacobians["det_jacs"].reshape(mesh.n_cells, mesh.n_sps_per_cell)
        div_oi = _scalar_convection_volume_overintegrated(
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
        # （默认 `2x`）决定，用 `_actual_over_order(oi)` 读实际值：
        #
        #   原生棱柱基（唯一实现，过积分上限 6）
        #     P1 oo=2: prism 4.0552e-01 -> 1.2170e-02   33.3x
        #     P2 oo=4: prism 1.2480e-02 -> 2.8663e-12   **机器零**
        #
        # 已删除的坍缩棱柱基上同一算例是（**史料，已不可复现**）：
        #     P1 oo=2: prism 4.9816e+00 -> 1.3602e-01   36.6x
        #     P1 oo=3（3x 档）: prism 4.9816e+00 -> 2.1243e-03  2345x
        #     P2 oo=3: prism 7.6441e-01 -> 9.5333e-03   80.2x
        #
        # 两条基的 coarse 误差**不同量级**（原生 P1 比坍缩小 12 倍、P2 小
        # 61 倍），所以"coarse 必须 > 0.5"那条防空转的下界当年必须分档 ——
        # 原生 P2 的 coarse 只有 1.2e-2，用 0.5 会把一条正常通过的判据
        # 判成"算例没造出混叠"。
        basis = resolve_prism_basis_mode()
        coarse_min, gain_min = _PRISM_EXPECT[order]
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
        #   over_order = resolve_tet_overintegration_order(order)（上限 6）
        #   order=1 -> 2（不足，残余 5.6e-2）
        #   order=2 -> 4（足够，机器零）
        # 也就是说 `2*order` 这条经验法则是为平均流的**二次**非线性设计
        # 的（见 `fr/overintegration_order.py::
        # resolve_overintegration_order_rule`），标量输运的三重乘积严格来说
        # 需要 `3*order`。不在这里改那条法则：算子与 jacobians_fine 是
        # **全局共享**的（平均流用同一份），改动会同时影响已验证的平均流
        # 路径，且会把 order=1 的棱柱细点数从 18 抬到 40（P2 OOM 有前科），
        # 属于独立一步。
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
        oi = get_overintegration_context(mesh, ops)
        det = mesh.jacobians["det_jacs"].reshape(n_cells, n_sps)

        div_oi = _scalar_diffusion_volume_overintegrated(
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

    默认 `2x` 必须复现此前已被长期验证的行为。"`3x` 只在 order=1 上与它
    不同"这句话只对已删除的坍缩棱柱基成立（order>=2 被那条
    `OVERINTEGRATION_MAX_ORDER=3` 卡住）：原生基的上限是 6，所以 P2 也会
    被这条规则改变（细点 75 -> 196），见下面的细点数判据。
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
        from autoflowcfd.fr.overintegration_order import (
            resolve_overintegration_order_rule,
        )
        old = self._env(None)
        try:
            assert resolve_overintegration_order_rule() == 2
        finally:
            self._restore(old)

    @pytest.mark.parametrize("v,expected", [("2x", 2), ("3x", 3), ("3X", 3)])
    def test_accepted_values(self, v, expected):
        from autoflowcfd.fr.overintegration_order import (
            resolve_overintegration_order_rule,
        )
        old = self._env(v)
        try:
            assert resolve_overintegration_order_rule() == expected
        finally:
            self._restore(old)

    @pytest.mark.parametrize("v", ["2", "3", "", "two"])
    def test_rejects_unknown(self, v):
        from autoflowcfd.fr.overintegration_order import (
            resolve_overintegration_order_rule,
        )
        old = self._env(v)
        try:
            with pytest.raises(ValueError, match="AFCFD_OVERINT_ORDER_RULE"):
                resolve_overintegration_order_rule()
        finally:
            self._restore(old)

    @pytest.mark.parametrize("order,nf_2x,nf_3x", [
        (1, 18, 40), (2, 75, 196), (3, 196, 196)])
    def test_fine_point_counts_per_rule(self, order, nf_2x, nf_3x):
        """两档规则实际构造出的细点数——直接读算子，不重算规则。

        原生棱柱的细点数是 `(oo+1)^2(oo+2)/2`，上限是 6（已删除的坍缩基
        那条条件数上限是 3），所以：

            rule  P1          P2           P3
            2x    oo=2 -> 18   oo=4 ->  75  oo=6 -> 196
            3x    oo=3 -> 40   oo=6 -> 196  oo=6 -> 196（被上限 6 卡住）

        也就是说"这条切换只影响 order=1"这句话**只对已删除的坍缩档成立**：
        原生档下 P2 也被它改变（75 -> 196）。
        """
        for v, want in (("2x", nf_2x), ("3x", nf_3x)):
            old = self._env(v)
            try:
                ops = generate_fr_operators(order)
                got = ops.overint_D_fine_prism.shape[0]
                assert got == want, (
                    f"rule={v} order={order}: 细点数 {got} != {want}")
            finally:
                self._restore(old)
