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
from autoflowcfd.fr.operators import generate_fr_operators

from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh


def _real_dof_mask(mesh, order):
    """(n_cells, n_sps) 布尔掩码，标出**真实自由度**。

    必须有这一步才能和解析参照比较：native 四面体只有前
    `(p+1)(p+2)(p+3)/6` 个 SP 是自由度，其余是零填充槽位，
    `D_native_tet_padded` 的填充行是**零行**，所以那些位置的散度恒为 0，
    与解析值无关。用"div(x,0,0)=1"这个手算算例定位到的：不加掩码时
    结果是 [0,1] 区间而不是恒 1，min=0 全部来自填充位。棱柱的
    `(order+1)^3` 个 SP 全是自由度。
    """
    n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
    n_native = (order + 1) * (order + 2) * (order + 3) // 6
    mask = np.ones((n_cells, n_sps), dtype=bool)
    mask[mesh.n_prism_cells:, n_native:] = False
    return mask


def _actual_over_order(oi):
    """从过积分上下文**实际读出** over_order，而不是在测试里重算规则。

    2026-09-15 的教训：本文件原来内部写死 `min(2*order, 3)`，而生产规则
    随后改成了 `3*order`（只影响 order=1，见 fr/operators.py 该处文档）。
    测试当时仍然通过，但走的是错误的分支、判据的理由已经和现实脱节。
    细点数 `n_fine = (over_order+1)^3`，反解即得。
    """
    n1d = round(oi["n_fine"] ** (1.0 / 3.0))
    assert n1d ** 3 == oi["n_fine"], f"n_fine={oi['n_fine']} 不是完全立方数"
    return n1d - 1


def _per_type_slices(mesh, order):
    """[("prism", 切片), ("tet", 切片)]，四面体那一片只取真实自由度。

    分类型统计是必须的：两类单元可达的精度不同（棱柱映射非仿射 ->
    adj(J) 是非平凡多项式 -> 乘积次数被进一步推高；native 四面体仿射 ->
    adj(J) 常数）。混在一起会掩盖"仿射单元上已经精确"这条最强证据。
    """
    n_native = (order + 1) * (order + 2) * (order + 3) // 6
    return [
        ("prism", (slice(0, mesh.n_prism_cells), slice(None))),
        ("tet", (slice(mesh.n_prism_cells, mesh.n_cells), slice(0, n_native))),
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

        # 实测（相对 L-inf）。注意 over_order 由
        # `AFCFD_OVERINT_ORDER_RULE` 决定（默认 `2x`），所以 order=1 有
        # 两组数字；断言用 `_actual_over_order(oi)` 读实际值来分派：
        #   order=1 over_order=2（默认）: prism 4.9816e+00 -> 1.3602e-01  36.6x
        #                                 tet   4.4823e-01 -> 5.6166e-02   8.0x
        #   order=1 over_order=3（3x 档）: prism 4.9816e+00 -> 2.1243e-03  2345x
        #                                 tet   4.4823e-01 -> 3.6341e-13  机器零
        #   order=2（两档都是 3）        : prism 7.6441e-01 -> 9.5333e-03  80.2x
        #                                 tet   5.6166e-02 -> 9.8492e-14  机器零
        err_co, err_oi = errs["prism"]
        assert err_co > 0.5, (
            f"order={order} prism: coarse 误差只有 {err_co:.3e}，这个算例没有"
            f"真正制造出混叠，判据失去意义——请加大 rho/u/phi 的非线性度")
        assert err_oi < err_co / 10.0, (
            f"order={order} prism: 去混叠误差 {err_oi:.3e} 相比 coarse 的 "
            f"{err_co:.3e} 改善不到 10 倍（实测 36.6x / 80.2x）")

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
    def test_fine_point_counts_per_rule(self, order, nf_2x, nf_3x):
        """两档实际构造出的细点数——直接读算子，不重算规则。

        order>=2 上两档相同，这是 `OVERINTEGRATION_MAX_ORDER = 3` 的
        直接后果，也是"这条切换只影响 order=1"这句话的依据。
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


class TestOverintegrationIsNoOpAtOrder3:
    """order=3 时去混叠**恰好什么也不做**，这是 `OVERINTEGRATION_MAX_ORDER`
    的直接后果，必须显式钉住而不是让它看起来"也生效了"。

        over_order = min(2*order, OVERINTEGRATION_MAX_ORDER=3)
        order=1 -> 2,  order=2 -> 3,  order=3 -> 3,  order>=4 -> 3

    order=3 时 over_order==order，"细"网格就是粗网格本身，插值/微分/限制
    三件套复合起来等于原路径。实测改善倍数恰好 1.00x。

    这条不是缺陷报告，是**范围说明**：本项目当前跑 P1/P2，P3 另有独立的
    内存边界（见 P3 face-flux-points OOM 记录）。真要让 P3 也受益必须提高
    那个上限，而它是平均流与标量输运共享的，属于独立一步。
    """

    def test_over_order_equals_order_at_3(self):
        """从**实际构造出的算子**读 over_order，不重算规则。"""
        from autoflowcfd.fr.collapsed_basis import OVERINTEGRATION_MAX_ORDER
        assert OVERINTEGRATION_MAX_ORDER == 3, (
            "上限被改动了——它有真实回归背景（放宽到 4 会让 P2 均匀自由"
            "流场残差从 1.06e-5 恶化到 5.6e-3），见该常量处文档")
        mesh = _build_synthetic_mixed_mesh(3)
        oi = tp._turb_overint_ops(mesh, generate_fr_operators(3))
        assert _actual_over_order(oi) == 3

    def test_no_accuracy_change_at_order3(self):
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

    def test_default_is_off(self):
        old = self._env(None)
        try:
            assert tp.resolve_turb_overintegration() == "off"
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
        """默认关闭时，公开接口必须逐位等于此前实现。

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
