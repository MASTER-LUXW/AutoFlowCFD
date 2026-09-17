"""粘性体积项去混叠（过积分）的验证（2026-09-15）。

## 修的是什么

去混叠机制（`fr/collapsed_basis.py::build_overintegration_operators` 有
完整动机与实测数字）一直**只接在平均流的无粘体积项**上。粘性项完全没有
（grep 确认 `fr_residual/viscous_flux.py` 里 overint/jacobians_fine 零
命中），而粘性通量里有 `tau`、`u·tau`、`k_cond*grad_T` 这些乘积，再乘
`adj(J)`，真实多项式次数远高于 order。

它此前无所谓的原因**已经消失**：legacy 模态滤波器下 `grad_vel` 是机器零
（实测胞内 |grad u|/(U/h)=7.7e-16），粘性体积项几乎只剩边界 IP 罚项；
一旦真正关掉滤波器（零阶数损失），`grad_vel` 变成 O(0.08) 的真实量，
这一项立刻活跃。

与 k/omega 扩散项那边不同，**这里没有"Gamma 自身混叠"那种局限**：本函数
手上有 Q/grad_vel/grad_T/mu_t，可以像无粘路径那样在 FINE 点重新求值
非线性通量函数本身。

## 判据

与 `test_turbulence_transport_overintegration.py` 同一套方法论：把
`grad_vel`/`grad_T`/`mu_t`/`Q` 都设成**显式多项式**（作为输入直接给定，
不经过 `compute_physical_gradient`，从而把体积项自身的混叠单独隔离出来），
参照值用**四阶中心差分对闭式通量函数求导**得到——被求导的是
`viscous_physical_flux_point` 这个项目自己的通量函数在解析场上的取值，
所以测的是**散度离散**而不是通量公式。

分单元类型统计：棱柱映射非仿射（实测同一单元内 det(J) 跨度 0.0722），
native 四面体仿射（跨度恰好 0），两类可达精度不同；四面体那一片只取
真实自由度（`D_native_tet_padded` 的填充行是零行）。
"""

import os

import numpy as np
import pytest

from autoflowcfd.core.fr_operators.flux_kernels import viscous_physical_flux_batch
from autoflowcfd.core.fr_operators.volume_contract import (
    contract_shared_operator_2axis, contravariant_flux_from_metric,
    get_overintegration_context,
)
from autoflowcfd.core.fr_residual import viscous_flux as vf
from autoflowcfd.fr.operators import generate_fr_operators

from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh

MU = 1.8e-5
PR = 0.72
PR_T = 0.9


def _per_type_slices(mesh, order):
    n_native = (order + 1) * (order + 2) * (order + 3) // 6
    return [
        ("prism", (slice(0, mesh.n_prism_cells), slice(None))),
        ("tet", (slice(mesh.n_prism_cells, mesh.n_cells), slice(0, n_native))),
    ]


class _ViscField:
    """Q / grad_vel / grad_T / mu_t 全部取显式一次多项式。

    `tau` 与 `mu_eff` 各含一次项 -> 乘积二次；`u·tau` 三次；再乘
    `adj(J)`（棱柱上非常数）还会更高。故意让次数超过 order。
    """

    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.q0 = np.array([1.225, 30.0, 2.0, -1.5, 101325.0])
        self.qg = rng.uniform(-0.05, 0.05, (5, 3)) * np.array(
            [[1.0], [30.0], [30.0], [30.0], [1.0e4]])
        self.gv0 = rng.uniform(-5.0, 5.0, (3, 3))
        self.gvg = rng.uniform(-2.0, 2.0, (3, 3, 3))
        self.gT0 = rng.uniform(-3.0, 3.0, 3)
        self.gTg = rng.uniform(-1.0, 1.0, (3, 3))
        self.mt0 = 3.0e-5
        self.mtg = rng.uniform(-2.0e-6, 2.0e-6, 3)

    def Q(self, X):
        return self.q0[None, :] + X @ self.qg.T

    def grad_vel(self, X):
        return self.gv0[None] + np.einsum("pd,abd->pab", X, self.gvg)

    def grad_T(self, X):
        return self.gT0[None, :] + X @ self.gTg.T

    def mu_t(self, X):
        return self.mt0 + X @ self.mtg

    def flux(self, X):
        """(n, 3, 5) 物理粘性通量，用项目自己的通量函数在解析场上求值。"""
        return viscous_physical_flux_batch(
            np.ascontiguousarray(self.Q(X)),
            np.ascontiguousarray(self.grad_vel(X)),
            np.ascontiguousarray(self.grad_T(X)),
            MU, PR, np.ascontiguousarray(self.mu_t(X)), PR_T,
        )

    def div_exact(self, X):
        """四阶中心差分对**闭式通量函数**求 div(G)，返回 (n, 5)。"""
        h = 1e-4 * max(np.abs(X).max(), 1.0)
        out = np.zeros((X.shape[0], 5))
        for d in range(3):
            e = np.zeros(3)
            e[d] = h
            out += (-self.flux(X + 2 * e)[:, d, :]
                    + 8 * self.flux(X + e)[:, d, :]
                    - 8 * self.flux(X - e)[:, d, :]
                    + self.flux(X - 2 * e)[:, d, :]) / (12 * h)
        return out


def _setup(order, seed=0):
    mesh = _build_synthetic_mixed_mesh(order)
    ops = generate_fr_operators(order)
    nc, ns = mesh.n_cells, mesh.n_sps_per_cell
    X = mesh.sps_coords.reshape(-1, 3)
    f = _ViscField(seed)
    return (mesh, ops, f.Q(X).reshape(nc, ns, 5),
            f.grad_vel(X).reshape(nc, ns, 3, 3),
            f.grad_T(X).reshape(nc, ns, 3),
            f.mu_t(X).reshape(nc, ns),
            f.div_exact(X).reshape(nc, ns, 5))


def _coarse_viscous_div(Q, gv, gT, mut, mesh, ops):
    """coarse 路径的粘性体积项 div_comp（复刻生产代码的非过积分分支）。"""
    nc, ns = mesh.n_cells, mesh.n_sps_per_cell
    n_prism = mesh.n_prism_cells
    det = mesh.jacobians["det_jacs"].reshape(nc, ns)
    inv = mesh.jacobians["inv_jacs"].reshape(nc, ns, 3, 3)
    tet_op = (ops.D_native_tet_padded
              if getattr(ops, "D_native_tet_padded", None) is not None
              else ops.D_3d_tet)
    div = np.zeros((nc, ns, 5))
    for lo, hi, op_D in ((0, n_prism, ops.D_3d_prism), (n_prism, nc, tet_op)):
        if hi <= lo:
            continue
        nb = hi - lo
        G_phys = viscous_physical_flux_batch(
            np.ascontiguousarray(Q[lo:hi].reshape(-1, 5)),
            np.ascontiguousarray(gv[lo:hi].reshape(-1, 3, 3)),
            np.ascontiguousarray(gT[lo:hi].reshape(-1, 3)),
            MU, PR, np.ascontiguousarray(mut[lo:hi].reshape(-1)), PR_T,
        ).reshape(nb, ns, 3, 5)
        G_tilde = contravariant_flux_from_metric(det[lo:hi], inv[lo:hi], G_phys)
        div[lo:hi] = contract_shared_operator_2axis(op_D, G_tilde)
    return div


class TestOverintegrationIsMoreAccurate:
    @pytest.mark.parametrize("order", [1, 2])
    def test_viscous_volume_term_vs_analytic(self, order):
        mesh, ops, Q, gv, gT, mut, exact = _setup(order)
        oi = get_overintegration_context(mesh, ops)
        assert oi is not None, (
            f"order={order} 下过积分算子/细点度量不可用——它们在 order>=1 时"
            f"应当无条件构造好")
        det = mesh.jacobians["det_jacs"].reshape(mesh.n_cells, mesh.n_sps_per_cell)

        div_oi = vf._viscous_volume_overintegrated(
            Q, gv, gT, mut, MU, PR, PR_T, oi, mesh.n_sps_per_cell) / det[..., None]
        div_co = _coarse_viscous_div(Q, gv, gT, mut, mesh, ops) / det[..., None]

        # 只看能量分量与动量分量（质量分量的粘性通量恒为零，见
        # viscous_flux.py::viscous_physical_flux_batch——G[...,0] 不写值）
        errs = {}
        for name, sl in _per_type_slices(mesh, order):
            e = exact[sl][..., 1:]
            sc = np.abs(e).max()
            errs[name] = (np.abs(div_co[sl][..., 1:] - e).max() / sc,
                          np.abs(div_oi[sl][..., 1:] - e).max() / sc)

        pr_co, pr_oi = errs["prism"]
        assert pr_co > 1e-3, (
            f"order={order} prism: coarse 误差只有 {pr_co:.3e}，这个算例没有"
            f"真正制造出混叠，判据失去意义")
        assert pr_oi < pr_co, (
            f"order={order} prism: 去混叠误差 {pr_oi:.3e} 不低于 coarse 的 "
            f"{pr_co:.3e}——去混叠没起作用")

        tet_co, tet_oi = errs["tet"]
        assert tet_oi <= tet_co, (
            f"order={order} tet: 去混叠误差 {tet_oi:.3e} 高于 coarse 的 "
            f"{tet_co:.3e}")

    @pytest.mark.parametrize("order", [1, 2])
    def test_mass_component_stays_zero(self, order):
        """粘性通量的质量分量恒为零 -> 它的散度也必须恒为零。

        这条同时是对"去混叠链路没有把别的分量串到质量分量上"的检查。
        """
        mesh, ops, Q, gv, gT, mut, _ = _setup(order)
        oi = get_overintegration_context(mesh, ops)
        div_oi = vf._viscous_volume_overintegrated(
            Q, gv, gT, mut, MU, PR, PR_T, oi, mesh.n_sps_per_cell)
        np.testing.assert_allclose(div_oi[..., 0], 0.0, rtol=0, atol=0)


class TestFreestreamPreservation:
    @pytest.mark.parametrize("order", [1, 2])
    def test_zero_gradients_give_zero_divergence(self, order):
        """梯度恒为零 -> 粘性通量恒为零 -> 散度必须精确为零。"""
        mesh = _build_synthetic_mixed_mesh(order)
        ops = generate_fr_operators(order)
        nc, ns = mesh.n_cells, mesh.n_sps_per_cell
        Q = np.zeros((nc, ns, 5))
        Q[..., 0] = 1.225
        Q[..., 1] = 30.0
        Q[..., 4] = 101325.0
        oi = get_overintegration_context(mesh, ops)
        div = vf._viscous_volume_overintegrated(
            Q, np.zeros((nc, ns, 3, 3)), np.zeros((nc, ns, 3)),
            np.zeros((nc, ns)), MU, PR, PR_T, oi, ns)
        np.testing.assert_allclose(div, 0.0, rtol=0, atol=0)


class TestSwitchSemantics:
    def _env(self, v):
        old = os.environ.get("AFCFD_VISC_OVERINT")
        if v is None:
            os.environ.pop("AFCFD_VISC_OVERINT", None)
        else:
            os.environ["AFCFD_VISC_OVERINT"] = v
        return old

    def _restore(self, old):
        if old is None:
            os.environ.pop("AFCFD_VISC_OVERINT", None)
        else:
            os.environ["AFCFD_VISC_OVERINT"] = old

    def test_default_is_on(self):
        """默认 `on`（2026-09-17 从 `off` 改）。

        依据（平板边界层算例 2304 单元，跑到 400 步的真实粘性梯度状态上求
        一次粘性残差，以 `on` + `AFCFD_OVERINT_ORDER_RULE=3x` 为参照）：

            off @ 2x（原默认）   能量分量相对差 0.632442
            on  @ 2x（新默认）   能量分量相对差 0.000000   <- 逐位相同

        即过积分的结果在生产过积分阶数上**已经收敛**（提到 3x 逐位不变），
        不过积分的差 63%。动量分量两档逐位相同（P1 下常粘度的 tau 是逐单元
        P0、精确可微分），差的只有能量分量——它含 `u = rho_u/rho` 与
        `T = p/(rho*R)` 这些**有理**函数，真实非多项式。

        代价：粘性项 +25%，整步约 +5.3%。

        同时这是 `AFCFD_FILTER_MODE` 默认从 `legacy` 改成 `sensor` 的**一致性
        要求**：legacy 滤波下 `grad_vel` 是机器零（7.7e-16）、粘性体积项几乎
        只剩边界罚项，那个"off 无所谓"的前提随之消失。
        """
        old = self._env(None)
        try:
            assert vf.resolve_viscous_overintegration() == "on"
        finally:
            self._restore(old)

    def test_off_stays_available_for_regression(self):
        """`off` 保留为合法档：逐位复现历史结果、以及隔离界面项的交叉
        对比测试（`test_fr_viscous_flux_kernel_crosscheck.py`）都要用它。"""
        old = self._env("off")
        try:
            assert vf.resolve_viscous_overintegration() == "off"
        finally:
            self._restore(old)

    @pytest.mark.parametrize("v,expected", [
        ("off", "off"), ("on", "on"), ("OFF", "off"), ("On", "on"),
    ])
    def test_accepted_values(self, v, expected):
        old = self._env(v)
        try:
            assert vf.resolve_viscous_overintegration() == expected
        finally:
            self._restore(old)

    @pytest.mark.parametrize("v", ["yes", "1", "true", "", "sensor"])
    def test_rejects_unknown(self, v):
        old = self._env(v)
        try:
            with pytest.raises(ValueError, match="AFCFD_VISC_OVERINT"):
                vf.resolve_viscous_overintegration()
        finally:
            self._restore(old)

    @pytest.mark.parametrize("order", [1, 2])
    def test_public_api_differs_exactly_by_the_two_volume_terms(self, order):
        """默认关闭 vs 打开，公开接口的差值必须**恰好等于**两种体积项之差。

        任何把去混叠误接进默认路径、或在链路上多改了别的东西（界面项、
        troubled-cell 抑制）的改动都会在这里失败。
        """
        from autoflowcfd.core.fr_residual.inviscid import primitive_to_conserved
        mesh = _build_synthetic_mixed_mesh(order)
        ops = generate_fr_operators(order)
        nc, ns = mesh.n_cells, mesh.n_sps_per_cell
        X = mesh.sps_coords.reshape(-1, 3)
        rng = np.random.default_rng(order)
        Qp = np.zeros((nc, ns, 5))
        Qp[..., 0] = 1.225
        Qp[..., 1] = (30.0 + 2.0 * X[:, 1]).reshape(nc, ns)
        Qp[..., 2] = (1.5 * X[:, 2]).reshape(nc, ns)
        Qp[..., 3] = (-0.8 * X[:, 0]).reshape(nc, ns)
        Qp[..., 4] = (101325.0 * (1.0 + 0.05 * X[:, 0])).reshape(nc, ns)
        U = primitive_to_conserved(Qp)

        old = self._env("off")
        try:
            res_off = vf.compute_viscous_residual_fr(U, mesh, ops, MU, PR)
        finally:
            self._restore(old)
        old = self._env("on")
        try:
            res_on = vf.compute_viscous_residual_fr(U, mesh, ops, MU, PR)
        finally:
            self._restore(old)

        # 两者确实不同（否则下面那条是平凡真）
        assert np.abs(res_on - res_off).max() > 0.0, "开关没有产生任何差异"

        # 差值应等于两种体积项之差——用生产代码同一套输入重算
        from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive
        from autoflowcfd.core.fr_residual.viscous_flux import compute_temperature
        Q = conserved_to_primitive(U[..., :5])
        T = compute_temperature(Q)
        from autoflowcfd.core.fr_operators.gradients import compute_physical_gradient
        gQ = compute_physical_gradient(Q, mesh, ops)
        gv = gQ[:, :, 1:4, :]
        gT = compute_physical_gradient(T[:, :, None], mesh, ops)[:, :, 0, :]
        mut = np.zeros((nc, ns))
        det = mesh.jacobians["det_jacs"].reshape(nc, ns)
        oi = get_overintegration_context(mesh, ops)
        vol_oi = vf._viscous_volume_overintegrated(
            Q, gv, gT, mut, MU, PR, PR_T, oi, ns) / det[..., None]
        vol_co = _coarse_viscous_div(Q, gv, gT, mut, mesh, ops) / det[..., None]
        np.testing.assert_allclose(res_on - res_off, vol_oi - vol_co,
                                   rtol=1e-9, atol=1e-9)
