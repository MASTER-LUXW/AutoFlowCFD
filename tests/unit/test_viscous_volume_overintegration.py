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

2026-10-01：生产体积项（`viscous_volume_term`）用的体积算子 K 同时减去了修正项的
本侧通量迹（`fr/face_flux_trace.py`），不再是单纯的散度；去混叠精度这一条因此对照
测试自己按 `c2f -> 细点通量 -> D_fine -> f2c` 拼出的过积分散度（与 K 的体积部分同一组
算子）。"解点上微分"那一档（`AFCFD_VISC_OVERINT=off`）同日删除，开关语义的用例一并删。
"""

import numpy as np
import pytest

from autoflowcfd.core.fr_operators.flux_kernels import viscous_physical_flux_batch
from autoflowcfd.core.fr_operators.volume_contract import (
    contract_shared_operator_2axis, contravariant_flux_from_metric,
    get_overintegration_context,
)
from autoflowcfd.core.fr_residual import viscous_flux as vf
from autoflowcfd.fr.native_prism.mode import resolve_prism_basis_mode
from autoflowcfd.fr.operators import generate_fr_operators

from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh

MU = 1.8e-5
PR = 0.72
PR_T = 0.9


def _per_type_slices(mesh, order):
    """两类单元各自的**真实自由度**切片。

    2026-09-20：棱柱那一段此前写的是 `slice(None)`（整个 `(p+1)^3` 宽度
    都算自由度），那只对坍缩棱柱基成立；原生棱柱基（当前默认）每单元只有
    `(p+1)^2(p+2)/2` 个真实自由度，把零填充槽位算进来会让误差被填充位
    主导、去混叠与不去混叠算出逐位相同的数字。真实自由度数走唯一入口。
    """
    from autoflowcfd.fr.native_padding import real_sps_per_cell

    n_real_prism, n_real_tet = real_sps_per_cell(order)
    return [
        ("prism", (slice(0, mesh.n_prism_cells), slice(0, n_real_prism))),
        ("tet", (slice(mesh.n_prism_cells, mesh.n_cells),
                 slice(0, n_real_tet))),
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


def _overint_viscous_div(Q, gv, gT, mut, mesh, ops):
    """过积分散度 `f2c·D_fine·G~(细点)`（K 的体积部分），参考空间。"""
    nc, ns = mesh.n_cells, mesh.n_sps_per_cell
    div = np.zeros((nc, ns, 5))
    for lo, hi, nf, det_f, inv_f, c2f, D_f, f2c in get_overintegration_context(mesh, ops)["segs"]:
        if hi <= lo:
            continue
        nb = hi - lo
        interp = lambda a: np.einsum("qs,cs...->cq...", c2f, a)
        G_phys = viscous_physical_flux_batch(
            np.ascontiguousarray(interp(Q[lo:hi]).reshape(-1, 5)),
            np.ascontiguousarray(interp(gv[lo:hi]).reshape(-1, 3, 3)),
            np.ascontiguousarray(interp(gT[lo:hi]).reshape(-1, 3)),
            MU, PR, np.ascontiguousarray(interp(mut[lo:hi]).reshape(-1)), PR_T,
        ).reshape(nb, nf, 3, 5)
        G_tilde = contravariant_flux_from_metric(np.ascontiguousarray(det_f), np.ascontiguousarray(inv_f), G_phys)
        div[lo:hi] = np.einsum("sq,cqv->csv", f2c, contract_shared_operator_2axis(D_f, G_tilde))
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

        div_oi = _overint_viscous_div(Q, gv, gT, mut, mesh, ops) / det[..., None]
        div_co = _coarse_viscous_div(Q, gv, gT, mut, mesh, ops) / det[..., None]

        # 只看能量分量与动量分量（质量分量的粘性通量恒为零，见
        # viscous_flux.py::viscous_physical_flux_batch——G[...,0] 不写值）
        errs = {}
        for name, sl in _per_type_slices(mesh, order):
            e = exact[sl][..., 1:]
            sc = np.abs(e).max()
            errs[name] = (np.abs(div_co[sl][..., 1:] - e).max() / sc,
                          np.abs(div_oi[sl][..., 1:] - e).max() / sc)

        # coarse 误差的下界按棱柱基分档（2026-09-20）：原生棱柱基的
        # coarse 本身就准得多（P1 实测 1.8e-3 vs 坍缩 2.7e-2、P2 实测
        # 4.2e-5 vs 坍缩 1.5e-3），用坍缩档那条 1e-3 会把一次正常通过
        # 判成"算例没造出混叠"。下界仍然保留 —— 它防的是"判据空转"。
        _CO_MIN = {("collapsed", 1): 1e-3, ("collapsed", 2): 1e-3,
                   ("native", 1): 1e-4, ("native", 2): 1e-6}
        basis = resolve_prism_basis_mode()
        pr_co, pr_oi = errs["prism"]
        assert pr_co > _CO_MIN[(basis, order)], (
            f"{basis} order={order} prism: coarse 误差只有 {pr_co:.3e}，"
            f"低于该档下界 {_CO_MIN[(basis, order)]:.1e} —— 这个算例没有"
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
        div_oi = vf.viscous_volume_term(Q, gv, gT, mut, MU, PR, PR_T, oi, mesh.n_sps_per_cell)
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
        div = vf.viscous_volume_term(
            Q, np.zeros((nc, ns, 3, 3)), np.zeros((nc, ns, 3)),
            np.zeros((nc, ns)), MU, PR, PR_T, oi, ns)
        np.testing.assert_allclose(div, 0.0, rtol=0, atol=0)
