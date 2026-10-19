# -*- coding: utf-8 -*-
"""过积分体积项与物理梯度的逐单元融合核（2026-10-15）与分块实现逐位相同。

融合核把 `解点 -> 细点插值 -> 通量 -> 逆变通量 -> 散度 -> 解点` 整条链放进每个单元的局部工作数组里，取代
"按 32768 单元分块、每步分配块大小临时数组"的实现（Windows 上新分配大数组的缺页代价是复用内存的 10 倍以上，
见 `fr_residual/inviscid.py::inviscid_volume_divergence` 文档）。这里的参照就是那条分块链，用仍保留的批量原语
（共享算子收缩、批量通量、逆变通量）逐步组装：融合只改执行方式，求和顺序必须处处相同，所以全部断言
`array_equal`。

两组输入：随机构造的过积分上下文（单元数远大于并行块数，覆盖块划分；四面体段的细点度量是步长 0 的广播视图，
与生产一致），与合成混合网格上的真实算子（order 1~3）。
"""

import numpy as np
import pytest

from autoflowcfd.core.fr_operators.flux_kernels import euler_physical_flux_batch, viscous_physical_flux_batch
from autoflowcfd.core.fr_operators.flux_kernels.viscous import smooth_nonnegative_array
from autoflowcfd.core.fr_operators.gradients import compute_physical_gradient
from autoflowcfd.core.fr_operators.volume_contract import (
    contract_shared_operator_1axis,
    contract_shared_operator_2axis,
    contravariant_flux_from_metric,
    get_overintegration_context,
)
from autoflowcfd.core.fr_residual.inviscid import inviscid_volume_divergence
from autoflowcfd.core.fr_residual.viscous_flux import viscous_volume_term
from autoflowcfd.core.turbulence.transport.convection import _scalar_convection_volume_overintegrated
from autoflowcfd.core.turbulence.transport.diffusion import _scalar_diffusion_volume_overintegrated
from autoflowcfd.fr.operators import generate_fr_operators

from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh

MU, PR, PR_T = 1.8e-5, 0.72, 0.9


# ------------------------------------------------------------------ 参照：分块链（批量原语逐步组装）

def _k_contract(K_all, combo, F_tilde):
    """逐单元 `K_all[combo[c]] : F_tilde[c]`：按组合分组、每组一次共享算子收缩（同一个 4 路累加核）。"""
    out = np.empty((F_tilde.shape[0], K_all.shape[1], F_tilde.shape[3]))
    for k in np.unique(combo):
        sel = combo == k
        out[sel] = contract_shared_operator_2axis(K_all[k], np.ascontiguousarray(F_tilde[sel]))
    return out


def _ref_inviscid(Q, oi):
    out = np.zeros(Q.shape[:2] + (5,))
    for (lo, hi, n_fine, det, inv, c2f, _D, _f2c), (K_all, combo) in zip(oi["segs"], oi["lifted_div"]):
        q_f = contract_shared_operator_1axis(c2f, Q[lo:hi])
        F = euler_physical_flux_batch(q_f.reshape(-1, 5)).reshape(hi - lo, n_fine, 3, 5)
        out[lo:hi] = _k_contract(K_all, combo, contravariant_flux_from_metric(det, inv, F))
    return out


def _ref_viscous(Q, gv, gT, mut, oi):
    out = np.zeros(Q.shape[:2] + (5,))
    for (lo, hi, n_fine, det, inv, c2f, _D, _f2c), (K_all, combo) in zip(oi["segs"], oi["lifted_div"]):
        nb, n_sps = hi - lo, Q.shape[1]
        q_f = contract_shared_operator_1axis(c2f, Q[lo:hi])
        gv_f = contract_shared_operator_1axis(c2f, np.ascontiguousarray(gv[lo:hi].reshape(nb, n_sps, 9)))
        gT_f = contract_shared_operator_1axis(c2f, np.ascontiguousarray(gT[lo:hi]))
        mt_f = contract_shared_operator_1axis(c2f, np.ascontiguousarray(mut[lo:hi][:, :, None]))[..., 0]
        G = viscous_physical_flux_batch(q_f.reshape(-1, 5), np.ascontiguousarray(gv_f.reshape(-1, 3, 3)),
                                        gT_f.reshape(-1, 3), MU, PR, np.ascontiguousarray(mt_f.reshape(-1)),
                                        PR_T).reshape(nb, n_fine, 3, 5)
        out[lo:hi] = _k_contract(K_all, combo, contravariant_flux_from_metric(det, inv, G))
    return out


def _ref_convection(phi, rho, vel, oi):
    out = np.empty(phi.shape)
    for lo, hi, _n_fine, det, inv, c2f, D_fine, f2c in oi["segs"]:
        phi_f = contract_shared_operator_1axis(c2f, np.ascontiguousarray(phi[lo:hi, :, None]))
        rho_f = contract_shared_operator_1axis(c2f, np.ascontiguousarray(rho[lo:hi, :, None]))
        u_f = contract_shared_operator_1axis(c2f, np.ascontiguousarray(vel[lo:hi]))
        F = (rho_f * phi_f * u_f)[..., None]
        div_f = contract_shared_operator_2axis(D_fine, contravariant_flux_from_metric(det, inv, F))
        out[lo:hi] = contract_shared_operator_1axis(f2c, div_f)[..., 0]
    return out


def _ref_diffusion(gamma, grad, band, oi):
    out = np.empty(gamma.shape)
    for lo, hi, _n_fine, det, inv, c2f, D_fine, f2c in oi["segs"]:
        g_f = smooth_nonnegative_array(
            contract_shared_operator_1axis(c2f, np.ascontiguousarray(gamma[lo:hi, :, None])), band)
        grad_f = contract_shared_operator_1axis(c2f, np.ascontiguousarray(grad[lo:hi]))
        G = (g_f * grad_f)[..., None]
        div_f = contract_shared_operator_2axis(D_fine, contravariant_flux_from_metric(det, inv, G))
        out[lo:hi] = contract_shared_operator_1axis(f2c, div_f)[..., 0]
    return out


# ------------------------------------------------------------------ 输入

def _random_context(rng, n_prism=1300, n_tet=1700, n_sps=10, n_fine_p=18, n_fine_t=20, n_combo=(3, 7)):
    """随机过积分上下文：两段、四面体段细点度量为逐单元常数的步长 0 广播（与 `get_overintegration_context` 同构）。"""
    n_cells = n_prism + n_tet
    det_p = rng.uniform(0.5, 2.0, (n_prism, n_fine_p))
    inv_p = rng.normal(size=(n_prism, n_fine_p, 3, 3))
    det_t = np.broadcast_to(rng.uniform(0.5, 2.0, n_tet)[:, None], (n_tet, n_fine_t))
    inv_t = np.broadcast_to(rng.normal(size=(n_tet, 3, 3))[:, None], (n_tet, n_fine_t, 3, 3))
    segs, lifted = [], []
    for lo, hi, nf, det, inv, nc in ((0, n_prism, n_fine_p, det_p, inv_p, n_combo[0]),
                                     (n_prism, n_cells, n_fine_t, det_t, inv_t, n_combo[1])):
        segs.append((lo, hi, nf, det, inv, rng.normal(size=(nf, n_sps)), rng.normal(size=(nf, nf, 3)),
                     rng.normal(size=(n_sps, nf))))
        lifted.append((rng.normal(size=(nc, n_sps, nf, 3)), rng.integers(0, nc, hi - lo)))
    return dict(segs=tuple(segs), lifted_div=tuple(lifted)), n_cells, n_sps


def _primitive_state(rng, n_cells, n_sps):
    Q = np.empty((n_cells, n_sps, 5))
    Q[..., 0] = rng.uniform(0.8, 1.3, (n_cells, n_sps))
    Q[..., 1:4] = rng.normal(scale=30.0, size=(n_cells, n_sps, 3))
    Q[..., 4] = rng.uniform(0.9e5, 1.1e5, (n_cells, n_sps))
    return Q


def _mesh_context(order):
    mesh = _build_synthetic_mixed_mesh(order)
    ops = generate_fr_operators(order)
    oi = get_overintegration_context(mesh, ops)
    assert oi is not None
    return mesh, ops, oi


# ------------------------------------------------------------------ 判据

class TestRandomContext:
    """单元数远大于并行块数（块边界、空块以外的全部划分路径）；四面体段度量是广播视图。"""

    def test_inviscid(self):
        rng = np.random.default_rng(0)
        oi, n, s = _random_context(rng)
        Q = _primitive_state(rng, n, s)
        assert np.array_equal(inviscid_volume_divergence(Q, oi), _ref_inviscid(Q, oi))

    def test_viscous_with_strided_velocity_gradient(self):
        """`grad_vel` 在生产里是 `grad_Q[:, :, 1:4, :]` 的切片视图；涡粘含负值（细点上走取用值）。"""
        rng = np.random.default_rng(1)
        oi, n, s = _random_context(rng)
        Q = _primitive_state(rng, n, s)
        grad_Q = rng.normal(size=(n, s, 5, 3))
        gT = rng.normal(size=(n, s, 3))
        mut = rng.normal(scale=3 * MU, size=(n, s))
        got = viscous_volume_term(Q, grad_Q[:, :, 1:4, :], gT, mut, MU, PR, PR_T, oi, s)
        assert np.array_equal(got, _ref_viscous(Q, grad_Q[:, :, 1:4, :], gT, mut, oi))

    def test_scalar_convection_with_strided_inputs(self):
        """`rho`/`velocity` 在生产里是原始变量数组的切片视图。"""
        rng = np.random.default_rng(2)
        oi, n, s = _random_context(rng)
        Q = _primitive_state(rng, n, s)
        phi = rng.normal(size=(n, s))
        got = _scalar_convection_volume_overintegrated(phi, Q[:, :, 0], Q[:, :, 1:4], oi, s)
        assert np.array_equal(got, _ref_convection(phi, Q[:, :, 0], Q[:, :, 1:4], oi))

    @pytest.mark.parametrize("band", [0.0, 0.3])
    def test_scalar_diffusion(self, band):
        """扩散系数含负值与窄带内的值（细点取用值 `smooth_nonnegative` 的三个分支都走到）。"""
        rng = np.random.default_rng(3)
        oi, n, s = _random_context(rng)
        gamma = rng.normal(loc=0.2, scale=0.4, size=(n, s))
        grad = rng.normal(size=(n, s, 3))
        got = _scalar_diffusion_volume_overintegrated(gamma, grad, oi, s, band)
        assert np.array_equal(got, _ref_diffusion(gamma, grad, band, oi))


@pytest.mark.parametrize("order", [1, 2, 3])
class TestRealOperators:
    """合成混合网格上的真实算子与细点度量（含原生基零填充槽位）。"""

    def test_mean_flow_and_scalar_volume_terms(self, order):
        mesh, ops, oi = _mesh_context(order)
        rng = np.random.default_rng(10 + order)
        n, s = mesh.n_cells, mesh.n_sps_per_cell
        Q = _primitive_state(rng, n, s)
        assert np.array_equal(inviscid_volume_divergence(Q, oi), _ref_inviscid(Q, oi))
        grad_Q = rng.normal(size=(n, s, 5, 3))
        gT, mut = rng.normal(size=(n, s, 3)), rng.uniform(0.0, 50 * MU, (n, s))
        assert np.array_equal(viscous_volume_term(Q, grad_Q[:, :, 1:4, :], gT, mut, MU, PR, PR_T, oi, s),
                              _ref_viscous(Q, grad_Q[:, :, 1:4, :], gT, mut, oi))
        phi = rng.normal(size=(n, s))
        assert np.array_equal(_scalar_convection_volume_overintegrated(phi, Q[:, :, 0], Q[:, :, 1:4], oi, s),
                              _ref_convection(phi, Q[:, :, 0], Q[:, :, 1:4], oi))
        assert np.array_equal(_scalar_diffusion_volume_overintegrated(mut, gT, oi, s, MU),
                              _ref_diffusion(mut, gT, MU, oi))

    def test_physical_gradient(self, order):
        """链式法则参照取原实现 `np.matmul(np.swapaxes(grad_comp, -1, -2), inv_jacs)`（求和顺序 m=0,1,2 相同）。"""
        mesh, ops, _oi = _mesh_context(order)
        rng = np.random.default_rng(20 + order)
        n, s, npr = mesh.n_cells, mesh.n_sps_per_cell, mesh.n_prism_cells
        wide = rng.normal(size=(n, s, 5))
        inv = mesh.jacobians["inv_jacs"].reshape(n, s, 3, 3)
        for field in (wide, wide[:, :, 1:4]):          # 连续与切片视图两种输入
            V = field.shape[2]
            ref = np.empty((n, s, V, 3))
            for lo, hi, D in ((0, npr, ops.D_native_prism_padded), (npr, n, ops.D_native_tet_padded)):
                D2 = np.ascontiguousarray(np.transpose(D, (0, 2, 1))).reshape(s * 3, s)
                gc = contract_shared_operator_1axis(D2, np.ascontiguousarray(field[lo:hi])).reshape(hi - lo, s, 3, V)
                ref[lo:hi] = np.matmul(np.swapaxes(gc, -1, -2), inv[lo:hi])
            assert np.array_equal(compute_physical_gradient(field, mesh, ops), ref)
