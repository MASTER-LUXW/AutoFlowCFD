"""Unit tests for core/utils/order_continuation.py's vectorized SPs
prolongation matrix.

`_build_linear_interp_matrix_3d` replaced a per-cell, per-variable Python
loop that constructed a fresh `scipy.interpolate.RegularGridInterpolator`
for every (cell, variable) pair - real performance bottleneck on
production-scale meshes (hundreds of thousands of cells) at every Order
Continuation phase transition.

真实 bug 修复（2026-09-02）：这些测试此前把"新实现是否与原始
scipy.interpolate.RegularGridInterpolator(method='linear') 逐单元循环
数值一致"当成正确性判据——但那个 scipy 循环本身就是 bug：它在旧 SPs
之间做分段多线性插值，不是对旧 SPs 隐含的那个多项式的精确求值。对
old_order<=1（P0->P1、P1->P2）分段线性恰好与真正的常数/线性多项式
重合，掩盖了这个问题；但对 old_order>=2（P2->P3，本项目实际会用到的
转换）二者系统性不同——用一个真实的三次多项式场测得两者相对误差高达
19.7%（`test_old_scipy_loop_method_was_wrong_for_p2_to_p3` 记录了这个
数字，作为这次修复"确实修了一个真bug、不是无意义的重构"的独立证据）。

新的判据（`TestBuildLinearInterpMatrix3D`）：验证新实现对旧 SPs 隐含的
多项式做精确解析延拓（Order Continuation 只单调升阶，新的张量积多项式
空间严格包含旧空间，因此精确延拓总是可能、不是近似）——用已知次数为
old_order 的随机张量积多项式场，在新 SPs 上直接解析求值作为 ground
truth，断言矩阵作用结果与其在浮点舍入误差范围内一致，而不是与旧的
（本身有 bug 的）scipy 循环比对。
"""

import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator

from autoflowcfd.core.utils.order_continuation import (
    _build_linear_interp_matrix_3d,
    _lagrange_basis_matrix_1d,
)
from autoflowcfd.fr.quadrature_points import gauss_legendre


def _random_tensor_poly_1d_coeffs(rng, degree: int) -> np.ndarray:
    return rng.uniform(-2.0, 2.0, degree + 1)


def _eval_poly_1d(coeffs: np.ndarray, x) -> np.ndarray:
    return sum(c * x ** i for i, c in enumerate(coeffs))


def _tensor_poly_nodal_values(coeffs_x, coeffs_y, coeffs_z, sps_1d: np.ndarray) -> np.ndarray:
    """在 sps_1d 张量积网格（C-order 嵌套：x 最外层、z 最内层，与
    `_build_linear_interp_matrix_3d` 的 meshgrid(indexing='ij')+ravel
    约定一致）上对可分离张量积多项式 phi(x,y,z)=px(x)*py(y)*pz(z) 逐点
    求值，返回展平后的一维数组，形状 (len(sps_1d)**3,)。"""
    xx, yy, zz = np.meshgrid(sps_1d, sps_1d, sps_1d, indexing='ij')
    vals = (
        _eval_poly_1d(coeffs_x, xx)
        * _eval_poly_1d(coeffs_y, yy)
        * _eval_poly_1d(coeffs_z, zz)
    )
    return vals.reshape(-1)


class TestLagrangeBasisMatrix1D:
    def test_reproduces_identity_at_matching_nodes(self):
        """节点自身处的取值必须精确是 Kronecker delta（0/1 浮点噪声都
        不该有），不能靠数值巧合得到。"""
        nodes, _ = gauss_legendre(3)
        L = _lagrange_basis_matrix_1d(nodes, nodes)
        np.testing.assert_array_equal(L, np.eye(len(nodes)))

    def test_single_node_broadcasts_constant_one(self):
        """P0 阶段每个方向只有 1 个参考点——退化的'Lagrange 基'是常数
        函数 1，必须原样广播到任意新节点，不能除以零/报错。"""
        old_nodes, _ = gauss_legendre(1)
        new_nodes, _ = gauss_legendre(2)
        L = _lagrange_basis_matrix_1d(old_nodes, new_nodes)
        np.testing.assert_allclose(L, np.ones((2, 1)))

    @pytest.mark.parametrize("old_order", [1, 2, 3])
    def test_exactly_reproduces_polynomial_of_matching_degree(self, old_order):
        """核心正确性判据：old_nodes 唯一确定一个次数为 old_order 的
        1D 多项式，L 必须能在任意新点上精确求出这个多项式的解析值
        （不是数值插值近似）。"""
        rng = np.random.default_rng(1234 + old_order)
        old_nodes, _ = gauss_legendre(old_order + 1)
        new_nodes, _ = gauss_legendre(old_order + 3)  # 任意更密的新节点集合
        coeffs = _random_tensor_poly_1d_coeffs(rng, old_order)

        old_vals = _eval_poly_1d(coeffs, old_nodes)
        L = _lagrange_basis_matrix_1d(old_nodes, new_nodes)
        new_vals_via_L = L @ old_vals
        new_vals_true = _eval_poly_1d(coeffs, new_nodes)

        np.testing.assert_allclose(new_vals_via_L, new_vals_true, atol=1e-10, rtol=1e-10)


class TestBuildLinearInterpMatrix3D:
    @pytest.mark.parametrize("old_order,new_order", [(0, 1), (1, 2), (2, 3)])
    def test_exactly_reproduces_tensor_polynomial_of_matching_degree(self, old_order, new_order):
        """决定性判据（取代此前"与旧 scipy 循环数值一致"的错误判据，
        见模块文档）：old SPs 隐含的次数为 old_order 的张量积多项式，
        必须在 new SPs 上被精确解析求值，机器精度级误差，涵盖本项目
        实际使用的全部三组阶数转换（P0->P1/P1->P2/P2->P3）。"""
        rng = np.random.default_rng(42 + old_order)
        old_sps_1d, _ = gauss_legendre(old_order + 1)
        new_sps_1d, _ = gauss_legendre(new_order + 1)

        coeffs_x = _random_tensor_poly_1d_coeffs(rng, old_order)
        coeffs_y = _random_tensor_poly_1d_coeffs(rng, old_order)
        coeffs_z = _random_tensor_poly_1d_coeffs(rng, old_order)

        old_vals = _tensor_poly_nodal_values(coeffs_x, coeffs_y, coeffs_z, old_sps_1d)
        new_vals_true = _tensor_poly_nodal_values(coeffs_x, coeffs_y, coeffs_z, new_sps_1d)

        W = _build_linear_interp_matrix_3d(old_sps_1d, new_sps_1d)
        new_vals_via_W = W @ old_vals

        np.testing.assert_allclose(new_vals_via_W, new_vals_true, atol=1e-9, rtol=1e-9)

    def test_vectorized_einsum_application_matches_per_cell_manual_application(self):
        """向量化 einsum 应用（生产路径：一次矩阵乘法覆盖全部单元/
        变量）必须与逐单元手动应用同一个 W 矩阵完全一致——这里只验证
        向量化本身没有引入错误，W 矩阵自身的正确性由上一个测试保证。"""
        old_order, new_order = 2, 3
        old_sps_1d, _ = gauss_legendre(old_order + 1)
        new_sps_1d, _ = gauss_legendre(new_order + 1)
        W = _build_linear_interp_matrix_3d(old_sps_1d, new_sps_1d)

        rng = np.random.default_rng(7)
        n_cells, n_vars = 12, 5
        old_n_sps = (old_order + 1) ** 3
        U_old = rng.standard_normal((n_cells, old_n_sps, n_vars))

        actual = np.einsum('ab,cbv->cav', W, U_old)

        expected = np.zeros((n_cells, (new_order + 1) ** 3, n_vars))
        for i in range(n_cells):
            for v in range(n_vars):
                expected[i, :, v] = W @ U_old[i, :, v]

        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_p0_single_point_grid_broadcasts_constant(self):
        """P0's reference 'grid' is a single point per axis - the matrix
        must broadcast that one value to every new SP, not raise."""
        old_sps_1d, _ = gauss_legendre(1)
        new_sps_1d, _ = gauss_legendre(2)

        W = _build_linear_interp_matrix_3d(old_sps_1d, new_sps_1d)

        assert W.shape == (8, 1)
        np.testing.assert_allclose(W, np.ones((8, 1)))

    def test_identity_when_grids_match(self):
        """Interpolating onto the exact same SPs must reproduce the field
        exactly (the matrix should behave as an identity for this case)."""
        sps_1d, _ = gauss_legendre(2)
        W = _build_linear_interp_matrix_3d(sps_1d, sps_1d)
        np.testing.assert_allclose(W, np.eye(8), atol=1e-12)

    def test_old_scipy_loop_method_was_wrong_for_p2_to_p3(self):
        """回归证据（不是本模块正常验证路径的一部分）：证明此前的
        scipy.interpolate.RegularGridInterpolator(method='linear') 逐
        单元循环对 P2->P3 转换确实有真实的、非平凡的错误（不是"少许
        精度差异"），量化这次修复解决的问题有多大。这个测试锁定的是
        *旧实现的错误程度*，新实现（上面几个测试）不应该、也不会复现
        这个误差。"""
        old_order, new_order = 2, 3
        rng = np.random.default_rng(42 + old_order)
        old_sps_1d, _ = gauss_legendre(old_order + 1)
        new_sps_1d, _ = gauss_legendre(new_order + 1)
        old_n1d, new_n1d = old_order + 1, new_order + 1

        coeffs_x = _random_tensor_poly_1d_coeffs(rng, old_order)
        coeffs_y = _random_tensor_poly_1d_coeffs(rng, old_order)
        coeffs_z = _random_tensor_poly_1d_coeffs(rng, old_order)

        old_vals_3d = _tensor_poly_nodal_values(
            coeffs_x, coeffs_y, coeffs_z, old_sps_1d
        ).reshape((old_n1d, old_n1d, old_n1d))
        new_vals_true = _tensor_poly_nodal_values(coeffs_x, coeffs_y, coeffs_z, new_sps_1d)

        new_xx, new_yy, new_zz = np.meshgrid(new_sps_1d, new_sps_1d, new_sps_1d, indexing='ij')
        new_pts = np.column_stack([new_xx.ravel(), new_yy.ravel(), new_zz.ravel()])
        old_scipy_interp = RegularGridInterpolator(
            (old_sps_1d, old_sps_1d, old_sps_1d), old_vals_3d,
            method='linear', bounds_error=False, fill_value=None,
        )
        old_method_vals = old_scipy_interp(new_pts)

        rel_err = np.max(np.abs(old_method_vals - new_vals_true)) / np.max(np.abs(new_vals_true))
        assert rel_err > 0.1, (
            f"预期旧 scipy 分段线性方法在 P2->P3 上有非平凡误差（历史实测约 19.7%），"
            f"实测 rel_err={rel_err:.4e} 远小于预期——如果这个断言开始失败，说明"
            f"random seed 巧合选到了一个旧方法意外精确的多项式，应换一个 seed 而不是"
            f"删除这个回归证据测试。"
        )
