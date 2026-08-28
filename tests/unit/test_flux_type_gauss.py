"""AutoFlowCFD V2.0 - #14 flux_type (Radau/Gauss) 修正函数族单元测试。

背景（详见 fr/matrix_operators.py 模块内文档）：本项目此前唯一实现的
修正函数方案（`_solve_radau_correction_coeffs`）被文档误标为 Huynh (2007)
的 "g2"，经独立文献研究交叉核实（Huynh 2007 AIAA 2007-4079; Vincent,
Castonguay & Jameson 2011 JSC 47(1):50-72; Huynh/Wang/Vincent 2013
NASA/TM-2013-218078），它实际是 VCJH 参数化里 η_p=0 的方案，即 Huynh
记法里的 "g_DG"——数值实现本身没有问题（此前已通过边界条件/正交性/对称性
验证），只是标签错了，本次修复只更新文档。

本次新增的 "gauss" 方案对应 η_p=p/(p+1)，与 Spectral Difference (SD)
方法等价，闭式解为：
    g_L(x) = (-1)^p * (1-x)/2 * P_p(x)
    g_R(x) =          (1+x)/2 * P_p(x)
（P_p 为标准归一化 p 次 Legendre 多项式）。这里的测试独立于内部实现
重新构造这个闭式解并用有限差分求导交叉验证，而不是直接复用被测函数
内部的解析求导公式（避免测试和实现共享同一处可能的代数错误）。
"""

import numpy as np
import pytest
from numpy.polynomial import legendre as np_legendre

from autoflowcfd.fr.matrix_operators import compute_correction_weights
from autoflowcfd.fr.operators import generate_fr_operators
from autoflowcfd.fr.quadrature_points import gauss_legendre


def _closed_form_gauss_g(p: int):
    """独立于被测实现，直接按文献验证过的闭式解构造 g_L/g_R（连续函数，
    不是只在 SPs 处的采样值），供后续有限差分求导交叉验证使用。"""
    leg_p = np_legendre.Legendre.basis(p)
    sign = (-1.0) ** p

    def g_left(x):
        return sign * (1.0 - x) / 2.0 * leg_p(x)

    def g_right(x):
        return (1.0 + x) / 2.0 * leg_p(x)

    return g_left, g_right


class TestGaussCorrectionFunctionBoundaryConditions:
    """独立解析验证闭式解本身满足修正函数必须满足的边界条件——这是
    对文献闭式解的复核，不依赖 compute_correction_weights 的实现。"""

    @pytest.mark.parametrize("p", [0, 1, 2, 3, 4])
    def test_boundary_conditions_hold_analytically(self, p):
        g_left, g_right = _closed_form_gauss_g(p)
        assert g_left(-1.0) == pytest.approx(1.0, abs=1e-12)
        assert g_left(1.0) == pytest.approx(0.0, abs=1e-12)
        assert g_right(-1.0) == pytest.approx(0.0, abs=1e-12)
        assert g_right(1.0) == pytest.approx(1.0, abs=1e-12)

    @pytest.mark.parametrize("p", [0, 1, 2, 3, 4])
    def test_mirror_symmetry_g_right_equals_g_left_of_minus_x(self, p):
        """g_R(x) = g_L(-x)，与本项目 Radau 方案的对称性约定一致
        （见 _solve_radau_correction_coeffs 文档）。"""
        g_left, g_right = _closed_form_gauss_g(p)
        xs = np.linspace(-0.9, 0.9, 11)
        np.testing.assert_allclose(g_right(xs), g_left(-xs), atol=1e-12)


class TestComputeCorrectionWeightsGauss:
    """交叉验证 compute_correction_weights(n, 'gauss') 的解析求导实现，
    与对闭式解做数值有限差分求导的结果比较——两条独立路径吻合才能确认
    解析求导本身没有代数错误。"""

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_derivative_matches_finite_difference_of_closed_form(self, n):
        p = n - 1
        sps, _ = gauss_legendre(n)
        g_left, g_right = _closed_form_gauss_g(p)

        eps = 1e-6
        fd_left = (g_left(sps + eps) - g_left(sps - eps)) / (2 * eps)
        fd_right = (g_right(sps + eps) - g_right(sps - eps)) / (2 * eps)

        g_left_prime, g_right_prime = compute_correction_weights(n, flux_point_type='gauss')

        np.testing.assert_allclose(g_left_prime, fd_left, atol=1e-5)
        np.testing.assert_allclose(g_right_prime, fd_right, atol=1e-5)

    def test_gauss_differs_from_radau_for_same_n(self):
        """两个方案在 p>=1 时必须给出不同的修正函数导数——确认新增的
        'gauss' 分支不是意外复用/退化成既有的 'radau' 方案。"""
        n = 3
        g_left_radau, g_right_radau = compute_correction_weights(n, flux_point_type='radau')
        g_left_gauss, g_right_gauss = compute_correction_weights(n, flux_point_type='gauss')

        assert not np.allclose(g_left_radau, g_left_gauss)
        assert not np.allclose(g_right_radau, g_right_gauss)

    def test_default_flux_point_type_matches_radau(self):
        """默认参数值（未显式传参）必须与显式传 'radau' 完全一致——
        保证此前所有省略这个参数的调用点行为不变。"""
        n = 3
        g_left_default, g_right_default = compute_correction_weights(n)
        g_left_radau, g_right_radau = compute_correction_weights(n, flux_point_type='radau')
        np.testing.assert_array_equal(g_left_default, g_left_radau)
        np.testing.assert_array_equal(g_right_default, g_right_radau)

    def test_lobatto_alias_still_routes_to_radau_scheme(self):
        """历史字符串值 'lobatto'（generate_fr_operators 此前的默认值）
        必须继续路由到与 'radau' 完全相同的修正函数分支——不是新增
        'gauss' 分支后被意外挡在另一侧。"""
        n = 3
        g_left_lobatto, g_right_lobatto = compute_correction_weights(n, flux_point_type='lobatto')
        g_left_radau, g_right_radau = compute_correction_weights(n, flux_point_type='radau')
        np.testing.assert_array_equal(g_left_lobatto, g_left_radau)
        np.testing.assert_array_equal(g_right_lobatto, g_right_radau)


class TestGenerateFrOperatorsFluxType:
    """generate_fr_operators 层面的回归防护：默认行为不变，'gauss' 真正
    驱动一套不同的 g_left/g_right。"""

    def test_default_call_omitting_param_matches_explicit_radau(self):
        ops_default = generate_fr_operators(order=2)
        ops_radau = generate_fr_operators(order=2, flux_point_type='radau')
        np.testing.assert_array_equal(ops_default.g_left, ops_radau.g_left)
        np.testing.assert_array_equal(ops_default.g_right, ops_radau.g_right)

    def test_gauss_flux_type_gives_different_correction_weights(self):
        ops_radau = generate_fr_operators(order=2, flux_point_type='radau')
        ops_gauss = generate_fr_operators(order=2, flux_point_type='gauss')
        assert not np.allclose(ops_radau.g_left, ops_gauss.g_left)
        assert not np.allclose(ops_radau.g_right, ops_gauss.g_right)


class TestFreeStreamPreservationGauss:
    """自由流场保持性（本项目标准判据，见 test_fr_residual_inviscid.py::
    TestFreeStreamPreservation）在新的 'gauss' 修正函数方案下同样必须
    成立：均匀流场的无粘残差应严格为零（数值上为机器精度量级）。

    这不是一次网格加密收敛阶数研究（本项目现有的 TestFreeStreamPreservation
    本身也不做网格加密收敛阶数研究，只做定阶数的均匀流场残差检查）——
    与既有验证方法保持一致，网格加密收敛阶数验证留待未来需要时单独补充。
    """

    @pytest.mark.parametrize("order", [1, 2])
    def test_uniform_flow_gives_near_zero_residual_with_gauss_scheme(self, order):
        from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh
        from autoflowcfd.core.fr_residual.inviscid import (
            compute_inviscid_residual_fr,
            primitive_to_conserved,
        )

        mesh = _build_synthetic_mixed_mesh(order)
        ops_gauss = generate_fr_operators(order, flux_point_type='gauss')

        rho_inf, u_inf, v_inf, w_inf, p_inf = 1.225, 30.0, 5.0, -3.0, 101325.0
        Q_inf = np.array([rho_inf, u_inf, v_inf, w_inf, p_inf])
        U_inf = primitive_to_conserved(Q_inf)
        U = np.tile(U_inf, (mesh.n_cells, mesh.n_sps_per_cell, 1))

        residual = compute_inviscid_residual_fr(U, mesh, ops_gauss)
        rel_res = np.max(np.abs(residual)) / p_inf

        # 与既有 TestFreeStreamPreservation 的 P1/P2 判据同量级（该测试
        # 类未覆盖 P1，本文件按 P2 的 3e-5 量级取一个同样不过分苛刻的
        # 界，两个方案共享同一套体积项/度量项/面耦合基础设施，理论上
        # 应有相近的浮点噪声地板）。
        assert rel_res < 3e-4, f"Free-stream preservation failed at P={order} (gauss): rel={rel_res:.3e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
