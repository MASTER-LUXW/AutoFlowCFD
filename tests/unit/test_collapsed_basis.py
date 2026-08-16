"""AutoFlowCFD V2.0 - 坍缩坐标单纯形模态基/微分矩阵单元测试。

核心判据：D = V_xi @ inv(V) 必须精确重现模态基自身的解析导数（这是
Vandermonde 矩阵求逆构造微分算子是否正确的直接检验，与具体几何映射
无关），且 Vandermonde 矩阵在当前生产阶数 P=2 下必须良态可逆。
"""

import numpy as np

from autoflowcfd.fr.collapsed_basis import (
    build_collapsed_diff_matrices,
    grad_jacobi_polynomial,
    jacobi_polynomial,
    prism_modal_basis_and_grad,
    tet_modal_basis_and_grad,
)
from autoflowcfd.fr.operators import gauss_legendre


def _ref_cube_sps(order: int) -> np.ndarray:
    n1d = order + 1
    sps_1d, _ = gauss_legendre(n1d)
    xx, yy, zz = np.meshgrid(sps_1d, sps_1d, sps_1d, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])


def test_jacobi_polynomial_low_order_sanity():
    x = np.array([-0.5, 0.0, 0.3, 0.9])
    assert np.allclose(jacobi_polynomial(x, 0, 0, 0), 1.0)
    assert np.allclose(jacobi_polynomial(x, 0, 0, 1), x)  # P_1^(0,0)(x) = x
    # numerical derivative check for a higher (alpha,beta,n) combination
    eps = 1e-6
    n, alpha, beta = 3, 2, 1
    analytic = grad_jacobi_polynomial(x, alpha, beta, n)
    numeric = (jacobi_polynomial(x + eps, alpha, beta, n) - jacobi_polynomial(x - eps, alpha, beta, n)) / (2 * eps)
    assert np.max(np.abs(analytic - numeric)) < 1e-6


def test_vandermonde_well_conditioned_at_production_order():
    """P=2 是本项目当前实际生产阶数（见 cube_demo 全流程验证）。"""
    order = 2
    ref = _ref_cube_sps(order)
    a, b, c = ref[:, 0], ref[:, 1], ref[:, 2]
    for cell_type, basis_fn in [("tet", tet_modal_basis_and_grad), ("prism", prism_modal_basis_and_grad)]:
        V, _, _, _ = basis_fn(a, b, c, order)
        cond = np.linalg.cond(V)
        assert cond < 1e7, f"{cell_type} Vandermonde ill-conditioned at P={order}: cond={cond:.3e}"


def test_diff_matrix_exactly_reproduces_mode_derivatives():
    """D = V_xi @ inv(V) 必须精确重现模态基自身的解析导数——与所用哪组
    (可逆的)基构造无关，是微分矩阵构造正确性的数学判据。"""
    order = 2
    ref = _ref_cube_sps(order)
    a, b, c = ref[:, 0], ref[:, 1], ref[:, 2]
    for cell_type, basis_fn in [("tet", tet_modal_basis_and_grad), ("prism", prism_modal_basis_and_grad)]:
        D = build_collapsed_diff_matrices(cell_type, order, ref)
        V, Va, Vb, Vc = basis_fn(a, b, c, order)
        n_modes = V.shape[1]
        max_err = 0.0
        for mode in range(n_modes):
            nodal_vals = V[:, mode]
            da = D[:, :, 0] @ nodal_vals
            db = D[:, :, 1] @ nodal_vals
            dc = D[:, :, 2] @ nodal_vals
            err = max(
                np.max(np.abs(da - Va[:, mode])),
                np.max(np.abs(db - Vb[:, mode])),
                np.max(np.abs(dc - Vc[:, mode])),
            )
            max_err = max(max_err, err)
        assert max_err < 1e-8, f"{cell_type}: D matrix does not reproduce mode derivatives, err={max_err:.3e}"
