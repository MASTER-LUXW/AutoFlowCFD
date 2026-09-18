"""AutoFlowCFD V2.0 - 物理空间梯度算子单元测试。

核心判据：对任意非退化四面体/棱柱单元，线性物理函数的梯度必须精确
恢复为其真实常数梯度（这是曲边/坍缩坐标度量项变换是否正确的直接检验）。
"""

from types import SimpleNamespace

import numpy as np
import pytest

from autoflowcfd.core.fr_operators.gradients import compute_physical_gradient, compute_physical_scalar_gradient
from autoflowcfd.grid.curved_mapping.curved_mapping import map_prism_to_physical, map_tet_to_physical
from autoflowcfd.fr.operators import generate_fr_operators, gauss_legendre
from autoflowcfd.fr.native_simplex_basis import build_native_tet_operators
from autoflowcfd.grid.high_order.high_order_mesh import HighOrderMesh


class _MockNodes:
    def __init__(self, coords):
        self._coords = coords

    def get_coordinates(self):
        return self._coords


class _MockCells:
    def __init__(self, connectivity):
        self.connectivity = connectivity


def _build_mesh(order):
    nodes = np.array(
        [
            [0, 0, 0], [1, 0, 0], [0.2, 1, 0], [0.1, 0.3, 1],
            [5, 0, 0], [6, 0, 0], [5, 1, 0], [5.3, 0.2, 1], [6.1, 0.1, 1], [5.2, 1.3, 1],
        ],
        dtype=float,
    )
    tet_conn = np.array([[0, 1, 2, 3]], dtype=np.int32)
    prism_conn = np.array([[4, 5, 6, 7, 8, 9]], dtype=np.int32)
    mock_volume = SimpleNamespace(
        cell_count=2,
        nodes=_MockNodes(nodes),
        cells=_MockCells(tet_conn),
        prism_cells=_MockCells(prism_conn),
        boundaries=None,
    )
    mesh = HighOrderMesh(order=order)
    mesh.load_from_volume_mesh(mock_volume, build_faces=False)
    return mesh


def test_linear_function_gradient_exact_for_tet_and_prism():
    # P=2 是本项目当前实际生产阶数，要求机器精度；P=3 容差沿用同一档
    # （四面体 native 单纯形基在高阶下的条件数特征与坍缩坐标不同，
    # 但同样在此判据下有界，见 native_simplex_basis.py 文档）。
    #
    # 2026-09-03 更正：四面体坍缩坐标基已删除（见 fr/operators.py 模块
    # 文档），四面体单元的梯度输出在"零填充块对角"约定下，填充行
    # （`[n_native:]`）恒为零梯度（`D_native_tet_padded` 的填充行本身
    # 是零，不是真实自由度的物理梯度取值，见 native_padding.py
    # 文档）——这是既有、已验证的设计不变量，不是本次改动引入的新
    # 近似；对比时必须只看真实自由度（`[:n_native]`），不能再要求
    # 填充行也精确等于常数梯度。
    tolerances = {1: 1e-9, 2: 1e-9, 3: 1e-6}
    for order in [1, 2, 3]:
        mesh = _build_mesh(order)
        n_native = build_native_tet_operators(order)[0].shape[0]
        a_coef = np.array([2.0, -3.0, 5.0])
        phi = mesh.sps_coords @ a_coef + 7.0  # (n_cells, n_sps)
        grad = compute_physical_scalar_gradient(phi, mesh, mesh.operators)
        # 单元全局索引约定"棱柱在前、四面体在后"（见 HighOrderMesh 模块
        # 文档）：cell 0 是棱柱（不受本次删除 collapsed 四面体基影响，
        # 全部检查），cell 1 是四面体（只检查真实自由度，填充行恒为零
        # 梯度，见上）。
        max_err_prism = np.max(np.abs(grad[0] - a_coef))
        max_err_tet = np.max(np.abs(grad[1, :n_native] - a_coef))
        max_err = max(max_err_tet, max_err_prism)
        assert max_err < tolerances[order], f"order={order}: max_err={max_err}"


def test_multi_variable_field_gradient_matches_scalar_case():
    mesh = _build_mesh(order=2)
    n_native = build_native_tet_operators(2)[0].shape[0]
    a1 = np.array([1.0, 0.0, 0.0])
    a2 = np.array([0.0, 2.0, 0.0])
    phi1 = mesh.sps_coords @ a1
    phi2 = mesh.sps_coords @ a2
    field = np.stack([phi1, phi2], axis=-1)  # (n_cells, n_sps, 2)
    grad = compute_physical_gradient(field, mesh, mesh.operators)  # (n_cells,n_sps,2,3)
    # 见 test_linear_function_gradient_exact_for_tet_and_prism 同一处
    # 更正说明（"棱柱在前、四面体在后"）：cell 0 是棱柱，全部检查；
    # cell 1 是四面体，只检查真实自由度 [:n_native]。
    assert np.allclose(grad[0, :, 0, :], a1, atol=1e-9)
    assert np.allclose(grad[0, :, 1, :], a2, atol=1e-9)
    assert np.allclose(grad[1, :n_native, 0, :], a1, atol=1e-9)
    assert np.allclose(grad[1, :n_native, 1, :], a2, atol=1e-9)


# ---------------------------------------------------------------------------
# 去混叠（过积分）**不适用于本算子** —— 一条被自己的数据证伪的假设
# ---------------------------------------------------------------------------
#
# 2026-09-15 排查"哪些地方缺反混叠"时，本算子曾被列为疑点：它算的是
# `grad_phys = inv_jac · (D · field)`，而 `inv_jac` 在非仿射单元上是
# **有理函数**，看起来又是一个"乘积"。
#
# 但那条类比不成立。去混叠要解决的是"**先混叠再求导**"——把一个真实次数
# 高于 order 的乘积投影到 degree-order 空间之后再对它微分，投影误差被
# 微分算子放大（`fr/collapsed_basis.py::build_overintegration_operators`
# 记录的实测：不去混叠的 P2 体积项对解析残差恒为 0 的线性剪切场算出的
# 残差是真值的 43~62 倍）。本算子里 `D · field` 在 field 自身的多项式
# 空间内是**精确**的，之后乘 `inv_jac` 只是**逐点求值**、不再求导，
# 所以那条放大链根本不存在。
#
# 实测证据（解空间内多项式、总次数 == order，梯度有闭式解）：
#
#     order=1: prism 相对误差 2.243e-14   tet(真实自由度) 3.697e-16
#     order=2: prism 相对误差 9.490e-14   tet(真实自由度) 2.794e-16
#     order=3: prism 相对误差 1.343e-12   tet(真实自由度) 7.717e-17
#
# 也就是机器精度。**不要给这个算子加过积分**：那只会增加成本而不改善
# 任何东西。去混叠有意义的位置是"乘积被求导"的那些项，现已全部覆盖：
# 无粘体积项（`fr_residual/inviscid.py`，一直开着）、粘性体积项
# （`fr_residual/viscous_flux.py`，AFCFD_VISC_OVERINT）、k/omega 对流与
# 扩散体积项（`core/turbulence/transport.py`，AFCFD_TURB_OVERINT）。

@pytest.mark.parametrize("order", [1, 2, 3])
def test_gradient_is_exact_for_in_space_polynomials(order):
    """本算子对解空间内的多项式必须是机器精度精确的。

    这条同时是"不需要给梯度算子加去混叠"的依据（见上方说明）：一旦有人
    改动实现导致这里退化，说明引入了真实的表示误差，那时才需要重新评估。
    """
    from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh

    mesh = _build_synthetic_mixed_mesh(order)
    ops = generate_fr_operators(order)
    n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
    n_prism = mesh.n_prism_cells
    n_native = (order + 1) * (order + 2) * (order + 3) // 6
    X = mesh.sps_coords.reshape(-1, 3)

    # 总次数 <= order 的全部单项式（严格落在解空间内）
    terms = [(a, b, c)
             for a in range(order + 1)
             for b in range(order + 1 - a)
             for c in range(order + 1 - a - b)]
    coeff = np.random.default_rng(order).uniform(-1.0, 1.0, len(terms))

    def f(Y):
        return sum(k * Y[:, 0] ** a * Y[:, 1] ** b * Y[:, 2] ** c
                   for k, (a, b, c) in zip(coeff, terms))

    def grad_f(Y):
        g = np.zeros((Y.shape[0], 3))
        for k, (a, b, c) in zip(coeff, terms):
            if a > 0:
                g[:, 0] += k * a * Y[:, 0] ** (a - 1) * Y[:, 1] ** b * Y[:, 2] ** c
            if b > 0:
                g[:, 1] += k * b * Y[:, 0] ** a * Y[:, 1] ** (b - 1) * Y[:, 2] ** c
            if c > 0:
                g[:, 2] += k * c * Y[:, 0] ** a * Y[:, 1] ** b * Y[:, 2] ** (c - 1)
        return g

    got = compute_physical_gradient(
        f(X).reshape(n_cells, n_sps, 1), mesh, ops)[:, :, 0, :]
    exact = grad_f(X).reshape(n_cells, n_sps, 3)
    scale = np.abs(exact).max()

    err_prism = np.abs(got[:n_prism] - exact[:n_prism]).max() / scale
    err_tet = np.abs(
        got[n_prism:, :n_native] - exact[n_prism:, :n_native]).max() / scale
    assert err_prism < 1e-11, (
        f"order={order} 棱柱梯度相对误差 {err_prism:.3e} 不是机器精度——"
        f"实测应为 ~1e-14~1e-12")
    assert err_tet < 1e-13, (
        f"order={order} 四面体（真实自由度）梯度相对误差 {err_tet:.3e} "
        f"不是机器精度——实测应为 ~1e-16")
