"""AutoFlowCFD V2.0 - native 四面体（路径C）体积项去混叠（过积分）算子
(`fr/native_tet/overintegration.py`) 决定性验证。
"""

import numpy as np
import pytest

from autoflowcfd.fr.native_tet.overintegration import build_native_tet_overintegration_operators
from autoflowcfd.fr.native_tet.basis import (
    build_native_tet_operators, map_native_tet_to_physical, compute_native_tet_jacobian,
    restricted_tet_modes,
)
from autoflowcfd.fr.collapsed_basis import OVERINTEGRATION_MAX_ORDER

TET_NODES = np.array([
    [0.1, -0.3, 0.2], [1.2, 0.0, -0.1], [-0.1, 1.1, 0.0], [0.0, 0.1, 1.3],
], dtype=float)


def _over_order(order):
    return min(2 * order, OVERINTEGRATION_MAX_ORDER)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_shapes(order):
    over_order = _over_order(order)
    ref_coarse, _ = build_native_tet_operators(order)
    ref_fine, D_fine = build_native_tet_operators(over_order)
    n_coarse, n_fine = ref_coarse.shape[0], ref_fine.shape[0]

    ref_fine_out, interp_c2f, D_fine_out, restrict_f2c = build_native_tet_overintegration_operators(
        order, over_order
    )
    assert ref_fine_out.shape == (n_fine, 3)
    assert interp_c2f.shape == (n_fine, n_coarse)
    assert D_fine_out.shape == (n_fine, n_fine, 3)
    np.testing.assert_allclose(D_fine_out, D_fine, atol=1e-13)
    assert restrict_f2c.shape == (n_coarse, n_fine)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_interp_c2f_exact_for_affine_field(order):
    """coarse 阶数模态基在 fine 点上取值——Q 本身次数<=order 时必须精确
    插值（不引入混叠），机器精度。"""
    over_order = _over_order(order)
    ref_coarse, _ = build_native_tet_operators(order)
    ref_fine, _, D_fine, _ = build_native_tet_overintegration_operators(order, over_order)

    phys_coarse = map_native_tet_to_physical(ref_coarse, TET_NODES)
    phys_fine = map_native_tet_to_physical(ref_fine, TET_NODES)

    def field(phys):
        x, y, z = phys[:, 0], phys[:, 1], phys[:, 2]
        return 1.0 + 2.0 * x - 3.0 * y + 0.5 * z

    _, interp_c2f, _, _ = build_native_tet_overintegration_operators(order, over_order)
    field_coarse = field(phys_coarse)
    field_fine_via_interp = interp_c2f @ field_coarse
    field_fine_exact = field(phys_fine)
    np.testing.assert_allclose(field_fine_via_interp, field_fine_exact, atol=1e-9)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_restrict_f2c_exact_for_fine_degree_field(order):
    """fine 阶数模态基在 coarse 点上取值——一个真正 degree<=over_order
    （可能 >order）的场，从 fine 节点值精确恢复到 coarse 点，必须与
    解析函数本身在 coarse 点上的值一致（这是"精确插值"，不是"模态
    截断投影"，过积分的目的就是要保留这部分高阶内容，见模块文档）。
    """
    over_order = _over_order(order)
    if over_order <= order:
        pytest.skip(f"order={order}: over_order={over_order} 未真正提升阶数，跳过")

    ref_coarse, _ = build_native_tet_operators(order)
    ref_fine, _, _, restrict_f2c = build_native_tet_overintegration_operators(order, over_order)

    phys_coarse = map_native_tet_to_physical(ref_coarse, TET_NODES)
    phys_fine = map_native_tet_to_physical(ref_fine, TET_NODES)

    # 构造一个真正 degree==over_order 的多项式场（超出 coarse 能精确
    # 表示的范围），验证 restrict_f2c 仍然精确保留这部分内容。
    def field_high_degree(phys):
        x, y, z = phys[:, 0], phys[:, 1], phys[:, 2]
        return x ** over_order + 0.3 * y ** over_order - 0.7 * z ** over_order + x * y * z

    field_fine = field_high_degree(phys_fine)
    field_coarse_via_restrict = restrict_f2c @ field_fine
    field_coarse_exact = field_high_degree(phys_coarse)
    np.testing.assert_allclose(field_coarse_via_restrict, field_coarse_exact, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_full_pipeline_kills_constant_field(order):
    """完整过积分链路（插值到细网格→微分→限制回粗网格）作用在常数场
    上必须恒为 0——最基本的相容性检验，任何一步有系统性错误都会破坏它。
    """
    over_order = _over_order(order)
    ref_coarse, _ = build_native_tet_operators(order)
    n_coarse = ref_coarse.shape[0]

    _, interp_c2f, D_fine, restrict_f2c = build_native_tet_overintegration_operators(order, over_order)

    const_coarse = np.full(n_coarse, 7.0)
    const_fine = interp_c2f @ const_coarse
    for m in range(3):
        deriv_fine = D_fine[:, :, m] @ const_fine
        deriv_coarse = restrict_f2c @ deriv_fine
        assert np.max(np.abs(deriv_coarse)) < 1e-8, f"order={order}, m={m}: 常数场经完整过积分链路后散度应为 0"
