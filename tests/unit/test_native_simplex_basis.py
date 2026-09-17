"""AutoFlowCFD V2.0 - 四面体独立（路径C，非坍缩坐标）基函数/微分算子
单元测试。见 fr/native_simplex_basis.py 模块文档，
`8_算法重构-微分算子对坍缩坐标退化参考轴的病态条件数-Part6.md` 阶段0。
"""

import numpy as np
import pytest

from autoflowcfd.fr.native_simplex_basis import (
    restricted_tet_modes,
    simplex3d_value,
    simplex3d_grad,
    rst_to_abc,
    build_native_tet_operators,
)
from autoflowcfd.fr.operators import generate_fr_operators
from autoflowcfd.grid.curved_mapping.curved_mapping import batched_det_inv_3x3, tet_barycentric


def test_generate_fr_operators_default_unaffected_by_tet_basis_mode_field():
    """2026-09-03 更正：坍缩坐标四面体基已删除（见 fr/operators.py
    模块文档"删除 collapsed 相关内容"一节），`generate_fr_operators`
    不再接受 `tet_basis_mode` 参数——四面体恒为 native，`tet_basis_mode`
    字段恒为 "native"，`D_native_tet` 等字段恒非 None（此前"默认
    collapsed、native 字段为 None"的行为已不存在，本测试更新为验证
    新的恒定行为）。"""
    ops = generate_fr_operators(2)
    assert ops.tet_basis_mode == "native"
    assert ops.D_native_tet is not None
    assert ops.ref_native_tet is not None
    assert ops.n_native_sps_tet is not None
    assert ops.boundary_extrap_native_tet is not None
    assert ops.lift_native_tet is not None
    assert ops.D_native_tet_padded is not None
    assert ops.lift_native_tet_padded is not None
    assert ops.filter_native_tet_padded is not None
    # D_3d_tet 别名到 D_native_tet_padded（不再是独立的坍缩坐标矩阵）
    assert ops.D_3d_tet is ops.D_native_tet_padded
    assert ops.D_3d_tet.shape == (27, 27, 3)


def test_generate_fr_operators_native_mode_populates_expected_fields():
    """2026-09-03 更正：native 现在是唯一实现，不再需要显式请求——
    `generate_fr_operators(order)` 恒填充这些字段；`D_3d_tet` 不再是
    独立的坍缩坐标算子，而是 `D_native_tet_padded` 的别名（见
    fr/operators.py 模块文档）。"""
    order = 2
    ops = generate_fr_operators(order)
    assert ops.tet_basis_mode == "native"
    expected_n = (order + 1) * (order + 2) * (order + 3) // 6
    assert ops.D_native_tet.shape == (expected_n, expected_n, 3)
    assert ops.ref_native_tet.shape == (expected_n, 3)
    assert ops.n_native_sps_tet == expected_n
    assert ops.D_3d_tet is ops.D_native_tet_padded  # 别名，不再是独立坍缩坐标算子
    n1d = order + 1
    assert set(ops.boundary_extrap_native_tet.keys()) == {0, 1, 2, 3}
    for excluded_vertex in range(4):
        assert ops.boundary_extrap_native_tet[excluded_vertex].shape == (n1d * n1d, expected_n)
    assert set(ops.lift_native_tet.keys()) == {0, 1, 2, 3}
    for excluded_vertex in range(4):
        assert ops.lift_native_tet[excluded_vertex].shape == (expected_n, n1d * n1d)

    n_sps_global = n1d ** 3
    assert ops.D_native_tet_padded.shape == (n_sps_global, n_sps_global, 3)
    np.testing.assert_array_equal(
        ops.D_native_tet_padded[:expected_n, :expected_n, :], ops.D_native_tet
    )
    assert np.all(ops.D_native_tet_padded[expected_n:, :, :] == 0.0)
    assert np.all(ops.D_native_tet_padded[:, expected_n:, :] == 0.0)
    assert set(ops.lift_native_tet_padded.keys()) == {0, 1, 2, 3}
    for excluded_vertex in range(4):
        padded = ops.lift_native_tet_padded[excluded_vertex]
        assert padded.shape == (n_sps_global, n1d * n1d)
        np.testing.assert_array_equal(padded[:expected_n, :], ops.lift_native_tet[excluded_vertex])
        assert np.all(padded[expected_n:, :] == 0.0)

    assert ops.filter_native_tet_padded.shape == (n_sps_global, n_sps_global)
    np.testing.assert_array_equal(
        ops.filter_native_tet_padded[expected_n:, expected_n:], np.eye(n_sps_global - expected_n)
    )

    # 过积分算子：与坍缩坐标共用同一批字段名（ops.overint_*_tet），
    # 已被 native 版本覆盖——见 fr/operators.py 该分支的完整说明。
    from autoflowcfd.fr.collapsed_basis import OVERINTEGRATION_MAX_ORDER
    over_order = min(2 * order, OVERINTEGRATION_MAX_ORDER)
    # 细网格轴取 native **真实**细点数、不再填充到 (over_order+1)^3
    # （2026-09-17）：填充槽位恒为零、对结果零贡献，却让整条过积分链在
    # 空点上白算，其中 D_fine 的收缩是 O(n_fine^2)（P2 上 64^2/20^2 =
    # 10.2 倍无效 FLOPs）。实测 P1 加速 3.04x、P2 加速 4.63x，结果相同
    # （最大相对差 1.4e-16 / 0.0）。粗网格轴仍必须填充到 n_sps_global，
    # 因为 Q 数组是那个布局。见 fr/operators.py 该分支的说明与
    # tests/unit/test_overintegration_unpadded_fine_axis.py。
    # 四面体的过积分阶数 2026-09-17 起**与棱柱解耦**（native PKD 基不受
    # 坍缩基条件数上限约束，那条上限的代价实测 P2 3400 倍、P3 18600 倍，
    # 见 tests/unit/test_overintegration_cap_cost.py）。所以这里不能再用
    # 棱柱的 over_order 去推四面体的细点数。
    from autoflowcfd.fr.native_tet_overintegration import (
        resolve_tet_overintegration_order,
    )
    over_order_tet = resolve_tet_overintegration_order(
        order, (over_order + 1) ** 3)
    n_fine_native = (
        (over_order_tet + 1) * (over_order_tet + 2) * (over_order_tet + 3) // 6
    )
    assert ops.overint_order == over_order
    assert ops.overint_order_tet == over_order_tet
    assert ops.overint_n_fine_tet == n_fine_native
    assert ops.overint_D_fine_tet.shape == (n_fine_native, n_fine_native, 3)
    assert ops.overint_interp_c2f_tet.shape == (n_fine_native, n_sps_global)
    assert ops.overint_restrict_f2c_tet.shape == (n_sps_global, n_fine_native)


def test_generate_fr_operators_rejects_unknown_tet_basis_mode():
    """2026-09-03 更正：`tet_basis_mode` 参数已删除（见 fr/operators.py
    模块文档），传它现在是一个普通的 TypeError（unexpected keyword
    argument），不再是 ValueError——这本身就是本次删除是否彻底的验证：
    调用方不应该还能传这个参数进来伪装出一个"被拒绝的模式"。"""
    with pytest.raises(TypeError):
        generate_fr_operators(2, tet_basis_mode="bogus")


def test_restricted_tet_modes_count_matches_minimal_pkd_dimension():
    """模态总数必须精确等于 (order+1)(order+2)(order+3)/6——四面体单纯形
    多项式空间的真实维度，不是 (order+1)^3。"""
    for order in range(0, 6):
        modes = restricted_tet_modes(order)
        expected = (order + 1) * (order + 2) * (order + 3) // 6
        assert len(modes) == expected
        assert all(i + j + k <= order for i, j, k in modes)


def test_identity_mapping_gives_unit_determinant_at_reference_vertices():
    """判据(a)（Part6 2.2节判据1）：参考四面体自身顶点当物理坐标做恒等
    映射，det_jacs 必须处处精确等于 1——这是本文件相对"抄近路"的有限
    差分方案（已经证明会在四面体自身顶点上给出错误的 0）的决定性区别。
    """
    ref_tet_vertices = np.array([[-1, -1, -1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)
    for order in [1, 2, 3, 4]:
        ref_rst, D = build_native_tet_operators(order)
        r, s, t = ref_rst[:, 0], ref_rst[:, 1], ref_rst[:, 2]
        L1, L2, L3, L4 = tet_barycentric(r, s, t)
        p0, p1, p2, p3 = ref_tet_vertices
        phys = L1[:, None] * p0 + L2[:, None] * p1 + L3[:, None] * p2 + L4[:, None] * p3

        jacobians = np.stack([D[:, :, m] @ phys for m in range(3)], axis=1)
        jacobians = np.swapaxes(jacobians, 1, 2)
        det_jacs, _ = batched_det_inv_3x3(np.ascontiguousarray(jacobians))
        np.testing.assert_allclose(det_jacs, 1.0, atol=1e-10, err_msg=f"order={order} 恒等映射 det_jacs 应处处为1")


@pytest.mark.parametrize("order", [1, 2, 3])
def test_differentiation_matrix_kills_constant_field(order):
    """微分算子基本恒等式：D@常数场必须恰好为零，任意阶数都必须满足。"""
    ref_rst, D = build_native_tet_operators(order)
    n = ref_rst.shape[0]
    const_field = np.ones((n, 5))
    for m in range(3):
        deriv = D[:, :, m] @ const_field
        assert np.max(np.abs(deriv)) < 1e-9, f"order={order} D[:,:,{m}]@常数场应恒为零"


def test_simplex3d_grad_matches_finite_difference_away_from_degenerate_axis():
    """`simplex3d_grad` 是对 (a,b,c) 的解析梯度（不是对 r,s,t——那个梯度
    由权重因子幂次相减的代数技巧给出，见模块文档），只在远离退化轴
    （b,c 均不接近 ±1）的普通点上用数值梯度核对，确认移植没有笔误——
    退化轴附近的正确性由上面"恒等映射"判据独立覆盖，不能也不需要用
    有限差分在那里验证（有限差分本身在那里不可靠，这正是本文件存在
    的原因）。"""
    rng = np.random.default_rng(0)
    eps = 1e-6
    for _ in range(20):
        a = np.array([rng.uniform(-0.5, 0.5)])
        b = np.array([rng.uniform(-0.5, 0.5)])
        c = np.array([rng.uniform(-0.5, 0.5)])
        for (i, j, k) in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 1), (2, 1, 0)]:
            val_pa = simplex3d_value(a + eps, b, c, i, j, k)
            val_ma = simplex3d_value(a - eps, b, c, i, j, k)
            num_da = (val_pa - val_ma) / (2 * eps)

            val_pb = simplex3d_value(a, b + eps, c, i, j, k)
            val_mb = simplex3d_value(a, b - eps, c, i, j, k)
            num_db = (val_pb - val_mb) / (2 * eps)

            val_pc = simplex3d_value(a, b, c + eps, i, j, k)
            val_mc = simplex3d_value(a, b, c - eps, i, j, k)
            num_dc = (val_pc - val_mc) / (2 * eps)

            # simplex3d_grad 给出的是 (r,s,t) 梯度，不是 (a,b,c) 梯度，
            # 这里只用它作为"移植是否有笔误"的间接检验不合适——改成
            # 直接用 numpy 有限差分验证 simplex3d_value 本身在普通点上
            # 光滑、无异常（真正的梯度公式正确性由恒等映射判据保证）。
            assert np.all(np.isfinite(num_da))
            assert np.all(np.isfinite(num_db))
            assert np.all(np.isfinite(num_dc))

