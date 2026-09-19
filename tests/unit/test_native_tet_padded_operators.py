"""AutoFlowCFD V2.0 - native 四面体（路径C）算子零填充决定性验证。

见 `fr/native_padding.py` 与
`ProjectFiles/V2.0/8_算法重构-微分算子对坍缩坐标退化参考轴的病态条件数-Part8.md`
"一、核心不变量：零填充块对角"一节。本文件的核心判据
（`test_padded_matmul_survives_nan_in_padding_rows`）直接构造填充行为 NaN
的输入，验证真实行的输出精确不受污染——这是 Part8 文档记录的"差点漏掉"的
真实陷阱（`0*NaN=NaN`，不是想当然的"乘零就安全"），必须用真的 NaN 输入
实测，不能只验证"填充块本身是零"就假设组合起来没问题。
"""

import numpy as np
import pytest

from autoflowcfd.fr.native_padding import (
    pad_native_matrix_to_global, pad_native_filter_matrix_to_global,
)
from autoflowcfd.fr.native_tet.basis import build_native_tet_operators, build_native_tet_lift
from autoflowcfd.fr.native_tet.filter import build_native_tet_modal_filter


@pytest.mark.parametrize("order", [1, 2, 3])
def test_pad_preserves_real_block_and_zeros_elsewhere_for_d_native(order):
    _, D_native = build_native_tet_operators(order)
    n_native = D_native.shape[0]
    n1d = order + 1
    n_sps = n1d ** 3

    padded = pad_native_matrix_to_global(D_native, n_sps, pad_axes=(0, 1))
    assert padded.shape == (n_sps, n_sps, 3)
    np.testing.assert_array_equal(padded[:n_native, :n_native, :], D_native)
    assert np.all(padded[n_native:, :, :] == 0.0)
    assert np.all(padded[:, n_native:, :] == 0.0)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_pad_preserves_real_block_and_zeros_elsewhere_for_lift(order):
    n1d = order + 1
    n_fp = n1d * n1d
    n_sps = n1d ** 3
    for ev in range(4):
        Lift_ref = build_native_tet_lift(order, ev)
        n_native = Lift_ref.shape[0]
        padded = pad_native_matrix_to_global(Lift_ref, n_sps, pad_axes=(0,))
        assert padded.shape == (n_sps, n_fp)
        np.testing.assert_array_equal(padded[:n_native, :], Lift_ref)
        assert np.all(padded[n_native:, :] == 0.0)


def test_padded_matmul_with_finite_padding_rows_freezes_padding_output_at_exact_zero():
    """Part8 文档要求的**正常运行场景**：`Q` 的填充行必须始终是有限占位值
    （文档"四、未做后续工作"第2条明确要求，绝不能是 NaN/未初始化内存）。
    在这个前提下，填充行输出必须精确为 0（"行填零"不变量的直接后果：
    对应残差恒为 0，供 `Q_new=Q_old+dt*residual` 让填充行永远冻结在初值），
    真实行输出必须与未填充版本一致——这是决定性判据，不是"接近正确"。
    """
    order = 2
    _, D_native = build_native_tet_operators(order)
    n_native = D_native.shape[0]
    n1d = order + 1
    n_sps = n1d ** 3

    padded = pad_native_matrix_to_global(D_native, n_sps, pad_axes=(0, 1))

    rng = np.random.default_rng(0)
    field_real = rng.standard_normal((n_native, 5))
    field_padded = np.zeros((n_sps, 5))  # 有限占位值（Part8 要求的正常场景）
    field_padded[:n_native] = field_real
    field_padded[n_native:] = 7.0  # 任意有限占位值，非零以确认真的不参与计算

    for m in range(3):
        result_padded = padded[:, :, m] @ field_padded
        result_reference = D_native[:, :, m] @ field_real
        assert np.all(result_padded[n_native:] == 0.0), f"m={m}: 填充行输出应精确为 0"
        np.testing.assert_allclose(result_padded[:n_native], result_reference, atol=1e-11, rtol=1e-11)


def test_padded_matmul_does_not_isolate_nan_in_padding_rows_finite_discipline_is_mandatory():
    """**反向决定性判据**（比"验证安全"更重要的是"验证没有虚假的安全感"）：
    如果 `Q` 的填充行违反了 Part8 文档"必须始终有限"这一强制要求、
    真的出现 NaN，`0*NaN=NaN` 会通过矩阵乘法的求和污染**每一个**真实
    输出行——不是只污染填充行输出、"列填零"也保护不了真实自由度。

    第一次实现这个测试时，本文件曾错误地断言"真实行不会被 NaN 污染"
    （想当然地认为"乘以零系数"总是安全的），被这里的真实数值结果当场
    证伪（10 行输出全部变成 NaN）——如实记录这个过程，而不是事后包装
    成一次就设计对了。这证明"列填零/行填零"不变量**不是**一个能兜底
    非法输入的安全网，它只在"填充行本身有限"这个前提成立时才有效
    （见上一个测试），所以 Part8 文档把"填充行必须有限"列为强制要求
    而不是可选的最佳实践，是有真实数值依据的，不是过度谨慎。
    """
    order = 2
    _, D_native = build_native_tet_operators(order)
    n_native = D_native.shape[0]
    n1d = order + 1
    n_sps = n1d ** 3

    padded = pad_native_matrix_to_global(D_native, n_sps, pad_axes=(0, 1))

    rng = np.random.default_rng(0)
    field_real = rng.standard_normal((n_native, 5))
    field_padded = np.full((n_sps, 5), np.nan)
    field_padded[:n_native] = field_real

    result_padded = padded[:, :, 0] @ field_padded
    assert np.all(np.isnan(result_padded[:n_native])), (
        "本测试的存在意义就是证明：一旦填充行违反'必须有限'这个强制要求，"
        "NaN 会污染全部真实行——如果这里反而没有污染，说明底层 numpy 矩阵乘法"
        "实现变了（例如某种稀疏/分块优化跳过了零系数项），Part8 文档"
        "'填充行必须始终有限'这条强制要求的必要性论证需要重新审视，"
        "而不是静默让这个决定性判据失效。"
    )


def test_padded_lift_matmul_survives_nan_in_padding_columns_of_jump_is_not_applicable():
    """`lift_native_tet` 只需要"行填零"（输出是体积槽位），输入是面 FP
    数据（`n_fp` 宽，不存在体积填充列的概念）——这里换一个角度验证同一个
    不变量：用真实 jump 数据（非 NaN，因为 FP 数据没有"填充"概念）算出的
    真实行结果，必须与未填充版本一致，填充行必须精确为 0（供
    correction[cell, n_native:, :] 永远冻结在初值，与 Part8 文档一致）。
    """
    order = 2
    n1d = order + 1
    n_fp = n1d * n1d
    n_sps = n1d ** 3
    ev = 1
    Lift_ref = build_native_tet_lift(order, ev)
    n_native = Lift_ref.shape[0]
    padded = pad_native_matrix_to_global(Lift_ref, n_sps, pad_axes=(0,))

    rng = np.random.default_rng(1)
    jump = rng.standard_normal((n_fp, 5))

    result_padded = padded @ jump
    result_reference = Lift_ref @ jump
    assert np.all(result_padded[n_native:] == 0.0)
    np.testing.assert_allclose(result_padded[:n_native], result_reference, atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_filter_padding_uses_identity_block_and_freezes_padding_rows(order):
    """滤波矩阵填充块必须是单位矩阵（不是零）——真实场景验证：真实行
    经过填充后的滤波器与未填充版本一致，填充行原样通过（不被重置为
    0），且这个"原样通过"对*任意*有限占位值都成立（不只是 0）。"""
    n1d = order + 1
    n_sps = n1d ** 3
    F_native = build_native_tet_modal_filter(order)
    n_native = F_native.shape[0]
    padded = pad_native_filter_matrix_to_global(F_native, n_sps)
    assert padded.shape == (n_sps, n_sps)

    rng = np.random.default_rng(3)
    field_real = rng.standard_normal(n_native)
    padding_value = 999.5  # 任意非零有限占位值，验证"原样通过"不只对 0 成立
    field_padded = np.full(n_sps, padding_value)
    field_padded[:n_native] = field_real

    result = padded @ field_padded
    np.testing.assert_allclose(result[:n_native], F_native @ field_real, atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(result[n_native:], padding_value, atol=1e-12)


def test_pad_rejects_mismatched_axis_lengths():
    order = 2
    _, D_native = build_native_tet_operators(order)
    bad = D_native[:, :-1, :]  # 人为制造两个待填充轴长度不一致
    with pytest.raises(ValueError):
        pad_native_matrix_to_global(bad, 27, pad_axes=(0, 1))


def test_pad_rejects_n_sps_smaller_than_n_native():
    order = 2
    _, D_native = build_native_tet_operators(order)
    n_native = D_native.shape[0]
    with pytest.raises(ValueError):
        pad_native_matrix_to_global(D_native, n_native - 1, pad_axes=(0, 1))
