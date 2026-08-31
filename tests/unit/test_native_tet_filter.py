"""AutoFlowCFD V2.0 - native 四面体（路径C）指数模态滤波器
(`fr/native_tet_filter.py`) 决定性验证。

判据设计直接对应 `fr/modal_filter.py` 模块文档记录的真实教训——该
文档明确警告"总阶数 i+j+k 归一化在坍缩坐标四面体基上会*放大*白噪声"
（真实测得标准差放大 17.5 倍，谱范数 192.9），而 native 基改用总阶数
`(i+j+k)/order` 归一化是因为它是这套单纯形基的标准判据（不是照抄
坍缩坐标的判据，两套基数学结构不同）——因此本文件的
`TestWhiteNoiseIsDamped` 必须真的验证 native 版本*没有*重蹈坍缩坐标
"总阶数判据在错误的基上放大噪声"这个覆辙，不能只做"看起来合理"的
弱检查。
"""

import numpy as np
import pytest

from autoflowcfd.fr.native_tet_filter import build_native_tet_modal_filter
from autoflowcfd.fr.native_simplex_basis import (
    build_native_tet_operators, restricted_tet_modes, simplex3d_value, rst_to_abc,
)


class TestOrderZeroIsIdentity:
    def test_order_0_is_identity(self):
        F = build_native_tet_modal_filter(0)
        assert np.allclose(F, np.eye(1))


class TestConstantFieldPreserved:
    """常数场（eta=0 模态）必须严格不衰减——自由流场保持性的前提。"""

    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_constant_field_preserved(self, order):
        F = build_native_tet_modal_filter(order)
        n = F.shape[0]
        field = np.full(n, 42.0)
        np.testing.assert_allclose(F @ field, field, atol=1e-8)


class TestTopModeStronglyDamped:
    """最高阶模态（i+j+k==order）必须被压到机器精度量级——抑制混叠
    失稳的核心机制，与坍缩坐标版本同一个设计承诺。"""

    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_top_mode_damped(self, order):
        modes = restricted_tet_modes(order)
        top_indices = [m for m, (i, j, k) in enumerate(modes) if i + j + k == order]
        assert len(top_indices) > 0

        ref_rst, _ = build_native_tet_operators(order)
        a, b, c = rst_to_abc(ref_rst[:, 0], ref_rst[:, 1], ref_rst[:, 2])
        V = np.column_stack([simplex3d_value(a, b, c, i, j, k) for (i, j, k) in modes])

        F = build_native_tet_modal_filter(order)
        for m in top_indices:
            modal_coeffs = np.zeros(len(modes))
            modal_coeffs[m] = 1.0
            nodal_field = V @ modal_coeffs
            filtered = F @ nodal_field
            assert np.max(np.abs(filtered)) < 1e-8 * np.max(np.abs(nodal_field)), (
                f"order={order}, mode index {m} (degree {order}) 未被充分压制"
            )


class TestWhiteNoiseIsNotAmplified:
    """决定性判据（对应 modal_filter.py 文档记录的真实反例）：对 native
    体积节点上的随机白噪声施加滤波，结果的标准差/谱范数都不能*放大*
    ——如果放大，说明 native 基上"总阶数"归一化重蹈了坍缩坐标那次
    "总阶数在错误的基上放大噪声"的覆辙,不能只凭"这是标准判据"就假设
    它对这套具体基一定生效。
    """

    @pytest.mark.parametrize("order", [1, 2, 3, 4])
    def test_white_noise_std_and_spectral_norm_not_amplified(self, order):
        """判据校准说明（第一版曾用错阈值，如实记录）：最初断言"谱范数
        必须 <=1"，被真实数值证伪（order=2~4 谱范数 1.58~2.19）——但
        进一步核对发现，`modal_filter.py` 模块文档记录的、**已经在生产
        代码里使用、经过 cube_demo/TGV/Couette 真实验证过的**坍缩坐标
        `max(i,j,k)` 归一化滤波器本身谱范数就是 5.78（远大于 1；那份
        文档同时记录了被拒绝的总阶数归一化方案谱范数高达 192.9，作为
        真正的"失败"对照）——"谱范数 <=1" 从来不是这类相似变换滤波器
        的正确判据（该矩阵是 `V@diag(sigma)@V^{-1}` 形式的相似变换，
        V 不是正交阵时，即使 sigma 逐分量都 <=1，谱范数也可能 >1，这是
        线性代数的一般事实，不是实现 bug）。改为对照生产已接受的量级
        （<=10，比坍缩坐标已验证可用的 5.78 留有余量，远低于被拒绝方案
        的 192.9）+ 随机白噪声中位数衰减比（更直接、更贴近实际使用场景
        的判据）两者共同判断。
        """
        F = build_native_tet_modal_filter(order)
        n = F.shape[0]

        spectral_norm = np.linalg.norm(F, ord=2)
        assert spectral_norm <= 10.0, (
            f"order={order}: native 模态滤波器谱范数 {spectral_norm:.4f} 超出"
            "生产已验证坍缩坐标滤波器（5.78）的合理量级（阈值10），可能"
            "重蹈'总阶数判据放大噪声'（被拒绝方案谱范数192.9）的覆辙。"
        )

        rng = np.random.default_rng(order * 1000 + 7)
        n_trials = 200
        std_ratios = np.empty(n_trials)
        for t in range(n_trials):
            noise = rng.standard_normal(n)
            filtered = F @ noise
            std_ratios[t] = np.std(filtered) / np.std(noise)
        assert np.median(std_ratios) < 1.0, (
            f"order={order}: 随机白噪声滤波后标准差中位数比值 {np.median(std_ratios):.4f} "
            ">= 1，说明滤波器对随机噪声没有起到压制作用（甚至放大），不满足"
            "抑制混叠失稳的设计目的。"
        )
