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


class TestFilterNeverAmplifiesInTheRightNorm:
    """决定性判据：滤波不得**在正确的范数下**放大。

    ## 判据校准的两次更正（如实记录，都是被自己的数据逼出来的）

    第一版断言"谱范数 <= 1"，被数值证伪（order 2~4 的谱范数 1.58~2.58）。
    那不是实现 bug：滤波矩阵是 `V @ diag(sigma) @ V^{-1}` 的相似变换，
    `V`（节点↔模态）**不是正交阵**，所以即使 `sigma` 逐分量 <= 1，节点
    2-范数意义下的谱范数也可以 > 1。生产里已验证可用的坍缩坐标滤波器
    谱范数本身就是 5.78（被拒绝的总阶数归一化方案是 192.9）。

    第二版改成"随机白噪声的**节点标准差**中位比值 < 1"。2026-09-17 把
    `AFCFD_FILTER_MODE` 默认从 `legacy` 改到 `sensor`（矩阵是严格的
    `project`：sigma 只取 0 或 1）之后，这条判据在 order=3/4 上失败
    （中位比值 1.0818 / 1.1496）。核对发现**判据本身不对**：节点标准差
    不是投影必须收缩的范数，理由与第一版同一件事（V 不正交）。

    ## 正确的不变量（实测，2026-09-17）

    PKD 模态基在参考四面体上 L2 正交，所以模态系数的 2-范数**就是** L2
    范数。一个正交投影必须同时满足两条，实测：

        order   节点std比中位   模态L2比中位    |F@F-F|max   节点谱范数
          1        0.0000        0.70921574     0.00e+00      1.000
          2        0.6681        0.87301575     1.39e-16      1.581
          3        1.0818        0.93717037     3.33e-16      1.861
          4        1.1496        0.95413076     7.77e-16      2.581

    模态 L2 比值**每阶都 <= 1**、幂等误差**每阶都是机器零**——`project`
    档是严格的正交投影，从不放大。节点 std 在高阶超过 1 是 V 不正交的
    必然结果，与"是否放大"无关。

    所以本类的判据是：
      1. 模态 L2 范数不增（真正的"不放大"）；
      2. 幂等（这是"按单元门控降一阶"这套用法要求的算子语义，见
         `fr/modal_filter.py` 里 `filter_sigma` 的注释）；
      3. 节点谱范数仍钉一个量级上界（挡住"总阶数归一化"那类 192.9 的
         真实失败），但不再要求 <= 1。
    """

    #: 实测节点谱范数（2026-09-17，留约 1.3 倍余量）。上界不是 1——见类文档。
    _SPECTRAL = {1: 1.0, 2: 1.581, 3: 1.861, 4: 2.581}

    def _modal_l2_ratio(self, F, order, n_trials=120):
        from autoflowcfd.fr.native_simplex_basis import (
            build_native_tet_operators, restricted_tet_modes,
        )
        from autoflowcfd.fr.native_tet_overintegration import (
            _native_modal_vandermonde,
        )

        ref, _ = build_native_tet_operators(order)
        V = _native_modal_vandermonde(ref, restricted_tet_modes(order))
        rng = np.random.default_rng(order * 1000 + 7)
        out = np.empty(n_trials)
        for t in range(n_trials):
            u = rng.standard_normal(F.shape[0])
            cu = np.linalg.solve(V, u)
            cf = np.linalg.solve(V, F @ u)
            out[t] = np.linalg.norm(cf) / np.linalg.norm(cu)
        return out

    @pytest.mark.parametrize("order", [1, 2, 3, 4])
    def test_modal_l2_norm_never_grows(self, order):
        """正交投影的定义性质：L2 范数不增。**每一次**试验都要满足，
        不是中位数满足——一次放大就说明它不是投影。"""
        F = build_native_tet_modal_filter(order)
        r = self._modal_l2_ratio(F, order)
        assert r.max() <= 1.0 + 1e-12, (
            f"order={order}: 模态 L2 比值最大 {r.max():.12f} > 1，"
            f"滤波在 L2 意义下放大了——它不是正交投影")

    @pytest.mark.parametrize("order", [1, 2, 3, 4])
    def test_filter_is_idempotent(self, order):
        """幂等是"按单元门控降一阶"这套用法的前提：非幂等的话被标记的
        单元会被反复削、一路掉到 P0，退化后更容易再次越界，形成正反馈
        （legacy 档 P2/P3 的 |F@F-F| 是 6.3e-2 / 3.5e-1，正是这个病）。"""
        F = build_native_tet_modal_filter(order)
        err = np.abs(F @ F - F).max()
        assert err < 1e-13, f"order={order}: |F@F-F| = {err:.3e}，不幂等"

    @pytest.mark.parametrize("order", [1, 2, 3, 4])
    def test_nodal_spectral_norm_stays_in_measured_range(self, order):
        """节点谱范数**不要求 <= 1**（V 不正交，见类文档），但要钉住量级
        ——挡住"总阶数归一化"那类真实失败（实测谱范数 192.9）。"""
        F = build_native_tet_modal_filter(order)
        got = np.linalg.norm(F, ord=2)
        assert got <= 1.3 * self._SPECTRAL[order], (
            f"order={order}: 节点谱范数 {got:.4f} 超出实测值 "
            f"{self._SPECTRAL[order]} 的 1.3 倍")

    @pytest.mark.parametrize("order", [1, 2, 3, 4])
    def test_nodal_std_ratio_matches_measurement(self, order):
        """把"节点 std 在高阶确实会 > 1"这个事实钉住。

        它曾被当作失败判据（第二版），实际是 V 不正交的必然结果。钉住它
        是为了让"哪天它变了"成为一个需要解释的信号，而不是让人再一次
        误以为滤波器坏了。
        """
        F = build_native_tet_modal_filter(order)
        rng = np.random.default_rng(order * 1000 + 7)
        ratios = np.array([
            np.std(F @ (u := rng.standard_normal(F.shape[0]))) / np.std(u)
            for _ in range(200)])
        # 实测（本测试自己的种子 order*1000+7、200 次试验，2026-09-17）
        expected = {1: 0.0000, 2: 0.7118, 3: 1.0158, 4: 1.1225}[order]
        assert np.median(ratios) == pytest.approx(expected, abs=0.02), (
            f"order={order}: 节点 std 中位比值 {np.median(ratios):.4f}，"
            f"实测 {expected}")
