"""模态滤波器"每个阶数损失一整阶"这一事实的回归测试（2026-09-15）。

## 发现

`fr/modal_filter.py` 的注释写的设计意图是"只有最高一两阶模态被**显著
压制**，不牺牲已解析到的真实物理精度"。实现与意图不符：

    FILTER_ALPHA = -log(eps) ≈ 36.04,  sigma(eta) = exp(-alpha * eta^8)
    eta = max(i,j,k) / order
    => sigma(eta=1) = exp(-36.04) = 2.2e-16   ← 清零，不是衰减

而 `max(i,j,k)` 判据下 eta=1 恰好覆盖该阶**全部新增模态**，于是：

    order=1 -> 保留 1/8   模态（只剩常数）      => P1 实际是 P0
    order=2 -> 保留 8/27  模态（只剩双线性）    => P2 实际是 P1
    order=3 -> 保留 27/64 模态                  => P3 实际是 P2

即保留集恰好是 {i,j,k <= order-1}，大小 order^3 —— **每个阶数精确损失
一整阶**。alpha=-ln(eps) 本身是 Hesthaven & Warburton 的标准取值，但那是
为**高阶**设计的（P8 上清掉第 8 阶无关紧要）；本项目只跑 P1/P2，正好落在
这个取值最糟的区间。

真实印证（79 万单元 cube_demo，P1，iter=300 检查点）：胞内 |grad u| 相对
参照剪切率 U/h 只有 7.7e-16（机器零），粘性残差几乎只来自边界 IP 罚项。

## 本文件的作用

把这些**可验证的事实**钉住，而不是只写在注释里：任何人调 alpha、换
归一化判据、或改 FILTER_ORDER，都会在这里被迫正面处理"损失多少阶"这个
问题，而不是让它继续静默存在。

同时钉住三档可切换模式（`AFCFD_FILTER_MODE`）的语义——它们是受控 A/B
与现场排查的入口，必须保证 off/mild 真的不损失阶数。
"""

import numpy as np
import pytest

from tests.unit._modal_filter_order_loss_common import (
    _expected_legacy_rank,
    _prism_filter,
    _real_block,
    _reload_with_env,
)
from tests.unit._modal_filter_order_loss_common import _restore_modules  # noqa: F401  autouse 夹具：进入本模块命名空间才生效


class TestLegacyFilterLosesExactlyOneOrder:
    """**legacy 档**：保留模态数恰好等于"少一阶"的模态数，即损失一整阶。

    2026-09-20：判据从"秩 == order^3"改成"秩 == 少一阶的模态数"
    （`_expected_legacy_rank`）。`order^3` 是**坍缩棱柱基**下那个数的
    具体取值；默认基改成 native 之后模态集不同（还多了零填充槽位的单位
    行），但**结论不变** —— 两条基都是精确损失一整阶。写成公式而不是
    写死数字，才是这条结论本身的判据。

    2026-09-18 更正：本类此前全部用 `AFCFD_FILTER_MODE=None`（**默认档**）
    —— 类名写着 legacy，测的却是默认值。历史上默认档恰好也把顶模态清零
    （legacy -> sensor 时 sensor 的矩阵是精确投影），所以一直通过；直到
    默认档的 sensor 改成顶模态 sigma=0.99 的**有界衰减**（见
    `fr/modal_filter.py::filter_sigma` 里"为什么 sensor 从精确投影改成
    有界衰减"一节）才暴露出来。现在显式指定 legacy。
    """

    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_rank_is_exactly_one_order_lower(self, order):
        """保留秩恰好等于"少一阶"的模态数（两条棱柱基各自的形式见
        `_expected_legacy_rank`）。"""
        mf, ops_mod = _reload_with_env(AFCFD_FILTER_MODE="legacy")
        F = _prism_filter(ops_mod, order)
        n = (order + 1) ** 3
        assert F.shape == (n, n)
        expected_rank = _expected_legacy_rank(order)
        rank = int(np.linalg.matrix_rank(F, 1e-10))
        assert rank == expected_rank, (
            f"order={order}: 滤波矩阵秩={rank}，期望 {expected_rank}。"
            f"秩变化意味着'损失几阶'变了——这是精度层面的行为改变，"
            f"必须显式确认而不是顺带改掉")

    def test_top_mode_sigma_is_annihilation_not_damping(self):
        mf, _ = _reload_with_env(AFCFD_FILTER_MODE="legacy")
        sigma_top = float(mf._exp_filter_sigma(np.array(1.0)))
        assert sigma_top < 1e-12, (
            f"sigma(eta=1)={sigma_top:.3e}，不再是'清零'量级——"
            f"若这是有意的放宽，请同步更新本文件与 modal_filter.py 的说明")
        # 常数模态必须完全保留（自由流场保持性）
        assert float(mf._exp_filter_sigma(np.array(0.0))) == 1.0

    def test_p1_annihilates_linear_intra_cell_content(self):
        """order=1 时线性场的**胞内**变化被抹到机器零——P1 退化为 P0
        最直接的证据（不依赖任何真实网格）。"""
        mf, ops_mod = _reload_with_env(AFCFD_FILTER_MODE="legacy")
        F = _real_block(_prism_filter(ops_mod, 1), 1)
        # order=1 的节点空间里，任何非常数向量都只能由常数模态 + 那些
        # 到顶的模态张成，所以用随机向量减去其常数部分就是"纯胞内变化"，
        # 不需要显式构造线性场坐标。
        # **只取真实自由度子块**（2026-09-20）：零填充槽位的滤波行是
        # 单位阵（`fr/native_padding.py` 的约定），把它们算进来等于在问
        # "单位阵有没有清零"，与本条要钉的性质无关。
        rng = np.random.default_rng(3)
        for _ in range(5):
            f = rng.standard_normal(F.shape[1])
            f_const = np.full_like(f, f.mean())
            var0 = np.abs(f - f_const).max()
            g = F @ f
            var1 = np.abs(g - np.full_like(g, g.mean())).max()
            assert var1 / max(var0, 1e-300) < 1e-12, (
                f"order=1 滤波后仍保留了 {var1/var0:.3e} 的非常数内容——"
                f"若这是有意的修复，请更新本用例")

    def test_constant_field_preserved_at_all_orders(self):
        """不管损失几阶，常数场必须逐位保留（自由流场保持性的前提）。"""
        mf, ops_mod = _reload_with_env(AFCFD_FILTER_MODE=None)
        for order in (1, 2, 3):
            F = _prism_filter(ops_mod, order)
            one = np.ones(F.shape[1])
            assert np.abs(F @ one - 1.0).max() < 1e-13


class TestSwitchableModesDoNotLoseOrder:
    """`AFCFD_FILTER_MODE=off/mild` 必须真的不损失阶数——它们是受控 A/B
    与现场排查的入口，如果也在清零就失去了意义。"""

    @pytest.mark.parametrize("order", [1, 2])
    def test_off_is_identity(self, order):
        mf, ops_mod = _reload_with_env(AFCFD_FILTER_MODE="off")
        F = _prism_filter(ops_mod, order)
        np.testing.assert_allclose(F, np.eye(F.shape[0]), rtol=0, atol=0)

    @pytest.mark.parametrize("order", [1, 2])
    def test_mild_keeps_full_rank(self, order):
        mf, ops_mod = _reload_with_env(AFCFD_FILTER_MODE="mild",
                                       AFCFD_FILTER_SIGMA_TOP="0.99")
        F = _prism_filter(ops_mod, order)
        n = (order + 1) ** 3
        rank = int(np.linalg.matrix_rank(F, 1e-10))
        assert rank == n, f"mild 档 order={order} 秩={rank}，应为满秩 {n}"

    def test_mild_sigma_top_matches_requested(self):
        for want in ("0.99", "0.9", "0.5"):
            mf, _ = _reload_with_env(AFCFD_FILTER_MODE="mild",
                                     AFCFD_FILTER_SIGMA_TOP=want)
            got = float(mf._exp_filter_sigma(np.array(1.0)))
            assert got == pytest.approx(float(want), rel=1e-12), (
                f"请求 sigma_top={want}，实得 {got}")

    def test_default_mode_is_off(self):
        """默认 `off`（2026-09-19 从 `sensor` 改）。

        ## 为什么此前那轮标定整体失效

        2026-09-19 在已上线的原生四面体路径上查出并修掉两个重大缺陷
        （见 `core/fr_operators/face_kernels.py::FlatFaceGeometry.
        ref_area_weight` 与 `core/fr_operators/troubled_cell.py::
        _outlier_ref_and_flag_kernel` 的文档）：DG 提升的界面项多乘了一个
        `|adj_row| ~ h^2`（上风耗散被系统性压制），以及 P2/P3 的四面体
        残差被机制3 **整体清零**（单元完全不演化）。此前"必须靠滤波才
        稳定"的印象是在那个前提下形成的。

        ## 修完之后用有解析解的算例重新判定

            档       Blasius cf 中位      TGV 动能（解析耗散率判据）
            off      +9.52%               通过
            sensor   +9.52%（与 off 逐位相同）  失败：能量净增长 +5.14%
            project  -87.69%              通过
            legacy   --                   失败：过耗散 7.6 倍

        `sensor` 在 Blasius 上与 `off` **逐位相同**（它实质上什么都没做），
        但在 TGV 上 BJ 判据对欠分辨光滑场 100% 标记、退化成"全局每 stage
        施加 mild 非幂等衰减"，450 次累积出非物理的能量增长。
        `project` 在 P1 上等于把被标记单元拍平成 P0，壁面剪应力塌 88%。

        完整依据见 `fr/modal_filter.py` 里"默认值 2026-09-19 改为 off"那节。
        """
        mf, ops_mod = _reload_with_env(AFCFD_FILTER_MODE=None)
        assert mf.FILTER_MODE == "off"

    def test_sensor_mode_matrix_is_bounded_damping_not_projection(self):
        """**2026-09-18 更正**：`sensor` 的矩阵是**有界衰减**，不是投影。

        此前这里断言"sensor 的矩阵与 project 相同：严格投影，P1 的秩仍是
        1"。那条设计被 Blasius 平板（本项目唯一有精确解的粘性算例）的
        实测否掉：精确投影把被标记单元一次清零，而单元内扩散重建壁面
        梯度约需 `h^2/nu/dt ~ 26` 步，0.83%/stage 的贴壁命中率意味着平均
        每 40 步就再清一次 —— 壁面剪应力被压掉 2.3 倍（du/dy 中位 561.9
        vs `off` 的 1312.7，cf -74.54% vs -6.33%）。换成顶模态
        sigma=0.99 的有界衰减后 cf 与 `off` **完全一致**（du/dy 1312.68），
        而滤波仍在工作（贴壁层每 stage 仍标记 3~7.5%）。
        完整推导与数据见 `fr/modal_filter.py::filter_sigma`。

        判据：矩阵**满秩**（不丢任何模态）、顶模态 sigma 恰好等于
        `AFCFD_FILTER_SIGMA_TOP`、常数模态严格为 1。
        """
        mf, ops_mod = _reload_with_env(AFCFD_FILTER_MODE="sensor")
        F1 = _prism_filter(ops_mod, 1)
        assert int(np.linalg.matrix_rank(F1, 1e-10)) == F1.shape[0], (
            "sensor 档的矩阵必须满秩——投影型会丢掉一整阶")
        sigma_top = float(mf.filter_sigma(np.array(1.0)))
        assert abs(sigma_top - 0.99) < 1e-12, (
            f"顶模态 sigma={sigma_top:.6g}，应当等于默认 "
            f"AFCFD_FILTER_SIGMA_TOP=0.99")
        assert float(mf.filter_sigma(np.array(0.0))) == 1.0, "常数模态必须严格保留"

    def test_sensor_mode_intermediate_modes_are_essentially_untouched(self):
        """有界衰减必须**只动顶模态**。

        这是"非幂等没关系"那条论证的前提：`mild` 的 alpha 下
        sigma(eta)=exp(-0.01005*eta^8)，P2/P3 的中间模态都在 1 的 1e-3
        以内。legacy 的 alpha 完全不同（P2 中间模态 0.8687、P3 的
        0.2450），那才是"会把中间模态一起反复削掉"的来源——若哪天
        默认 alpha 变大到动了中间模态，这条会失败，届时必须重新评估
        "门控 + 非幂等算子"的正当性而不是放宽本判据。
        """
        mf, _ = _reload_with_env(AFCFD_FILTER_MODE="sensor")
        for etas in ([0.0, 0.5], [0.0, 1 / 3, 2 / 3]):
            sig = np.asarray(mf.filter_sigma(np.array(etas)))
            assert np.all(sig > 1.0 - 1e-3), (
                f"中间模态 sigma={sig} 偏离 1 超过 1e-3")

    def test_legacy_stays_available_for_regression(self):
        """`legacy` 是唯一能复现历史结果的档，必须保留为合法取值。"""
        mf, ops_mod = _reload_with_env(AFCFD_FILTER_MODE="legacy")
        assert mf.FILTER_MODE == "legacy"
