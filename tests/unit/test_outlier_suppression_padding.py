"""机制3（`suppress_residual_outliers`）与原生基零填充槽位的相互作用。

## 这个文件钉住的是一个真实的、严重的生产缺陷（2026-09-18 发现并修复）

机制3 按"同单元其余 SP 的中位数"建立异常判据的参照量：

    ref[c,v] = max( median_s |residual[c,s,v]|,
                    1e-9 * mean_s |U[c,s,v]|,
                    1e-300 )
    |residual| > 1e4 * ref  ->  判为异常、清零

而原生基（四面体、以及 2026-09-18 起可选的原生棱柱）的**零填充槽位残差
恒为零**（那是"零填充块对角"不变量刻意保证的：填充槽位不被时间推进改写）。
于是中位数被那些零拖下去：

    order   n_sps   真实槽位   零填充   中位数落点
      P1      8        4         4      sorted[3],sorted[4] 平均 = 最小真实值/2  > 0
      P2     27       10        17      sorted[13] 落在零区                      = 0
      P3     64       20        44      sorted[32] 落在零区                      = 0

**后果**：P2/P3 上参照量塌到 `1e-9 * mean|U|` 这个地板（对密度约 1.2e-9），
阈值 `1e4 * 1.2e-9 = 1.2e-5`，而真实残差是 1e5 量级 —— **整个四面体单元的
残差被全部清零，单元完全不演化**。P1 侥幸逃过（中位数非零），所以 P1 的
生产运行看起来正常，掩盖了这条缺陷。

这也是"native 解决了 P2 灾难性发散"这个既有结论必须重新审视的原因：
残差被清零的单元当然不会发散。

## 修法

中位数与均值都只在**真实槽位**上统计，判据也只作用在真实槽位 ——
`fr/native_padding.py::real_sps_per_cell` 就是"哪些槽位是真的"的唯一
判据来源，机制3 这个调用点此前漏掉了它。
"""

import numpy as np
import pytest

from autoflowcfd.core.fr_operators.troubled_cell import (
    RESIDUAL_OUTLIER_FACTOR,
    suppress_residual_outliers,
)
from autoflowcfd.fr.native_padding import native_tet_n_real_sps


def _case(order, n_prism=1, n_tet=1, res_scale=1.0e5):
    """构造一个 (n_prism 棱柱 + n_tet 四面体) 的残差/场对。

    四面体的填充槽位残差恒为零（生产不变量），真实槽位给一个**均匀**的
    量级 —— 均匀是关键：机制3 的设计意图是抓"同单元里个别 SP 凸出"，
    均匀的真实残差本来就不该被判为异常。
    """
    n_sps = (order + 1) ** 3
    n_real_tet = native_tet_n_real_sps(order)
    n_cells = n_prism + n_tet
    res = np.zeros((n_cells, n_sps, 5))
    fld = np.zeros((n_cells, n_sps, 5))
    fld[:, :, 0] = 1.225
    fld[:, :, 1] = 1.225 * 30.0
    fld[:, :, 4] = 2.5e5
    # 棱柱：全槽位都是真实自由度（坍缩基）
    res[:n_prism, :, :] = res_scale
    # 四面体：只有前 n_real_tet 个是真实的
    res[n_prism:, :n_real_tet, :] = res_scale
    return res, fld, n_real_tet


class TestPaddingDoesNotZeroRealResiduals:
    """**核心判据**：均匀的真实残差一个都不能被清零。"""

    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_uniform_real_residual_survives(self, order):
        res, fld, n_real = _case(order)
        out = suppress_residual_outliers(res, fld, n_prism=1)
        tet_real = out[1, :n_real]
        assert np.all(tet_real != 0.0), (
            f"order={order}: 四面体真实槽位被清零了 "
            f"{int(np.sum(tet_real == 0.0))}/{tet_real.size} 个 —— "
            f"零填充槽位把中位数拖到了 0")
        np.testing.assert_allclose(tet_real, res[1, :n_real])

    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_padding_slots_stay_zero(self, order):
        res, fld, n_real = _case(order)
        out = suppress_residual_outliers(res, fld, n_prism=1)
        n_sps = (order + 1) ** 3
        if n_real < n_sps:
            assert np.all(out[1, n_real:] == 0.0)

    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_prism_cells_unaffected(self, order):
        """棱柱段（坍缩基、无填充）必须逐位不变。"""
        res, fld, _n = _case(order)
        out = suppress_residual_outliers(res, fld, n_prism=1)
        np.testing.assert_array_equal(out[0], res[0])

    def test_p2_is_the_regression_case(self):
        """P2 是这条缺陷最直接的复现点（17/27 槽位是零 -> 中位数取到零区）。

        修复前这里的 `out[1, :10]` 全是 0。
        """
        res, fld, n_real = _case(2)
        assert n_real == 10
        out = suppress_residual_outliers(res, fld, n_prism=1)
        assert float(np.abs(out[1, :n_real]).max()) > 0.0


class TestRealOutliersAreStillCaught:
    """修掉填充干扰**不能**让机制3 失效 —— 否则是用一个缺陷换另一个。"""

    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_single_spiky_sp_in_a_tet_is_zeroed(self, order):
        res, fld, n_real = _case(order)
        # 在真实槽位里塞一个远超同单元中位数的尖峰
        res[1, 0, 0] = res[1, 1, 0] * RESIDUAL_OUTLIER_FACTOR * 10.0
        out = suppress_residual_outliers(res, fld, n_prism=1)
        assert out[1, 0, 0] == 0.0, "真实的单点异常没有被抓住"
        # 同单元其余真实槽位不受牵连
        assert np.all(out[1, 1:n_real, 0] != 0.0)

    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_spiky_sp_in_a_prism_is_zeroed(self, order):
        res, fld, _n = _case(order)
        res[0, 0, 0] = res[0, 1, 0] * RESIDUAL_OUTLIER_FACTOR * 10.0
        out = suppress_residual_outliers(res, fld, n_prism=1)
        assert out[0, 0, 0] == 0.0


class TestScalarFieldsAndShapes:
    def test_single_variable_arrays_work(self):
        """湍流输运那几个调用点传的是 `(n_cells, n_sps, 1)`。"""
        order = 2
        n_sps = (order + 1) ** 3
        n_real = native_tet_n_real_sps(order)
        res = np.zeros((2, n_sps, 1))
        fld = np.full((2, n_sps, 1), 0.5)
        res[0, :, 0] = 3.0
        res[1, :n_real, 0] = 3.0
        out = suppress_residual_outliers(res, fld, n_prism=1)
        assert np.all(out[1, :n_real, 0] != 0.0)

    def test_n_prism_equal_to_n_cells_is_pure_collapsed(self):
        order = 2
        n_sps = (order + 1) ** 3
        res = np.full((3, n_sps, 5), 7.0)
        fld = np.full((3, n_sps, 5), 1.0)
        out = suppress_residual_outliers(res, fld, n_prism=3)
        np.testing.assert_array_equal(out, res)

    def test_n_prism_out_of_range_raises(self):
        order = 1
        n_sps = (order + 1) ** 3
        res = np.zeros((2, n_sps, 5))
        fld = np.ones((2, n_sps, 5))
        with pytest.raises(ValueError, match="n_prism"):
            suppress_residual_outliers(res, fld, n_prism=3)
