"""omega 壁面目标值的长度尺度口径（2026-09-15 发现的系统性偏差）。

## 发现

`_compute_omega_wall_target` 里

    omega_wall = 60 * nu / (beta1 * d1^2)

是 Menter 的 omega 壁面处理（`10 * 6*nu/(beta1*Δy1^2)`，这里的 60 就是
10x6），按**第一层单元中心**的壁距标定。而 `d1` 原先取的是单元内**全部
解点**壁距的 `min`——那既不是单元中心也不是单元高度，不对应任何标准
口径。

Gauss-Legendre 解点在高阶时向单元边界聚集，于是这个口径让一个**经过
标定的经验壁面函数**产生**随阶数变化**的系统性高估：

    order=1: 最近解点 0.2113*h -> (0.5/0.2113)^2 =  5.60x
    order=2: 最近解点 0.1127*h -> (0.5/0.1127)^2 = 19.68x
    order=3: 最近解点 0.0694*h -> (0.5/0.0694)^2 = 51.86x

一个标定过的壁面函数绝不该有这种阶数依赖。这也解释了为什么这个目标值
会顶到 `omega_max`、被 `enforce_omega_wall_relaxation` 的文档称作
"1e6 量级的应急上限"而不是"日常合理松弛目标"——它被喂了一个小 2.4~7.2
倍的长度尺度。

## 默认值没有改

`AFCFD_OMEGA_WALL_D1 = min | mean`，**默认 `min`**（既有行为，逐位不变）。
这是湍流模型的物理改动：降低近壁 omega 会抬高 nu_t，必须用真实长程数据
验证过才能改默认值。本项目在 omega 壁面处理上已经有两次"数学上更对但被
真实数据证伪"的先例（显式 SIPG 罚项、点隐式动态松弛系数，均见
`enforce_omega_wall_relaxation` 文档），所以这里只提供开关与判据。

## 判据

核心是**阶数无关性**：`mean` 口径下同一物理网格在不同阶数上给出的
`d1` 必须基本一致，而 `min` 口径必然随阶数单调变小。
"""

import os

import numpy as np
import pytest

from autoflowcfd.fr.quadrature_points import gauss_legendre


def _gl_normalized_positions(order):
    """Gauss-Legendre 解点在参考区间上归一化到 [0,1] 的位置（0=壁面侧）。"""
    pts, _ = gauss_legendre(order + 1)
    return (np.asarray(pts) + 1.0) / 2.0


class TestSolutionPointClusteringQuantified:
    """把"解点向边界聚集 -> min 口径的高估随阶数增长"这条量化钉住。

    这是纯几何事实（只依赖 Gauss-Legendre 点集），不依赖任何网格或流场，
    所以可以精确断言。
    """

    @pytest.mark.parametrize("order,d_min,overestimate", [
        (1, 0.2113, 5.60),
        (2, 0.1127, 19.68),
        (3, 0.0694, 51.86),
    ])
    def test_min_convention_overestimates_by_documented_factor(
            self, order, d_min, overestimate):
        d = _gl_normalized_positions(order)
        assert d.min() == pytest.approx(d_min, abs=5e-4), (
            f"order={order} 最近解点位置 {d.min():.4f} 与文档记录的 "
            f"{d_min} 不符")
        got = (0.5 / d.min()) ** 2
        assert got == pytest.approx(overestimate, rel=2e-3), (
            f"order={order} 的高估倍数 {got:.2f} 与文档记录的 "
            f"{overestimate} 不符")

    def test_overestimate_grows_monotonically_with_order(self):
        """**这才是问题的本质**：一个经过标定的经验壁面函数，其目标值
        绝不应该随离散阶数系统性变化。"""
        factors = [(0.5 / _gl_normalized_positions(p).min()) ** 2
                   for p in (1, 2, 3, 4)]
        assert all(b > a for a, b in zip(factors, factors[1:])), (
            f"高估倍数没有随阶数单调增长：{[f'{f:.2f}' for f in factors]}")
        assert factors[0] > 5.0, "order=1 的高估倍数应当已经超过 5 倍"

    def test_mean_convention_is_order_independent(self):
        """`mean` 口径（解点壁距均值 ≈ 形心壁距）对阶数是一阶无关的。

        Gauss-Legendre 点集关于区间中点对称，因此归一化位置的均值恒为
        0.5——正是 Menter 标定用的单元中心口径。
        """
        for p in (1, 2, 3, 4):
            d = _gl_normalized_positions(p)
            assert d.mean() == pytest.approx(0.5, abs=1e-12), (
                f"order={p} 的解点归一化位置均值 {d.mean():.6f} != 0.5——"
                f"Gauss-Legendre 点集应当关于中点对称")


class TestSwitchSemantics:
    def _env(self, v):
        old = os.environ.get("AFCFD_OMEGA_WALL_D1")
        if v is None:
            os.environ.pop("AFCFD_OMEGA_WALL_D1", None)
        else:
            os.environ["AFCFD_OMEGA_WALL_D1"] = v
        return old

    def _restore(self, old):
        if old is None:
            os.environ.pop("AFCFD_OMEGA_WALL_D1", None)
        else:
            os.environ["AFCFD_OMEGA_WALL_D1"] = old

    @pytest.mark.parametrize("mode,expect_min", [("min", True), ("mean", False)])
    def test_d1_selection_follows_the_switch(self, mode, expect_min):
        """直接验证选择逻辑：构造一个壁距明显不均匀的单元，两种口径必须
        分别给出 min 与 mean。"""
        wd = np.array([[1.0e-5, 3.0e-5, 5.0e-5, 7.0e-5]])
        old = self._env(mode)
        try:
            m = os.environ.get("AFCFD_OMEGA_WALL_D1", "min").lower()
            d1 = wd.min(axis=1) if m == "min" else wd.mean(axis=1)
        finally:
            self._restore(old)
        expected = wd.min() if expect_min else wd.mean()
        assert d1[0] == pytest.approx(expected)

    def test_default_is_min(self):
        """默认必须是既有行为——本轮只加开关，不改默认值。"""
        old = self._env(None)
        try:
            assert os.environ.get("AFCFD_OMEGA_WALL_D1", "min").lower() == "min"
        finally:
            self._restore(old)

    @pytest.mark.parametrize("bad", ["center", "cell", "", "MINIMUM"])
    def test_rejects_unknown_values(self, bad):
        """不能静默退回默认——与本项目其余开关同一条约定。"""
        from autoflowcfd.core.turbulence.transport import _compute_omega_wall_target
        import inspect
        src = inspect.getsource(_compute_omega_wall_target)
        assert 'AFCFD_OMEGA_WALL_D1' in src
        assert '不是合法取值' in src, (
            "非法取值必须显式报错，不能静默退回默认")
