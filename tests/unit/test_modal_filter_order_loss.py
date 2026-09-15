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

import importlib
import os

import numpy as np
import pytest


def _reload_with_env(**env):
    """在给定环境变量下重新导入 modal_filter 与 operators（模块级常量
    在导入时求值，必须重载才能生效）。"""
    old = {k: os.environ.get(k) for k in env}
    try:
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        import autoflowcfd.fr.modal_filter as mf
        importlib.reload(mf)
        import autoflowcfd.fr.operators as ops_mod
        importlib.reload(ops_mod)
        return mf, ops_mod
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.fixture(autouse=True)
def _restore_modules():
    """每个用例结束后把两个模块恢复成默认环境下的状态，避免污染其它
    测试文件（它们会 import 这两个模块的模块级常量/算子）。"""
    yield
    _reload_with_env(AFCFD_FILTER_MODE=None, AFCFD_FILTER_SIGMA_TOP=None)


def _prism_filter(ops_mod, order):
    return np.asarray(ops_mod.generate_fr_operators(order).filter_prism)


class TestLegacyFilterLosesExactlyOneOrder:
    """默认（legacy）行为：保留模态数恰好是 order^3，即损失一整阶。"""

    @pytest.mark.parametrize("order,expected_rank", [(1, 1), (2, 8), (3, 27)])
    def test_rank_equals_order_cubed(self, order, expected_rank):
        mf, ops_mod = _reload_with_env(AFCFD_FILTER_MODE=None)
        F = _prism_filter(ops_mod, order)
        n = (order + 1) ** 3
        assert F.shape == (n, n)
        rank = int(np.linalg.matrix_rank(F, 1e-10))
        assert rank == expected_rank, (
            f"order={order}: 滤波矩阵秩={rank}，期望 {expected_rank}=order^3。"
            f"秩变化意味着'损失几阶'变了——这是精度层面的行为改变，"
            f"必须显式确认而不是顺带改掉")

    def test_top_mode_sigma_is_annihilation_not_damping(self):
        mf, _ = _reload_with_env(AFCFD_FILTER_MODE=None)
        sigma_top = float(mf._exp_filter_sigma(np.array(1.0)))
        assert sigma_top < 1e-12, (
            f"sigma(eta=1)={sigma_top:.3e}，不再是'清零'量级——"
            f"若这是有意的放宽，请同步更新本文件与 modal_filter.py 的说明")
        # 常数模态必须完全保留（自由流场保持性）
        assert float(mf._exp_filter_sigma(np.array(0.0))) == 1.0

    def test_p1_annihilates_linear_intra_cell_content(self):
        """order=1 时线性场的**胞内**变化被抹到机器零——P1 退化为 P0
        最直接的证据（不依赖任何真实网格）。"""
        mf, ops_mod = _reload_with_env(AFCFD_FILTER_MODE=None)
        F = _prism_filter(ops_mod, 1)
        # order=1 的节点空间里，任何非常数向量都只能由常数模态 + 那些
        # max(i,j,k)=1 的模态张成，所以用随机向量减去其常数部分就是
        # "纯胞内变化"，不需要显式构造线性场坐标。
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

    def test_default_mode_is_legacy(self):
        """默认必须保持既有行为——本轮只加可切换档位，不改默认。"""
        mf, ops_mod = _reload_with_env(AFCFD_FILTER_MODE=None)
        assert mf.FILTER_MODE == "legacy"
        assert int(np.linalg.matrix_rank(_prism_filter(ops_mod, 1), 1e-10)) == 1


class TestSensorModeIsNotSilentlyIgnored:
    """`AFCFD_FILTER_MODE=sensor` 只在单机 CPU 推进循环里接线过。

    legacy/off/mild 三档是在算子构造期改滤波矩阵本身，对全部后端自动
    生效；sensor 档必须在推进循环里逐单元求 Persson-Peraire 指示器，
    只能逐后端实现。如果别的后端"读不到这个分支所以按 legacy 跑"，同一
    个环境变量在不同后端就意味着不同的数值方案且毫无提示——本项目不
    接受这种静默行为，所以 `resolve_filter_mode` 直接报错。
    """

    @pytest.mark.parametrize("backend", ["cpu-mpi", "gpu-single", "gpu-mpi"])
    def test_sensor_raises_on_unwired_backends(self, backend):
        from autoflowcfd.core.fr_solver.filter import resolve_filter_mode
        old = os.environ.get("AFCFD_FILTER_MODE")
        os.environ["AFCFD_FILTER_MODE"] = "sensor"
        try:
            with pytest.raises(NotImplementedError, match="sensor"):
                resolve_filter_mode(backend)
        finally:
            if old is None:
                os.environ.pop("AFCFD_FILTER_MODE", None)
            else:
                os.environ["AFCFD_FILTER_MODE"] = old

    def test_sensor_is_allowed_on_cpu_single(self):
        from autoflowcfd.core.fr_solver.filter import resolve_filter_mode
        old = os.environ.get("AFCFD_FILTER_MODE")
        os.environ["AFCFD_FILTER_MODE"] = "sensor"
        try:
            assert resolve_filter_mode("cpu-single") == "sensor"
        finally:
            if old is None:
                os.environ.pop("AFCFD_FILTER_MODE", None)
            else:
                os.environ["AFCFD_FILTER_MODE"] = old

    @pytest.mark.parametrize("mode", ["legacy", "off", "mild"])
    @pytest.mark.parametrize("backend", ["cpu-single", "cpu-mpi", "gpu-single", "gpu-mpi"])
    def test_other_modes_pass_on_every_backend(self, mode, backend):
        """这三档不需要逐后端接线，任何后端都不该拦。"""
        from autoflowcfd.core.fr_solver.filter import resolve_filter_mode
        old = os.environ.get("AFCFD_FILTER_MODE")
        os.environ["AFCFD_FILTER_MODE"] = mode
        try:
            assert resolve_filter_mode(backend) == mode
        finally:
            if old is None:
                os.environ.pop("AFCFD_FILTER_MODE", None)
            else:
                os.environ["AFCFD_FILTER_MODE"] = old

    def test_default_is_legacy_everywhere(self):
        from autoflowcfd.core.fr_solver.filter import resolve_filter_mode
        old = os.environ.pop("AFCFD_FILTER_MODE", None)
        try:
            for b in ("cpu-single", "cpu-mpi", "gpu-single", "gpu-mpi"):
                assert resolve_filter_mode(b) == "legacy"
        finally:
            if old is not None:
                os.environ["AFCFD_FILTER_MODE"] = old


class TestCompoundingOverStages:
    """把"每个 RK stage 都施加 => 任何 sigma<1 都会复合累积"这条量化清楚。

    这条决定了"只把 alpha 调小"为什么不是正确方向：它只是把清零推迟。
    """

    @pytest.mark.parametrize("sigma_top,stages", [(0.99, 300), (0.9, 300), (0.5, 60)])
    def test_mild_filter_still_annihilates_over_many_stages(self, sigma_top, stages):
        mf, ops_mod = _reload_with_env(AFCFD_FILTER_MODE="mild",
                                       AFCFD_FILTER_SIGMA_TOP=str(sigma_top))
        F = _prism_filter(ops_mod, 1)
        rng = np.random.default_rng(11)
        f = rng.standard_normal(F.shape[1])
        var0 = np.abs(f - f.mean()).max()
        g = f.copy()
        for _ in range(stages):
            g = F @ g
        var1 = np.abs(g - g.mean()).max()
        retain = var1 / max(var0, 1e-300)
        expected = sigma_top ** stages
        # 只断言"确实在指数衰减、量级不慢于 sigma^stages"，并且**不低于
        # 双精度噪声底**时才比较——实测 sigma=0.5 施加 60 次后
        # retain=4.89e-17 而 sigma^60=8.67e-19，差 56 倍纯粹是因为
        # 已经触到噪声底（这里的 var 是相对 O(1) 量的差，分辨率 ~1e-17），
        # 不是衰减变慢。
        assert retain <= max(10.0 * expected, 1e-14)
        assert retain < 0.2, (
            f"sigma_top={sigma_top} 施加 {stages} 次后仍保留 {retain:.3e}——"
            f"本用例的前提（会复合累积）不成立，请重新评估")
