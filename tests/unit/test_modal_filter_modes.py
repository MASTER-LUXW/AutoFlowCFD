"""模态滤波档位：sensor 档不被静默忽略、两种基分别检查、恒等滤波短路、多级累积（从 test_modal_filter_order_loss.py 拆出，背景见该文件模块文档）。"""

import os
import numpy as np
import pytest

from autoflowcfd.core.fr_solver.filter import _SENSOR_MODE_SUPPORTED_BACKENDS

from tests.unit._modal_filter_order_loss_common import (
    _ALL_FILTER_BACKENDS,
    _expected_legacy_rank,
    _prism_filter,
    _real_block,
    _reload_with_env,
)
from tests.unit._modal_filter_order_loss_common import _restore_modules  # noqa: F401  autouse 夹具：进入本模块命名空间才生效


class TestSensorModeIsNotSilentlyIgnored:
    """`AFCFD_FILTER_MODE=sensor` 必须逐后端接线。

    legacy/off/mild 三档是在算子构造期改滤波矩阵本身，对全部后端自动
    生效；sensor 档必须在推进循环里逐单元求传感器指示器，只能逐后端
    实现。如果未接线的后端"读不到这个分支所以按 legacy 跑"，同一个环境
    变量在不同后端就意味着不同的数值方案且毫无提示——本项目不接受这种
    静默行为：显式请求直接报错，默认值退到 `project` 并打一条量化了
    数值后果的警告。

    **2026-09-18：四条后端全部接线完成**，于是 `resolve_filter_mode`
    里那条"默认值退到 project 并打警告"的分支成了死代码，已删除——
    现在无论显式请求还是默认值，未接线的后端一律报错。本类原先针对
    "未接线后端"的参数化判据随之删除（留着会是空参数集，静默地什么
    都不测），换成下面两条：全部后端都必须已接线、未知后端名必须报错。
    """

    def test_every_real_backend_is_wired(self):
        """四条真实后端全部在已接线列表里。

        将来新增后端时这条会失败——那正是要的：新后端必须显式接线，
        不能靠一条 warning 悄悄降级成 project（那会精确抹掉最高一阶
        多项式内容，P1 退化成 P0）。
        """
        missing = [b for b in _ALL_FILTER_BACKENDS
                   if b not in _SENSOR_MODE_SUPPORTED_BACKENDS]
        assert not missing, f"这些后端还没接线 sensor 门控：{missing}"

    def test_unknown_backend_name_raises(self):
        """拼错/未知的后端名必须报错，不能静默按某一档跑。"""
        from autoflowcfd.core.fr_solver.filter import resolve_filter_mode
        old = os.environ.pop("AFCFD_FILTER_MODE", None)
        try:
            with pytest.raises(NotImplementedError, match="gpu-rocm"):
                resolve_filter_mode("gpu-rocm")
        finally:
            if old is not None:
                os.environ["AFCFD_FILTER_MODE"] = old

    @pytest.mark.parametrize("backend", _ALL_FILTER_BACKENDS)
    def test_sensor_is_allowed_on_wired_backends(self, backend):
        from autoflowcfd.core.fr_solver.filter import resolve_filter_mode
        old = os.environ.get("AFCFD_FILTER_MODE")
        os.environ["AFCFD_FILTER_MODE"] = "sensor"
        try:
            assert resolve_filter_mode(backend) == "sensor"
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

    def test_default_resolves_to_off_on_every_backend(self):
        """默认值（2026-09-19 起 `off`）的**逐后端**解析。

        `off` 不需要任何后端接线（它是恒等滤波、`build_filter_func` 直接
        返回 None），所以四条后端都必须解析成 `off`；未知后端名报错
        （见 `test_unknown_backend_name_raises`）。
        """
        from autoflowcfd.core.fr_solver.filter import resolve_filter_mode
        old = os.environ.pop("AFCFD_FILTER_MODE", None)
        try:
            for b in _ALL_FILTER_BACKENDS:
                assert resolve_filter_mode(b) == "off", b
        finally:
            if old is not None:
                os.environ["AFCFD_FILTER_MODE"] = old

    def test_two_resolvers_share_one_default(self):
        """**真实 bug 回归（2026-09-17）**：滤波档有两个独立解析器——
        `fr/modal_filter.py` 定矩阵的 sigma、`fr_solver/filter.py` 定
        `step.py` 走不走门控分支。把默认从 `legacy` 改成 `sensor` 时只改了
        前者，于是默认路径变成"矩阵是精确投影、但全局逐 stage 施加"，
        功能上等于 legacy（实测两者在 P1 上逐位相同），壁面剪应力照样被
        清零（平板边界层算例 du/dy 从 1734 变成 0）。两处必须同源。
        """
        import importlib

        from autoflowcfd.core.fr_solver import filter as f
        import autoflowcfd.fr.modal_filter as mf

        old = os.environ.pop("AFCFD_FILTER_MODE", None)
        try:
            mf = importlib.reload(mf)
            assert f.resolve_filter_mode("cpu-single") == mf.FILTER_MODE, (
                "两个解析器的默认值不一致——会出现「矩阵按一档、施加方式"
                "按另一档」这种没人能从日志里看出来的组合")
        finally:
            if old is not None:
                os.environ["AFCFD_FILTER_MODE"] = old

    def test_explicit_sensor_resolves_on_every_backend(self):
        """显式请求 sensor 在四条后端上都能满足（全部已接线）。"""
        import pytest as _pytest

        from autoflowcfd.core.fr_solver.filter import resolve_filter_mode
        old = os.environ.get("AFCFD_FILTER_MODE")
        os.environ["AFCFD_FILTER_MODE"] = "sensor"
        try:
            for b in _ALL_FILTER_BACKENDS:
                assert resolve_filter_mode(b) == "sensor", b
            # 未知后端名现在被**无条件**的后端校验先拦住（2026-09-19：
            # 那条校验此前寄生在 `mode == "sensor"` 判据里，默认值改成
            # `off` 之后就会静默放行，所以提到了前面）。两种情形都必须
            # 报 NotImplementedError，只是先触发的那条变了。
            with _pytest.raises(NotImplementedError, match="未知后端标识"):
                resolve_filter_mode("gpu-rocm")
        finally:
            if old is None:
                os.environ.pop("AFCFD_FILTER_MODE", None)
            else:
                os.environ["AFCFD_FILTER_MODE"] = old


class TestBothBasesMustBeCheckedSeparately:
    """**两套基的滤波矩阵必须分别自证**——真实 bug 回归（2026-09-15）。

    `fr/native_tet/filter.py::build_native_tet_modal_filter` 此前只短路
    `order == 0`，漏了 `FILTER_MODE == "off"`（`fr/modal_filter.py` 的
    棱柱/坍缩两个构造函数都有那条短路）。后果是
    **`AFCFD_FILTER_MODE=off` 只关掉了棱柱的滤波器，四面体照旧每个
    RK stage 被清掉一整阶**：

        MODE=off order=1: prism 秩 8/8 是单位阵 | tet 秩 5/8 与 legacy 完全相同
        MODE=off order=2: prism 秩 27/27      | tet 秩 21/27

    79 万单元 cube_demo 的 `n_prism=136980`，四面体 654512 个占 **82.7%**
    ——也就是说"关掉滤波器"的几轮对照实验里，绝大多数单元根本没被关掉，
    整组实验的前提是错的。排查时之所以漏掉，是因为诊断日志只打印了
    `filter_prism` 的秩、用它代表了另一套基。

    本类对**两个矩阵**逐一断言，覆盖三档全部组合。
    """

    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_off_makes_both_matrices_exactly_identity(self, order):
        _, ops_mod = _reload_with_env(AFCFD_FILTER_MODE="off")
        ops = ops_mod.generate_fr_operators(order)
        for name in ("filter_prism", "filter_tet"):
            M = np.asarray(getattr(ops, name))
            np.testing.assert_array_equal(
                M, np.eye(M.shape[0]),
                err_msg=f"off 档 order={order} 的 {name} 不是单位阵——"
                        f"该档位承诺零阶数损失，任何一套基漏掉都会让"
                        f"对照实验的前提失效")

    @pytest.mark.parametrize("order", [1, 2])
    def test_legacy_annihilates_in_both_bases(self, order):
        """**legacy 档**两套基都必须真的在清零（否则'损失一整阶'这个
        事实本身就只对其中一套成立）。

        2026-09-18：此前写的是"默认档"并用 `AFCFD_FILTER_MODE=None`，
        见 TestLegacyFilterLosesExactlyOneOrder 的同一处更正。"""
        _, ops_mod = _reload_with_env(AFCFD_FILTER_MODE="legacy")
        ops = ops_mod.generate_fr_operators(order)
        n = (order + 1) ** 3
        rank_p = int(np.linalg.matrix_rank(np.asarray(ops.filter_prism), 1e-10))
        rank_t = int(np.linalg.matrix_rank(np.asarray(ops.filter_tet), 1e-10))
        expect_p = _expected_legacy_rank(order)
        assert rank_p == expect_p, (
            f"prism 秩 {rank_p} != 期望 {expect_p}"
            f"（见 `_expected_legacy_rank`：两条棱柱基各自的形式）")
        # native 四面体：真实自由度里只剩 i+j+k<=order-1 的那些，
        # 再加 (n - n_native) 个单位阵填充行。
        n_native = (order + 1) * (order + 2) * (order + 3) // 6
        n_keep = order * (order + 1) * (order + 2) // 6
        assert rank_t == n_keep + (n - n_native), (
            f"tet 秩 {rank_t} 与'保留 i+j+k<=order-1 + 填充行'不符"
            f"（期望 {n_keep}+{n - n_native}）")

    @pytest.mark.parametrize("order", [1, 2])
    def test_mild_keeps_both_matrices_full_rank(self, order):
        _, ops_mod = _reload_with_env(AFCFD_FILTER_MODE="mild",
                                      AFCFD_FILTER_SIGMA_TOP="0.99")
        ops = ops_mod.generate_fr_operators(order)
        n = (order + 1) ** 3
        for name in ("filter_prism", "filter_tet"):
            M = np.asarray(getattr(ops, name))
            assert int(np.linalg.matrix_rank(M, 1e-10)) == n, (
                f"mild 档 order={order} 的 {name} 不满秩")
            assert not np.array_equal(M, np.eye(n)), (
                f"mild 档 {name} 退化成了单位阵（那是 off 档的语义）")


class TestIdentityFilterIsShortCircuited:
    """滤波矩阵是单位阵时必须**整个跳过**调用，而不是白乘一遍。

    `AFCFD_FILTER_MODE=off` 下两个矩阵都是单位阵（mild 档取
    sigma_top=1.0 也一样）。79 万单元 P1 下 U 是 (791492,8,5) ≈ 253MB，
    一步三个 RK stage 白读写约 1.5GB 纯内存带宽；k/omega 两个标量场各
    ≈ 50MB、每步一次。`TimeIntegrator.step`/`step_dual_time` 对
    `filter_func=None` 有显式支持（分布式路径在 n_sps==1 时本来就传
    None），所以 `build_filter_func` 直接返回 None。

    判据刻意**看矩阵内容**而不是环境变量：那样连 sigma_top=1.0 这种
    等价配置也一并短路，也不依赖"环境变量与算子构造保持同步"这个隐含
    假设。
    """

    def _solver_stub(self, ops_mod, order):
        from types import SimpleNamespace
        ops = ops_mod.generate_fr_operators(order)
        mesh = SimpleNamespace(n_cells=4, n_sps_per_cell=(order + 1) ** 3,
                               n_prism_cells=2)
        return SimpleNamespace(mesh=mesh, ops=ops)

    @pytest.mark.parametrize("order", [1, 2])
    def test_off_returns_none(self, order):
        from autoflowcfd.core.fr_solver.filter import build_filter_func
        _, ops_mod = _reload_with_env(AFCFD_FILTER_MODE="off")
        assert build_filter_func(self._solver_stub(ops_mod, order)) is None

    @pytest.mark.parametrize("order", [1, 2])
    def test_legacy_returns_callable(self, order):
        """legacy 档必须仍然返回可调用对象——短路不能误伤真实滤波档。

        2026-09-18 改成显式 legacy（此前用默认档，见
        TestLegacyFilterLosesExactlyOneOrder 的同一处更正）。默认档现在是
        `sensor`（顶模态 0.99 有界衰减），它的矩阵同样不是单位阵、同样
        必须返回可调用对象——由 `test_default_mode_matrix_is_bounded_
        damping` 覆盖。"""
        from autoflowcfd.core.fr_solver.filter import build_filter_func
        _, ops_mod = _reload_with_env(AFCFD_FILTER_MODE="legacy")
        f = build_filter_func(self._solver_stub(ops_mod, order))
        assert f is not None and callable(f)

    def test_mild_with_sigma_top_one_also_short_circuits(self):
        """sigma_top=1.0 与 off 数学等价，也必须短路。"""
        from autoflowcfd.core.fr_solver.filter import build_filter_func
        _, ops_mod = _reload_with_env(AFCFD_FILTER_MODE="mild",
                                      AFCFD_FILTER_SIGMA_TOP="1.0")
        assert build_filter_func(self._solver_stub(ops_mod, 1)) is None

    @pytest.mark.parametrize("mode,expected_identity", [
        ("off", True), ("legacy", False),
    ])
    def test_turbulence_side_helper_agrees(self, mode, expected_identity):
        """k/omega 那一侧用的是同一套判据（独立实现在 turbulence.py，
        因为那里不经过 build_filter_func）。两者必须给出一致的判断。"""
        from autoflowcfd.core.fr_solver.turbulence import (
            _filter_matrices_are_identity,
        )
        # 两档都**显式**传（此前 legacy 那一支传 None、靠"默认值就是
        # legacy"，那个假设在 2026-09-17 默认改成 sensor 时就已过时、
        # 只因 sensor 也非恒等而侥幸通过；2026-09-19 默认改成 off 之后
        # 就直接失败了）。
        _, ops_mod = _reload_with_env(AFCFD_FILTER_MODE=mode)
        ops = ops_mod.generate_fr_operators(1)
        assert _filter_matrices_are_identity(ops) is expected_identity


class TestCompoundingOverStages:
    """把"每个 RK stage 都施加 => 任何 sigma<1 都会复合累积"这条量化清楚。

    这条决定了"只把 alpha 调小"为什么不是正确方向：它只是把清零推迟。
    """

    @pytest.mark.parametrize("sigma_top,stages", [(0.99, 300), (0.9, 300), (0.5, 60)])
    def test_mild_filter_still_annihilates_over_many_stages(self, sigma_top, stages):
        mf, ops_mod = _reload_with_env(AFCFD_FILTER_MODE="mild",
                                       AFCFD_FILTER_SIGMA_TOP=str(sigma_top))
        # 只取真实自由度子块，理由同
        # `test_p1_annihilates_linear_intra_cell_content`。
        F = _real_block(_prism_filter(ops_mod, 1), 1)
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
