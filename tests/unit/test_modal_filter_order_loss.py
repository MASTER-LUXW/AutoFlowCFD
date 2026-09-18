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

from autoflowcfd.core.fr_solver.filter import _SENSOR_MODE_SUPPORTED_BACKENDS


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

    def test_default_mode_is_sensor(self):
        """默认 `sensor`（2026-09-17 从 `legacy` 改）。

        依据（P1 实测，平板边界层算例，同一初场同一 CFL）：

            legacy  176.2 ms/step  res 7.9426e+04
            project 156.9 ms/step  res 7.9426e+04  <- 与 legacy 逐位相同
            off     179.7 ms/step  res 1.6658e+05
            sensor+persson 154.3  res 1.6658e+05   <- 与 off 逐位相同
            sensor+bounds  183.1  res 7.7280e+04   <- 残差最低，+3.9%

        `legacy == project` 在 P1 上成立是因为 P1 的顶模态就是全部非常数
        内容，两档都把它清零、都让 P1 退化成 P0。`sensor+bounds` 在等熵涡
        精确解上保住收敛阶 2.16/2.18（设计阶 2）。完整依据见
        `fr/modal_filter.py` 里 `_FILTER_MODE` 上方那节。
        """
        mf, ops_mod = _reload_with_env(AFCFD_FILTER_MODE=None)
        assert mf.FILTER_MODE == "sensor"
        # sensor 档的矩阵与 project 相同：严格投影，P1 的秩仍是 1
        # （门控在应用层，不在矩阵里——见 modal_filter.py 的说明）
        assert int(np.linalg.matrix_rank(_prism_filter(ops_mod, 1), 1e-10)) == 1

    def test_legacy_stays_available_for_regression(self):
        """`legacy` 是唯一能复现历史结果的档，必须保留为合法取值。"""
        mf, ops_mod = _reload_with_env(AFCFD_FILTER_MODE="legacy")
        assert mf.FILTER_MODE == "legacy"


#: 全部后端标识。哪些**已接线**不在这里硬编码——从
#: `_SENSOR_MODE_SUPPORTED_BACKENDS` 读（同一个语义只允许一个事实来源；
#: 硬编码一份会让"接线完成但测试仍钉着旧状态"这种失败反复出现，
#: 2026-09-18 接线 cpu-mpi 时就是这么被绊了一次）。
_ALL_FILTER_BACKENDS = ("cpu-single", "cpu-mpi", "gpu-single", "gpu-mpi")
_WIRED_BACKENDS = [b for b in _ALL_FILTER_BACKENDS
                   if b in _SENSOR_MODE_SUPPORTED_BACKENDS]
_UNWIRED_BACKENDS = [b for b in _ALL_FILTER_BACKENDS
                     if b not in _SENSOR_MODE_SUPPORTED_BACKENDS]


class TestSensorModeIsNotSilentlyIgnored:
    """`AFCFD_FILTER_MODE=sensor` 必须逐后端接线。

    legacy/off/mild 三档是在算子构造期改滤波矩阵本身，对全部后端自动
    生效；sensor 档必须在推进循环里逐单元求传感器指示器，只能逐后端
    实现。如果未接线的后端"读不到这个分支所以按 legacy 跑"，同一个环境
    变量在不同后端就意味着不同的数值方案且毫无提示——本项目不接受这种
    静默行为：显式请求直接报错，默认值退到 `project` 并打一条量化了
    数值后果的警告。

    接线进度（2026-09-18：`cpu-mpi` 已补齐，见
    `core/mpi/distributed_solver.py::_build_sensor_gated_filter_func_
    distributed` 与 `tests/unit/test_sensor_gate_distributed.py`）由
    `_SENSOR_MODE_SUPPORTED_BACKENDS` 单一决定，本类全部判据从它派生。
    """

    def test_wired_and_unwired_sets_are_both_nonempty(self):
        """两个集合都非空，否则下面的参数化测试会静默变成空集合。

        接线全部完成之后这条会失败——那时应当删掉"未接线"那几条测试，
        而不是让它们静默地什么都不测。
        """
        assert _WIRED_BACKENDS, "没有任何已接线后端，参数化测试成了空集"
        assert _UNWIRED_BACKENDS, (
            "全部后端都已接线——请删除本类中针对未接线后端的判据，"
            "而不是留着空参数化")

    @pytest.mark.parametrize("backend", _UNWIRED_BACKENDS)
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

    @pytest.mark.parametrize("backend", _WIRED_BACKENDS)
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

    def test_default_is_sensor_where_wired_and_project_elsewhere(self):
        """默认值（2026-09-17 起 `sensor`）的**逐后端**解析。

        已接线的后端拿到 `sensor`；未接线的后端默认值退到 `project`
        ——同一个精确投影矩阵、但**全局逐 RK stage
        施加**。为什么默认可以退而显式请求不可以，见
        `resolve_filter_mode` 里那段说明：本项目禁止的是**无声**地把
        明确请求换掉，而默认值必须让每个后端都能跑起来，退档时打一条
        量化了数值后果的警告。
        """
        from autoflowcfd.core.fr_solver.filter import resolve_filter_mode
        old = os.environ.pop("AFCFD_FILTER_MODE", None)
        try:
            for b in _WIRED_BACKENDS:
                assert resolve_filter_mode(b) == "sensor", b
            for b in _UNWIRED_BACKENDS:
                assert resolve_filter_mode(b) == "project", b
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

    def test_explicit_sensor_on_unwired_backend_raises(self):
        """显式请求得不到满足必须报错，不能退档。"""
        import pytest as _pytest

        from autoflowcfd.core.fr_solver.filter import resolve_filter_mode
        old = os.environ.get("AFCFD_FILTER_MODE")
        os.environ["AFCFD_FILTER_MODE"] = "sensor"
        try:
            for b in _WIRED_BACKENDS:
                assert resolve_filter_mode(b) == "sensor", b
            for b in _UNWIRED_BACKENDS:
                with _pytest.raises(NotImplementedError, match="尚未在后端"):
                    resolve_filter_mode(b)
        finally:
            if old is None:
                os.environ.pop("AFCFD_FILTER_MODE", None)
            else:
                os.environ["AFCFD_FILTER_MODE"] = old


class TestBothBasesMustBeCheckedSeparately:
    """**两套基的滤波矩阵必须分别自证**——真实 bug 回归（2026-09-15）。

    `fr/native_tet_filter.py::build_native_tet_modal_filter` 此前只短路
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
        """默认档两套基都必须真的在清零（否则'损失一整阶'这个事实本身
        就只对其中一套成立）。"""
        _, ops_mod = _reload_with_env(AFCFD_FILTER_MODE=None)
        ops = ops_mod.generate_fr_operators(order)
        n = (order + 1) ** 3
        rank_p = int(np.linalg.matrix_rank(np.asarray(ops.filter_prism), 1e-10))
        rank_t = int(np.linalg.matrix_rank(np.asarray(ops.filter_tet), 1e-10))
        assert rank_p == order ** 3, f"prism 秩 {rank_p} != order^3={order**3}"
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
        """默认档必须仍然返回可调用对象——短路不能误伤生产默认路径。"""
        from autoflowcfd.core.fr_solver.filter import build_filter_func
        _, ops_mod = _reload_with_env(AFCFD_FILTER_MODE=None)
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
        _, ops_mod = _reload_with_env(
            AFCFD_FILTER_MODE=(None if mode == "legacy" else mode))
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
