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
from tests.unit._filter_mode import (
    reload_filter_modules,
    restore_default_filter_modules,
)


@pytest.fixture
def project_mode():
    """在 `project` 档（严格投影：sigma 只取 0 或 1）下运行本用例。

    **为什么需要它（2026-09-18）**：本文件此前用**默认档**去测"顶模态被
    压到机器精度""矩阵幂等""节点 std 比值等于某个实测值"——那些都是
    **投影型**矩阵的性质，不是"滤波器"的性质。默认档的 sensor 从精确
    投影改成顶模态 sigma=0.99 的有界衰减（理由与 Blasius 实测见
    `fr/modal_filter.py::filter_sigma`）之后它们全部失败，暴露出这些测试
    从来不是在测自己文档里说的那一档。
    """
    reload_filter_modules(AFCFD_FILTER_MODE="project")
    yield
    restore_default_filter_modules()
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
    """**project/legacy 档**：最高阶模态（i+j+k==order）被压到机器精度
    量级，与坍缩坐标版本同一个设计承诺。

    默认档（sensor）现在是顶模态 sigma=0.99 的**有界衰减**而不是清零，
    对应性质见本文件 `TestDefaultModeIsBoundedDamping`。
    """

    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_top_mode_damped(self, order, project_mode):
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
    def test_filter_is_idempotent(self, order, project_mode):
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
    def test_nodal_std_ratio_matches_measurement(self, order, project_mode):
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


class TestSensorModeIsBoundedDamping:
    """**`sensor`/`mild` 档**的 native 四面体矩阵：有界衰减，不是投影。

    2026-09-18 引入这种矩阵，完整推导见 `fr/modal_filter.py::filter_sigma`。
    判据方向与上面那几类刻意相反：这里要求矩阵**满秩**且**非幂等**，因为
    "只要传感器还在报就持续慢慢耗散"是那一档的设计意图；而 L2 不增
    （不放大）这条对两档都必须成立。

    **不再叫"默认档"（2026-09-19）**：默认值已改成 `off`（恒等滤波），
    依据是修掉原生四面体路径两个重大缺陷之后用有解析解的算例重新判定 ——
    `sensor` 在 Blasius 上与 `off` 逐位相同（实质无操作），但在 TGV 上
    因 BJ 判据 100% 标记而退化成全局施加，450 次累积出非物理的能量增长
    （+5.14%）。完整依据见 `fr/modal_filter.py` 里"默认值 2026-09-19 改为
    off"那一节。本类保留是因为 `sensor`/`mild` 仍是合法档，其矩阵契约
    必须继续被钉住。
    """

    @pytest.fixture(autouse=True)
    def _sensor_mode(self):
        """本类**显式**在 `sensor` 档下运行（默认档已是 `off`=恒等）。"""
        reload_filter_modules(AFCFD_FILTER_MODE="sensor")
        yield
        restore_default_filter_modules()

    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_matrix_is_full_rank(self, order):
        F = build_native_tet_modal_filter(order)
        assert int(np.linalg.matrix_rank(F, 1e-10)) == F.shape[0], (
            f"order={order}: sensor 档矩阵不满秩——投影型会丢掉一整阶，"
            f"而实测那会把壁面剪应力压掉 2.3 倍")

    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_top_mode_is_damped_to_sigma_top_not_zero(self, order):
        """顶模态被乘上 0.99，而不是清零。"""
        from autoflowcfd.fr.native_simplex_basis import (
            build_native_tet_operators, restricted_tet_modes,
            simplex3d_value, rst_to_abc,
        )

        modes = restricted_tet_modes(order)
        ref_rst, _ = build_native_tet_operators(order)
        a, b, c = rst_to_abc(ref_rst[:, 0], ref_rst[:, 1], ref_rst[:, 2])
        V = np.column_stack(
            [simplex3d_value(a, b, c, i, j, k) for (i, j, k) in modes])
        F = build_native_tet_modal_filter(order)
        tops = [m for m, (i, j, k) in enumerate(modes) if i + j + k == order]
        assert tops
        for m in tops:
            coeff = np.zeros(len(modes))
            coeff[m] = 1.0
            nodal = V @ coeff
            got = F @ nodal
            # 特征向量：结果应当是同一个模态乘 0.99
            ratio = np.linalg.norm(got) / np.linalg.norm(nodal)
            assert abs(ratio - 0.99) < 1e-9, (
                f"order={order} mode {m}: 顶模态被乘了 {ratio:.9f}，"
                f"应当是 0.99（默认 AFCFD_FILTER_SIGMA_TOP）")

    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_is_not_idempotent_by_design(self, order):
        """非幂等是**刻意的**：重复施加持续耗散。

        这条与 `TestFilterNeverAmplifiesInTheRightNorm::
        test_filter_is_idempotent`（project 档）方向相反，两者都对——
        它们测的是两个不同的档。
        """
        F = build_native_tet_modal_filter(order)
        err = np.max(np.abs(F @ F - F))
        assert err > 1e-6, (
            f"order={order}: 默认档矩阵是幂等的（|F@F-F|={err:.2e}）——"
            f"那说明它退回成了投影型")

    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_constant_field_still_exactly_preserved(self, order):
        """自由流场保持性：常数场必须逐位不变（sigma(0)=1 严格）。"""
        F = build_native_tet_modal_filter(order)
        ones = np.ones(F.shape[0])
        np.testing.assert_allclose(F @ ones, ones, rtol=0, atol=1e-14)
