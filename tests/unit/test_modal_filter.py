"""AutoFlowCFD V2.0 - 指数模态滤波器 (fr/modal_filter.py) 单元测试。

此前这个模块（构造函数 build_tet_modal_filter/build_prism_modal_filter
及其在 core/fr_solver/filter.py 里的调用点）完全没有单元测试覆盖。

2026-08-29 调查记录：定量复现过一个真实现象——cube_demo 真实网格上
P1 求解 60 步之后，每个单元内部 8 个 SP 的速度分量彼此只差 ~1e-14
（机器精度噪声），单元间却相差 27 m/s；根因是 `eta=max(i,j,k)/order`
归一化在 order<=1 时必然把唯一的非常数模态归一化到 eta=1.0，被滤波器
在每个 RK stage 压到机器精度量级，本质上把 P1 求解器拍平成 P0。

**曾尝试但已放弃的"修复"**：(a) order<=1 返回单位矩阵；(b) order>=2
改成"只压制 max(i,j,k)==order 的模态，其余模态 sigma 严格等于 1.0"
的硬截断。两者在真实验证中都被证伪——(a) 用于 cube_demo 真实 P1 层流
求解，10 步内发散到 NaN（比原来的缓慢发散更差）；(b) 用于
tests/validation/test_tgv.py（P2 四面体）与
tests/validation/test_couette.py（P2 棱柱）都在数步内复现本模块
文档记载的原始灾难性混叠失稳（数值放大到 ~1e78 量级）。说明滤波器
对"中间"模态的光滑衰减不是可以去掉的多余抑制，而是真实承担着抑制
混叠失稳的数值稳定性职责——这与它会在足够多次迭代后磨灭这些模态
携带的真实物理梯度内容，是当前设计里一对尚未解决的真实张力，
不是简单调整 sigma 分段就能兼得的问题。已回退到原始实现（见
fr/modal_filter.py 模块文档"2026-08-29 调查记录"一节）。

本文件因此只测试*当前实际生效*的原始行为：常数场恒等、最高阶模态
被强力压制，并把"中间模态确实会被逐步磨灭"这一确认过的、暂未解决的
限制记录成一个量化回归测试（不是要修复它，只是不让这个已知限制在
未来被意外改得更好或更差却没人注意到）。任何后续想再次尝试放松
中间模态压制强度的人，必须先让 tests/validation/test_couette.py 和
tests/validation/test_tgv.py 通过，这两个是真正有效的稳定性回归测试。
"""

import numpy as np

import pytest

from autoflowcfd.fr.modal_filter import build_prism_modal_filter, build_tet_modal_filter
from autoflowcfd.fr.operators import gauss_legendre
from tests.unit._filter_mode import (
    reload_filter_modules,
    restore_default_filter_modules,
)


@pytest.fixture
def legacy_mode():
    """在 `legacy` 档下运行本用例，退出时恢复默认档。

    **为什么需要它（2026-09-18）**：本文件此前全部用**默认档**去测
    "顶模态被压到机器精度""线性模态在 180 次 stage 后被磨灭"这类性质
    ——而那些性质是 `legacy`（全局清零型）的性质，不是"滤波器"的性质。
    默认档从 legacy 改到 sensor（顶模态 sigma=0.99 的有界衰减，见
    `fr/modal_filter.py::filter_sigma`）之后它们全部失败，暴露出这些
    测试从来不是在测自己文档里说的那一档。
    """
    reload_filter_modules(AFCFD_FILTER_MODE="legacy")
    yield
    restore_default_filter_modules()


def _ref_cube_sps(order: int) -> np.ndarray:
    n1d = order + 1
    sps_1d, _ = gauss_legendre(n1d)
    xx, yy, zz = np.meshgrid(sps_1d, sps_1d, sps_1d, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])


def _linear_field(ref_cube_sps: np.ndarray) -> np.ndarray:
    a, b, c = ref_cube_sps[:, 0], ref_cube_sps[:, 1], ref_cube_sps[:, 2]
    return 2.0 * a - 3.0 * b + 5.0 * c + 7.0


class TestOrderZeroIsIdentity:
    """order==0 只有一个常数自由度，没有任何模态需要滤波。"""

    def test_tet_order_0_is_identity(self):
        ref = _ref_cube_sps(0)
        F = build_tet_modal_filter(0, ref)
        assert np.allclose(F, np.eye(ref.shape[0]))

    def test_prism_order_0_is_identity(self):
        ref = _ref_cube_sps(0)
        F = build_prism_modal_filter(0, ref)
        assert np.allclose(F, np.eye(ref.shape[0]))


class TestConstantFieldPreserved:
    """常数场（eta=0 模态）在任意阶数下都必须严格不衰减——这是自由
    流场保持性的前提，本模块文档也以此为核心设计承诺之一。
    """

    def test_tet_constant_field_preserved_order_1(self):
        order = 1
        ref = _ref_cube_sps(order)
        F = build_tet_modal_filter(order, ref)
        field = np.full(ref.shape[0], 42.0)
        assert np.allclose(F @ field, field, atol=1e-9)

    def test_tet_constant_field_preserved_order_2(self):
        order = 2
        ref = _ref_cube_sps(order)
        F = build_tet_modal_filter(order, ref)
        field = np.full(ref.shape[0], 42.0)
        assert np.allclose(F @ field, field, atol=1e-9)

    def test_prism_constant_field_preserved_order_2(self):
        order = 2
        ref = _ref_cube_sps(order)
        F = build_prism_modal_filter(order, ref)
        field = np.full(ref.shape[0], -13.5)
        assert np.allclose(F @ field, field, atol=1e-9)


class TestTopModeStronglyDamped:
    """**legacy 档**：最高阶模态（max(i,j,k)==order）在每个阶数下都被
    压到机器精度量级。

    2026-08-29 的两次"减少滤波强度"尝试证明：在**全局逐 stage 施加**的
    前提下一旦削弱这一点（哪怕只放松"次高阶"模态），
    tests/validation/test_tgv.py 与 test_couette.py 会在数步内复现灾难性
    失稳。本组是那次教训的永久回归防线。

    **范围更正（2026-09-18）**：那条教训的前提是"全局施加"。默认档现在
    是**逐单元门控 + 顶模态有界衰减**（sensor + sigma_top=0.99），它不
    依赖"一次清零"来稳定——靠的是"只要传感器还在报就持续慢慢耗散"。
    Blasius 平板（唯一有精确解的粘性算例）实测：一次清零会把壁面剪应力
    压掉 2.3 倍（cf -74.5% vs -6.3%），有界衰减下 cf 与无滤波完全一致。
    所以本组现在显式跑在 `legacy` 档，默认档的矩阵性质见
    `test_modal_filter_order_loss.py::
    test_default_mode_matrix_is_bounded_damping_not_projection`。
    """

    def _top_mode_nodal_field(self, order, basis_fn, ref):
        from autoflowcfd.fr.collapsed_basis import tet_modal_basis_and_grad  # noqa: F401

        a, b, c = ref[:, 0], ref[:, 1], ref[:, 2]
        V, _, _, _ = basis_fn(a, b, c, order)
        n1d = order + 1
        top_flat = (n1d - 1) * n1d * n1d + (n1d - 1) * n1d + (n1d - 1)
        top_modal = np.zeros(V.shape[0])
        top_modal[top_flat] = 1.0
        return V @ top_modal

    def test_tet_top_mode_damped_order_1(self, legacy_mode):
        from autoflowcfd.fr.collapsed_basis import tet_modal_basis_and_grad

        order = 1
        ref = _ref_cube_sps(order)
        F = build_tet_modal_filter(order, ref)
        top_nodal = self._top_mode_nodal_field(order, tet_modal_basis_and_grad, ref)
        filtered = F @ top_nodal
        assert np.max(np.abs(filtered)) < 1e-8 * np.max(np.abs(top_nodal))

    def test_tet_top_mode_damped_order_2(self, legacy_mode):
        from autoflowcfd.fr.collapsed_basis import tet_modal_basis_and_grad

        order = 2
        ref = _ref_cube_sps(order)
        F = build_tet_modal_filter(order, ref)
        top_nodal = self._top_mode_nodal_field(order, tet_modal_basis_and_grad, ref)
        filtered = F @ top_nodal
        assert np.max(np.abs(filtered)) < 1e-8 * np.max(np.abs(top_nodal))


class TestKnownLimitationModesErodeOverManyIterations:
    """**legacy 档**的量化记录：P1 的线性模态在 60 个完整 SSP-RK3 步
    （=180 次 stage 滤波）后被磨灭到原幅值跨度的机器精度量级。

    这正是 cube_demo 真实复现里"单元内 8 个 SP 彼此只差 ~1e-14"的量化
    重现。

    **2026-09-18：这条限制已经被解决，不再是"暂不修复"。** 默认档改成
    逐单元门控 + 顶模态有界衰减（sigma_top=0.99）后：
      * 只有被传感器标记的单元才被施加（Blasius 上全域 0.2%、贴壁层
        7.5%），光滑区一个单元都不动；
      * 每次只衰减顶模态 1%，而物理残差每步都在重建梯度 —— 两者达到
        平衡的结果是壁面 du/dy 与**完全不滤波**逐位量级一致
        （1312.68 vs 1312.68）。
    本组保留为 legacy 档的历史记录（`legacy` 仍是唯一能复现历史结果的
    档），新默认档的对应性质见
    `test_modal_filter_order_loss.py::
    test_default_mode_intermediate_modes_are_essentially_untouched`。
    """

    def test_tet_order_1_linear_mode_erodes_after_many_iterations(self, legacy_mode):
        order = 1
        ref = _ref_cube_sps(order)
        F = build_tet_modal_filter(order, ref)
        field = _linear_field(ref)
        filtered = field.copy()
        for _ in range(3 * 60):  # 60 个完整 SSP-RK3 步
            filtered = F @ filtered
        original_spread = np.max(field) - np.min(field)
        filtered_spread = np.max(filtered) - np.min(filtered)
        # 记录已确认的磨灭幅度（不是期望值，是现状）：远小于原跨度
        assert filtered_spread < 1e-10 * original_spread

    def test_prism_order_1_linear_mode_erodes_after_many_iterations(self, legacy_mode):
        order = 1
        ref = _ref_cube_sps(order)
        F = build_prism_modal_filter(order, ref)
        field = _linear_field(ref)
        filtered = field.copy()
        for _ in range(3 * 60):
            filtered = F @ filtered
        original_spread = np.max(field) - np.min(field)
        filtered_spread = np.max(filtered) - np.min(filtered)
        assert filtered_spread < 1e-10 * original_spread
