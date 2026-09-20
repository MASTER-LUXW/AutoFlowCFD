"""模态滤波器传感器门控的纯数组内核与 k/omega 独立门控维度（2026-09-15）。

## 背景：两个维度此前被静默混在一起

`filter_scalar_field`（k/omega 滤波）直接用 `ops.filter_prism`、**完全不经过**
平均流那边的传感器门控。于是 `AFCFD_FILTER_MODE=sensor` 的真实语义一直是
"平均流门控 + k/omega 仍被完整清掉一整阶"。

79 万单元 cube_demo 真实网格同检查点 250 步对照决定性分离出了这一点：

    档       平均流        k/omega     om_max（起始 1.63e4）
    legacy   全局清零      全局清零    受控
    off      不滤波        **不滤波**  step100 达 1.65e5，增速持续加速
    sensor   门控(≈不滤波)  全局清零    step140 才 7.32e4，增速持续减速

off 与 sensor 的**平均流**轨迹几乎逐位相同（step100 残差都是 2.211e9、
Cd 3.0831 vs 3.0830），说明传感器在平均流上几乎不触发；两者 om_max 差一个
量级的原因**全部**在 k/omega 那一维。既然两维效果可以完全分离，就不能再让
一个环境变量同时决定它们——新增 `AFCFD_FILTER_TURB_GATE`（默认 "all"，与
此前行为逐位一致）。

## 顺带修掉的一处真实缺陷

第一版 `build_sensor_gated_filter_func` 靠"把当前 stage 的解塞进
`solver.state.U` 再调用 solver 版传感器"来取指示器。但 `state.Q` 是**普通
缓存数组**、只由 `_update_primitives()` 刷新，所以那样读到的是**步首**的
Q、不是当前 stage 的解。改成纯数组内核后直接读 `U[:,:,0]`——密度既是守恒量
第一分量也是原始量第一分量，正是想要的场，且不再有副作用。

## 本文件的判据

围绕"门控只改变对哪些单元施加、不改变矩阵本身"这条不变量：全 True 时必须
与全局版**逐位**一致，全 False 时必须**逐位**等于输入，中间情形只有被标记
的单元变化。
"""

import os

import numpy as np
import pytest

from autoflowcfd.core.fr_operators.artificial_viscosity import (
    compute_troubled_cell_mask,
)
from autoflowcfd.core.fr_solver.filter import (
    build_sensor_gated_filter_func_arrays,
    compute_turb_troubled_mask,
    filter_scalar_field,
    filter_scalar_field_gated,
    resolve_turb_filter_gate,
)
from autoflowcfd.fr.operators import generate_fr_operators


@pytest.fixture(scope="module")
def ops1():
    """order=1 的算子，**显式在 `sensor` 档下**构造。

    本文件测的是"门控只改变对哪些单元施加、不改变矩阵本身"这条不变量，
    所以需要一个**非恒等**的滤波矩阵。默认档已于 2026-09-19 改成 `off`
    （恒等滤波，依据见 `fr/modal_filter.py` 里"默认值 2026-09-19 改为
    off"那一节），在默认档下这些判据会退化成"什么都没变"而假通过 ——
    所以这里显式指定档位，不依赖默认值。
    """
    from tests.unit._filter_mode import (
        reload_filter_modules,
        restore_default_filter_modules,
    )

    _, ops_mod = reload_filter_modules(AFCFD_FILTER_MODE="sensor")
    try:
        yield ops_mod.generate_fr_operators(1)
    finally:
        restore_default_filter_modules()


#: order=1 native 四面体的真实自由度个数（`(p+1)(p+2)(p+3)/6`）。
#: 后 4 个槽位是零填充，按 `fr/native_padding.py` 的约定"初始化时
#: 复制真实 SP #0"、之后残差行填零、滤波行是单位阵——**永远冻结在初值**，
#: 不是自由度。往那里注入尖峰是构造不出物理场的，传感器也**必须**忽略它
#: （见 `TestNativeTetPaddingIsIgnored`）。
N_NATIVE_TET_P1 = 4


def _spiky(n_cells, n_sps, spike_cells, rng, sp_index=None):
    """光滑背景 + 指定单元内的 SP 级尖峰。

    背景刻意取"单元间缓变、单元内几乎常数"——那是已解析的物理，传感器
    不该命中；尖峰是单元内相邻解点间的数量级跳变，正是 2026-09-12 记录的
    omega 失控的形态。

    尖峰位置默认随机取**真实自由度**范围内的槽位（`< N_NATIVE_TET_P1`）：
    棱柱 8 个 SP 全是自由度，native 四面体只有前 4 个是，取前 4 个才能
    同时对两种单元类型都构造出真实的胞内跳变。
    """
    phi = 1.0 + 0.05 * np.arange(n_cells, dtype=float)[:, None] * np.ones(n_sps)
    hi = min(n_sps, N_NATIVE_TET_P1)
    for c in spike_cells:
        j = sp_index if sp_index is not None else int(rng.integers(0, hi))
        phi[c, j] *= 1.0e4
    return phi


class TestTroubledCellMask:
    def test_constant_field_flags_nothing(self):
        phi = np.full((8, 8), 3.7)
        assert not compute_troubled_cell_mask(phi, 1, n_prism=4).any()

    def test_cellwise_smooth_field_flags_nothing(self):
        """单元间缓变、单元内常数：已解析的物理，不该被当成欠分辨。"""
        phi = 1.0 + 0.05 * np.arange(8, dtype=float)[:, None] * np.ones(8)
        assert not compute_troubled_cell_mask(phi, 1, n_prism=4).any()

    def test_sp_level_spike_flags_exactly_those_cells(self):
        rng = np.random.default_rng(7)
        spikes = [1, 5, 6]
        phi = _spiky(8, 8, spikes, rng)
        mask = compute_troubled_cell_mask(phi, 1, n_prism=4)
        np.testing.assert_array_equal(
            np.flatnonzero(mask), np.array(spikes),
            err_msg="被标记的单元与注入尖峰的单元不一致")

    def test_order_zero_flags_nothing(self):
        """order==0 没有可截断的最高阶模态——与传感器 order==0 短路一致。"""
        rng = np.random.default_rng(7)
        phi = _spiky(8, 1, [1, 5], rng)
        assert not compute_troubled_cell_mask(phi, 0, n_prism=4).any()

    def test_both_orderings_agree_on_prism_first_layout(self):
        """`n_prism` 与等价的 `cell_is_prism` 必须给出同一个掩码。

        分布式 local 排列里棱柱/四面体交错，只能用后者；这条保证两条
        表达方式在它们重合的情形下不会分叉。
        """
        rng = np.random.default_rng(11)
        phi = _spiky(10, 8, [2, 7, 9], rng)
        n_prism = 4
        cip = np.arange(10) < n_prism
        np.testing.assert_array_equal(
            compute_troubled_cell_mask(phi, 1, n_prism=n_prism),
            compute_troubled_cell_mask(phi, 1, cell_is_prism=cip))

    def test_scale_invariance(self):
        """Persson-Peraire 的 S_e 是能量比值，对场的整体缩放不变。

        这条是"平均流可以直接用守恒密度、不必先反算原始密度"的依据。
        """
        rng = np.random.default_rng(3)
        phi = _spiky(8, 8, [1, 4], rng)
        base = compute_troubled_cell_mask(phi, 1, n_prism=4)
        for s in (1e-6, 3.7, 1e5):
            np.testing.assert_array_equal(
                compute_troubled_cell_mask(phi * s, 1, n_prism=4), base,
                err_msg=f"缩放 {s} 改变了判据")

    @pytest.mark.parametrize("kwargs", [
        {},
        {"n_prism": 4, "cell_is_prism": np.arange(8) < 4},
    ])
    def test_requires_exactly_one_partition_spec(self, kwargs):
        """都不给或都给都要报错——那是调用方对索引空间没有明确认知的信号。"""
        phi = np.ones((8, 8))
        with pytest.raises(ValueError, match="恰好一个|只能给一个"):
            compute_troubled_cell_mask(phi, 1, **kwargs)

    def test_cell_is_prism_shape_is_validated(self):
        phi = np.ones((8, 8))
        with pytest.raises(ValueError, match="形状"):
            compute_troubled_cell_mask(phi, 1, cell_is_prism=np.ones(5, bool))


class TestNativeTetPaddingIsIgnored:
    """native 四面体的零填充槽位必须被传感器完全忽略（2026-09-15）。

    这是排查中发现的一处**真实缺陷**：传感器的 `cell_type="tet"` 分支一直
    用 `collapsed_basis.py::tet_modal_basis_and_grad`（坍缩坐标族，
    `(order+1)^3` 个模态），而坍缩坐标四面体基已于 2026-09-03 整体删除、
    四面体现在唯一实现是 native PKD/Dubiner 基（`i+j+k<=order`）。于是
    传感器既用了错的基，又把冻结的填充槽位当成真实自由度喂进了指标——
    实测推进 10 步后填充块与真实 SP#0 已相差 3.4%，指标里混进一个纯人造
    的阶跃。现在四面体走 `compute_persson_peraire_sensor_native_tet`。
    """

    def test_spike_in_padded_slot_is_ignored(self):
        rng = np.random.default_rng(41)
        phi = _spiky(6, 8, [], rng)
        phi[2, 7] *= 1.0e6          # 填充槽位（order=1 下索引 4..7）
        phi[4, 5] *= 1.0e6
        # n_prism=0 -> 全部按四面体处理
        assert not compute_troubled_cell_mask(phi, 1, n_prism=0).any(), (
            "填充槽位里的值影响了判据——那些槽位不是自由度，冻结在初值，"
            "混进指标会随推进步数产生一个纯人造的阶跃")

    def test_spike_in_real_dof_is_still_caught(self):
        rng = np.random.default_rng(41)
        phi = _spiky(6, 8, [1, 3], rng, sp_index=2)   # 索引 2 < 4，真实自由度
        mask = compute_troubled_cell_mask(phi, 1, n_prism=0)
        np.testing.assert_array_equal(np.flatnonzero(mask), np.array([1, 3]))

    def test_native_sensor_uses_only_real_dofs(self):
        """直接判据：改动填充列不改变 s_e，改动真实列会改变。"""
        from autoflowcfd.core.fr_operators.artificial_viscosity import (
            compute_persson_peraire_sensor_native_tet,
        )
        base = np.full((3, 8), 2.5)
        s0 = compute_persson_peraire_sensor_native_tet(base, 1)
        pad = base.copy(); pad[:, 4:] = 1.0e9
        np.testing.assert_allclose(
            compute_persson_peraire_sensor_native_tet(pad, 1), s0, rtol=0, atol=0)
        real = base.copy(); real[:, 1] = 1.0e9
        assert np.all(compute_persson_peraire_sensor_native_tet(real, 1) > s0 + 10.0)

    def test_constant_field_energy_ratio_is_machine_zero(self):
        from autoflowcfd.core.fr_operators.artificial_viscosity import (
            compute_persson_peraire_sensor_native_tet,
        )
        s_e = compute_persson_peraire_sensor_native_tet(np.full((3, 8), 7.0), 1)
        assert np.all(s_e < -20.0), f"常数场 s_e={s_e}，应为机器零量级"

    @pytest.mark.parametrize("order,n_native,n_top", [(1, 4, 3), (2, 10, 6), (3, 20, 10)])
    def test_native_mode_counts(self, order, n_native, n_top):
        """`n_native=(p+1)(p+2)(p+3)/6`，顶模态是 `i+j+k==order` 的那些。

        钉住这两个数：任何人把判据改回张量积族的 `max(i,j,k)`、或者
        把模态集合改成 `(order+1)^3`，这里会立刻失败。
        """
        from autoflowcfd.core.fr_operators.artificial_viscosity.sensor_operators import (
            _build_native_tet_sensor_operators,
        )
        V_inv, top_mask, got_n = _build_native_tet_sensor_operators(order)
        assert got_n == n_native == (order + 1) * (order + 2) * (order + 3) // 6
        assert V_inv.shape == (n_native, n_native)
        assert int(top_mask.sum()) == n_top

    def test_order_zero_returns_neg_inf(self):
        from autoflowcfd.core.fr_operators.artificial_viscosity import (
            compute_persson_peraire_sensor_native_tet,
        )
        s_e = compute_persson_peraire_sensor_native_tet(np.full((3, 1), 2.0), 0)
        assert np.all(np.isneginf(s_e))


class TestGatedScalarFilterMatchesGlobalWhenUngated:
    """门控只改变"对哪些单元施加"，不改变矩阵本身。"""

    def test_all_troubled_is_bit_identical_to_global(self, ops1):
        rng = np.random.default_rng(5)
        phi = rng.standard_normal((10, 8)) + 5.0
        n_prism = 4
        got = filter_scalar_field_gated(
            phi, ops1.filter_prism, ops1.filter_tet,
            np.ones(10, dtype=bool), n_prism=n_prism)
        want = filter_scalar_field(phi, n_prism, ops1.filter_prism, ops1.filter_tet)
        np.testing.assert_allclose(got, want, rtol=0, atol=0)

    def test_none_troubled_is_bit_identical_to_input(self, ops1):
        rng = np.random.default_rng(5)
        phi = rng.standard_normal((10, 8)) + 5.0
        got = filter_scalar_field_gated(
            phi, ops1.filter_prism, ops1.filter_tet,
            np.zeros(10, dtype=bool), n_prism=4)
        np.testing.assert_allclose(got, phi, rtol=0, atol=0)

    def test_only_flagged_cells_change(self, ops1):
        rng = np.random.default_rng(5)
        phi = rng.standard_normal((10, 8)) + 5.0
        troubled = np.zeros(10, dtype=bool)
        troubled[[1, 7]] = True
        got = filter_scalar_field_gated(
            phi, ops1.filter_prism, ops1.filter_tet, troubled, n_prism=4)
        changed = np.flatnonzero(np.abs(got - phi).max(axis=1) > 0)
        np.testing.assert_array_equal(changed, np.array([1, 7]))

    def test_does_not_mutate_input(self, ops1):
        rng = np.random.default_rng(5)
        phi = rng.standard_normal((10, 8)) + 5.0
        before = phi.copy()
        filter_scalar_field_gated(
            phi, ops1.filter_prism, ops1.filter_tet,
            np.ones(10, dtype=bool), n_prism=4)
        np.testing.assert_allclose(phi, before, rtol=0, atol=0)

    def test_interleaved_layout_matches_equivalent_prism_first(self, ops1):
        rng = np.random.default_rng(9)
        phi = rng.standard_normal((10, 8)) + 5.0
        troubled = np.zeros(10, dtype=bool)
        troubled[[0, 3, 8]] = True
        a = filter_scalar_field_gated(
            phi, ops1.filter_prism, ops1.filter_tet, troubled, n_prism=4)
        b = filter_scalar_field_gated(
            phi, ops1.filter_prism, ops1.filter_tet, troubled,
            cell_is_prism=(np.arange(10) < 4))
        np.testing.assert_allclose(a, b, rtol=0, atol=0)


class TestTurbTroubledMaskIsUnionOfKAndOmega:
    """k 与 omega 各自都会混叠，任一出问题该单元就需要滤波 —— 取并集。

    为什么不复用平均流的掩码：实测 off/sensor 两档平均流轨迹几乎逐位相同、
    om_max 却差一个量级，就是"密度光滑而 omega 有尖峰"的直接证据。
    """

    def test_spike_only_in_k_is_flagged(self):
        rng = np.random.default_rng(21)
        k = _spiky(8, 8, [2], rng)
        om = 1.0 + 0.05 * np.arange(8, dtype=float)[:, None] * np.ones(8)
        mask = compute_turb_troubled_mask(k, om, 1, n_prism=4)
        np.testing.assert_array_equal(np.flatnonzero(mask), np.array([2]))

    def test_spike_only_in_omega_is_flagged(self):
        rng = np.random.default_rng(22)
        k = 1.0 + 0.05 * np.arange(8, dtype=float)[:, None] * np.ones(8)
        om = _spiky(8, 8, [5], rng)
        mask = compute_turb_troubled_mask(k, om, 1, n_prism=4)
        np.testing.assert_array_equal(np.flatnonzero(mask), np.array([5]))

    def test_union_not_intersection(self):
        rng = np.random.default_rng(23)
        k = _spiky(8, 8, [1], rng)
        om = _spiky(8, 8, [6], rng)
        mask = compute_turb_troubled_mask(k, om, 1, n_prism=4)
        np.testing.assert_array_equal(np.flatnonzero(mask), np.array([1, 6]))


class TestTurbGateResolution:
    def _with_env(self, value):
        old = os.environ.get("AFCFD_FILTER_TURB_GATE")
        if value is None:
            os.environ.pop("AFCFD_FILTER_TURB_GATE", None)
        else:
            os.environ["AFCFD_FILTER_TURB_GATE"] = value
        return old

    def _restore(self, old):
        if old is None:
            os.environ.pop("AFCFD_FILTER_TURB_GATE", None)
        else:
            os.environ["AFCFD_FILTER_TURB_GATE"] = old

    def test_default_is_all(self):
        old = self._with_env(None)
        try:
            assert resolve_turb_filter_gate() == "all"
        finally:
            self._restore(old)

    @pytest.mark.parametrize("value,expected", [
        ("all", "all"), ("sensor", "sensor"),
        ("ALL", "all"), ("Sensor", "sensor"),
    ])
    def test_accepted_values(self, value, expected):
        old = self._with_env(value)
        try:
            assert resolve_turb_filter_gate() == expected
        finally:
            self._restore(old)

    @pytest.mark.parametrize("value", ["legacy", "off", "mild", "", "yes"])
    def test_rejects_unknown_values(self, value):
        """不能静默退回默认——那正是本轮要消除的那类静默行为。"""
        old = self._with_env(value)
        try:
            with pytest.raises(ValueError, match="AFCFD_FILTER_TURB_GATE"):
                resolve_turb_filter_gate()
        finally:
            self._restore(old)

    def test_read_at_call_time_not_import_time(self):
        """运行期重读：这一维不影响算子构造，所以不需要 reload 模块。"""
        old = self._with_env("all")
        try:
            assert resolve_turb_filter_gate() == "all"
            os.environ["AFCFD_FILTER_TURB_GATE"] = "sensor"
            assert resolve_turb_filter_gate() == "sensor"
        finally:
            self._restore(old)


class TestMeanFlowGatedBuilder:
    def test_smooth_state_is_returned_untouched(self, ops1):
        """没有单元被标记时必须原样返回（连 reshape 都不该改变内容）。"""
        n_cells, n_sps, n_var = 10, 8, 5
        U = np.zeros((n_cells, n_sps, n_var))
        U[..., 0] = 1.0 + 0.05 * np.arange(n_cells)[:, None]
        U[..., 4] = 2.5e5
        f = build_sensor_gated_filter_func_arrays(
            n_cells, n_sps, 1, ops1.filter_prism, ops1.filter_tet, n_prism=4)
        flat = U.reshape(n_cells * n_sps, n_var)
        np.testing.assert_allclose(f(flat.copy()), flat, rtol=0, atol=0)

    def test_only_spiky_cells_are_filtered(self, ops1):
        rng = np.random.default_rng(31)
        n_cells, n_sps, n_var = 10, 8, 5
        U = np.zeros((n_cells, n_sps, n_var))
        U[..., 0] = _spiky(n_cells, n_sps, [3, 8], rng)
        U[..., 1] = 30.0
        U[..., 4] = 2.5e5
        f = build_sensor_gated_filter_func_arrays(
            n_cells, n_sps, 1, ops1.filter_prism, ops1.filter_tet, n_prism=4)
        flat = U.reshape(n_cells * n_sps, n_var)
        out = f(flat.copy()).reshape(n_cells, n_sps, n_var)
        changed = np.flatnonzero(np.abs(out - U).reshape(n_cells, -1).max(axis=1) > 0)
        np.testing.assert_array_equal(changed, np.array([3, 8]))

    def test_requires_exactly_one_partition_spec(self, ops1):
        with pytest.raises(ValueError, match="只能给一个"):
            build_sensor_gated_filter_func_arrays(
                10, 8, 1, ops1.filter_prism, ops1.filter_tet)
