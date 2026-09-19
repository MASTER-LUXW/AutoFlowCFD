"""
native 四面体过积分的**细网格轴不再填充**——纯速度收益，结果不变。

## 缺口

`pad_native_matrix_to_global` 的"零填充块对角"不变量保证填充槽位恒为
零、对结果零贡献。但此前三个过积分算子的**细网格轴**也被填到了与棱柱
共用的 `(over_order+1)^3` 宽度，而 native 四面体在 `over_order` 下只有
`(oo+1)(oo+2)(oo+3)/6` 个**真实**细点：

    P1 (oo=2)   真实 10   填充 27
    P2 (oo=3)   真实 20   填充 64

于是整条过积分链（插值 -> 物理通量 -> 逆变通量 -> 散度 -> 限制）都在这些
恒零的空点上白算，其中 `D_fine` 的收缩是 **O(n_fine^2)**：P2 上是
`64^2/20^2 = 10.2` 倍的无效 FLOPs。四面体在本项目两张真实 ANSA 网格里
占约 83% 的单元。

实测（20000 个四面体单元跑完整链路，与填充版逐项对比）：

    P1  加速 3.04x   最大相对差 1.385e-16（舍入）
    P2  加速 4.63x   最大相对差 0.000e+00（完全相同）

**粗网格轴仍然必须填充**：`Q` 是 `(n_cells, n_sps_global, 5)` 的填充布局
（native 四面体的真实自由度只占前 n_native 个槽位），所以 interp 的**列**、
restrict 的**行**必须是 n_sps_global 宽。只有细网格轴是过积分链内部自己的
中间维度，可以取真实长度。

## 本文件覆盖

1. 三个算子的形状（细轴真实长度、粗轴填充宽度）
2. **等价性**：不填充与显式填充跑完整链路结果相同（这是正确性的核心）
3. 共享上下文 helper 的新契约：每段自带 n_fine 与已切好的度量
4. 四面体段度量"切前 n_fine_tet 列"为什么恒等于真实细点上的度量
"""

import numpy as np
import pytest

from autoflowcfd.core.fr_operators.flux_kernels import euler_physical_flux_batch
from autoflowcfd.core.fr_operators.volume_contract import (
    contract_shared_operator_1axis,
    contract_shared_operator_2axis,
    contravariant_flux_from_metric,
)
from autoflowcfd.fr.native_tet_overintegration import (
    build_native_tet_overintegration_operators,
)
from autoflowcfd.fr.native_padding import pad_native_matrix_to_global
from autoflowcfd.fr.operators import generate_fr_operators


def _n_fine_native(oo):
    return (oo + 1) * (oo + 2) * (oo + 3) // 6


def _over_order(order):
    """棱柱的过积分阶数（受坍缩基条件数上限约束）。"""
    from autoflowcfd.fr.collapsed_basis import (
        OVERINTEGRATION_MAX_ORDER,
        resolve_overintegration_order_rule,
    )
    return min(resolve_overintegration_order_rule() * order,
               OVERINTEGRATION_MAX_ORDER)


def _over_order_tet(order):
    """四面体的过积分阶数——**与棱柱不同**（2026-09-17 起解耦）。

    native PKD 基不受坍缩基条件数上限约束，实际阶数由
    `resolve_tet_overintegration_order` 按"经验法则 / 四面体自己的上限 /
    与棱柱共用的 jacobians_fine 布局宽度"三重约束定。P2 因此是 4（不是 3），
    P3 是 5（理想 6 被布局夹住）。
    """
    from autoflowcfd.fr.native_tet_overintegration import (
        resolve_tet_overintegration_order,
    )
    return resolve_tet_overintegration_order(order, (_over_order(order) + 1) ** 3)


class TestOperatorShapes:
    """细轴取真实长度、粗轴保持填充宽度。"""

    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_shapes(self, order):
        ops = generate_fr_operators(order)
        oo = _over_order_tet(order)
        nf = _n_fine_native(oo)
        ns = (order + 1) ** 3

        assert ops.overint_order_tet == oo, (
            f"P{order}: 四面体过积分阶数 {ops.overint_order_tet} != {oo}")

        assert ops.overint_D_fine_tet.shape == (nf, nf, 3), (
            f"P{order}: D_fine_tet 形状 {ops.overint_D_fine_tet.shape} != "
            f"({nf},{nf},3) —— 细网格轴不该被填充")
        assert ops.overint_interp_c2f_tet.shape == (nf, ns), (
            f"P{order}: interp c2f 形状应为 (n_fine_native, n_sps_global)")
        assert ops.overint_restrict_f2c_tet.shape == (ns, nf), (
            f"P{order}: restrict f2c 形状应为 (n_sps_global, n_fine_native)")

    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_declared_count_matches_shape(self, order):
        ops = generate_fr_operators(order)
        assert ops.overint_n_fine_tet == ops.overint_D_fine_tet.shape[0]

    @pytest.mark.parametrize("order", [1, 2])
    def test_fine_axis_is_strictly_narrower_than_padded(self, order):
        """真实细点数必须**严格小于**填充宽度——否则这项改动没有收益，
        说明填充宽度的定义变了，本文件的前提需要复核。"""
        oo = _over_order(order)
        assert _n_fine_native(oo) < (oo + 1) ** 3


class TestEquivalenceWithPaddedVersion:
    """**核心正确性**：不填充与显式填充跑完整过积分链结果相同。

    这条是整项改动能成立的唯一依据——填充槽位恒为零所以"零贡献"是
    `pad_native_matrix_to_global` 的设计不变量，但不变量要被真实
    验证过才算。
    """

    @pytest.mark.parametrize("order", [1, 2])
    def test_full_chain_matches(self, order):
        oo = _over_order(order)
        n_sps_g = (order + 1) ** 3
        n_fine_g = (oo + 1) ** 3
        ref_f, c2f_n, Df_n, f2c_n = build_native_tet_overintegration_operators(
            order, oo)
        nf = ref_f.shape[0]
        n_nat = c2f_n.shape[1]

        # 显式填充版（改动前的构造方式）
        Df_pad = pad_native_matrix_to_global(Df_n, n_fine_g, pad_axes=(0, 1))
        c2f_pad = pad_native_matrix_to_global(
            pad_native_matrix_to_global(c2f_n, n_sps_g, pad_axes=(1,)),
            n_fine_g, pad_axes=(0,))
        f2c_pad = pad_native_matrix_to_global(
            pad_native_matrix_to_global(f2c_n, n_fine_g, pad_axes=(1,)),
            n_sps_g, pad_axes=(0,))
        # 只填粗轴（改动后）
        c2f_u = pad_native_matrix_to_global(c2f_n, n_sps_g, pad_axes=(1,))
        f2c_u = pad_native_matrix_to_global(f2c_n, n_sps_g, pad_axes=(0,))

        rng = np.random.default_rng(20260917)
        NC = 64
        Q = np.zeros((NC, n_sps_g, 5))
        Q[:, :n_nat, 0] = 1.225 * (1 + 0.05 * rng.standard_normal((NC, n_nat)))
        Q[:, :n_nat, 1] = 30.0 * rng.standard_normal((NC, n_nat))
        Q[:, :n_nat, 4] = 101325.0 / 0.4 * (
            1 + 0.02 * rng.standard_normal((NC, n_nat)))
        det = np.abs(rng.standard_normal(NC)) * 1e-3 + 1e-3
        inv = np.tile(np.eye(3), (NC, 1, 1)) * (
            1.0 + 0.1 * rng.standard_normal((NC, 1, 1)))

        def chain(c2f, Df, f2c, n_fine):
            dj = np.ascontiguousarray(
                np.broadcast_to(det[:, None], (NC, n_fine)))
            ij = np.ascontiguousarray(
                np.broadcast_to(inv[:, None, :, :], (NC, n_fine, 3, 3)))
            Qf = contract_shared_operator_1axis(c2f, Q)
            F = euler_physical_flux_batch(Qf.reshape(-1, 5)).reshape(
                NC, n_fine, 3, 5)
            Ft = contravariant_flux_from_metric(dj, ij, F)
            dv = contract_shared_operator_2axis(Df, Ft)
            return contract_shared_operator_1axis(f2c, dv)

        a = chain(c2f_pad, Df_pad, f2c_pad, n_fine_g)
        b = chain(c2f_u, Df_n, f2c_u, nf)
        rel = np.abs(a - b).max() / max(np.abs(a).max(), 1e-300)
        assert rel < 1e-13, (
            f"P{order}: 去掉细轴填充改变了结果，最大相对差 {rel:.3e} "
            f"—— 填充槽位本该恒为零、零贡献")

    @pytest.mark.parametrize("order", [1, 2])
    def test_padded_slots_are_zero(self, order):
        """直接验证不变量本身：真实块之外全为零。"""
        oo = _over_order(order)
        ops = generate_fr_operators(order)
        n_nat_sps = (order + 1) * (order + 2) * (order + 3) // 6
        # interp 的列（粗轴）在 n_nat_sps 之后必须全零
        assert np.all(ops.overint_interp_c2f_tet[:, n_nat_sps:] == 0.0)
        # restrict 的行（粗轴）同理
        assert np.all(ops.overint_restrict_f2c_tet[n_nat_sps:, :] == 0.0)


class TestContextContract:
    """共享上下文 helper 的新契约。"""

    def _fake(self, order, n_prism, n_tet):
        from types import SimpleNamespace

        ops = generate_fr_operators(order)
        n_fine_prism = (_over_order(order) + 1) ** 3
        n_cells = n_prism + n_tet
        rng = np.random.default_rng(3)
        det = rng.random(n_cells * n_fine_prism) + 1.0
        inv = rng.random((n_cells * n_fine_prism, 3, 3))
        # 四面体段：把每个单元的全部细点槽位填成**同一个常数**，复现
        # `compute_native_tet_jacobians` 的真实行为（直边四面体的 Jacobian
        # 逐单元为常数）
        det = det.reshape(n_cells, n_fine_prism)
        inv = inv.reshape(n_cells, n_fine_prism, 3, 3)
        det[n_prism:] = det[n_prism:, :1]
        inv[n_prism:] = inv[n_prism:, :1]
        mesh = SimpleNamespace(
            n_cells=n_cells, n_prism_cells=n_prism,
            n_sps_per_cell_fine=n_fine_prism,
            jacobians_fine={"det_jacs": det.ravel(),
                            "inv_jacs": inv.reshape(-1, 3, 3)},
        )
        return mesh, ops, det, inv

    @pytest.mark.parametrize("order", [1, 2])
    def test_segments_carry_own_n_fine(self, order):
        from autoflowcfd.core.fr_operators.volume_contract import (
            get_overintegration_context,
        )

        mesh, ops, _det, _inv = self._fake(order, 5, 7)
        oi = get_overintegration_context(mesh, ops)
        assert oi is not None
        assert "n_fine" not in oi and "det_fine" not in oi, (
            "共享键必须已移除——保留它们会让漏改的消费点静默用棱柱的 "
            "n_fine 去切四面体段")
        (p_lo, p_hi, p_nf, p_det, p_inv, *_), \
            (t_lo, t_hi, t_nf, t_det, t_inv, *_) = oi["segs"]
        assert (p_lo, p_hi) == (0, 5) and (t_lo, t_hi) == (5, 12)
        assert p_nf == (_over_order(order) + 1) ** 3
        assert t_nf == ops.overint_D_fine_tet.shape[0]
        assert p_det.shape == (5, p_nf) and p_inv.shape == (5, p_nf, 3, 3)
        assert t_det.shape == (7, t_nf) and t_inv.shape == (7, t_nf, 3, 3)

    @pytest.mark.parametrize("order", [1, 2])
    def test_tet_metric_is_the_broadcast_per_cell_constant(self, order):
        """四面体段的度量必须恒等于该单元的常数度量。

        实现是"取第 0 列广播"（2026-09-17 第二次改动；第一版是"切前
        n_fine_tet 列"，两者在直边四面体上等价，但切列额外要求
        `n_fine_tet <= n_fine_prism`，把 P3 从理想的 oo=6 夹到 5）。
        """
        from autoflowcfd.core.fr_operators.volume_contract import (
            get_overintegration_context,
        )

        mesh, ops, det, inv = self._fake(order, 5, 7)
        oi = get_overintegration_context(mesh, ops)
        _, (_, _, t_nf, t_det, t_inv, *_) = oi["segs"]
        for i in range(7):
            assert np.all(t_det[i] == det[5 + i, 0])
            assert np.all(t_inv[i] == inv[5 + i, 0])

    def test_tet_fine_points_may_exceed_the_prism_layout_width(self):
        """四面体细点数**可以**超过棱柱布局宽度——那条约束已被移除。

        它曾经是个真实约束（度量靠"切前 n_fine_tet 列"），并且把 P3 的
        过积分阶数从理想的 6（84 个细点）夹到 5（56），而去混叠误差在
        `oo = 2*order` 处断崖式下降：P3 oo=5 是 3.37e-3、oo=6 是 4.80e-6。
        改成"第 0 列广播"后宽度不再相关。这条测试把**移除**钉住，避免有人
        因为"看起来越界"又把限制加回去。

        用 **P3 的真实生产组合**来测（2026-09-19 改）：坍缩棱柱 oo=3 ->
        64 个细点，native 四面体 oo=6 -> 84 个，天然就是 84 > 64。
        此前这里是"把 `n_sps_per_cell_fine` 人为设成比四面体细点数还小
        一格"的合成 mesh，那个组合在生产里不可能出现 —— 而棱柱段现在
        也会校验"算子 n_fine == 布局宽度"（原生/坍缩两档下都是恒等式），
        合成值直接撞上那道闸。用真实组合既保住了原意，又不再依赖一个
        构造不出来的状态。
        """
        from types import SimpleNamespace

        from autoflowcfd.core.fr_operators.volume_contract import (
            get_overintegration_context,
        )

        order = 3
        ops = generate_fr_operators(order)
        n_fine_tet = ops.overint_D_fine_tet.shape[0]
        n_fine_prism = ops.overint_D_fine_prism.shape[0]
        assert n_fine_tet > n_fine_prism, (
            f"P{order} 下四面体细点数 {n_fine_tet} 应当超过棱柱的 "
            f"{n_fine_prism} —— 这条测试的前提就是这个不等式")

        n_cells, n_prism, n_tet = 4, 2, 2
        det = np.arange(
            1.0, n_cells * n_fine_prism + 1.0).reshape(n_cells, n_fine_prism)
        inv = (np.tile(np.eye(3), (n_cells, n_fine_prism, 1, 1))
               * det[:, :, None, None])
        # 四面体段按真实行为填成逐单元常数
        det[n_prism:] = det[n_prism:, :1]
        inv[n_prism:] = inv[n_prism:, :1]
        mesh = SimpleNamespace(
            n_cells=n_cells, n_prism_cells=n_prism,
            n_sps_per_cell_fine=n_fine_prism,
            jacobians_fine={"det_jacs": det.ravel(),
                            "inv_jacs": inv.reshape(-1, 3, 3)},
        )
        oi = get_overintegration_context(mesh, ops)
        assert oi is not None
        _, (_, _, t_nf, t_det, t_inv, *_) = oi["segs"]
        assert t_nf == n_fine_tet > n_fine_prism
        assert t_det.shape == (n_tet, n_fine_tet)
        assert t_inv.shape == (n_tet, n_fine_tet, 3, 3)
        for i in range(n_tet):
            assert np.all(t_det[i] == det[n_prism + i, 0])
            assert np.all(t_inv[i] == inv[n_prism + i, 0])

    def test_prism_layout_width_mismatch_is_rejected(self):
        """棱柱段的"算子 n_fine == 布局宽度"必须是**硬失败**。

        两者在两档下都是恒等式（坍缩 `(oo+1)^3`、原生
        `(oo+1)^2(oo+2)/2`，见 `fr/overintegration_order.prism_n_fine`）。
        不相等只可能是算子与网格几何用了不同的 over_order 或不同的棱柱
        基档 —— 那会静默切出一个"看起来合法"的错误度量数组，所以这里
        必须报错而不是取其中一个。

        （历史：这两处的一致性此前只靠两个文件各自的注释互相提醒"必须
        逐字一致"，阶数公式在 `fr/operators.py` 与
        `high_order_mesh_order.py` 里各写一遍。现在公式合并到
        `overintegration_order.py` 唯一入口，这条是运行期的第二道闸。）
        """
        from types import SimpleNamespace

        from autoflowcfd.core.fr_operators.volume_contract import (
            get_overintegration_context,
        )

        ops = generate_fr_operators(2)
        n_fine_prism = ops.overint_D_fine_prism.shape[0]
        wrong = n_fine_prism - 1
        n_cells = 4
        det = np.ones((n_cells, wrong))
        inv = np.tile(np.eye(3), (n_cells, wrong, 1, 1))
        mesh = SimpleNamespace(
            n_cells=n_cells, n_prism_cells=2,
            n_sps_per_cell_fine=wrong,
            jacobians_fine={"det_jacs": det.ravel(),
                            "inv_jacs": inv.reshape(-1, 3, 3)},
        )
        with pytest.raises(ValueError, match="棱柱过积分细点数不一致"):
            get_overintegration_context(mesh, ops)


class TestAllConsumersUsePerSegmentMetric:
    """四个消费点（CPU 三处 + GPU 四处）都必须按段取度量。

    漏改任何一处会静默用棱柱的 n_fine 去切四面体段——形状不匹配还算好，
    更糟的是切出一个"看起来合法"的错误数组。
    """

    _FILES = [
        "src/autoflowcfd/core/fr_residual/inviscid.py",
        "src/autoflowcfd/core/fr_residual/viscous_flux.py",
        "src/autoflowcfd/core/turbulence/transport.py",
        "src/autoflowcfd/core/gpu/residual/gpu_inviscid_volume.py",
        "src/autoflowcfd/core/gpu/residual/gpu_viscous.py",
        "src/autoflowcfd/core/gpu/turbulence/gpu_scalar_transport.py",
    ]

    @pytest.mark.parametrize("rel", _FILES)
    def test_no_shared_fine_metric_indexing_left(self, rel):
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[2]
        src = (root / rel).read_text(encoding="utf-8")
        for banned in ('oi["n_fine"]', 'oi["det_fine"]', 'oi["inv_fine"]',
                       "adj_j_fine[lo:hi]",
                       "det_jacs_fine[c0:c1]", "inv_jacs_fine[c0:c1]"):
            assert banned not in src, (
                f"{rel} 里仍有共享细点度量的旧取法 `{banned}`")
