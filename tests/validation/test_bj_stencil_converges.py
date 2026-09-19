"""BJ 越界判据的邻域模板：面邻居**不随加密收敛**，顶点邻居收敛。

## 这条测试回答什么

项目问题清单里长期记着一条"BJ 判据在欠解析光滑场上标记 100%"，指的是
TGV（三重周期、纯四面体、P2）。它被当成"这条判据在生产上不可用"的依据。
2026-09-19 做网格加密扫描，把它变成了一条**可判定**的结论：

    n   单元数   h       模板     越界max/尺度  越界中位/尺度  默认标记%
    4      384  0.864   面        6.232e-01     4.457e-01     100.00%
    4      384  0.864   顶点      6.232e-01     3.732e-01     100.00%
    8     3072  0.432   面        3.794e-01     1.831e-01     100.00%
    8     3072  0.432   顶点      2.089e-01     **0.0**        25.00%
   16    24576  0.216   面        1.950e-01     8.038e-02     100.00%
   16    24576  0.216   顶点      8.576e-02     **0.0**         6.18%

**面模板的标记比例在三档加密上恒为 100% —— 它不收敛。** 顶点模板
100% -> 25% -> 6.18%，大致 ∝ h^2（单元数 x4、标记比例 /4）。

（n=4 是每波长 4 个单元，两种模板都标记全部单元。那一档**本来就该被
标记** —— 解在那个分辨率上确实没被解析，不是误判。判据要求的是"随着
分辨率提高，误判要消失"，这正是上表的第 3 列在说的事。）

## 机理

BJ 的包络是"本单元与其邻居的**单元均值**"张成的区间。一个四面体只有
4 个面邻居，它们在三维里不能把本单元夹住，于是 O(h |grad u|) 的**合法
光滑变化**落在包络之外。换成顶点邻域（共享任一顶点的全部单元）之后包络
才真正包住本单元。顶点（node-based）模板本来就是非结构限制器里
Barth-Jespersen 的常见实现形式，理由正是这一条。

越界量的观测阶数（`rho_u`，相邻两档比值）：

    面      p(max) = 0.716 / 0.960     （即 O(h^1)）
    顶点    p(max) = 1.577 / 1.284

## 两条被自己的数据否掉的方案（如实记录）

1. **经典 TVB（Cockburn-Shu 的 `M h^2` 容差）修不了它。** 面模板下越界
   量是 O(h^1)，而 `M h^2` 在 h 小时衰减更快 —— 细网格上判据仍会标记
   全部单元。这条方案此前被评估为"工业界的答案"，是加密扫描把它否掉的。
2. **"100% 来自 BJ 统计了零填充槽位"。** TGV P2 纯四面体下 63% 的槽位
   是填充，看起来很像。**证伪**：n=8 上"只统计真实槽位"标记 100.00%、
   "含填充"98.96%，只差 32/3072 = 1.04% 的单元（n=4 上 0 个）。也就是
   填充**不是** 100% 的成因 —— 成因是模板。

   （但填充**不是完全中性**的，那 1.04% 是真的；`row_is_prism`/
   `n_real_*` 那两个排除参数仍然必要，依据是另一个测量："TGV P2 推进
   30 步后填充值已越出真实槽位区间 14% 的区间宽"。第一版把这条写成
   "完全相同"，实测不成立，已更正。）

   为什么填充的影响这么小：原生基的填充槽位**不是零** ——
   `build_order_geometry` 把填充 SP 的坐标设成该单元 SP#0 的坐标，所以
   那里的解是 SP#0 的重复值、落在单元自身区间内。

## 一条方法论教训

第一版这批测量**漏了 `_set_tgv_ic`**（`test_tgv._build_tgv_solver` 只建
solver，初场由调用方单独施加），于是量的是**均匀场** —— 均匀场上 BJ
当然标记 0.00%，据此一度得出"C2 已过时"的结论并写进了报告。补上初场后
标记是 100%，C2 成立。判据脚本必须自证"被测的场真的是那个场"。
"""

import numpy as np
import pytest

from ._tgv_mesh import build_triply_periodic_tet_mesh

#: 加密扫描用的两档。`n=16` 是 24576 单元，构造一次约 30 s ——
#: 那一档的数字记在模块文档里，测试里只跑 4 与 8：`n=8` 已经足够体现
#: "面模板恒 100%、顶点模板掉到 25%" 这个定性差别。
_N_COARSE = 4
_N_FINE = 8

#: `n=8` 上两种模板的实测标记比例（见模块文档表格）。阈值留余量：
#: 面模板必须**仍然**是 100%（它不收敛，这是负控制），顶点模板必须
#: 明显低于它。
_FACE_FLAGGED_AT_FINE = 0.999
_VERTEX_FLAGGED_AT_FINE_MAX = 0.40


def _tgv_state(n):
    """在 `n^3` 网格上构造 TGV **解析初场**，返回诊断需要的量。

    这里直接写初场而不是调 `test_tgv._set_tgv_ic`：那个函数要一个已经
    建好的 solver，而这条测试只需要"解点上的守恒变量 + 网格"，建 solver
    是纯开销。式子与 `_set_tgv_ic` 同一份（TGV 的标准初场）。
    """
    from .test_tgv import GAMMA, L, LC, ORDER, P_INF, RHO_INF, U0

    mesh = build_triply_periodic_tet_mesh(order=ORDER, n=n, L=L)
    xyz = mesh.sps_coords
    x, y, z = xyz[:, :, 0] / LC, xyz[:, :, 1] / LC, xyz[:, :, 2] / LC
    u = U0 * np.sin(x) * np.cos(y) * np.cos(z)
    v = -U0 * np.cos(x) * np.sin(y) * np.cos(z)
    p = (P_INF + RHO_INF * U0 ** 2 / 16.0
         * (np.cos(2 * x) + np.cos(2 * y)) * (np.cos(2 * z) + 2.0))
    rho = np.full_like(u, RHO_INF)
    field = np.empty(u.shape + (5,))
    field[..., 0] = rho
    field[..., 1] = rho * u
    field[..., 2] = rho * v
    field[..., 3] = 0.0
    field[..., 4] = p / (GAMMA - 1.0) + 0.5 * rho * (u ** 2 + v ** 2)
    # 前提自证：这个场在解点上真的有变化（第一版漏施初场、量了均匀场，
    # 见模块文档"方法论教训"）。
    assert float(field[..., 1].max() - field[..., 1].min()) > 0.1 * RHO_INF * U0
    return mesh, field


def _flagged_fraction(mesh, field, *, use_vertex):
    from autoflowcfd.core.fr_operators.bounds_sensor import (
        compute_bounds_violation_mask,
    )
    from autoflowcfd.core.fr_operators.vertex_stencil import (
        build_vertex_stencil,
    )
    from autoflowcfd.fr.native_padding import real_sps_per_cell

    fc = mesh.face_connectivity
    n_real_prism, n_real_tet = real_sps_per_cell(mesh.order)
    from .test_tgv import P_INF, RHO_INF, U0

    ref = np.array([RHO_INF, RHO_INF * U0, RHO_INF * U0, RHO_INF * U0, P_INF])
    stencil = build_vertex_stencil(mesh) if use_vertex else None
    if use_vertex:
        assert stencil is not None, "顶点模板没建出来 —— 网格缺单元-顶点连接"
    mask = compute_bounds_violation_mask(
        field,
        owner_cell=np.asarray(fc.owner_cell),
        neighbor_cell=np.asarray(fc.neighbor_cell),
        is_boundary=np.asarray(fc.is_boundary),
        ref_scales=ref,
        row_is_prism=np.arange(mesh.n_cells) < mesh.n_prism_cells,
        n_real_prism=n_real_prism,
        n_real_tet=n_real_tet,
        vertex_stencil=stencil,
    )
    return float(np.count_nonzero(mask)) / mask.size


@pytest.fixture(scope="module")
def coarse():
    return _tgv_state(_N_COARSE)


@pytest.fixture(scope="module")
def fine():
    return _tgv_state(_N_FINE)


class TestFaceStencilDoesNotConverge:
    """**负控制**：面邻居模板的标记比例在加密下不下降。"""

    def test_face_flags_everything_at_both_resolutions(self, coarse, fine):
        frac_c = _flagged_fraction(*coarse, use_vertex=False)
        frac_f = _flagged_fraction(*fine, use_vertex=False)
        assert frac_c >= _FACE_FLAGGED_AT_FINE, (
            f"n={_N_COARSE} 面模板只标记 {100*frac_c:.2f}%")
        assert frac_f >= _FACE_FLAGGED_AT_FINE, (
            f"n={_N_FINE} 面模板标记 {100*frac_f:.2f}% —— 记录是 100%，"
            f"它一旦收敛说明判据被改过，应当来更新本文件的记录")


class TestVertexStencilConverges:
    """顶点邻居模板的标记比例随加密真正下降 —— 这是 C2 的修复依据。"""

    def test_vertex_flags_far_fewer_cells_on_the_finer_mesh(self, fine):
        frac = _flagged_fraction(*fine, use_vertex=True)
        assert frac <= _VERTEX_FLAGGED_AT_FINE_MAX, (
            f"n={_N_FINE} 顶点模板标记 {100*frac:.2f}% —— 记录是 25.00%")

    def test_refinement_reduces_the_flagged_fraction(self, coarse, fine):
        """加密必须让标记比例**真正下降**（面模板做不到这一点）。"""
        frac_c = _flagged_fraction(*coarse, use_vertex=True)
        frac_f = _flagged_fraction(*fine, use_vertex=True)
        assert frac_f < 0.5 * frac_c, (
            f"顶点模板加密后标记比例 {100*frac_c:.2f}% -> "
            f"{100*frac_f:.2f}%，没有显著下降")

    def test_median_overshoot_vanishes_on_the_finer_mesh(self, fine):
        """过半单元在顶点包络下**完全不越界**（面模板下中位越界 1.83e-01）。

        这条比"标记比例"更直接：它说的是越界量本身的分布，而不是经过
        容差之后的判定结果。
        """
        mesh, field = fine
        from autoflowcfd.core.fr_operators.vertex_stencil import (
            accumulate_vertex_envelope,
            build_vertex_stencil,
        )
        from autoflowcfd.core.fr_operators.bounds_sensor import (
            _scatter_minmax,
        )
        from autoflowcfd.fr.native_padding import real_sps_per_cell
        from .test_tgv import RHO_INF, U0

        n_real = real_sps_per_cell(mesh.order)[1]      # 纯四面体网格
        real = field[:, :n_real, 1]                     # rho_u
        cm, cmax, cmin = real.mean(axis=1), real.max(axis=1), real.min(axis=1)
        fc = mesh.face_connectivity
        oc, nc = np.asarray(fc.owner_cell), np.asarray(fc.neighbor_cell)
        interior = ~np.asarray(fc.is_boundary)

        def envelope(use_vertex):
            nb_max, nb_min = cm.copy(), cm.copy()
            _scatter_minmax(np, nb_max, nb_min, oc[interior], cm[nc[interior]])
            _scatter_minmax(np, nb_max, nb_min, nc[interior], cm[oc[interior]])
            if use_vertex:
                accumulate_vertex_envelope(
                    np, _scatter_minmax, nb_max, nb_min, cm,
                    build_vertex_stencil(mesh))
            over = np.maximum(np.maximum(cmax - nb_max, nb_min - cmin), 0.0)
            return float(np.median(over)) / (RHO_INF * U0)

        med_face = envelope(False)
        med_vertex = envelope(True)
        assert med_vertex == 0.0, (
            f"顶点包络下的中位越界 {med_vertex:.4e} —— 记录是恰好 0")
        assert med_face > 0.05, (
            f"面包络下的中位越界只有 {med_face:.4e} —— 记录是 1.83e-01，"
            f"它一旦变小说明面模板或算例被改过")


class TestPaddingSlotsAreNotTheCause:
    """**被证伪的假设**钉成测试：填充槽位不是"标记 100%"的成因。

    如实记录一处**更正**：第一版这条断言的是"含/不含填充逐位相同"，
    实测**不成立** —— n=8 上"只算真实槽位"100.00%、"含填充"98.96%，
    32/3072 个单元（1.04%）不同（n=4 上 0 个）。

    所以正确的陈述是两句，不是一句：

      * 填充**不是** 100% 标记的成因（它只影响约 1% 的单元，不是 100%）；
      * 但它**不是完全中性**的 —— 这正是 `row_is_prism`/`n_real_*` 那两个
        参数存在的理由（2026-09-19 早先那批改动引入，依据是另一个测量：
        "TGV P2 推进 30 步后填充值已越出真实槽位区间 14% 的区间宽"）。

    本类因此断言"差异很小"而不是"零差异"；把它写成零差异会在第一次
    真实运行里失败、并让人误以为填充排除没生效。
    """

    #: 含/不含填充在 n=8 上的实测标记比例差（1.04%）。阈值取 3% 留余量：
    #: 判据是"远小于 100%"，不是钉死在 1.04%。
    _MAX_PADDING_INDUCED_DIFF = 0.03

    def test_including_padding_slots_changes_nothing(self, fine):
        mesh, field = fine
        from autoflowcfd.core.fr_operators.bounds_sensor import (
            compute_bounds_violation_mask,
        )
        from autoflowcfd.fr.native_padding import real_sps_per_cell
        from .test_tgv import P_INF, RHO_INF, U0

        n_real_prism, n_real_tet = real_sps_per_cell(mesh.order)
        assert n_real_tet < field.shape[1], (
            "这个网格没有填充槽位，本条测试的前提不成立")
        fc = mesh.face_connectivity
        common = dict(
            owner_cell=np.asarray(fc.owner_cell),
            neighbor_cell=np.asarray(fc.neighbor_cell),
            is_boundary=np.asarray(fc.is_boundary),
            ref_scales=np.array([RHO_INF, RHO_INF * U0, RHO_INF * U0,
                                 RHO_INF * U0, P_INF]),
        )
        only_real = compute_bounds_violation_mask(
            field, **common,
            row_is_prism=np.arange(mesh.n_cells) < mesh.n_prism_cells,
            n_real_prism=n_real_prism, n_real_tet=n_real_tet)
        with_padding = compute_bounds_violation_mask(field, **common)
        diff = float(np.count_nonzero(only_real != with_padding)) / len(only_real)
        assert diff <= self._MAX_PADDING_INDUCED_DIFF, (
            f"填充槽位改变了 {100*diff:.2f}% 的判定 —— 记录是 1.04%。"
            f"它一旦变大，'填充不是 100% 标记的成因'这条结论就要重测")
        # 而两者都**远高于**顶点模板的比例 —— 这才是 100% 的真正成因
        # （模板，不是填充）。
        assert only_real.mean() > 0.9 and with_padding.mean() > 0.9, (
            f"面模板下含/不含填充分别标记 {100*only_real.mean():.2f}% / "
            f"{100*with_padding.mean():.2f}% —— 记录都是约 100%")
