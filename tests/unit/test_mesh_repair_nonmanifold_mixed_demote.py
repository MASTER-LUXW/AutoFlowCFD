"""折叠角棱柱 -> 四面体降级拆分的单元测试。

核心判据：精确的单角折叠（v_c == w_c，corner in {0,1,2}）必须分解为
恰好 2 个非退化四面体，体积精确等于原折叠棱柱体积（compute_prism_volumes
给出的参考值），不丢弃任何体积。

注意（诚实记录，避免误导）：`_split_collapsed_corner_to_2_tets` 最初
是作为"修复旧版通用 3-way 拆分产生体积不均衡 sliver"的动机写的，但
已用符号推导+本文件的测试证明：对这个特定简并情形，旧版通用拆分
丢弃退化候选后剩下的 2 个候选与本函数直接构造的 2 个四面体是逐节点
集合完全相同的——即两者产生完全相同的输出几何，不存在实际的体积
不均衡 bug。本文件的测试因此把"与旧版逐位一致"作为回归判据（而不是
"比旧版更好"），价值在于验证这个更直接、更少一次浪费计算的实现是
正确的等价替代，不是验证一个不存在的 bug 被修复了。
"""

import numpy as np
import pytest

from autoflowcfd.grid.mesh_gen.repair.mesh_repair_nonmanifold_mixed_demote import (
    demote_invalid_prisms_to_tets,
    _split_collapsed_corner_to_2_tets,
)
from autoflowcfd.grid.validation.quality_metrics import compute_prism_volumes


def _tet_volumes(nodes: np.ndarray, tets: np.ndarray) -> np.ndarray:
    p = nodes[tets]
    a = p[:, 1] - p[:, 0]
    b = p[:, 2] - p[:, 0]
    c = p[:, 3] - p[:, 0]
    return np.abs(np.einsum('ij,ij->i', a, np.cross(b, c))) / 6.0


# 一个规则三棱柱的 6 个节点：底 v0,v1,v2，顶 w0,w1,w2（垂直挤出，高度2）。
_REGULAR_PRISM_NODES = np.array([
    [0, 0, 0], [1, 0, 0], [0, 1, 0],
    [0, 0, 2], [1, 0, 2], [0, 1, 2],
], dtype=float)


class TestSplitCollapsedCornerToTwoTets:
    """_split_collapsed_corner_to_2_tets 的几何正确性。"""

    @pytest.mark.parametrize("corner", [0, 1, 2])
    def test_volume_exactly_conserved_and_balanced(self, corner):
        row = [0, 1, 2, 3, 4, 5]
        row[3 + corner] = row[corner]  # 折叠该角：w_c = v_c
        prism = np.array([row], dtype=np.int64)

        ref_volume = compute_prism_volumes(_REGULAR_PRISM_NODES, prism)[0]
        tets = _split_collapsed_corner_to_2_tets(prism, corner)

        assert tets.shape == (2, 4)
        vols = _tet_volumes(_REGULAR_PRISM_NODES, tets)

        assert np.all(vols > 1e-12), "两个子四面体都必须是非退化的正体积"
        assert np.isclose(vols.sum(), ref_volume, rtol=1e-10), (
            f"分解后总体积 {vols.sum()} 必须精确等于原折叠棱柱体积 {ref_volume}"
        )
        # 规则棱柱的对称几何下，两个子四面体体积应严格相等（quad 被
        # 对角线精确二等分），非对称情况只需保证同量级（不是本测试
        # 覆盖范围）。
        assert np.isclose(vols[0], vols[1], rtol=1e-10)

    @pytest.mark.parametrize("corner", [0, 1, 2])
    def test_identical_node_sets_to_legacy_generic_split(self, corner):
        """核心回归判据（见模块/文件顶部"注意"说明）：本函数与旧版通用
        3-way 拆分 + 丢弃退化候选，对单角折叠情形必须产出逐节点集合
        完全相同的 2 个四面体——不是"体积总量相同"这种弱判据，而是
        "同一个四面体"（绕向可以不同，但顶点集合必须相同）。这是
        "本函数不改变输出几何、只是更直接的等价实现"这一结论的直接
        证据。
        """
        from autoflowcfd.grid.mesh_gen.repair.mesh_repair_nonmanifold_mixed_demote import (
            _split_prisms_to_tets,
        )
        row = [0, 1, 2, 3, 4, 5]
        row[3 + corner] = row[corner]
        prism = np.array([row], dtype=np.int64)

        new_tets = _split_collapsed_corner_to_2_tets(prism, corner)
        legacy_tets = _split_prisms_to_tets(prism)  # (3,4)，含 1 个退化行

        new_sets = {frozenset(t) for t in new_tets}
        legacy_valid_sets = {frozenset(t) for t in legacy_tets if len(set(t)) == 4}

        assert len(new_sets) == 2
        assert new_sets == legacy_valid_sets


class TestDemoteInvalidPrismsToTets:
    """demote_invalid_prisms_to_tets 端到端行为：干净单角折叠 vs. 杂乱重复模式。"""

    @pytest.mark.parametrize("corner", [0, 1, 2])
    def test_clean_single_collapse_uses_exact_2tet_split(self, corner):
        row = [0, 1, 2, 3, 4, 5]
        row[3 + corner] = row[corner]
        prism_cells = np.array([row], dtype=np.int64)
        bl_cell_groups = np.array(['WALL'], dtype=object)

        new_prism, new_groups, extra_tets, extra_groups = demote_invalid_prisms_to_tets(
            prism_cells, bl_cell_groups
        )

        assert len(new_prism) == 0, "折叠棱柱本身必须从存活棱柱数组中移除"
        assert len(new_groups) == 0
        assert extra_tets.shape == (2, 4), "干净单角折叠必须精确产出 2 个四面体，不多不少"
        assert list(extra_groups) == ['WALL', 'WALL'], "子四面体必须继承源棱柱的边界组标签"

        vols = _tet_volumes(_REGULAR_PRISM_NODES, extra_tets)
        ref_volume = compute_prism_volumes(_REGULAR_PRISM_NODES, prism_cells)[0]
        assert np.all(vols > 1e-12)
        assert np.isclose(vols.sum(), ref_volume, rtol=1e-10)

    def test_no_duplicates_returns_input_unchanged(self):
        prism_cells = np.array([[0, 1, 2, 3, 4, 5]], dtype=np.int64)
        bl_cell_groups = np.array(['WALL'], dtype=object)

        new_prism, new_groups, extra_tets, extra_groups = demote_invalid_prisms_to_tets(
            prism_cells, bl_cell_groups
        )

        assert np.array_equal(new_prism, prism_cells)
        assert np.array_equal(new_groups, bl_cell_groups)
        assert len(extra_tets) == 0
        assert len(extra_groups) == 0

    def test_empty_input(self):
        prism_cells = np.empty((0, 6), dtype=np.int64)
        bl_cell_groups = np.empty((0,), dtype=object)

        new_prism, new_groups, extra_tets, extra_groups = demote_invalid_prisms_to_tets(
            prism_cells, bl_cell_groups
        )
        assert len(new_prism) == 0
        assert len(extra_tets) == 0

    def test_messy_duplicate_pattern_falls_back_to_generic_split_without_crashing(self):
        """两对重复（非单一竖直对折叠）不满足"干净单角折叠"的判据，
        必须走通用兜底路径而不是被误当成单角折叠处理（否则
        _split_collapsed_corner_to_2_tets 的"corner 处 apex 唯一"假设
        不成立，会静默产出错误的拓扑）。这里只验证不崩溃、返回形状
        合法——这类输入本就是本模块文档字符串说明的"远少于情况1"的
        兜底场景，不要求精确体积判据。
        """
        # v0==v1（同一底面内重复，不是竖直对）且 w0==w1（顶面同样重复）:
        # 两对重复，且都不是 v_i==w_i 形式。
        prism_cells = np.array([[0, 0, 2, 3, 3, 5]], dtype=np.int64)
        bl_cell_groups = np.array(['WALL'], dtype=object)

        new_prism, new_groups, extra_tets, extra_groups = demote_invalid_prisms_to_tets(
            prism_cells, bl_cell_groups
        )
        assert len(new_prism) == 0
        # 通用拆分对这个特定简并输入可能产出 0~3 个非退化子四面体，
        # 只要求形状一致、不崩溃。
        assert extra_tets.shape[1] == 4
        assert len(extra_groups) == len(extra_tets)

    def test_mixed_batch_clean_and_messy_together(self):
        """同一批里既有干净单角折叠又有杂乱模式，两条路径的结果必须
        正确拼接（不能因为分组处理而互相覆盖或丢行）。
        """
        clean_row = [0, 1, 2, 3, 4, 5]
        clean_row[3 + 1] = clean_row[1]  # corner=1 折叠
        messy_row = [10, 10, 12, 13, 13, 15]  # 两对重复，非竖直对
        good_row = [20, 21, 22, 23, 24, 25]  # 无重复，应原样保留

        prism_cells = np.array([clean_row, messy_row, good_row], dtype=np.int64)
        bl_cell_groups = np.array(['WALL', 'INLET', 'OUTLET'], dtype=object)

        new_prism, new_groups, extra_tets, extra_groups = demote_invalid_prisms_to_tets(
            prism_cells, bl_cell_groups
        )

        assert len(new_prism) == 1
        assert np.array_equal(new_prism[0], good_row)
        assert new_groups[0] == 'OUTLET'
        # 干净折叠贡献 2 个四面体，杂乱模式贡献若干个（>=0）。
        assert extra_tets.shape[1] == 4
        assert 'WALL' in extra_groups  # 来自 clean_row 的子四面体
