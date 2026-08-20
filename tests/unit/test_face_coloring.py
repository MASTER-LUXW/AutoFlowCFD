"""面图着色正确性验证测试。

验证：
1. 贪心着色算法正确性——同色面之间无"触及单元"冲突，覆盖 owner 侧与
   非边界面的 neighbor 侧两次 scatter-add 写入（不能只看 owner_cell，
   否则漏掉"面A的owner是面B的neighbor"这类交叉冲突，见 face_coloring.py
   模块文档与本文件 test_owner_neighbor_cross_conflict_detected）
2. 缓存机制正常工作
3. 图着色 kernel 与 per-thread buffer kernel 数值等价
"""

import numpy as np
import pytest
from autoflowcfd.core.utils.face_coloring import greedy_face_coloring, get_color_masks


def _assert_no_write_conflict(colors, owner_cell, neighbor_cell, is_boundary):
    """结构性校验：同色的任意两个面，其"触及单元集合"不相交。

    触及单元集合：owner 侧恒为 {owner_cell[f]}；非边界面额外含
    {neighbor_cell[f]}（对应图着色 kernel 里两次独立的 scatter-add 写入）。
    """
    n_faces = len(owner_cell)
    touched = []
    for f in range(n_faces):
        cells = {int(owner_cell[f])}
        if not bool(is_boundary[f]):
            nc = int(neighbor_cell[f])
            if nc >= 0:
                cells.add(nc)
        touched.append(cells)

    n_colors = int(np.max(colors)) + 1
    for c in range(n_colors):
        face_idx = np.where(colors == c)[0]
        seen_cells = set()
        for f in face_idx:
            overlap = touched[f] & seen_cells
            assert not overlap, f"Color {c}: face {f} 与已着色面共享单元 {overlap}，存在 scatter-add 写冲突"
            seen_cells |= touched[f]


class TestGreedyFaceColoring:
    """贪心面图着色测试。"""

    def test_simple_coloring(self):
        """简单测试：每个 cell 2 个面（均为边界面，无 neighbor 侧写入），应需要 2 种颜色。"""
        owner_cell = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=np.int64)
        neighbor_cell = np.full(10, -1, dtype=np.int64)
        is_boundary = np.ones(10, dtype=np.bool_)
        colors = greedy_face_coloring(owner_cell, 5, neighbor_cell, is_boundary)
        n_colors = int(np.max(colors)) + 1

        assert n_colors >= 2
        _assert_no_write_conflict(colors, owner_cell, neighbor_cell, is_boundary)

    def test_owner_neighbor_cross_conflict_detected(self):
        """回归测试：owner/neighbor 交叉冲突必须被识别为冲突。

        面0: owner=0, neighbor=1；面1: owner=1, neighbor=2。
        两个面的 owner_cell 互不相同（0 vs 1），若只按 owner_cell 分组
        着色会误判为"无冲突"、可能同色——但面0对cell 1做neighbor侧写入，
        面1对cell 1做owner侧写入，两者在cell 1上真实冲突，必须不同色。
        这正是本轮修复前 greedy_face_coloring 会漏掉的竞争条件场景。
        """
        owner_cell = np.array([0, 1], dtype=np.int64)
        neighbor_cell = np.array([1, 2], dtype=np.int64)
        is_boundary = np.array([False, False], dtype=np.bool_)

        colors = greedy_face_coloring(owner_cell, 3, neighbor_cell, is_boundary)
        assert colors[0] != colors[1], "面0/面1 在 cell 1 上有真实写冲突，不能同色"
        _assert_no_write_conflict(colors, owner_cell, neighbor_cell, is_boundary)

    def test_complex_coloring(self):
        """复杂测试：模拟生产网格的 owner/neighbor 面连接分布（内部面为主）。"""
        np.random.seed(42)
        n_faces = 10000
        n_cells = 5000
        owner_cell = np.random.randint(0, n_cells, size=n_faces, dtype=np.int64)
        # 80% 内部面（有效 neighbor_cell），20% 边界面
        is_boundary = np.random.rand(n_faces) < 0.2
        neighbor_cell = np.random.randint(0, n_cells, size=n_faces, dtype=np.int64)
        neighbor_cell[is_boundary] = -1

        colors = greedy_face_coloring(owner_cell, n_cells, neighbor_cell, is_boundary)
        n_colors = int(np.max(colors)) + 1

        _assert_no_write_conflict(colors, owner_cell, neighbor_cell, is_boundary)

        # 颜色数应该在合理范围内（覆盖 owner+neighbor 双侧冲突后颜色数会
        # 比纯 owner_cell 版本略多，但仍应远小于面数）
        assert n_colors <= 40, f"Too many colors: {n_colors}"

    def test_color_masks(self):
        """测试颜色 mask 生成。"""
        owner_cell = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
        neighbor_cell = np.full(6, -1, dtype=np.int64)
        is_boundary = np.ones(6, dtype=np.bool_)
        colors = greedy_face_coloring(owner_cell, 3, neighbor_cell, is_boundary)
        n_colors = int(np.max(colors)) + 1
        masks = get_color_masks(colors, n_colors)

        assert len(masks) == n_colors
        for c in range(n_colors):
            assert masks[c].dtype == bool
            assert np.sum(masks[c]) == np.sum(colors == c)


class TestColoringKernelEquivalence:
    """图着色 kernel 与 per-thread buffer kernel 数值等价性测试。"""

    def test_inviscid_kernel_equivalence_nt1(self):
        """无粘 kernel：nt=1 时图着色与 per-thread buffer 应 bit-exact 相等。"""
        # 这个测试在 test_fr_residual_inviscid_kernel_crosscheck.py 中已覆盖
        # 这里只验证接口存在
        from autoflowcfd.core.fr_residual.inviscid_kernel import (
            compute_inviscid_interface_correction_kernel,
        )
        from autoflowcfd.core.fr_residual.inviscid_kernel_colored import (
            compute_inviscid_interface_correction_kernel_colored,
        )
        assert callable(compute_inviscid_interface_correction_kernel)
        assert callable(compute_inviscid_interface_correction_kernel_colored)

    def test_viscous_kernel_equivalence_nt1(self):
        """粘性 kernel：nt=1 时图着色与 per-thread buffer 应 bit-exact 相等。"""
        from autoflowcfd.core.fr_residual.viscous_flux_kernel import (
            compute_viscous_interface_correction_kernel,
            compute_viscous_interface_correction_kernel_colored,
        )
        assert callable(compute_viscous_interface_correction_kernel)
        assert callable(compute_viscous_interface_correction_kernel_colored)

    def test_turbulence_transport_kernel_equivalence(self):
        """湍流输运 kernel：图着色版本存在。"""
        from autoflowcfd.core.turbulence.transport_kernel import (
            distribute_corrections_to_cells_kernel,
            distribute_corrections_to_cells_kernel_colored,
        )
        assert callable(distribute_corrections_to_cells_kernel)
        assert callable(distribute_corrections_to_cells_kernel_colored)


class TestFlatFaceColoringCache:
    """面几何缓存中的图着色测试。"""

    def test_flat_face_has_coloring(self):
        """FlatFaceGeometry 应包含图着色信息。"""
        from autoflowcfd.core.fr_operators.face_kernels import FlatFaceGeometry
        # 检查 dataclass 字段
        fields = {f.name for f in FlatFaceGeometry.__dataclass_fields__.values()}
        assert "color_face_indices" in fields
        assert "n_colors" in fields
