"""逐层反应式自碰撞冻结机制（mesh_front_collision.py）的单元测试，
mesh_extrusion.extrude_layers 现在每层之后都调用它——为什么静态的、基于
未变形表面的估计（mesh_layer_step 里的斜接，或先验的 thickness_limit）
单靠自身无法保证挤出前沿不折叠到自己身上（不论 growth_rate/bl_layers/
transition_growth_rate 取什么），见该模块自己的文档。
"""

import numpy as np
import pytest

from autoflowcfd.grid.mesh_gen.extrusion.mesh_extrusion import extrude_layers
from autoflowcfd.grid.mesh_gen.utils.mesh_front_collision import (
    CONVERGENCE_SAFETY_FRACTION,
    clamp_budget_for_convergence,
    find_self_colliding_faces,
    freeze_self_colliding_nodes,
)
from autoflowcfd.grid.mesh_gen.utils.mesh_utils import compute_face_normals

# 两个在三维里真正相交、不共享顶点的三角形——直接对照
# overlap_geometry.triangle_triangle_intersect 验证过。三角形 A 平放在 z=0
# 且内部包含原点；三角形 B 的边 B0-B1 是线段 x=0,y=0,z∈[-2,2]，正好穿过
# 那个内部点。
A0, A1, A2 = np.array([-2., -2., 0.]), np.array([2., -2., 0.]), np.array([0., 2., 0.])
B0, B1, B2 = np.array([0., 0., -2.]), np.array([0., 0., 2.]), np.array([0., 3., 0.])


class TestFindSelfCollidingFaces:
    def test_crossing_triangles_are_detected(self):
        nodes = np.array([A0, A1, A2, B0, B1, B2])
        faces = np.array([[0, 1, 2], [3, 4, 5]])
        colliding = find_self_colliding_faces(nodes, faces)
        assert sorted(colliding.tolist()) == [0, 1]

    def test_well_separated_triangles_are_not_flagged(self):
        far = np.array([B0, B1, B2]) + 100.0
        nodes = np.vstack([[A0, A1, A2], far])
        faces = np.array([[0, 1, 2], [3, 4, 5]])
        assert len(find_self_colliding_faces(nodes, faces)) == 0

    def test_adjacent_faces_sharing_a_vertex_are_never_flagged(self):
        """共享一个中心顶点的平面三角形扇是普通、合法的网格拓扑，不是自碰撞——
        尽管每一对扇叶都在那个共享顶点上相接。
        """
        center = np.array([0., 0., 0.])
        n = 8
        rim = [np.array([np.cos(t), np.sin(t), 0.])
               for t in np.linspace(0, 2 * np.pi, n, endpoint=False)]
        nodes = np.array([center] + rim)
        faces = np.array([[0, 1 + i, 1 + (i + 1) % n] for i in range(n)])
        assert len(find_self_colliding_faces(nodes, faces)) == 0

    def test_empty_face_array_returns_empty(self):
        nodes = np.zeros((0, 3))
        faces = np.zeros((0, 3), dtype=np.int64)
        result = find_self_colliding_faces(nodes, faces)
        assert len(result) == 0


class TestFreezeSelfCollidingNodes:
    def _two_face_setup(self):
        """faces=[0,1,2]（A）与 [3,4,5]（B）；`current` 里 B 在很远处（上一层，
        已被接受、无碰撞），`new` 里 B 被拉回来与 A 真正相交。
        """
        new_nodes = np.array([A0, A1, A2, B0, B1, B2])
        current_nodes = new_nodes.copy()
        current_nodes[3:] = np.array([B0, B1, B2]) + 100.0
        faces = np.array([[0, 1, 2], [3, 4, 5]])
        return new_nodes, current_nodes, faces

    def test_colliding_nodes_are_rolled_back_and_frozen(self):
        new_nodes, current_nodes, faces = self._two_face_setup()
        budget = np.full(6, np.inf)

        frozen = freeze_self_colliding_nodes(new_nodes, current_nodes, faces, budget)

        assert sorted(frozen.tolist()) == [0, 1, 2, 3, 4, 5]
        assert np.array_equal(new_nodes, current_nodes)
        assert np.all(budget == 0.0)
        # 回退后的结果本身必须无碰撞——冻结到上一层（已被接受的那层）
        # 绝不会让情况变坏。
        assert len(find_self_colliding_faces(new_nodes, faces)) == 0

    def test_uninvolved_face_is_left_untouched(self):
        """第三个、离得很远的面不应受冻结碰撞对的影响——冻结只作用于出问题的
        节点，不是整层。
        """
        new_nodes, current_nodes, faces = self._two_face_setup()
        # 既远离原点（A/B 自己的坐标）**又**远离 +100（这个设置里 B 被回退到的
        # 位置），所以绝不会意外落在两者附近。
        far_face = np.array([[1e5, 1e5, 1e5], [1e5 + 1, 1e5, 1e5], [1e5, 1e5 + 1, 1e5]])
        new_nodes = np.vstack([new_nodes, far_face])
        current_nodes = np.vstack([current_nodes, far_face + np.array([0., 0., 5.])])
        faces = np.vstack([faces, [[6, 7, 8]]])
        budget = np.full(9, np.inf)
        original_far = new_nodes[6:].copy()

        frozen = freeze_self_colliding_nodes(new_nodes, current_nodes, faces, budget)

        assert set(frozen.tolist()) == {0, 1, 2, 3, 4, 5}
        assert np.array_equal(new_nodes[6:], original_far), "uninvolved face must not move"
        assert np.all(budget[6:] == np.inf), "uninvolved nodes must keep their budget"

    def test_no_collision_freezes_nothing(self):
        nodes = np.array([A0, A1, A2])
        faces = np.array([[0, 1, 2]])
        budget = np.full(3, np.inf)

        frozen = freeze_self_colliding_nodes(nodes.copy(), nodes.copy(), faces, budget)

        assert len(frozen) == 0
        assert np.all(budget == np.inf)

    def test_two_independent_collisions_are_both_resolved(self):
        """同一层里别处两对互不相关的碰撞（例如同一物体的两个不同尖角）必须在
        一次调用里都被抓到并冻结，而不是只处理找到的第一对。
        """
        new1, cur1, _ = self._two_face_setup()
        offset = np.array([500., 0., 0.])
        new2, cur2 = new1 + offset, cur1 + offset
        new_nodes = np.vstack([new1, new2])
        current_nodes = np.vstack([cur1, cur2])
        faces = np.vstack([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]])
        budget = np.full(12, np.inf)

        frozen = freeze_self_colliding_nodes(new_nodes, current_nodes, faces, budget)

        assert set(frozen.tolist()) == set(range(12))
        assert len(find_self_colliding_faces(new_nodes, faces)) == 0

    def test_partially_frozen_pair_only_refreezes_the_still_moving_side(self):
        """碰撞对的一侧在更早的层已被冻结（预算已为 0，extrude_single_layer 已把
        它的位移钳为 0——这一层它本来就没动）、另一侧仍在正常推进时，只有仍在
        移动的那一侧的节点作为新冻结返回；已冻结的一侧没有可回退的东西。
        """
        new_nodes, current_nodes, faces = self._two_face_setup()
        new_nodes[:3] = current_nodes[:3]  # triangle A already frozen: unmoved
        budget = np.array([0., 0., 0., np.inf, np.inf, np.inf])

        frozen = freeze_self_colliding_nodes(new_nodes, current_nodes, faces, budget)

        assert set(frozen.tolist()) == {3, 4, 5}
        assert np.array_equal(new_nodes, current_nodes)
        assert list(budget) == [0., 0., 0., 0., 0., 0.]
        assert len(find_self_colliding_faces(new_nodes, faces)) == 0


class TestClampBudgetForConvergence:
    def _facing_pair(self, gap=0.05):
        """两个三角形，互为唯一的近邻，相距 `gap`，绕向使法向相对（B 的法向是
        -z，正对着 A 的 +z 朝下）——真正相向汇聚的一对，而不只是两个恰好平行的
        邻近三角形（这个函数为什么必须看方向而不只是邻近度，见
        CONVERGING_DOT_THRESHOLD 的注释）。
        """
        a = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        b = np.array([[0., 0., gap], [1., 0., gap], [0., 1., gap]])
        nodes = np.vstack([a, b])
        faces = np.array([[0, 1, 2], [3, 5, 4]])  # note: 3,5,4 reverses B's winding
        return nodes, faces

    def test_tightens_to_safety_fraction_of_current_gap(self):
        gap = 0.05
        nodes, faces = self._facing_pair(gap)
        budget = np.full(6, np.inf)

        clamp_budget_for_convergence(nodes, faces, budget)

        expected = CONVERGENCE_SAFETY_FRACTION * gap
        assert budget == pytest.approx(np.full(6, expected))

    def test_fraction_is_strictly_below_one_half(self):
        """安全裕度的全部意义：两侧各自用完由"严格小于半个间隙"导出的预算，
        永远不会恰好相遇（见 CONVERGENCE_SAFETY_FRACTION 的注释）——防止有人
        把它"简化"回精确的 0.5，悄悄重新引入恰好重合的汇聚。
        """
        assert CONVERGENCE_SAFETY_FRACTION < 0.5

    def test_never_loosens_an_already_tighter_budget(self):
        """被更早的层（或 find_self_colliding_faces 自己的冻结）冻结的节点必须
        保持冻结——这个函数只能收紧预算，不能恢复。
        """
        nodes, faces = self._facing_pair(gap=0.05)
        budget = np.array([0.001] * 3 + [np.inf] * 3)

        clamp_budget_for_convergence(nodes, faces, budget)

        assert budget[0] == pytest.approx(0.001)
        assert budget[3] == pytest.approx(CONVERGENCE_SAFETY_FRACTION * 0.05)

    def test_well_separated_faces_are_not_clamped(self):
        nodes, faces = self._facing_pair(gap=0.05)
        nodes[3:] += 1000.0  # 把第二个三角形推到很远
        budget = np.full(6, np.inf)

        clamp_budget_for_convergence(nodes, faces, budget)

        assert np.all(np.isinf(budget))

    def test_close_but_diverging_pair_across_a_convex_edge_is_not_clamped(self):
        """cube_demo 上发现的一个真实、严重缺陷的回归测试：跨在普通**凸**棱边
        两侧的两个小三角形（材料在 x<0 与 y<0，共享边在 x=0,y=0）从第一层起就在
        共享边附近靠得很近——这是特征附近正常、形状正确的网格加密，不是缺陷——
        并且各自沿自己的法向（约 (1,0,0) 与 (0,1,0)）移动会**增大**它们的间距
        （凸棱的前沿在挤出时发散，mesh_layer_step.py 的斜接正是为它而设）。
        只按邻近度钳制（完全没有方向过滤）无法把它与真正的汇聚区分开，于是几乎
        沿立方体的每条棱都冻结了节点——直接测量：由此产生的冻结、近乎重复的几何
        送进 tetgen 之后，重叠单元比未修复的基线**多 131 倍**（132,260 对 1,004）。
        见 CONVERGING_CLOSING_RATE_THRESHOLD 的注释——包括为什么单纯的法向点积
        判据（第二个尝试）**同样**不够，只是它失效的情形更窄、这个夹具恰好没有
        覆盖（尖锐的凸楔/薄翅）。
        """
        a0, a1, a2 = np.array([0., -0.2, 0.]), np.array([0., -0.05, 0.]), np.array([0., -0.125, 0.1])
        b0, b1, b2 = np.array([-0.2, 0., 0.]), np.array([-0.05, 0., 0.]), np.array([-0.125, 0., 0.1])
        nodes = np.array([a0, a1, a2, b0, b1, b2])
        faces = np.array([[0, 1, 2], [3, 4, 5]])
        # 确认夹具确实是一对靠得很近的三角形（只看邻近度的朴素版本**会**错误地
        # 钳制它）——不是因为附近什么都没有而空洞通过的测试。
        centroid_a, centroid_b = nodes[:3].mean(axis=0), nodes[3:].mean(axis=0)
        assert np.linalg.norm(centroid_a - centroid_b) < 0.2
        budget = np.full(6, np.inf)

        clamp_budget_for_convergence(nodes, faces, budget)

        assert np.all(np.isinf(budget))

    def test_sharp_convex_wedge_is_not_clamped(self):
        """修上面凸棱缺陷时发现的第二个、更隐蔽的误报的回归测试：单纯的
        dot(normal_a, normal_b) < 0 过滤（第一个尝试的修法）对尖锐的**凸**楔
        （例如薄翅或翼型后缘）**同样**是错的——它的两个面法向近乎**相反**纯粹
        因为楔角是锐角（对称的 10 度楔给出 dot=-0.98），但两个表面在向外挤出时
        确实是**发散**的，与任何其它凸特征一样；材料薄并不改变偏移面的移动方向。
        这个函数实际使用的闭合速率判据在单纯法向点积判据出错的地方给出正确结果
        （直接验证过：这个夹具是 +0.35，见 CONVERGING_CLOSING_RATE_THRESHOLD 的
        注释）。
        """
        half_angle = np.deg2rad(5.0)
        top_dir = np.array([np.cos(half_angle), np.sin(half_angle), 0.])
        bot_dir = np.array([np.cos(half_angle), -np.sin(half_angle), 0.])
        a0, a1, a2 = 0.8 * top_dir, 1.0 * top_dir, 0.9 * top_dir + np.array([0., 0., 0.1])
        b0, b1, b2 = 0.8 * bot_dir, 1.0 * bot_dir, 0.9 * bot_dir + np.array([0., 0., 0.1])
        nodes = np.array([a0, a1, a2, b0, b1, b2])
        faces = np.array([[0, 1, 2], [3, 4, 5]])
        centroid_a, centroid_b = nodes[:3].mean(axis=0), nodes[3:].mean(axis=0)
        assert np.linalg.norm(centroid_a - centroid_b) < 0.2  # confirm genuinely close
        budget = np.full(6, np.inf)

        clamp_budget_for_convergence(nodes, faces, budget)

        assert np.all(np.isinf(budget))

    def test_already_intersecting_pair_clamps_straight_to_zero(self):
        """候选对已经（恰好）相交的情形在实际中不应出现（见函数文档——按归纳
        current_nodes 总是已经无碰撞），但必须防御性地处理
        （triangle_triangle_min_distance 只对不相交的一对有意义），而不是崩溃或
        静默跳过钳制。需要一对**既**重叠**又**汇聚（闭合速率为负，见
        CONVERGING_CLOSING_RATE_THRESHOLD）的三角形——两个倾斜、互相穿过的三角形，
        固定进这个测试之前直接对照 triangle_triangle_intersect 与闭合速率公式
        验证过。
        """
        a0, a1, a2 = np.array([0., 0., 0.4]), np.array([1., 0., 0.1]), np.array([0., 1., -0.1])
        b0, b1, b2 = np.array([0., 0., -0.1]), np.array([1., 0., -0.1]), np.array([0., 1., 0.1])
        nodes = np.array([a0, a1, a2, b0, b1, b2])
        faces = np.array([[0, 1, 2], [3, 4, 5]])
        budget = np.full(6, np.inf)

        clamp_budget_for_convergence(nodes, faces, budget)

        assert np.all(budget == 0.0)

    def test_empty_face_array_does_not_crash(self):
        nodes = np.zeros((0, 3))
        faces = np.zeros((0, 3), dtype=np.int64)
        budget = np.zeros(0)
        clamp_budget_for_convergence(nodes, faces, budget)  # must not raise


class TestExtrudeLayersNeverProducesASelfIntersectingLayer:
    """端到端回归：两片相对的平面片，之间只有 0.05m 的窄缝（与车身底部贴近
    地面是同一类缺陷——见 mesh_tetgen_core.compute_local_thickness_limit 的
    文档），相向挤出且**不**提供先验的 thickness_limit，所以唯一能阻止两个
    前沿交叉的就是被测的反应式冻结。生长参数是普通默认值，没有为了让测试
    通过而调——要点（按本项目的明确要求）是任何层数或增长率都不应产生重叠。
    """

    def _facing_patches(self, gap=0.05):
        # 片 A：z=0 上的单位正方形，绕向使法向为 +z。
        a = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
        # 片 B：同一投影位置、z=gap，绕向使法向为 -z
        # （正对着 A 朝下）。
        b = np.array([[0., 0., gap], [1., 0., gap], [1., 1., gap], [0., 1., gap]])
        nodes = np.vstack([a, b])
        faces = np.array([
            [0, 1, 2], [0, 2, 3],   # A, normal +z
            [4, 6, 5], [4, 7, 6],   # B, normal -z
        ])
        normals = compute_face_normals(nodes, faces)
        assert normals[0][2] == pytest.approx(1.0)
        assert normals[2][2] == pytest.approx(-1.0)
        return nodes, faces, normals

    def test_facing_fronts_never_cross_across_any_layer(self):
        surface_nodes, surface_faces, normals = self._facing_patches(gap=0.05)
        bounding_box = {
            'min': np.array([-10., -10., -10.]),
            'max': np.array([10., 10., 10.]),
        }

        all_nodes, layer_connectivity = extrude_layers(
            surface_nodes, surface_faces, normals, bounding_box,
            growth_rate=1.2, min_cell_size=0.005,
            bl_layers=20,
        )

        n_layers = len(layer_connectivity)
        npl = len(surface_nodes)
        for k in range(n_layers):
            layer_nodes = all_nodes[k * npl:(k + 1) * npl]
            colliding = find_self_colliding_faces(layer_nodes, surface_faces)
            assert len(colliding) == 0, (
                f"layer {k} self-intersects (faces {colliding.tolist()}) - "
                f"the reactive freeze failed to stop the fronts from crossing"
            )
            # A 的四个节点（0-3，向 +z 生长）绝不能越过 B 对应的四个节点（4-7，向
            # -z 生长）："从未交叉"的物理含义，在上面几何自相交检查之外直接检查。
            assert np.all(layer_nodes[0:4, 2] <= layer_nodes[4:8, 2] + 1e-12)

        # 机制必须真的起过作用（这个间隙足够窄，按这些设置不受约束地生长会在
        # 20 层之内早早把它填满）——否则上面的测试只是因为没有任何东西长到足够远
        # 而空洞通过。
        final_a_z = all_nodes[(n_layers - 1) * npl + 0, 2]
        final_b_z = all_nodes[(n_layers - 1) * npl + 4, 2]
        assert final_b_z - final_a_z < 0.05, "fronts should have advanced close to the gap"
        unconstrained_estimate = 0.005 * (1.2 ** (n_layers - 1))
        assert final_a_z < unconstrained_estimate, (
            "front A should have been frozen short of its unconstrained growth"
        )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
