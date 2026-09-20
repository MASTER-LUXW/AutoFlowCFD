"""原生棱柱基的面算子：面通量点、体积->面外插、DG 提升。

## 本文件钉住的性质

**精确性**（硬判据，任何正确实现都必须满足）：
  * 每个面恒有 `(order+1)^2` 个通量点，且确实落在那个面上；
  * 外插矩阵对空间内**全部**多项式给出精确的面上取值；
  * 模方闭式解与独立数值积分一致（与四面体那份闭式解同一条验收标准 ——
    推导完不假设成立，先用不依赖被测代码的积分核对）；
  * 提升算子满足**精确守恒恒等式** `1^T M_ref Lift_ref = 1^T`：把任意
    面通量跳跃提升到体积、再对常数测试函数取矩，必须**恰好**等于该面
    跳跃的加权和。守恒是这个算子唯一不能错的性质。

## 为什么要这一套

坍缩棱柱基的 `max|D|` 随阶数爆炸（P1 2.05 -> P3 560.1），后果是自由流
保持性只有 ~1e-9、以及只在被三角化的两个参考轴上长出的伪横流（P1 饱和、
P2 无界增长导致发散，1152 单元干净网格第 75 步）。完整数据见
`fr/native_prism/triangle_basis.py` 模块文档。
"""

import numpy as np
import pytest

from autoflowcfd.fr.native_prism.basis import (
    build_native_prism_nodes,
    build_native_prism_vandermonde,
    native_prism_exact_jacobian,
    native_prism_n_sps,
    restricted_prism_modes,
)
from autoflowcfd.fr.native_prism.face import (
    PRISM_FACE_IDS,
    build_all_native_prism_face_operators,
    build_native_prism_boundary_extrap,
    build_native_prism_lift,
    native_prism_face_points,
    native_prism_face_points_physical,
    native_prism_mode_norm_squared,
)

_ORDERS = [1, 2, 3]


class TestFacePoints:
    @pytest.mark.parametrize("order", _ORDERS)
    @pytest.mark.parametrize("face_id", PRISM_FACE_IDS)
    def test_count_is_uniform(self, order, face_id):
        """`_KernelFaceData` 的 flat 数组假设全网格每个面的通量点数统一。

        棱柱的两个三角形封盖与三个侧四边形**必须**给出同样多的点，否则
        面数据没法放进同一个 flat 数组 —— 这是 native 四面体当年记录在案
        的那条必要修正，棱柱同样受约束。
        """
        fp = native_prism_face_points(order, face_id)
        assert fp.shape == ((order + 1) ** 2, 3)

    @pytest.mark.parametrize("order", _ORDERS)
    def test_points_lie_on_the_claimed_face(self, order):
        """点必须真的在那个面上，且在参考棱柱内。"""
        tol = 1e-12
        for face_id in PRISM_FACE_IDS:
            fp = native_prism_face_points(order, face_id)
            r, s, t = fp[:, 0], fp[:, 1], fp[:, 2]
            # 参考棱柱内
            assert np.all(r >= -1.0 - tol), f"face {face_id}: r 越界"
            assert np.all(s >= -1.0 - tol), f"face {face_id}: s 越界"
            assert np.all(r + s <= tol), f"face {face_id}: r+s 越界"
            assert np.all(np.abs(t) <= 1.0 + tol), f"face {face_id}: t 越界"
            # 各面自己的约束
            if face_id == 0:
                assert np.allclose(t, -1.0), "face 0 应当在 t=-1"
            elif face_id == 1:
                assert np.allclose(t, 1.0), "face 1 应当在 t=+1"
            elif face_id == 2:
                # 排除 V0 -> 边 V1-V2 是斜边 r+s=0
                assert np.allclose(r + s, 0.0, atol=tol), "face 2 应在 r+s=0"
            elif face_id == 3:
                # 排除 V1 -> 边 V0-V2 是 r=-1
                assert np.allclose(r, -1.0, atol=tol), "face 3 应在 r=-1"
            else:
                # 排除 V2 -> 边 V0-V1 是 s=-1
                assert np.allclose(s, -1.0, atol=tol), "face 4 应在 s=-1"

    def test_side_faces_span_the_full_extrusion(self):
        """侧面的通量点必须铺满挤出方向，不能退化成一条线。"""
        fp = native_prism_face_points(2, 3)
        assert len(np.unique(np.round(fp[:, 2], 12))) == 3
        assert len(np.unique(np.round(fp[:, 0] + fp[:, 1], 12))) >= 2

    def test_bad_face_id_raises(self):
        with pytest.raises(ValueError, match="face_id"):
            native_prism_face_points(2, 5)

    def test_physical_points_match_the_cell(self):
        """物理点必须落在该物理面上（用一个右棱柱直接验证）。"""
        tri = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.25]])
        nodes = np.vstack([tri, tri + [0.0, 0.1, 0.0]])
        bottom = native_prism_face_points_physical(2, 0, nodes)
        top = native_prism_face_points_physical(2, 1, nodes)
        assert np.allclose(bottom[:, 1], 0.0)
        assert np.allclose(top[:, 1], 0.1)


class TestModeNormsAgainstQuadrature:
    """闭式模方 vs **独立**数值积分。

    积分完全不依赖被测的闭式表达式：在参考棱柱上用足够高阶的
    Gauss-Legendre 张量积 + Duffy 变换求积，被积函数直接取
    `build_native_prism_vandermonde` 给出的模态值。
    """

    @staticmethod
    def _quadrature(n_quad):
        """参考棱柱上的求积点与权重（Duffy 三角形 x 直线）。"""
        from autoflowcfd.fr.quadrature_points import gauss_legendre

        xa, wa = gauss_legendre(n_quad)
        xb, wb = gauss_legendre(n_quad)
        xt, wt = gauss_legendre(n_quad)
        A, B, T = np.meshgrid(xa, xb, xt, indexing="ij")
        WA, WB, WT = np.meshgrid(wa, wb, wt, indexing="ij")
        a, b, t = A.ravel(), B.ravel(), T.ravel()
        # Duffy: r = (1+a)(1-b)/2 - 1, s = b, Jacobian dr ds = (1-b)/2 da db
        r = (1.0 + a) * (1.0 - b) / 2.0 - 1.0
        s = b
        w = WA.ravel() * WB.ravel() * WT.ravel() * (1.0 - b) / 2.0
        return np.column_stack([r, s, t]), w

    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_closed_form_matches_quadrature(self, order):
        pts, w = self._quadrature(2 * order + 6)
        V, _, _, _ = build_native_prism_vandermonde(order, pts)
        modes = restricted_prism_modes(order)
        worst = 0.0
        detail = ""
        for m, (i, j, k) in enumerate(modes):
            num = float(np.sum(w * V[:, m] ** 2))
            ana = native_prism_mode_norm_squared(i, j, k)
            rel = abs(num - ana) / abs(ana)
            if rel > worst:
                worst, detail = rel, f"(i,j,k)=({i},{j},{k}) 数值 {num:.6e} 闭式 {ana:.6e}"
        assert worst < 1e-10, f"order={order}: 最大相对偏差 {worst:.3e}  {detail}"

    @pytest.mark.parametrize("order", [1, 2])
    def test_modes_are_orthogonal(self, order):
        """提升算子的推导依赖模态**正交**（非归一），这条要单独钉住。"""
        pts, w = self._quadrature(2 * order + 6)
        V, _, _, _ = build_native_prism_vandermonde(order, pts)
        gram = (V * w[:, None]).T @ V
        diag = np.diag(gram).copy()
        off = gram - np.diag(diag)
        assert np.max(np.abs(off)) / np.max(np.abs(diag)) < 1e-10, (
            "非对角元不可忽略，模态不正交 -> 提升算子的 diag(1/N) 形式失效")


class TestExtrapolationExactness:
    """**硬判据**：外插必须对空间内全部多项式精确。"""

    @pytest.mark.parametrize("order", _ORDERS)
    def test_extrapolates_every_polynomial_in_the_space(self, order):
        ref = build_native_prism_nodes(order)
        V_sps, _, _, _ = build_native_prism_vandermonde(order, ref)
        worst = 0.0
        detail = ""
        for face_id in PRISM_FACE_IDS:
            E = build_native_prism_boundary_extrap(order, face_id)
            fp = native_prism_face_points(order, face_id)
            V_fp, _, _, _ = build_native_prism_vandermonde(order, fp)
            # 逐个模态当场：节点值 V_sps[:, m]，面上精确值 V_fp[:, m]
            for m in range(V_sps.shape[1]):
                got = E @ V_sps[:, m]
                exact = V_fp[:, m]
                err = float(np.max(np.abs(got - exact)))
                rel = err / max(float(np.max(np.abs(exact))), 1.0)
                if rel > worst:
                    worst, detail = rel, f"face {face_id} mode {m}"
        assert worst < 1e-11, f"order={order}: 最大相对误差 {worst:.3e} ({detail})"

    @pytest.mark.parametrize("order", _ORDERS)
    def test_reproduces_constants(self, order):
        """`E @ 1 = 1` —— 均匀流下左右两侧必须外插出**同一个**常数，
        否则界面会跳出一个伪跳跃，那正是自由流保持性的直接来源。
        """
        for face_id in PRISM_FACE_IDS:
            E = build_native_prism_boundary_extrap(order, face_id)
            res = float(np.max(np.abs(E @ np.ones(E.shape[1]) - 1.0)))
            assert res < 1e-13, f"order={order} face={face_id}: {res:.3e}"


class TestLiftConservation:
    """提升算子的**精确守恒恒等式** `1^T M_ref Lift_ref = 1^T`。

    推导：`M_ref = V^-T diag(N) V^-1`、`Lift_ref = V diag(1/N) B^T`，故
    `M_ref Lift_ref = V^-T B^T`；再左乘 `1^T`（常数场的模态系数只有第 0
    个非零、且恰好是 `1/phi_000`）即得 `1^T`。

    物理含义：把任意面跳跃提升进体积、再对常数测试函数取矩，结果**恰好**
    等于该面跳跃的加权和 —— 也就是"进出这个单元的通量不会凭空增减"。
    守恒是这个算子唯一不能错的性质，所以用恒等式而不是容差比较来钉它。
    """

    @staticmethod
    def _m_ref(order):
        ref = build_native_prism_nodes(order)
        V, _, _, _ = build_native_prism_vandermonde(order, ref)
        modes = restricted_prism_modes(order)
        N = np.array([native_prism_mode_norm_squared(i, j, k)
                      for (i, j, k) in modes])
        Vinv = np.linalg.inv(V)
        return Vinv.T @ (N[:, None] * Vinv)

    @pytest.mark.parametrize("order", _ORDERS)
    def test_conservation_identity(self, order):
        M = self._m_ref(order)
        for face_id in PRISM_FACE_IDS:
            L = build_native_prism_lift(order, face_id)
            got = np.ones(M.shape[0]) @ (M @ L)
            err = float(np.max(np.abs(got - 1.0)))
            assert err < 1e-11, (
                f"order={order} face={face_id}: 守恒恒等式偏差 {err:.3e}"
                f"（1^T M Lift 应当逐项为 1）")

    @pytest.mark.parametrize("order", _ORDERS)
    def test_algebraic_form_matches_the_derivation(self, order):
        """`M_ref @ Lift_ref == V^-T @ B^T` —— 推导里那一步化简的直接检验。"""
        M = self._m_ref(order)
        ref = build_native_prism_nodes(order)
        V, _, _, _ = build_native_prism_vandermonde(order, ref)
        for face_id in PRISM_FACE_IDS:
            L = build_native_prism_lift(order, face_id)
            fp = native_prism_face_points(order, face_id)
            B, _, _, _ = build_native_prism_vandermonde(order, fp)
            expect = np.linalg.inv(V).T @ B.T
            assert np.allclose(M @ L, expect, rtol=1e-9, atol=1e-11), (
                f"order={order} face={face_id}")

    @pytest.mark.parametrize("order", _ORDERS)
    def test_shapes(self, order):
        n_sps = native_prism_n_sps(order)
        for face_id in PRISM_FACE_IDS:
            assert build_native_prism_lift(order, face_id).shape == (
                n_sps, (order + 1) ** 2)
            assert build_native_prism_boundary_extrap(
                order, face_id).shape == ((order + 1) ** 2, n_sps)


class TestBulkBuilder:
    @pytest.mark.parametrize("order", _ORDERS)
    def test_matches_individual_builders(self, order):
        extrap, lift = build_all_native_prism_face_operators(order)
        assert set(extrap) == set(PRISM_FACE_IDS)
        assert set(lift) == set(PRISM_FACE_IDS)
        for f in PRISM_FACE_IDS:
            assert np.array_equal(
                extrap[f], build_native_prism_boundary_extrap(order, f))
            assert np.array_equal(lift[f], build_native_prism_lift(order, f))


class TestGeometryJacobian:
    """`native_prism_exact_jacobian` —— 原生坐标下的解析雅可比。"""

    def test_right_prism_metric_is_constant(self):
        """顶面是底面纯平移时，雅可比逐点恒定。

        这是"自由流保持性回到机器零"的结构性依据：度量恒定 + `D@1` 机器零
        => 均匀流下体积项散度机器零。坍缩基在同样的单元上做不到（它的
        `d(r,s)/d(a,b)` 带 `(1-b)/4` 因子、随点变化）。
        """
        tri = np.array([[0.0, 0.0, 0.0], [1e-2, 0.0, 0.0],
                        [0.0, 0.0, 2.5e-3]])
        nodes = np.vstack([tri, tri + [0.0, 1e-5, 0.0]])
        ref = build_native_prism_nodes(2)
        jac = native_prism_exact_jacobian(ref, nodes)
        spread = np.max(np.abs(jac - jac[0])) / np.max(np.abs(jac))
        assert spread < 1e-14, f"右棱柱的雅可比不恒定，相对散布 {spread:.3e}"

    def test_matches_finite_difference_of_the_map(self):
        """解析雅可比必须与几何映射的数值微分一致（防公式抄错）。"""
        from autoflowcfd.fr.native_prism.basis import (
            map_native_prism_to_physical,
        )

        nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.1, 0.0], [0.2, 0.0, 1.0],
                          [0.05, 1.0, 0.0], [1.1, 1.2, 0.1], [0.1, 1.0, 1.2]])
        pts = np.array([[-0.5, -0.5, 0.3], [-0.9, -0.05, -0.7],
                        [-0.2, -0.7, 0.0]])
        jac = native_prism_exact_jacobian(pts, nodes)
        h = 1e-6
        for p in range(pts.shape[0]):
            for m in range(3):
                dp = np.zeros(3)
                dp[m] = h
                fd = (map_native_prism_to_physical(
                          (pts[p] + dp)[None, :], nodes)
                      - map_native_prism_to_physical(
                          (pts[p] - dp)[None, :], nodes)) / (2.0 * h)
                assert np.allclose(jac[p, :, m], fd.ravel(), atol=1e-7), (
                    f"点 {p} 方向 {m}: 解析 {jac[p, :, m]} vs 数值 {fd.ravel()}")

    def test_rejects_wrong_node_count(self):
        with pytest.raises(ValueError, match=r"\(6, 3\)"):
            native_prism_exact_jacobian(
                build_native_prism_nodes(1), np.zeros((4, 3)))


class TestCubeFaceMapping:
    """`(axis, side)` 立方体面 <-> 原生棱柱 `face_id` 的对应表。

    **为什么必须几何验证**：面 id 配错不会报错，只会静默地对某个面用错
    外插/提升矩阵 —— 与当年多 GPU"四面体拿到棱柱矩阵"完全同一类缺陷
    （那条是靠真实网格发散才暴露的）。

    **判据怎么选**：第一版判"通量点落在该面顶点张成的平面上"，那是错的
    —— 不规则棱柱的**四边形侧面本来就不共面**（实测非平面性 2e-2，与面
    id 无关）。改用形状无关的精确判据：几何映射对 6 个顶点是**线性**的，
    所以每个通量点都是 6 个顶点的一组权重；某个面的点必须**只**依赖
    `PRISM_CUBE_FACES` 给出的那几个顶点，被排除的顶点权重恒为零。
    """

    @staticmethod
    def _vertex_weights(order, face_id):
        """通量点对 6 个顶点的权重 `(n_fp, 6)`。

        只用公开的 `native_prism_face_points_physical`：映射对顶点线性，
        所以把第 i 个顶点设成 `[1,1,1]`、其余设成 0，返回值就是第 i 个
        顶点的权重（三列相同）。这样不会拿被测公式去验证它自己。
        """
        w = np.empty(((order + 1) ** 2, 6))
        for i in range(6):
            nodes = np.zeros((6, 3))
            nodes[i] = 1.0
            got = native_prism_face_points_physical(order, face_id, nodes)
            assert np.allclose(got[:, 0], got[:, 1]) and np.allclose(
                got[:, 0], got[:, 2]), "权重提取假设被破坏"
            w[:, i] = got[:, 0]
        return w

    def test_each_native_face_depends_only_on_its_cube_faces_vertices(self):
        from autoflowcfd.fr.native_prism.face import (
            NATIVE_PRISM_FACE_TO_CUBE_FACE,
            cube_face_to_native_prism_face,
        )
        from autoflowcfd.grid.curved_mapping.curved_mapping import (
            PRISM_CUBE_FACES,
        )

        axis_name = {0: "a", 1: "b", 2: "c"}
        for face_id, (axis, side) in NATIVE_PRISM_FACE_TO_CUBE_FACE.items():
            key = f"{axis_name[axis]}={'+1' if side > 0 else '-1'}"
            on_face = set(PRISM_CUBE_FACES[key])
            w = self._vertex_weights(3, face_id)
            off = [i for i in range(6) if i not in on_face]
            worst = float(np.max(np.abs(w[:, off])))
            assert worst < 1e-14, (
                f"face_id={face_id} 被映射到立方体面 {key}（物理顶点 "
                f"{sorted(on_face)}），但它的通量点对面外顶点 {off} 仍有"
                f"权重 {worst:.3e} —— 对应表配错了")
            # 面上的顶点必须真的被用到（否则"权重为零"这条判据会退化成
            # 对任何配错都成立）
            assert np.max(np.abs(w[:, sorted(on_face)])) > 0.1
            assert cube_face_to_native_prism_face(axis, side) == face_id

    def test_weights_form_a_partition_of_unity(self):
        """权重和恒为 1 —— 映射必须是顶点的仿射组合。"""
        for face_id in PRISM_FACE_IDS:
            w = self._vertex_weights(2, face_id)
            assert np.allclose(w.sum(axis=1), 1.0, atol=1e-14)

    def test_mapping_is_a_bijection_onto_the_five_real_faces(self):
        from autoflowcfd.fr.native_prism.face import (
            NATIVE_PRISM_FACE_TO_CUBE_FACE,
        )
        from autoflowcfd.grid.curved_mapping.curved_mapping import (
            PRISM_CUBE_FACES,
        )

        assert set(NATIVE_PRISM_FACE_TO_CUBE_FACE) == set(PRISM_FACE_IDS)
        assert len(set(NATIVE_PRISM_FACE_TO_CUBE_FACE.values())) == 5
        assert len(PRISM_CUBE_FACES) == 5, (
            "PRISM_CUBE_FACES 不再是 5 个面，对应表必须重新核对")

    def test_degenerate_cube_face_is_rejected(self):
        """`(1, +1)` 是坍缩参考立方体上的退化面（坍缩成一条侧棱）。

        静默返回某个 face_id 会让一条不存在的面参与界面项组装。
        """
        from autoflowcfd.fr.native_prism.face import (
            cube_face_to_native_prism_face,
        )

        with pytest.raises(ValueError, match="退化面"):
            cube_face_to_native_prism_face(1, 1.0)


class TestRealSpsCountsFollowTheActivePrismBasis:
    """`real_sps_per_cell` —— "哪些槽位是真的"的唯一判据来源。

    **为什么这条必须有测试**：`reduce_*_over_real_sps` 原先写死"棱柱用满
    全部槽位"。原生棱柱一上线棱柱也有填充槽位，而那些槽位冻结在初值、
    随推进变馊（实测 10 步后偏差 3.4%）。漏改不会报错，只会让 checkpoint
    的单元均值、人工粘性尺度、omega 壁面目标值里的 rho、GPU 局部 dt 的
    min 全部**静默**算错。
    """

    @pytest.mark.parametrize("order", [1, 2, 3, 4])
    def test_collapsed_prism_uses_every_slot(self, order, monkeypatch):
        from autoflowcfd.fr.native_padding import real_sps_per_cell

        monkeypatch.setenv("AFCFD_PRISM_BASIS", "collapsed")  # 2026-09-20 起默认已是 native，坍缩档必须显式指定
        n_prism, n_tet = real_sps_per_cell(order)
        assert n_prism == (order + 1) ** 3, "坍缩棱柱没有填充槽位"
        assert n_tet == (order + 1) * (order + 2) * (order + 3) // 6

    @pytest.mark.parametrize("order", [1, 2, 3, 4])
    def test_native_prism_reports_its_own_count(self, order, monkeypatch):
        from autoflowcfd.fr.native_padding import real_sps_per_cell

        monkeypatch.setenv("AFCFD_PRISM_BASIS", "native")
        n_prism, _ = real_sps_per_cell(order)
        assert n_prism == native_prism_n_sps(order)
        assert n_prism < (order + 1) ** 3

    def test_reduction_excludes_prism_padding_under_native(self, monkeypatch):
        """决定性判据：棱柱填充槽位塞进一个巨大值，均值必须不受影响。"""
        from autoflowcfd.fr.native_padding import (
            reduce_per_cell_over_real_sps,
        )

        monkeypatch.setenv("AFCFD_PRISM_BASIS", "native")
        f = np.zeros((4, 27))
        f[:, :18] = 2.0
        f[:2, 18:] = 1e6          # 棱柱段的填充槽位：变馊的初值
        got = reduce_per_cell_over_real_sps(f, 2, 2, "mean")
        assert np.allclose(got[:2], 2.0), (
            f"棱柱填充槽位被算进均值了：{got[:2]}")

    def test_collapsed_reduction_is_bit_identical_to_plain_mean(
            self, monkeypatch):
        """坍缩模式下必须与"直接对整个 SP 轴求均值"**逐位**相同。

        这条保证改动对已长期验证的默认路径零影响。
        """
        from autoflowcfd.fr.native_padding import (
            reduce_per_cell_over_real_sps,
        )

        monkeypatch.setenv("AFCFD_PRISM_BASIS", "collapsed")  # 2026-09-20 起默认已是 native，坍缩档必须显式指定
        rng = np.random.default_rng(11)
        f = rng.normal(size=(6, 27))
        got = reduce_per_cell_over_real_sps(f, 4, 2, "mean")
        assert np.array_equal(got[:4], f[:4].mean(axis=1))

    def test_row_masked_reduction_excludes_prism_padding(self, monkeypatch):
        """逐行掩码版（按 owner_cell 索引出来的逐面数组）同样要正确。"""
        from autoflowcfd.fr.native_padding import reduce_rows_over_real_sps

        monkeypatch.setenv("AFCFD_PRISM_BASIS", "native")
        f = np.zeros((3, 27))
        f[:, :10] = 5.0
        f[:, 10:18] = 5.0
        f[:, 18:] = -1e6
        is_prism = np.array([True, False, True])
        got = reduce_rows_over_real_sps(f, is_prism, 2, "mean")
        assert np.allclose(got, 5.0), got

    def test_bad_mode_raises_rather_than_falling_back(self, monkeypatch):
        """拼错环境变量必须报错。

        静默退回默认值会让 A/B 对照失去意义 —— 本项目已经吃过一次
        "固定 CFL 请求被静默丢弃、两条不同配置给出逐位相同轨迹"的亏。
        """
        from autoflowcfd.fr.native_prism.mode import resolve_prism_basis_mode

        monkeypatch.setenv("AFCFD_PRISM_BASIS", "Native ")
        assert resolve_prism_basis_mode() == "native", "应当容忍大小写与空格"
        monkeypatch.setenv("AFCFD_PRISM_BASIS", "nativ")
        with pytest.raises(ValueError, match="AFCFD_PRISM_BASIS"):
            resolve_prism_basis_mode()


class TestOperatorsAndGeometrySwitchTogether:
    """`FROperators` 与 `build_order_geometry` 必须**一起**切换棱柱基。

    原生 `D` 作用在**原生棱柱节点**上的场，坍缩档的 `sps_coords`/
    `jacobians` 建在张量积立方体节点上。把原生算子套到坍缩节点采样的场上
    不会报错，只会给出错的导数 —— 所以两边读同一个 `AFCFD_PRISM_BASIS`，
    这里钉住"确实读到了同一个值"。
    """

    @staticmethod
    def _ops(order):
        from autoflowcfd.fr.operators import generate_fr_operators

        return generate_fr_operators(order)

    def test_collapsed_leaves_every_native_field_none(self, monkeypatch):
        monkeypatch.setenv("AFCFD_PRISM_BASIS", "collapsed")  # 2026-09-20 起默认已是 native，坍缩档必须显式指定
        ops = self._ops(2)
        assert ops.prism_basis_mode == "collapsed"
        for name in ("D_native_prism", "ref_native_prism",
                     "n_native_sps_prism", "boundary_extrap_native_prism",
                     "lift_native_prism", "D_native_prism_padded",
                     "lift_native_prism_padded",
                     "filter_native_prism_padded"):
            assert getattr(ops, name) is None, f"{name} 应当是 None"
        assert np.abs(ops.D_3d_prism).max() > 20.0, "坍缩档的 max|D| 应当很大"

    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_native_aliases_the_old_field_names(self, order, monkeypatch):
        """`D_3d_prism`/`filter_prism` 必须别名到填充好的原生版本。

        这是让"任何无条件读旧字段名的消费点自动拿到原生结果"成立的关键，
        与四面体当年完全同一个做法。
        """
        monkeypatch.setenv("AFCFD_PRISM_BASIS", "native")
        ops = self._ops(order)
        n_global = (order + 1) ** 3
        assert ops.prism_basis_mode == "native"
        assert ops.D_3d_prism is ops.D_native_prism_padded
        assert ops.filter_prism is ops.filter_native_prism_padded
        assert ops.D_3d_prism.shape == (n_global, n_global, 3)
        assert ops.filter_prism.shape == (n_global, n_global)
        assert ops.n_native_sps_prism == native_prism_n_sps(order)

    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_padding_block_is_exactly_zero(self, order, monkeypatch):
        """填充块必须**恰好**为零（"零填充块对角"不变量）。

        行填零 -> 填充槽位的残差恒为零、不被时间推进改写；
        列填零 -> 填充行里任何有限数值都不污染真实输出。
        """
        monkeypatch.setenv("AFCFD_PRISM_BASIS", "native")
        ops = self._ops(order)
        nr = ops.n_native_sps_prism
        D = ops.D_native_prism_padded
        assert np.all(D[nr:, :, :] == 0.0)
        assert np.all(D[:, nr:, :] == 0.0)
        for mat in ops.lift_native_prism_padded.values():
            assert np.all(mat[nr:, :] == 0.0)

    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_native_operator_magnitude_is_far_smaller(self, order,
                                                      monkeypatch):
        """`max|D_3d_prism|` 必须大幅下降 —— 自由流保持性的直接控制量
        （实测误差严格等于 `eps * max|D| / det(J)`）。
        """
        monkeypatch.setenv("AFCFD_PRISM_BASIS", "collapsed")  # 2026-09-20 起默认已是 native，坍缩档必须显式指定
        mag_c = float(np.abs(self._ops(order).D_3d_prism).max())
        monkeypatch.setenv("AFCFD_PRISM_BASIS", "native")
        mag_n = float(np.abs(self._ops(order).D_3d_prism).max())
        gain = mag_c / mag_n
        floor = {1: 1.5, 2: 4.0, 3: 30.0}[order]
        assert gain >= floor, (
            f"order={order}: max|D| 只改善了 {gain:.1f} 倍 "
            f"(native {mag_n:.3f} vs collapsed {mag_c:.3f})")

    def test_bad_face_key_is_rejected_not_silently_mapped(self, monkeypatch):
        """按 `(axis, side)` 取原生棱柱面算子只允许走对应表。"""
        from autoflowcfd.fr.native_prism.face import (
            cube_face_to_native_prism_face,
        )

        monkeypatch.setenv("AFCFD_PRISM_BASIS", "native")
        ops = self._ops(2)
        # 5 个合法立方体面都要能取到算子
        for axis in range(3):
            for side in (-1.0, 1.0):
                if (axis, side) == (1, 1.0):
                    continue
                fid = cube_face_to_native_prism_face(axis, side)
                assert fid in ops.boundary_extrap_native_prism
                assert fid in ops.lift_native_prism_padded


class TestNativePrismGeometry:
    """生产几何在原生档下的两条硬性质。

    **网格在坍缩档下建、几何在原生档下重算**：`load_from_volume_mesh`
    在原生档下会撞上面编码的硬护栏（残差 kernel 还没适配，见那条护栏的
    说明）。而几何是 `(mesh, order, mode)` 的**纯函数**，所以直接调
    `build_order_geometry` 就能在不碰面编码的前提下验证它 —— 这不是绕过
    护栏，护栏挡的正是"原生几何配坍缩面编码去跑残差"那件事。
    """

    @staticmethod
    def _geometry(order, monkeypatch):
        import sys

        sys.path.insert(0, "tests/validation")
        from _channel_mesh import build_channel_mesh_prism
        from autoflowcfd.grid.high_order.high_order_mesh_order import (
            build_order_geometry,
        )

        monkeypatch.setenv("AFCFD_PRISM_BASIS", "collapsed")  # 2026-09-20 起默认已是 native，坍缩档必须显式指定
        mesh = build_channel_mesh_prism(order, nx=3, ny=2, nz=2,
                                        Lx=0.1, H=0.01, Lz=0.004)
        monkeypatch.setenv("AFCFD_PRISM_BASIS", "native")
        return mesh, build_order_geometry(mesh, order)

    def test_right_prism_metric_is_constant_through_production_geometry(
            self, monkeypatch):
        """挤出网格（右棱柱）的 `det_jacs` 在整张网格上**恒定**。

        坍缩档做不到：它的参考点在退化边附近聚集，同一张网格里 `det_jacs`
        实测变化 7.9 倍。度量恒定 + `D@1` 机器零 = 均匀流下体积项散度机器零，
        这就是"自由流保持性"的结构性依据。
        """
        _mesh, geom = self._geometry(2, monkeypatch)
        det = np.abs(geom["jacobians"]["det_jacs"])
        assert det.min() > 0.0
        spread = det.max() / det.min()
        assert spread < 1.0 + 1e-12, f"度量不恒定，极值比 {spread:.6f}"

    def test_collapsed_metric_really_does_vary(self, monkeypatch):
        """自检：同一张网格在坍缩档下 `det_jacs` **确实**随点变化。

        否则上面那条"恒定"判据无从判断是原生基的功劳还是网格太规整。
        """
        import sys

        sys.path.insert(0, "tests/validation")
        from _channel_mesh import build_channel_mesh_prism
        from autoflowcfd.grid.high_order.high_order_mesh_order import (
            build_order_geometry,
        )

        monkeypatch.setenv("AFCFD_PRISM_BASIS", "collapsed")  # 2026-09-20 起默认已是 native，坍缩档必须显式指定
        mesh = build_channel_mesh_prism(2, nx=3, ny=2, nz=2,
                                        Lx=0.1, H=0.01, Lz=0.004)
        det = np.abs(build_order_geometry(mesh, 2)["jacobians"]["det_jacs"])
        assert det.max() / det.min() > 2.0, (
            f"坍缩档的度量极值比只有 {det.max() / det.min():.3f}，"
            f"这张网格区分不出两条基")

    def test_padding_slots_copy_real_sp0(self, monkeypatch):
        """填充槽位必须复制真实 SP #0（有限、物理上合法的占位值）。

        留 `np.zeros` 默认值对应原点，会被后处理/可视化误当成真实几何
        位置；而 `det_jacs` 是**除数**，留 0 直接是除零。
        """
        mesh, geom = self._geometry(2, monkeypatch)
        n_sps = mesh.n_sps_per_cell
        n_real = native_prism_n_sps(2)
        n_prism = mesh.n_prism_cells
        assert n_real < n_sps
        det = geom["jacobians"]["det_jacs"].reshape(-1, n_sps)[:n_prism]
        coords = geom["sps_coords"][:n_prism]
        assert np.array_equal(det[:, n_real:],
                              np.repeat(det[:, :1], n_sps - n_real, axis=1))
        assert np.array_equal(
            coords[:, n_real:],
            np.repeat(coords[:, :1], n_sps - n_real, axis=1))
        assert np.all(np.isfinite(det)) and np.all(np.abs(det) > 0.0)

    @pytest.mark.parametrize("order", [1, 2])
    def test_solver_path_builds_end_to_end_under_native(self, monkeypatch,
                                                        order):
        """原生档下建带面的网格必须**成功**，并且确实走的是原生那一套。

        这条曾经是一条"必须硬失败"的护栏测试（2026-09-18）：当时残差
        kernel 判断"是不是 native 面"的判据是 `code >= 6`，原生棱柱面的
        10~14 号编码会被当成四面体面、按 `code-6` 取到 4~8 行，而那些
        数组只有 4 行、numba nopython 不做边界检查。

        护栏已于 2026-09-19 移除 —— 原生算子改成**堆叠**成一个数组
        （行 0~3 四面体、行 4~8 棱柱，索引恒为 `code - 6`），于是既有的
        `code >= 6` 写法对两类原生面原样成立。这里改成正向判据，并额外
        钉住三件事，任何一件退化都说明分派又断了：

          1. 面编码里真的出现了原生棱柱编码 [10,15)（而不是悄悄退回坍缩）；
          2. `n_sps_per_cell_fine` 是**原生**细点数 `(oo+1)^2(oo+2)/2`
             而不是坍缩的 `(oo+1)^3`（过积分接线生效的标志）；
          3. 六个 `overint_*` 算子全部非 None，且棱柱段的 `n_fine` 与上面
             那个布局宽度相等 —— 缺一个会让整条过积分链（**含四面体段**）
             静默退回 coarse 路径。
        """
        import sys

        sys.path.insert(0, "tests/validation")
        from _channel_mesh import build_channel_mesh_prism

        from autoflowcfd.core.fr_operators.volume_contract import (
            get_overintegration_context,
        )
        from autoflowcfd.fr.overintegration_order import (
            prism_n_fine, resolve_prism_overintegration_order,
        )
        from autoflowcfd.grid.connectivity.face_connectivity import (
            NATIVE_PRISM_FACE_CODE_RANGE,
        )

        monkeypatch.setenv("AFCFD_PRISM_BASIS", "native")
        mesh = build_channel_mesh_prism(order, nx=2, ny=2, nz=2,
                                        Lx=0.05, H=0.01, Lz=0.004)

        lo, hi = NATIVE_PRISM_FACE_CODE_RANGE
        codes = np.concatenate([
            np.asarray(mesh.face_connectivity.owner_cube_face),
            np.asarray(mesh.face_connectivity.neighbor_cube_face)])
        assert np.any((codes >= lo) & (codes < hi)), (
            "原生档下面编码里没有任何原生棱柱面 —— 说明 "
            "with_native_face_codes 的 prism_native 分派没生效")

        oo = resolve_prism_overintegration_order(order)
        expect_fine = prism_n_fine(oo)
        assert expect_fine == (oo + 1) ** 2 * (oo + 2) // 2
        assert expect_fine < (oo + 1) ** 3, "原生细点数应当少于坍缩张量积"
        assert mesh.n_sps_per_cell_fine == expect_fine

        ctx = get_overintegration_context(mesh, mesh.operators)
        assert ctx is not None, (
            "六个 overint_* 算子有缺失 —— 过积分会整体（含四面体段）"
            "静默退回 coarse 路径")
        prism_seg = ctx["segs"][0]
        assert prism_seg[2] == expect_fine


class TestTwoPrismMapsAreTheSameGeometry:
    """坍缩 `(a,b,c)` 与原生 `(r,s,t)` 描述的是**同一个**几何映射。

    这条是让整个"面通量点定位层"可以原样复用的支点：坍缩档用 Newton 在
    `(a,b,c)` 里定位面点（还有 numba 预计算路径），而
    `(r,s) = cube_to_tri_rs(a,b)`、`t = c` 是闭式变换 —— 只要两个映射恒等，
    原生档就不需要另写一套定位器，只需要把定位结果换算过去、再换成原生
    Vandermonde。

    所以这条**必须**是逐位恒等而不是"数值接近"：它是复用而不是近似替代的
    依据。
    """

    _NODES = np.array([
        [0.00, 0.00, 0.00], [1.00, 0.10, 0.00], [0.20, 0.00, 1.00],
        [0.05, 1.00, 0.00], [1.10, 1.20, 0.10], [0.10, 1.00, 1.20],
    ])

    @staticmethod
    def _sample(n=2000, seed=3):
        rng = np.random.default_rng(seed)
        return rng.uniform(-1.0, 1.0, size=(n, 3))

    def test_maps_agree_bit_for_bit(self):
        from autoflowcfd.fr.native_prism.basis import (
            map_native_prism_to_physical,
        )
        from autoflowcfd.grid.curved_mapping.curved_mapping import (
            cube_to_tri_rs,
            map_prism_to_physical,
        )

        abc = self._sample()
        r, s = cube_to_tri_rs(abc[:, 0], abc[:, 1])
        rst = np.column_stack([r, s, abc[:, 2]])
        p_col = map_prism_to_physical(abc, self._NODES)
        p_nat = map_native_prism_to_physical(rst, self._NODES)
        assert np.array_equal(p_col, p_nat), (
            f"两个映射不是逐位恒等，max|diff|="
            f"{float(np.abs(p_col - p_nat).max()):.3e} —— 复用定位层的依据"
            f"不成立")

    def test_metric_relation_is_the_duffy_jacobian(self):
        """`det_collapsed = det_native * (1-b)/2`。

        `(r,s,t) <- (a,b,c)` 的雅可比是三角阵，行列式恰好
        `(dr/da)(ds/db)(dt/dc) = (1-b)/2`。这条同时交叉验证两个解析雅可比
        实现（任一处抄错公式都会让比值偏离）。
        """
        from autoflowcfd.fr.native_prism.basis import (
            native_prism_exact_jacobian,
        )
        from autoflowcfd.grid.curved_mapping.curved_mapping import (
            cube_to_tri_rs,
        )
        from autoflowcfd.grid.curved_mapping.curved_mapping_exact_jacobian import (
            prism_exact_jacobian,
        )

        abc = self._sample(n=500, seed=5)
        # 避开退化边附近（那里 (1-b)/2 -> 0，比值本身失去意义）
        abc = abc[abc[:, 1] < 0.9]
        r, s = cube_to_tri_rs(abc[:, 0], abc[:, 1])
        rst = np.column_stack([r, s, abc[:, 2]])
        d_col = np.linalg.det(prism_exact_jacobian(abc, self._NODES))
        d_nat = np.linalg.det(native_prism_exact_jacobian(rst, self._NODES))
        expect = d_nat * (1.0 - abc[:, 1]) / 2.0
        rel = np.abs(d_col / expect - 1.0)
        assert float(rel.max()) < 1e-12, (
            f"度量关系不成立，最大相对偏差 {float(rel.max()):.3e}")

    def test_collapsed_metric_degenerates_where_native_does_not(self):
        """同一批点上，坍缩度量在 `b -> 1` 附近趋零，原生几乎不变。

        这就是"坍缩棱柱在退化边一带病态"的直接来源：`1/det(J)` 在残差里
        是个因子，度量趋零处它被放大。

        判据是**两者的散布之比**，不是"原生恒定"：本测试单元刻意取得不
        规则（顶面不是底面的纯平移），原生度量本来就会随点小幅变化 ——
        逐点恒定只对右棱柱成立，那条由
        `TestNativePrismGeometry::test_right_prism_metric_is_constant_
        through_production_geometry` 单独覆盖。
        """
        from autoflowcfd.fr.native_prism.basis import (
            native_prism_exact_jacobian,
        )
        from autoflowcfd.grid.curved_mapping.curved_mapping import (
            cube_to_tri_rs,
        )
        from autoflowcfd.grid.curved_mapping.curved_mapping_exact_jacobian import (
            prism_exact_jacobian,
        )

        b_vals = np.array([-0.9, 0.0, 0.9, 0.99, 0.999])
        abc = np.column_stack([np.zeros_like(b_vals), b_vals,
                               np.zeros_like(b_vals)])
        r, s = cube_to_tri_rs(abc[:, 0], abc[:, 1])
        rst = np.column_stack([r, s, abc[:, 2]])
        d_col = np.abs(np.linalg.det(prism_exact_jacobian(abc, self._NODES)))
        d_nat = np.abs(np.linalg.det(
            native_prism_exact_jacobian(rst, self._NODES)))
        assert d_col[-1] / d_col[0] < 1e-2, (
            f"坍缩度量没有在退化边附近趋零：{d_col}")
        spread_col = d_col.max() / d_col.min()
        spread_nat = d_nat.max() / d_nat.min()
        assert spread_nat < 1.1, (
            f"原生度量散布 {spread_nat:.4f} 过大（这批点在同一个单元里）")
        assert spread_col / spread_nat > 100.0, (
            f"坍缩散布 {spread_col:.1f} 相对原生 {spread_nat:.4f} 只差 "
            f"{spread_col / spread_nat:.1f} 倍，两条基区分不开")


class TestFaceAdjRowsAndWeights:
    """面的 `adj_row` 与参考求积权重。

    **判据用闭合面恒等式** `sum_faces sum_fp (adj_row * w_ref) = 0`：
    它一次把三件事同时钉死 ——
      * 5 个面的参考余向量**符号**（任一个反了都会留下 O(总面积) 的残差）；
      * 三角形封盖的 **Duffy 因子** `(1-s)/2`（漏掉会留下 O(1) 残差）；
      * 斜边面**不归一化**余向量的处理（`|(1,1,0)|=sqrt2` 恰好补偿
        `lambda` 参数化的参考边长元）。

    这就是几何守恒律（GCL）在面一级的形式：闭合曲面上 `∮ n dA = 0`。
    """

    #: **正定向**的贴壁薄右棱柱（三角形 V0->V1->V2 与挤出方向成右手系，
    #: `det(J) = +3.125e-11`）。定向是 `adj_row` 朝外的**前提**：`det(J)<0`
    #: 时 `adj = det * inv(J)` 整体反号、法向全部朝内，而闭合面恒等式
    #: **测不出**这种全局翻转（全翻之后求和仍然为零）—— 所以下面既有闭合
    #: 恒等式、也必须有逐面朝向的绝对核对。生产路径由
    #: `curved_mapping_orientation.fix_prism_orientation` 保证定向，两条基的
    #: `det` 同号（相差 `(1-b)/2 > 0`），所以那个既有修正对原生基同样有效。
    _RIGHT_THIN = np.vstack([
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.5e-3], [1e-2, 0.0, 0.0]]),
        np.array([[0.0, 1e-5, 0.0], [0.0, 1e-5, 2.5e-3], [1e-2, 1e-5, 0.0]]),
    ])
    _IRREGULAR = np.array([
        [0.00, 0.00, 0.00], [1.00, 0.10, 0.00], [0.20, 0.00, 1.00],
        [0.05, 1.00, 0.00], [1.10, 1.20, 0.10], [0.10, 1.00, 1.20],
    ])

    @staticmethod
    def _plain_weights(order):
        """**平凡**张量积求积权重 —— 全部 15 种面编码共用的那一套。

        原生棱柱三角封盖的 Duffy 因子 `(1-s)/2` 现在乘在 `adj_row` 里
        （2026-09-19，与 native 四面体三角形面统一约定，见
        `native_prism/face.py` 里 `_FACE_REF_COVECTOR` 上方那段实测
        依据），所以这里配的就是下游 `FlatFaceGeometry.ref_area_weight`
        那一份，不再有逐面的参考权重函数。
        """
        from autoflowcfd.fr.quadrature_points import gauss_legendre

        _pts, w1d = gauss_legendre(order + 1)
        w1, w2 = np.meshgrid(w1d, w1d, indexing="ij")
        return (w1 * w2).ravel()

    @classmethod
    def _accumulate(cls, order, nodes):
        from autoflowcfd.fr.native_prism.face import (
            native_prism_face_adj_rows,
        )

        w = cls._plain_weights(order)
        total = np.zeros(3)
        area = 0.0
        for f in PRISM_FACE_IDS:
            adj = native_prism_face_adj_rows(order, f, nodes)
            total += (adj * w[:, None]).sum(axis=0)
            area += float((np.linalg.norm(adj, axis=1) * w).sum())
        return total, area

    @pytest.mark.parametrize("order", [1, 2, 3])
    @pytest.mark.parametrize("cell", ["right_thin", "irregular"])
    def test_closed_surface_identity(self, order, cell):
        nodes = (self._RIGHT_THIN if cell == "right_thin"
                 else self._IRREGULAR)
        total, area = self._accumulate(order, nodes)
        rel = float(np.linalg.norm(total)) / max(area, 1e-300)
        assert rel < 1e-13, (
            f"order={order} {cell}: |sum n dA| / 总面积 = {rel:.3e}"
            f"（余向量符号 / Duffy 因子 / 权重三者之一出错）")

    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_total_area_matches_the_analytic_value(self, order):
        """总面积必须等于手算值 —— 闭合恒等式对"整体缩放"不敏感，
        必须再钉一条绝对量，否则权重整体差一个常数因子也能通过。
        """
        dx, dz, dy = 1e-2, 2.5e-3, 1e-5
        cap = 0.5 * dx * dz                       # 三角形面积
        edges = dx + dz + np.hypot(dx, dz)        # 三条边长
        expect = 2.0 * cap + edges * dy
        _total, area = self._accumulate(order, self._RIGHT_THIN)
        assert abs(area / expect - 1.0) < 1e-12, (
            f"order={order}: 总面积 {area:.6e} vs 解析 {expect:.6e}")

    @pytest.mark.parametrize("order", [1, 2])
    def test_flipping_one_covector_breaks_the_identity(self, order,
                                                       monkeypatch):
        """反转任一个面的余向量必须让恒等式**失败** —— 否则这条判据
        对符号错误没有区分力。
        """
        from autoflowcfd.fr.native_prism import face as npf

        for f in PRISM_FACE_IDS:
            orig = npf._FACE_REF_COVECTOR[f]
            monkeypatch.setitem(npf._FACE_REF_COVECTOR, f,
                                (-orig[0], orig[1]))
            total, area = self._accumulate(order, self._IRREGULAR)
            rel = float(np.linalg.norm(total)) / area
            monkeypatch.setitem(npf._FACE_REF_COVECTOR, f, orig)
            assert rel > 1e-3, (
                f"face {f} 的余向量反转后恒等式仍然通过（相对 {rel:.3e}），"
                f"判据对符号错误没有区分力")

    @pytest.mark.parametrize("order", [1, 2])
    def test_dropping_the_duffy_factor_breaks_the_identity(self, order,
                                                           monkeypatch):
        """去掉三角形封盖的 Duffy 因子必须让恒等式失败。

        因子现在乘在 `adj_row` 里（由 `_FACE_REF_COVECTOR` 的 `is_cap`
        标记控制），所以"去掉"的做法是把两个封盖的标记改成 False。
        """
        from autoflowcfd.fr.native_prism import face as npf

        for f in (0, 1):
            cov, _is_cap = npf._FACE_REF_COVECTOR[f]
            monkeypatch.setitem(npf._FACE_REF_COVECTOR, f, (cov, False))
        total, area = self._accumulate(order, self._IRREGULAR)
        assert float(np.linalg.norm(total)) / area > 1e-3, (
            "漏掉 Duffy 因子后恒等式仍然通过，判据没有区分力")

    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_plain_weight_gives_the_exact_area_like_native_tet(self, order):
        """**约定的钉子**：平凡张量积权重直接给出精确面积。

        这就是"Duffy 因子必须在 adj 行里"的判据。native 四面体三角形面
        本来就满足它（`_native_tet_adj_row_batched` 的行里含着因子，实测
        四个面的 `sum_p w_p |adj_row_p|` 到 1e-15 等于精确三角面积），
        棱柱如果把因子放在权重里，这里的单位直棱柱底/顶面就会算成
        1.000000 而不是 0.500000 —— 恰好差 2 倍。

        下游只有**一套** `(n_fp,)` 的参考权重（`FlatFaceGeometry.
        ref_area_weight`、`compute_exact_face_normals_and_weights` 的
        `true_area_weight = mag * w_fp`、GPU 侧同名字段），对全部 15 种
        面编码共用，所以这条不成立就等于面积权重错。
        """
        from autoflowcfd.fr.native_prism.face import (
            native_prism_face_adj_rows,
        )

        unit = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0],
        ])
        expect = {0: 0.5, 1: 0.5, 2: np.sqrt(2.0), 3: 1.0, 4: 1.0}
        w = self._plain_weights(order)
        for f in PRISM_FACE_IDS:
            adj = native_prism_face_adj_rows(order, f, unit)
            got = float((np.linalg.norm(adj, axis=1) * w).sum())
            assert abs(got / expect[f] - 1.0) < 1e-12, (
                f"order={order} face {f}: 平凡权重给出面积 {got:.6f}，"
                f"精确值 {expect[f]:.6f}")

    def test_adj_rows_point_outward(self):
        """右棱柱上逐面核对法向朝外（与手算方向比对）。"""
        from autoflowcfd.fr.native_prism.face import (
            native_prism_face_adj_rows,
        )

        expect = {
            0: np.array([0.0, -1.0, 0.0]),   # t=-1 -> 底面三角形（y=0），-y
            1: np.array([0.0, 1.0, 0.0]),    # t=+1 -> 顶面三角形，+y
            3: np.array([0.0, 0.0, -1.0]),   # r=-1 -> 边 V0V2 沿 x、在 z=0 上
            4: np.array([-1.0, 0.0, 0.0]),   # s=-1 -> 边 V0V1 沿 z、在 x=0 上
        }
        for f, n_exp in expect.items():
            adj = native_prism_face_adj_rows(2, f, self._RIGHT_THIN)
            n = adj / np.linalg.norm(adj, axis=1, keepdims=True)
            assert np.allclose(n, n_exp[None, :], atol=1e-12), (
                f"face {f} 法向 {n[0]} 不是期望的 {n_exp}")
        # 斜边面：法向在 x-z 平面内、指向远离原点一侧
        adj = native_prism_face_adj_rows(2, 2, self._RIGHT_THIN)
        n = adj / np.linalg.norm(adj, axis=1, keepdims=True)
        assert np.allclose(n[:, 1], 0.0, atol=1e-12), "斜边面法向不该有 y 分量"
        assert np.all(n[:, 0] > 0.0) and np.all(n[:, 2] > 0.0), (
            f"斜边面法向 {n[0]} 应当同时指向 +x 与 +z")

    def test_negative_orientation_flips_every_normal_inward(self):
        """负定向单元的法向**全部朝内** —— 记录"定向是前提"这件事。

        闭合面恒等式对全局翻转没有区分力（全翻之后求和仍然为零），所以
        这条单独钉住。生产路径由 `fix_prism_orientation` 保证定向。
        """
        from autoflowcfd.fr.native_prism.basis import (
            build_native_prism_nodes,
            native_prism_exact_jacobian,
        )
        from autoflowcfd.fr.native_prism.face import (
            native_prism_face_adj_rows,
        )

        flipped = self._RIGHT_THIN[[0, 2, 1, 3, 5, 4]]
        det = np.linalg.det(
            native_prism_exact_jacobian(build_native_prism_nodes(2), flipped))
        assert np.all(det < 0.0), "构造的对照单元并不是负定向"
        adj = native_prism_face_adj_rows(2, 0, flipped)
        n = adj / np.linalg.norm(adj, axis=1, keepdims=True)
        assert np.allclose(n, np.array([0.0, 1.0, 0.0])[None, :], atol=1e-12), (
            f"负定向单元底面法向 {n[0]} 应当朝内（+y）")
        # 而闭合恒等式照样成立 —— 这就是它测不出全局翻转的证据
        total, area = self._accumulate(2, flipped)
        assert float(np.linalg.norm(total)) / area < 1e-13
