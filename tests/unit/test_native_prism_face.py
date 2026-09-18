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
`fr/native_triangle_basis.py` 模块文档。
"""

import numpy as np
import pytest

from autoflowcfd.fr.native_prism_basis import (
    build_native_prism_nodes,
    build_native_prism_vandermonde,
    native_prism_exact_jacobian,
    native_prism_n_sps,
    restricted_prism_modes,
)
from autoflowcfd.fr.native_prism_face import (
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
        from autoflowcfd.fr.native_prism_basis import (
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
        from autoflowcfd.fr.native_prism_face import (
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
        from autoflowcfd.fr.native_prism_face import (
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
        from autoflowcfd.fr.native_prism_face import (
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

        monkeypatch.delenv("AFCFD_PRISM_BASIS", raising=False)
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

        monkeypatch.delenv("AFCFD_PRISM_BASIS", raising=False)
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
        from autoflowcfd.fr.prism_basis_mode import resolve_prism_basis_mode

        monkeypatch.setenv("AFCFD_PRISM_BASIS", "Native ")
        assert resolve_prism_basis_mode() == "native", "应当容忍大小写与空格"
        monkeypatch.setenv("AFCFD_PRISM_BASIS", "nativ")
        with pytest.raises(ValueError, match="AFCFD_PRISM_BASIS"):
            resolve_prism_basis_mode()
