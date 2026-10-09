"""原生棱柱基：两套棱柱映射是同一几何、面邻接行与求积权重（从 test_native_prism_face.py 拆出，背景见该文件模块文档）。"""

import numpy as np
import pytest

from autoflowcfd.fr.native_prism.face import PRISM_FACE_IDS


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
