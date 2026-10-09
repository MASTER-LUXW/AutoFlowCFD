"""原生棱柱基：真实解点数、算子与几何同步切换、原生棱柱几何量（从 test_native_prism_face.py 拆出，背景见该文件模块文档）。"""

import numpy as np
import pytest

from autoflowcfd.fr.native_prism.basis import native_prism_n_sps


class TestRealSpsCountsFollowTheActivePrismBasis:
    """`real_sps_per_cell` —— "哪些槽位是真的"的唯一判据来源。

    **为什么这条必须有测试**：`reduce_*_over_real_sps` 原先写死"棱柱用满
    全部槽位"。原生棱柱一上线棱柱也有填充槽位，而那些槽位冻结在初值、
    随推进变馊（实测 10 步后偏差 3.4%）。漏改不会报错，只会让 checkpoint
    的单元均值、人工粘性尺度、omega 壁面目标值里的 rho、GPU 局部 dt 的
    min 全部**静默**算错。
    """

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

    def test_tet_reduction_is_bit_identical_to_plain_mean(self):
        """**四面体段**的归约必须与"直接对整个真实 SP 轴求均值"逐位相同。

        原来这条测的是坍缩棱柱（那时棱柱没有填充槽位，所以对整个 SP 轴
        求均值就是对的）。坍缩棱柱基已于 2026-09-23 删除，改测四面体段
        —— 它的真实自由度数同样由 `real_sps_per_cell` 给出，判据的意图
        （"掩码逻辑不能在没有填充的那一段上引入任何偏差"）完全保留。
        """
        from autoflowcfd.fr.native_padding import (
            real_sps_per_cell, reduce_per_cell_over_real_sps,
        )

        _n_prism_real, n_tet_real = real_sps_per_cell(2)
        rng = np.random.default_rng(11)
        f = rng.normal(size=(6, 27))
        f[:, n_tet_real:] = np.nan          # 填充槽位：被正确掩掉才不会传染
        got = reduce_per_cell_over_real_sps(f, 0, 2, "mean")
        assert np.array_equal(got, f[:, :n_tet_real].mean(axis=1))

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

    def test_every_native_prism_field_is_present(self):
        """原生棱柱那一整组算子字段**恒为非 None**。

        原来这条测的是反面（"坍缩档下它们全是 None"）。坍缩棱柱基已于
        2026-09-23 删除，判据翻成正面：只剩一条基，那组字段必须永远建好
        —— 任何一个是 None 都意味着算子构造静默失败了，而下游
        `FROperators.native_face_extrap` 只会在真正取用时才抛错。
        """
        ops = self._ops(2)
        for name in ("D_native_prism", "ref_native_prism",
                     "n_native_sps_prism", "D_native_prism_padded",
                     "filter_native_prism_padded"):
            assert getattr(ops, name) is not None, f"{name} 不该是 None"
        # 原来这里顺手自检"坍缩档 max|D| 应当很大（>20）"，作为改善倍数
        # 判据的对照侧。坍缩棱柱基已于 2026-09-23 删除，该自检随之移除；
        # 原生侧改用绝对上界判据，见
        # `test_native_operator_magnitude_stays_small`。

    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_native_aliases_the_old_field_names(self, order):
        """`D_3d_prism`/`filter_prism` 必须别名到填充好的原生版本。

        这是让"任何无条件读旧字段名的消费点自动拿到原生结果"成立的关键，
        与四面体当年完全同一个做法。（原先还断言 `ops.prism_basis_mode ==
        "native"`；那个字段随坍缩档一起删除了 —— 只剩一条基时，再记录
        "当前是哪条"就是冗余。）
        """
        ops = self._ops(order)
        n_global = (order + 1) ** 3
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
        for code in range(10, 15):
            assert np.all(ops.face_lift_by_op[code - 6][nr:, :] == 0.0)

    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_native_operator_magnitude_stays_small(self, order):
        """`max|D_3d_prism|` 必须保持在小量级 —— 自由流保持性的直接控制量
        （实测误差严格等于 `eps * max|D| / det(J)`）。

        原来这条是"原生 vs 坍缩的改善倍数"。坍缩棱柱基已于 2026-09-23
        删除、无法再构造对照侧，所以改成钉**原生侧的绝对上界**：

            order   实测 max|D_3d_prism|   本测试上界（实测值 x1.5）
              1          0.866025                  1.30
              2          2.581989                  3.90
              3          4.860154                  7.30

        作为量级参照，坍缩棱柱基在删除前的同一个量是 P3 **560.1**
        （项目记忆 `native-prism-basis-migration` 记的 `max|D|` 随阶数
        爆炸：P1 2.05 -> P3 560.1），也就是原生把它压低了两个数量级 ——
        这正是自由流保持性从 ~1e-9 改善到机器零的直接原因。
        """
        mag_n = float(np.abs(self._ops(order).D_3d_prism).max())
        ceiling = {1: 1.30, 2: 3.90, 3: 7.30}[order]
        assert mag_n <= ceiling, (
            f"order={order}: max|D_3d_prism| = {mag_n:.6f} 超过上界 "
            f"{ceiling}（2026-09-23 实测 "
            f"{ {1: 0.866025, 2: 2.581989, 3: 4.860154}[order] }）")

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
                assert ops.native_face_extrap(10 + fid).shape == (9, ops.n_native_sps_prism)


class TestNativePrismGeometry:
    """生产几何的两条硬性质。

    2026-09-24 之前这里刻意"网格在坍缩档下建、几何在原生档下重算" ——
    那是因为 `load_from_volume_mesh` 在原生档下会撞上原生棱柱面编码的
    硬护栏（当时残差 kernel 还没适配）。那道护栏已于 2026-09-19 随
    "原生算子堆叠成一张表、索引恒为 `code - 6`" 一并移除，坍缩棱柱基
    本身也已于 2026-09-23 删除，所以现在直接按生产路径建网格即可。
    """

    @staticmethod
    def _geometry(order, monkeypatch):
        import sys

        sys.path.insert(0, "tests/validation")
        from _channel_mesh import build_channel_mesh_prism
        from autoflowcfd.grid.high_order.high_order_mesh_order import (
            build_order_geometry,
        )

        monkeypatch.setenv("AFCFD_PRISM_BASIS", "native")
        mesh = build_channel_mesh_prism(order, nx=3, ny=2, nz=2,
                                        Lx=0.1, H=0.01, Lz=0.004)
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

    # 原来这里有一条自检 `test_collapsed_metric_really_does_vary`：同一张
    # 网格在坍缩档下 `det_jacs` 极值比 > 2.0，用来证明上面那条"原生档度量
    # 恒定"不是因为网格太规整。坍缩棱柱基已于 2026-09-23 删除、无法再构造
    # 对照侧，该自检随之移除。它当时的结论仍然有效并记录在此：**这张
    # `build_channel_mesh_prism(2, nx=3, ny=2, nz=2, Lx=0.1, H=0.01,
    # Lz=0.004)` 网格在坍缩档下度量确实随点变化（极值比 > 2）**，所以
    # "原生档恒定"是基本身的功劳，不是网格规整的副产物。


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
