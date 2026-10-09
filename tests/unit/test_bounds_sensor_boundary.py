"""BJ 型邻居极值判据：边界 Dirichlet 补全包络、边界面排除、滤波门控接线（从 test_bounds_sensor.py 拆出，背景见该文件模块文档）。"""

import numpy as np
import pytest

from autoflowcfd.core.fr_operators.bounds_sensor import compute_bounds_violation_mask

from tests.unit._bounds_sensor_common import _line_mesh


class TestBoundaryDirichletCompletesTheEnvelope:
    """边界单元的包络必须用**物理边界值**补全 —— 真实缺陷回归（2026-09-18）。

    边界面不提供"邻居单元均值"（用幽灵态会把边界条件的物理跳跃、例如
    壁面镜像的法向速度反号，误判成越界）。此前的做法是单纯"排除"，于是
    该单元在那个方向上的包络变成**单侧**的——而本判据的设计不变量是
    "线性场恒不触发"，单侧包络直接破坏了它：贴壁单元靠壁那个解点必然
    低于本单元均值、也低于上方邻居均值，而包络下界恰好就是本单元均值。

    工程后果（Blasius 平板，有精确解，nx=16，400 步，**固定 CFL 0.03、
    同步数**的干净对照；壁面剪应力用胞内两点斜率提取，该提取本身已用
    "把解析解直接放到同一网格上"验证到中位 -0.49%、最大 1.82%）：

        off / sensor+persson（掩码恒空）   du/dy 1312.7   cf  -6.33%
        sensor+bounds（修复前）            du/dy   93.0   cf -93.37%
        sensor+bounds（补上壁面 Dirichlet）du/dy  561.9   cf -74.54%

    全域标记率只有 0.400%，但**贴壁那一层是 14.393%**（集中 36 倍，逐
    变量 rho_u 14.143% / rho_w 0.250%），而壁面剪应力恰好只由这一层决定。
    补上无滑移壁面的动量 Dirichlet 值（静止壁 rho*u_wall = 0）之后贴壁层
    命中率降到 **0.831%**，du/dy 的**最大值 1316.3 已与 `off` 的 1312.7
    吻合**。

    **cf 中位仍是 -74.54%**：余下的差距来自"被标记单元上的动作是精确
    投影、一次把非常数内容清零"（单元内扩散重建梯度约需
    `h^2/nu/dt ~ 26` 步，而 0.83%/stage 平均每 40 步就再清一次，所以
    最大值对了、中位恢复不到位）。那是另一个独立问题，本类只钉包络。

    **两条被自己数据否掉的替代补全方案**（不要再试）：
      1. 用 owner 自身解点极值撑开 -> 边界单元恒不可能被标记，而 `nz=1`
         的准二维算例里每个单元都贴着两个对称面，实测标记率变成全域
         0.000%、判据整个失效；
      2. 把内部包络对本单元均值作**镜像** -> 线性场不变量恢复了（贴壁层
         14.393% -> 2.099%），但贴边界单元内部一个 50 倍的**真实**过冲
         也抓不到——镜像半宽由"本单元均值与邻居均值之差"决定，而单元
         自身就是异常值时这个差正好被它自己撑大。单邻居情形下仅凭单元
         均值在信息上**无法**区分两者，必须引入边界条件这份外部信息。
    """

    def _one_sided_reference(self, q, owner, neigh, bnd, **kw):
        """修复前的做法：边界面单纯排除、不补全。"""
        field = q[:, :, None] if q.ndim == 2 else q
        n_cells = field.shape[0]
        interior = ~bnd
        o_i, n_i = owner[interior], neigh[interior]
        mask = np.zeros(n_cells, dtype=bool)
        rel_tol = kw.get("rel_tol", 0.1)
        abs_frac = kw.get("abs_frac", 1e-3)
        refs = kw.get("ref_scales")
        for v in range(field.shape[2]):
            g = field[:, :, v]
            cm, cx, cn = g.mean(1), g.max(1), g.min(1)
            nb_max, nb_min = cm.copy(), cm.copy()
            np.maximum.at(nb_max, o_i, cm[n_i])
            np.maximum.at(nb_max, n_i, cm[o_i])
            np.minimum.at(nb_min, o_i, cm[n_i])
            np.minimum.at(nb_min, n_i, cm[o_i])
            scale = (float(refs[v]) if refs is not None
                     else float(np.sqrt(np.mean(cm ** 2))))
            tol = rel_tol * (nb_max - nb_min) + abs_frac * max(scale, 1e-300)
            mask |= (cx > nb_max + tol) | (cn < nb_min - tol)
        return mask

    def test_old_one_sided_form_did_flag_boundary_cells(self):
        """先证明这不是假想的问题：旧做法在线性场上**确实**标记两端。"""
        n = 30
        c = np.arange(n, dtype=float)
        b = 1.0
        q = np.stack([2.0 + b * (c - 0.5), 2.0 + b * (c + 0.5)], axis=1)
        owner, neigh, bnd = _line_mesh(n)
        old = self._one_sided_reference(q, owner, neigh, bnd)
        assert old[0] and old[-1], (
            "旧的单侧包络本该在线性场的两端单元上触发——若这条不成立，"
            "下面那条'修复后干净'就没有意义")
        assert not old[1:-1].any(), "旧做法在内部单元上本来是干净的"

    def _linear_dirichlet(self, n, a, b, owner, bnd):
        """线性场 `a + b*x` 在两端边界面上的真实边界值。

        单元 i 形心在 x=i、半宽 0.5，所以左端边界面在 x=-0.5、右端在
        x=(n-1)+0.5。
        """
        table = np.full((owner.size, 1), np.nan)
        x_face = {0: -0.5, n - 1: (n - 1) + 0.5}
        for f in np.flatnonzero(bnd):
            table[f, 0] = a + b * x_face[int(owner[f])]
        return table

    def test_with_dirichlet_values_boundary_cells_are_clean(self):
        """给出真实边界值后，线性场在**两端单元**上也不触发。"""
        n = 30
        c = np.arange(n, dtype=float)
        owner, neigh, bnd = _line_mesh(n)
        for b in (1.0, -3.7, 1e4):
            q = np.stack([2.0 + b * (c - 0.5), 2.0 + b * (c + 0.5)], axis=1)
            table = self._linear_dirichlet(n, 2.0, b, owner, bnd)
            mask = compute_bounds_violation_mask(
                q, owner, neigh, bnd, bnd_dirichlet=table)
            assert not mask.any(), (
                f"b={b}: 给了边界值仍标记了 {mask.sum()} 个（含两端）")

    def test_table_of_all_nan_is_bit_identical_to_not_passing_it(self):
        """全 NaN 的表必须与不传表**逐位相同**。

        NaN 的语义是"退回排除"，而排除对包络初值 `cell_mean` 恰好是
        无操作——这条把该等价性钉住，避免将来 NaN 分支被改成别的行为
        而没人发现。
        """
        rng = np.random.default_rng(11)
        n = 50
        q = rng.normal(size=(n, 3)) * 5.0 + 20.0
        owner, neigh, bnd = _line_mesh(n, n_sps=3)
        m1 = compute_bounds_violation_mask(q, owner, neigh, bnd)
        m2 = compute_bounds_violation_mask(
            q, owner, neigh, bnd,
            bnd_dirichlet=np.full((owner.size, 1), np.nan))
        np.testing.assert_array_equal(m1, m2)

    def test_interior_cells_are_bit_identical_to_the_old_form(self):
        """补全只影响**贴边界**的单元，内部单元必须逐位不变。

        否则这次修复会悄悄改变所有算例的内部行为。
        """
        rng = np.random.default_rng(3)
        n = 60
        q = rng.normal(size=(n, 4)) * 10.0 + 100.0
        owner, neigh, bnd = _line_mesh(n, n_sps=4)
        new = compute_bounds_violation_mask(q, owner, neigh, bnd)
        old = self._one_sided_reference(q, owner, neigh, bnd)
        np.testing.assert_array_equal(
            new[1:-1], old[1:-1],
            err_msg="内部单元的判据被这次修复改变了")

    def test_real_overshoot_at_a_boundary_cell_is_still_caught(self):
        """补全不能把判据在边界单元上废掉：真实过冲仍要被抓到。

        这一条正是另外两个补全方案被否掉的原因——两者都会漏掉这个 50
        倍的过冲。
        """
        n = 20
        q = np.full((n, 2), 1.0)
        q[0] = [1.0, 50.0]                 # 贴边界单元内部一个巨大过冲
        owner, neigh, bnd = _line_mesh(n)
        table = np.full((owner.size, 1), np.nan)
        table[bnd, 0] = 1.0                # 该边界上的真实值就是 1.0
        mask = compute_bounds_violation_mask(
            q, owner, neigh, bnd, bnd_dirichlet=table)
        assert mask[0], "边界单元上的真实过冲没有被抓到——判据被废掉了"


class TestBoundaryFacesExcluded:
    """边界面不参与邻域构造。

    若把边界面也算进去（用 neighbor_cell 的占位值 -1 去索引 cell_mean），
    numpy 的负索引会**静默**取到最后一个单元的均值——那是个安静的错误
    答案，比崩溃更难发现。本类同时钉住"不崩"和"结果与只用内部面一致"。
    """

    def test_placeholder_neighbor_on_boundary_is_not_read(self):
        n = 10
        q = np.full((n, 2), 1.0)
        q[0, :] = 100.0            # 让最后一个单元的均值与首个差别极大
        owner, neigh, bnd = _line_mesh(n)

        m1 = compute_bounds_violation_mask(q, owner, neigh, bnd)
        # 把边界面的占位邻居换成另一个同样非法的值：结果必须逐位相同
        neigh2 = neigh.copy()
        neigh2[bnd] = -999
        m2 = compute_bounds_violation_mask(q, owner, neigh2, bnd)
        assert np.array_equal(m1, m2), "边界面的占位 neighbor 被读取了"


class TestFilterGateWiring:
    """门控入口的接线：bounds 档缺面连接必须报错而不是静默退回 persson。"""

    FS = {'rho_inf': 1.225, 'vel_inf': 33.33, 'p_inf': 101325.0}

    def test_bounds_without_connectivity_raises(self):
        from autoflowcfd.core.fr_solver.filter import (
            build_sensor_gated_filter_func_arrays,
        )

        with pytest.raises(ValueError, match='面连接'):
            build_sensor_gated_filter_func_arrays(
                10, 8, 1, np.eye(8), np.eye(8), n_prism=5, sensor='bounds',
                freestream=self.FS)

    def test_bounds_without_freestream_raises(self):
        """缺 freestream 必须报错而不是静默退回全场 RMS。

        用全场 RMS 做绝对地板在真实 checkpoint 上实测标记了 **11.08%**
        的单元（事先写定的判据上限是 3%），所以"静默退回"等于静默启用
        一个已知不可用的配置。
        """
        from autoflowcfd.core.fr_solver.filter import (
            build_sensor_gated_filter_func_arrays,
        )

        owner, neigh, bnd = _line_mesh(10)
        with pytest.raises(ValueError, match='freestream'):
            build_sensor_gated_filter_func_arrays(
                10, 8, 1, np.eye(8), np.eye(8), n_prism=5, sensor='bounds',
                owner_cell=owner, neighbor_cell=neigh, is_boundary=bnd)

    def test_unknown_sensor_raises(self):
        from autoflowcfd.core.fr_solver.filter import (
            build_sensor_gated_filter_func_arrays,
        )

        with pytest.raises(ValueError, match='未知 sensor'):
            build_sensor_gated_filter_func_arrays(
                10, 8, 1, np.eye(8), np.eye(8), n_prism=5, sensor='nope')

    def test_ref_scales_length_is_validated(self):
        with pytest.raises(ValueError, match='ref_scales 长度'):
            compute_bounds_violation_mask(
                np.ones((5, 2, 5)), *_line_mesh(5), ref_scales=[1.0, 2.0])

    def test_ref_scales_are_used_as_the_floor(self):
        """给出 ref_scales 时地板必须由它决定，而不是由场自身的 RMS。

        构造：场量级 1e6，过冲 0.5。若地板用场 RMS（1e-3 * 1e6 = 1000），
        0.5 的过冲被完全淹没、不触发；若地板用 ref_scales=1（1e-3），
        0.5 远超地板、必须触发。两种结果必须分得开。
        """
        n = 21
        q = np.full((n, 2), 1e6)
        q[10, 1] = 1e6 + 0.5
        assert not compute_bounds_violation_mask(q, *_line_mesh(n))[10]
        assert compute_bounds_violation_mask(
            q, *_line_mesh(n), ref_scales=[1.0])[10]

    def test_bounds_gate_only_touches_flagged_cells(self):
        """真正跑一遍门控回调：只有越界单元被滤波矩阵作用，其余逐位不变。

        用一个"把非常数内容清零"的滤波矩阵（P1 legacy 档的极限形态）
        做算子，这样"被作用过"可以直接由"解点值变成了单元均值"判定。
        """
        from autoflowcfd.core.fr_solver.filter import (
            build_sensor_gated_filter_func_arrays,
        )

        n_cells, n_sps, n_var = 21, 2, 5
        # 清零非常数模态：F = (1/n_sps) * ones —— 每个解点都变成均值
        F = np.full((n_sps, n_sps), 1.0 / n_sps)
        owner, neigh, bnd = _line_mesh(n_cells)

        U = np.ones((n_cells, n_sps, n_var))
        U[:, 1, :] = 1.001                      # 微小光滑变化，不该触发
        U[10, 1, 0] = 9.0                       # cell10 明显过冲

        func = build_sensor_gated_filter_func_arrays(
            n_cells, n_sps, 1, F, F, n_prism=n_cells // 2,
            sensor='bounds', owner_cell=owner, neighbor_cell=neigh,
            is_boundary=bnd,
            # 来流参考量级刻意取得很小（rho_inf=1 -> 地板 1e-3），
            # 以便这个合成场（量级 ~1）的过冲判定与单元测试的直觉一致
            freestream={'rho_inf': 1.0, 'vel_inf': 1.0, 'p_inf': 1.0})
        out = func(U.reshape(n_cells * n_sps, n_var).copy()).reshape(U.shape)

        # cell10 的解点被压成均值
        assert np.allclose(out[10, 0, 0], out[10, 1, 0])
        assert out[10, 0, 0] == pytest.approx((1.0 + 9.0) / 2)
        # 其余单元逐位不变
        others = [c for c in range(n_cells) if c != 10]
        assert np.array_equal(out[others], U[others])
