"""
AutoFlowCFD V2.0 - 邻居极值越界（BJ 型）troubled-cell 判据。

## 为什么要这个判据（一句话）

Persson-Peraire 的 `s0 = -4*log10(order)` 在 order=1 时为 0，门限退化成
"顶模态能量占比 >= 10%"，而 P1 的顶模态**就是**全部非常数模态——它在
生产阶数 P1 上原理上不适用（实测 A/B 前 51 步残差与 Cd 逐字符相同）。
BJ 型判据不依赖模态分解，没有这个退化。完整推导与真实网格实测见
`core/fr_operators/bounds_sensor.py` 模块文档。

## 本文件覆盖的性质（按重要性排序）

1. **线性场恒不触发** —— 这是判据可用的前提。一维均匀网格上线性场的
   单元解点极值严格落在邻居均值区间内部（`形心 ± h/2*grad`
   vs `形心 ± h*grad`），所以限制器在线性场上不激活、不损失阶数。
   这条性质**与阶数无关**，正是 Persson-Peraire 缺的那一半。
2. 均匀场恒不触发（含浮点噪声下）
3. 真实过冲一定被抓到，且只抓那一个单元
4. 边界面不参与邻域构造（否则壁面镜像的法向速度反号会被误判）
5. 多变量取并集（真实实测里越界量最大的是横向动量与压力，只探密度
   会漏掉——与人工粘性 `DEFAULT_SENSOR_VAR_INDEX = 0` 只探密度而 P2
   失效模态在能量上，是同一类问题）
6. 形状/索引不自洽时报错，不静默广播成"恒不触发"
"""

import numpy as np
import pytest

from autoflowcfd.core.fr_operators.bounds_sensor import (
    compute_bounds_violation_mask,
    resolve_troubled_sensor,
)


def _line_mesh(n_cells, n_sps=2):
    """一维单元链的面连接：cell i 与 cell i+1 相邻，两端是边界面。

    返回 (owner, neighbor, is_boundary)。
    """
    owner = list(range(n_cells - 1)) + [0, n_cells - 1]
    neigh = list(range(1, n_cells)) + [-1, -1]
    bnd = [False] * (n_cells - 1) + [True, True]
    return (np.array(owner), np.array(neigh), np.array(bnd, dtype=bool))


class TestNoFalsePositives:
    """判据在"本该不动"的场上必须一个单元都不标记。"""

    def test_uniform_field(self):
        n = 20
        q = np.full((n, 2), 1.225)
        mask = compute_bounds_violation_mask(q, *_line_mesh(n))
        assert not mask.any()

    def test_uniform_field_with_roundoff_noise(self):
        """均匀区 nb_max == nb_min，没有绝对地板会被 ~1e-16 的噪声触发、
        把全场标成可疑 —— 这正是 `abs_frac` 存在的理由。"""
        rng = np.random.default_rng(7)
        n = 200
        q = 1.225 * (1.0 + rng.normal(scale=1e-15, size=(n, 2)))
        mask = compute_bounds_violation_mask(q, *_line_mesh(n))
        assert not mask.any(), f"{mask.sum()} 个单元被浮点噪声误判"

    def test_linear_field_is_never_flagged(self):
        """**最重要的一条**：线性场恒不触发。

        单元 i 的形心在 x=i，半宽 0.5，解点取形心 ± 0.5；线性场
        q = a + b*x 的单元解点极值为 `a+b*i ± 0.5*b`，而邻居均值为
        `a+b*(i±1)` —— 解点极值严格在内部。判据因此不激活。
        """
        n = 30
        centroids = np.arange(n, dtype=float)
        for b in (1.0, -3.7, 1e4, 1e-6):
            q = np.stack([2.0 + b * (centroids - 0.5),
                          2.0 + b * (centroids + 0.5)], axis=1)
            mask = compute_bounds_violation_mask(q, *_line_mesh(n))
            # 两端单元只有一个邻居，邻域区间单侧收窄，是 BJ 的已知边缘
            # 效应；内部单元必须干净。
            assert not mask[1:-1].any(), (
                f"b={b}: 线性场在内部单元上触发了 {mask[1:-1].sum()} 次"
            )

    def test_smooth_quadratic_is_not_flagged_at_default_tol(self):
        """光滑二次场也不该被标记（默认 rel_tol=0.1 的设计意图）。"""
        n = 40
        c = np.arange(n, dtype=float)
        q = np.stack([(c - 0.5) ** 2, (c + 0.5) ** 2], axis=1)
        mask = compute_bounds_violation_mask(q, *_line_mesh(n))
        assert not mask[1:-1].any(), f"{mask[1:-1].sum()} 个内部单元被误判"


class TestTruePositives:
    """真实过冲必须被抓到，且只抓该单元。"""

    def test_single_cell_overshoot(self):
        n = 21
        q = np.full((n, 2), 1.0)
        # 第 10 个单元的一个解点冲到 5.0：均值 3.0 远超邻居均值 1.0，
        # 解点最大值 5.0 更是远超
        q[10, 1] = 5.0
        mask = compute_bounds_violation_mask(q, *_line_mesh(n))
        assert mask[10], "明显过冲没有被抓到"
        assert mask.sum() == 1, f"除了 cell10 还标了 {np.flatnonzero(mask)}"

    def test_single_cell_undershoot(self):
        n = 21
        q = np.full((n, 2), 1.0)
        q[10, 0] = -3.0
        mask = compute_bounds_violation_mask(q, *_line_mesh(n))
        assert mask[10]
        assert mask.sum() == 1

    def test_overshoot_magnitude_threshold_is_respected(self):
        """越界幅度小于 rel_tol*邻域跨度 时不触发，大于时触发 ——
        证明 rel_tol 真的在起作用，不是个没接上的参数。"""
        n = 21
        c = np.arange(n, dtype=float)
        q = np.stack([c - 0.5, c + 0.5], axis=1)      # 线性场，邻域跨度 2
        base = q.copy()

        # 小幅越界：把 cell10 的上解点再抬 0.05（远小于 0.1*2=0.2）
        q_small = base.copy()
        q_small[10, 1] += 0.05
        assert not compute_bounds_violation_mask(q_small, *_line_mesh(n))[10]

        # 大幅越界：抬 2.0
        q_big = base.copy()
        q_big[10, 1] += 2.0
        assert compute_bounds_violation_mask(q_big, *_line_mesh(n))[10]

    def test_rel_tol_zero_is_stricter(self):
        """rel_tol=0 必须比默认档更严。

        注意凸起幅度的选取：线性场里解点极值只到形心 ±0.5，而邻居均值
        区间是 ±1，所以判据本身还有 0.5 的余量——**凸起必须先越过这
        0.5 才谈得上 rel_tol 起不起作用**。第一版这条测试取 0.05 的
        凸起，那根本没出区间，与 rel_tol 无关，是测试设计错误（已被
        这条测试自己抓到）。
        """
        n = 21
        c = np.arange(n, dtype=float)
        q = np.stack([c - 0.5, c + 0.5], axis=1)
        q[10, 1] += 0.6          # 越过 0.5 的余量，但小于 rel_tol 容差
        assert not compute_bounds_violation_mask(q, *_line_mesh(n))[10]
        assert compute_bounds_violation_mask(
            q, *_line_mesh(n), rel_tol=0.0, abs_frac=0.0)[10]


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


class TestMultiVariableUnion:
    def test_any_variable_violation_flags_the_cell(self):
        n = 21
        q = np.ones((n, 2, 5))
        q[..., 1] *= 30.0          # 一个量级不同的分量
        # 只有第 3 个变量（w 动量）过冲
        q[10, 1, 3] = 50.0
        mask = compute_bounds_violation_mask(q, *_line_mesh(n))
        assert mask[10]
        assert mask.sum() == 1

    def test_2d_input_is_treated_as_single_variable(self):
        n = 11
        q2 = np.ones((n, 2))
        q2[5, 1] = 9.0
        q3 = q2[:, :, None]
        assert np.array_equal(
            compute_bounds_violation_mask(q2, *_line_mesh(n)),
            compute_bounds_violation_mask(q3, *_line_mesh(n)),
        )

    def test_scale_is_per_variable(self):
        """绝对地板按**每个变量自己**的 RMS 缩放。

        若用全场共同的尺度，量级小的变量（例如 rho~1.2）会被量级大的
        变量（rho_E~2.5e5）的地板完全淹没、判据在它上面恒不触发——
        这与项目记忆里"把全部残差分量用共同的 q_inf*U 归一化"那次
        真实错误是同一类。
        """
        n = 21
        q = np.ones((n, 2, 2))
        q[..., 1] *= 2.5e5                  # 第二个变量量级大 2e5 倍
        q[10, 1, 0] = 6.0                   # 只在小量级变量上过冲
        mask = compute_bounds_violation_mask(q, *_line_mesh(n))
        assert mask[10], "小量级变量上的过冲被大量级变量的地板淹没了"


class TestInputValidation:
    """形状/索引不自洽必须报错，不能静默变成"恒不触发"。"""

    def test_bad_field_ndim(self):
        with pytest.raises(ValueError, match='field_nodal'):
            compute_bounds_violation_mask(np.ones(5), *_line_mesh(5))

    def test_mismatched_face_arrays(self):
        with pytest.raises(ValueError, match='同长度一维数组'):
            compute_bounds_violation_mask(
                np.ones((5, 2)), np.zeros(3, int), np.zeros(4, int),
                np.zeros(3, bool))

    def test_owner_out_of_range(self):
        with pytest.raises(ValueError, match='owner_cell 越界'):
            compute_bounds_violation_mask(
                np.ones((5, 2)), np.array([9]), np.array([1]),
                np.array([False]))

    def test_interior_neighbor_out_of_range(self):
        with pytest.raises(ValueError, match='neighbor_cell 越界'):
            compute_bounds_violation_mask(
                np.ones((5, 2)), np.array([0]), np.array([99]),
                np.array([False]))


class TestSensorResolution:
    def test_default_is_persson(self, monkeypatch):
        """默认必须保持既有行为逐位不变。"""
        monkeypatch.delenv('AFCFD_TROUBLED_SENSOR', raising=False)
        assert resolve_troubled_sensor() == 'persson'

    def test_empty_env_is_default(self, monkeypatch):
        monkeypatch.setenv('AFCFD_TROUBLED_SENSOR', '  ')
        assert resolve_troubled_sensor() == 'persson'

    @pytest.mark.parametrize('raw,expect', [
        ('bounds', 'bounds'), ('BOTH', 'both'), (' Persson ', 'persson'),
    ])
    def test_env_values(self, monkeypatch, raw, expect):
        monkeypatch.setenv('AFCFD_TROUBLED_SENSOR', raw)
        assert resolve_troubled_sensor() == expect

    def test_invalid_raises(self, monkeypatch):
        monkeypatch.setenv('AFCFD_TROUBLED_SENSOR', 'bound')
        with pytest.raises(ValueError, match='AFCFD_TROUBLED_SENSOR'):
            resolve_troubled_sensor()


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
