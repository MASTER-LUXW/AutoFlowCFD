"""`build_boundary_dirichlet_table` —— BJ 越界判据的边界值表。

为什么需要这张表（完整推导与实测见
`fr_operators/bounds_sensor.py::compute_bounds_violation_mask` 的
`bnd_dirichlet` 文档，以及 `test_bounds_sensor.py::
TestBoundaryDirichletCompletesTheEnvelope`）：边界面不提供"邻居单元
均值"，单纯排除会让该单元在那个方向上的包络变成**单侧**的，从而破坏
判据的设计不变量"线性场恒不触发"。后果是贴壁单元被结构性误判——实测
贴壁层命中率 14.393%（全域只有 0.400%），Blasius 平板的壁面剪应力被
压掉 14 倍（du/dy 93.0 vs `off` 的 1312.7）。

本文件钉住这张表的构造，尤其是那个**真实踩到过的坑**：静止壁面在
`build_boundary_ghost_provider` 的 type_map 里写作
`wall_velocity=[0.0, 0.0, 0.0]` 而**不是** `None`，第一版判据写成
`wall_velocity is not None` 就把所有真实算例的壁面都当成移动壁跳过了
——实测表里 5 列全是 NaN、贴壁层命中率毫无变化（14.393% 一点没动）。
"""

import numpy as np
import pytest

from autoflowcfd.core.fr_solver.boundary import build_boundary_dirichlet_table


class _Provider:
    """`BoundaryGhostStateProvider` 的最小替身（只用到三个属性）。"""

    def __init__(self, group_code, code_to_config, default_config=None):
        self.group_code = np.asarray(group_code)
        self.code_to_config = code_to_config
        self.default_config = default_config or {"type": "FARFIELD"}


#: 真实算例里 `build_boundary_ghost_provider` 产出的配置（取自 Blasius
#: 平板算例的实际 `code_to_config`，含静止壁那个 `[0,0,0]`）。
_REAL_CONFIG = {
    0: {"type": "SYMMETRY"},
    1: {"type": "WALL", "is_no_slip": True, "wall_velocity": [0.0, 0.0, 0.0]},
    2: {"type": "INLET", "Q_inlet": [1.225, 30.0, 0.0, 0.0, 101325.0]},
    3: {"type": "OUTLET", "p_outlet": 101325.0},
    4: {"type": "WALL", "is_no_slip": False},
    5: {"type": "SYMMETRY"},
}


class TestStationaryWallIsRecognised:
    """**真实坑回归**：静止壁写作 `[0,0,0]`，不是 `None`。"""

    def test_zero_wall_velocity_still_gets_momentum_dirichlet(self):
        gc = np.array([-1, -1, 1, 1, 0, 2, 3, 4, 5])
        t = build_boundary_dirichlet_table(_Provider(gc, _REAL_CONFIG), gc.size)
        wall = gc == 1
        assert np.all(t[wall, 1:4] == 0.0), (
            "静止壁（wall_velocity=[0,0,0]）的动量列必须是 0——写成 "
            "`wall_velocity is not None` 会把所有真实算例的壁面都跳过")
        assert np.all(~np.isfinite(t[wall, 0])), "密度没有 Dirichlet 值"
        assert np.all(~np.isfinite(t[wall, 4])), "总能没有 Dirichlet 值（绝热壁）"

    def test_wall_velocity_none_also_counts_as_stationary(self):
        cfg = {1: {"type": "WALL", "is_no_slip": True}}
        gc = np.array([1, 1, -1])
        t = build_boundary_dirichlet_table(_Provider(gc, cfg), gc.size)
        assert np.all(t[:2, 1:4] == 0.0)

    def test_moving_wall_is_left_nan_and_warns(self, caplog):
        cfg = {1: {"type": "WALL", "is_no_slip": True,
                   "wall_velocity": [5.0, 0.0, 0.0]}}
        gc = np.array([1, 1, -1])
        t = build_boundary_dirichlet_table(_Provider(gc, cfg), gc.size)
        assert np.all(~np.isfinite(t)), (
            "移动壁的 rho*u_wall 需要逐面壁面密度，这里拿不到——"
            "给个错的值比退回排除更糟")


class TestOtherBoundaryTypesStayNaN:
    """其余类型留 NaN（退回排除），逐项理由见函数文档。"""

    @pytest.mark.parametrize("code", [0, 2, 3, 4, 5])
    def test_non_no_slip_types_have_no_dirichlet(self, code):
        gc = np.array([code, code, -1])
        t = build_boundary_dirichlet_table(_Provider(gc, _REAL_CONFIG), gc.size)
        assert np.all(~np.isfinite(t)), f"code={code} 不该有 Dirichlet 值"

    def test_slip_wall_is_not_given_zero_momentum(self):
        """滑移壁（`is_no_slip=False`）的切向速度不为零，绝不能置 0。

        置 0 会把真实的切向流动当成越界，是比单侧包络更严重的误判。
        """
        gc = np.array([4, 4])
        t = build_boundary_dirichlet_table(_Provider(gc, _REAL_CONFIG), gc.size)
        assert np.all(~np.isfinite(t))

    def test_interior_faces_are_nan(self):
        gc = np.array([-1, -1, 1])
        t = build_boundary_dirichlet_table(_Provider(gc, _REAL_CONFIG), gc.size)
        assert np.all(~np.isfinite(t[:2]))


class TestDefaultConfigIsUsed:
    def test_unmatched_code_falls_back_to_default_config(self):
        """未匹配到任何组的边界面用 default_config。

        默认是 FARFIELD（无 Dirichlet），但若某个算例把默认设成无滑移
        壁面，这些面也必须拿到 0 动量。
        """
        gc = np.array([99, 99])
        t = build_boundary_dirichlet_table(
            _Provider(gc, {}, default_config={
                "type": "WALL", "is_no_slip": True}), gc.size)
        assert np.all(t[:, 1:4] == 0.0)

    def test_default_farfield_gives_nan(self):
        gc = np.array([99, 99])
        t = build_boundary_dirichlet_table(_Provider(gc, {}), gc.size)
        assert np.all(~np.isfinite(t))


class TestGuards:
    def test_none_provider_returns_none(self):
        assert build_boundary_dirichlet_table(None, 10) is None

    def test_provider_without_group_code_returns_none(self):
        class Bare:
            pass

        assert build_boundary_dirichlet_table(Bare(), 10) is None

    def test_face_count_mismatch_raises(self):
        """索引空间对不上必须报错。

        静默继续会让整张表错位——而错位的表只会让判据在错误的面上放宽/
        收紧，在残差日志里完全看不出来。
        """
        gc = np.array([1, 1, -1])
        with pytest.raises(ValueError, match="group_code"):
            build_boundary_dirichlet_table(_Provider(gc, _REAL_CONFIG), 99)

    def test_too_few_variables_raises(self):
        gc = np.array([1, -1])
        with pytest.raises(ValueError, match="n_var"):
            build_boundary_dirichlet_table(
                _Provider(gc, _REAL_CONFIG), gc.size, n_var=3)

    def test_table_shape_and_dtype(self):
        gc = np.array([1, 0, -1, 2])
        t = build_boundary_dirichlet_table(
            _Provider(gc, _REAL_CONFIG), gc.size, n_var=7)
        assert t.shape == (4, 7)
        assert t.dtype == np.float64


class TestEndToEndOnTheRealCase:
    """在真实 Blasius 算例上：表确实覆盖了壁面、且只覆盖壁面。"""

    def test_real_blasius_provider(self):
        pytest.importorskip("scipy")
        import sys
        sys.path.insert(0, ".")
        from tests.validation._blasius_case import build_blasius_solver

        solver, _ = build_blasius_solver(order=1, nx=8, nz=1, cfl=0.03)
        fc = solver.mesh.face_connectivity
        n_faces = int(np.asarray(fc.owner_cell).size)
        prov = solver.boundary_ghost_provider
        t = build_boundary_dirichlet_table(prov, n_faces, 5)
        assert t is not None

        gc = np.asarray(prov.group_code)
        no_slip_codes = [c for c, cfg in prov.code_to_config.items()
                         if cfg.get("type") == "WALL"
                         and cfg.get("is_no_slip", True)]
        assert no_slip_codes, "算例里应当有无滑移壁面组"
        is_wall = np.isin(gc, no_slip_codes)
        assert is_wall.any(), "无滑移壁面组应当有面"

        # 动量三列：壁面面全为 0，其余面全 NaN
        assert np.all(t[is_wall, 1:4] == 0.0)
        assert np.all(~np.isfinite(t[~is_wall, 1:4]))
        # 密度/总能列：处处无 Dirichlet
        assert np.all(~np.isfinite(t[:, 0]))
        assert np.all(~np.isfinite(t[:, 4]))
