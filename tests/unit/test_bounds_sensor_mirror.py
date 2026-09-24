"""BJ 越界判据在**镜像型边界**（对称面 / 滑移壁）上的包络补全。

## 它修的是什么

判据的设计不变量是"线性场恒不触发"。边界面不提供邻居单元均值，单纯排除
会让该单元的包络在那个方向上**单侧**收窄，于是线性场也会触发 —— 这正是
`bnd_dirichlet` 那一半修掉的缺陷（无滑移壁，实测代价 cf -93%，见
`test_boundary_dirichlet_table.py`）。

对称面与滑移壁补不了静态值表：它们的外侧"邻居"是本单元的**镜像**，动量
均值 `m' = m - 2 (m . n) n` 依赖解、每个 stage 都不同。所以由判据内部就地
算，构造方只给法向。

**我先前的论证是错的**（已在代码里更正）：原话是"对称面上 `m_n = 0`，
所以排除即精确"。`m_n = 0` 只在**面上那一点**成立，镜像邻居的**单元均值**
`-<m_n>` 一般不为零。本文件第一条测试就是那个错误的反例。

## 为什么切向分量是无操作、而这是**正确**的

镜像单元是本单元对该平面的反射，切向动量在反射下不变，所以镜像单元的
切向均值**恰好等于**本单元均值 —— 把它计入包络是恒等操作。这不是"这一半
没修"，而是精确的外侧均值本身就等于它。因此只有动量三列参与，标量列
（密度/总能）同理无操作。
"""

from tests.unit._patch_pkg import patch_pkg_attr

import numpy as np
import pytest

from autoflowcfd.core.fr_operators.bounds_sensor import (
    compute_bounds_violation_mask,
)
from autoflowcfd.core.fr_solver.boundary import build_boundary_mirror_normals

_N = 8
_N_SPS = 4
_REF = np.array([1.0, 1.0, 1.0, 1.0, 2.5])


def _chain():
    """一维单元链，两端是边界面、法向 ±z。"""
    owner = np.array([0] + list(range(_N - 1)) + [_N - 1])
    neigh = np.array([-1] + list(range(1, _N)) + [-1])
    bnd = np.array([True] + [False] * (_N - 1) + [True])
    z = (np.arange(_N) + 0.5)[:, None] + np.linspace(
        -0.4, 0.4, _N_SPS)[None, :]
    nrm = np.full((owner.size, 3), np.nan)
    nrm[0] = [0.0, 0.0, -1.0]
    nrm[-1] = [0.0, 0.0, 1.0]
    return owner, neigh, bnd, z, nrm


def _base():
    f = np.zeros((_N, _N_SPS, 5))
    f[:, :, 0] = 1.0
    f[:, :, 4] = 2.5
    return f


def _hits(field, owner, neigh, bnd, mirror=None):
    return compute_bounds_violation_mask(
        field, owner, neigh, bnd, ref_scales=_REF,
        bnd_mirror_normal=mirror)


class TestMirrorRestoresTheDesignInvariant:
    def test_linear_normal_momentum_no_longer_triggers(self):
        """线性法向动量 + 两端对称面：2 次误判 -> 0。

        这就是"排除即精确"那条错误论证的反例。
        """
        owner, neigh, bnd, z, nrm = _chain()
        f = _base()
        f[:, :, 3] = 0.3 * z + 0.05
        assert int(_hits(f, owner, neigh, bnd).sum()) == 2
        assert int(_hits(f, owner, neigh, bnd, nrm).sum()) == 0

    def test_realistic_quasi_2d_structure(self):
        """真实准二维解的结构（切向常数 + 法向线性）同样 2 -> 0。

        Blasius 平板验证算例展向两面就是 SYMMETRY、法向分量正是 `rho_w`
        —— 那个算例的开放问题恰好是展向 `w` 的非物理增长。
        """
        owner, neigh, bnd, z, nrm = _chain()
        f = _base()
        f[:, :, 1] = 0.7
        f[:, :, 3] = 0.3 * z + 0.05
        assert int(_hits(f, owner, neigh, bnd).sum()) == 2
        assert int(_hits(f, owner, neigh, bnd, nrm).sum()) == 0

    def test_constant_tangential_never_triggered_either_way(self):
        """切向常数场两种情形都不触发（镜像对切向是恒等操作）。"""
        owner, neigh, bnd, _z, nrm = _chain()
        f = _base()
        f[:, :, 1] = 0.7
        assert int(_hits(f, owner, neigh, bnd).sum()) == 0
        assert int(_hits(f, owner, neigh, bnd, nrm).sum()) == 0


class TestRealViolationsStillCaught:
    """补包络**不能**把判据变钝 —— 否则就是用另一个缺陷换掉这个缺陷。"""

    def test_in_cell_oscillation_is_still_flagged(self):
        owner, neigh, bnd, z, nrm = _chain()
        f = _base()
        f[:, :, 3] = 0.3 * z + 0.05
        f[4, :, 3] += np.array([2.0, -2.0, 2.0, -2.0])
        m = _hits(f, owner, neigh, bnd, nrm)
        assert bool(m[4]), "真实越界单元没有被标记"
        assert int(m.sum()) == 1, f"只应标记单元4，实际 {np.flatnonzero(m)}"

    def test_oscillation_at_the_boundary_cell_is_still_flagged(self):
        """紧贴对称面的那个单元里的振荡同样要命中。

        补包络是为了消除**结构性误判**，不是为了让边界单元免检。
        """
        owner, neigh, bnd, _z, nrm = _chain()
        f = _base()
        f[:, :, 3] = 0.0
        f[0, :, 3] = np.array([3.0, -3.0, 3.0, -3.0])
        assert bool(_hits(f, owner, neigh, bnd, nrm)[0])


class TestMirrorIsExactForTheGhostMean:
    def test_normal_component_mirror_equals_minus_own_mean(self):
        """法向列的外侧均值必须正好是 `-<m_n>`。

        把法向动量整体平移会改变 `<m_n>` 的大小与符号，而镜像值随之改变；
        若实现里用的是别的值（例如仍然当作 0），平移后必然重新误判。
        """
        owner, neigh, bnd, z, nrm = _chain()
        for shift in (0.0, 1.0, -3.0):
            f = _base()
            f[:, :, 3] = 0.3 * z + 0.05 + shift
            assert int(_hits(f, owner, neigh, bnd, nrm).sum()) == 0, (
                f"shift={shift} 下线性场被误判")

    def test_orientation_of_the_normal_does_not_matter(self):
        """`n` 与 `-n` 必须给出同一个结果（公式里 `n` 出现两次）。"""
        owner, neigh, bnd, z, nrm = _chain()
        f = _base()
        f[:, :, 3] = 0.3 * z + 0.05
        a = _hits(f, owner, neigh, bnd, nrm)
        b = _hits(f, owner, neigh, bnd, -nrm)
        assert np.array_equal(a, b)

    def test_oblique_normal(self):
        """斜法向：镜像必须作用在**法向分量**上，不是某个坐标轴上。"""
        owner, neigh, bnd, _z, _nrm = _chain()
        d = np.array([1.0, 0.0, 1.0]) / np.sqrt(2.0)
        nrm = np.full((owner.size, 3), np.nan)
        nrm[0] = d
        nrm[-1] = d
        t = (np.arange(_N) + 0.5)[:, None] + np.linspace(
            -0.4, 0.4, _N_SPS)[None, :]
        f = _base()
        # 动量完全沿 d、大小线性 -> 纯法向分量
        for c in range(3):
            f[:, :, 1 + c] = (0.3 * t + 0.05) * d[c]
        assert int(_hits(f, owner, neigh, bnd).sum()) == 2
        assert int(_hits(f, owner, neigh, bnd, nrm).sum()) == 0


class TestGuards:
    def test_wrong_shape_is_rejected(self):
        owner, neigh, bnd, _z, _n = _chain()
        f = _base()
        with pytest.raises(ValueError, match="bnd_mirror_normal 形状"):
            _hits(f, owner, neigh, bnd, np.zeros((owner.size, 2)))

    def test_zero_length_normal_is_rejected(self):
        """零长度法向必须报错而不是静默退回单侧包络。

        静默退回是"看不出来的错误"：日志里没有任何迹象，只有壁面剪应力
        悄悄被压低。
        """
        owner, neigh, bnd, _z, nrm = _chain()
        f = _base()
        bad = nrm.copy()
        bad[0] = [0.0, 0.0, 0.0]
        with pytest.raises(ValueError, match="零长度法向"):
            _hits(f, owner, neigh, bnd, bad)

    def test_too_few_variables_is_rejected(self):
        owner, neigh, bnd, _z, nrm = _chain()
        f = np.zeros((_N, _N_SPS, 3))
        with pytest.raises(ValueError, match="至少 4 个变量"):
            compute_bounds_violation_mask(
                f, owner, neigh, bnd, bnd_mirror_normal=nrm)

    def test_all_nan_normals_are_a_no_op(self):
        """全 NaN（没有任何镜像面）必须与不给这个参数逐位一致。"""
        owner, neigh, bnd, z, _n = _chain()
        f = _base()
        f[:, :, 3] = 0.3 * z + 0.05
        allnan = np.full((owner.size, 3), np.nan)
        assert np.array_equal(_hits(f, owner, neigh, bnd),
                              _hits(f, owner, neigh, bnd, allnan))


class _Provider:
    def __init__(self, group_code, cfgs):
        self.group_code = np.asarray(group_code)
        self.code_to_config = cfgs
        self.default_config = {"type": "FARFIELD"}


class TestMirrorNormalTable:
    """`build_boundary_mirror_normals` —— 哪些面算镜像面。"""

    _GC = np.array([-1, 0, 1, 2, 3, -1])
    _CFG = {
        0: {"type": "SYMMETRY"},
        1: {"type": "WALL", "is_no_slip": False, "physical_no_slip": False},
        2: {"type": "WALL", "is_no_slip": True, "physical_no_slip": True},
        3: {"type": "INLET"},
    }

    def _table(self, normals=None):
        n = self._GC.size
        if normals is None:
            normals = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
        return build_boundary_mirror_normals(
            _Provider(self._GC, self._CFG), normals, n)

    def test_symmetry_and_slip_wall_are_mirrors(self):
        t = self._table()
        assert np.all(np.isfinite(t[1]))   # SYMMETRY
        assert np.all(np.isfinite(t[2]))   # 滑移壁

    def test_no_slip_wall_inlet_and_interior_are_not(self):
        """无滑移壁由 Dirichlet 表负责，两张表互斥。"""
        t = self._table()
        for i in (0, 3, 4, 5):
            assert np.all(np.isnan(t[i])), f"面 {i} 不该是镜像面"

    def test_per_flux_point_normals_are_reduced(self):
        """(n_faces, n_fp, 3)（分布式的 `true_normal`）要被归约成逐面。"""
        n = self._GC.size
        fp = np.tile(np.array([0.0, 0.0, 1.0]), (n, 3, 1))
        t = self._table(fp)
        assert t.shape == (n, 3)
        assert np.allclose(t[1], [0.0, 0.0, 1.0])

    def test_none_when_no_mirror_faces(self):
        gc = np.array([-1, 2, 2])
        t = build_boundary_mirror_normals(
            _Provider(gc, self._CFG), np.zeros((3, 3)), 3)
        assert t is None, "没有镜像面时应返回 None（调用方退回排除）"

    def test_none_without_provider_or_normals(self):
        assert build_boundary_mirror_normals(None, np.zeros((3, 3)), 3) is None
        assert build_boundary_mirror_normals(
            _Provider(self._GC, self._CFG), None, self._GC.size) is None

    def test_index_space_mismatch_raises(self):
        with pytest.raises(ValueError, match="group_code 长度"):
            build_boundary_mirror_normals(
                _Provider(self._GC, self._CFG), np.zeros((99, 3)), 99)


class TestSingleBackendDeliversBothTables:
    """cpu-single（生产默认后端）必须把**两张表**都交到判据内核。

    漏掉任一张都不会报错，只会让边界旁的单元被结构性误判 —— 那在残差
    日志里完全看不出来。分布式路径的同类测试见
    `test_sensor_gate_distributed.py`。
    """

    def _solver(self):
        from types import SimpleNamespace

        owner, neigh, bnd, z, _n = _chain()
        n_faces = owner.size
        normal = np.zeros((n_faces, 3))
        normal[:, 2] = 1.0
        gc = np.full(n_faces, -1, dtype=np.int64)
        gc[0] = 5          # 对称面
        gc[-1] = 6         # 无滑移壁
        prov = SimpleNamespace(
            group_code=gc,
            code_to_config={
                5: {"type": "SYMMETRY"},
                6: {"type": "WALL", "is_no_slip": True,
                    "physical_no_slip": True,
                    "wall_velocity": [0.0, 0.0, 0.0]},
            },
            default_config={"type": "FARFIELD"})
        fc = SimpleNamespace(owner_cell=owner, neighbor_cell=neigh,
                             is_boundary=bnd, normal=normal)
        ident = np.eye(_N_SPS)
        return SimpleNamespace(
            mesh=SimpleNamespace(n_cells=_N, n_sps_per_cell=_N_SPS,
                                 n_prism_cells=_N, face_connectivity=fc),
            ops=SimpleNamespace(filter_prism=ident, filter_tet=ident),
            order=1, current_order=1,
            # 键名此前全错（"rho"/"u"/"p"），被生产代码的 `.get(k, 兜底)`
            # 静默吞掉、实际用的是 p_inf=101325。2026-09-24 生产代码改为直接
            # 取键后才暴露。按作者本意给单位量级。
            freestream={"rho_inf": 1.0, "vel_inf": 1.0, "p_inf": 1.0},
            boundary_ghost_provider=prov), owner, neigh, bnd, z

    def test_both_tables_reach_the_kernel(self, monkeypatch):
        monkeypatch.setenv("AFCFD_TROUBLED_SENSOR", "bounds")
        from autoflowcfd.core.fr_operators import bounds_sensor as bs
        from autoflowcfd.core.fr_solver.filter import (
            build_sensor_gated_filter_func,
        )

        solver, owner, _neigh, _bnd, z = self._solver()
        seen = {}
        orig = bs.compute_bounds_violation_mask

        def spy(*a, **kw):
            seen["bd"] = kw.get("bnd_dirichlet")
            seen["mir"] = kw.get("bnd_mirror_normal")
            return orig(*a, **kw)

        patch_pkg_attr(monkeypatch, bs, "compute_bounds_violation_mask", spy)
        ff = build_sensor_gated_filter_func(solver)
        f = _base()
        f[:, :, 3] = 0.3 * z + 0.05
        ff(f.reshape(_N * _N_SPS, 5).copy())

        bd = seen.get("bd")
        mir = seen.get("mir")
        assert bd is not None, "没收到 bnd_dirichlet"
        assert mir is not None, "没收到 bnd_mirror_normal"
        assert np.asarray(bd).shape == (owner.size, 5)
        assert np.asarray(mir).shape == (owner.size, 3)
        # 无滑移壁那一面进 Dirichlet 表、对称面那一面进镜像表，互斥
        assert np.all(np.asarray(bd)[-1, 1:4] == 0.0)
        assert np.all(np.isnan(np.asarray(bd)[0]))
        assert np.all(np.isfinite(np.asarray(mir)[0]))
        assert np.all(np.isnan(np.asarray(mir)[-1]))

    def test_tables_are_built_once_and_cached(self, monkeypatch):
        """两张表只依赖静态 BC 配置与几何，不能每个 RK stage 重建一次。"""
        monkeypatch.setenv("AFCFD_TROUBLED_SENSOR", "bounds")
        from autoflowcfd.core.fr_solver import boundary as bnd_mod
        from autoflowcfd.core.fr_solver.filter import (
            build_sensor_gated_filter_func,
        )

        solver, _o, _n, _b, z = self._solver()
        calls = {"n": 0}
        orig = bnd_mod.build_boundary_dirichlet_table

        def counting(*a, **kw):
            calls["n"] += 1
            return orig(*a, **kw)

        # 必须连子模块一起换：`boundary` 2026-09-24 拆成子包，真正调用
        # `build_boundary_dirichlet_table` 的是**同一个子模块**
        # `boundary/tables.py` 里的 `make_bj_boundary_tables`，它走的是
        # tables.py 自己的全局，不是包属性。对包 setattr 会**成功**（这个
        # 名字被 __init__ re-export 了）却完全不起作用 —— 计数恒为 0。
        patch_pkg_attr(monkeypatch, bnd_mod, "build_boundary_dirichlet_table",
                       counting)
        ff = build_sensor_gated_filter_func(solver)
        f = _base()
        f[:, :, 3] = 0.3 * z + 0.05
        flat = f.reshape(_N * _N_SPS, 5)
        for _ in range(4):
            ff(flat.copy())
        assert calls["n"] == 1, f"表被重建了 {calls['n']} 次"
