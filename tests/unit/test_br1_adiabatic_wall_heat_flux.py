"""BR1 边界温度梯度按热边界类型分派的验证（2026-09-15）。

## 修的是什么

壁面 ghost 态一律复制 rho/p（温度无跳跃 ⇒ 语义绝热，见
`fr_ghost_state.wall_ghost_state` 的"热边界条件"一节），但粘性界面项此前
把**温度梯度**也一律取内部值（`gT_n = gT_o`），而边界 IP 罚项只覆盖动量
分量（`for v in range(1,4)`）。于是面上平均法向温度梯度等于内部值、一般
非零——实现出来的壁面热条件既不是绝热（q_w=0）也不是等温，而是"按内部
梯度透射"，对能量方程不自洽。

现在 WALL/SYMMETRY 改为法向分量镜像，INLET/OUTLET/FARFIELD 保持透射。

## 判据分三层

1. **算子层（精确恒等式）**：`mirror_normal_component` 的代数性质——
   BR1 面平均的法向分量精确为零、对合、保模、与法向朝向/缩放无关。
2. **几何层**：镜像用的逆变行 `adj_row` 与物理面法向 `true_normal` 在真实
   合成网格上确实平行——否则"法向分量为零"这句话指的就不是物理法向。
3. **端到端（fail-then-pass）**：整条 `compute_viscous_residual_fr` 路径上，
   WALL 与 FARFIELD 的能量残差现在不同；把分派掩码强制成全 False（即
   修复前的行为）后两者重新变得逐位相同——这条断言在修复前必然失败，
   是这次改动真正生效的直接证据。

第 3 层特意用"速度场恒为零 + 温度场非均匀"的构造：此时粘性应力张量
tau ≡ 0、粘性功 u·tau ≡ 0，能量通量**只剩传导项 q = -k∇T**，于是能量分量
的任何差异都只能来自 ∇T 的处理，不会被动量/粘性功的差异污染。
"""

import numpy as np
import pytest

from autoflowcfd.boundary.fr_ghost_state import (
    ADIABATIC_THERMAL_BC_TYPES,
    BoundaryGhostStateProvider,
    build_boundary_adiabatic_mask,
)
from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
from autoflowcfd.core.fr_operators.flux_kernels import mirror_normal_component
from autoflowcfd.core.fr_residual.inviscid import (
    DefaultGhostProvider, primitive_to_conserved,
)
from autoflowcfd.core.fr_residual import viscous_flux as vf
from autoflowcfd.fr.operators import generate_fr_operators

from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh

MU = 1.8e-5
PR = 0.72
Q_FREE = np.array([1.225, 30.0, 0.0, 0.0, 101325.0])
_MASK_SYMBOL = "autoflowcfd.boundary.fr_ghost_state.build_boundary_adiabatic_mask"


def _force_transmissive(monkeypatch):
    """复刻修复前的行为：分派掩码恒为全 False（一律透射）。"""
    monkeypatch.setattr(
        _MASK_SYMBOL,
        lambda n_faces, is_boundary, gp: np.zeros(n_faces, dtype=np.bool_))


# ---------------------------------------------------------------------------
# 1. 算子层：精确代数性质
# ---------------------------------------------------------------------------

class TestMirrorNormalComponent:
    @pytest.fixture
    def rng(self):
        return np.random.default_rng(20260915)

    def test_br1_average_has_exactly_zero_normal_component(self, rng):
        """核心判据：BR1 面平均 0.5*(g + mirror(g)) 的法向分量精确为零。

        这正是"离散传导热通量 a·q_avg 恒等于零"的充要条件——
        a·q_avg = -k*(a·∇T_avg)，而 a·∇T_avg 就是这里断言为零的量。
        """
        for _ in range(200):
            g = rng.standard_normal(3) * 10 ** rng.uniform(-6, 6)
            a = rng.standard_normal(3) * 10 ** rng.uniform(-6, 6)
            g_avg = 0.5 * (g + mirror_normal_component(g, a))
            proj = float(np.dot(g_avg, a))
            scale = np.linalg.norm(g) * np.linalg.norm(a)
            assert abs(proj) <= 1e-14 * scale, (
                f"a·g_avg={proj:.3e}（相对尺度 {scale:.3e}）不是机器零——"
                f"壁面热通量就不会精确为零")

    def test_is_involutive(self, rng):
        """镜像两次回到原值——镜像反射的定义性质（能抓出"写成了投影"）。"""
        for _ in range(100):
            g = rng.standard_normal(3)
            a = rng.standard_normal(3)
            back = mirror_normal_component(mirror_normal_component(g, a), a)
            assert np.abs(back - g).max() < 1e-13

    def test_is_isometric(self, rng):
        """|mirror(g)| == |g|：反射保模，只改方向不改量级。"""
        for _ in range(100):
            g = rng.standard_normal(3)
            a = rng.standard_normal(3)
            m = mirror_normal_component(g, a)
            rel = abs(np.linalg.norm(m) - np.linalg.norm(g)) / np.linalg.norm(g)
            assert rel < 1e-13

    def test_independent_of_normal_orientation_and_scale(self, rng):
        """结果只依赖面的方向，不依赖内/外法向约定，也不依赖 |adj_row|。

        这条保证了"用逆变行而不是单位法向"是安全的——逆变行的模含面积/
        雅可比因子，一旦结果依赖它，不同网格尺度下行为就会不一致。
        """
        for _ in range(100):
            g = rng.standard_normal(3)
            a = rng.standard_normal(3)
            base = mirror_normal_component(g, a)
            for s in (-1.0, -3.7e5, 2.0, 1.1e-7):
                np.testing.assert_allclose(
                    mirror_normal_component(g, a * s), base,
                    rtol=1e-12, atol=1e-14)

    def test_tangential_component_is_preserved(self, rng):
        """只镜像法向分量：切向分量逐分量不变。

        物理上必须如此——绝热约束的是 dT/dn，沿壁面的温度变化是真实的
        物理量，不能被一起抹掉。
        """
        for _ in range(100):
            g = rng.standard_normal(3)
            a = rng.standard_normal(3)
            n = a / np.linalg.norm(a)
            m = mirror_normal_component(g, a)
            t_g = g - np.dot(g, n) * n
            t_m = m - np.dot(m, n) * n
            assert np.abs(t_m - t_g).max() < 1e-13 * max(np.abs(g).max(), 1.0)

    def test_degenerate_adj_row_returns_input(self):
        """|adj_row|=0（退化面）时原样返回，不产生 NaN。

        这种面的通量投影 a·q 本来就是零，镜不镜像都不影响结果；关键是
        不能除零把 NaN 播进残差。
        """
        g = np.array([1.0, -2.0, 3.0])
        out = mirror_normal_component(g, np.zeros(3))
        np.testing.assert_array_equal(out, g)
        assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# 2. BC 分类掩码
# ---------------------------------------------------------------------------

def _provider_all(bc_type, flat, **cfg):
    group_code = np.full(flat.n_faces, -1, dtype=np.int64)
    group_code[flat.is_boundary] = 0
    conf = {"type": bc_type}
    conf.update(cfg)
    return BoundaryGhostStateProvider(
        group_code, {0: conf}, {"type": "FARFIELD", "Q_free": Q_FREE})


class TestAdiabaticMask:
    @pytest.fixture(scope="class")
    def flat(self):
        mesh = _build_synthetic_mixed_mesh(1)
        return get_flat_face_geometry(mesh, generate_fr_operators(1))

    @pytest.mark.parametrize("bc_type,expected", [
        ("WALL", True), ("SYMMETRY", True),
        ("INLET", False), ("OUTLET", False), ("FARFIELD", False),
    ])
    def test_classification(self, flat, bc_type, expected):
        p = _provider_all(bc_type, flat)
        mask = build_boundary_adiabatic_mask(flat.n_faces, flat.is_boundary, p)
        bnd = np.nonzero(flat.is_boundary)[0]
        assert bnd.size > 0
        assert bool(mask[bnd].all()) is expected
        assert bool(mask[bnd].any()) is expected

    @pytest.mark.parametrize("is_no_slip", [True, False])
    def test_both_wall_branches_are_adiabatic(self, flat, is_no_slip):
        """滑移壁（含 WMLES 激活时的壁面）与无滑移壁都是绝热的——本项目
        既没有等温壁 BC 也没有壁面热通量模型，两个分支的 ghost 态都复制
        rho/p（见 test_wall_ghost_shared_construction.py 判据 6）。"""
        p = _provider_all("WALL", flat, is_no_slip=is_no_slip)
        mask = build_boundary_adiabatic_mask(flat.n_faces, flat.is_boundary, p)
        assert mask[flat.is_boundary].all()

    def test_interior_faces_never_marked(self, flat):
        p = _provider_all("WALL", flat)
        mask = build_boundary_adiabatic_mask(flat.n_faces, flat.is_boundary, p)
        assert not mask[~flat.is_boundary].any(), (
            "内部面被标记了——内部面两侧都有真实的局部梯度，镜像会破坏"
            "已验证正确的内部粘性耦合")

    def test_mixed_groups_are_per_face(self, flat):
        """不同边界组必须逐面区分，不能"有一个 WALL 就全标"。"""
        bnd = np.nonzero(flat.is_boundary)[0]
        assert bnd.size >= 4
        group_code = np.full(flat.n_faces, -1, dtype=np.int64)
        group_code[bnd] = np.arange(bnd.size) % 2
        p = BoundaryGhostStateProvider(
            group_code,
            {0: {"type": "WALL", "is_no_slip": True},
             1: {"type": "FARFIELD", "Q_free": Q_FREE}},
            {"type": "FARFIELD", "Q_free": Q_FREE})
        mask = build_boundary_adiabatic_mask(flat.n_faces, flat.is_boundary, p)
        np.testing.assert_array_equal(mask[bnd], (np.arange(bnd.size) % 2) == 0)

    def test_unmatched_faces_follow_default_config(self, flat):
        """未匹配任何组的边界面（group_code=-1）走 default_config——
        不能静默当成透射，否则默认 SYMMETRY 的工况会漏掉。"""
        bnd = np.nonzero(flat.is_boundary)[0]
        group_code = np.full(flat.n_faces, -1, dtype=np.int64)  # 全部未匹配
        p = BoundaryGhostStateProvider(group_code, {}, {"type": "SYMMETRY"})
        mask = build_boundary_adiabatic_mask(flat.n_faces, flat.is_boundary, p)
        assert mask[bnd].all()

    def test_default_provider_is_all_transmissive(self, flat):
        """DefaultGhostProvider 的 Q_ghost=Q_owner 本身就是纯透射，没有
        BC 语义可分派——必须全 False，保证既有行为逐位不变。"""
        mask = build_boundary_adiabatic_mask(
            flat.n_faces, flat.is_boundary, DefaultGhostProvider())
        assert not mask.any()
        assert mask.shape == (flat.n_faces,)
        assert mask.dtype == np.bool_

    def test_cached_per_provider(self, flat):
        p = _provider_all("WALL", flat)
        m1 = build_boundary_adiabatic_mask(flat.n_faces, flat.is_boundary, p)
        m2 = build_boundary_adiabatic_mask(flat.n_faces, flat.is_boundary, p)
        assert m1 is m2, "掩码没有按 provider 缓存——每次残差求值都要重扫边界面"

    def test_adiabatic_type_set_is_explicit(self):
        assert ADIABATIC_THERMAL_BC_TYPES == frozenset({"WALL", "SYMMETRY"})


# ---------------------------------------------------------------------------
# 3. 几何层：逆变行与物理法向平行
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("order", [1, 2])
def test_adj_row_is_parallel_to_true_normal(order):
    """镜像用的是逆变行 adj_row，断言它与物理面法向 true_normal 平行。

    若不平行，"法向分量精确为零"这句话指的就不是物理意义上的法向，
    整个绝热判据就失去物理含义。判据用叉乘模 / 模积（=|sin theta|）。
    """
    mesh = _build_synthetic_mixed_mesh(order)
    flat = get_flat_face_geometry(mesh, generate_fr_operators(order))
    bnd = np.nonzero(flat.is_boundary & flat.owner_is_primary)[0]
    assert bnd.size > 0

    a = flat.owner_adj_row_exact[bnd]          # (nb, n_fp, 3)
    n = flat.true_normal[bnd]                  # (nb, n_fp, 3)
    cross = np.cross(a, n)
    denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(n, axis=-1)
    good = denom > 0
    sin_theta = np.linalg.norm(cross, axis=-1)[good] / denom[good]
    assert sin_theta.max() < 1e-12, (
        f"逆变行与物理法向最大夹角 |sin|={sin_theta.max():.3e}，不平行")


# ---------------------------------------------------------------------------
# 4. 端到端：fail-then-pass
# ---------------------------------------------------------------------------

def _zero_velocity_nonuniform_T_state(mesh):
    """速度恒为零、温度非均匀的状态场。

    速度为零 ⇒ tau ≡ 0 且粘性功 u·tau ≡ 0 ⇒ 能量通量只剩传导项
    q = -k∇T，把这次改动要考察的量单独隔离出来。温度非均匀通过让压力
    随坐标变化实现（密度保持常数），保证 ∇T 在边界面上确有非零法向分量。
    """
    x = mesh.sps_coords.reshape(mesh.n_cells, mesh.n_sps_per_cell, 3)
    Q = np.zeros((mesh.n_cells, mesh.n_sps_per_cell, 5))
    Q[..., 0] = 1.225
    Q[..., 1:4] = 0.0
    Q[..., 4] = 101325.0 * (1.0 + 0.05 * x[..., 0]
                            + 0.03 * x[..., 1] - 0.02 * x[..., 2])
    return primitive_to_conserved(Q)


def _energy_residual(U, mesh, ops, provider):
    res = vf.compute_viscous_residual_fr(U, mesh, ops, MU, PR,
                                         boundary_ghost_provider=provider)
    return res[..., 4]


@pytest.mark.parametrize("order", [1, 2])
def test_wall_and_farfield_differ_now_and_coincide_when_forced_off(order, monkeypatch):
    """fail-then-pass 的核心用例。

    速度恒为零时，WALL 与 FARFIELD 的能量粘性残差在**修复前**必然逐位
    相同：两者的 ∇T 处理都是"取内部值"，而 tau/粘性功都是零，能量通量
    只剩 q=-k∇T_avg，与 ghost 态里的 rho/p 无关。

    修复后 WALL 走法向镜像、FARFIELD 走透射，两者必须不同；把分派掩码
    强制成全 False（复刻修复前的行为）后又必须重新变回逐位相同。
    后半句同时证明了"差异确实来自这次分派，而不是别的副作用"。
    """
    mesh = _build_synthetic_mixed_mesh(order)
    ops = generate_fr_operators(order)
    flat = get_flat_face_geometry(mesh, ops)
    U = _zero_velocity_nonuniform_T_state(mesh)

    wall = _provider_all("WALL", flat, is_no_slip=True)
    farf = _provider_all("FARFIELD", flat, Q_free=Q_FREE)

    e_wall = _energy_residual(U, mesh, ops, wall)
    e_farf = _energy_residual(U, mesh, ops, farf)
    diff = np.abs(e_wall - e_farf).max()
    scale = max(np.abs(e_farf).max(), 1e-300)
    assert diff / scale > 1e-6, (
        f"WALL 与 FARFIELD 的能量残差仍然几乎相同（相对差 {diff/scale:.3e}）"
        f"——边界温度梯度分派没有生效")

    _force_transmissive(monkeypatch)
    e_wall_old = _energy_residual(
        U, mesh, ops, _provider_all("WALL", flat, is_no_slip=True))
    e_farf_old = _energy_residual(
        U, mesh, ops, _provider_all("FARFIELD", flat, Q_free=Q_FREE))
    np.testing.assert_allclose(
        e_wall_old, e_farf_old, rtol=0, atol=0,
        err_msg="修复前 WALL/FARFIELD 的能量残差应逐位相同")


@pytest.mark.parametrize("order", [1, 2])
def test_symmetry_matches_wall_for_energy_at_zero_velocity(order):
    """速度恒为零时 SYMMETRY 与 WALL 的能量残差必须逐位相同。

    两者都被归为绝热类、走同一个法向镜像；而 symmetry_ghost_state 与
    is_no_slip=True 的 wall_ghost_state 对 u≡0 的场给出同样的 ghost
    速度（都是零），rho/p 也都照抄。这条把"分类正确"与"镜像正确"分开
    检验：任何一处把 SYMMETRY 漏掉，这里立刻失败。
    """
    mesh = _build_synthetic_mixed_mesh(order)
    ops = generate_fr_operators(order)
    flat = get_flat_face_geometry(mesh, ops)
    U = _zero_velocity_nonuniform_T_state(mesh)

    e_wall = _energy_residual(
        U, mesh, ops, _provider_all("WALL", flat, is_no_slip=True))
    e_sym = _energy_residual(U, mesh, ops, _provider_all("SYMMETRY", flat))
    np.testing.assert_allclose(e_wall, e_sym, rtol=0, atol=0)


@pytest.mark.parametrize("order", [1, 2])
def test_mass_and_momentum_residuals_are_untouched(order, monkeypatch):
    """这次改动只动能量方程的传导项：质量/动量分量必须逐位不变。

    用同一个 WALL provider 跑两遍（一遍真实掩码、一遍强制全 False），
    比较前 4 个分量。这条是对"没有顺手改到已验证的动量路径"的直接保证。
    """
    mesh = _build_synthetic_mixed_mesh(order)
    ops = generate_fr_operators(order)
    flat = get_flat_face_geometry(mesh, ops)

    # 这里用**有速度**的场，否则动量分量恒为零、断言退化成平凡真
    x = mesh.sps_coords.reshape(mesh.n_cells, mesh.n_sps_per_cell, 3)
    Q = np.zeros((mesh.n_cells, mesh.n_sps_per_cell, 5))
    Q[..., 0] = 1.225
    Q[..., 1] = 30.0 + 2.0 * x[..., 1]
    Q[..., 2] = 1.5 * x[..., 2]
    Q[..., 3] = -0.8 * x[..., 0]
    Q[..., 4] = 101325.0 * (1.0 + 0.05 * x[..., 0] + 0.03 * x[..., 1])
    U = primitive_to_conserved(Q)

    res_new = vf.compute_viscous_residual_fr(
        U, mesh, ops, MU, PR,
        boundary_ghost_provider=_provider_all("WALL", flat, is_no_slip=True))
    _force_transmissive(monkeypatch)
    res_old = vf.compute_viscous_residual_fr(
        U, mesh, ops, MU, PR,
        boundary_ghost_provider=_provider_all("WALL", flat, is_no_slip=True))

    np.testing.assert_allclose(
        res_new[..., :4], res_old[..., :4], rtol=0, atol=0,
        err_msg="质量/动量分量被改动了——本次改动只应影响能量传导项")
    assert np.abs(res_new[..., 4] - res_old[..., 4]).max() > 0.0, (
        "能量分量完全没变——说明镜像根本没生效"
        "（该网格上边界 ∇T 法向分量非零）")


@pytest.mark.parametrize("order", [1, 2])
def test_default_ghost_provider_behavior_is_bit_identical(order, monkeypatch):
    """DefaultGhostProvider 路径必须逐位不变（它没有 BC 语义可分派）。"""
    mesh = _build_synthetic_mixed_mesh(order)
    ops = generate_fr_operators(order)
    U = _zero_velocity_nonuniform_T_state(mesh)

    res_new = vf.compute_viscous_residual_fr(
        U, mesh, ops, MU, PR, boundary_ghost_provider=None)
    _force_transmissive(monkeypatch)
    res_old = vf.compute_viscous_residual_fr(
        U, mesh, ops, MU, PR, boundary_ghost_provider=None)
    np.testing.assert_allclose(res_new, res_old, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# 5. 混合拆分面（B-8）的两个半区 —— 手工合成配置
# ---------------------------------------------------------------------------
#
# 为什么要手工合成：B-8 混合拆分面是 BL 挤出在几何尖角棱处产生拓扑缝隙的
# 固有产物（见 fr/face_flux_points/merge.py 的"混合分组检测"一节），只在
# 真实带边界层的网格上出现——`_build_synthetic_mixed_mesh` 实测
# `mixed_nb_partner`/`mixed_ow_partner` **全为 -1**，也就是说本文件前面
# 那些端到端用例一次都没有走进这两个分支。
#
# 而这次改动在每条 kernel 里都有 3 个分派点：真边界面 + B-8 owner 半区 +
# B-8 neighbor 半区。只验证第一个是不够的。
#
# 做法：取真实合成网格的 flat 几何，把某个**内部面** f_int 手工登记成
# "混合拆分面，配对面是边界面 bf、整张面都取边界半区"。这个拓扑在物理上
# 不成立，但它让 kernel 真的走进 B-8 分支，而分支里的**分派逻辑**正是要
# 验证的东西。
#
# 归因设计（关键）：保持 flat 与 provider 完全不变，只切换绝热掩码
# （bf 绝热 vs 全透射）。此时两次计算唯一的差别就是 ∇T 的处理，且只发生在
# 两个地方：(a) bf 自己的真边界面分支、(b) f_int 的 B-8 半区。因此只要
# **挑一个 owner 与 bf 的 owner 不同的 f_int**，那么 f_int 的 owner 单元上
# 出现的任何差异就只可能来自 (b)。

import dataclasses


def _pick_faces_for_mixed(flat, need_neighbor_side):
    """挑一对 (f_int, bf)：f_int 是内部面，bf 是边界面，且两者的相关
    单元不同 —— 差异归因的前提。"""
    bnd = np.nonzero(flat.is_boundary & flat.owner_is_primary)[0]
    interior = np.nonzero((~flat.is_boundary) & flat.owner_is_primary
                          & flat.neighbor_is_primary)[0]
    assert bnd.size > 0 and interior.size > 0
    for f_int in interior:
        target = (flat.neighbor_cell[f_int] if need_neighbor_side
                  else flat.owner_cell[f_int])
        for bf in bnd:
            if flat.owner_cell[bf] != target:
                return int(f_int), int(bf), int(target)
    raise AssertionError("找不到满足归因条件的 (f_int, bf) 组合")


def _flat_with_mixed(flat, f_int, bf, side):
    """返回一份把 f_int 登记成混合拆分面的 flat 副本（整张面取边界半区）。"""
    nb_partner = flat.mixed_nb_partner.copy()
    nb_mask = flat.mixed_nb_mask.copy()
    ow_partner = flat.mixed_ow_partner.copy()
    ow_mask = flat.mixed_ow_mask.copy()
    if side == "owner":
        nb_partner[f_int] = bf
        nb_mask[f_int, :] = True
    else:
        ow_partner[f_int] = bf
        ow_mask[f_int, :] = True
    return dataclasses.replace(
        flat,
        mixed_nb_partner=nb_partner, mixed_nb_mask=nb_mask,
        mixed_ow_partner=ow_partner, mixed_ow_mask=ow_mask,
    )


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("side", ["owner", "neighbor"])
def test_mixed_split_half_dispatches_grad_T_by_partner_bc(order, side, monkeypatch):
    """B-8 两个半区都必须按**配对面**的热边界类型分派 ∇T。

    判据：
    1. 目标单元（owner 半区看 f_int 的 owner，neighbor 半区看 f_int 的
       neighbor）的**能量**残差在"bf 绝热"与"全透射"两种掩码下必须不同
       —— 说明 B-8 分支真的读了 `bnd_adiabatic[配对面]` 并做了镜像。
    2. 同一单元的质量/动量分量必须逐位相同 —— 说明只动了能量传导项。
    """
    mesh = _build_synthetic_mixed_mesh(order)
    ops = generate_fr_operators(order)
    flat = get_flat_face_geometry(mesh, ops)
    f_int, bf, target_cell = _pick_faces_for_mixed(flat, side == "neighbor")
    flat2 = _flat_with_mixed(flat, f_int, bf, side)
    U = _zero_velocity_nonuniform_T_state(mesh)

    # bf 是 WALL（绝热），其余边界面 FARFIELD（透射）——只让 bf 这一个面
    # 的分类在两次运行间起作用。
    group_code = np.full(flat.n_faces, -1, dtype=np.int64)
    group_code[flat.is_boundary] = 1
    group_code[bf] = 0
    provider = BoundaryGhostStateProvider(
        group_code,
        {0: {"type": "WALL", "is_no_slip": True},
         1: {"type": "FARFIELD", "Q_free": Q_FREE}},
        {"type": "FARFIELD", "Q_free": Q_FREE})
    assert build_boundary_adiabatic_mask(
        flat.n_faces, flat.is_boundary, provider)[bf], "bf 应被判为绝热面"

    res_adiabatic = vf.compute_viscous_residual_fr(
        U, mesh, ops, MU, PR, boundary_ghost_provider=provider,
        flat_face_override=flat2)

    _force_transmissive(monkeypatch)
    provider2 = BoundaryGhostStateProvider(
        group_code,
        {0: {"type": "WALL", "is_no_slip": True},
         1: {"type": "FARFIELD", "Q_free": Q_FREE}},
        {"type": "FARFIELD", "Q_free": Q_FREE})
    res_transmissive = vf.compute_viscous_residual_fr(
        U, mesh, ops, MU, PR, boundary_ghost_provider=provider2,
        flat_face_override=flat2)

    e_diff = np.abs(res_adiabatic[target_cell, :, 4]
                    - res_transmissive[target_cell, :, 4]).max()
    e_scale = max(np.abs(res_transmissive[target_cell, :, 4]).max(), 1e-300)
    assert e_diff / e_scale > 1e-6, (
        f"side={side} order={order}: 单元 {target_cell}（f_int={f_int} 的"
        f"{'neighbor' if side == 'neighbor' else 'owner'}，与 bf={bf} 的 owner "
        f"{flat.owner_cell[bf]} 不同）能量残差没变（相对差 {e_diff/e_scale:.3e}）"
        f"——B-8 {side} 半区没有按配对面的热边界类型分派 ∇T")

    np.testing.assert_allclose(
        res_adiabatic[target_cell, :, :4], res_transmissive[target_cell, :, :4],
        rtol=0, atol=0,
        err_msg=f"side={side}: B-8 半区的质量/动量分量被改动了")
