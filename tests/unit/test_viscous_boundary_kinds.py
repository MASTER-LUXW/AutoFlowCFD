"""粘性通量的逐面边界种类（`core/fr_operators/flux_kernels/viscous_bc.py`）。

## 修的是什么（2026-09-30；取代 2026-09-15 的"绝热面掩码"）

边界面的公共粘性通量此前除绝热类（WALL/SYMMETRY）把 ∇T 法向镜像外，一律
"本侧梯度、只罚速度"。于是出口/远场/入口的温度、出口与对称面切向的速度在
扩散算子里没有任何边界条件，公共法向粘性通量就等于本单元自己的通量，离散
扩散算子失去强制性 —— Blasius P2 隐式稳态发散的根因（完整论证与实测见
`viscous_bc.py` 模块文档）。强制性本身的判据在
`test_viscous_heat_conduction_sign.py::test_viscous_operators_are_coercive_with_mixed_boundaries`。

## 判据

1. **算子层（精确恒等式）**：`mirror_normal_component` 与
   `mirror_velocity_gradient` 的代数性质（面平均后法向热通量/切向牵引
   精确为零、对合、与法向朝向和缩放无关）。
2. **分类**：每种 BC 配置落到正确的种类，逐面、按 `default_config`、缓存。
3. **几何层**：镜像用的逆变行 `adj_row` 与物理面法向平行。
4. **端到端（精确守恒恒等式）**：边界法向粘性通量为零的种类（绝热类的
   能量、镜像类的切向动量、Neumann 的全部分量）下，全域积分的粘性残差
   恒为零——旧处理下对称面切向应力与出口热通量都原样穿过边界，这两条
   必然失败。
5. **混合拆分面（B-8）两个半区**按配对面的种类分派。
"""

import dataclasses

import numpy as np
import pytest

from autoflowcfd.boundary.fr_ghost_state import (
    ADIABATIC_THERMAL_BC_TYPES,
    BoundaryGhostStateProvider,
    build_viscous_boundary_kind,
)
from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
from autoflowcfd.core.fr_operators.flux_kernels import (
    VBC_DIRICHLET, VBC_INLET, VBC_INTERIOR, VBC_MIRROR, VBC_NEUMANN, VBC_NOSLIP_WALL,
    mirror_normal_component, mirror_velocity_gradient,
)
from autoflowcfd.core.fr_residual.inviscid import (
    DefaultGhostProvider, primitive_to_conserved,
)
from autoflowcfd.core.fr_residual import viscous_flux as vf
from autoflowcfd.fr.native_padding import real_sps_per_cell
from autoflowcfd.fr.native_prism.quadrature import build_native_prism_sp_weights
from autoflowcfd.fr.operators import generate_fr_operators

from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh
from tests.validation._channel_mesh import build_channel_mesh_prism, build_face_exact_ghost_provider

MU = 1.8e-5
PR = 0.72
Q_FREE = np.array([1.225, 30.0, 0.0, 0.0, 101325.0])
_KIND_SYMBOL = "autoflowcfd.boundary.fr_ghost_state.build_viscous_boundary_kind"


# ---------------------------------------------------------------------------
# 1. 算子层：精确代数性质
# ---------------------------------------------------------------------------

class TestMirrorNormalComponent:
    @pytest.fixture
    def rng(self):
        return np.random.default_rng(20260915)

    def test_average_has_exactly_zero_normal_component(self, rng):
        """面平均 0.5*(g + mirror(g)) 的法向分量精确为零 —— 离散传导热通量
        a·q_avg = -k*(a·∇T_avg) 恒等于零的充要条件。"""
        for _ in range(200):
            g = rng.standard_normal(3) * 10 ** rng.uniform(-6, 6)
            a = rng.standard_normal(3) * 10 ** rng.uniform(-6, 6)
            g_avg = 0.5 * (g + mirror_normal_component(g, a))
            proj = float(np.dot(g_avg, a))
            scale = np.linalg.norm(g) * np.linalg.norm(a)
            assert abs(proj) <= 1e-14 * scale

    def test_is_involutive_and_isometric(self, rng):
        for _ in range(100):
            g = rng.standard_normal(3)
            a = rng.standard_normal(3)
            m = mirror_normal_component(g, a)
            assert np.abs(mirror_normal_component(m, a) - g).max() < 1e-13
            assert abs(np.linalg.norm(m) - np.linalg.norm(g)) < 1e-13 * np.linalg.norm(g)

    def test_independent_of_normal_orientation_and_scale(self, rng):
        """只依赖面的方向，不依赖内/外法向约定与 |adj_row|（逆变行的模含面积/
        雅可比因子，结果若依赖它，不同网格尺度下行为就会不一致）。"""
        for _ in range(100):
            g = rng.standard_normal(3)
            a = rng.standard_normal(3)
            base = mirror_normal_component(g, a)
            for s in (-1.0, -3.7e5, 2.0, 1.1e-7):
                np.testing.assert_allclose(mirror_normal_component(g, a * s), base, rtol=1e-12, atol=1e-14)

    def test_tangential_component_is_preserved(self, rng):
        """绝热约束的是 dT/dn，沿面的温度变化是真实物理量，不能被一起抹掉。"""
        for _ in range(100):
            g = rng.standard_normal(3)
            a = rng.standard_normal(3)
            n = a / np.linalg.norm(a)
            m = mirror_normal_component(g, a)
            assert np.abs((m - np.dot(m, n) * n) - (g - np.dot(g, n) * n)).max() < 1e-13

    def test_degenerate_adj_row_returns_input(self):
        g = np.array([1.0, -2.0, 3.0])
        out = mirror_normal_component(g, np.zeros(3))
        np.testing.assert_array_equal(out, g)


class TestMirrorVelocityGradient:
    @pytest.fixture
    def rng(self):
        return np.random.default_rng(20260930)

    def test_average_has_zero_tangential_traction(self, rng):
        """核心判据：面平均梯度的应变法-切分量精确为零 ⇒ 公共牵引只有法向分量。"""
        for _ in range(200):
            g = rng.standard_normal((3, 3)) * 10 ** rng.uniform(-4, 4)
            a = rng.standard_normal(3) * 10 ** rng.uniform(-4, 4)
            n = a / np.linalg.norm(a)
            g_avg = 0.5 * (g + mirror_velocity_gradient(g, a))
            Sn = 0.5 * (g_avg + g_avg.T) @ n
            assert np.abs(Sn - np.dot(n, Sn) * n).max() <= 1e-13 * np.abs(g).max()

    def test_preserves_normal_normal_and_tangential_strain(self, rng):
        """法-法与切-切应变不变（对称面上这两者是真实物理量）。"""
        for _ in range(100):
            g = rng.standard_normal((3, 3))
            a = rng.standard_normal(3)
            n = a / np.linalg.norm(a)
            t = np.cross(n, rng.standard_normal(3))
            t /= np.linalg.norm(t)
            m = mirror_velocity_gradient(g, a)
            assert abs(n @ m @ n - n @ g @ n) < 1e-13
            assert abs(t @ m @ t - t @ g @ t) < 1e-13

    def test_equals_gradient_of_mirrored_field(self, rng):
        """定义性质：镜像场 `u_g(x) = R u(Rx)` 的梯度就是 `R ∇u R`；对合、与法向
        朝向与缩放无关。"""
        for _ in range(100):
            g = rng.standard_normal((3, 3))
            a = rng.standard_normal(3)
            n = a / np.linalg.norm(a)
            R = np.eye(3) - 2.0 * np.outer(n, n)
            m = mirror_velocity_gradient(g, a)
            np.testing.assert_allclose(m, R @ g @ R, atol=1e-13)
            np.testing.assert_allclose(mirror_velocity_gradient(m, a), g, atol=1e-13)
            for s in (-1.0, 3.3e4, 2.1e-6):
                np.testing.assert_allclose(mirror_velocity_gradient(g, a * s), m, atol=1e-12)

    def test_degenerate_adj_row_returns_input(self):
        g = np.arange(9.0).reshape(3, 3)
        np.testing.assert_array_equal(mirror_velocity_gradient(g, np.zeros(3)), g)


# ---------------------------------------------------------------------------
# 2. 逐面种类分类
# ---------------------------------------------------------------------------

def _provider_all(bc_type, flat, **cfg):
    group_code = np.full(flat.n_faces, -1, dtype=np.int64)
    group_code[flat.is_boundary] = 0
    conf = {"type": bc_type}
    conf.update(cfg)
    return BoundaryGhostStateProvider(group_code, {0: conf}, {"type": "FARFIELD", "Q_free": Q_FREE})


class TestViscousBoundaryKind:
    @pytest.fixture(scope="class")
    def flat(self):
        return get_flat_face_geometry(_build_synthetic_mixed_mesh(1), generate_fr_operators(1))

    @pytest.mark.parametrize("bc_type,cfg,expected", [
        ("WALL", {"is_no_slip": True}, VBC_NOSLIP_WALL),
        ("WALL", {}, VBC_NOSLIP_WALL),
        ("WALL", {"is_no_slip": False}, VBC_MIRROR),
        ("SYMMETRY", {}, VBC_MIRROR),
        ("FARFIELD", {"Q_free": Q_FREE}, VBC_DIRICHLET),
        ("INLET", {"Q_inlet": Q_FREE}, VBC_INLET),
        ("OUTLET", {"p_outlet": 101325.0}, VBC_NEUMANN),
    ])
    def test_classification(self, flat, bc_type, cfg, expected):
        kind = build_viscous_boundary_kind(flat.n_faces, flat.is_boundary, _provider_all(bc_type, flat, **cfg))
        assert flat.is_boundary.any()
        assert (kind[flat.is_boundary] == expected).all()
        assert (kind[~flat.is_boundary] == VBC_INTERIOR).all()
        assert kind.dtype == np.int8

    def test_mixed_groups_are_per_face(self, flat):
        bnd = np.nonzero(flat.is_boundary)[0]
        group_code = np.full(flat.n_faces, -1, dtype=np.int64)
        group_code[bnd] = np.arange(bnd.size) % 2
        p = BoundaryGhostStateProvider(
            group_code,
            {0: {"type": "WALL", "is_no_slip": True}, 1: {"type": "OUTLET", "p_outlet": 1e5}},
            {"type": "FARFIELD", "Q_free": Q_FREE})
        kind = build_viscous_boundary_kind(flat.n_faces, flat.is_boundary, p)
        np.testing.assert_array_equal(
            kind[bnd], np.where(np.arange(bnd.size) % 2 == 0, VBC_NOSLIP_WALL, VBC_NEUMANN))

    def test_unmatched_faces_follow_default_config(self, flat):
        p = BoundaryGhostStateProvider(np.full(flat.n_faces, -1, dtype=np.int64), {}, {"type": "SYMMETRY"})
        kind = build_viscous_boundary_kind(flat.n_faces, flat.is_boundary, p)
        assert (kind[flat.is_boundary] == VBC_MIRROR).all()

    def test_default_provider_is_neumann(self, flat):
        """DefaultGhostProvider 的幽灵态是本侧延拓 —— 按延拓分量的规则取零法向
        粘性通量，而不是"本侧通量原样穿过边界"那个不施加任何条件的旧处理。"""
        kind = build_viscous_boundary_kind(flat.n_faces, flat.is_boundary, DefaultGhostProvider())
        assert (kind[flat.is_boundary] == VBC_NEUMANN).all()
        assert (kind[~flat.is_boundary] == VBC_INTERIOR).all()

    def test_wrapper_sharing_bc_semantics_is_classified(self, flat):
        """委托给 BoundaryGhostStateProvider、共享其 BC 属性的包装层（如验证算例的
        逐点剖面入口）按内层语义分类。2026-09-30 实测：按 isinstance 判时这类
        包装层整层落进兜底，无滑移壁被当成零通量边界，Blasius cf 只剩 12%~19%。"""
        inner = _provider_all("WALL", flat, is_no_slip=True)

        class Wrapper:
            def __init__(self, p):
                self._p = p
                self.group_code = p.group_code
                self.code_to_config = p.code_to_config
                self.default_config = p.default_config

            def __call__(self, face_idx, Q_owner_fp, true_normal):
                return self._p(face_idx, Q_owner_fp, true_normal)

        kind = build_viscous_boundary_kind(flat.n_faces, flat.is_boundary, Wrapper(inner))
        assert (kind[flat.is_boundary] == VBC_NOSLIP_WALL).all()

    def test_provider_without_bc_semantics_is_rejected(self, flat):
        """没有可读 BC 语义的 provider 不能被静默猜成某种边界。"""
        with pytest.raises(TypeError):
            build_viscous_boundary_kind(flat.n_faces, flat.is_boundary, lambda f, q, n: q)

    def test_cached_per_provider(self, flat):
        p = _provider_all("WALL", flat)
        assert (build_viscous_boundary_kind(flat.n_faces, flat.is_boundary, p)
                is build_viscous_boundary_kind(flat.n_faces, flat.is_boundary, p))

    def test_closed_boundary_type_set_is_explicit(self):
        assert ADIABATIC_THERMAL_BC_TYPES == frozenset({"WALL", "SYMMETRY"})


# ---------------------------------------------------------------------------
# 3. 几何层：逆变行与物理法向平行
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("order", [1, 2])
def test_adj_row_is_parallel_to_true_normal(order):
    """镜像用的是逆变行 adj_row；若它与物理面法向不平行，"法向分量为零"指的
    就不是物理法向。判据用叉乘模 / 模积（= |sin theta|）。"""
    mesh = _build_synthetic_mixed_mesh(order)
    flat = get_flat_face_geometry(mesh, generate_fr_operators(order))
    bnd = np.nonzero(flat.is_boundary & flat.owner_is_primary)[0]
    assert bnd.size > 0
    a = flat.owner_adj_row_exact[bnd]
    n = flat.true_normal[bnd]
    denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(n, axis=-1)
    good = denom > 0
    sin_theta = np.linalg.norm(np.cross(a, n), axis=-1)[good] / denom[good]
    assert sin_theta.max() < 1e-12


# ---------------------------------------------------------------------------
# 4. 端到端：精确守恒恒等式
# ---------------------------------------------------------------------------
#
# 粘性残差按原生棱柱解点求积权重与 det(J) 积分，等于全部边界面上公共法向粘性
# 通量之和（内部面两侧公共通量与罚项反号相消；提升算子积分精确）。于是边界
# 法向通量为零的种类下全域积分恒为零。只取两条对**线性**通量成立的恒等式：
# 静止场的能量（纯热传导）与剪切流的动量（应力对速度梯度线性）。有速度时能量
# 通量 `tau.u` 是非线性的，通量点上按外插状态重算与外插通量多项式之差是混叠
# 量级，那条不是这里的判据。

LX, HY, LZ = 0.4, 0.1, 0.08


def _channel(order):
    mesh = build_channel_mesh_prism(order, 3, 3, 2, LX, HY, LZ)
    ops = generate_fr_operators(order)
    nr, _ = real_sps_per_cell(order)
    w = build_native_prism_sp_weights(order)
    dj = np.asarray(mesh.jacobians["det_jacs"]).reshape(mesh.n_cells, mesh.n_sps_per_cell)[:, :nr]
    return mesh, ops, (lambda r: (r[:, :nr] * dj[..., None] * w[None, :, None]).sum((0, 1)))


def _run(mesh, ops, Q, bc):
    prov = build_face_exact_ghost_provider(mesh, LX, HY, LZ, bc)
    return vf.compute_viscous_residual_fr(primitive_to_conserved(Q), mesh, ops, MU, PR,
                                          boundary_ghost_provider=prov)


@pytest.mark.parametrize("order", [1, 2])
def test_adiabatic_and_neumann_boundaries_conserve_energy_exactly(order):
    """静止、非均匀温度：无滑移壁/滑移壁/对称面（∇T 镜像）与出口（零法向通量）
    上边界热通量为零，全域能量积分残差恒为零。旧处理下出口热通量原样穿过边界。"""
    mesh, ops, integ = _channel(order)
    X = np.asarray(mesh.sps_coords)
    Q = np.zeros(X.shape[:2] + (5,))
    Q[..., 0] = 1.225
    Q[..., 4] = 101325.0 * (1 + 0.05 * X[..., 0] / LX + 0.03 * np.sin(7 * X[..., 1] / HY)
                            - 0.02 * X[..., 2] / LZ)
    bc = {"z_min": {"type": "SYMMETRY"}, "z_max": {"type": "SYMMETRY"},
          "wall_bottom": {"type": "WALL", "is_no_slip": True},
          "wall_top": {"type": "WALL", "is_no_slip": False},
          "x_min": {"type": "OUTLET", "p_outlet": 101325.0},
          "x_max": {"type": "OUTLET", "p_outlet": 101325.0}}
    r = _run(mesh, ops, Q, bc)
    total, scale = integ(r)[4], integ(np.abs(r))[4]
    assert abs(total) <= 1e-10 * scale, f"能量积分 {total:.3e}（尺度 {scale:.3e}）"


@pytest.mark.parametrize("order", [1, 2])
def test_mirror_and_neumann_boundaries_carry_no_shear(order):
    """剪切流 `u = u(y, z)`：y/z 面对称（切向牵引为零、法向速度本就为零）、x 面
    出口（零法向通量），全域 x 动量积分残差恒为零。旧处理下对称面把 `mu du/dy`
    原样当作边界剪应力。"""
    mesh, ops, integ = _channel(order)
    X = np.asarray(mesh.sps_coords)
    Q = np.zeros(X.shape[:2] + (5,))
    Q[..., 0] = 1.225
    Q[..., 4] = 101325.0
    Q[..., 1] = 300.0 * X[..., 1] + 40.0 * np.cos(9.0 * X[..., 2] / LZ) * X[..., 1]
    bc = {n: {"type": "SYMMETRY"} for n in ("z_min", "z_max", "wall_bottom", "wall_top")}
    bc["x_min"] = {"type": "OUTLET", "p_outlet": 101325.0}
    bc["x_max"] = {"type": "OUTLET", "p_outlet": 101325.0}
    r = _run(mesh, ops, Q, bc)
    total, scale = integ(r)[1], integ(np.abs(r))[1]
    assert abs(total) <= 1e-10 * scale, f"x 动量积分 {total:.3e}（尺度 {scale:.3e}）"


@pytest.mark.parametrize("order", [1, 2])
def test_symmetry_matches_wall_for_energy_at_zero_velocity(order):
    """速度恒为零时对称面与无滑移壁的能量残差逐位相同：两者都镜像 ∇T、都不罚
    温度，静止场上两种幽灵态也相同。"""
    mesh = _build_synthetic_mixed_mesh(order)
    ops = generate_fr_operators(order)
    flat = get_flat_face_geometry(mesh, ops)
    x = mesh.sps_coords.reshape(mesh.n_cells, mesh.n_sps_per_cell, 3)
    Q = np.zeros((mesh.n_cells, mesh.n_sps_per_cell, 5))
    Q[..., 0] = 1.225
    Q[..., 4] = 101325.0 * (1.0 + 0.05 * x[..., 0] + 0.03 * x[..., 1] - 0.02 * x[..., 2])
    U = primitive_to_conserved(Q)
    e_wall = vf.compute_viscous_residual_fr(
        U, mesh, ops, MU, PR, boundary_ghost_provider=_provider_all("WALL", flat, is_no_slip=True))[..., 4]
    e_sym = vf.compute_viscous_residual_fr(
        U, mesh, ops, MU, PR, boundary_ghost_provider=_provider_all("SYMMETRY", flat))[..., 4]
    np.testing.assert_allclose(e_wall, e_sym, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# 5. 混合拆分面（B-8）的两个半区 —— 手工合成配置
# ---------------------------------------------------------------------------
#
# B-8 混合拆分面只在真实带边界层的网格上出现（`_build_synthetic_mixed_mesh` 上
# `mixed_nb_partner`/`mixed_ow_partner` 全为 -1），所以把某个内部面 f_int 手工
# 登记成"配对面是边界面 bf、整张面取边界半区"，让 kernel 真的走进 B-8 分支。
# 归因：provider 与 flat 不变，只把 bf 的种类在 Dirichlet 与 Neumann 之间切换。
# 静止、非均匀温度、幽灵态是无滑移壁（温度与本侧相同）时：Dirichlet 下公共
# 通量等于本侧通量且温度罚项为零（跳变量为零），Neumann 下跳变量是
# `-a·G(本侧)`（非零热通量）——能量不同、质量/动量逐位相同。挑一个相关单元
# 与 bf 的 owner 不同的 f_int，差异就只能来自 B-8 半区。

def _pick_faces_for_mixed(flat, need_neighbor_side):
    """挑一对 (f_int, bf)：f_int 是内部面，bf 是边界面，且两者的相关单元不同。"""
    bnd = np.nonzero(flat.is_boundary & flat.owner_is_primary)[0]
    interior = np.nonzero((~flat.is_boundary) & flat.owner_is_primary & flat.neighbor_is_primary)[0]
    assert bnd.size > 0 and interior.size > 0
    for f_int in interior:
        target = flat.neighbor_cell[f_int] if need_neighbor_side else flat.owner_cell[f_int]
        for bf in bnd:
            if flat.owner_cell[bf] != target:
                return int(f_int), int(bf), int(target)
    raise AssertionError("找不到满足归因条件的 (f_int, bf) 组合")


def _flat_with_mixed(flat, f_int, bf, side):
    """返回一份把 f_int 登记成混合拆分面的 flat 副本（整张面取边界半区）。"""
    nb_partner, nb_mask = flat.mixed_nb_partner.copy(), flat.mixed_nb_mask.copy()
    ow_partner, ow_mask = flat.mixed_ow_partner.copy(), flat.mixed_ow_mask.copy()
    if side == "owner":
        nb_partner[f_int] = bf
        nb_mask[f_int, :] = True
    else:
        ow_partner[f_int] = bf
        ow_mask[f_int, :] = True
    return dataclasses.replace(flat, mixed_nb_partner=nb_partner, mixed_nb_mask=nb_mask,
                               mixed_ow_partner=ow_partner, mixed_ow_mask=ow_mask)


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("side", ["owner", "neighbor"])
def test_mixed_split_half_dispatches_by_partner_kind(order, side, monkeypatch):
    mesh = _build_synthetic_mixed_mesh(order)
    ops = generate_fr_operators(order)
    flat = get_flat_face_geometry(mesh, ops)
    f_int, bf, target_cell = _pick_faces_for_mixed(flat, side == "neighbor")
    flat2 = _flat_with_mixed(flat, f_int, bf, side)
    x = mesh.sps_coords.reshape(mesh.n_cells, mesh.n_sps_per_cell, 3)
    Q = np.zeros((mesh.n_cells, mesh.n_sps_per_cell, 5))
    Q[..., 0] = 1.225
    Q[..., 4] = 101325.0 * (1.0 + 0.05 * x[..., 0] + 0.03 * x[..., 1] - 0.02 * x[..., 2])
    U = primitive_to_conserved(Q)
    provider = _provider_all("WALL", flat, is_no_slip=True)

    def run(kind_bf):
        kind = np.where(flat.is_boundary, VBC_NOSLIP_WALL, VBC_INTERIOR).astype(np.int8)
        kind[bf] = kind_bf
        monkeypatch.setattr(_KIND_SYMBOL, lambda n, b, g: kind)
        return vf.compute_viscous_residual_fr(U, mesh, ops, MU, PR, boundary_ghost_provider=provider,
                                              flat_face_override=flat2)

    r_dir, r_neu = run(VBC_DIRICHLET), run(VBC_NEUMANN)
    e_diff = np.abs(r_dir[target_cell, :, 4] - r_neu[target_cell, :, 4]).max()
    e_scale = max(np.abs(r_neu[target_cell, :, 4]).max(), 1e-300)
    assert e_diff / e_scale > 1e-6, (
        f"side={side} order={order}: 单元 {target_cell} 的能量残差没变 —— B-8 {side} 半区"
        f"没有按配对面的种类分派")
    np.testing.assert_allclose(r_dir[target_cell, :, :4], r_neu[target_cell, :, :4], rtol=0, atol=0)
