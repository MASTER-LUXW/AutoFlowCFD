"""湍流输运过积分测试（test_turbulence_transport_overintegration*.py）共用的辅助函数与解析场。"""

import numpy as np

from autoflowcfd.fr.operators import generate_fr_operators

from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh


def _real_dof_mask(mesh, order):
    """(n_cells, n_sps) 布尔掩码，标出**真实自由度**。

    必须有这一步才能和解析参照比较：原生基只有前若干个 SP 是自由度，
    其余是零填充槽位，填充行的散度恒为 0，与解析值无关。用
    "div(x,0,0)=1"这个手算算例定位到的：不加掩码时结果是 [0,1] 区间
    而不是恒 1，min=0 全部来自填充位。

    **棱柱同样可能有填充槽位（2026-09-20）**：原先这里写死"棱柱的
    `(order+1)^3` 个 SP 全是自由度"，那只对坍缩棱柱基成立；原生棱柱基
    （2026-09-20 起是默认）每单元只有 `(p+1)^2(p+2)/2` 个真实自由度。
    漏掉这一点的直接后果是本文件的误差被填充槽位主导：去混叠与不去
    混叠算出**逐位相同**的误差（实测 8.241e-01 vs 8.241e-01），判据
    完全失效。真实自由度数走唯一入口 `fr/native_padding.real_sps_per_cell`。
    """
    from autoflowcfd.fr.native_padding import real_sps_per_cell

    n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
    n_real_prism, n_real_tet = real_sps_per_cell(order)
    mask = np.ones((n_cells, n_sps), dtype=bool)
    mask[:mesh.n_prism_cells, n_real_prism:] = False
    mask[mesh.n_prism_cells:, n_real_tet:] = False
    return mask


def _actual_over_order(oi):
    """从过积分上下文**实际读出** over_order，而不是在测试里重算规则。

    2026-09-15 的教训：本文件原来内部写死 `min(2*order, 3)`，而生产规则
    随后改成了 `3*order`（只影响 order=1，见 fr/operators.py 该处文档）。
    测试当时仍然通过，但走的是错误的分支、判据的理由已经和现实脱节。
    取**棱柱段**的细点数反解：`n_fine_prism = (over_order+1)^3`。

    2026-09-17 起上下文不再有共享的 `oi["n_fine"]`——native 四面体过积分
    的细网格轴不再填充到棱柱宽度，两段的 n_fine 不同了。所以这里明确
    取棱柱段，而不是随便拿一段来反解。

    **反解方式按棱柱基分档（2026-09-20）**：原先写死"开立方"
    （`n_fine_prism = (oo+1)^3`），那只对坍缩棱柱基成立；原生棱柱基的
    细点数是 `(oo+1)^2(oo+2)/2`（P1 oo=2 -> 18，不是完全立方数，原先
    会直接断言失败）。两档统一向唯一入口 `prism_n_fine` 反查，而不是在
    测试里再写一份公式。
    """
    from autoflowcfd.fr.overintegration_order import prism_n_fine

    n_fine_prism = oi["segs"][0][2]
    for oo in range(1, 32):
        if prism_n_fine(oo) == n_fine_prism:
            return oo
    raise AssertionError(
        f"棱柱段 n_fine={n_fine_prism} 不对应任何 over_order —— "
        f"细点数公式与 `fr/overintegration_order.prism_n_fine` 脱节了")


def _per_type_slices(mesh, order):
    """[("prism", 切片), ("tet", 切片)]，四面体那一片只取真实自由度。

    分类型统计是必须的：两类单元可达的精度不同（棱柱映射非仿射 ->
    adj(J) 是非平凡多项式 -> 乘积次数被进一步推高；native 四面体仿射 ->
    adj(J) 常数）。混在一起会掩盖"仿射单元上已经精确"这条最强证据。
    """
    # 两类单元各自的真实自由度数走唯一入口（原生棱柱基下棱柱同样有
    # 零填充槽位，见 `_real_dof_mask` 里那段 2026-09-20 的说明）。
    from autoflowcfd.fr.native_padding import real_sps_per_cell

    n_real_prism, n_real_tet = real_sps_per_cell(order)
    return [
        ("prism", (slice(0, mesh.n_prism_cells), slice(0, n_real_prism))),
        ("tet", (slice(mesh.n_prism_cells, mesh.n_cells),
                 slice(0, n_real_tet))),
    ]


def _coarse_convection_div(scalar_field, rho, velocity, mesh, ops):
    """coarse 路径的对流体积项 div_F（复刻生产代码的非过积分分支）。"""
    from autoflowcfd.core.fr_operators.volume_contract import (
        contravariant_flux_from_metric,
    )
    from autoflowcfd.core.turbulence.transport_kernel import (
        scalar_convection_volume_kernel,
    )
    n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
    n_prism = mesh.n_prism_cells
    det = mesh.jacobians["det_jacs"].reshape(n_cells, n_sps)
    inv = mesh.jacobians["inv_jacs"].reshape(n_cells, n_sps, 3, 3)
    rho_u = rho[:, :, None] * velocity
    rho_u_tilde = contravariant_flux_from_metric(det, inv, rho_u[..., None])[..., 0]
    tet_op = (ops.D_native_tet_padded
              if getattr(ops, "D_native_tet_padded", None) is not None
              else ops.D_3d_tet)
    div = np.empty((n_cells, n_sps))
    if n_prism > 0:
        scalar_convection_volume_kernel(
            np.ascontiguousarray(scalar_field[:n_prism]),
            np.ascontiguousarray(rho_u_tilde[:n_prism]),
            np.ascontiguousarray(ops.D_3d_prism), div[:n_prism])
    if n_cells > n_prism:
        scalar_convection_volume_kernel(
            np.ascontiguousarray(scalar_field[n_prism:]),
            np.ascontiguousarray(rho_u_tilde[n_prism:]),
            np.ascontiguousarray(tet_op), div[n_prism:])
    return div


class _PolyField:
    """rho / u / phi 都取显式一次多项式，三重乘积是三次。

        rho = r0 + rx*x + ry*y + rz*z
        u_i = a_i + b_i*x + c_i*y + d_i*z
        phi = p0 + px*x + py*y + pz*z

    `div(rho*u*phi)` 用 sympy 级别的手工展开太易错，改用**中心差分对
    解析函数本身求导**——被求导的是闭式表达式而不是离散场，步长取
    1e-5 相对尺度，四阶中心差分的截断误差远低于本用例要分辨的
    "混叠误差 vs 机器零"这个量级差。
    """

    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.r = rng.uniform(0.8, 1.3, 4)      # r0, rx, ry, rz
        self.u = rng.uniform(-1.0, 1.0, (3, 4))
        self.p = rng.uniform(0.5, 1.5, 4)

    def rho(self, X):
        return self.r[0] + X @ self.r[1:]

    def vel(self, X):
        return self.u[:, 0][None, :] + X @ self.u[:, 1:].T

    def phi(self, X):
        return self.p[0] + X @ self.p[1:]

    def flux(self, X):
        """rho*u*phi，形状 (n,3)。"""
        return self.rho(X)[:, None] * self.vel(X) * self.phi(X)[:, None]

    def div_exact(self, X, h=None):
        """四阶中心差分求 div(rho*u*phi)（对闭式函数求导，不是对离散场）。"""
        scale = max(np.abs(X).max(), 1.0)
        h = h if h is not None else 1e-4 * scale
        out = np.zeros(X.shape[0])
        for d in range(3):
            e = np.zeros(3); e[d] = h
            f_p2 = self.flux(X + 2 * e)[:, d]
            f_p1 = self.flux(X + e)[:, d]
            f_m1 = self.flux(X - e)[:, d]
            f_m2 = self.flux(X - 2 * e)[:, d]
            out += (-f_p2 + 8 * f_p1 - 8 * f_m1 + f_m2) / (12 * h)
        return out


def _setup(order, seed=0):
    mesh = _build_synthetic_mixed_mesh(order)
    ops = generate_fr_operators(order)
    X = mesh.sps_coords.reshape(-1, 3)
    f = _PolyField(seed)
    n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
    phi = f.phi(X).reshape(n_cells, n_sps)
    rho = f.rho(X).reshape(n_cells, n_sps)
    vel = f.vel(X).reshape(n_cells, n_sps, 3)
    exact = f.div_exact(X).reshape(n_cells, n_sps)
    return mesh, ops, phi, rho, vel, exact


#: `order -> (coarse 误差下界, 去混叠必须达到的改善倍数)`。
#: 数值来自实测（见 `test_convection_volume_term_vs_analytic` 里的记录
#: 表），下界都比实测值留了余量：它们是"判据没有空转"的防护，不是目标值。
#: 2026-09-23 之前这张表的键是 `(棱柱基, order)` —— 坍缩棱柱基删除后只剩
#: 原生一档，所以退化成按 order 分档。
_PRISM_EXPECT = {
    1: (0.1, 10.0),
    2: (1.0e-3, 1.0e6),
}
