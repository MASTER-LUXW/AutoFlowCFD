"""湍流标量对流算子对均匀场与常数平移的保持性（`core/turbulence/transport/convection.py`）。

被求解的方程是 `rho dphi/dt + rho u . grad(phi) = ...`（未知量是 k 与 `w = ln omega` 本身，
不是 `rho phi`），一致的空间离散是对流形式。界面项早已是对流形式的跳变量
`m (phi_up - phi_side)`（均匀场为零）；2026-10-02 之前体积项却是守恒形式
`-div(rho u phi)`，两者只差 `-phi div_vol(rho u)`——只有离散体积散度逐点为零时才消失，
而高阶 FR 的平均流只在"体积散度 + 界面修正"整体上满足连续性。`w = ln omega` 的绝对值
约 8~14，于是伪源 `-w div_vol(rho u)/rho` 在 plate_demo P1 平板前缘锐边上达 +1.1e5 1/s，
压过 omega 耗散把 123 个解点一路推到 `omega_max` 钳位（钳位处残差不可微，JFNK 湍流
GMRES 每步跑满 200 次）。

判据（质量通量刻意取非无散的光滑场，否则本测试没有检验力——第一条断言钉住这个前提）：

1. 均匀标量 `phi = c` 的对流残差为零（相对伪源量级到机器精度）；
2. 平移不变：`R(phi + c) = R(phi)`（对 `ln omega` 这种不以零为基准的量是关键性质）；
3. 对 `phi` 仍精确线性（上风选择只取决于质量通量符号）。
"""

import numpy as np
import pytest

from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
from autoflowcfd.core.turbulence.transport import compute_scalar_convection_residual
from autoflowcfd.fr.native_padding import real_sps_per_cell
from autoflowcfd.fr.operators import generate_fr_operators
from tests.validation._channel_mesh import build_channel_mesh, build_channel_mesh_prism

_BUILDERS = {"tet": lambda *a: build_channel_mesh(*a, rotate_odd_tets=True), "prism": build_channel_mesh_prism}


def _setup(kind, order):
    mesh = _BUILDERS[kind](order, 2, 2, 2, 0.4, 0.1, 0.08)
    ops = generate_fr_operators(order)
    flat = get_flat_face_geometry(mesh, ops)
    n_prism, n_tet = real_sps_per_cell(order)
    n_real = np.where(np.arange(mesh.n_cells) < mesh.n_prism_cells, n_prism, n_tet)
    real = np.arange(mesh.n_sps_per_cell)[None, :] < n_real[:, None]
    X = np.asarray(mesh.sps_coords)
    x, y, z = X[..., 0], X[..., 1], X[..., 2]
    rho = 1.2 + 0.05 * np.sin(7.0 * x) * np.cos(11.0 * y)
    vel = np.stack([30.0 + 40.0 * x, 5.0 + 60.0 * y * z, -3.0 + 25.0 * z], axis=-1)   # div(rho u) != 0
    bnd = np.asarray(flat.is_boundary).astype(bool)
    return mesh, ops, real, rho, vel, bnd, X


def _conv(phi, rho, vel, mesh, ops, bnd, freestream):
    return compute_scalar_convection_residual(phi, rho, vel, mesh, ops, open_boundary_face=bnd,
                                              freestream_value=freestream) / rho


@pytest.mark.parametrize("kind", ["tet", "prism"])
@pytest.mark.parametrize("order", [1, 2])
def test_uniform_scalar_has_zero_convection_residual(kind, order):
    mesh, ops, real, rho, vel, bnd, X = _setup(kind, order)
    c = 13.8
    r = _conv(np.full(real.shape, c), rho, vel, mesh, ops, bnd, c)[real]
    # 前提：质量通量确实非无散（守恒形式在这里给出 O(c * div) 的伪源）
    linear = X[..., 0] - 0.2
    scale = np.abs(_conv(c + linear, rho, vel, mesh, ops, bnd, c)[real]).max()
    assert scale > 1.0
    assert np.abs(r).max() <= 1e-12 * c * scale, (
        f"{kind} P{order}: 均匀标量的对流残差 max |R| = {np.abs(r).max():.3e}（应为 0）")


@pytest.mark.parametrize("kind", ["tet", "prism"])
@pytest.mark.parametrize("order", [1, 2])
def test_convection_is_invariant_to_constant_shift_and_linear(kind, order):
    mesh, ops, real, rho, vel, bnd, X = _setup(kind, order)
    phi = np.sin(5.0 * X[..., 0]) + 0.3 * X[..., 1] * X[..., 2] + 2.0
    psi = np.cos(3.0 * X[..., 2]) - X[..., 0]
    c = 9.0
    r_phi = _conv(phi, rho, vel, mesh, ops, bnd, 0.0)
    r_shift = _conv(phi + c, rho, vel, mesh, ops, bnd, c)
    ref = np.abs(r_phi[real]).max()
    assert np.abs(r_shift - r_phi)[real].max() <= 1e-11 * ref, "常数平移改变了对流残差"
    # 精确线性（来流值随之线性组合）
    r_psi = _conv(psi, rho, vel, mesh, ops, bnd, 0.0)
    r_comb = _conv(2.0 * phi - 3.0 * psi, rho, vel, mesh, ops, bnd, 0.0)
    assert np.abs(r_comb - (2.0 * r_phi - 3.0 * r_psi))[real].max() <= 1e-11 * ref
