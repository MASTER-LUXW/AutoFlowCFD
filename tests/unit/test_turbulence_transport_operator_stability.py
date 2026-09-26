"""湍流标量对流/扩散算子的线性稳定性：精确算子矩阵的谱全部 `Re(lambda) <= 0`。

回归对象（2026-09-26）：`core/turbulence/transport/face_frames.py` 模块文档记录
的两个缺陷——neighbor 侧用了 owner 通量点顺序的跳变量（对流与扩散）、扩散在
neighbor 侧符号相反（反扩散）。两者都只在"两侧通量点顺序不一致"的面上暴露：
此前的 CPU/GPU 对照测试用的合成网格只有 2 个内部面、顺序恰好一致，于是 CPU 与
GPU 错得一模一样却互相吻合。这里用的小通道网格两类单元都有顺序不一致的面
（下面第一条断言钉住这个前提，否则本测试没有检验力）。

为什么可以用精确谱（项目记忆 `fd_jacobian_spectrum_unreliable` 说数值雅可比谱
不可靠，那是对 AUSM+up/幽灵态的不可微处）：`rho*u` 固定时对流算子对 `phi`
**精确线性**（上风选择只取决于质量通量的符号），`Gamma` 固定时扩散算子同样
精确线性，所以对单位向量逐列作用得到的就是算子矩阵本身，不是差分近似。
"""

import numpy as np
import pytest

from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
from autoflowcfd.core.turbulence.transport import (
    compute_scalar_convection_residual,
    compute_scalar_diffusion_residual,
)
from autoflowcfd.fr.native_padding import real_sps_per_cell
from autoflowcfd.fr.operators import generate_fr_operators
from tests.validation._channel_mesh import build_channel_mesh, build_channel_mesh_prism

_BUILDERS = {"tet": build_channel_mesh, "prism": build_channel_mesh_prism}


def _setup(kind, order):
    mesh = _BUILDERS[kind](order, 2, 2, 2, 0.4, 0.1, 0.08)
    ops = generate_fr_operators(order)
    flat = get_flat_face_geometry(mesh, ops)
    n_prism, n_tet = real_sps_per_cell(order)
    n_real = np.where(np.arange(mesh.n_cells) < mesh.n_prism_cells, n_prism, n_tet)
    real = np.arange(mesh.n_sps_per_cell)[None, :] < n_real[:, None]
    return mesh, ops, flat, real


def _mismatched_faces(mesh, flat):
    X = np.asarray(mesh.sps_coords)
    E = np.asarray(flat.boundary_extrap_native)
    own, nei = np.asarray(flat.owner_cell), np.asarray(flat.neighbor_cell)
    f = np.nonzero((nei >= 0) & np.asarray(flat.neighbor_is_primary).astype(bool))[0]
    xo = np.einsum("fis,fsd->fid", E[np.asarray(flat.owner_cube_face)[f] - 6], X[own[f]])
    xn = np.einsum("fis,fsd->fid", E[np.asarray(flat.neighbor_cube_face)[f] - 6], X[nei[f]])
    return int((np.linalg.norm(xo - xn, axis=-1).max(axis=1) > 1e-9).sum())


def _operator_spectrum(apply, real):
    idx = np.argwhere(real)
    n = len(idx)
    A = np.empty((n, n))
    for j, (c, s) in enumerate(idx):
        e = np.zeros(real.shape)
        e[c, s] = 1.0
        A[:, j] = apply(e)[real]
    return np.linalg.eigvals(A)


@pytest.mark.parametrize("kind", ["tet", "prism"])
@pytest.mark.parametrize("order", [1, 2])
def test_scalar_convection_operator_is_stable(kind, order):
    """均匀来流、开放边界流入取零：迎风 DG 对流算子的谱必须全部在闭左半平面。"""
    mesh, ops, flat, real = _setup(kind, order)
    assert _mismatched_faces(mesh, flat) > 0, "网格没有两侧顺序不一致的面，本测试失去检验力"
    shape = real.shape
    rho = np.full(shape, 1.2)
    vel = np.broadcast_to(np.array([30.0, 11.0, -7.0]), shape + (3,)).copy()
    bnd = np.asarray(flat.is_boundary).astype(bool)

    def apply(phi):
        return compute_scalar_convection_residual(phi, rho, vel, mesh, ops, open_boundary_face=bnd,
                                                  freestream_value=0.0) / rho

    lam = _operator_spectrum(apply, real)
    assert lam.real.max() <= 1e-9 * np.abs(lam).max(), (
        f"{kind} P{order}: 对流算子有增长模态 max Re(lambda) = {lam.real.max():.3e}"
        f"（谱半径 {np.abs(lam).max():.3e}）")


@pytest.mark.parametrize("kind", ["tet", "prism"])
@pytest.mark.parametrize("order", [1, 2])
def test_scalar_diffusion_operator_is_stable(kind, order):
    """均匀 Gamma：IIPG 内罚扩散算子的谱必须全部在闭左半平面（强制性）。"""
    mesh, ops, flat, real = _setup(kind, order)
    assert _mismatched_faces(mesh, flat) > 0, "网格没有两侧顺序不一致的面，本测试失去检验力"
    gamma = np.full(real.shape, 1.2 * 1e-3)

    def apply(phi):
        return compute_scalar_diffusion_residual(phi, gamma, mesh, ops) / 1.2

    lam = _operator_spectrum(apply, real)
    assert lam.real.max() <= 1e-9 * np.abs(lam).max(), (
        f"{kind} P{order}: 扩散算子有增长模态 max Re(lambda) = {lam.real.max():.3e}"
        f"（谱半径 {np.abs(lam).max():.3e}）")


def test_scalar_diffusion_is_conservative_across_unequal_cells():
    """全边界齐次 Neumann 时扩散残差的守恒加权和恒为零（守恒量是
    `sum_s W_cs U_cs`，W 与正性限制器同一份）。内罚项长度尺度必须两侧单值
    （`FlatFaceGeometry.ip_length`）：两侧各用自己的 `V/A` 时，相邻体积不等的面上
    两侧罚通量不等。混合网格棱柱/四面体相邻体积比为 2。"""
    from autoflowcfd.core.time_integration.positivity import build_positivity_limiter
    from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh

    mesh = _build_synthetic_mixed_mesh(1)
    ops = generate_fr_operators(1)
    flat = get_flat_face_geometry(mesh, ops)
    v = np.asarray(mesh.get_all_cell_volumes()).ravel()
    own, nei = np.asarray(flat.owner_cell), np.asarray(flat.neighbor_cell)
    inner = nei >= 0
    assert (np.maximum(v[own[inner]], v[nei[inner]]) / np.minimum(v[own[inner]], v[nei[inner]])).max() > 1.5
    W = build_positivity_limiter(mesh, ops).W
    phi = np.random.default_rng(0).standard_normal(W.shape)
    R = compute_scalar_diffusion_residual(phi, np.full(W.shape, 1e-3), mesh, ops)
    assert abs((W * R).sum()) <= 1e-13 * np.abs(W * R).sum()
