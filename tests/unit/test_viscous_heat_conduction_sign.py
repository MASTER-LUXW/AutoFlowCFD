"""粘性能量通量的热传导方向（2026-09-26 修复的一阶物理缺陷的回归测试）。

能量方程的粘性通量 `G_E = tau.u - q`，`q = -k grad(T)`，即 `G_E = tau.u + k grad(T)`，
残差按 `dU/dt = -div F + div G` 组装。此前 CPU 两份实现与 GPU 版都写成
`tau.u + q`（热传导反扩散），见 `flux_kernels.viscous_physical_flux_point` 文档。

三层判据，互相独立：
1. 逐点公式：静止流体里 `G_E` 必须与 `grad(T)` 同向（热量顺梯度反方向流动，
   通量 `-q` 顺梯度方向）；
2. 热斑：均匀压力、静止、局部高温，粘性残差在热斑处 `d(rho E)/dt < 0`；
3. 线性化谱：均匀静止基态上纯粘性算子（线性算子，数值雅可比可靠）没有
   正实部特征值——反扩散时有数十个、特征向量全在能量分量上。
"""

import numpy as np
import pytest

from autoflowcfd.core.fr_operators.flux_kernels import CP_AIR, viscous_physical_flux_point
from autoflowcfd.core.fr_residual.viscous_flux import compute_viscous_residual_fr, viscous_physical_flux
from autoflowcfd.fr.native_padding import real_sps_per_cell
from autoflowcfd.fr.operators import generate_fr_operators
from tests.validation._channel_mesh import (
    build_channel_mesh, build_channel_mesh_prism, build_face_exact_ghost_provider,
)

RHO, P_INF, MU, PR = 1.225, 101325.0, 1.8e-5, 0.72
LX, H, LZ = 0.4, 0.1, 0.08


def test_pointwise_energy_flux_follows_fourier_law():
    Q = np.array([RHO, 0.0, 0.0, 0.0, P_INF])
    gT = np.array([3.0, -2.0, 0.5])
    mu_t = 7.0 * MU
    G = viscous_physical_flux_point(Q, np.zeros((3, 3)), gT, MU, PR, mu_t, 0.9)
    k = MU * CP_AIR / PR + mu_t * CP_AIR / 0.9
    np.testing.assert_allclose(G[:, 4], k * gT, rtol=1e-14)
    # numpy 向量化版本同一公式
    Gv = viscous_physical_flux(Q[None], np.zeros((1, 3, 3)), gT[None], MU, PR, np.array([mu_t]), 0.9)[0]
    np.testing.assert_allclose(Gv[:, 4], k * gT, rtol=1e-14)


def _channel(kind, order):
    build = build_channel_mesh if kind == "tet" else build_channel_mesh_prism
    mesh = build(order, 3, 3, 2, LX, H, LZ)
    bc = {n: {"type": "SYMMETRY"} for n in ("z_min", "z_max", "wall_bottom", "wall_top", "x_min", "x_max")}
    return mesh, build_face_exact_ghost_provider(mesh, LX, H, LZ, bc), generate_fr_operators(order)


@pytest.mark.parametrize("kind", ["tet", "prism"])
def test_hot_spot_cools_down(kind):
    mesh, prov, ops = _channel(kind, 2)
    X = np.asarray(mesh.sps_coords)
    r2 = ((X - np.array([0.2, 0.05, 0.04])) ** 2).sum(-1)
    T = 288.0 * (1.0 + 0.1 * np.exp(-r2 / 0.03 ** 2))
    U = np.zeros(X.shape[:2] + (5,))
    U[..., 0] = P_INF / (287.0 * T)
    U[..., 4] = P_INF / 0.4
    dUdt = compute_viscous_residual_fr(U, mesh, ops, MU, PR, boundary_ghost_provider=prov)
    assert dUdt[..., 4][r2 < 0.015 ** 2].mean() < 0.0


@pytest.mark.parametrize("kind", ["tet", "prism"])
def test_viscous_operator_has_no_growing_mode(kind):
    mesh, prov, ops = _channel(kind, 1)
    nc, ns = mesh.n_cells, mesh.n_sps_per_cell
    U = np.zeros((nc, ns, 5))
    U[..., 0] = RHO
    U[..., 4] = P_INF / 0.4
    mu_t = np.full((nc, ns), 20.0 * MU)

    def dUdt(u):
        return compute_viscous_residual_fr(u.reshape(nc, ns, 5), mesh, ops, MU, PR, mu_t_field=mu_t,
                                           boundary_ghost_provider=prov).reshape(-1)

    npr, nte = real_sps_per_cell(1)
    nreal = np.where(np.arange(nc) < mesh.n_prism_cells, npr, nte)
    idx = np.array([(c * ns + s) * 5 + v for c in range(nc) for s in range(nreal[c]) for v in range(5)])
    u0 = U.reshape(-1)
    r0 = dUdt(u0)
    A = np.empty((idx.size, idx.size))
    for j, k in enumerate(idx):
        up = u0.copy()
        h = 1e-6 * max(abs(up[k]), 1.0)
        up[k] += h
        A[:, j] = (dUdt(up)[idx] - r0[idx]) / h
    lam = np.linalg.eigvals(A)
    assert lam.real.max() <= 1e-8 * np.abs(lam).max(), (
        f"{kind}: 纯粘性算子有增长模态 max Re = {lam.real.max():.3e}（谱半径 {np.abs(lam).max():.3e}）")
