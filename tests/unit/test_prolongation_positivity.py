"""升阶延拓后在新阶数点集上的正性限制（`_limit_prolongated_state`）。

构造在 P1 点集上可容许（先施加 P1 的正性限制器）的强变化状态，延拓到 P2：
低阶多项式在 P2 的解点 / 面通量点 / 过积分细点上可以出现 rho 或 p 非正（判据
本身先检查确实出现了，否则测试没有意义）；施加后新点集上处处为正、守恒的单元
均值不变。
"""

import numpy as np

from autoflowcfd.core.time_integration.positivity import build_positivity_limiter
from autoflowcfd.fr.operators import generate_fr_operators
from tests.validation._channel_mesh import build_channel_mesh_prism

GAMMA = 1.4


def _point_set_min(lim, U):
    """限制器点集（解点 + 全部通量求值点）上 rho 与 p 的最小值。"""
    n_cells, n_sps = lim.W.shape
    U3 = U.reshape(n_cells, n_sps, -1)[..., :5]
    out = []
    for sel, nr, E in ((lim.cell_is_prism, lim.n_real_prism, lim.E_prism),
                       (~lim.cell_is_prism, lim.n_real_tet, lim.E_tet)):
        if not sel.any():
            continue
        u = U3[sel, :nr]
        P = np.concatenate([u, np.einsum("qs,csv->cqv", E[:, :nr], u)], axis=1)
        rho = P[..., 0]
        p = (GAMMA - 1.0) * (P[..., 4] - 0.5 * (P[..., 1:4] ** 2).sum(-1) / rho)
        out.append((rho.min(), p.min()))
    return min(o[0] for o in out), min(o[1] for o in out)


def _cell_means(lim, U):
    n_cells, n_sps = lim.W.shape
    U3 = U.reshape(n_cells, n_sps, -1)[..., :5]
    return (lim.W[..., None] * U3).sum(1) / lim.W.sum(1)[:, None]


def test_prolongated_state_is_admissible_on_new_point_set():
    from autoflowcfd.core.fr_solver import FRSolver

    mesh = build_channel_mesh_prism(2, 3, 3, 2, 0.4, 0.1, 0.08)
    s = FRSolver(mesh=mesh, order=2, turb_model_name="NONE", n_vars=5, rho_inf=1.225, vel_inf=30.0,
                 p_inf=101325.0, mu_molecular=1.8e-5)
    # 回到 P1：几何、算子、状态
    s.mesh.set_order(1)
    s.ops = generate_fr_operators(1)
    s.current_order = 1
    n_cells, n_sps1 = mesh.n_cells, mesh.n_sps_per_cell
    rng = np.random.default_rng(7)
    rho = 1.225 * np.exp(1.5 * rng.standard_normal((n_cells, n_sps1)))
    vel = 30.0 * rng.standard_normal((n_cells, n_sps1, 3))
    p = 101325.0 * np.exp(1.5 * rng.standard_normal((n_cells, n_sps1)))
    U = np.empty((n_cells, n_sps1, 5))
    U[..., 0] = rho
    U[..., 1:4] = rho[..., None] * vel
    U[..., 4] = p / (GAMMA - 1.0) + 0.5 * rho * (vel ** 2).sum(-1)
    lim1 = build_positivity_limiter(s.mesh, s.ops, order=1)
    lim1(U.reshape(-1, 5))
    assert min(_point_set_min(lim1, U)) > 0.0
    s.state.U = U
    s.state.n_sps = n_sps1
    s.state.Q = np.zeros_like(U)
    s.state._update_primitives()

    s._interpolate_to_new_order(2)
    s.mesh.set_order(2)
    s.ops = generate_fr_operators(2)
    lim2 = build_positivity_limiter(s.mesh, s.ops, order=2)
    before = s.state.U.copy()
    assert min(_point_set_min(lim2, before)) <= 0.0, "构造的状态在 P2 点集上本来就可容许，判据无效"

    s._limit_prolongated_state()
    after = s.state.U
    assert min(_point_set_min(lim2, after)) > 0.0
    np.testing.assert_allclose(_cell_means(lim2, after), _cell_means(lim2, before), rtol=1e-12, atol=1e-9)
