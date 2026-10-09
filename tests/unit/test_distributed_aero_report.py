"""分布式求解收尾的气动系数（2026-10-09，`core/mpi/global_view.py`）。

此前分布式 `solve steady/transient/resume` 不报告气动系数。现在各 rank 的 local 段汇总到 root，与单机同一个
函数：同一网格、同一状态、同一涡粘场下，分布式全局视图给出的系数必须与单机逐位相同。
"""

import numpy as np
import pytest

from autoflowcfd.core.time_integration import TimeIntegrationScheme
from autoflowcfd.postprocess.fr_coefficients import compute_aerodynamic_coefficients_fr
from tests.validation._channel_mesh import build_channel_mesh_prism

_BC = {"x_min": "VELOCITY_INLET", "x_max": "PRESSURE_OUTLET", "wall_bottom": "WALL", "wall_top": "WALL",
       "z_min": "SYMMETRY", "z_max": "SYMMETRY"}


def _mesh():
    mesh = build_channel_mesh_prism(1, 3, 3, 2, 0.4, 0.1, 0.08)
    mesh.boundary_bc_types = dict(_BC)
    return mesh


def _state(shape):
    from autoflowcfd.core.fr_residual.inviscid import primitive_to_conserved

    rng = np.random.default_rng(3)
    Q = np.empty(shape[:2] + (5,))
    Q[..., 0] = 1.225 * (1 + 0.02 * rng.standard_normal(shape[:2]))
    Q[..., 1] = 10.0 * (1 + 0.1 * rng.standard_normal(shape[:2]))
    Q[..., 2] = rng.standard_normal(shape[:2])
    Q[..., 3] = rng.standard_normal(shape[:2])
    Q[..., 4] = 101325.0 * (1 + 0.001 * rng.standard_normal(shape[:2]))
    return np.stack([primitive_to_conserved(q) for q in Q.reshape(-1, 5)]).reshape(shape[:2] + (5,))


@pytest.mark.parametrize("with_eddy_viscosity", [False, True])
def test_distributed_view_gives_the_single_machine_coefficients(with_eddy_viscosity):
    from autoflowcfd.core.fr_solver import FRSolver
    from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver
    from autoflowcfd.core.mpi.global_view import gather_global_view
    from autoflowcfd.fr.operators import generate_fr_operators

    single = FRSolver(mesh=_mesh(), order=1, turb_model_name="NONE", vel_inf=10.0,
                      time_scheme=TimeIntegrationScheme.SSP_RK3)
    dmesh = _mesh()
    dist = DistributedFRSolver(mesh=dmesh, ops=generate_fr_operators(1), face_connectivity=dmesh.face_connectivity,
                               n_ranks=1, order=1, turb_model_name="none", vel_inf=10.0,
                               time_scheme=TimeIntegrationScheme.SSP_RK3)
    U = _state(single.state.U.shape)
    single.state.U = U.copy()
    single.state._update_primitives()
    n_local = dist.partition.n_local_cells
    dist.state.U[:n_local] = U[dist.partition.local_cells]
    if with_eddy_viscosity:
        mu_t = 1e-4 * (1.0 + np.random.default_rng(5).random(U.shape[:2]))
        single._get_turbulent_viscosity_field = lambda: mu_t
        dist._prev_mu_t_local = mu_t[dist.partition.local_cells]

    view = gather_global_view(dist)
    want = compute_aerodynamic_coefficients_fr(single, reference_area=0.01)
    got = compute_aerodynamic_coefficients_fr(view, reference_area=0.01)
    for name in ("Cd", "Cl", "Cs", "Cm", "Cy", "Cr"):
        assert getattr(got, name) == pytest.approx(getattr(want, name), rel=1e-12, abs=1e-15), name
    assert abs(want.Cd) > 0.0
