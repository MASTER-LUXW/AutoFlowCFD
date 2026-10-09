"""完全分布式网格加载测试（test_distributed_mesh_loader*.py）共用的夹具与辅助函数。"""

import numpy as np
import pytest

from autoflowcfd.core.mpi.distributed_mesh_loader import build_fully_distributed_rank_package
from autoflowcfd.core.fr_residual.inviscid import primitive_to_conserved
from autoflowcfd.fr.operators import generate_fr_operators

from tests.unit._wall_source import synthetic_wall_source
from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh


class _FakeHaloExchange:
    def __init__(self, U_extended: np.ndarray):
        self._U_extended = U_extended

    def exchange(self, U_local: np.ndarray) -> np.ndarray:
        return self._U_extended


@pytest.fixture(scope="module")
def mesh_and_ops():
    order = 1
    mesh = _build_synthetic_mixed_mesh(order)
    ops = generate_fr_operators(order)
    assert mesh.n_prism_cells == 2
    assert mesh.n_cells == 4
    return mesh, ops


def _nonuniform_U(mesh, rng):
    rho_inf, u_inf, v_inf, w_inf, p_inf = 1.225, 30.0, 5.0, -3.0, 101325.0
    n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
    Q = np.zeros((n_cells, n_sps, 5))
    Q[..., 0] = rho_inf * (1.0 + rng.uniform(-0.02, 0.02, size=(n_cells, n_sps)))
    Q[..., 1] = u_inf + rng.uniform(-3.0, 3.0, size=(n_cells, n_sps))
    Q[..., 2] = v_inf + rng.uniform(-2.0, 2.0, size=(n_cells, n_sps))
    Q[..., 3] = w_inf + rng.uniform(-2.0, 2.0, size=(n_cells, n_sps))
    Q[..., 4] = p_inf * (1.0 + rng.uniform(-0.01, 0.01, size=(n_cells, n_sps)))
    return np.stack(
        [primitive_to_conserved(Q[c, s]) for c in range(n_cells) for s in range(n_sps)]
    ).reshape(n_cells, n_sps, 5)


def _build_all_packages(mesh, ops, n_ranks=2):
    fc = mesh.face_connectivity
    cell_partition = np.array([0, 1, 0, 1], dtype=np.int32)
    freestream = {"rho_inf": 1.225, "vel_inf": 33.33, "p_inf": 101325.0}

    import types
    from autoflowcfd.core.fr_solver.boundary import build_boundary_ghost_provider
    root_stub = types.SimpleNamespace(
        mesh=mesh, freestream=freestream, turb_model_name="NONE", wmles_model=None,
    )
    boundary_ghost_provider_global = build_boundary_ghost_provider(root_stub, bc_overrides={})

    packages = [
        build_fully_distributed_rank_package(
            mesh, ops, fc, cell_partition, r, n_ranks,
            boundary_ghost_provider_global, freestream, mu_molecular=1.8e-5, mach_ref=0.2,
            order=mesh.order, enable_viscous=True,
            wall_distance_source=synthetic_wall_source(mesh),
        )
        for r in range(n_ranks)
    ]
    return packages, boundary_ghost_provider_global, cell_partition
