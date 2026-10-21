"""分布式求解器两条构造路径（传统模式 `__init__` 与完全分布式 `from_fully_distributed_package`）共用与网格来源无关的
装配（I-110，`core/mpi/distributed_solver/assembly.py`）：两条路径得到同一组属性、同一组设置值，只在几何来源相关的
少数属性上不同。此前两条路径各写一遍几十个属性，多次出现"某属性只在一条路径设置"。"""

import numpy as np
import pytest

from autoflowcfd.core.mpi.distributed_mesh_loader import build_fully_distributed_rank_package
from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver
from autoflowcfd.core.time_integration.base import TimeIntegrationScheme
from autoflowcfd.fr.operators import generate_fr_operators
from tests.unit._boundary_physics import ghost_provider
from tests.unit._distributed_run_options import distributed_run_options
from tests.unit._wall_source import synthetic_wall_source
from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh

_FREESTREAM = {"rho_inf": 1.225, "vel_inf": 33.33, "p_inf": 101325.0, "aoa_deg": 4.0, "aos_deg": -1.5}
#: 只属于一条路径的属性：传统模式的分区与全局 DES 网格尺度 / 壁距来源；完全分布式的 root 上下文与随包下发的着色
_TRADITIONAL_ONLY = {"_oc_cell_partition", "_wall_distance_source", "_iddes_h_max", "_iddes_h_wn"}
_FULLY_ONLY = {"_root_context", "_block_jacobi_colors_local", "_coupling_colors_local"}


@pytest.mark.parametrize("model", ["NONE", "SST", "IDDES"])
def test_both_constructors_assemble_the_same_solver(model):
    from autoflowcfd.core.fr_solver.mach_ref import resolve_mach_ref
    from autoflowcfd.core.turbulence.des import compute_h_max_and_h_wn

    mesh = _build_synthetic_mixed_mesh(1)
    ops = generate_fr_operators(1)
    options = distributed_run_options(turb_model_name=model, turbulence_intensity=0.03, viscosity_ratio=6.0,
                                      cfl_start=0.02, cfl_max=0.08, cfl_min=0.01, low_mach_precond=False)
    traditional = DistributedFRSolver(
        mesh=mesh, ops=ops, face_connectivity=mesh.face_connectivity, n_ranks=1, rank=0, order=1,
        mu_molecular=1.8e-5, wall_distance_source=synthetic_wall_source(mesh),
        **{k: v for k, v in _FREESTREAM.items()}, **options)
    mach_ref = resolve_mach_ref(_FREESTREAM["rho_inf"], _FREESTREAM["vel_inf"], _FREESTREAM["p_inf"])
    h_max, h_wn = compute_h_max_and_h_wn(mesh) if model == "IDDES" else (None, None)
    package = build_fully_distributed_rank_package(
        mesh, ops, mesh.face_connectivity, np.zeros(mesh.n_cells, dtype=np.int32), 0, 1,
        ghost_provider(mesh, _FREESTREAM, model), _FREESTREAM, 1.8e-5, mach_ref, 1, True,
        wall_distance_source=synthetic_wall_source(mesh), h_max_global=h_max, h_wn_global=h_wn, **options)
    fully = DistributedFRSolver.from_fully_distributed_package(package, n_ranks=1, rank=0)

    a, b = set(vars(traditional)), set(vars(fully))
    assert a - b <= _TRADITIONAL_ONLY, a - b
    assert b - a <= _FULLY_ONLY, b - a
    for name in ("turb_model_name", "freestream", "mu_molecular", "_turbulence_intensity", "_viscosity_ratio",
                 "_sem_num_eddies", "wall_treatment", "artificial_viscosity_enabled", "order", "current_order",
                 "low_mach_precond_enabled"):
        assert getattr(traditional, name) == getattr(fully, name), name
    assert type(traditional.turb_model) is type(fully.turb_model)
    assert type(traditional.ddes_model) is type(fully.ddes_model)
    np.testing.assert_array_equal(traditional.state.U, fully.state.U)
    assert traditional.time_integrator.scheme is fully.time_integrator.scheme is TimeIntegrationScheme.SSP_RK3
