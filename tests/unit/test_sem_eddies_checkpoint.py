"""合成湍流入口的涡核数随 checkpoint 持久化，续算沿用（I-190）；完全分布式加载的紧凑包携带它并严格读取。

此前 `sem_num_eddies` 不进 checkpoint：LES/WMLES/DDES/IDDES 续算按密度判据重新取涡核数，自定义
`--sem-num-eddies` 的算例续算被静默换参数；分布式求解器本身也不保存这个值。
"""

from types import SimpleNamespace

import click
import numpy as np
import pytest

from autoflowcfd.cli.solve.checkpoint_io.rebuild import resolve_resume_sem_num_eddies
from autoflowcfd.core.freestream_defaults import MU_MOLECULAR, P_INF, RHO_INF
from autoflowcfd.core.time_integration import TimeIntegrationScheme
from autoflowcfd.core.turbulence.registry import uses_synthetic_inlet
from autoflowcfd.core.utils.checkpoint_physics import SEM_AUTO, physics_metadata
from tests.unit._distributed_run_options import distributed_run_options
from tests.validation._channel_mesh import build_channel_mesh_prism

_BC = {"x_min": "VELOCITY_INLET", "x_max": "PRESSURE_OUTLET", "wall_bottom": "WALL", "wall_top": "WALL",
       "z_min": "SYMMETRY", "z_max": "SYMMETRY"}


def _stub(n):
    return SimpleNamespace(freestream={"rho_inf": 1.2, "vel_inf": 10.0, "p_inf": 1e5}, mu_molecular=1.8e-5,
                           _turbulence_intensity=0.01, _viscosity_ratio=5.0, artificial_viscosity_enabled=False,
                           artificial_viscosity_alpha=1.0, wall_treatment="resolved", _sem_num_eddies=n)


def test_metadata_records_the_eddy_count_and_auto():
    assert physics_metadata(_stub(77))["sem_num_eddies"] == 77
    assert physics_metadata(_stub(None))["sem_num_eddies"] == SEM_AUTO


def test_resume_rules():
    assert [uses_synthetic_inlet(m) for m in ("ddes", "IDDES", "les", "wmles", "sst", "sa", None)] == \
        [True, True, True, True, False, False, False]
    assert resolve_resume_sem_num_eddies({"sem_num_eddies": 77}, "ddes") == 77
    assert resolve_resume_sem_num_eddies({"sem_num_eddies": SEM_AUTO}, "les") is None
    assert resolve_resume_sem_num_eddies({}, "sa") is None              # 不用合成湍流入口：不需要记录
    with pytest.raises(click.ClickException, match="sem_num_eddies"):
        resolve_resume_sem_num_eddies({}, "ddes")                       # 此前静默按密度判据重取


def _mesh():
    mesh = build_channel_mesh_prism(1, 3, 3, 2, 0.4, 0.1, 0.08)
    mesh.boundary_bc_types = dict(_BC)
    return mesh


_KW = dict(turb_model_name="LES", time_scheme=TimeIntegrationScheme.DUAL_TIME, vel_inf=10.0, rho_inf=RHO_INF,
           p_inf=P_INF, mu_molecular=MU_MOLECULAR, sem_num_eddies=55)


def test_single_node_checkpoint_records_the_user_eddy_count(tmp_path):
    from autoflowcfd.cli.solve.checkpoint_io.write import write_checkpoint
    from autoflowcfd.core.fr_solver import FRSolver
    from autoflowcfd.core.utils.checkpoint import CheckpointManager

    solver = FRSolver(mesh=_mesh(), order=1, **_KW)
    path = write_checkpoint(solver, str(tmp_path), 0, "m.pkl", order=1, turbulence_model="les", backend="cpu",
                            quiet=True)
    meta = CheckpointManager().load(path)[3]
    assert int(meta["sem_num_eddies"]) == 55
    assert resolve_resume_sem_num_eddies(meta, "les") == 55


def test_distributed_checkpoint_records_the_user_eddy_count(tmp_path):
    from autoflowcfd.core.mpi.distributed_checkpoint import distributed_load_checkpoint, distributed_save_checkpoint
    from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver
    from autoflowcfd.fr.operators import generate_fr_operators

    mesh = _mesh()
    solver = DistributedFRSolver(mesh=mesh, ops=generate_fr_operators(1), face_connectivity=mesh.face_connectivity,
                                 n_ranks=1, order=1, **_KW)
    assert solver._sem_num_eddies == 55
    path = distributed_save_checkpoint(solver, str(tmp_path), 0, "m.pkl", 1, "les", "cpu")
    _U, meta, _it = distributed_load_checkpoint(path, solver)
    assert int(meta["sem_num_eddies"]) == 55


def test_fully_distributed_package_carries_the_eddy_count():
    import types

    from autoflowcfd.core.fr_solver.boundary import build_boundary_ghost_provider
    from autoflowcfd.core.mpi.distributed_mesh_loader import build_fully_distributed_rank_package
    from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver
    from autoflowcfd.fr.operators import generate_fr_operators

    mesh = _mesh()
    ops = generate_fr_operators(1)
    freestream = {"rho_inf": RHO_INF, "vel_inf": 10.0, "p_inf": P_INF}
    stub = types.SimpleNamespace(mesh=mesh, ops=ops, freestream=freestream, turb_model_name="LES", wall_model=None,
                                 _turbulence_intensity=0.01, _viscosity_ratio=5.0, mu_molecular=MU_MOLECULAR,
                                 _sem_num_eddies=55)
    package = build_fully_distributed_rank_package(
        mesh, ops, mesh.face_connectivity, np.zeros(mesh.n_cells, dtype=np.int32), rank=0, n_ranks=1,
        boundary_ghost_provider_global=build_boundary_ghost_provider(stub, bc_overrides={}),
        freestream=freestream, mu_molecular=MU_MOLECULAR, mach_ref=0.05, order=1, enable_viscous=True,
        **distributed_run_options(turb_model_name="LES", sem_num_eddies=55))
    assert package["sem_num_eddies"] == 55
    assert DistributedFRSolver.from_fully_distributed_package(package, n_ranks=1)._sem_num_eddies == 55
    # 打包端恒写的键缺失时报错（此前 .get(键, 缺省) 静默换成缺省值）
    del package["viscosity_ratio"]
    with pytest.raises(KeyError, match="viscosity_ratio"):
        DistributedFRSolver.from_fully_distributed_package(package, n_ranks=1)


def test_fully_distributed_load_builds_the_synthetic_inlet(tmp_path):
    """真实的完全分布式加载（root 用桩对象构造边界幽灵态）：LES + 速度入口构造出合成湍流入口、涡核数为用户值。
    2026-10-20 以前桩上缺 `ops`，这里抛 AttributeError——LES/WMLES/DDES/IDDES 的完全分布式路径跑不起来。"""
    from unittest.mock import patch

    from autoflowcfd.core.mpi.distributed_mesh_loader import distributed_mesh_load_v2

    mesh = _mesh()
    with patch("autoflowcfd.cli.solve.mesh_loader.load_mesh_for_solver", return_value=(mesh, None)):
        package, root_context = distributed_mesh_load_v2(
            "m.pkl", 1, None, 1, freestream={"rho_inf": RHO_INF, "vel_inf": 10.0, "p_inf": P_INF},
            mu_molecular=MU_MOLECULAR, mach_ref=0.05, enable_viscous=True,
            **distributed_run_options(turb_model_name="LES", time_scheme=TimeIntegrationScheme.DUAL_TIME,
                                      sem_num_eddies=55))
    provider = root_context["boundary_ghost_provider_global"]
    sem_states = [c["sem"] for c in provider.code_to_config.values() if c.get("sem") is not None]
    assert len(sem_states) == 1 and sem_states[0].sem.num_eddies == 55
    assert package["sem_num_eddies"] == 55
