"""边界幽灵态提供者按显式物理设置构造（I-106）：此前接收整个求解器对象，完全分布式加载的 root 为此伪造
`SimpleNamespace` 桩（`wall_model=object()` 只为让"是否为 None"走对分支）。"""

import inspect
from types import SimpleNamespace

from autoflowcfd.core.fr_solver.boundary import BoundaryPhysics, boundary_physics, build_boundary_ghost_provider
from tests.unit._module_source import module_source


def test_builder_takes_mesh_ops_and_physics():
    assert list(inspect.signature(build_boundary_ghost_provider).parameters) == [
        "mesh", "ops", "physics", "bc_overrides"]


def test_boundary_physics_reads_the_solver_settings():
    solver = SimpleNamespace(freestream={"rho_inf": 1.0}, mu_molecular=2e-5, turb_model_name="WMLES",
                             wall_model=object(), _turbulence_intensity=0.03, _viscosity_ratio=7.0,
                             _sem_num_eddies=12)
    assert boundary_physics(solver) == BoundaryPhysics({"rho_inf": 1.0}, 2e-5, "WMLES", True, 0.03, 7.0, 12)


def test_fully_distributed_root_builds_no_solver_stub():
    source = module_source("autoflowcfd.core.mpi.distributed_mesh_loader.fully_distributed")
    assert "SimpleNamespace(\n            mesh=mesh" not in source and "root_solver_stub" not in source
