"""完全分布式加载的求解包对湍流模型名的校验与单机同一规则（`registry.require_supported`，I-213）。

2026-10-22 以前 `build_fully_distributed_rank_package` 要求调用方传规范化的大写名，小写的合法名被当成"无法识别"、
抛 NotImplementedError；单机 FRSolver 与分布式求解器构造都是按大小写不敏感规范化。未知模型名在所有入口都是
ValueError（参数错误），不是"尚未实现"。
"""

import pytest

from autoflowcfd.core.mpi.distributed_mesh_loader import build_fully_distributed_rank_package
from tests.unit._boundary_physics import ghost_provider
from tests.unit._distributed_mesh_loader_common import mesh_and_ops  # noqa: F401  (fixture)
from tests.unit._distributed_run_options import distributed_run_options
from tests.unit._wall_source import synthetic_wall_source

import numpy as np


def _package(mesh, ops, name):
    freestream = {"rho_inf": 1.225, "vel_inf": 33.33, "p_inf": 101325.0}
    return build_fully_distributed_rank_package(
        mesh, ops, mesh.face_connectivity, np.array([0, 0, 0, 0], dtype=np.int32), 0, 1,
        ghost_provider(mesh, freestream, "NONE"), freestream, mu_molecular=1.8e-5, mach_ref=0.2,
        order=mesh.order, enable_viscous=True, wall_distance_source=synthetic_wall_source(mesh),
        **distributed_run_options(turb_model_name=name))


def test_lowercase_model_name_is_normalized_like_single_node(mesh_and_ops):  # noqa: F811
    mesh, ops = mesh_and_ops
    assert _package(mesh, ops, "none")["turb_model_name"] == "NONE"


def test_unknown_model_name_is_a_value_error(mesh_and_ops):  # noqa: F811
    mesh, ops = mesh_and_ops
    with pytest.raises(ValueError, match="未知湍流模型"):
        _package(mesh, ops, "k-epsilon")
