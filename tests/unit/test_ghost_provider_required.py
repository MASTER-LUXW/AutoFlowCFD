"""边界幽灵态提供者必须显式给出（I-5，2026-10-19）。

此前残差 / 梯度核心函数收到 None 时静默换成零梯度外插（边界条件对残差不可见），网格没有面连接关系时构建提供者
只打警告、返回 None。单 GPU 与多 GPU 都因此出过"全部边界退化成镜像内部值"的真实缺陷。
"""

from types import SimpleNamespace

import numpy as np
import pytest

from autoflowcfd.boundary.fr_ghost_state import DefaultGhostProvider, require_ghost_provider
from autoflowcfd.core.fr_operators.corrected_gradient import source_velocity_gradient
from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
from autoflowcfd.core.fr_residual.inviscid import (
    compute_inviscid_residual_fr, mean_flow_primitives, primitive_to_conserved,
)
from autoflowcfd.core.fr_residual.viscous_flux import compute_viscous_residual_fr
from autoflowcfd.core.fr_solver.boundary.ghost import build_boundary_ghost_provider
from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh


@pytest.fixture(scope="module")
def case():
    mesh = _build_synthetic_mixed_mesh(1)
    Q = np.tile([1.2, 30.0, 1.0, -2.0, 1.0e5], (mesh.n_cells, mesh.n_sps_per_cell, 1))
    return mesh, Q, primitive_to_conserved(Q)


def test_helper_returns_the_provider_and_refuses_none():
    p = DefaultGhostProvider()
    assert require_ghost_provider(p, "x") is p
    with pytest.raises(ValueError, match="必须显式给出边界幽灵态提供者"):
        require_ghost_provider(None, "x")


def test_core_functions_refuse_a_missing_provider(case):
    mesh, Q, U = case
    flat = get_flat_face_geometry(mesh, mesh.operators)
    with pytest.raises(TypeError):                         # 无粘：参数必填
        compute_inviscid_residual_fr(U, mesh, mesh.operators, mach_ref=0.1)
    with pytest.raises(ValueError, match="compute_inviscid_residual_fr"):
        compute_inviscid_residual_fr(U, mesh, mesh.operators, None, mach_ref=0.1)
    with pytest.raises(ValueError, match="compute_viscous_residual_fr"):
        compute_viscous_residual_fr(U, mesh, mesh.operators, 1.8e-5, 0.72)
    with pytest.raises(ValueError, match="mean_flow_primitives"):
        mean_flow_primitives(U, flat, None)
    with pytest.raises(ValueError, match="source_velocity_gradient"):
        source_velocity_gradient(Q, mesh, mesh.operators, flat, None)


def test_explicit_zero_gradient_provider_still_works(case):
    """零梯度外插是合法的显式选择：均匀流的无粘残差为零（自由流保持）。"""
    mesh, _, U = case
    r = compute_inviscid_residual_fr(U, mesh, mesh.operators, DefaultGhostProvider(), mach_ref=0.1)
    assert np.max(np.abs(r)) < 1e-6 * 1.2 * 30.0


def test_building_a_provider_without_faces_is_an_error():
    with pytest.raises(ValueError, match="面连接关系"):
        build_boundary_ghost_provider(SimpleNamespace(face_connectivity=None), None, None, {})
