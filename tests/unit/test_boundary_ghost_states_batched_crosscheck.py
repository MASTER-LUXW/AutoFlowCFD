"""边界幽灵态批量接口的交叉验证。

`BoundaryGhostStateProvider.ghost_states` 按边界组批量构造幽灵态；基类 `GhostStateProvider.ghost_states`
是逐面调用 `__call__` 的参照实现。两者在同一份含 WALL（无滑移/滑移）/FARFIELD/INLET/OUTLET/SYMMETRY、
原生四面体面与棱柱四边形混合拆分面的合成网格上必须逐位相等（底层是同一批幽灵态函数，只是调用分组不同）。
残差组装入口 `compute_boundary_ghost_states` 对任何 provider 都只走 `ghost_states` 这一条路径。
"""

import numpy as np
import pytest

from autoflowcfd.boundary.fr_ghost_state import (
    BoundaryGhostStateProvider, DefaultGhostProvider, GhostStateProvider,
)
from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
from autoflowcfd.core.fr_residual.inviscid_kernel import compute_boundary_ghost_states
from autoflowcfd.fr.operators import generate_fr_operators
from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh

Q_FREE = np.array([1.225, 30.0, 0.0, 0.0, 101325.0])


def _build_multi_bc_provider(flat, rng=None):
    """全部边界面循环分配 5 个边界组（每种类型都覆盖到），未分组面走默认 SYMMETRY。"""
    boundary_idx = np.flatnonzero(flat.is_boundary)
    assert boundary_idx.size > 0
    group_code = np.full(flat.n_faces, -1, dtype=np.int64)
    group_code[boundary_idx] = np.arange(boundary_idx.size) % 5
    code_to_config = {
        0: {"type": "WALL", "is_no_slip": True},
        1: {"type": "WALL", "is_no_slip": False},
        2: {"type": "FARFIELD", "Q_free": Q_FREE},
        3: {"type": "INLET", "Q_inlet": Q_FREE},
        4: {"type": "OUTLET", "p_outlet": 101000.0},
    }
    return BoundaryGhostStateProvider(group_code, code_to_config)


def _random_primitives(mesh, seed):
    rng = np.random.default_rng(seed)
    shape = (mesh.n_cells, mesh.n_sps_per_cell)
    Q = np.zeros(shape + (5,))
    Q[..., 0] = 1.225 * (1.0 + rng.uniform(-0.05, 0.05, size=shape))
    Q[..., 1] = 30.0 + rng.uniform(-5.0, 5.0, size=shape)
    Q[..., 2] = rng.uniform(-5.0, 5.0, size=shape)
    Q[..., 3] = rng.uniform(-5.0, 5.0, size=shape)
    Q[..., 4] = 101325.0 * (1.0 + rng.uniform(-0.02, 0.02, size=shape))
    return Q


def _active_boundary_inputs(flat, Q):
    faces = np.flatnonzero(flat.is_boundary & (flat.owner_is_primary | flat.mixed_bnd_face))
    E_o = flat.boundary_extrap_native[flat.owner_face_op[faces]]
    Q_o = np.einsum('fps,fsv->fpv', E_o, Q[flat.owner_cell[faces]])
    return faces, Q_o, flat.true_normal[faces]


@pytest.mark.parametrize("order", [1, 2])
def test_batched_matches_per_face_reference(order):
    mesh = _build_synthetic_mixed_mesh(order)
    flat = get_flat_face_geometry(mesh, generate_fr_operators(order))
    Q = _random_primitives(mesh, order * 100 + 7)
    provider = _build_multi_bc_provider(flat)
    faces, Q_o, normals = _active_boundary_inputs(flat, Q)

    batched = provider.ghost_states(faces, Q_o, normals)
    per_face = GhostStateProvider.ghost_states(provider, faces, Q_o, normals)
    np.testing.assert_array_equal(batched, per_face)

    # 判据有区分度：全部当 FARFIELD 给出不同结果
    farfield_only = BoundaryGhostStateProvider(
        np.zeros(flat.n_faces, dtype=np.int64), {0: {"type": "FARFIELD", "Q_free": Q_FREE}})
    assert not np.array_equal(batched, farfield_only.ghost_states(faces, Q_o, normals))


@pytest.mark.parametrize("order", [1, 2])
def test_assembly_entry_uses_provider_batch(order):
    """残差组装入口给出的幽灵态与直接调用 provider 批量接口逐位一致，非活跃面行为零。"""
    mesh = _build_synthetic_mixed_mesh(order)
    flat = get_flat_face_geometry(mesh, generate_fr_operators(order))
    Q = _random_primitives(mesh, order * 200 + 3)
    provider = _build_multi_bc_provider(flat)
    faces, Q_o, normals = _active_boundary_inputs(flat, Q)

    Q_ghost = compute_boundary_ghost_states(flat, Q, provider)
    np.testing.assert_array_equal(Q_ghost[faces], provider.ghost_states(faces, Q_o, normals))
    inactive = np.setdiff1d(np.arange(flat.n_faces), faces)
    assert not Q_ghost[inactive].any()


def test_default_provider_is_owner_extrapolation():
    order = 1
    mesh = _build_synthetic_mixed_mesh(order)
    flat = get_flat_face_geometry(mesh, generate_fr_operators(order))
    Q = _random_primitives(mesh, 11)
    faces, Q_o, _ = _active_boundary_inputs(flat, Q)
    Q_ghost = compute_boundary_ghost_states(flat, Q, DefaultGhostProvider())
    np.testing.assert_array_equal(Q_ghost[faces], Q_o)


def test_single_face_call_matches_batch():
    """单面接口 `__call__` 是批量接口的特例。"""
    order = 1
    mesh = _build_synthetic_mixed_mesh(order)
    flat = get_flat_face_geometry(mesh, generate_fr_operators(order))
    Q = _random_primitives(mesh, 5)
    provider = _build_multi_bc_provider(flat)
    faces, Q_o, normals = _active_boundary_inputs(flat, Q)
    batched = provider.ghost_states(faces, Q_o, normals)
    for k in range(min(len(faces), 12)):
        np.testing.assert_array_equal(provider(int(faces[k]), Q_o[k], normals[k]), batched[k])
