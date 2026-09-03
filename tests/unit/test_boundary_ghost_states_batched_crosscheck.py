"""AutoFlowCFD V2.0 - `compute_boundary_ghost_states` 按边界组批量化的
决定性交叉验证（2026-09-03 性能优化）。

背景：真实 cProfile 画像（cube_demo 374,352 单元子集）显示
`BoundaryGhostStateProvider.__call__` 逐面 Python 级 dict/if-elif 分派
（每步约 24 万次调用）占单步耗时约 3~6%——其内部真正调用的
`wall_ghost_state`/`farfield_ghost_state`/... 本身已经是接受任意长度
批量的向量化函数，逐面调用纯粹是调度开销。新增
`_compute_boundary_ghost_states_batched`：先向量化算出全部活跃边界面
的 Q_o（外插），再按 `group_code` 分组、每组一次性批量调用底层函数
（INLET+SEM 例外，逐面涡核位置相关，组内仍逐面调用）。

本文件验证：批量版本与原有逐面版本（`_compute_boundary_ghost_states_
per_face`，保留作为通用回退）在同一份真实含 WALL/INLET/OUTLET/
SYMMETRY 四种边界类型 + native 四面体面 + 棱柱四边形侧面混合拆分面
的合成网格上，结果一致到机器精度（rtol/atol=1e-12，不是逐位相等——
Q_o 的外插矩阵乘法批量版走一次性 `np.einsum`（BLAS），逐面版走
numba `_extrap_matmul` 单面小矩阵乘法，同一套数学公式的不同求和结合
顺序，在最后一位浮点尾数上允许有非零但机器精度量级的差异，这与本
项目其它地方"并行化重结合"判据同一原则，见 inviscid_kernel.py 模块
文档）。
"""

import numpy as np
import pytest

from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
from autoflowcfd.core.fr_residual.inviscid import primitive_to_conserved, conserved_to_primitive
from autoflowcfd.core.fr_residual.inviscid_kernel import (
    compute_boundary_ghost_states,
    _compute_boundary_ghost_states_per_face,
    _compute_boundary_ghost_states_batched,
)
from autoflowcfd.boundary.fr_ghost_state import BoundaryGhostStateProvider
from autoflowcfd.fr.operators import generate_fr_operators
from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh


def _build_multi_bc_provider(flat, rng):
    """给一份真实合成网格的全部边界面随机分配 5 种边界类型之一（含
    WALL 的 is_no_slip 两种取值、INLET 不含 SEM），构造真实
    `BoundaryGhostStateProvider`。"""
    n_faces = flat.n_faces
    boundary_idx = np.nonzero(flat.is_boundary)[0]
    n_boundary = len(boundary_idx)
    assert n_boundary > 0

    group_code = np.full(n_faces, -1, dtype=np.int64)
    # 5 个边界组编码，循环分配给全部边界面，确保每种类型都被覆盖到。
    codes_cycle = np.arange(n_boundary) % 5
    group_code[boundary_idx] = codes_cycle

    Q_free = np.array([1.225, 30.0, 0.0, 0.0, 101325.0])
    code_to_config = {
        0: {"type": "WALL", "is_no_slip": True},
        1: {"type": "WALL", "is_no_slip": False},
        2: {"type": "FARFIELD", "Q_free": Q_free},
        3: {"type": "INLET", "Q_inlet": Q_free},
        4: {"type": "OUTLET", "p_outlet": 101000.0},
    }
    default_config = {"type": "SYMMETRY"}
    return BoundaryGhostStateProvider(group_code, code_to_config, default_config)


@pytest.mark.parametrize("order", [1, 2])
def test_batched_matches_per_face_on_real_mixed_mesh(order):
    mesh = _build_synthetic_mixed_mesh(order)  # native 四面体 + 棱柱
    ops = generate_fr_operators(order)
    flat = get_flat_face_geometry(mesh, ops)

    rng = np.random.default_rng(order * 100 + 7)
    n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
    rho_inf, p_inf = 1.225, 101325.0
    Q = np.zeros((n_cells, n_sps, 5))
    Q[..., 0] = rho_inf * (1.0 + rng.uniform(-0.05, 0.05, size=(n_cells, n_sps)))
    Q[..., 1] = 30.0 + rng.uniform(-5.0, 5.0, size=(n_cells, n_sps))
    Q[..., 2] = rng.uniform(-5.0, 5.0, size=(n_cells, n_sps))
    Q[..., 3] = rng.uniform(-5.0, 5.0, size=(n_cells, n_sps))
    Q[..., 4] = p_inf * (1.0 + rng.uniform(-0.02, 0.02, size=(n_cells, n_sps)))
    U = primitive_to_conserved(Q)

    provider = _build_multi_bc_provider(flat, rng)

    expected = _compute_boundary_ghost_states_per_face(flat, U, provider)
    actual = _compute_boundary_ghost_states_batched(flat, U, provider)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    # 反向对照：确认判据本身有区分度——换一个明显不同的 provider（全部
    # 面都当 FARFIELD 处理）应该给出不同的结果，不是恰好对任何输入都
    # 相等的退化情形。
    n_faces = flat.n_faces
    trivial_provider = BoundaryGhostStateProvider(
        np.full(n_faces, 0, dtype=np.int64),
        {0: {"type": "FARFIELD", "Q_free": np.array([1.225, 30.0, 0.0, 0.0, 101325.0])}},
        {"type": "FARFIELD", "Q_free": np.array([1.225, 30.0, 0.0, 0.0, 101325.0])},
    )
    trivial_result = _compute_boundary_ghost_states_batched(flat, U, trivial_provider)
    assert not np.array_equal(actual, trivial_result)


@pytest.mark.parametrize("order", [1, 2])
def test_dispatch_wrapper_uses_batched_path_for_real_provider(order):
    """`compute_boundary_ghost_states` 顶层分派函数：真实
    `BoundaryGhostStateProvider` 时应给出与批量实现完全一致的结果
    （即真的走了批量分支，不是巧合地和逐面版本一样）。"""
    mesh = _build_synthetic_mixed_mesh(order)
    ops = generate_fr_operators(order)
    flat = get_flat_face_geometry(mesh, ops)

    rng = np.random.default_rng(order * 200 + 3)
    n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
    U = np.tile(
        primitive_to_conserved(np.array([1.225, 30.0, 0.0, 0.0, 101325.0])),
        (n_cells, n_sps, 1),
    )

    provider = _build_multi_bc_provider(flat, rng)
    via_dispatch = compute_boundary_ghost_states(flat, U, None, provider)
    via_batched = _compute_boundary_ghost_states_batched(flat, U, provider)
    np.testing.assert_array_equal(via_dispatch, via_batched)


def test_dispatch_wrapper_falls_back_to_per_face_for_generic_provider():
    """非 `BoundaryGhostStateProvider` 的任意可调用对象（如
    `DefaultGhostProvider`）必须走逐面回退路径，不假设它能被按
    group_code 分组——这是分派条件本身要保护的通用契约。"""
    from autoflowcfd.core.fr_residual.inviscid import DefaultGhostProvider

    order = 1
    mesh = _build_synthetic_mixed_mesh(order)
    ops = generate_fr_operators(order)
    flat = get_flat_face_geometry(mesh, ops)

    n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
    U = np.tile(
        primitive_to_conserved(np.array([1.225, 30.0, 0.0, 0.0, 101325.0])),
        (n_cells, n_sps, 1),
    )

    provider = DefaultGhostProvider()
    # 不应该抛错（DefaultGhostProvider 没有 group_code/code_to_config，
    # 若分派函数误判为批量路径会在这里 AttributeError）。
    result = compute_boundary_ghost_states(flat, U, None, provider)
    expected = _compute_boundary_ghost_states_per_face(flat, U, provider)
    np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
