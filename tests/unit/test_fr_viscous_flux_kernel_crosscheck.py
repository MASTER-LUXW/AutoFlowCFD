"""粘性残差新旧界面项实现逐位对比 (性能优化验证阶段一)。

与 test_fr_residual_inviscid_kernel_crosscheck.py 同一个验证思路（先
用新旧实现在同一个网格、同一个状态场上分别算一遍完整粘性残差，逐位
对比，再接入生产代码）。均匀常数流场（BR1 对常数场应恒为零梯度、恒为
零跳跃）判据用相对 p_inf 的量，理由与无粘残差的自由流场判据完全一致：
`jump = G_tilde_common - G_tilde_own` 是灾难性抵消结果，对浮点运算
结合顺序（numpy einsum/BLAS vs numba 编译后的逐点标量三项和）天然敏感，
不代表逻辑不一致（已用无粘残差交叉验证定位、确认过这个机制，见该文件
判据部分的详细文档）。
"""

import numba
import numpy as np
import pytest

from autoflowcfd.core.fr_residual.inviscid import (
    conserved_to_primitive, primitive_to_conserved, DefaultGhostProvider,
)
from autoflowcfd.core.fr_residual.viscous_flux import compute_viscous_residual_fr
from autoflowcfd.core.fr_operators.gradients import compute_physical_gradient
from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
from autoflowcfd.core.fr_residual.inviscid_kernel import compute_boundary_ghost_states
from autoflowcfd.core.fr_residual.viscous_flux_kernel import compute_viscous_interface_correction_kernel
from autoflowcfd.core.fr_operators.flux_kernels import viscous_physical_flux_batch
from autoflowcfd.core.fr_operators.volume_contract import contract_shared_operator_2axis
from autoflowcfd.core.fr_operators.troubled_cell import suppress_residual_outliers

from .test_fr_residual_inviscid import _build_synthetic_mixed_mesh

MU = 1.8e-5
PR = 0.72
PR_T = 0.9


def _compute_temperature(Q):
    R_AIR = 287.0
    rho = np.maximum(Q[..., 0], 1e-10)
    return Q[..., 4] / (rho * R_AIR)


def _compute_residual_via_new_kernel(U, mesh, ops, mu_t_field=None, boundary_ghost_provider=None):
    """与 fr_viscous_flux.py::compute_viscous_residual_fr 完全对应的
    "新版"：体积项逐字复制（未改动），界面项换成新 kernel。"""
    ghost_provider = boundary_ghost_provider if boundary_ghost_provider is not None else DefaultGhostProvider()
    n_cells = mesh.n_cells
    n_sps = mesh.n_sps_per_cell
    n_prism = mesh.n_prism_cells
    mu_t_field = np.zeros((n_cells, n_sps)) if mu_t_field is None else mu_t_field

    Q = conserved_to_primitive(U[..., :5])
    T = _compute_temperature(Q)

    grad_Q = compute_physical_gradient(Q, mesh, ops)
    grad_vel = grad_Q[:, :, 1:4, :]
    grad_T = compute_physical_gradient(T[:, :, None], mesh, ops)[:, :, 0, :]

    det_jacs = mesh.jacobians["det_jacs"].reshape(n_cells, n_sps)
    inv_jacs = mesh.jacobians["inv_jacs"].reshape(n_cells, n_sps, 3, 3)
    adj_j = det_jacs[..., None, None] * inv_jacs

    # --- 体积项（与 fr_viscous_flux.py 当前实现逐字一致，性能优化后已改用
    # viscous_physical_flux_batch/matmul/tensordot，理由见该文件与
    # fr_volume_contract.py 模块文档；同步更新原因见
    # test_fr_residual_inviscid_kernel_crosscheck.py 同类改动的注释）---
    Q_flat = np.ascontiguousarray(Q.reshape(-1, 5))
    grad_vel_flat = np.ascontiguousarray(grad_vel.reshape(-1, 3, 3))
    grad_T_flat = np.ascontiguousarray(grad_T.reshape(-1, 3))
    mu_t_flat = np.ascontiguousarray(mu_t_field.reshape(-1))
    G_phys = viscous_physical_flux_batch(
        Q_flat, grad_vel_flat, grad_T_flat, MU, PR, mu_t_flat, PR_T
    ).reshape(n_cells, n_sps, 3, 5)
    G_tilde = np.matmul(adj_j, G_phys)
    div_comp = np.zeros((n_cells, n_sps, 5))
    if n_prism > 0:
        div_comp[:n_prism] = contract_shared_operator_2axis(ops.D_3d_prism, G_tilde[:n_prism])
    if n_cells > n_prism:
        div_comp[n_prism:] = contract_shared_operator_2axis(ops.D_3d_tet, G_tilde[n_prism:])
    residual = div_comp / det_jacs[..., None]

    # --- 界面项：新 kernel ---
    flat = get_flat_face_geometry(mesh, ops)
    Q_ghost = compute_boundary_ghost_states(flat, Q, adj_j, ghost_provider)
    n_threads = numba.get_num_threads()
    correction = compute_viscous_interface_correction_kernel(
        Q, grad_vel, grad_T, mu_t_field,
        adj_j, det_jacs, MU, PR, PR_T,
        flat.owner_cell, flat.neighbor_cell, flat.is_boundary,
        flat.owner_axis, flat.owner_side, flat.neighbor_axis, flat.neighbor_side,
        flat.owner_is_primary, flat.neighbor_is_primary,
        flat.neighbor_src0_cell, flat.neighbor_src0_mat,
        flat.neighbor_src1_idx, flat.neighbor_src1_cell, flat.neighbor_src1_mat,
        flat.owner_src0_cell, flat.owner_src0_mat,
        flat.owner_src1_idx, flat.owner_src1_cell, flat.owner_src1_mat,
        flat.boundary_extrap, flat.g_left, flat.g_right, Q_ghost,
        flat.dist_fp_of_sp, flat.dist_axis_coord_of_sp,
        n_prism, n_threads,
    )
    residual = residual + correction
    return suppress_residual_outliers(residual, U[..., :5])


@pytest.mark.parametrize("order,rel_tol", [(1, 1e-9), (2, 1e-7), (3, 1e-3)])
def test_new_kernel_matches_old_loop_uniform_flow(order, rel_tol):
    mesh = _build_synthetic_mixed_mesh(order)
    rho_inf, u_inf, v_inf, w_inf, p_inf = 1.225, 30.0, 5.0, -3.0, 101325.0
    Q_inf = np.array([rho_inf, u_inf, v_inf, w_inf, p_inf])
    U_inf = primitive_to_conserved(Q_inf)
    U = np.tile(U_inf, (mesh.n_cells, mesh.n_sps_per_cell, 1))

    old_residual = compute_viscous_residual_fr(U, mesh, mesh.operators, MU, PR)
    new_residual = _compute_residual_via_new_kernel(U, mesh, mesh.operators)

    max_diff = np.max(np.abs(old_residual - new_residual))
    rel_diff = max_diff / p_inf
    assert rel_diff < rel_tol, f"P={order}: max|old-new|={max_diff:.3e}, rel={rel_diff:.3e}"


@pytest.mark.parametrize("order", [1, 2])
def test_new_kernel_matches_old_loop_nonuniform_perturbed_flow(order):
    """非均匀扰动流场（含湍流涡粘场），覆盖 sources 求和、gv/gT 平均、
    mu_t 耦合路径。P=3 排除，理由同无粘残差交叉验证（该合成网格在 P=3
    下即使很小的扰动也会让未改动的旧实现自身给出病态量级的结果）。"""
    mesh = _build_synthetic_mixed_mesh(order)
    rng = np.random.default_rng(order * 2000 + 3)

    rho_inf, u_inf, v_inf, w_inf, p_inf = 1.225, 30.0, 5.0, -3.0, 101325.0
    Q_inf = np.array([rho_inf, u_inf, v_inf, w_inf, p_inf])
    U_inf = primitive_to_conserved(Q_inf)
    U = np.tile(U_inf, (mesh.n_cells, mesh.n_sps_per_cell, 1))

    n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
    Q = conserved_to_primitive(U)
    Q[..., 0] *= 1.0 + rng.uniform(-0.05, 0.05, size=(n_cells, n_sps))
    Q[..., 1] += rng.uniform(-5.0, 5.0, size=(n_cells, n_sps))
    Q[..., 2] += rng.uniform(-5.0, 5.0, size=(n_cells, n_sps))
    Q[..., 3] += rng.uniform(-5.0, 5.0, size=(n_cells, n_sps))
    Q[..., 4] *= 1.0 + rng.uniform(-0.05, 0.05, size=(n_cells, n_sps))
    U = primitive_to_conserved(Q)
    mu_t_field = rng.uniform(0.0, 5e-3, size=(n_cells, n_sps))

    old_residual = compute_viscous_residual_fr(U, mesh, mesh.operators, MU, PR, mu_t_field=mu_t_field)
    new_residual = _compute_residual_via_new_kernel(U, mesh, mesh.operators, mu_t_field=mu_t_field)

    max_diff = np.max(np.abs(old_residual - new_residual))
    scale = max(np.max(np.abs(old_residual)), 1.0)
    assert max_diff < max(1e-9, scale * 1e-10), f"P={order}: max|old-new|={max_diff:.3e}, scale={scale:.3e}"


# --- 多核并行 (阶段二) 验证：nt=1 vs nt=16，理由/判据哲学同
# test_fr_residual_inviscid_kernel_crosscheck.py 同名部分 ---
def _run_with_threads(order, nonuniform, n_threads):
    numba.set_num_threads(n_threads)
    mesh = _build_synthetic_mixed_mesh(order)
    rho_inf, u_inf, v_inf, w_inf, p_inf = 1.225, 30.0, 5.0, -3.0, 101325.0
    Q_inf = np.array([rho_inf, u_inf, v_inf, w_inf, p_inf])
    U_inf = primitive_to_conserved(Q_inf)
    U = np.tile(U_inf, (mesh.n_cells, mesh.n_sps_per_cell, 1))
    mu_t_field = None
    if nonuniform:
        rng = np.random.default_rng(order * 4000 + 13)
        n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
        Q = conserved_to_primitive(U)
        Q[..., 0] *= 1.0 + rng.uniform(-0.05, 0.05, size=(n_cells, n_sps))
        Q[..., 1] += rng.uniform(-5.0, 5.0, size=(n_cells, n_sps))
        Q[..., 2] += rng.uniform(-5.0, 5.0, size=(n_cells, n_sps))
        Q[..., 3] += rng.uniform(-5.0, 5.0, size=(n_cells, n_sps))
        Q[..., 4] *= 1.0 + rng.uniform(-0.05, 0.05, size=(n_cells, n_sps))
        U = primitive_to_conserved(Q)
        mu_t_field = rng.uniform(0.0, 5e-3, size=(n_cells, n_sps))
    return _compute_residual_via_new_kernel(U, mesh, mesh.operators, mu_t_field=mu_t_field)


@pytest.mark.parametrize("order,rel_tol", [(1, 1e-9), (2, 1e-7), (3, 1e-3)])
def test_parallel_nt1_matches_nt16_uniform_flow(order, rel_tol):
    r1 = _run_with_threads(order, nonuniform=False, n_threads=1)
    r16 = _run_with_threads(order, nonuniform=False, n_threads=16)
    max_diff = np.max(np.abs(r1 - r16))
    rel_diff = max_diff / 101325.0
    assert rel_diff < rel_tol, f"P={order}: max|nt1-nt16|={max_diff:.3e}, rel={rel_diff:.3e}"


@pytest.mark.parametrize("order", [1, 2])
def test_parallel_nt1_matches_nt16_nonuniform_flow(order):
    r1 = _run_with_threads(order, nonuniform=True, n_threads=1)
    r16 = _run_with_threads(order, nonuniform=True, n_threads=16)
    max_diff = np.max(np.abs(r1 - r16))
    scale = max(np.max(np.abs(r1)), 1.0)
    assert max_diff < max(1e-9, scale * 1e-10), f"P={order}: max|nt1-nt16|={max_diff:.3e}, scale={scale:.3e}"


def test_parallel_degenerate_cell_no_blowup():
    """退化/近共面单元场景，理由/构造方式同
    test_fr_residual_inviscid_kernel_crosscheck.py 同名测试。"""
    from types import SimpleNamespace
    from autoflowcfd.grid.high_order.high_order_mesh import HighOrderMesh
    from .test_fr_residual_inviscid import _MockNodes, _MockCells

    nodes = np.array(
        [
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1.0 / 3, 1.0 / 3, 1e-6],
            [10, 0, 0], [11, 0, 0], [10, 1, 0], [10, 0, 1], [11, 0, 1], [10, 1, 1],
        ],
        dtype=float,
    )
    tet_conn = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int32)
    nodes = np.vstack([nodes, [[9, -1, 0], [9, -1, 1]]])
    prism_conn = np.array([[5, 6, 7, 8, 9, 10], [5, 7, 11, 8, 10, 12]], dtype=np.int32)
    mock_volume = SimpleNamespace(
        cell_count=len(tet_conn) + len(prism_conn),
        nodes=_MockNodes(nodes), cells=_MockCells(tet_conn), prism_cells=_MockCells(prism_conn),
    )
    mesh = HighOrderMesh(order=2)
    mesh.load_from_volume_mesh(mock_volume)

    rho_inf, u_inf, v_inf, w_inf, p_inf = 1.225, 30.0, 5.0, -3.0, 101325.0
    Q_inf = np.array([rho_inf, u_inf, v_inf, w_inf, p_inf])
    U_inf = primitive_to_conserved(Q_inf)
    U = np.tile(U_inf, (mesh.n_cells, mesh.n_sps_per_cell, 1))

    numba.set_num_threads(1)
    r1 = _compute_residual_via_new_kernel(U, mesh, mesh.operators)
    numba.set_num_threads(16)
    r16 = _compute_residual_via_new_kernel(U, mesh, mesh.operators)

    assert np.all(np.isfinite(r1)) and np.all(np.isfinite(r16)), "退化单元场景不应产生 NaN/Inf"
    max_diff = np.max(np.abs(r1 - r16))
    max_val = max(np.max(np.abs(r1)), 1.0)
    rel_diff = max_diff / max_val
    assert rel_diff < 1e-10, f"退化单元场景 nt1 vs nt16 相对差异异常放大: rel={rel_diff:.3e}, max_val={max_val:.3e}"
