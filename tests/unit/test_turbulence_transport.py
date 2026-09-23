"""Unit tests for core/turbulence/transport.py's WALL k=0 Dirichlet fix.

`extrapolate_scalar_to_faces_kernel` previously always used a Neumann
(ghost=owner) default at every boundary face, including WALL, for k and
omega alike - a known, documented approximation (see the kernel's own
docstring history). k is exactly zero at a no-slip wall (a standard SST/
k-omega boundary condition, not an approximation), so for k specifically a
Dirichlet-zero ghost (ghost = -owner, mirroring the mean-flow no-slip wall
ghost state formula) is now applied wherever `_compute_wall_dirichlet_
face_mask` identifies a face as WALL-typed via the solver's
`boundary_ghost_provider`. omega is left on the Neumann default (its
analytic near-wall value needs additional wall-distance data, tracked as
separate future work).

These tests pin the kernel-level Dirichlet-zero mirroring in isolation and
the mask builder's face classification / defensive fallback.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from autoflowcfd.core.turbulence.transport_kernel import extrapolate_scalar_to_faces_kernel
from autoflowcfd.core.turbulence.transport import _compute_wall_dirichlet_face_mask


class TestExtrapolateScalarToFacesKernelWallDirichlet:
    def test_wall_face_mirrors_to_zero_non_wall_stays_neumann(self):
        """Two boundary faces on the same single cell (value 5.0): face 0 is
        flagged WALL-Dirichlet-zero, face 1 is not. Expect ghost = -owner for
        face 0 (enforces phi=0 at the wall) and ghost = owner for face 1
        (unchanged Neumann default)."""
        n_faces, n_fp, n_sps = 2, 1, 1
        scalar_sps = np.array([[5.0]])

        # 原生自身面外插表按 `code - 6` 索引（四面体 [6,10)、棱柱
        # [10,15) 同一张表，长 9）；这里取 code=6 并置成恒等。
        boundary_extrap_native = np.zeros((9, n_fp, n_sps))
        boundary_extrap_native[0] = np.array([[1.0]])

        owner_cell = np.array([0, 0], dtype=np.int64)

        # Both faces are boundary faces: no real neighbor source.
        neighbor_src0_cell = np.array([-1, -1], dtype=np.int64)
        neighbor_src0_mat = np.zeros((n_faces, n_fp, n_sps))
        neighbor_src1_idx = np.array([-1, -1], dtype=np.int64)
        neighbor_src1_cell = np.empty((0,), dtype=np.int64)
        neighbor_src1_mat = np.empty((0, n_fp, n_sps))

        wall_dirichlet_zero_face = np.array([True, False])
        # 无混合拆分面（B-8）：partner 全 -1、掩码全 False，混合覆盖循环不生效。
        mixed_nb_partner = np.full(n_faces, -1, dtype=np.int64)
        mixed_nb_mask = np.zeros((n_faces, n_fp), dtype=np.bool_)
        # 无非零 Dirichlet 目标值（本测试只覆盖 k 的 Dirichlet-zero 分支）。
        has_wall_dirichlet_value = np.zeros(n_faces, dtype=np.bool_)
        wall_dirichlet_value_face = np.zeros((n_faces, n_fp), dtype=np.float64)
        owner_cube_face = np.full(n_faces, 6, dtype=np.int64)

        phi_owner, phi_neighbor = extrapolate_scalar_to_faces_kernel(
            scalar_sps,
            neighbor_src0_cell, neighbor_src0_mat,
            neighbor_src1_idx, neighbor_src1_cell, neighbor_src1_mat,
            owner_cell, n_faces, n_fp, n_sps,
            wall_dirichlet_zero_face,
            mixed_nb_partner, mixed_nb_mask,
            has_wall_dirichlet_value, wall_dirichlet_value_face,
            owner_cube_face, boundary_extrap_native,
        )

        np.testing.assert_allclose(phi_owner, [[5.0], [5.0]])
        np.testing.assert_allclose(phi_neighbor[0], [-5.0])  # WALL: Dirichlet-zero mirror
        np.testing.assert_allclose(phi_neighbor[1], [5.0])   # non-WALL: unchanged Neumann

    def test_all_false_mask_reproduces_old_neumann_only_behavior(self):
        """A mask of all False must reproduce the pre-fix behavior exactly
        (ghost = owner at every boundary face) - guards against the new
        parameter silently changing existing (non-WALL) callers."""
        n_faces, n_fp, n_sps = 1, 1, 1
        scalar_sps = np.array([[3.0]])
        # 原生自身面外插表：编码 [6,10) 四面体、[10,15) 棱柱，统一按
        # `code - 6` 索引，所以表长 9。这里用 code=6（四面体 v0 面）。
        boundary_extrap_native = np.zeros((9, n_fp, n_sps))
        boundary_extrap_native[0] = np.array([[1.0]])
        owner_cell = np.array([0], dtype=np.int64)
        neighbor_src0_cell = np.array([-1], dtype=np.int64)
        neighbor_src0_mat = np.zeros((n_faces, n_fp, n_sps))
        neighbor_src1_idx = np.array([-1], dtype=np.int64)
        neighbor_src1_cell = np.empty((0,), dtype=np.int64)
        neighbor_src1_mat = np.empty((0, n_fp, n_sps))
        wall_dirichlet_zero_face = np.array([False])
        mixed_nb_partner = np.full(n_faces, -1, dtype=np.int64)
        mixed_nb_mask = np.zeros((n_faces, n_fp), dtype=np.bool_)
        has_wall_dirichlet_value = np.zeros(n_faces, dtype=np.bool_)
        wall_dirichlet_value_face = np.zeros((n_faces, n_fp), dtype=np.float64)
        owner_cube_face = np.full(n_faces, 6, dtype=np.int64)

        _, phi_neighbor = extrapolate_scalar_to_faces_kernel(
            scalar_sps,
            neighbor_src0_cell, neighbor_src0_mat,
            neighbor_src1_idx, neighbor_src1_cell, neighbor_src1_mat,
            owner_cell, n_faces, n_fp, n_sps,
            wall_dirichlet_zero_face,
            mixed_nb_partner, mixed_nb_mask,
            has_wall_dirichlet_value, wall_dirichlet_value_face,
            owner_cube_face, boundary_extrap_native,
        )
        np.testing.assert_allclose(phi_neighbor, [[3.0]])

    def test_wall_dirichlet_value_face_mirrors_to_target(self):
        """omega 解析壁面值分支：WALL 面标记 has_wall_dirichlet_value=True
        且给定 target=12.0 时，ghost 应满足 (ghost+owner)/2 == target
        （即 ghost = 2*target - owner），而不是 Dirichlet-zero 或 Neumann。"""
        n_faces, n_fp, n_sps = 1, 1, 1
        scalar_sps = np.array([[5.0]])
        # 原生自身面外插表：编码 [6,10) 四面体、[10,15) 棱柱，统一按
        # `code - 6` 索引，所以表长 9。这里用 code=6（四面体 v0 面）。
        boundary_extrap_native = np.zeros((9, n_fp, n_sps))
        boundary_extrap_native[0] = np.array([[1.0]])
        owner_cell = np.array([0], dtype=np.int64)
        neighbor_src0_cell = np.array([-1], dtype=np.int64)
        neighbor_src0_mat = np.zeros((n_faces, n_fp, n_sps))
        neighbor_src1_idx = np.array([-1], dtype=np.int64)
        neighbor_src1_cell = np.empty((0,), dtype=np.int64)
        neighbor_src1_mat = np.empty((0, n_fp, n_sps))
        wall_dirichlet_zero_face = np.array([False])
        mixed_nb_partner = np.full(n_faces, -1, dtype=np.int64)
        mixed_nb_mask = np.zeros((n_faces, n_fp), dtype=np.bool_)
        has_wall_dirichlet_value = np.array([True])
        wall_dirichlet_value_face = np.array([[12.0]])
        owner_cube_face = np.full(n_faces, 6, dtype=np.int64)

        phi_owner, phi_neighbor = extrapolate_scalar_to_faces_kernel(
            scalar_sps,
            neighbor_src0_cell, neighbor_src0_mat,
            neighbor_src1_idx, neighbor_src1_cell, neighbor_src1_mat,
            owner_cell, n_faces, n_fp, n_sps,
            wall_dirichlet_zero_face,
            mixed_nb_partner, mixed_nb_mask,
            has_wall_dirichlet_value, wall_dirichlet_value_face,
            owner_cube_face, boundary_extrap_native,
        )

        np.testing.assert_allclose(phi_owner, [[5.0]])
        np.testing.assert_allclose(phi_neighbor, [[19.0]])  # 2*12.0 - 5.0
        face_avg = 0.5 * (phi_owner + phi_neighbor)
        np.testing.assert_allclose(face_avg, [[12.0]])


class TestComputeWallDirichletFaceMask:
    def test_identifies_wall_coded_faces_only(self):
        mesh = SimpleNamespace(face_connectivity=SimpleNamespace(n_faces=5))
        provider = SimpleNamespace(
            group_code=np.array([-1, 0, 1, 1, 2]),
            code_to_config={
                0: {"type": "FARFIELD"},
                1: {"type": "WALL"},
                2: {"type": "OUTLET"},
            },
        )
        solver = SimpleNamespace(mesh=mesh, boundary_ghost_provider=provider)

        mask = _compute_wall_dirichlet_face_mask(solver)

        np.testing.assert_array_equal(mask, [False, False, True, True, False])

    def test_no_wall_groups_returns_all_false(self):
        mesh = SimpleNamespace(face_connectivity=SimpleNamespace(n_faces=3))
        provider = SimpleNamespace(
            group_code=np.array([-1, 0, 0]),
            code_to_config={0: {"type": "FARFIELD"}},
        )
        solver = SimpleNamespace(mesh=mesh, boundary_ghost_provider=provider)

        mask = _compute_wall_dirichlet_face_mask(solver)

        np.testing.assert_array_equal(mask, [False, False, False])

    def test_missing_provider_metadata_falls_back_to_all_false(self):
        """A ghost provider without group_code/code_to_config (e.g. a plain
        callable used by some tests) must not raise - it degrades to the
        pre-fix Neumann-everywhere default, not a new failure mode."""
        mesh = SimpleNamespace(face_connectivity=SimpleNamespace(n_faces=4))
        solver = SimpleNamespace(mesh=mesh, boundary_ghost_provider=lambda f, q, n: q)

        mask = _compute_wall_dirichlet_face_mask(solver)

        np.testing.assert_array_equal(mask, [False, False, False, False])

    def test_no_ghost_provider_falls_back_to_all_false(self):
        mesh = SimpleNamespace(face_connectivity=SimpleNamespace(n_faces=2))
        solver = SimpleNamespace(mesh=mesh, boundary_ghost_provider=None)

        mask = _compute_wall_dirichlet_face_mask(solver)

        np.testing.assert_array_equal(mask, [False, False])

    def test_slip_wall_excluded_only_no_slip_wall_included(self):
        """真实 bug 回归测试（2026-09-12，cube_demo 791,492 单元真实网格
        P1 阶段 omega 独立于历史、确定性地在 7286 个单元收敛到同一数值
        ~984312 的排查发现）：`is_no_slip=False` 的滑移壁（如 cube_demo
        的风洞外壁 "tunnel"，物理上零剪切、不是真实边界层）不应该被
        当作需要 Wilcox omega 壁面解析式处理的壁面——否则这些单元的
        omega 会被 `enforce_omega_wall_relaxation` 每步强行拉向一个
        物理上荒谬的目标值（决定性验证：手术式重置 omega 场后单步内
        从 2268 跳到 501133，精确等于 0.5*2268+0.5*1e6，即 relax=0.5
        松弛向 omega_max 安全上限 1e6 的结果）。只有 `is_no_slip=True`
        （默认值，与 body 表面这类真实固壁一致）的 WALL 编码才应计入。
        """
        mesh = SimpleNamespace(face_connectivity=SimpleNamespace(n_faces=4))
        provider = SimpleNamespace(
            group_code=np.array([0, 1, 2, -1]),
            code_to_config={
                0: {"type": "WALL", "is_no_slip": True},   # body（真实固壁）
                1: {"type": "WALL", "is_no_slip": False},  # tunnel（滑移壁）
                2: {"type": "OUTLET"},
            },
        )
        solver = SimpleNamespace(mesh=mesh, boundary_ghost_provider=provider)

        mask = _compute_wall_dirichlet_face_mask(solver)

        np.testing.assert_array_equal(mask, [True, False, False, False])


class _MockNodes:
    def __init__(self, coords):
        self._coords = coords

    def get_coordinates(self):
        return self._coords


class _MockCells:
    def __init__(self, connectivity):
        self.connectivity = connectivity


def _build_synthetic_mixed_mesh(order: int):
    """2 个共享面的四面体 + 2 个共享侧面的棱柱——与
    test_fr_residual_inviscid.py::_build_synthetic_mixed_mesh 完全相同
    的几何构造（本文件独立保留一份，避免跨测试文件的私有函数依赖）。"""
    from autoflowcfd.grid.high_order.high_order_mesh import HighOrderMesh

    nodes = np.array(
        [
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1],
            [10, 0, 0], [11, 0, 0], [10, 1, 0], [10, 0, 1], [11, 0, 1], [10, 1, 1],
        ],
        dtype=float,
    )
    tet_conn = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int32)
    nodes = np.vstack([nodes, [[9, -1, 0], [9, -1, 1]]])
    prism_conn = np.array(
        [
            [5, 6, 7, 8, 9, 10],
            [5, 7, 11, 8, 10, 12],
        ],
        dtype=np.int32,
    )

    mock_volume = SimpleNamespace(
        cell_count=len(tet_conn) + len(prism_conn),
        nodes=_MockNodes(nodes),
        cells=_MockCells(tet_conn),
        prism_cells=_MockCells(prism_conn),
    )

    mesh = HighOrderMesh(order=order)
    mesh.load_from_volume_mesh(mock_volume)
    return mesh


class TestComputeOmegaWallTarget:
    """真实 bug 回归测试（V2.0 专家组盲审发现）：omega 壁面解析式
    `omega_wall = 60*nu/(beta1*d1^2)`（Wilcox 标准公式），此前完全没有
    接入、omega 恒用 Neumann 默认。"""

    def test_matches_wilcox_formula_on_wall_faces_only(self):
        from autoflowcfd.core.turbulence.transport import _compute_omega_wall_target

        mesh = _build_synthetic_mixed_mesh(order=1)
        ops = mesh.operators
        n_faces = mesh.face_connectivity.n_faces
        n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell

        # 构造一个逐 SP 不同的合成壁面距离场（避免退化成常数掩盖 bug）。
        rng = np.random.default_rng(0)
        wall_distance = rng.uniform(0.001, 0.1, size=(n_cells, n_sps))

        # 挑 2 个真实存在的边界面标记为 WALL（不依赖真实 BoundaryGhostProvider，
        # 只测 _compute_omega_wall_target 本身的公式/取值逻辑）。
        boundary_idx = np.nonzero(mesh.face_connectivity.is_boundary)[0]
        assert len(boundary_idx) >= 2, "synthetic mesh 应该有边界面"
        wall_mask = np.zeros(n_faces, dtype=np.bool_)
        wall_mask[boundary_idx[:2]] = True

        mu = 1.8e-5
        rho = np.full((n_cells, n_sps), 1.2)
        beta1 = 0.075
        turb_model = SimpleNamespace(beta1=beta1)
        solver = SimpleNamespace(
            mesh=mesh, ops=ops, wall_distance=wall_distance, turb_model=turb_model,
        )

        omega_wall_value_face, has_value = _compute_omega_wall_target(solver, wall_mask, mu, rho)

        np.testing.assert_array_equal(has_value, wall_mask)
        # 非 WALL 面必须恒为 0（不会被消费，但仍应是良定义的值，不是 NaN/垃圾）。
        non_wall = ~wall_mask
        np.testing.assert_array_equal(omega_wall_value_face[non_wall], 0.0)

        from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
        flat = get_flat_face_geometry(mesh, ops)
        for f in np.nonzero(wall_mask)[0]:
            owner = flat.owner_cell[f]
            d1_expected = max(np.min(wall_distance[owner]), 1e-8)
            nu = mu / rho[owner].mean()
            omega_expected = 60.0 * nu / (beta1 * d1_expected**2)
            np.testing.assert_allclose(omega_wall_value_face[f], omega_expected, rtol=1e-10)
            assert np.isfinite(omega_expected) and omega_expected > 0

    def test_degenerate_tiny_wall_distance_is_capped_at_omega_max(self):
        """真实 bug 回归测试（2026-09-05，cube_demo 791,492 单元真实网格
        验证决定性发现）：`d1 = np.maximum(d1, 1e-8)` 只防止除零，不防止
        结果本身失控——真实网格上至少有一个 WALL 面 owner 单元的
        wall_distance 恰好卡在这个 1e-8 下限，代入 Wilcox 公式算出
        omega_wall~1e14，比"远超任何工程壁面 omega 值"的安全上限
        omega_max(=1e6) 还大 8 个数量级。之前唯一的消费者（对流项上风
        ghost）碰巧被无滑移壁面处近零的对流通量掩盖了这个问题，
        2026-09-04/05 新增的消费者（enforce_omega_wall_relaxation 等）
        没有这层天然保护，完全暴露：'正确地'把 omega 松弛向一个物理上
        荒谬的目标值，几步内就把全域 omega_mean 打到 1e12 量级。"""
        from autoflowcfd.core.turbulence.transport import _compute_omega_wall_target

        mesh = _build_synthetic_mixed_mesh(order=1)
        ops = mesh.operators
        n_faces = mesh.face_connectivity.n_faces
        n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell

        # 故意构造一个退化的、恰好卡在 1e-8 下限的壁面距离场（模拟真实
        # 网格上观测到的退化单元），其余单元用正常量级的距离。
        wall_distance = np.full((n_cells, n_sps), 0.01)
        wall_distance[0, :] = 1e-9  # 比下限还小，会被 np.maximum 夹到 1e-8

        boundary_idx = np.nonzero(mesh.face_connectivity.is_boundary)[0]
        assert len(boundary_idx) >= 1
        wall_mask = np.zeros(n_faces, dtype=np.bool_)
        wall_mask[boundary_idx] = True

        mu = 1.8e-5
        rho = np.full((n_cells, n_sps), 1.2)
        omega_max = 1e6
        turb_model = SimpleNamespace(beta1=0.075, omega_max=omega_max)
        solver = SimpleNamespace(
            mesh=mesh, ops=ops, wall_distance=wall_distance, turb_model=turb_model,
        )

        omega_wall_value_face, has_value = _compute_omega_wall_target(solver, wall_mask, mu, rho)

        # 不管退化程度多严重（哪怕 d1 被夹到 1e-8 这个极限），输出必须
        # 被限制在 omega_max 以内，不能是物理上荒谬的 1e14 量级。
        assert np.all(omega_wall_value_face <= omega_max + 1e-6)
        assert np.all(np.isfinite(omega_wall_value_face))
        # 至少要有一个面真的撞到了上限（否则这个测试没有真正覆盖退化场景）。
        assert np.any(np.isclose(omega_wall_value_face[wall_mask], omega_max))

    def test_zero_wall_mask_gives_no_dirichlet_faces(self):
        from autoflowcfd.core.turbulence.transport import _compute_omega_wall_target

        mesh = _build_synthetic_mixed_mesh(order=1)
        n_faces = mesh.face_connectivity.n_faces
        n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell

        wall_mask = np.zeros(n_faces, dtype=np.bool_)
        solver = SimpleNamespace(
            mesh=mesh, ops=mesh.operators,
            wall_distance=np.full((n_cells, n_sps), 0.01),
            turb_model=SimpleNamespace(beta1=0.075),
        )

        omega_wall_value_face, has_value = _compute_omega_wall_target(
            solver, wall_mask, 1.8e-5, np.full((n_cells, n_sps), 1.2)
        )

        assert not np.any(has_value)
        np.testing.assert_array_equal(omega_wall_value_face, 0.0)


class TestEnforceOmegaWallRelaxation:
    """真实 bug 回归测试：omega 壁面 Wilcox 解析值只在对流项生效、扩散侧
    没有闭合这个已知架构缺口的缓解措施。

    2026-09-05 这一天先后尝试过两版更"数学严谨"的替代方案，都被真实
    791,492 单元网格验证证伪并撤销：
    (1) 显式 SIPG 罚项——刚性 c_pen/d1 在细网格近壁单元上用显式积分
        必然超调，20 步内把 omega_mean 打到 5.4e11；
    (2) 点隐式动态松弛系数 relax_eff=dt*c_wall/(1+dt*c_wall)——数学上
        确实排除了(1)的超调，但对*所有*足够刚性（d1 小）的单元都会
        让 relax_eff 趋近 1，几步内把大量边界层单元强行拉到 target
        （即便 target 本身已经被 omega_max 上限保护），这个物理扰动
        幅度本身就会通过 D_k=rho*beta_star*k*omega 把 k 场冲垮
        （2 步内 k_mean 从 38 打到 0.17，非 NaN/Inf，但同样不可接受）。

    现在保留、并已用真实生产续算验证 900+ 步（平均流场零漂移）的是
    最初这一版：固定 `relax`（默认 0.5，与 c_wall/dt 无关）——对所有
    命中的单元一视同仁地只走一半路程，天然温和，不会在几步内把耦合
    的湍流子系统冲垮。测试要点：(1) 基本混合公式正确；(2) 不超调；
    (3) 即使 wall_distance 极小（配合 `_compute_omega_wall_target` 的
    omega_max 上限）也不会 NaN/Inf/超调，但也*不应该*像已撤销的点
    隐式版本那样几步内就完全等于 target——固定 relax 就是要更温和。
    """

    def _build_solver(self, order=1, nu_t_value=0.5, wall_distance_value=0.01):
        from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry

        mesh = _build_synthetic_mixed_mesh(order=order)
        ops = mesh.operators
        fc = mesh.face_connectivity
        n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell

        wall_faces = np.nonzero(fc.is_boundary)[0]
        assert len(wall_faces) > 0, "synthetic mesh 应该有边界面"
        group_code = np.where(fc.is_boundary, 0, -1)
        boundary_ghost_provider = SimpleNamespace(
            group_code=group_code,
            code_to_config={0: {"type": "WALL"}},
        )

        rho_inf, u_inf, p_inf = 1.225, 30.0, 101325.0
        Q = np.zeros((n_cells, n_sps, 5))
        Q[..., 0] = rho_inf
        Q[..., 1] = u_inf
        Q[..., 4] = p_inf

        k_field = np.full((n_cells, n_sps), 1.0)
        omega_field = np.full((n_cells, n_sps), 1.0)  # 远低于解析壁面值，模拟塌陷场景
        nu_t = np.full((n_cells, n_sps), nu_t_value)
        turb_model = SimpleNamespace(
            k_field=k_field, omega_field=omega_field, nu_t=nu_t,
            beta1=0.075, sigma_w2=0.856,
        )

        solver = SimpleNamespace(
            mesh=mesh, ops=ops,
            state=SimpleNamespace(Q=Q, n_cells=n_cells),
            mu_molecular=1.8e-5,
            wall_distance=np.full((n_cells, n_sps), wall_distance_value),
            turb_model=turb_model,
            boundary_ghost_provider=boundary_ghost_provider,
            _turbulence_flat_face_override=None,
        )
        flat = get_flat_face_geometry(mesh, ops)
        return solver, flat

    def test_relaxed_value_matches_fixed_blend_no_overshoot(self):
        from autoflowcfd.core.turbulence.transport import (
            enforce_omega_wall_relaxation, _compute_wall_dirichlet_face_mask,
            _compute_omega_wall_target,
        )

        solver, flat = self._build_solver()
        omega_before = solver.turb_model.omega_field.copy()

        wall_mask = _compute_wall_dirichlet_face_mask(solver)
        rho = solver.state.Q[:, :, 0]
        omega_wall_value_face, has_wall = _compute_omega_wall_target(
            solver, wall_mask, solver.mu_molecular, rho,
        )
        wall_face_idx = np.nonzero(has_wall)[0]
        owner_cells = np.unique(flat.owner_cell[wall_face_idx])
        target_by_cell = {}
        for f in wall_face_idx:
            oc = int(flat.owner_cell[f])
            target_by_cell.setdefault(oc, []).append(omega_wall_value_face[f, 0])

        enforce_omega_wall_relaxation(solver, dt=1e-3)  # dt 未使用，见函数文档
        omega_after = solver.turb_model.omega_field

        for oc, targets in target_by_cell.items():
            target_avg = np.mean(targets)
            expected = 0.5 * omega_before[oc, 0] + 0.5 * target_avg  # 默认 relax=0.5
            np.testing.assert_allclose(omega_after[oc, 0], expected, rtol=1e-10)
            lo, hi = sorted([omega_before[oc, 0], target_avg])
            assert lo - 1e-6 <= omega_after[oc, 0] <= hi + 1e-6, (
                f"cell {oc}: old={omega_before[oc,0]}, target={target_avg}, "
                f"after={omega_after[oc,0]} — 超出 [old,target] 区间，说明发生了超调"
            )
        # 非 WALL owner 的单元不应该被这个函数动过。
        non_hit = np.array([c for c in range(solver.state.n_cells) if c not in owner_cells])
        if len(non_hit) > 0:
            np.testing.assert_array_equal(omega_after[non_hit], omega_before[non_hit])

    def test_custom_relax_coefficient_is_honored(self):
        from autoflowcfd.core.turbulence.transport import (
            enforce_omega_wall_relaxation, _compute_wall_dirichlet_face_mask,
            _compute_omega_wall_target,
        )

        solver, flat = self._build_solver()
        omega_before = solver.turb_model.omega_field.copy()

        wall_mask = _compute_wall_dirichlet_face_mask(solver)
        rho = solver.state.Q[:, :, 0]
        omega_wall_value_face, has_wall = _compute_omega_wall_target(solver, wall_mask, solver.mu_molecular, rho)
        wall_face_idx = np.nonzero(has_wall)[0]

        relax = 0.2
        enforce_omega_wall_relaxation(solver, dt=1e-3, relax=relax)
        omega_after = solver.turb_model.omega_field

        seen_cells = set()
        for f in wall_face_idx:
            oc = int(flat.owner_cell[f])
            if oc in seen_cells:
                continue
            seen_cells.add(oc)
            target = omega_wall_value_face[f, 0]
            expected = (1.0 - relax) * omega_before[oc, 0] + relax * target
            np.testing.assert_allclose(omega_after[oc, 0], expected, rtol=1e-10)

    def test_extreme_wall_distance_stays_bounded_and_gentle_not_full_target(self):
        """模拟退化的、极小 wall_distance 近壁单元（真实网格上确实观测到
        这类单元，见 TestComputeOmegaWallTarget 的 omega_max 上限回归
        测试）——即使解析目标值已被 omega_max 上限保护，固定 relax=0.5
        的松弛结果也必须仍然只是"半程混合"，绝不能像 2026-09-05 当天
        被证伪撤销的点隐式版本那样让近壁单元几步内就完全等于
        target（那个版本的动态 relax_eff 会在这种极端刚性场景下趋近
        1，把 k 场冲垮）——这是这次改回固定 relax 的核心诉求，必须
        有测试钉住，防止未来又不小心重新引入"隐式趋近 1"的行为。"""
        from autoflowcfd.core.turbulence.transport import (
            enforce_omega_wall_relaxation, _compute_wall_dirichlet_face_mask,
            _compute_omega_wall_target,
        )

        solver, flat = self._build_solver(wall_distance_value=1e-7)
        omega_before = solver.turb_model.omega_field.copy()

        wall_mask = _compute_wall_dirichlet_face_mask(solver)
        rho = solver.state.Q[:, :, 0]
        omega_wall_value_face, has_wall = _compute_omega_wall_target(
            solver, wall_mask, solver.mu_molecular, rho,
        )
        wall_face_idx = np.nonzero(has_wall)[0]

        enforce_omega_wall_relaxation(solver, dt=1.0)  # dt 故意取极端值，但函数文档明确它不参与计算
        omega_after = solver.turb_model.omega_field

        assert np.all(np.isfinite(omega_after))
        seen_cells = set()
        for f in wall_face_idx:
            oc = int(flat.owner_cell[f])
            if oc in seen_cells:
                continue
            seen_cells.add(oc)
            target = omega_wall_value_face[f, 0]
            # target 本身必须已经被 omega_max（默认 1e6）保护，不是失控的 1e14+。
            assert target <= 1e6 + 1e-6
            expected = 0.5 * omega_before[oc, 0] + 0.5 * target
            np.testing.assert_allclose(omega_after[oc, 0], expected, rtol=1e-10)
            # 关键判据：即使这是"最刚性"的极端场景，固定 relax 也只走半程，
            # 绝不应该等于（或极接近）target 本身。
            assert not np.isclose(omega_after[oc, 0], target, rtol=1e-2)
