"""AutoFlowCFD V2.0 - native 四面体 + WMLES 支持验证（2026-09-02）。

真实 bug（排查分布式 WMLES 支持时顺带发现，见
core/utils/solver_helpers.py 模块文档"native tet + WMLES"一节）：
`compute_wmles_wall_stress_correction`此前对四面体单元恒用坍缩坐标
专属的`ops.boundary_extrap_tet[(axis,side)]`/1D Radau/VCJH 修正函数
导数（`_distribute_from_face`），未区分`tet_basis_mode`。native 面的
`owner_axis`/`owner_side`存的是复用的 excluded_vertex/哑值（见
face_kernels.py::FlatFaceGeometry 字段文档），拿去查坍缩坐标专用的
矩阵字典在语义上是错的——修复前若真的用这些伪值去查
`ops.boundary_extrap_tet`，六个合法 `(axis,side)` 组合是
`{(0,-1),(0,1),(1,-1),(1,1),(2,-1),(2,1)}`，而 native 面存的
`owner_side`是恒定的哑值（见下方决定性验证），大概率不在这个集合里，
直接 KeyError；即使凑巧落在合法集合里，取到的也是语义不匹配的
坍缩坐标矩阵。

本文件验证修复后的行为：native 四面体 WALL 面能正确走
`boundary_extrap_native_tet`/`lift_native_tet_padded`分支，产出
有限、非零、只作用在正确 owner 单元上的动量修正。
"""

import numpy as np
import pytest

from autoflowcfd.fr.operators import generate_fr_operators
from autoflowcfd.core.turbulence.wmles import WMLESModel
from autoflowcfd.core.fr_solver.boundary import build_boundary_ghost_provider
from autoflowcfd.core.utils.solver_helpers import compute_wmles_wall_stress_correction
from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive, primitive_to_conserved
from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh


def _uniform_freestream_U(mesh) -> np.ndarray:
    rho_inf, u_inf, v_inf, w_inf, p_inf = 1.225, 30.0, 5.0, -3.0, 101325.0
    Q_inf = np.array([rho_inf, u_inf, v_inf, w_inf, p_inf])
    U_inf = primitive_to_conserved(Q_inf)
    return np.tile(U_inf, (mesh.n_cells, mesh.n_sps_per_cell, 1))


class TestNativeTetWmlesBoundaryExtrap:
    def test_old_collapsed_extrap_would_give_wrong_answer_on_native_face(self):
        """决定性前提验证：修复前的做法（对 native 四面体 WALL 面仍用
        `ops.boundary_extrap_tet[(owner_axis,owner_side)]`外插）不只是
        理论上"语义不对"，在这个具体合成网格上真的会暴露问题——
        `owner_axis`/`owner_side`对 native 面存的是复用的 excluded_
        vertex/哑值，逐面检查发现两种暴露方式都出现了：有的面伪键
        根本不在 6 个合法 (axis,side) 组合里，直接 KeyError；有的面
        伪键凑巧撞上某个真实键，但取到的矩阵语义不对，外插结果与正确
        的 native 外插（`boundary_extrap_native_tet`，按需 pad）不同。
        这证明这是一个需要真正修复的真实 bug，而不是理论上的边界情况。"""
        from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
        from autoflowcfd.fr.native_padding import pad_native_matrix_to_global

        order = 2
        mesh = _build_synthetic_mixed_mesh(order, tet_basis_mode="native")
        ops = generate_fr_operators(order)
        flat = get_flat_face_geometry(mesh, ops)
        n_sps = mesh.n_sps_per_cell

        native_faces = np.nonzero(flat.owner_cube_face >= 6)[0]
        assert len(native_faces) > 0, "test setup must produce at least one native tet face"

        rng = np.random.default_rng(0)
        field = rng.uniform(-1.0, 1.0, size=(n_sps, 3))

        # 逐个 native 面检查修复前的做法（用 owner_axis/owner_side 当
        # 坍缩坐标键查 ops.boundary_extrap_tet）会不会暴露问题——两种
        # 暴露方式都足以证明这是真实 bug：(a) 伪键根本不在 6 个合法
        # (axis,side) 组合里，直接 KeyError；(b) 伪键凑巧撞上某个真实
        # 键，但取到的矩阵语义不对，外插结果与正确的 native 外插不同。
        found_keyerror = False
        found_wrong_value = False
        for f in native_faces:
            axis, side = int(flat.owner_axis[f]), float(flat.owner_side[f])
            excluded_vertex = int(flat.owner_cube_face[f]) - 6
            E_correct = pad_native_matrix_to_global(
                ops.boundary_extrap_native_tet[excluded_vertex], n_sps, pad_axes=(1,)
            )
            result_correct = E_correct @ field
            try:
                E_wrong = ops.boundary_extrap_tet[(axis, side)]
            except KeyError:
                found_keyerror = True
                continue
            result_wrong = E_wrong @ field
            if result_wrong.shape != result_correct.shape or not np.allclose(result_wrong, result_correct):
                found_wrong_value = True

        assert found_keyerror or found_wrong_value, (
            "修复前的坍缩坐标外插矩阵在这个网格的全部 native 面上都恰好"
            "得到与正确 native 外插相同的结果——说明这份合成网格无法暴露"
            "该 bug，需要换一个反例网格"
        )

    def test_wmles_wall_stress_on_native_tet_face(self):
        order = 2
        mesh = _build_synthetic_mixed_mesh(order, tet_basis_mode="native")
        ops = generate_fr_operators(order)
        n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
        n_prism = mesh.n_prism_cells
        mu, rho_inf = 1.8e-5, 1.225

        # 找一个真实的四面体（native）边界面，把它的 owner 单元标成 WALL。
        fc = mesh.face_connectivity
        tet_boundary_faces = np.nonzero(fc.is_boundary & (fc.owner_cell >= n_prism))[0]
        assert len(tet_boundary_faces) > 0, "test setup must have a real tet boundary face"
        boundary_face = int(tet_boundary_faces[0])
        wall_cell = int(fc.owner_cell[boundary_face])
        mesh.boundary_groups = {"wall_group": np.array([wall_cell], dtype=np.int64)}
        mesh.boundary_bc_types = {"wall_group": "WALL"}

        import types
        root_stub = types.SimpleNamespace(
            mesh=mesh, freestream={"rho_inf": rho_inf, "vel_inf": 30.0, "p_inf": 101325.0},
            turb_model_name="WMLES", wmles_model=object(),
        )
        provider = build_boundary_ghost_provider(root_stub, bc_overrides={})
        wmles_model = WMLESModel(nu=mu / rho_inf)
        wall_distance = np.full((n_cells, n_sps), 1e-2)

        U = _uniform_freestream_U(mesh)
        Q = conserved_to_primitive(U[..., :5])
        facade = types.SimpleNamespace(
            wmles_model=wmles_model, mesh=mesh, ops=ops, wall_distance=wall_distance,
            state=types.SimpleNamespace(U=U, Q=Q), boundary_ghost_provider=provider,
        )

        correction = compute_wmles_wall_stress_correction(facade)

        assert correction is not None, "test setup must produce at least one real WALL face"
        assert correction.shape == (n_cells, n_sps, 5)
        assert np.all(np.isfinite(correction))
        # 只有分量 1:4（动量）非零。
        assert np.allclose(correction[..., 0], 0.0)
        assert np.allclose(correction[..., 4], 0.0)
        # 修正只施加在 wall_cell 上，其余单元恒为零。
        other_cells = [c for c in range(n_cells) if c != wall_cell]
        for c in other_cells:
            assert np.allclose(correction[c], 0.0), f"correction leaked onto non-WALL cell {c}"
        assert np.any(correction[wall_cell, :, 1:4] != 0.0), "wall cell should get a nonzero momentum correction"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
