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
    def test_the_collapsed_extrap_table_is_gone(self):
        """原「修复前的做法会出错」那条前提验证已随被测对象一并删除。

        它当年逐面证明了：对原生四面体 WALL 面沿用
        `ops.boundary_extrap_tet[(owner_axis, owner_side)]` 外插，两种
        暴露方式都会真的出现 —— 有的面伪键根本不在 6 个合法 (axis,side)
        组合里、直接 `KeyError`；有的面伪键凑巧撞上某个真实键、取到的
        矩阵语义不对，外插结果与正确的原生外插不同。那份证据保留在
        `ProjectFiles/V2.0/19_重大问题修复-湍流模型跨后端逐项排查与完全
        分布式加载补齐.md`。

        2026-09-24 起 `FROperators` 上已经没有 `boundary_extrap_tet` /
        `boundary_extrap_prism` 这两张坍缩外插表（随坍缩 1D 分布机制一并
        删除，见 `fr_residual/inviscid_kernel.py::
        compute_inviscid_interface_correction_kernel` 文档），那条错误做法
        在代码层面已经不可能被写出来。这里把「表确实没了」钉住，取代
        原来那条无法再运行的前提验证。
        """
        ops = generate_fr_operators(2)
        for gone in ("boundary_extrap_tet", "boundary_extrap_prism",
                     "g_left", "g_right", "L_interp"):
            assert not hasattr(ops, gone), (
                f"FROperators 又出现了 {gone} —— 那条坍缩路径已于 "
                f"2026-09-24 删除，重新加回它需要先恢复一条一维张量积"
                f"离散路径（见 ProjectFiles/V2.0/27_...md）")


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
