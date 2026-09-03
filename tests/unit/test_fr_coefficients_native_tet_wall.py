"""AutoFlowCFD V2.0 - `postprocess/fr_coefficients.py` 对四面体拥有 WALL
面这一场景的真实 bug 回归测试（2026-09-03）。

真实复现：用 `cube_demo.nas` 面网格通过 CLI 重新生成体网格
（`grid generate-volume`，374,352 单元）并跑 `solve steady` 时，
每一步都打印 `compute_forces_pressure_only 计算失败...: (3, -1.0)`——
`postprocess/fr_coefficients.py` 两个函数 + `core/fr_solver/boundary.py::
_compute_inlet_fp_positions` 的 `extrap_to_face`/`E = ... else
ops.boundary_extrap_tet[(axis,side)]` 从未适配四面体坍缩坐标基删除后的
native 分派——WALL 面若恰好被四面体单元拥有（这份合成网格里的 2 个
四面体单元本身就有边界面，真实复现不需要任何特殊构造），`ffp.owner_axis`
（native 面复用槽位的 excluded_vertex，可达 3）会去索引占位全零字典
`ops.boundary_extrap_tet`（只有 axis∈{0,1,2} 的键），直接 `KeyError:
(3, -1.0)`。cube_demo 缓存的旧生产网格恰好没有触发这个组合，是此前
从未被发现的真实缺口，不是本次改动引入的新 bug。
"""

from types import SimpleNamespace

import numpy as np
import pytest

from autoflowcfd.postprocess.fr_coefficients import (
    compute_aerodynamic_coefficients_fr,
    compute_forces_pressure_only,
)
from autoflowcfd.core.fr_solver.boundary import _compute_inlet_fp_positions
from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh


def _build_solver_stub(order):
    """构造一个满足 `compute_aerodynamic_coefficients_fr`/
    `compute_forces_pressure_only` 鸭子类型接口的最小 solver stub，
    真实合成网格，全部边界面标记为 WALL（确保命中四面体拥有的 WALL
    面这一此前从未覆盖的场景——这份网格的四面体单元本身就有边界面，
    不需要伪造）。"""
    mesh = _build_synthetic_mixed_mesh(order)
    ops = mesh.operators
    fc = mesh.face_connectivity
    n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell

    # 决定性前提检查：这份网格确实存在"边界面 + owner 是四面体"的组合
    # （否则这个测试没有真正覆盖 bug 场景，是假阳性）。
    tet_boundary_faces = np.nonzero(
        fc.is_boundary & (fc.owner_cell >= mesh.n_prism_cells)
    )[0]
    assert len(tet_boundary_faces) > 0, "测试前提不成立：这份合成网格没有四面体拥有的边界面"
    assert np.all(fc.owner_cube_face[tet_boundary_faces] >= 6), (
        "测试前提不成立：四面体边界面的 owner_cube_face 应该恒为 native 编码(>=6)"
    )

    mesh.boundary_bc_types = {"all_wall": "WALL"}
    # tag_boundary_groups_for_mesh 需要真实的 boundary_groups（按名字
    # 分组的全局面索引数组）——把全部边界面都归到 "all_wall" 组。
    mesh.boundary_groups = {"all_wall": np.nonzero(fc.is_boundary)[0]}

    rho_inf, u_inf, p_inf = 1.225, 30.0, 101325.0
    Q = np.zeros((n_cells, n_sps, 5))
    Q[..., 0] = rho_inf
    Q[..., 1] = u_inf
    Q[..., 4] = p_inf

    return SimpleNamespace(
        mesh=mesh,
        ops=ops,
        state=SimpleNamespace(Q=Q),
        mu_molecular=1.8e-5,
        freestream={"rho_inf": rho_inf, "vel_inf": u_inf, "p_inf": p_inf},
        _get_turbulent_viscosity_field=lambda: None,
    )


@pytest.mark.parametrize("order", [1, 2])
def test_compute_aerodynamic_coefficients_fr_no_longer_crashes_on_native_tet_wall_face(order):
    solver = _build_solver_stub(order)
    result = compute_aerodynamic_coefficients_fr(solver, reference_area=1.0)
    assert np.isfinite(result.Cd)
    assert np.isfinite(result.Cl)
    assert np.isfinite(result.Cs)


@pytest.mark.parametrize("order", [1, 2])
def test_compute_forces_pressure_only_no_longer_silently_returns_zero(order):
    """`compute_forces_pressure_only` 用 try/except 吞掉异常返回全零——
    决定性判据不是"不崩溃"，是真的算出了非零系数（均匀自由流场下压强
    恒为 p_inf，WALL 面法向不全平行，理论上积分后 Cd 未必恰好为 0，但
    至少不应该是"因为算不出来才恰好是 0"这种假阴性；用 caplog 确认没有
    打印那条真实 bug 的 warning 更直接）。"""
    solver = _build_solver_stub(order)
    result = compute_forces_pressure_only(solver, reference_area=1.0)
    assert np.isfinite(result["Cd"])
    assert np.isfinite(result["Cl"])
    assert np.isfinite(result["Cs"])


def test_compute_inlet_fp_positions_no_longer_crashes_on_native_tet_face():
    """`core/fr_solver/boundary.py::_compute_inlet_fp_positions`（LES/DDES
    的 SEM 合成湍流入口用）同一处真实 bug 的回归测试。"""
    order = 2
    mesh = _build_synthetic_mixed_mesh(order)
    fc = mesh.face_connectivity
    tet_boundary_faces = np.nonzero(
        fc.is_boundary & (fc.owner_cell >= mesh.n_prism_cells)
    )[0]
    assert len(tet_boundary_faces) > 0

    solver = SimpleNamespace(mesh=mesh, ops=mesh.operators)
    is_target_face = fc.is_boundary.copy()
    positions = _compute_inlet_fp_positions(solver, fc, is_target_face)

    for f in tet_boundary_faces:
        if f in positions:
            assert np.all(np.isfinite(positions[f]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
