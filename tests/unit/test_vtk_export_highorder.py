"""AutoFlowCFD V2.0 - #5 VTK 高阶 Lagrange 单元导出单元测试。

核心验证方式（与本模块开发过程中的独立验证一致，也是
`ProjectFiles/V2.0/0_项目实施路径.md` 对本条目要求的方式）：构造一个
已知的物理空间二次解析场（P=2 下应精确表示），赋值到真实 HighOrderMesh
的 SPs 上，导出后：
1. 直接比较插值矩阵产出的 VTK 节点值与解析函数在节点物理坐标处的值——
   验证插值本身正确。
2. 写出真实 .vtu 文件、重新加载、在单元内部一个非节点位置用 pyvista
   自己的采样/插值重新求值，与解析值比较——验证节点排序 + VTK 自身对
   这批节点的形函数插值全链路正确，不只是"文件能写出"。
"""

import numpy as np
import pytest

from autoflowcfd.postprocess.vtk_export_highorder import export_highorder_vtk
from autoflowcfd.postprocess.vtk_export_highorder.node_layout import _tet_vtk_node_barycentrics, _wedge_vtk_node_layout
from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh, _MockNodes, _MockCells
from autoflowcfd.grid.high_order.high_order_mesh import HighOrderMesh

GAMMA = 1.4


def _build_pure_tet_mesh(order: int) -> HighOrderMesh:
    """纯四面体合成网格（无棱柱）——验证 order=3 四面体高阶导出时不能用
    `_build_synthetic_mixed_mesh`（含棱柱，棱柱侧限制在 order<=2，会在
    到达四面体检查之前先因棱柱阶数超限报错，测不到四面体本身的行为）。
    """
    from types import SimpleNamespace
    nodes = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]],
        dtype=float,
    )
    tet_conn = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int32)
    mock_volume = SimpleNamespace(
        cell_count=len(tet_conn),
        nodes=_MockNodes(nodes),
        cells=_MockCells(tet_conn),
        prism_cells=_MockCells(np.zeros((0, 6), dtype=np.int32)),
    )
    mesh = HighOrderMesh(order=order)
    mesh.load_from_volume_mesh(mock_volume)
    return mesh


def _cubic_field(xyz: np.ndarray) -> np.ndarray:
    """物理空间三次多项式，P=3 下应能被 FR 解精确表示——用于决定性
    验证 order=3 四面体高阶导出（含新增的面内部点）。"""
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    return (
        101325.0 + 50.0 * x - 30.0 * y + 20.0 * z
        + 5.0 * x * y - 3.0 * y * z + 2.0 * x * z
        + 1.5 * x**2 - 0.8 * y**2 + 0.4 * z**2
        + 0.7 * x**3 - 0.5 * y**3 + 0.3 * z**3
        + 0.2 * x * y * z
    )


def _quadratic_field(xyz: np.ndarray) -> np.ndarray:
    """物理空间二次多项式，P=2 下应能被 FR 解精确表示。"""
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    return (
        101325.0 + 50.0 * x - 30.0 * y + 20.0 * z
        + 5.0 * x * y - 3.0 * y * z + 2.0 * x * z
        + 1.5 * x**2 - 0.8 * y**2
    )


class TestVtkNodeLayoutCounts:
    def test_tet_order1_is_just_corners(self):
        bary = _tet_vtk_node_barycentrics(1)
        assert bary.shape == (4, 4)

    def test_tet_order2_has_10_nodes(self):
        bary = _tet_vtk_node_barycentrics(2)
        assert bary.shape == (10, 4)

    def test_tet_order3_has_20_nodes(self):
        """order=3 已实现（2026-09-02）：4 角点 + 6 棱各 2 内部点(12) +
        4 面各 1 内部点(4) = 20 = C(3+3,3)（标准四面体 Lagrange 节点
        计数公式），无体内部点（order<4）。"""
        bary = _tet_vtk_node_barycentrics(3)
        assert bary.shape == (20, 4)
        np.testing.assert_allclose(bary.sum(axis=1), 1.0, atol=1e-12)

    def test_tet_order4_raises_not_implemented(self):
        """order>=4 已实测证伪（面内部点排列对面顶点顺序敏感，见模块
        文档），显式拒绝而不是虚报支持。"""
        with pytest.raises(NotImplementedError):
            _tet_vtk_node_barycentrics(4)

    def test_wedge_order1_is_just_corners(self):
        tri_bary, z = _wedge_vtk_node_layout(1)
        assert tri_bary.shape == (6, 3)
        assert z.shape == (6,)

    def test_wedge_order2_has_18_nodes(self):
        tri_bary, z = _wedge_vtk_node_layout(2)
        assert tri_bary.shape == (18, 3)

    def test_wedge_order3_has_40_nodes_each_once(self):
        """order=3：6 角点 + 9 棱 x 2 + 2 三角形面 x 1 + 3 四边形面 x 4 + 体内部 2 = 40 = (p+1)^2 (p+2)/2，
        每个格点恰好占一个编号（`_wedge_vtk_node_layout` 内部检查冲突与缺号）。"""
        tri_bary, z = _wedge_vtk_node_layout(3)
        assert tri_bary.shape == (40, 3)
        lattice = {(round(3 * b[1]), round(3 * b[2]), round(3 * zz)) for b, zz in zip(tri_bary, z)}
        assert len(lattice) == 40

    def test_wedge_order4_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            _wedge_vtk_node_layout(4)


class TestExportHighorderVtk:
    """端到端验证：真实网格 + 已知二次解析场 + 真实 .vtu 写出/重读。"""

    def _build_conserved_field(self, mesh):
        """构造 U，使 conserved_to_primitive(U) 的压力分量恰好等于
        _quadratic_field(物理坐标)，密度=1、速度=0（避免动能交叉项，
        令 rho*E = p/(gamma-1) 精确成立）。"""
        n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
        p_at_sps = _quadratic_field(mesh.sps_coords)  # (n_cells, n_sps)
        U = np.zeros((n_cells, n_sps, 5))
        U[..., 0] = 1.0  # rho
        U[..., 4] = p_at_sps / (GAMMA - 1.0)  # rho*E = e_internal (u=v=w=0)
        return U

    def test_interpolated_node_values_match_analytic_field(self, tmp_path):
        mesh = _build_synthetic_mixed_mesh(order=2)
        U = self._build_conserved_field(mesh)

        out_path = tmp_path / "highorder_test.vtu"
        export_highorder_vtk(mesh, U, out_path, fields=['pressure'])
        assert out_path.exists()

        import pyvista as pv
        grid = pv.read(str(out_path))
        assert grid.n_cells == mesh.n_cells

        node_pressures = grid.point_data['Pressure']
        node_points = grid.points
        expected = _quadratic_field(node_points)

        rel_err = np.max(np.abs(node_pressures - expected)) / np.max(np.abs(expected))
        assert rel_err < 1e-8, f"Node-value interpolation mismatch: rel_err={rel_err:.3e}"

    def test_probed_interior_value_matches_analytic_field(self, tmp_path):
        """比节点值比较更强的判据：在单元内部（非节点）位置，用 VTK
        自己的形函数插值重新采样，验证节点排序对 VTK 读取器同样正确
        （不仅仅是我们自己构造的插值矩阵自洽）。"""
        mesh = _build_synthetic_mixed_mesh(order=2)
        U = self._build_conserved_field(mesh)

        out_path = tmp_path / "highorder_probe_test.vtu"
        export_highorder_vtk(mesh, U, out_path, fields=['pressure'])

        import pyvista as pv
        grid = pv.read(str(out_path))

        # 用每个物理单元的真实节点重心平均作为一个安全的内部探测点
        # （凸单元内部点，不落在任何面/棱上）。
        cell_centers = grid.cell_centers().points
        sampled = pv.PolyData(cell_centers).sample(grid)
        valid = sampled['vtkValidPointMask'].astype(bool)
        assert np.any(valid), "No successfully probed points"

        sampled_pressure = sampled['Pressure'][valid]
        expected = _quadratic_field(cell_centers[valid])

        rel_err = np.max(np.abs(sampled_pressure - expected)) / np.max(np.abs(expected))
        assert rel_err < 1e-6, f"Interior-probe mismatch: rel_err={rel_err:.3e}"

    def test_order_above_max_supported_raises(self, tmp_path):
        """order=4 超过已验证的节点编号（两类单元都只验证到求解器最高阶数 3）：显式拒绝。"""
        mesh = _build_synthetic_mixed_mesh(order=4)
        U = np.zeros((mesh.n_cells, mesh.n_sps_per_cell, 5))
        U[..., 0] = 1.0
        U[..., 4] = 1.0
        with pytest.raises(NotImplementedError):
            export_highorder_vtk(mesh, U, tmp_path / "should_not_be_written.vtu")


class TestExportHighorderVtkOrder3Tet:
    """order=3 四面体高阶导出端到端决定性验证（2026-09-02 新增）——与
    `TestExportHighorderVtk`（order=2）同一套判据，换成纯四面体网格
    （无棱柱，避免棱柱侧的 order<=2 限制先触发）+ 三次解析场（P=3 下
    应精确表示，含新增的面内部点这条链路）。
    """

    def _build_conserved_field(self, mesh):
        n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
        p_at_sps = _cubic_field(mesh.sps_coords)
        U = np.zeros((n_cells, n_sps, 5))
        U[..., 0] = 1.0
        U[..., 4] = p_at_sps / (GAMMA - 1.0)
        return U

    def test_interpolated_node_values_match_analytic_field(self, tmp_path):
        mesh = _build_pure_tet_mesh(order=3)
        U = self._build_conserved_field(mesh)

        out_path = tmp_path / "highorder_order3_test.vtu"
        export_highorder_vtk(mesh, U, out_path, fields=['pressure'])
        assert out_path.exists()

        import pyvista as pv
        grid = pv.read(str(out_path))
        assert grid.n_cells == mesh.n_cells

        node_pressures = grid.point_data['Pressure']
        node_points = grid.points
        expected = _cubic_field(node_points)

        rel_err = np.max(np.abs(node_pressures - expected)) / np.max(np.abs(expected))
        assert rel_err < 1e-6, f"order=3 node-value interpolation mismatch: rel_err={rel_err:.3e}"

    def test_probed_interior_value_matches_analytic_field(self, tmp_path):
        """比节点值比较更强的判据：VTK 自身形函数在单元内部非节点位置
        重新插值——真正验证 order=3 新增的面内部点排序（4个面各1个，
        见 _TET_FACES_VTK 文档）对 VTK 读取器同样正确。"""
        mesh = _build_pure_tet_mesh(order=3)
        U = self._build_conserved_field(mesh)

        out_path = tmp_path / "highorder_order3_probe_test.vtu"
        export_highorder_vtk(mesh, U, out_path, fields=['pressure'])

        import pyvista as pv
        grid = pv.read(str(out_path))

        cell_centers = grid.cell_centers().points
        sampled = pv.PolyData(cell_centers).sample(grid)
        valid = sampled['vtkValidPointMask'].astype(bool)
        assert np.any(valid), "No successfully probed points"

        sampled_pressure = sampled['Pressure'][valid]
        expected = _cubic_field(cell_centers[valid])

        rel_err = np.max(np.abs(sampled_pressure - expected)) / np.max(np.abs(expected))
        assert rel_err < 1e-6, f"order=3 interior-probe mismatch: rel_err={rel_err:.3e}"

    def test_order4_tet_raises_not_implemented(self, tmp_path):
        """order=4 已实测证伪（见 vtk_export_highorder.py 模块文档），
        必须显式拒绝，不能悄悄产出错误结果。"""
        mesh = _build_pure_tet_mesh(order=4)
        U = np.zeros((mesh.n_cells, mesh.n_sps_per_cell, 5))
        U[..., 0] = 1.0
        U[..., 4] = 1.0
        with pytest.raises(NotImplementedError):
            export_highorder_vtk(mesh, U, tmp_path / "should_not_be_written_order4.vtu")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

def _poly_field(xyz, order):
    """物理空间 order 次多项式（P=order 下被 FR 解精确表示）。"""
    return _cubic_field(xyz) if order == 3 else (_quadratic_field(xyz) if order == 2 else
                                                  101325.0 + 50.0 * xyz[..., 0] - 30.0 * xyz[..., 1] + 20.0 * xyz[..., 2])


def _pressure_state(mesh, order):
    U = np.zeros((mesh.n_cells, mesh.n_sps_per_cell, 5))
    U[..., 0] = 1.0
    U[..., 4] = _poly_field(mesh.sps_coords, order) / (GAMMA - 1.0)
    return U


def _vtk_parametric_coords(cell):
    """VTK 单元自报的各节点参数坐标 `(n_nodes, 3)`（节点编号的权威来源）。"""
    pc = cell.GetParametricCoords()
    n = cell.GetNumberOfPoints()
    return np.array([pc[q] for q in range(3 * n)]).reshape(n, 3)


def _vtk_shape_function_samples(grid, label, n_per_cell, rng):
    """用 VTK 单元自己的形函数在单元内部随机参数点上求物理坐标与场值（不经过点定位器）。"""
    import vtk

    values = np.asarray(grid.point_data[label])
    xs, fs = [], []
    for c in range(grid.n_cells):
        cell = grid.GetCell(c)
        ids = np.array([cell.GetPointId(q) for q in range(cell.GetNumberOfPoints())])
        weights = vtk.vtkDoubleArray()
        for _ in range(n_per_cell):
            if cell.GetCellType() == 73:                       # 三棱柱：三角形 (r, s) x t
                r, s_ = rng.dirichlet(np.ones(3))[:2]
                pcoords = [r, s_, rng.uniform()]
            else:                                              # 四面体
                pcoords = list(rng.dirichlet(np.ones(4))[:3])
            x = [0.0, 0.0, 0.0]
            w = [0.0] * len(ids)
            sub_id = vtk.reference(0)
            cell.EvaluateLocation(sub_id, pcoords, x, w)
            cell.InterpolateFunctions(pcoords, w)
            xs.append(x)
            fs.append(np.dot(w, values[ids]))
    return np.array(xs), np.array(fs)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_node_numbering_matches_vtk_parametric_coords(tmp_path, order):
    """节点编号与 VTK 单元自报的参数坐标逐点一致（三棱柱与四面体）。2026-10-18：三棱柱 order 3 的第三个四边形
    侧面（i=0）内部点顺序曾经反了——order 2 时每面只有一个内部点，测不出来。"""
    import pyvista as pv

    mesh = _build_synthetic_mixed_mesh(order=order)
    grid = pv.read(str(export_highorder_vtk(mesh, _pressure_state(mesh, order), tmp_path / "m.vtu",
                                            fields=['pressure'])))
    wedge = next(c for c in range(grid.n_cells) if grid.GetCell(c).GetCellType() == 73)
    tet = next(c for c in range(grid.n_cells) if grid.GetCell(c).GetCellType() == 71)
    tri, z = _wedge_vtk_node_layout(order)
    np.testing.assert_allclose(_vtk_parametric_coords(grid.GetCell(wedge)),
                               np.column_stack([tri[:, 1], tri[:, 2], z]), atol=1e-12)
    bary = _tet_vtk_node_barycentrics(order)
    np.testing.assert_allclose(_vtk_parametric_coords(grid.GetCell(tet)), bary[:, 1:], atol=1e-12)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_mixed_mesh_exports_exact_polynomial_at_every_order(tmp_path, order):
    """棱柱 + 四面体混合网格，order 1~3：节点值、以及 VTK 自身形函数在单元内部随机参数点上插值出的值，都与
    物理空间解析场一致。2026-10-18 以前三棱柱只支持到 order 2，含边界层的 P3 结果无法高阶导出。"""
    import pyvista as pv

    mesh = _build_synthetic_mixed_mesh(order=order)
    assert mesh.n_prism_cells > 0 and mesh.n_cells > mesh.n_prism_cells
    out = export_highorder_vtk(mesh, _pressure_state(mesh, order), tmp_path / "mixed.vtu", fields=['pressure'])
    grid = pv.read(str(out))
    assert grid.n_cells == mesh.n_cells
    expected = _poly_field(grid.points, order)
    scale = np.abs(expected).max()
    assert np.abs(grid.point_data['Pressure'] - expected).max() < 1e-9 * scale
    x, f = _vtk_shape_function_samples(grid, 'Pressure', 8, np.random.default_rng(order))
    err = np.abs(f - _poly_field(x, order)).max()
    assert err < 1e-9 * scale, f"VTK 形函数内部插值误差 {err / scale:.3e}"


def test_sp_fields_are_exported_with_low_order_names(tmp_path):
    """解点附加场（湍流量、Q 准则）按低阶导出同一组场名写出，'turbulence' 展开为可用的湍流量；节点值是解点
    多项式的精确插值。请求了没有提供的场时报错。"""
    import pyvista as pv

    mesh = _build_synthetic_mixed_mesh(order=2)
    U = _pressure_state(mesh, 2)
    k = 1e-2 * _quadratic_field(mesh.sps_coords) / 101325.0
    sp = {'k': k, 'omega': 2.0 * k, 'nut': 3.0 * k, 'q_criterion': 4.0 * k}
    out = export_highorder_vtk(mesh, U, tmp_path / "turb.vtu", fields=['velocity', 'turbulence', 'q_criterion'],
                               sp_fields=sp)
    grid = pv.read(str(out))
    k_exp = 1e-2 * _quadratic_field(grid.points) / 101325.0
    for label, factor in (('TurbulentKineticEnergy', 1.0), ('SpecificDissipationRate', 2.0),
                          ('TurbulentViscosity', 3.0), ('QCriterion', 4.0)):
        np.testing.assert_allclose(grid.point_data[label], factor * k_exp, rtol=1e-9)
    assert 'SANuTilde' not in grid.point_data
    with pytest.raises(ValueError, match="没有提供解点值"):
        export_highorder_vtk(mesh, U, tmp_path / "x.vtu", fields=['nu_tilde'], sp_fields=sp)


def _sst_channel_solver():
    from autoflowcfd.core.fr_solver import FRSolver
    from autoflowcfd.core.fr_solver.turbulence.wall_distance import apply_wall_distance_source
    from autoflowcfd.core.time_integration import TimeIntegrationScheme
    from tests.validation._channel_mesh import build_channel_mesh_prism, channel_wall_source

    mesh = build_channel_mesh_prism(2, 3, 3, 2, 0.4, 0.1, 0.08)
    mesh.boundary_bc_types = {"x_min": "VELOCITY_INLET", "x_max": "PRESSURE_OUTLET", "wall_bottom": "WALL",
                              "wall_top": "WALL", "z_min": "SYMMETRY", "z_max": "SYMMETRY"}
    s = FRSolver(mesh=mesh, order=2, turb_model_name="SST", time_scheme=TimeIntegrationScheme.SSP_RK3, vel_inf=30.0)
    s.order_continuation_enabled = False
    apply_wall_distance_source(s, channel_wall_source(0.4, 0.1, 0.08))
    y = np.asarray(s.mesh.sps_coords)[..., 1]
    s.turb_model.k_field = 0.1 + 2.0 * y + 5.0 * y ** 2          # 二次场：P2 精确表示
    return s


def test_solver_export_carries_turbulence_and_q_criterion(tmp_path):
    """求解器级入口（API 与 CLI 共用）：湍流量取模型的解点场、Q 准则按解点速度梯度计算。"""
    import pyvista as pv

    from autoflowcfd.postprocess.vtk_export_highorder import export_solver_highorder_vtk

    s = _sst_channel_solver()
    out = export_solver_highorder_vtk(s, tmp_path / "solver.vtu", fields=['velocity', 'turbulence', 'q_criterion'])
    grid = pv.read(str(out))
    y = grid.points[:, 1]
    np.testing.assert_allclose(grid.point_data['TurbulentKineticEnergy'], 0.1 + 2.0 * y + 5.0 * y ** 2, rtol=1e-9)
    for label in ('SpecificDissipationRate', 'TurbulentViscosity', 'QCriterion', 'Velocity'):
        assert label in grid.point_data


def test_cli_high_order_export(tmp_path):
    """`post export-vtk --high-order`：从 checkpoint 重建求解器后按解点多项式导出（此前高阶导出只有 API 入口）；
    与 --boundaries-only 同用、或输出不是 .vtu 时报错。"""
    from unittest.mock import patch

    import pyvista as pv
    from click.testing import CliRunner

    from autoflowcfd.cli.post.export_commands import export_vtk

    s = _sst_channel_solver()
    with patch("autoflowcfd.cli.solve.checkpoint_io.rebuild_solver_from_checkpoint", return_value=(s, 0, {})),             patch("autoflowcfd.cli.post.helpers._locate_checkpoint", return_value=tmp_path / "ckpt.h5"):
        ok = CliRunner().invoke(export_vtk, ["--case", str(tmp_path), "--high-order", "-o", str(tmp_path / "ho.vtu"),
                                             "--variables", "pressure", "--variables", "k"])
        assert ok.exit_code == 0, ok.output
        grid = pv.read(str(tmp_path / "ho.vtu"))
        assert grid.n_cells == s.mesh.n_cells and 'TurbulentKineticEnergy' in grid.point_data
        bad = CliRunner().invoke(export_vtk, ["--case", str(tmp_path), "--high-order", "-o", str(tmp_path / "x.vtk")])
        assert bad.exit_code != 0 and ".vtu" in bad.output

