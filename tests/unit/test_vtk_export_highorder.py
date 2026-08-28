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

from autoflowcfd.postprocess.vtk_export_highorder import (
    _MAX_SUPPORTED_ORDER,
    _tet_barycentric_to_cube,
    _tet_vtk_node_barycentrics,
    _tri_barycentric_to_cube_ab,
    _wedge_vtk_node_layout,
    export_highorder_vtk,
)
from autoflowcfd.grid.curved_mapping.curved_mapping import (
    cube_to_tet_rst,
    cube_to_tri_rs,
    tet_barycentric,
    tri_barycentric,
)
from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh

GAMMA = 1.4


def _quadratic_field(xyz: np.ndarray) -> np.ndarray:
    """物理空间二次多项式，P=2 下应能被 FR 解精确表示。"""
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    return (
        101325.0 + 50.0 * x - 30.0 * y + 20.0 * z
        + 5.0 * x * y - 3.0 * y * z + 2.0 * x * z
        + 1.5 * x**2 - 0.8 * y**2
    )


class TestBarycentricCubeInversionRoundTrip:
    """独立数值往返验证 _tet_barycentric_to_cube/_tri_barycentric_to_cube_ab
    与本项目自己已验证的正向 Duffy 映射（curved_mapping.py）互为逆映射。"""

    def test_tet_inverse_matches_forward_mapping(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(-0.9, 0.9, 20)
        b = rng.uniform(-0.9, 0.9, 20)
        c = rng.uniform(-0.9, 0.9, 20)
        r, s, t = cube_to_tet_rst(a, b, c)
        L1, L2, L3, L4 = tet_barycentric(r, s, t)
        bary = np.column_stack([L1, L2, L3, L4])

        cube_recovered = _tet_barycentric_to_cube(bary)
        np.testing.assert_allclose(cube_recovered, np.column_stack([a, b, c]), atol=1e-10)

    def test_tri_inverse_matches_forward_mapping(self):
        rng = np.random.default_rng(1)
        a = rng.uniform(-0.9, 0.9, 20)
        b = rng.uniform(-0.9, 0.9, 20)
        r, s = cube_to_tri_rs(a, b)
        l1, l2, l3 = tri_barycentric(r, s)
        tri_bary = np.column_stack([l1, l2, l3])

        ab_recovered = _tri_barycentric_to_cube_ab(tri_bary)
        np.testing.assert_allclose(ab_recovered, np.column_stack([a, b]), atol=1e-10)


class TestVtkNodeLayoutCounts:
    def test_tet_order1_is_just_corners(self):
        bary = _tet_vtk_node_barycentrics(1)
        assert bary.shape == (4, 4)

    def test_tet_order2_has_10_nodes(self):
        bary = _tet_vtk_node_barycentrics(2)
        assert bary.shape == (10, 4)

    def test_tet_order3_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            _tet_vtk_node_barycentrics(3)

    def test_wedge_order1_is_just_corners(self):
        tri_bary, z = _wedge_vtk_node_layout(1)
        assert tri_bary.shape == (6, 3)
        assert z.shape == (6,)

    def test_wedge_order2_has_18_nodes(self):
        tri_bary, z = _wedge_vtk_node_layout(2)
        assert tri_bary.shape == (18, 3)

    def test_wedge_order3_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            _wedge_vtk_node_layout(3)


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
        mesh = _build_synthetic_mixed_mesh(order=3)
        assert mesh.order > _MAX_SUPPORTED_ORDER
        U = np.zeros((mesh.n_cells, mesh.n_sps_per_cell, 5))
        U[..., 0] = 1.0
        U[..., 4] = 1.0
        with pytest.raises(NotImplementedError):
            export_highorder_vtk(mesh, U, tmp_path / "should_not_be_written.vtu")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
