"""core/fr_operators/troubled_cell.py 的几何退化诊断单元测试。

本文件原先还覆盖"机制3"（`suppress_residual_outliers` 与它内部的
`_median_abs_over_sps_kernel` 中位数 kernel）。**机制3 已于 2026-09-19
整体删除**（真实网格消融对照证明它触发了但只把残差轨迹改变 ~1e-10
相对量、不改变发散结局，完整记录见 `core/fr_residual/inviscid.py`），
对应的两个测试类随之删除。

留下的是 `precompute_cell_face_misalignment`（机制2 的面法向失配
诊断量）的独立 oracle 对照 —— 机制1/2 只产出诊断量与日志，不干预
残差，所以它们的测试与机制3 的删除无关。
"""

import numpy as np
import pytest

from autoflowcfd.core.fr_operators.troubled_cell import (
    precompute_cell_face_misalignment,
)


def _reference_cell_face_misalignment(mesh):
    """`precompute_cell_face_misalignment` 的 numba 核的独立逐面 Python 循环
    参照实现（它防的真实性能缺陷：对每个面取 `mesh.face_flux_points[f]` 每次
    访问都会惰性*构造*一个完整的 FaceFluxPointGeometry 对象——生产网格上
    188 万次这样的构造实测要几分钟，修法见
    `_cell_face_misalignment_kernel` 的文档）。

    2026-08-23 更新（见 fr/face_flux_points/exact_normal.py 模块文档）：
    `own_dir_outward` 此前在这里由自己独立的一份解点网格 Lagrange 外插从
    `adj_j` 算出——是真实局部度量方向的*近似*，有真实的截断误差（低阶更差）。
    生产核现在读取网格加载时预先算好的逐通量点精确 adj(J) 行，不再外插——
    这份参照实现必须用同样的精确值（仍在这里独立计算，经
    `compute_exact_adj_rows` 而不是核自己预先算好的数组，所以对*归约循环*
    的逻辑它仍是真正的参照而不是同义反复），否则就是拿新核去和一个过时、
    精度更低、算的还是另一个量的近似比较。
    """
    from autoflowcfd.fr.face_flux_points.exact_normal import compute_exact_adj_rows

    fc = mesh.face_connectivity
    ffp_list = mesh.face_flux_points
    n_prism = mesh.n_prism_cells
    n1d = mesh.n_points_1d
    sps_1d = ffp_list._sps_1d

    # native 四面体（路径C）：四面体坍缩坐标基删除后合成测试网格默认
    # 即为 native（2026-09-03），必须传 `code_arr` 让 `compute_exact_
    # adj_rows` 按 `owner_cube_face`/`neighbor_cube_face>=6` 分派到它
    # 已有的 native 分支（`_native_tet_adj_row_batched`）——不传的话
    # `axis_arr`（对 native 面是复用槽位的 excluded_vertex）会与坍缩
    # 坐标 (axis,side) 语义发生数值碰撞，见该函数 `code_arr` 参数文档
    # "真实 bug 修复"一节。
    owner_adj_row = compute_exact_adj_rows(
        fc.n_faces, n1d, sps_1d, n_prism,
        cell_arr=fc.owner_cell.astype(np.int64),
        axis_arr=ffp_list.owner_axis.astype(np.int64),
        side_arr=ffp_list.owner_side.astype(np.float64),
        prism_conn=mesh._fixed_prism_conn if mesh._fixed_prism_conn is not None else np.empty((0, 6), dtype=np.int64),
        tet_conn=mesh._fixed_tet_conn if mesh._fixed_tet_conn is not None else np.empty((0, 4), dtype=np.int64),
        node_coords=mesh._node_coords,
        code_arr=fc.owner_cube_face.astype(np.int64),
    )
    neighbor_adj_row = compute_exact_adj_rows(
        fc.n_faces, n1d, sps_1d, n_prism,
        cell_arr=fc.neighbor_cell.astype(np.int64),
        axis_arr=ffp_list.neighbor_axis.astype(np.int64),
        side_arr=ffp_list.neighbor_side.astype(np.float64),
        prism_conn=mesh._fixed_prism_conn if mesh._fixed_prism_conn is not None else np.empty((0, 6), dtype=np.int64),
        tet_conn=mesh._fixed_tet_conn if mesh._fixed_tet_conn is not None else np.empty((0, 4), dtype=np.int64),
        node_coords=mesh._node_coords,
        valid_mask=~fc.is_boundary,
        code_arr=fc.neighbor_cube_face.astype(np.int64),
    )

    def own_dir(row, side):
        mag = np.linalg.norm(row, axis=-1)
        return (row / np.maximum(mag[:, None], 1e-300)) * side

    # native 四面体（路径C）同一处修复（2026-09-03，四面体坍缩坐标基
    # 删除后合成测试网格默认即为 native，此前只有显式请求 native 时才
    # 会走到这个分支，这个参考实现从未被真正练到过）：`ffp.owner_side`/
    # `ffp.neighbor_side` 对 native 面是复用槽位哑值（恒为 -1.0），不
    # 代表真正的定向语义——与 `_cell_face_misalignment_kernel` 生产
    # 实现同一处修复同一个理由（见该函数文档"native 四面体...真实
    # bug 修复"一节）：native 面（`owner_cube_face>=6`）的 side 因子
    # 固定为 +1，不使用复用槽位值。
    cell_misalign = np.zeros(mesh.n_cells)
    for f in range(fc.n_faces):
        ffp = ffp_list[f]
        if ffp.owner_is_primary:
            owner_cell = int(fc.owner_cell[f])
            side_o = 1.0 if int(fc.owner_cube_face[f]) >= 6 else ffp.owner_side
            d = own_dir(owner_adj_row[f], side_o)
            misalign = 1.0 - np.sum(d * ffp.true_normal, axis=-1)
            cell_misalign[owner_cell] = max(cell_misalign[owner_cell], float(misalign.max()))
        if (not fc.is_boundary[f]) and ffp.neighbor_is_primary:
            neighbor_cell = int(fc.neighbor_cell[f])
            side_n = 1.0 if int(fc.neighbor_cube_face[f]) >= 6 else ffp.neighbor_side
            d = own_dir(neighbor_adj_row[f], side_n)
            misalign = 1.0 - np.sum(d * (-ffp.true_normal), axis=-1)
            cell_misalign[neighbor_cell] = max(cell_misalign[neighbor_cell], float(misalign.max()))
    return cell_misalign


class TestPrecomputeCellFaceMisalignment:
    @pytest.mark.parametrize("order", [0, 1, 2, 3])
    def test_matches_per_face_reference_loop(self, order):
        from .test_fr_residual_inviscid import _build_synthetic_mixed_mesh

        mesh = _build_synthetic_mixed_mesh(order)
        actual = precompute_cell_face_misalignment(mesh)
        expected = _reference_cell_face_misalignment(mesh)
        np.testing.assert_allclose(actual, expected, atol=1e-10)
