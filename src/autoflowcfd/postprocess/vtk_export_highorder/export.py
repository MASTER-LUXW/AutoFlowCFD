"""AutoFlowCFD V2.0 - 高阶 Lagrange 单元 .vtu 导出本体。

从 `vtk_export_highorder.py` 拆出（2026-09-20）。纯搬家，逻辑未改。
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from .interp_matrices import (
    _build_native_prism_vtk_lagrange_export_data,
    _build_native_tet_vtk_lagrange_export_data,
    _build_vtk_lagrange_export_data,
)
from .node_layout import (
    _MAX_SUPPORTED_ORDER_TET,
    _MAX_SUPPORTED_ORDER_WEDGE,
    _VTK_LAGRANGE_TETRAHEDRON,
    _VTK_LAGRANGE_WEDGE,
    _tet_vtk_node_barycentrics,
)

_FIELD_LABELS = {
    'velocity': 'Velocity',
    'pressure': 'Pressure',
    'density': 'Density',
}


def export_highorder_vtk(
    mesh,
    U: np.ndarray,
    output_path,
    fields: Optional[List[str]] = None,
    binary: bool = True,
) -> Path:
    """把 FR 解场（真实分段多项式，非单元中心平均值）导出为 VTK 高阶
    Lagrange 单元 (.vtu)。见模块文档的完整技术说明。

    Args:
        mesh: HighOrderMesh 实例（需要 order/_ref_cube_sps/n_prism_cells/
            n_cells/_node_coords/_fixed_tet_conn/_fixed_prism_conn，均为
            该类已建立、被 fr/ 模块跨模块访问的内部几何属性）
        U: 守恒变量场，形状 (n_cells, n_sps, n_vars)，n_vars>=5（前 5 个
            分量是 (rho,rho*u,rho*v,rho*w,rho*E)；湍流分量若存在直接
            忽略，本次只导出平均流场）
        output_path: 输出 .vtu 路径
        fields: 要导出的场，默认 ['velocity', 'pressure']，可选值见
            `_FIELD_LABELS`
        binary: 是否用二进制+压缩写入（pyvista/VTK 默认行为）

    Returns:
        输出文件路径

    Raises:
        NotImplementedError: mesh.order > _MAX_SUPPORTED_ORDER_TET（3，
            四面体侧的决定性验证上限，见 _tet_vtk_node_barycentrics
            文档），或网格含棱柱单元且 mesh.order >
            _MAX_SUPPORTED_ORDER_WEDGE（2，见 _wedge_vtk_node_layout
            文档）。
    """
    import pyvista as pv

    from ...core.fr_residual.inviscid import conserved_to_primitive
    from ...grid.curved_mapping.curved_mapping import map_prism_to_physical, tet_barycentric
    from ...fr.native_tet.basis import build_native_tet_operators

    order = mesh.order
    n_prism_check = mesh.n_prism_cells
    n_tet_check = mesh.n_cells - n_prism_check
    if n_tet_check > 0 and order > _MAX_SUPPORTED_ORDER_TET:
        raise NotImplementedError(
            f"export_highorder_vtk：四面体高阶导出目前只支持 order<="
            f"{_MAX_SUPPORTED_ORDER_TET}（见 _tet_vtk_node_barycentrics "
            f"文档），当前网格 order={order}。"
        )
    if n_prism_check > 0 and order > _MAX_SUPPORTED_ORDER_WEDGE:
        raise NotImplementedError(
            f"export_highorder_vtk：网格含 {n_prism_check} 个棱柱单元，"
            f"棱柱/wedge 高阶导出目前只支持 order<={_MAX_SUPPORTED_ORDER_WEDGE}"
            f"（见 _wedge_vtk_node_layout 文档），当前网格 order={order}。"
            f"四面体单元本身可以用到 order={_MAX_SUPPORTED_ORDER_TET}——"
            f"如果这个网格没有棱柱单元就不会触发这个限制。"
        )
    if fields is None:
        fields = ['velocity', 'pressure']
    unknown = set(fields) - set(_FIELD_LABELS)
    if unknown:
        raise ValueError(f"Unknown fields: {unknown}. Valid: {sorted(_FIELD_LABELS)}")

    output_path = Path(output_path)
    if not output_path.suffix:
        output_path = output_path.with_suffix('.vtu')

    Q = conserved_to_primitive(U[..., :5])  # (n_cells, n_sps, 5)

    ref_cube_sps = mesh._ref_cube_sps
    n_prism = mesh.n_prism_cells
    n_cells = mesh.n_cells

    field_component = {
        'velocity': lambda q: q[..., 1:4],
        'pressure': lambda q: q[..., 4:5],
        'density': lambda q: q[..., 0:1],
    }

    all_points: List[np.ndarray] = []
    field_values: Dict[str, List[np.ndarray]] = {name: [] for name in fields}
    cells_flat: List[int] = []
    cell_types: List[int] = []
    degrees_rows: List[np.ndarray] = []

    node_offset = 0

    # --- 棱柱：按棱柱基分派（2026-09-20）---
    # 坍缩档走坍缩模态基 + `map_prism_to_physical`；原生档走原生模态基 +
    # `map_native_prism_to_physical`，且场值只取前 `n_native` 行（零填充
    # 槽位不携带真实场值）。两条的理由见
    # `_build_native_prism_vtk_lagrange_export_data` 文档。
    if n_prism > 0:
        # 棱柱恒为原生基（坍缩档已于 2026-09-23 删除）。**曾经的真实缺陷**：
        # 这里一度无条件把插值矩阵拟合在坍缩节点上，对原生基**不报错、
        # 静默给错值**（节点值相对误差 1.66e-2，2026-09-20 修复）。
        from ...fr.native_prism.basis import (
            map_native_prism_to_physical, native_prism_n_sps,
        )

        n_real_prism = native_prism_n_sps(order)
        target_ref, E = _build_native_prism_vtk_lagrange_export_data(order)
        prism_map = map_native_prism_to_physical
        n_vtk_nodes = target_ref.shape[0]
        for local_i in range(n_prism):
            global_cell = local_i
            cell_nodes_phys = mesh._node_coords[mesh._fixed_prism_conn[local_i]]
            phys_pts = prism_map(target_ref, cell_nodes_phys)
            all_points.append(phys_pts)

            q_at_sps = Q[global_cell]  # (n_sps, 5)
            if n_real_prism is not None:
                q_at_sps = q_at_sps[:n_real_prism]
            for name in fields:
                comp = field_component[name](q_at_sps)  # (n_sps, k)
                field_values[name].append(E @ comp)

            node_ids = np.arange(node_offset, node_offset + n_vtk_nodes)
            cells_flat.append(n_vtk_nodes)
            cells_flat.extend(node_ids.tolist())
            cell_types.append(_VTK_LAGRANGE_WEDGE)
            degrees_rows.append([float(order), float(order), float(order)])
            node_offset += n_vtk_nodes

    # --- 四面体：native 单纯形基插值（2026-09-03 更正，见
    # `_build_native_tet_vtk_lagrange_export_data` 文档"2026-09-03 更正"
    # 一节）——物理坐标直接用重心坐标仿射组合，不经过参考立方体；场值
    # 插值矩阵宽度是 `n_native`（真实自由度数），不是全局 padded 宽度，
    # 必须对 `q_at_sps` 先按 `[:n_native]` 切片，填充行不携带真实场值。
    n_tet = n_cells - n_prism
    if n_tet > 0:
        ref_rst, _ = build_native_tet_operators(order)
        n_native = ref_rst.shape[0]
        target_rst, E = _build_native_tet_vtk_lagrange_export_data(order)
        n_vtk_nodes = target_rst.shape[0]
        r_t, s_t, t_t = target_rst[:, 0], target_rst[:, 1], target_rst[:, 2]
        L0, L1, L2, L3 = tet_barycentric(r_t, s_t, t_t)

        for local_i in range(n_tet):
            global_cell = n_prism + local_i
            p0, p1, p2, p3 = mesh._node_coords[mesh._fixed_tet_conn[local_i]]
            phys_pts = (
                L0[:, None] * p0 + L1[:, None] * p1 + L2[:, None] * p2 + L3[:, None] * p3
            )
            all_points.append(phys_pts)

            q_at_native_sps = Q[global_cell, :n_native]  # (n_native, 5)——只取真实自由度
            for name in fields:
                comp = field_component[name](q_at_native_sps)  # (n_native, k)
                field_values[name].append(E @ comp)

            node_ids = np.arange(node_offset, node_offset + n_vtk_nodes)
            cells_flat.append(n_vtk_nodes)
            cells_flat.extend(node_ids.tolist())
            cell_types.append(_VTK_LAGRANGE_TETRAHEDRON)
            degrees_rows.append([float(order), float(order), float(order)])
            node_offset += n_vtk_nodes

    points = np.concatenate(all_points, axis=0)
    grid = pv.UnstructuredGrid(np.array(cells_flat, dtype=np.int64), np.array(cell_types, dtype=np.uint8), points)
    grid.cell_data['HigherOrderDegrees'] = np.array(degrees_rows, dtype=np.float64)

    for name in fields:
        vals = np.concatenate(field_values[name], axis=0)
        label = _FIELD_LABELS[name]
        grid.point_data[label] = vals[:, 0] if vals.shape[1] == 1 else vals

    grid.save(str(output_path), binary=binary)
    logger.success(f"High-order VTK Lagrange cells exported: {output_path}")
    return output_path
