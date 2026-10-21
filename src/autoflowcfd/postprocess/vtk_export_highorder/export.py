"""AutoFlowCFD V2.0 - 高阶 Lagrange 单元 .vtu 导出本体。

每个单元导出为一个 VTK 高阶 Lagrange 单元（四面体 71、三棱柱 73），节点值是 FR 解多项式在 VTK 等距节点上的
精确取值（`interp_matrices.py`）。全部单元共用同一组形函数/插值矩阵，坐标、场值与连接表整体向量化构造。

可导出的场：平均流（速度、压力、密度）与解点上的附加场 `sp_fields`（湍流量 k/omega/nu_tilde/nut、Q 准则，
场名与低阶导出相同，`postprocess/vtk_export_fields.py`）。
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from loguru import logger

from ..vtk_export_fields import FIELD_LABELS, TURBULENCE_FIELDS, expand_fields
from .interp_matrices import native_prism_export_data, native_tet_export_data
from .node_layout import (
    _MAX_SUPPORTED_ORDER_TET,
    _MAX_SUPPORTED_ORDER_WEDGE,
    _VTK_LAGRANGE_TETRAHEDRON,
    _VTK_LAGRANGE_WEDGE,
)

#: 平均流场 -> 原始变量列
_MEAN_FLOW = {'velocity': slice(1, 4), 'pressure': slice(4, 5), 'density': slice(0, 1)}
#: 写进文件的数组名
_LABELS = {**FIELD_LABELS, 'density': 'Density'}
#: 可请求的场名（'turbulence' 展开为 sp_fields 里可用的全部湍流量）
VALID_HIGHORDER_FIELDS = set(_MEAN_FLOW) | {'turbulence', 'q_criterion'} | set(TURBULENCE_FIELDS)


def export_highorder_vtk(
    mesh,
    U: np.ndarray,
    output_path,
    fields: Optional[List[str]] = None,
    sp_fields: Optional[Dict[str, np.ndarray]] = None,
    binary: bool = True,
) -> Path:
    """把 FR 解场（真实分段多项式）导出为 VTK 高阶 Lagrange 单元 (.vtu)。

    Args:
        mesh: HighOrderMesh（order、n_prism_cells、n_cells、_node_coords、_fixed_prism_conn、_fixed_tet_conn）
        U: 守恒变量 `(n_cells, n_sps, >=5)`
        output_path: 输出路径（无后缀时补 .vtu）
        fields: 要导出的场（默认速度、压力），取值见 `VALID_HIGHORDER_FIELDS`
        sp_fields: 解点上的附加场 `{场名: (n_cells, n_sps)}`（`core/turbulence/output.py::turbulence_sp_fields`、
            `postprocess/q_criterion.py::q_criterion_sp_field`）；请求了其中的场却没有给出时报错
        binary: 二进制写出

    Raises:
        NotImplementedError: 阶数超过已验证的节点编号（`node_layout.py`）
    """
    import pyvista as pv

    from ...core.fr_residual.inviscid import conserved_to_primitive
    from ...fr.native_padding import real_sps_per_cell

    sp_fields = dict(sp_fields or {})
    names = expand_fields(fields, sp_fields, valid=VALID_HIGHORDER_FIELDS)
    missing = [n for n in names if n not in _MEAN_FLOW and n not in sp_fields]
    if missing:
        raise ValueError(f"请求的场 {missing} 没有提供解点值（sp_fields 只有 {sorted(sp_fields)}）")

    order = int(mesh.order)
    n_prism, n_cells = int(mesh.n_prism_cells), int(mesh.n_cells)
    n_tet = n_cells - n_prism
    if n_tet > 0 and order > _MAX_SUPPORTED_ORDER_TET:
        raise NotImplementedError(f"四面体高阶导出只验证到 order={_MAX_SUPPORTED_ORDER_TET}，当前 order={order}")
    if n_prism > 0 and order > _MAX_SUPPORTED_ORDER_WEDGE:
        raise NotImplementedError(f"三棱柱高阶导出只验证到 order={_MAX_SUPPORTED_ORDER_WEDGE}，当前 order={order}")

    output_path = Path(output_path)
    if not output_path.suffix:
        output_path = output_path.with_suffix('.vtu')

    Q = conserved_to_primitive(np.asarray(U)[..., :5])
    values = {n: Q[..., _MEAN_FLOW[n]] if n in _MEAN_FLOW else np.asarray(sp_fields[n])[..., None] for n in names}
    npr, nte = real_sps_per_cell(order)
    nodes = np.asarray(mesh._node_coords)

    points: list = []
    cells: list = []
    types: list = []
    data: Dict[str, list] = {n: [] for n in names}
    offset = 0
    for count, lo, n_real, conn, build, vtk_type in (
            (n_prism, 0, npr, mesh._fixed_prism_conn, native_prism_export_data, _VTK_LAGRANGE_WEDGE),
            (n_tet, n_prism, nte, mesh._fixed_tet_conn, native_tet_export_data, _VTK_LAGRANGE_TETRAHEDRON)):
        if count == 0:
            continue
        N, E = build(order)                                     # (n_vtk, n_vertices), (n_vtk, n_real)
        n_vtk = N.shape[0]
        points.append(np.einsum('nv,cvd->cnd', N, nodes[np.asarray(conn)]).reshape(-1, 3))
        for n in names:
            data[n].append(np.einsum('ns,csk->cnk', E, values[n][lo:lo + count, :n_real]).reshape(count * n_vtk, -1))
        ids = offset + np.arange(count * n_vtk, dtype=np.int64).reshape(count, n_vtk)
        cells.append(np.column_stack([np.full(count, n_vtk, dtype=np.int64), ids]).ravel())
        types.append(np.full(count, vtk_type, dtype=np.uint8))
        offset += count * n_vtk

    grid = pv.UnstructuredGrid(np.concatenate(cells), np.concatenate(types), np.concatenate(points))
    # 必须显式给出各方向阶数：VTK 无法从节点数可靠推断（三棱柱缺这个数组时 VTK 底层直接崩溃）
    grid.cell_data['HigherOrderDegrees'] = np.full((n_cells, 3), float(order))
    for n in names:
        vals = np.concatenate(data[n])
        grid.point_data[_LABELS[n]] = vals[:, 0] if vals.shape[1] == 1 else vals
    grid.save(str(output_path), binary=binary)
    logger.success(f"High-order VTK Lagrange cells exported: {output_path}")
    return output_path


def export_solver_highorder_vtk(solver, output_path, fields: Optional[List[str]] = None, binary: bool = True) -> Path:
    """求解器当前状态的高阶导出（API `export_vtk(high_order=True)` 与 CLI `post export --high-order` 共用）：
    湍流量取湍流模型的解点场，请求了 Q 准则时按解点速度梯度计算。"""
    from ...core.turbulence.output import turbulence_sp_fields

    host = solver.host_view() if hasattr(solver, "host_view") else solver
    sp_fields = turbulence_sp_fields(getattr(host, "turb_model", None))
    if fields is not None and 'q_criterion' in fields:
        from ..q_criterion import q_criterion_sp_field

        sp_fields['q_criterion'] = q_criterion_sp_field(host)
    return export_highorder_vtk(host.mesh, host.state.U, output_path, fields=fields, sp_fields=sp_fields,
                                binary=binary)
