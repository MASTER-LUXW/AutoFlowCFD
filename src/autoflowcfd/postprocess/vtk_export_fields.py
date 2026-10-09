"""VTKExporter 的场数据计算辅助函数。

从 vtk_export.py 中拆分出来（该文件超过 400 行硬性拆分阈值）：边界
分区分类（_boundary_zone_ids）和单元中心/节点场数据计算
（_cell_fields/_cell_to_node/_point_fields）是相对独立的一组逻辑，
被 legacy 和 xml 两种写入路径共用。原来的 `VTKExporter` 方法体原样
搬到这里，改写成以 `exporter`（原来的 `self`）为第一个参数的模块级
函数；`VTKExporter` 上仍保留同名方法作为薄委托包装，外部调用方
（包括 `exporter._cell_fields(...)` 这类直接访问）行为不变。
"""

import numpy as np
from typing import Dict, List

from loguru import logger


#: 没有记录边界条件类型的组（外部体网格不带边界三角形时的 'exterior' 等）
_UNKNOWN_TYPE = "UNKNOWN"


def boundary_zone_ids(exporter, tri_conn: np.ndarray):
    """每个外部面三角形的 BoundaryID（边界组）与 BoundaryTypeID（该组的边界条件类型）。

    逐面定组，与体网格导出、格式转换同一个函数（`mesh_boundary.exterior_faces_by_group`）；类型取网格记录的
    `bc_types`（求解器用的就是它）。2026-10-09 以前按 owner 单元定组：同时贴着两个边界的角点单元，它的面
    全部标成后遍历到的那个组；类型则是在这里按组名另猜一遍，与求解器实际使用的类型可以不同。

    Args:
        tri_conn: (n_boundary_faces, 3) 外部面的节点编号

    Returns:
        (boundary_id, type_id, id_legend, type_legend)——前两个是 (n_boundary_faces,) 的 int32 数组，对照表是
        "<id>=<name>" 形式的 List[str]。

    Raises:
        ValueError: 有外部面不属于任何边界组
    """
    from autoflowcfd.grid.mesh_gen.utils.mesh_boundary import exterior_faces_by_group

    grid = exporter.grid_data
    faces_by_group = exterior_faces_by_group(grid)
    boundary_names = list(faces_by_group)
    surface = getattr(grid, "surface_mesh", None)
    bc_types = dict(surface["boundaries"].bc_types) if surface is not None else dict(grid.boundaries.bc_types)
    group_type = [str(bc_types.get(name, _UNKNOWN_TYPE)) for name in boundary_names]
    type_names = sorted(set(group_type))

    # 外部面按排序后的节点三元组对号（两边来自同一张网格的同一套面提取）
    n_nodes = int(grid.node_count)

    def _key(tris):
        t = np.sort(np.asarray(tris, dtype=np.int64), axis=1)
        return (t[:, 0] * n_nodes + t[:, 1]) * n_nodes + t[:, 2]

    keys = np.concatenate([_key(faces_by_group[name]) for name in boundary_names])
    ids = np.concatenate([np.full(len(faces_by_group[name]), gi, dtype=np.int32)
                          for gi, name in enumerate(boundary_names)])
    order = np.argsort(keys)
    keys, ids = keys[order], ids[order]
    query = _key(tri_conn)
    pos = np.minimum(np.searchsorted(keys, query), len(keys) - 1)
    found = keys[pos] == query
    if not found.all():
        raise ValueError(f"{int((~found).sum())}/{len(query)} 个外部面不属于任何边界组")
    boundary_id = ids[pos]
    type_id = np.asarray([type_names.index(t) for t in group_type], dtype=np.int32)[boundary_id]

    id_legend = [f"{i}={name}" for i, name in enumerate(boundary_names)]
    type_legend = [f"{i}={name}" for i, name in enumerate(type_names)]

    return boundary_id, type_id, id_legend, type_legend


def cell_fields(exporter, fields: List[str]) -> Dict[str, np.ndarray]:
    """在单元中心分辨率上计算每个请求的场（标量 (n_cells,)，矢量
    (n_cells, 3)），解数据不可用时（例如空的 SolutionVector）套用
    与旧的纯节点写入器相同的兜底常数。

    Returns:
        场名（'velocity'、'pressure'、湍流键、'q_criterion'）到其原始
        逐单元数组的字典——正是 CELL_DATA 写入的内容，也是
        POINT_DATA 插值的数据源。
    """
    n_cells = exporter.grid_data.cell_count
    has_data = exporter.solution.data is not None and exporter.solution.n_cells > 0
    out: Dict[str, np.ndarray] = {}

    if 'velocity' in fields:
        if has_data:
            u, v, w = exporter.solution.get_velocity()
            out['velocity'] = np.column_stack([u, v, w])
        else:
            logger.warning("Solution data not available. Using zero velocity.")
            out['velocity'] = np.zeros((n_cells, 3))

    if 'pressure' in fields:
        if has_data:
            out['pressure'] = exporter.solution.get_pressure()
        else:
            logger.warning("Solution data not available. Using uniform pressure.")
            out['pressure'] = np.full(n_cells, 101325.0)

    # 湍流量：取自湍流模型的输运场与涡粘的单元平均（`core/turbulence/output.py`），不再从守恒解
    # 的第 6、7 列推算（那是 SST 状态数组里从未更新的历史槽位）
    for key in ('k', 'omega', 'nu_tilde', 'nut'):
        if key not in fields:
            continue
        value = exporter.turbulence.get(key)
        if value is not None and len(value) == n_cells:
            out[key] = value
        else:
            logger.warning(f"Turbulence field '{key}' not available (laminar case, model without it, or "
                           f"checkpoint predates turb_cell_* fields); writing zeros")
            out[key] = np.zeros(n_cells)

    # Q-Criterion 涡识别准则 (P-02)
    if 'q_criterion' in fields:
        from .q_criterion import compute_q_criterion_from_grid_solution
        q_val = compute_q_criterion_from_grid_solution(exporter.grid_data, exporter.solution)
        if q_val is not None and len(q_val) == n_cells:
            out['q_criterion'] = q_val
        else:
            logger.warning("Q-Criterion computation unavailable; writing zeros")
            out['q_criterion'] = np.zeros(n_cells)

    return out


def cell_to_node(exporter, cell_values: np.ndarray, n_points: int, fallback: float = 0.0) -> np.ndarray:
    """把逐单元标量场插值成逐节点值（对每个节点相连的单元做体积
    加权平均——见 _field_utils.cell_to_node）。"""
    from ._field_utils import cell_to_node as _cell_to_node_impl

    conn = np.asarray(exporter.grid_data.cells.connectivity)
    volumes = getattr(exporter.grid_data.cells, "volumes", None)
    return _cell_to_node_impl(conn, cell_values, n_points, volumes=volumes, fallback=fallback)


def point_fields(exporter, cell_fields_data: Dict[str, np.ndarray], n_points: int) -> Dict[str, np.ndarray]:
    """把 `cell_fields_data` 里每个单元中心场都插值到节点。"""
    out: Dict[str, np.ndarray] = {}
    for name, arr in cell_fields_data.items():
        if arr.ndim == 2:
            fallback = 0.0 if name == 'velocity' else 0.0
            out[name] = np.column_stack([
                exporter._cell_to_node(arr[:, i], n_points, fallback=float(np.mean(arr[:, i])) if len(arr) else 0.0)
                for i in range(arr.shape[1])
            ])
        else:
            fallback = 101325.0 if name == 'pressure' else 0.0
            out[name] = exporter._cell_to_node(arr, n_points, fallback=fallback)
    return out
