"""VTK 导出的场数据：边界分区编号、单元场组装、单元 -> 节点插值。"""

from typing import Dict, List, Optional

import numpy as np

from ._field_utils import cell_to_node

#: 没有记录边界条件类型的组（外部体网格不带边界三角形时的 'exterior' 等）
_UNKNOWN_TYPE = "UNKNOWN"

#: 湍流单元场的键（取自湍流模型输运场与涡粘的单元积分平均，`core/turbulence/output.py`）
TURBULENCE_FIELDS = ('k', 'omega', 'nu_tilde', 'nut')

#: 可请求的场名（'turbulence' 展开为当前可用的全部湍流量）
VALID_FIELDS = {'velocity', 'pressure', 'turbulence', 'q_criterion'} | set(TURBULENCE_FIELDS)

#: 写进文件的数组名
FIELD_LABELS = {
    'velocity': 'Velocity',
    'pressure': 'Pressure',
    'k': 'TurbulentKineticEnergy',
    'omega': 'SpecificDissipationRate',
    'nu_tilde': 'SANuTilde',
    'nut': 'TurbulentViscosity',
    'q_criterion': 'QCriterion',
}


def expand_fields(fields: Optional[List[str]], turbulence: Dict[str, np.ndarray], valid=VALID_FIELDS) -> List[str]:
    """校验场名并把 'turbulence' 展开为可用的湍流量；请求了 'turbulence' 却没有任何湍流场时报错。
    `valid`：可请求的场名（低阶导出 `VALID_FIELDS`，高阶导出另含 density，见 `vtk_export_highorder`）。"""
    if fields is None:
        fields = ['velocity', 'pressure']
    invalid = set(fields) - set(valid)
    if invalid:
        raise ValueError(f"无效的场名 {sorted(invalid)}，可用：{sorted(valid)}")
    out: List[str] = []
    for name in fields:
        if name == 'turbulence':
            available = [k for k in TURBULENCE_FIELDS if k in turbulence]
            if not available:
                raise ValueError("请求了 'turbulence'，但没有任何湍流场（层流算例，或 checkpoint 早于 turb_cell_* 字段）")
            out.extend(k for k in available if k not in out)
        elif name not in out:
            out.append(name)
    return out


def boundary_zone_ids(grid, tri_conn: np.ndarray):
    """外部三角面 -> (BoundaryID, BoundaryTypeID, id 图例, 类型图例)。

    边界组的面集合由 `mesh_boundary.exterior_faces_by_group` 逐面给出（有面网格数据时按几何包含逐面
    归属），这里按排序后的节点三元组对号。
    """
    from autoflowcfd.grid.mesh_gen.utils.mesh_boundary import exterior_faces_by_group

    faces_by_group = exterior_faces_by_group(grid)
    boundary_names = list(faces_by_group)
    surface = getattr(grid, "surface_mesh", None)
    bc_types = dict(surface["boundaries"].bc_types) if surface is not None else dict(grid.boundaries.bc_types)
    group_type = [str(bc_types.get(name, _UNKNOWN_TYPE)) for name in boundary_names]
    type_names = sorted(set(group_type))

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


def cell_fields(grid, solution, turbulence: Dict[str, np.ndarray], q_criterion: Optional[np.ndarray],
                fields: List[str]) -> Dict[str, np.ndarray]:
    """请求场的单元值（单元顺序：棱柱在前、四面体在后）。缺数据的场报错，不写占位值。"""
    n_cells = int(grid.cell_count)
    out: Dict[str, np.ndarray] = {}
    if 'velocity' in fields or 'pressure' in fields:
        if solution is None or solution.data is None or solution.n_cells != n_cells:
            raise ValueError("没有与网格单元数一致的解数据，无法导出速度/压力")
        if 'velocity' in fields:
            out['velocity'] = np.column_stack(solution.get_velocity())
        if 'pressure' in fields:
            out['pressure'] = solution.get_pressure()
    for key in TURBULENCE_FIELDS:
        if key not in fields:
            continue
        value = turbulence.get(key)
        if value is None or len(value) != n_cells:
            raise ValueError(f"湍流场 '{key}' 不可用（层流、模型不含该量，或 checkpoint 早于 turb_cell_* 字段）")
        out[key] = value
    if 'q_criterion' in fields:
        if q_criterion is None or len(q_criterion) != n_cells:
            raise ValueError("Q 准则需要由求解器状态算出后传入（postprocess/q_criterion.py::q_criterion_cell_field）")
        out['q_criterion'] = np.asarray(q_criterion, dtype=np.float64)
    return out


def point_fields(grid, cell_fields_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """单元场按体积加权平均到节点（混合网格，`_field_utils.cell_to_node`）。"""
    return {name: cell_to_node(grid, arr) for name, arr in cell_fields_data.items()}
