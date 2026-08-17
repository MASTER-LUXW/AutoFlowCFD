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


def boundary_zone_ids(exporter, owner_cells: np.ndarray):
    """把每个边界面的 owner 四面体映射到 BoundaryID（按边界组名）
    和 BoundaryTypeID（按名称模式匹配的分类桶）。

    Returns:
        (boundary_id, type_id, id_legend, type_legend)——前两个是
        (n_boundary_faces,) 的 int32 数组，对照表是 "<id>=<name>"
        形式的 List[str]。
    """
    boundary_names = exporter.grid_data.boundaries.boundary_names
    name_to_id = {name: i for i, name in enumerate(boundary_names)}
    type_to_id = {t: i for i, t in enumerate(exporter._BC_TYPE_NAMES)}
    unclassified_id = len(boundary_names)

    def _classify(name: str) -> str:
        """基于名称模式匹配进行边界类型分类"""
        name_upper = name.upper()
        if any(prefix in name_upper for prefix in ['WALL', 'SOLID']):
            return 'WALL'
        elif 'GROUND' in name_upper:
            return 'GROUND'
        elif any(prefix in name_upper for prefix in ['INLET', 'INTAKE', 'ENTRY']):
            return 'INLET'
        elif any(prefix in name_upper for prefix in ['OUTLET', 'OUTFLOW', 'EXIT']):
            return 'OUTLET'
        elif 'SYM' in name_upper or 'MIRROR' in name_upper:
            return 'SYMMETRY'
        elif any(prefix in name_upper for prefix in ['FARFIELD', 'FAR_FIELD', 'BOUNDARY']):
            return 'FARFIELD'
        else:
            return 'WALL'

    # 向量化的 单元 -> id 查找：构建一个按单元 id 索引的稠密数组
    # （哨兵值 = unclassified/WALL），而不是用 Python 字典 + 逐
    # owner 单元的列表推导 + .get() 调用——对真实网格 owner_cells
    # 可能有 1e5-1e6 量级，这样改成了一次花式索引 gather。
    n_cells = exporter.grid_data.cell_count
    cell_to_name_id = np.full(n_cells, unclassified_id, dtype=np.int32)
    cell_to_type_id = np.full(n_cells, type_to_id['WALL'], dtype=np.int32)
    for name in boundary_names:
        btype = _classify(name)
        nid = name_to_id[name]
        tid = type_to_id.get(btype, type_to_id['WALL'])
        cells = np.asarray(exporter.grid_data.boundaries.get_cell_indices(name), dtype=np.int64)
        cell_to_name_id[cells] = nid
        cell_to_type_id[cells] = tid

    boundary_id = cell_to_name_id[owner_cells]
    type_id = cell_to_type_id[owner_cells]

    id_legend = [f"{i}={name}" for name, i in sorted(name_to_id.items(), key=lambda kv: kv[1])]
    if np.any(boundary_id == unclassified_id):
        n_unclassified = int(np.sum(boundary_id == unclassified_id))
        logger.warning(
            f"{n_unclassified} boundary faces have no matching boundary "
            f"group; tagged BoundaryID={unclassified_id} (<UNCLASSIFIED>)"
        )
        id_legend.append(f"{unclassified_id}=<UNCLASSIFIED>")
    type_legend = [f"{i}={name}" for name, i in sorted(type_to_id.items(), key=lambda kv: kv[1])]

    return boundary_id, type_id, id_legend, type_legend


def cell_fields(exporter, fields: List[str]) -> Dict[str, np.ndarray]:
    """在单元中心分辨率上计算每个请求的场（标量 (n_cells,)，矢量
    (n_cells, 3)），解数据不可用时（例如空的 SolutionVector）套用
    与旧的纯节点写入器相同的兜底常数。

    Returns:
        场名（'velocity'、'pressure'、'k'、'omega'、'nut'）到其原始
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

    need_turb = 'k' in fields or 'omega' in fields or 'nut' in fields
    if need_turb:
        k = omega = np.array([])
        if has_data:
            k, omega = exporter.solution.get_turbulence()
            if len(k) == 0:
                logger.warning(
                    "Solution has no turbulence columns (need >=7 variables); "
                    "writing zero for k/omega/nut"
                )
        k_out = k if len(k) == n_cells else np.full(n_cells, 0.0)
        omega_out = omega if len(omega) == n_cells else np.full(n_cells, 0.0)

        if 'k' in fields:
            out['k'] = k_out
        if 'omega' in fields:
            out['omega'] = omega_out
        if 'nut' in fields:
            if exporter.mu_t is not None and len(exporter.mu_t) == n_cells and has_data:
                rho = np.maximum(exporter.solution.get_density(), 1e-10)
                out['nut'] = exporter.mu_t / rho
            elif len(k_out) > 0 and np.any(omega_out > 0):
                logger.warning(
                    "Exact solver mu_t not available (checkpoint predates "
                    "extra_fields support, or turbulence disabled); "
                    "'nut' is the simplified nu_t = k/omega estimate, "
                    "not the actual SST-blended, a1-limited eddy "
                    "viscosity the solver used."
                )
                out['nut'] = k_out / np.maximum(omega_out, 1e-10)
            else:
                out['nut'] = np.zeros(n_cells)

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
