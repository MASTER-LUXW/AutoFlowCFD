"""低阶 VTK 导出：单元场（CELL_DATA）+ 体积加权节点场（POINT_DATA），legacy（.vtk）或 XML（.vtu）。

- 每个场同时写 CELL_DATA（求解器单元积分平均，不经插值，保留局部极值）与 POINT_DATA（混合网格
  体积加权节点平均，供平滑着色/等值面）。
- legacy 支持 big-endian 二进制载荷；XML 委托 VTK 自己的写入器（binary 时 zlib 压缩，.vtu 默认）。
- 湍流量取自湍流模型输运场与涡粘的单元积分平均（`core/turbulence/output.py::turbulence_cell_means`，
  checkpoint 里是 `turb_cell_*` 字段）；Q 准则由求解器状态算出后传入（`postprocess/q_criterion.py`）。
- `export_boundaries()` 只导出命名边界面片：BoundaryID / BoundaryTypeID（int32，名称图例作为 field data
  嵌入；legacy 二进制无法嵌入字符串 FIELD 块时图例记日志）+ owner 单元的精确场值。
- 高阶（拉格朗日单元）导出见 `vtk_export_highorder/`。

Example:
    >>> exporter = VTKExporter(grid_data, solution, turbulence=turbulence_cell_means(model, W))
    >>> exporter.export("output.vtk", binary=True)
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from loguru import logger

from ..core.backend.base import SolutionVector
from . import vtk_export_legacy, vtk_export_xml
from .vtk_export_fields import boundary_zone_ids, cell_fields, expand_fields, point_fields

_WRITERS = {'legacy': (vtk_export_legacy, '.vtk', False), 'xml': (vtk_export_xml, '.vtu', True)}


def _writer(fmt: str):
    if fmt not in _WRITERS:
        raise ValueError(f"不支持的格式 {fmt!r}（'legacy' 或 'xml'）")
    return _WRITERS[fmt]


class VTKExporter:
    """低阶 VTK 导出器。

    Args:
        grid_data: 体网格（VolumeMeshData；棱柱在前、四面体在后）或面网格 GridData
        solution: 单元积分平均的守恒解（`SolutionVector`，单元顺序同网格）
        turbulence: 湍流单元场 `{键: (n_cells,)}`（键 k / omega / nu_tilde / nut）
        q_criterion: Q 准则单元场 `(n_cells,)`（请求 'q_criterion' 时必须给出）
    """

    def __init__(self, grid_data, solution: Optional[SolutionVector], turbulence: Optional[Dict[str, np.ndarray]] = None,
                 q_criterion: Optional[np.ndarray] = None):
        self.grid_data = grid_data
        self.solution = solution
        self.turbulence = {k: np.asarray(v, dtype=np.float64) for k, v in (turbulence or {}).items()}
        self.q_criterion = q_criterion

    def export(self, output_path: str, fields: Optional[List[str]] = None, format: str = 'legacy',
               binary: Optional[bool] = None) -> Path:
        """导出体网格场。`binary` 缺省时 legacy 写 ASCII、xml 写二进制+压缩。"""
        module, suffix, default_binary = _writer(format)
        names = expand_fields(fields, self.turbulence)
        path = Path(output_path)
        if not path.suffix:
            path = path.with_suffix(suffix)
        cells = cell_fields(self.grid_data, self.solution, self.turbulence, self.q_criterion, names)
        module.export_volume(path, self.grid_data, cells, point_fields(self.grid_data, cells),
                             default_binary if binary is None else bool(binary))
        logger.success(f"VTK 文件已导出：{path}")
        return path

    def export_boundaries(self, output_path: str, fields: Optional[List[str]] = None, format: str = 'legacy',
                          binary: Optional[bool] = None) -> Path:
        """只导出命名边界面片（外部三角面），每面带 BoundaryID/BoundaryTypeID 与 owner 单元的场值。

        需要体网格（VolumeMeshData：命名边界组 + 面提取）；面片上不做节点平均（节点被内部网格共享，
        仅边界的节点平均有歧义），只携带逐面精确值。
        """
        if not hasattr(self.grid_data, 'ensure_faces_exist'):
            raise ValueError("export_boundaries 需要体网格（VolumeMeshData），不是面网格 GridData")
        module, suffix, default_binary = _writer(format)
        names = expand_fields(fields, self.turbulence)
        faces = self.grid_data.ensure_faces_exist()
        if faces.node_connectivity is None:
            raise RuntimeError("面数据没有 node_connectivity（旧版本 .pkl 网格缓存），请重新生成体网格")
        bidx = faces.get_boundary_face_indices()
        owners = faces.connectivity[bidx, 0].astype(np.int64)
        tri_conn = faces.node_connectivity[bidx]
        boundary_id, type_id, id_legend, type_legend = boundary_zone_ids(self.grid_data, tri_conn)
        cells = cell_fields(self.grid_data, self.solution, self.turbulence, self.q_criterion, names)
        path = Path(output_path)
        if not path.suffix:
            path = path.with_suffix(suffix)
        module.export_boundaries(path, self.grid_data, tri_conn, {k: v[owners] for k, v in cells.items()},
                                 boundary_id, type_id, id_legend, type_legend,
                                 default_binary if binary is None else bool(binary))
        logger.success(f"VTK 边界面片已导出：{path}")
        return path
