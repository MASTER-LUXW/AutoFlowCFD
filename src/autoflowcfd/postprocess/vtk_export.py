"""VTK 场数据导出模块。

本模块提供把 CFD 仿真结果导出为 VTK 格式的工具，供 ParaView 及其它
VTK 兼容查看器可视化使用。

核心组件:
    - VTKExporter: VTK 文件生成的主导出器
    - 支持速度、压力、湍流变量导出

拆分说明（本文件原有 756 行，超过 400 行硬性拆分阈值）：
    - 边界分区分类 + 场数据计算（_boundary_zone_ids/_cell_fields/
      _cell_to_node/_point_fields）已搬到 vtk_export_fields.py
    - legacy VTK (.vtk) ASCII/二进制写入细节已搬到 vtk_export_legacy.py
    - XML VTK (.vtu) 写入细节已搬到 vtk_export_xml.py
    这些方法在 `VTKExporter` 上仍然保留同名薄委托包装（方法体内部
    lazy import 对应模块的同名函数并转发调用），外部调用方
    （包括直接访问 `exporter._write_points(...)` 这类用法，例如
    cli/post/helpers.py 里的 `_export_point_fields_vtk`）行为完全不变。

保真度说明（为什么这不只是旧的简化版导出器）：
    - 每个场都**同时**写 CELL_DATA（求解器实际产生的、未插值的原始
      单元中心值）和 POINT_DATA（体积加权的节点插值，用于平滑等值面
      渲染）——而不是只写 POINT_DATA。主流求解器（Fluent/OpenFOAM/
      STAR-CCM+）都是有限体积/单元中心的，它们的 VTK 系列输出总是
      保留未平滑的逐单元值；只导出 POINT_DATA 会悄悄丢失局部极值
      （例如壁面峰值压力/剪切）。
    - Legacy .vtk 支持 `binary=True` 选项（符合 VTK legacy 规范的
      big-endian 二进制载荷）——对真实的（10 万+单元）工业网格是必需的，
      ASCII 文本既在磁盘上大得多，读写也慢得多。
    - `format='xml'` 是一个真正的写入器（委托给 pyvista/VTK 自己的
      vtkXMLUnstructuredGridWriter），不是以前那个悄悄退化成 legacy 的
      占位实现——带 binary+zlib 压缩的 XML VTU（`binary=True`，xml 的
      默认值）是当前主流 CFD 后处理工具（OpenFOAM 的 foamToVTK、
      ParaView 原生写入器）实际使用的现代标准格式。
    - 湍流量（`k`/`omega`/`nu_tilde`/`nut`）取自湍流模型输运场与涡粘的单元平均
      （`core/turbulence/output.py::turbulence_cell_means`，checkpoint 里是 `turb_cell_*`
      字段）。2026-10-04 以前 `k`/`omega` 读的是守恒解第 6、7 列——SST 状态数组里从未
      更新的历史槽位，导出的永远是初值；`nut` 在缺 `mu_t` 字段时（写 checkpoint 的一方从未
      写过它）退化成用这两列算的估计。
    - `export_boundaries()` 只导出命名的边界面片（WALL/INLET/OUTLET/...
      表面三角形），标记稳定的整数 BoundaryID/BoundaryTypeID（并把
      名称对照表作为 field data 嵌入）——这是 Fluent/OpenFOAM/STAR-CCM+
      使用的按面片划分的工作流程，而不是永远只能看到没有边界身份信息
      的整个体网格。

Example:
    >>> from autoflowcfd.postprocess import VTKExporter
    >>> exporter = VTKExporter(grid_data, solution, turbulence=turbulence_cell_means(model, n_prism, order))
    >>> exporter.export("output.vtk", binary=True)
"""

import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
from loguru import logger

from ..grid.structures import GridData
from ..core.backend.base import SolutionVector

# VTK legacy 单元类型代码（见 VTK 文件格式规范）。
_VTK_TRIANGLE = 5
_VTK_TETRA = 10
_VTK_WEDGE = 13  # 三棱柱——VTK 自己的节点顺序正好与本项目的
                 # (v0,v1,v2,w0,w1,w2) 约定直接一致（两个三角形"底面"
                 # 依次列出），不需要重新排序

#: 湍流输出键（`core/turbulence/output.py`：SST 有 k/omega/nut，SA-neg 有 nu_tilde/nut）。
_TURBULENCE_FIELDS = ('k', 'omega', 'nu_tilde', 'nut')
#: `turbulence` 是"该模型全部湍流输出"的别名，导出时展开（`_expand_fields`）。
_VALID_FIELDS = {'velocity', 'pressure', 'turbulence', 'q_criterion'} | set(_TURBULENCE_FIELDS)


class VTKExporter:
    """VTK 场数据导出器。

    把流场数据导出为 VTK 格式，供 ParaView 可视化。同时支持 legacy VTK
    （ASCII 或二进制）和基于 XML 的 VTK（.vtu，ASCII 或二进制+压缩）
    两种格式，每种格式的每个场都同时带有原始单元中心求解器值
    （CELL_DATA）和节点插值值（POINT_DATA）。

    Attributes:
        grid_data: 网格数据对象
        solution: 流场解向量
        turbulence: 湍流单元场 `{键: (n_cells,)}`（键见 `_TURBULENCE_FIELDS`）。请求了而缺失的
            湍流量写零并记录警告。

    Example:
        >>> exporter = VTKExporter(grid_data, solution, turbulence=turbulence)
        >>> exporter.export("result.vtk", fields=['velocity', 'pressure'], binary=True)
    """

    def __init__(
        self,
        grid_data: GridData,
        solution: SolutionVector,
        turbulence: Optional[Dict[str, np.ndarray]] = None,
    ):
        """初始化 VTK 导出器。

        Args:
            grid_data: 网格数据对象
            solution: 流场解向量
            turbulence: 可选的湍流单元场 `{键: (n_cells,)}`（键见 `_TURBULENCE_FIELDS`），取自
                湍流模型的输运场与涡粘（`core/turbulence/output.py::turbulence_cell_means`，
                checkpoint 里是 `turb_cell_*` 字段）

        Raises:
            ValueError: 网格或解数据无效
        """
        self.grid_data = grid_data
        self.solution = solution
        self.turbulence = {k: np.asarray(v, dtype=np.float64) for k, v in (turbulence or {}).items()}

        logger.info(
            f"VTKExporter initialized:\n"
            f"  Nodes:  {grid_data.metadata.node_count}\n"
            f"  Cells:  {grid_data.metadata.cell_count}"
        )

    def export(
        self,
        output_path: str,
        fields: Optional[List[str]] = None,
        format: str = 'legacy',
        binary: Optional[bool] = None,
    ) -> Path:
        """把流场导出为 VTK 文件。

        Args:
            output_path: 输出文件路径 (.vtk 或 .vtu)
            fields: 要导出的场（默认：全部可用场）
                   可选值：['velocity', 'pressure', 'k', 'omega', 'nut']
            format: VTK 格式（'legacy' 或 'xml'）
            binary: 写二进制载荷而不是 ASCII 文本。默认 'legacy' 为
                False（与以前行为一致），'xml' 为 True（带 binary+zlib
                压缩的 .vtu 是真实网格规模下的标准选择；传 False 强制
                ASCII XML）。

        Returns:
            Path: 导出文件的路径

        Raises:
            ValueError: 格式或场名无效
            IOError: 文件写入错误

        Example:
            >>> path = exporter.export("result.vtk", binary=True)
            >>> print(f"Exported to: {path}")
        """
        fields = self._expand_fields(fields)

        output_path = Path(output_path)

        if format == 'legacy':
            if not output_path.suffix:
                output_path = output_path.with_suffix('.vtk')
            self._export_legacy(output_path, fields, binary=bool(binary))
        elif format == 'xml':
            if not output_path.suffix:
                output_path = output_path.with_suffix('.vtu')
            self._export_xml(output_path, fields, binary=(True if binary is None else binary))
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'legacy' or 'xml'")

        logger.success(f"VTK file exported: {output_path}")
        return output_path

    def export_boundaries(
        self,
        output_path: str,
        fields: Optional[List[str]] = None,
        format: str = 'legacy',
        binary: Optional[bool] = None,
    ) -> Path:
        """只导出命名的边界面片（WALL/INLET/OUTLET/... 表面三角形），
        每个面片带以下标记：

          - BoundaryID（CELL_DATA，int32）：按边界组名稳定分配的分区
            id。id->名称对照表作为 field data 嵌入——一个名为
            'BoundaryID_to_Name' 的 "<id>=<name>" 字符串数组（legacy
            ASCII：DATASET 之后紧跟的一个 FIELD FieldData 块；xml：
            field_data，两者都能在 ParaView 的 Field Data 检查器里读到）
            ——而不是把名称本身当作逐单元字段，因为字符串类型的
            CELL_DATA 无法可靠地经过 VTK 自己的读取器往返（已实测验证：
            数组能列出来，但读回时是 NULL）。整数 id + 对照表正是
            OpenFOAM/Fluent 内部使用的 zone_id + 名称表模式。例外情况：
            legacy 格式 + binary=True 时没有嵌入对照表——VTK 9.3 自己的
            legacy 读取器打不开任何包含字符串 FIELD 块的二进制 .vtk
            文件（已用独立于本写入器的最小复现验证过）；这种情况下
            对照表改为记录日志，BoundaryID 本身不受影响。如果既要二进制
            又要嵌入对照表，优先用 format='xml'（binary+对照表这个组合
            默认就能正常工作）。
          - BoundaryTypeID（CELL_DATA，int32）：该面所属边界组的边界条件
            类型，取网格记录的 `bc_types`（求解器用的就是它）。对照表：
            'BoundaryTypeID_to_Name'。
          - 请求的流场，直接取自每个三角形的 owner 单元（原始、未插值
            的值；这里没有 point-data 处理——仅边界的节点平均在节点被
            内部网格共享时会有歧义，所以这个导出刻意只携带精确的逐面
            值）。

        让你可以在 ParaView 里只打开表面面片，按精确命名的分区或按
        物理角色着色/过滤——这正是 Fluent/OpenFOAM/STAR-CCM+ 使用的
        面片式工作流程——而不是永远只能看到没有边界身份信息的整个
        体网格。

        Args:
            output_path: 输出文件路径 (.vtk 或 .vtu)
            fields: 要导出的场（默认：['velocity', 'pressure']）
            format: VTK 格式（'legacy' 或 'xml'）
            binary: 见 export()；默认值相同。

        Returns:
            Path: 导出文件的路径

        Raises:
            ValueError: 格式/场名无效，或 grid_data 是裸面网格 GridData
                而不是 VolumeMeshData（没有逐四面体边界组/面提取可用来
                推导面片）
        """
        if not hasattr(self.grid_data, 'ensure_faces_exist'):
            raise ValueError(
                "export_boundaries requires a VolumeMeshData grid (named "
                "boundary groups over tetrahedra + face extraction), not "
                "a bare surface GridData."
            )
        fields = self._expand_fields(fields)

        faces = self.grid_data.ensure_faces_exist()
        if faces.node_connectivity is None:
            raise RuntimeError(
                "Face data has no node_connectivity - if this grid came "
                "from a cached volume_mesh.pkl built before boundary "
                "export support was added, regenerate the volume mesh."
            )

        bidx = faces.get_boundary_face_indices()
        owner_cells = faces.connectivity[bidx, 0].astype(np.int64)
        tri_conn = faces.node_connectivity[bidx]

        boundary_id, type_id, id_legend, type_legend = self._boundary_zone_ids(tri_conn)

        full_cell_fields = self._cell_fields(fields)
        boundary_fields = {k: v[owner_cells] for k, v in full_cell_fields.items()}

        output_path = Path(output_path)
        if format == 'legacy':
            if not output_path.suffix:
                output_path = output_path.with_suffix('.vtk')
            self._export_boundaries_legacy(
                output_path, fields, tri_conn, boundary_fields,
                boundary_id, type_id, id_legend, type_legend, binary=bool(binary),
            )
        elif format == 'xml':
            if not output_path.suffix:
                output_path = output_path.with_suffix('.vtu')
            self._export_boundaries_xml(
                output_path, fields, tri_conn, boundary_fields,
                boundary_id, type_id, id_legend, type_legend,
                binary=(True if binary is None else binary),
            )
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'legacy' or 'xml'")

        logger.success(f"VTK boundary patches exported: {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # 边界分区分类 + 场数据计算——已搬到 vtk_export_fields.py，这里只
    # 保留薄委托包装，保证 `exporter._xxx(...)` 这种直接方法调用（类
    # 内部互相调用，以及 cli/post/helpers.py 等外部调用方）行为不变。
    # ------------------------------------------------------------------

    def _boundary_zone_ids(self, tri_conn: np.ndarray):
        from .vtk_export_fields import boundary_zone_ids  # 见 vtk_export_fields.boundary_zone_ids
        return boundary_zone_ids(self, tri_conn)

    def _cell_fields(self, fields: List[str]) -> Dict[str, np.ndarray]:
        from .vtk_export_fields import cell_fields  # 见 vtk_export_fields.cell_fields
        return cell_fields(self, fields)

    def _cell_to_node(self, cell_values: np.ndarray, n_points: int, fallback: float = 0.0) -> np.ndarray:
        from .vtk_export_fields import cell_to_node  # 见 vtk_export_fields.cell_to_node
        return cell_to_node(self, cell_values, n_points, fallback=fallback)

    def _point_fields(self, cell_fields: Dict[str, np.ndarray], n_points: int) -> Dict[str, np.ndarray]:
        from .vtk_export_fields import point_fields  # 见 vtk_export_fields.point_fields
        return point_fields(self, cell_fields, n_points)

    # ------------------------------------------------------------------
    # Legacy VTK (.vtk)——ASCII 或二进制。已搬到 vtk_export_legacy.py，
    # 这里只保留薄委托包装。
    # ------------------------------------------------------------------

    _FIELD_LABELS = {
        'velocity': 'Velocity',
        'pressure': 'Pressure',
        'k': 'TurbulentKineticEnergy',
        'omega': 'SpecificDissipationRate',
        'nu_tilde': 'SANuTilde',
        'nut': 'TurbulentViscosity',
        'q_criterion': 'QCriterion',
    }

    def _expand_fields(self, fields: Optional[List[str]]) -> List[str]:
        """校验场名并展开 `turbulence` 别名（该模型实际有的湍流输出，`self.turbulence` 的键）。"""
        if fields is None:
            fields = ['velocity', 'pressure']
        invalid_fields = set(fields) - _VALID_FIELDS
        if invalid_fields:
            raise ValueError(
                f"Invalid fields: {invalid_fields}. "
                f"Valid fields: {_VALID_FIELDS}"
            )
        out = []
        for name in fields:
            if name == 'turbulence':
                if not self.turbulence:
                    logger.warning("'turbulence' requested but no turbulence fields are available "
                                   "(laminar case or checkpoint predates turb_cell_* fields)")
                out.extend(k for k in _TURBULENCE_FIELDS if k in self.turbulence and k not in out)
            elif name not in out:
                out.append(name)
        return out

    def _export_legacy(self, output_path: Path, fields: List[str], binary: bool) -> None:
        from .vtk_export_legacy import export_legacy  # 见 vtk_export_legacy.export_legacy
        export_legacy(self, output_path, fields, binary)

    @staticmethod
    def _wl(f, text: str, binary: bool) -> None:
        from .vtk_export_legacy import wl  # 见 vtk_export_legacy.wl
        wl(f, text, binary)

    def _write_points(self, f, binary: bool) -> None:
        from .vtk_export_legacy import write_points  # 见 vtk_export_legacy.write_points
        write_points(self, f, binary)

    def _write_cells(self, f, binary: bool) -> None:
        from .vtk_export_legacy import write_cells  # 见 vtk_export_legacy.write_cells
        write_cells(self, f, binary)

    def _write_cells_mixed(self, f, prism_conn: np.ndarray, tet_conn: np.ndarray, binary: bool) -> None:
        from .vtk_export_legacy import write_cells_mixed  # 见 vtk_export_legacy.write_cells_mixed
        write_cells_mixed(self, f, prism_conn, tet_conn, binary)

    def _write_cells_from(self, f, conn: np.ndarray, binary: bool) -> None:
        from .vtk_export_legacy import write_cells_from  # 见 vtk_export_legacy.write_cells_from
        write_cells_from(self, f, conn, binary)

    def _write_field_block(self, f, fields: List[str], values: Dict[str, np.ndarray], binary: bool) -> None:
        from .vtk_export_legacy import write_field_block  # 见 vtk_export_legacy.write_field_block
        write_field_block(self, f, fields, values, binary)

    def _write_scalar(self, f, name: str, values: np.ndarray, binary: bool, int_type: bool = False) -> None:
        from .vtk_export_legacy import write_scalar  # 见 vtk_export_legacy.write_scalar
        write_scalar(self, f, name, values, binary, int_type=int_type)

    def _write_field_data_legacy(self, f, entries: Dict[str, List[str]], binary: bool) -> None:
        from .vtk_export_legacy import write_field_data_legacy  # 见 vtk_export_legacy.write_field_data_legacy
        write_field_data_legacy(self, f, entries, binary)

    def _write_vector(self, f, name: str, values: np.ndarray, binary: bool) -> None:
        from .vtk_export_legacy import write_vector  # 见 vtk_export_legacy.write_vector
        write_vector(self, f, name, values, binary)

    def _export_boundaries_legacy(
        self, output_path: Path, fields: List[str], tri_conn: np.ndarray,
        boundary_fields: Dict[str, np.ndarray], boundary_id: np.ndarray, type_id: np.ndarray,
        id_legend: List[str], type_legend: List[str], binary: bool,
    ) -> None:
        from .vtk_export_legacy import export_boundaries_legacy  # 见 vtk_export_legacy.export_boundaries_legacy
        export_boundaries_legacy(
            self, output_path, fields, tri_conn, boundary_fields,
            boundary_id, type_id, id_legend, type_legend, binary,
        )

    # ------------------------------------------------------------------
    # XML VTK (.vtu)——委托给 pyvista/VTK 自己的写入器。已搬到
    # vtk_export_xml.py，这里只保留薄委托包装。
    # ------------------------------------------------------------------

    def _export_xml(self, output_path: Path, fields: List[str], binary: bool) -> None:
        from .vtk_export_xml import export_xml  # 见 vtk_export_xml.export_xml
        export_xml(self, output_path, fields, binary)

    def _export_boundaries_xml(
        self, output_path: Path, fields: List[str], tri_conn: np.ndarray,
        boundary_fields: Dict[str, np.ndarray], boundary_id: np.ndarray, type_id: np.ndarray,
        id_legend: List[str], type_legend: List[str], binary: bool,
    ) -> None:
        from .vtk_export_xml import export_boundaries_xml  # 见 vtk_export_xml.export_boundaries_xml
        export_boundaries_xml(
            self, output_path, fields, tri_conn, boundary_fields,
            boundary_id, type_id, id_legend, type_legend, binary,
        )
