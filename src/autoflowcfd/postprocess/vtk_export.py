"""VTK 场数据导出模块。

本模块提供把 CFD 仿真结果导出为 VTK 格式的工具，供 ParaView 及其它
VTK 兼容查看器可视化使用。

Key Components:
    - VTKExporter: VTK 文件生成的主导出器
    - 支持速度、压力、湍流变量导出

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
    - `mu_t`（湍流动力粘度），如果提供了，是求解器自己算出的 SST 混合
      值（见 core/turbulence_sst.py 的 `compute_eddy_viscosity`），通过
      CheckpointManager 的 extra_fields 持久化保存——而不是在它不可用时
      （例如加上这个功能之前保存的旧 checkpoint）退化用的简化
      nu_t = k/omega 估计值。
    - `export_boundaries()` 只导出命名的边界面片（WALL/INLET/OUTLET/...
      表面三角形），标记稳定的整数 BoundaryID/BoundaryTypeID（并把
      名称对照表作为 field data 嵌入）——这是 Fluent/OpenFOAM/STAR-CCM+
      使用的按面片划分的工作流程，而不是永远只能看到没有边界身份信息
      的整个体网格。

Example:
    >>> from autoflowcfd.postprocess import VTKExporter
    >>> exporter = VTKExporter(grid_data, solution, mu_t=mu_t)
    >>> exporter.export("output.vtk", binary=True)
"""

import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
from loguru import logger

from ..grid.structures import GridData
from ..core.backend.base import SolutionVector
from ._field_utils import cell_to_node

# VTK legacy 单元类型代码（见 VTK 文件格式规范）。
_VTK_TRIANGLE = 5
_VTK_TETRA = 10
_VTK_WEDGE = 13  # 三棱柱——VTK 自己的节点顺序正好与本项目的
                 # (v0,v1,v2,w0,w1,w2) 约定直接一致（两个三角形"底面"
                 # 依次列出），不需要重新排序

_VALID_FIELDS = {'velocity', 'pressure', 'k', 'omega', 'nut', 'turbulence'}


class VTKExporter:
    """VTK 场数据导出器。

    把流场数据导出为 VTK 格式，供 ParaView 可视化。同时支持 legacy VTK
    （ASCII 或二进制）和基于 XML 的 VTK（.vtu，ASCII 或二进制+压缩）
    两种格式，每种格式的每个场都同时带有原始单元中心求解器值
    （CELL_DATA）和节点插值值（POINT_DATA）。

    Attributes:
        grid_data: 网格数据对象
        solution: 流场解向量
        mu_t: 可选的 (n_cells,) 湍流动力粘度，求解器实际算出的值（来自
            CheckpointManager 的 extra_fields）。缺失时，'nut' 导出会
            退化成简化的 k/omega 估计并记录警告。

    Example:
        >>> exporter = VTKExporter(grid_data, solution, mu_t=mu_t)
        >>> exporter.export("result.vtk", fields=['velocity', 'pressure'], binary=True)
    """

    def __init__(
        self,
        grid_data: GridData,
        solution: SolutionVector,
        mu_t: Optional[np.ndarray] = None,
    ):
        """初始化 VTK 导出器。

        Args:
            grid_data: 网格数据对象
            solution: 流场解向量
            mu_t: 可选的、求解器算出的精确逐单元湍流动力粘度
                (Pa.s)，形状 (n_cells,)

        Raises:
            ValueError: 网格或解数据无效
        """
        self.grid_data = grid_data
        self.solution = solution
        self.mu_t = np.asarray(mu_t, dtype=np.float64) if mu_t is not None else None

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
        if fields is None:
            fields = ['velocity', 'pressure']

        invalid_fields = set(fields) - _VALID_FIELDS
        if invalid_fields:
            raise ValueError(
                f"Invalid fields: {invalid_fields}. "
                f"Valid fields: {_VALID_FIELDS}"
            )

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
          - BoundaryTypeID（CELL_DATA，int32）：更粗粒度的物理角色分类
            （WALL/GROUND/INLET/OUTLET/SYMMETRY/FARFIELD），用的是与
            实际求解路径*完全相同*的分类方式
            （BoundaryConditionHandler._classify）——而不是这里独立
            重新推导、可能悄悄偏离求解器实际处理方式的猜测。对照表：
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
        if fields is None:
            fields = ['velocity', 'pressure']
        invalid_fields = set(fields) - _VALID_FIELDS
        if invalid_fields:
            raise ValueError(
                f"Invalid fields: {invalid_fields}. "
                f"Valid fields: {_VALID_FIELDS}"
            )

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

        boundary_id, type_id, id_legend, type_legend = self._boundary_zone_ids(owner_cells)

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

    _BC_TYPE_NAMES = ['WALL', 'GROUND', 'INLET', 'OUTLET', 'SYMMETRY', 'FARFIELD']

    def _boundary_zone_ids(self, owner_cells: np.ndarray):
        """把每个边界面的 owner 四面体映射到 BoundaryID（按边界组名）
        和 BoundaryTypeID（按名称模式匹配的分类桶）。

        Returns:
            (boundary_id, type_id, id_legend, type_legend)——前两个是
            (n_boundary_faces,) 的 int32 数组，对照表是 "<id>=<name>"
            形式的 List[str]。
        """
        boundary_names = self.grid_data.boundaries.boundary_names
        name_to_id = {name: i for i, name in enumerate(boundary_names)}
        type_to_id = {t: i for i, t in enumerate(self._BC_TYPE_NAMES)}
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
        n_cells = self.grid_data.cell_count
        cell_to_name_id = np.full(n_cells, unclassified_id, dtype=np.int32)
        cell_to_type_id = np.full(n_cells, type_to_id['WALL'], dtype=np.int32)
        for name in boundary_names:
            btype = _classify(name)
            nid = name_to_id[name]
            tid = type_to_id.get(btype, type_to_id['WALL'])
            cells = np.asarray(self.grid_data.boundaries.get_cell_indices(name), dtype=np.int64)
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

    # ------------------------------------------------------------------
    # 共用的场计算——CELL_DATA 原始值和 POINT_DATA 节点插值值的唯一数据
    # 源，保证这两种表示不会悄悄产生分歧。
    # ------------------------------------------------------------------

    def _cell_fields(self, fields: List[str]) -> Dict[str, np.ndarray]:
        """在单元中心分辨率上计算每个请求的场（标量 (n_cells,)，矢量
        (n_cells, 3)），解数据不可用时（例如空的 SolutionVector）套用
        与旧的纯节点写入器相同的兜底常数。

        Returns:
            场名（'velocity'、'pressure'、'k'、'omega'、'nut'）到其原始
            逐单元数组的字典——正是 CELL_DATA 写入的内容，也是
            POINT_DATA 插值的数据源。
        """
        n_cells = self.grid_data.cell_count
        has_data = self.solution.data is not None and self.solution.n_cells > 0
        out: Dict[str, np.ndarray] = {}

        if 'velocity' in fields:
            if has_data:
                u, v, w = self.solution.get_velocity()
                out['velocity'] = np.column_stack([u, v, w])
            else:
                logger.warning("Solution data not available. Using zero velocity.")
                out['velocity'] = np.zeros((n_cells, 3))

        if 'pressure' in fields:
            if has_data:
                out['pressure'] = self.solution.get_pressure()
            else:
                logger.warning("Solution data not available. Using uniform pressure.")
                out['pressure'] = np.full(n_cells, 101325.0)

        need_turb = 'k' in fields or 'omega' in fields or 'nut' in fields
        if need_turb:
            k = omega = np.array([])
            if has_data:
                k, omega = self.solution.get_turbulence()
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
                if self.mu_t is not None and len(self.mu_t) == n_cells and has_data:
                    rho = np.maximum(self.solution.get_density(), 1e-10)
                    out['nut'] = self.mu_t / rho
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

        return out

    def _cell_to_node(self, cell_values: np.ndarray, n_points: int, fallback: float = 0.0) -> np.ndarray:
        """把逐单元标量场插值成逐节点值（对每个节点相连的单元做体积
        加权平均——见 _field_utils.cell_to_node）。"""
        conn = np.asarray(self.grid_data.cells.connectivity)
        volumes = getattr(self.grid_data.cells, "volumes", None)
        return cell_to_node(conn, cell_values, n_points, volumes=volumes, fallback=fallback)

    def _point_fields(self, cell_fields: Dict[str, np.ndarray], n_points: int) -> Dict[str, np.ndarray]:
        """把 `cell_fields` 里每个单元中心场都插值到节点。"""
        out: Dict[str, np.ndarray] = {}
        for name, arr in cell_fields.items():
            if arr.ndim == 2:
                fallback = 0.0 if name == 'velocity' else 0.0
                out[name] = np.column_stack([
                    self._cell_to_node(arr[:, i], n_points, fallback=float(np.mean(arr[:, i])) if len(arr) else 0.0)
                    for i in range(arr.shape[1])
                ])
            else:
                fallback = 101325.0 if name == 'pressure' else 0.0
                out[name] = self._cell_to_node(arr, n_points, fallback=fallback)
        return out

    # ------------------------------------------------------------------
    # Legacy VTK (.vtk)——ASCII 或二进制
    # ------------------------------------------------------------------

    _FIELD_LABELS = {
        'velocity': 'Velocity',
        'pressure': 'Pressure',
        'k': 'TurbulentKineticEnergy',
        'omega': 'SpecificDissipationRate',
        'nut': 'TurbulentViscosity',
    }

    def _export_legacy(self, output_path: Path, fields: List[str], binary: bool) -> None:
        """导出为 legacy VTK 格式（VTK Legacy 规范，DataFile Version
        3.0——经典的 CELLS/CELL_TYPES 布局，不是 VTK 9 更新的
        OFFSETS/CONNECTIVITY 变体，以便和旧版读取器保持最大兼容性）。"""
        logger.info(f"Exporting to legacy VTK format ({'binary' if binary else 'ASCII'}): {output_path}")

        n_points = self.grid_data.nodes.count
        n_cells = self.grid_data.cell_count
        cell_fields = self._cell_fields(fields)
        point_fields = self._point_fields(cell_fields, n_points)

        try:
            mode = 'wb' if binary else 'w'
            with open(output_path, mode) as f:
                self._wl(f, "# vtk DataFile Version 3.0\n", binary)
                self._wl(f, f"AutoFlowCFD Export - {output_path.name}\n", binary)
                self._wl(f, ("BINARY\n" if binary else "ASCII\n"), binary)
                self._wl(f, "\n", binary)
                self._wl(f, "DATASET UNSTRUCTURED_GRID\n", binary)
                self._wl(f, "\n", binary)

                self._write_points(f, binary)
                self._write_cells(f, binary)

                # CELL_DATA：原始、未插值的求解器值。
                self._wl(f, f"CELL_DATA {n_cells}\n", binary)
                self._write_field_block(f, fields, cell_fields, binary)

                # POINT_DATA：体积加权插值，用于平滑等值面渲染。
                self._wl(f, f"POINT_DATA {n_points}\n", binary)
                self._write_field_block(f, fields, point_fields, binary)

            logger.info("Legacy VTK file written successfully")

        except IOError as e:
            logger.error(f"Failed to write VTK file: {e}")
            raise

    @staticmethod
    def _wl(f, text: str, binary: bool) -> None:
        """写一行头部/关键字，二进制模式下编码为字节。"""
        f.write(text.encode('ascii') if binary else text)

    def _write_points(self, f, binary: bool) -> None:
        nodes = self.grid_data.nodes
        n_points = nodes.count
        coords = np.column_stack([nodes.x, nodes.y, nodes.z]).astype(np.float64)

        self._wl(f, f"POINTS {n_points} double\n", binary)
        if binary:
            f.write(coords.astype('>f8').tobytes())
            f.write(b'\n')
        else:
            np.savetxt(f, coords, fmt="%.6e")
        self._wl(f, "\n", binary)

    def _write_cells(self, f, binary: bool) -> None:
        """把单元连接关系写入 VTK 文件。

        从 connectivity 数组自身的形状检测每个单元实际的节点数
        （3=三角形，4=四面体）——如果设置了 grid_data.prism_cells，则
        先写三棱柱（6 节点 wedge，全局索引 [0, n_prism)），再写四面体
        （[n_prism, n_prism+n_tet)），与本项目的全局单元索引约定一致
        （见 PrismCells/face_extractor.extract_faces_mixed）。
        """
        prism_cells_obj = getattr(self.grid_data, 'prism_cells', None)
        if prism_cells_obj is not None:
            self._write_cells_mixed(f, prism_cells_obj.connectivity, self.grid_data.cells.connectivity, binary)
        else:
            self._write_cells_from(f, self.grid_data.cells.connectivity, binary)

    def _write_cells_mixed(self, f, prism_conn: np.ndarray, tet_conn: np.ndarray, binary: bool) -> None:
        """为三棱柱(wedge)+四面体混合网格写 CELLS/CELL_TYPES——legacy
        VTK 的 CELLS 格式里每行可以有不同的顶点数（每行开头的整数就是
        该行的顶点数），所以三棱柱和四面体可以直接拼接成一个块；
        CELL_TYPES 携带每行的类型代码（_VTK_WEDGE 还是 _VTK_TETRA）。"""
        prism_conn = np.asarray(prism_conn, dtype=np.int32)
        tet_conn = np.asarray(tet_conn, dtype=np.int32)
        n_prism = len(prism_conn)
        n_tet = len(tet_conn)
        n_cells = n_prism + n_tet
        total_ints = n_prism * 7 + n_tet * 5  # (1 个计数 + 6 个顶点) 或 (1 个计数 + 4 个顶点)

        self._wl(f, f"CELLS {n_cells} {total_ints}\n", binary)
        if binary:
            if n_prism:
                prism_lines = np.hstack([np.full((n_prism, 1), 6, dtype=np.int32), prism_conn])
                f.write(prism_lines.astype('>i4').tobytes())
            if n_tet:
                tet_lines = np.hstack([np.full((n_tet, 1), 4, dtype=np.int32), tet_conn])
                f.write(tet_lines.astype('>i4').tobytes())
            f.write(b'\n')
        else:
            if n_prism:
                prism_lines = np.hstack([np.full((n_prism, 1), 6, dtype=np.int32), prism_conn])
                np.savetxt(f, prism_lines, fmt="%d")
            if n_tet:
                tet_lines = np.hstack([np.full((n_tet, 1), 4, dtype=np.int32), tet_conn])
                np.savetxt(f, tet_lines, fmt="%d")
        self._wl(f, "\n", binary)

        cell_types = np.concatenate([
            np.full(n_prism, _VTK_WEDGE, dtype=np.int32),
            np.full(n_tet, _VTK_TETRA, dtype=np.int32),
        ])
        self._wl(f, f"CELL_TYPES {n_cells}\n", binary)
        if binary:
            f.write(cell_types.astype('>i4').tobytes())
            f.write(b'\n')
        else:
            np.savetxt(f, cell_types.reshape(-1, 1), fmt="%d")
        self._wl(f, "\n", binary)

    def _write_cells_from(self, f, conn: np.ndarray, binary: bool) -> None:
        """从显式的 connectivity 数组写 CELLS/CELL_TYPES——同时供整体
        体网格导出（_write_cells）和边界面导出（同一份节点数组上的
        另一组更小的三角形）共用。"""
        conn = np.asarray(conn, dtype=np.int32)
        n_cells = conn.shape[0]
        nodes_per_cell = conn.shape[1]

        vtk_type = {3: _VTK_TRIANGLE, 4: _VTK_TETRA}.get(nodes_per_cell)
        if vtk_type is None:
            raise ValueError(
                f"Unsupported cell connectivity width {nodes_per_cell} "
                f"(expected 3 for triangles or 4 for tetrahedra)"
            )

        counts = np.full((n_cells, 1), nodes_per_cell, dtype=np.int32)
        cell_lines = np.hstack([counts, conn])

        self._wl(f, f"CELLS {n_cells} {n_cells * (nodes_per_cell + 1)}\n", binary)
        if binary:
            f.write(cell_lines.astype('>i4').tobytes())
            f.write(b'\n')
        else:
            np.savetxt(f, cell_lines, fmt="%d")
        self._wl(f, "\n", binary)

        self._wl(f, f"CELL_TYPES {n_cells}\n", binary)
        types_arr = np.full(n_cells, vtk_type, dtype=np.int32)
        if binary:
            f.write(types_arr.astype('>i4').tobytes())
            f.write(b'\n')
        else:
            np.savetxt(f, types_arr, fmt="%d")
        self._wl(f, "\n", binary)

    def _write_field_block(self, f, fields: List[str], values: Dict[str, np.ndarray], binary: bool) -> None:
        if 'velocity' in fields and 'velocity' in values:
            self._write_vector(f, "Velocity", values['velocity'], binary)
        if 'pressure' in fields and 'pressure' in values:
            self._write_scalar(f, "Pressure", values['pressure'], binary)
        if 'k' in fields and 'k' in values:
            self._write_scalar(f, "TurbulentKineticEnergy", values['k'], binary)
        if 'omega' in fields and 'omega' in values:
            self._write_scalar(f, "SpecificDissipationRate", values['omega'], binary)
        if 'nut' in fields and 'nut' in values:
            self._write_scalar(f, "TurbulentViscosity", values['nut'], binary)

    def _write_scalar(self, f, name: str, values: np.ndarray, binary: bool, int_type: bool = False) -> None:
        vtk_type_name = "int" if int_type else "double"
        np_dtype = '>i4' if int_type else '>f8'
        self._wl(f, f"SCALARS {name} {vtk_type_name} 1\n", binary)
        self._wl(f, "LOOKUP_TABLE default\n", binary)
        if binary:
            f.write(np.ascontiguousarray(values).astype(np_dtype).tobytes())
            f.write(b'\n')
        else:
            np.savetxt(f, values, fmt="%d" if int_type else "%.6e")
        self._wl(f, "\n", binary)

    def _write_field_data_legacy(self, f, entries: Dict[str, List[str]], binary: bool) -> None:
        """写一个 FIELD FieldData 块（全局元数据，例如
        BoundaryID->名称对照表）。

        只在 ASCII 模式下写出：实测发现，VTK 9.3 自己的
        vtkUnstructuredGridReader 只要文件的数据模式是 BINARY，就无法
        解析**任何**包含字符串类型 FIELD 块的 legacy 文件——用一个独立
        于本写入器的最小手写复现确认过（不管 field 块在二进制载荷之前
        还是之后，都是同样失败；同一个块放在 ASCII 模式文件里能正确
        读回）。与其生成一个 VTK 自己的读取器都打不开的二进制 .vtk，
        binary=True 时改为把对照表记录到日志——无论如何，数值型的
        BoundaryID/BoundaryTypeID CELL_DATA 都不受影响。XML（.vtu）
        没有这个问题（见 _export_boundaries_xml），是这类导出推荐使用
        的格式。
        """
        if not entries:
            return
        if binary:
            for name, values in entries.items():
                logger.info(f"{name}: " + ", ".join(values))
            return
        self._wl(f, f"FIELD FieldData {len(entries)}\n", binary)
        for name, values in entries.items():
            self._wl(f, f"{name} 1 {len(values)} string\n", binary)
            for v in values:
                self._wl(f, f"{v}\n", binary)
        self._wl(f, "\n", binary)

    def _write_vector(self, f, name: str, values: np.ndarray, binary: bool) -> None:
        self._wl(f, f"VECTORS {name} double\n", binary)
        if binary:
            f.write(np.ascontiguousarray(values, dtype=np.float64).astype('>f8').tobytes())
            f.write(b'\n')
        else:
            np.savetxt(f, values, fmt="%.6e")
        self._wl(f, "\n", binary)

    def _export_boundaries_legacy(
        self, output_path: Path, fields: List[str], tri_conn: np.ndarray,
        boundary_fields: Dict[str, np.ndarray], boundary_id: np.ndarray, type_id: np.ndarray,
        id_legend: List[str], type_legend: List[str], binary: bool,
    ) -> None:
        logger.info(f"Exporting boundary patches to legacy VTK format ({'binary' if binary else 'ASCII'}): {output_path}")
        n_tri = tri_conn.shape[0]

        try:
            mode = 'wb' if binary else 'w'
            with open(output_path, mode) as f:
                self._wl(f, "# vtk DataFile Version 3.0\n", binary)
                self._wl(f, f"AutoFlowCFD Boundary Export - {output_path.name}\n", binary)
                self._wl(f, ("BINARY\n" if binary else "ASCII\n"), binary)
                self._wl(f, "\n", binary)
                self._wl(f, "DATASET UNSTRUCTURED_GRID\n", binary)
                self._write_field_data_legacy(f, {
                    'BoundaryID_to_Name': id_legend,
                    'BoundaryTypeID_to_Name': type_legend,
                }, binary)
                self._wl(f, "\n", binary)

                self._write_points(f, binary)
                self._write_cells_from(f, tri_conn, binary)

                self._wl(f, f"CELL_DATA {n_tri}\n", binary)
                self._write_scalar(f, "BoundaryID", boundary_id, binary, int_type=True)
                self._write_scalar(f, "BoundaryTypeID", type_id, binary, int_type=True)
                self._write_field_block(f, fields, boundary_fields, binary)

            logger.info("Legacy VTK boundary file written successfully")

        except IOError as e:
            logger.error(f"Failed to write VTK boundary file: {e}")
            raise

    # ------------------------------------------------------------------
    # XML VTK (.vtu)——委托给 pyvista/VTK 自己的写入器
    # ------------------------------------------------------------------

    def _export_xml(self, output_path: Path, fields: List[str], binary: bool) -> None:
        """导出为基于 XML 的 VTK 格式（.vtu），当前主流 CFD 后处理工具
        采用的现代标准格式。从与 legacy 写入器相同的单元/节点场数据
        构建一个 pyvista.UnstructuredGrid，交给 VTK 自己的
        vtkXMLUnstructuredGridWriter 序列化（binary=True 时带
        binary+zlib 压缩）——这样不需要自己手写 XML appended-data 的
        二进制编码，pyvista/VTK 已经正确实现了这一点，ParaView 自身
        读写用的也是这一套。
        """
        import pyvista as pv

        logger.info(f"Exporting to XML VTK format ({'binary' if binary else 'ASCII'}): {output_path}")

        nodes = self.grid_data.nodes
        n_points = nodes.count
        points = np.column_stack([nodes.x, nodes.y, nodes.z]).astype(np.float64)

        conn = np.asarray(self.grid_data.cells.connectivity, dtype=np.int64)
        nodes_per_cell = conn.shape[1]
        cell_type = {3: pv.CellType.TRIANGLE, 4: pv.CellType.TETRA}.get(nodes_per_cell)
        if cell_type is None:
            raise ValueError(
                f"Unsupported cell connectivity width {nodes_per_cell} "
                f"(expected 3 for triangles or 4 for tetrahedra)"
            )

        grid = pv.UnstructuredGrid({cell_type: conn}, points)

        cell_fields = self._cell_fields(fields)
        point_fields = self._point_fields(cell_fields, n_points)
        for key, arr in cell_fields.items():
            grid.cell_data[self._FIELD_LABELS[key]] = arr
        for key, arr in point_fields.items():
            grid.point_data[self._FIELD_LABELS[key]] = arr

        grid.save(str(output_path), binary=binary)
        logger.info("XML VTK file written successfully")

    def _export_boundaries_xml(
        self, output_path: Path, fields: List[str], tri_conn: np.ndarray,
        boundary_fields: Dict[str, np.ndarray], boundary_id: np.ndarray, type_id: np.ndarray,
        id_legend: List[str], type_legend: List[str], binary: bool,
    ) -> None:
        """把边界面片导出为 .vtu——见 export_boundaries。
        BoundaryID/BoundaryTypeID -> 名称对照表放在 field_data（全局
        元数据，不是逐单元）里：已实测验证，逐单元的*字符串*类型
        CELL_DATA 数组经过 VTK XML 写入器/读取器往返后不会保留（数组
        列出来了，但读回时是空指针），而 field_data 字符串数组能作为
        vtkStringArray 正确往返——无论是直接通过
        vtkXMLUnstructuredGridReader 还是通过 pyvista.read()。
        """
        import pyvista as pv

        logger.info(f"Exporting boundary patches to XML VTK format ({'binary' if binary else 'ASCII'}): {output_path}")

        nodes = self.grid_data.nodes
        points = np.column_stack([nodes.x, nodes.y, nodes.z]).astype(np.float64)

        grid = pv.UnstructuredGrid({pv.CellType.TRIANGLE: np.asarray(tri_conn, dtype=np.int64)}, points)
        grid.cell_data['BoundaryID'] = boundary_id
        grid.cell_data['BoundaryTypeID'] = type_id
        for key, arr in boundary_fields.items():
            grid.cell_data[self._FIELD_LABELS[key]] = arr
        grid.field_data['BoundaryID_to_Name'] = np.array(id_legend)
        grid.field_data['BoundaryTypeID_to_Name'] = np.array(type_legend)

        grid.save(str(output_path), binary=binary)
        logger.info("XML VTK boundary file written successfully")
