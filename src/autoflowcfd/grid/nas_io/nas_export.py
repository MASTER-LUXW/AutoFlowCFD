"""NAS 文件导出模块。

将 VolumeMeshData 导出为 Nastran (.nas) 格式，用于可视化和后处理。
边界组的 PSHELL/CTRIA3/PSOLID 元数据写入部分拆到了同目录下的
nas_export_boundary.py，本文件只保留节点/体单元几何的写入与编排。

主要组件：
    - export_volume_mesh_to_nas：主导出函数
    - _write_header：写入 NAS 文件头
    - _write_nodes：写入 GRID* 卡片（大字段，保留 10 位有效数字）
    - _write_tetrahedra：写入 CTETRA 卡片
    - _write_pentahedra：写入 CPENTA 卡片
"""

import numpy as np
from pathlib import Path
from loguru import logger

from .nas_export_boundary import write_boundaries as _write_boundaries


def export_volume_mesh_to_nas(
    volume_mesh,
    output_path: str,
    include_boundaries: bool = True,
    scale_factor: float = 1000.0
) -> str:
    """将 VolumeMeshData 导出为 Nastran (.nas) 格式。

    将四面体体网格转换为 Nastran 格式：
    - 节点的 GRID 卡片（默认：毫米，匹配本项目的 .nas 导入约定
      ——NASParser 默认 units='mm'，所以此函数默认值和 NASParser
      的往返保持一致）
    - 四面体元素的 CTETRA 卡片
    - 边界组的 PSHELL/PSET 卡片（可选）

    Args:
        volume_mesh: VolumeMeshData 对象，包含 nodes、cells、boundaries
            （内部存储为米，SI 单位）。
        output_path: 输出文件路径（.nas 扩展名）。
        include_boundaries: 是否包含边界组信息。
        scale_factor: 坐标缩放因子，应用于内部基于米的坐标
            （默认 1000.0 写入毫米，匹配 NASParser 的默认导入单位）。
            传 1.0 写入米。

    Returns:
        str: 导出文件的路径

    Example:
        >>> from autoflowcfd.grid import NASParser
        >>> parser = NASParser('surface.nas')
        >>> volume_mesh = parser.parse(generate_volume_mesh=True)
        >>> export_volume_mesh_to_nas(volume_mesh, 'volume_mesh.nas')
    """
    output_path = Path(output_path)

    # 确保 .nas 扩展名
    if output_path.suffix.lower() != '.nas':
        output_path = output_path.with_suffix('.nas')

    logger.info(f"Exporting volume mesh to NAS: {output_path}")
    logger.info(f"  Nodes: {volume_mesh.node_count:,}")
    logger.info(f"  Cells: {volume_mesh.cell_count:,}")
    logger.info(f"  Total volume: {volume_mesh.total_volume:.6e} m^3")

    write_boundaries = bool(
        include_boundaries and volume_mesh.boundaries and volume_mesh.boundaries.groups
    )
    n_boundary_groups = len(volume_mesh.boundaries.groups) if write_boundaries else 0
    # 下方边界组用的是 PSHELL PID 1..n_boundary_groups，所以体网格的
    # PSOLID 属性必须在这个范围之外——否则一旦分组数 >= 4（一个很常见
    # 的情况：inlet/outlet/wall/symmetry/ground）就会与某个边界的
    # PSHELL PID 冲突。
    solid_pid = n_boundary_groups + 1

    prism_cells = getattr(volume_mesh, 'prism_cells', None)
    has_prisms = prism_cells is not None and prism_cells.count > 0

    try:
        with open(output_path, 'w') as f:
            # 写入 header
            _write_header(f, volume_mesh)

            # 写入节点（GRID 卡片）
            logger.info("Writing nodes...")
            _write_nodes(f, volume_mesh.nodes, scale_factor)

            # 写入体单元。先写棱柱（BL 区域，CPENTA），再写
            # 四面体（核心区域，CTETRA）——元素 ID 遵循与网格自身
            # 单元索引相同的全局排序约定（[0, n_prism) 棱柱，
            # [n_prism, n_prism+n_tet) 四面体——见
            # PrismCells/face_extractor.extract_faces_mixed），所以
            # 下面的边界组单元索引直接与元素 ID 对齐，无需额外重映射。
            n_prism = 0
            if has_prisms:
                logger.info("Writing pentahedral (BL prism) elements...")
                n_prism = _write_pentahedra(f, prism_cells.connectivity, solid_pid)

            logger.info("Writing tetrahedral elements...")
            n_tets = _write_tetrahedra(f, volume_mesh.cells.connectivity, solid_pid, start_eid=n_prism + 1)

            # 写入边界信息（可选）：边界面作为 CTRIA3 元素，
            # 引用每组的 PSHELL 属性，使组在 ANSA/Nastran 中
            # 实际可选，而不是空的属性定义。
            if write_boundaries:
                logger.info("Writing boundary groups...")
                _write_boundaries(
                    f, volume_mesh, solid_pid=solid_pid, start_eid=n_prism + n_tets + 1
                )

            # 每个 Bulk Data deck 必须以 ENDDATA 结束，无论
            # 是否写入了边界组。
            f.write("ENDDATA\n")
            f.write("$ End of file\n")

        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
        logger.success(
            f"Volume mesh exported successfully: {output_path}\n"
            f"  File size: {file_size:.2f} MB"
        )

        return str(output_path)

    except Exception as e:
        logger.error(f"Failed to export volume mesh: {e}")
        raise RuntimeError(f"NAS export failed: {e}")


def _write_header(f, volume_mesh) -> None:
    """写入 NAS 文件头与元数据。

    Args:
        f: 文件句柄
        volume_mesh: VolumeMeshData 对象
    """
    from datetime import datetime
    
    # 注释行如实写明来源（2026-10-09 以前写的是 "$ANSA_VERSION;21.0.1;" 与 "file created by ANSA"——
    # 文件并不是 ANSA 生成的；没有任何解析器依赖这两行，ANSA 导入标准 Nastran bulk data 也不需要它们）
    timestamp = datetime.now().strftime("%a %b %d %H:%M:%S %Y")
    f.write("$\n")
    f.write(f"$ AutoFlowCFD Volume Mesh Export  {timestamp}\n")
    f.write(f"$ Nodes: {volume_mesh.node_count:,}\n")
    f.write(f"$ Elements: {volume_mesh.cell_count:,}\n")
    f.write(f"$ Total Volume: {volume_mesh.total_volume:.6e} m^3\n")
    f.write("$\n")
    f.write("$\n")
    f.write("$\n")
    f.write("BEGIN BULK                                                                      \n")


def _format_large_field_float(value: float) -> str:
    """16 字符大字段浮点数（10 位有效数字，`-1.234567890E+03` 恰好 16 字符）。"""
    return f"{value:16.9E}"


def _write_nodes(f, nodes, scale_factor: float) -> None:
    """写入所有节点的 GRID 卡片（Nastran 大字段格式 `GRID*` + `*` 续行）。

    小字段每个坐标只有 8 个字符：以毫米写出时，量级上千的坐标只剩 3 位小数（1 µm）。2026-10-08 cube_demo
    实测：同一份生成网格的相邻单元体积比在内存里是 33.20，导出、再导入后变成 222.69（偏斜度 0.975 -> 0.995），
    边界层与锐边处的近退化单元被坐标截断进一步压扁。大字段保留 10 位有效数字。

        行 1：列 1-8 "GRID*"，9-24 节点 ID，25-40 坐标系 ID（0 = 全局），41-56 X，57-72 Y，73-80 续行标识（空）
        行 2：列 1-8 "*"（续行），9-24 Z

    Args:
        f: 文件句柄
        nodes: NodeArray，包含 x, y, z 坐标
        scale_factor: 坐标缩放因子
    """
    n_nodes = len(nodes.x)
    # 批量写入：行累积在列表中，每批一次 writelines()
    batch_size = 1000
    for start_idx in range(0, n_nodes, batch_size):
        end_idx = min(start_idx + batch_size, n_nodes)
        lines = []
        for i in range(start_idx, end_idx):
            x = _format_large_field_float(nodes.x[i] * scale_factor)
            y = _format_large_field_float(nodes.y[i] * scale_factor)
            z = _format_large_field_float(nodes.z[i] * scale_factor)
            lines.append(f"GRID*   {i + 1:>16}{0:>16}{x}{y}\n*       {z}\n")
        f.writelines(lines)
        if (start_idx + batch_size) % 10000 == 0:
            logger.debug(f"  Written {start_idx + batch_size}/{n_nodes} nodes")
    logger.info(f"  Total nodes written: {n_nodes:,}")


def _write_tetrahedra(f, connectivity: np.ndarray, solid_pid: int, start_eid: int = 1) -> int:
    """写入四面体元素的 CTETRA 卡片。

    ANSA Nastran CTETRA 卡片格式（固定宽度字段）：
    CTETRA      EID       PID      G1       G2       G3       G4

    字段宽度：8-8-8-8-8-8 字符

    Args:
        f: 文件句柄
        connectivity: 四面体连接数组，shape=(n_tets, 4)
        solid_pid: 体单元的 PSOLID 属性 ID。必须与
            _write_boundaries 写入的 PSOLID 卡片匹配（或不与任何
            PSHELL PID 重复），以避免重复的 Bulk Data 条目。
        start_eid: 使用的第一个元素 ID（默认 1）——当棱柱
            (CPENTA) 元素已在此调用之前写入并占据 [1, start_eid)
            时非 1（见 export_volume_mesh_to_nas：棱柱占据
            与代码库其他地方相同的 [0, n_prism) 全局单元索引范围，
            所以元素 ID 与边界组单元索引保持一致）。

    Returns:
        int: 写入的四面体数量（使用的元素 ID 为
        start_eid..start_eid+n_tets-1），调用方可继续编号
        （例如边界 CTRIA3 卡片）而不与这些元素 ID 冲突。
    """
    n_tets = len(connectivity)

    # 批量写入以提高性能
    batch_size = 1000

    for start_idx in range(0, n_tets, batch_size):
        end_idx = min(start_idx + batch_size, n_tets)

        for i in range(start_idx, end_idx):
            elem_id = start_eid + i
            g1 = int(connectivity[i, 0]) + 1  # 从 0-indexed 转换为 1-indexed
            g2 = int(connectivity[i, 1]) + 1
            g3 = int(connectivity[i, 2]) + 1
            g4 = int(connectivity[i, 3]) + 1

            # ANSA 格式：固定宽度字段
            line = f"CTETRA{elem_id:>10}{solid_pid:>8}{g1:>8}{g2:>8}{g3:>8}{g4:>8}\n"
            f.write(line)

        if (start_idx + batch_size) % 10000 == 0:
            logger.debug(f"  Written {start_idx + batch_size}/{n_tets} elements")

    logger.info(f"  Total elements written: {n_tets:,}")
    return n_tets


def _write_pentahedra(f, connectivity: np.ndarray, solid_pid: int, start_eid: int = 1) -> int:
    """写入三角棱柱（BL 区域）元素的 CPENTA 卡片。

    ANSA Nastran CPENTA 卡片格式（固定宽度字段）：
    CPENTA      EID       PID      G1       G2       G3       G4       G5       G6

    G1-G3 是一个三角端盖，G4-G6 是另一个，Gi+3 在 Gi "上方"——
    完全匹配 PrismCells 的 (v0,v1,v2,w0,w1,w2) 约定（w_i 是 v_i
    的拉伸），所以此处的连接无需重排。

    卡片名 + EID + PID + 6 个网格 ID = 9 个字段，适合 Nastran 每行
    10 个（8 字符）字段的小字段布局，有余量——不需要续行卡片
    （与本文件其他地方的 PSHELL 不同，PSHELL 的数据多于一行放不下）。

    Args:
        f: 文件句柄
        connectivity: 棱柱连接数组，shape=(n_prism, 6)
        solid_pid: 体单元的 PSOLID 属性 ID（与 _write_tetrahedra 共享
            ——两个区域属于同一个实体部件）
        start_eid: 使用的第一个元素 ID（默认 1）

    Returns:
        int: 写入的棱柱数量（使用的元素 ID 为
        start_eid..start_eid+n_prism-1）
    """
    n_prism = len(connectivity)
    batch_size = 1000

    for start_idx in range(0, n_prism, batch_size):
        end_idx = min(start_idx + batch_size, n_prism)

        for i in range(start_idx, end_idx):
            elem_id = start_eid + i
            g = [int(connectivity[i, k]) + 1 for k in range(6)]

            line = (
                f"CPENTA{elem_id:>10}{solid_pid:>8}{g[0]:>8}{g[1]:>8}{g[2]:>8}"
                f"{g[3]:>8}{g[4]:>8}{g[5]:>8}\n"
            )
            f.write(line)

        if (start_idx + batch_size) % 10000 == 0:
            logger.debug(f"  Written {start_idx + batch_size}/{n_prism} elements")

    logger.info(f"  Total pentahedral elements written: {n_prism:,}")
    return n_prism


