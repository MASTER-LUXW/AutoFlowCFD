"""NAS 解析器：节点（节点）提取。

流式解析 Nastran 格式文件里的 GRID 节点卡片。
"""

import re
import numpy as np
from loguru import logger

from ..structures import NodeArray
from .nas_parser_exceptions import NASParseError
from .nas_parser_utils import parse_nastran_float

# 超过这个比例的 GRID 行被丢弃（解析错误 + 不可解析的行），就视为
# 系统性的格式/编码不匹配而不是偶然噪声，大声报错而不是静默返回一个
# 缺失大块真实几何但仍然报告"成功"的网格。仅在有意义的样本量
# (MIN_LINES_FOR_DROP_CHECK) 时才强制执行——小文件里几行 GRID 有一行
# 坏的不是系统性损坏的证据，而真实网格成千上万行里同样比例就是。
MAX_DROP_FRACTION = 0.05
MIN_LINES_FOR_DROP_CHECK = 20


def parse_nodes_from_nas(
    file_path: str,
    encoding: str = 'UTF-8'
) -> tuple:
    """解析节点数据（流式）。

    使用流式方式从 NAS 文件解析 GRID 卡片，高效处理大文件。
    支持逗号分隔和固定格式的 Nastran GRID 卡片。

    Args:
        file_path: NAS 文件路径
        encoding: 文件编码

    Returns:
        tuple: (NodeArray, dict) - 节点数组和 node_id_to_index 映射

    Raises:
        NASParseError: 节点解析失败
    """
    x_coords = []
    y_coords = []
    z_coords = []
    node_id_to_index = {}

    grid_pattern_comma = re.compile(
        r'^\s*GRID\s*,\s*\d+\s*,\s*\S*\s*,\s*(.+)$',
        re.IGNORECASE
    )

    node_count = 0
    parse_errors = 0
    skipped_lines = 0
    duplicate_node_ids = 0

    def _record_node(node_id: int, x: float, y: float, z: float) -> None:
        """存储解析的 GRID 卡片，重复 ID 时原地更新。

        重复的节点 ID 以前只会追加第二个条目并将 node_id_to_index
        指向它，把第一次出现的坐标留下作为一个活跃的、未引用的
        节点——使 node_count 膨胀一个断开的幻影点。这里改用 Nastran
        自身的约定（给定 ID 的最后一张卡片获胜）：覆盖现有槽位
        而不是增长数组。
        """
        nonlocal node_count, duplicate_node_ids
        existing_idx = node_id_to_index.get(node_id)
        if existing_idx is not None:
            x_coords[existing_idx] = x
            y_coords[existing_idx] = y
            z_coords[existing_idx] = z
            duplicate_node_ids += 1
            return
        x_coords.append(x)
        y_coords.append(y)
        z_coords.append(z)
        node_id_to_index[node_id] = node_count
        node_count += 1
        if node_count % 10000 == 0:
            logger.debug(f"Parsed {node_count:,} nodes...")

    try:
        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
            for line in f:
                line_stripped = line.strip()
                
                if not line_stripped or line_stripped.startswith('$') or line_stripped.startswith('#'):
                    continue
                
                if not line_stripped.upper().startswith('GRID'):
                    continue
                
                try:
                    parsed = False
                    
                    # 尝试 comma-separated 格式
                    match = grid_pattern_comma.match(line_stripped)
                    
                    if match:
                        all_parts = [p.strip() for p in line_stripped.split(',')]
                        if len(all_parts) >= 6:
                            try:
                                node_id = int(all_parts[1])
                                x = parse_nastran_float(all_parts[3])
                                y = parse_nastran_float(all_parts[4])
                                z = parse_nastran_float(all_parts[5])

                                if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                                    _record_node(node_id, x, y, z)
                                    parsed = True
                            except (ValueError, IndexError) as e:
                                logger.debug(f"Invalid coords: {line_stripped} - {e}")
                                parse_errors += 1

                    if not parsed:
                        # Fixed 格式 parsing
                        if len(line) >= 48:
                            node_id_str = line[8:16].strip()
                            x_str = line[24:32].strip()
                            y_str = line[32:40].strip()
                            z_str = line[40:48].strip()

                            if node_id_str and x_str and y_str and z_str:
                                try:
                                    node_id = int(node_id_str)
                                    x = parse_nastran_float(x_str)
                                    y = parse_nastran_float(y_str)
                                    z = parse_nastran_float(z_str)

                                    if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                                        _record_node(node_id, x, y, z)
                                        parsed = True
                                except (ValueError, IndexError):
                                    pass

                        if not parsed:
                            # Free-field (whitespace-separated, no commas) fallback.
                            # Field layout mirrors the comma format: ID, CP, X, Y, Z
                            # (5 fields when CP is explicit, 4 when CP is omitted).
                            # Blindly treating parts[1] as X - as this used to do -
                            # silently read an explicit CP value as the X coordinate
                            # and shifted Y/Z by one field whenever CP wasn't blank.
                            parts = line_stripped[4:].split()

                            if len(parts) >= 5:
                                try:
                                    node_id = int(parts[0])
                                    x = parse_nastran_float(parts[2])
                                    y = parse_nastran_float(parts[3])
                                    z = parse_nastran_float(parts[4])

                                    if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                                        _record_node(node_id, x, y, z)
                                        parsed = True
                                except (ValueError, IndexError):
                                    skipped_lines += 1
                            elif len(parts) == 4:
                                try:
                                    node_id = int(parts[0])
                                    x = parse_nastran_float(parts[1])
                                    y = parse_nastran_float(parts[2])
                                    z = parse_nastran_float(parts[3])

                                    if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                                        _record_node(node_id, x, y, z)
                                        parsed = True
                                except (ValueError, IndexError):
                                    skipped_lines += 1
                            else:
                                skipped_lines += 1
                    
                    if not parsed:
                        skipped_lines += 1
                        
                except Exception as e:
                    logger.debug(f"Error parsing GRID: {line_stripped} - {e}")
                    parse_errors += 1
    
    except Exception as e:
        raise NASParseError(f"Failed to read nodes: {str(e)}") from e
    
    if node_count == 0:
        logger.error("No valid GRID cards found")
        return (NodeArray(x=np.array([]), y=np.array([]), z=np.array([])), {})

    total_grid_lines = node_count + parse_errors + skipped_lines
    dropped = parse_errors + skipped_lines
    drop_fraction = dropped / total_grid_lines if total_grid_lines else 0.0
    if total_grid_lines >= MIN_LINES_FOR_DROP_CHECK and drop_fraction > MAX_DROP_FRACTION:
        raise NASParseError(
            f"{dropped}/{total_grid_lines} ({drop_fraction:.1%}) GRID lines could "
            f"not be parsed - this exceeds the {MAX_DROP_FRACTION:.0%} threshold "
            f"for incidental noise, and almost always means the file's actual "
            f"GRID card format/encoding doesn't match what this parser expects "
            f"(e.g. wrong column alignment for fixed-width cards, or a wrong "
            f"--encoding). Proceeding would silently produce a mesh missing a "
            f"large fraction of its true node count while still reporting "
            f"'success' - check the file's actual GRID card layout and encoding."
        )

    if parse_errors > 0:
        logger.warning(f"Encountered {parse_errors} parsing errors")
    if skipped_lines > 0:
        logger.info(f"Skipped {skipped_lines} GRID lines")
    if duplicate_node_ids > 0:
        logger.warning(
            f"{duplicate_node_ids} GRID cards reused an already-seen node ID; "
            f"kept the last card's coordinates for each (Nastran convention) "
            f"instead of creating an orphaned duplicate node"
        )
    
    x_array = np.array(x_coords, dtype=np.float64)
    y_array = np.array(y_coords, dtype=np.float64)
    z_array = np.array(z_coords, dtype=np.float64)
    
    logger.info(f"Successfully parsed {node_count:,} nodes")
    
    return (NodeArray(x=x_array, y=y_array, z=z_array), node_id_to_index)
