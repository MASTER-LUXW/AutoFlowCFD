"""NAS 解析器：节点（node）提取。

流式解析 Nastran 格式文件里的 GRID 节点卡片。
"""

import re
import numpy as np
from loguru import logger

from ..structures import NodeArray
from .nas_parser_exceptions import NASParseError
from .nas_parser_utils import parse_nastran_float

# Above this fraction of GRID lines dropped (parse errors + unparseable
# lines), treat it as a systematic format/encoding mismatch rather than
# incidental noise and fail loudly instead of silently returning a mesh
# missing a large chunk of its true geometry while still reporting success.
# Only enforced once there's a meaningful sample size
# (MIN_LINES_FOR_DROP_CHECK) - a handful of GRID lines in a small file
# having one bad line is not evidence of systematic corruption the way the
# same ratio would be across the tens of thousands of lines in a real mesh.
MAX_DROP_FRACTION = 0.05
MIN_LINES_FOR_DROP_CHECK = 20


def parse_nodes_from_nas(
    file_path: str,
    encoding: str = 'UTF-8'
) -> tuple:
    """解析节点数据(流式)
    
    Parses GRID cards from NAS file using streaming approach to handle
    large files efficiently. Supports both comma-separated and fixed-format
    Nastran GRID cards.
    
    Args:
        file_path: Path to NAS file
        encoding: File encoding
        
    Returns:
        tuple: (NodeArray, dict) - Node array and node_id_to_index mapping
        
    Raises:
        NASParseError: If node parsing fails
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
        """Store a parsed GRID card, updating in place on a duplicate ID.

        A repeated node ID used to just append a second entry and repoint
        node_id_to_index at it, leaving the first occurrence's coordinates
        behind as a live, unreferenced node - inflating node_count with a
        disconnected phantom point. Nastran's own convention (last card for
        a given ID wins) is applied here instead: overwrite the existing
        slot rather than growing the array.
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
                    
                    # Try comma-separated format
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
                        # Fixed format parsing
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
