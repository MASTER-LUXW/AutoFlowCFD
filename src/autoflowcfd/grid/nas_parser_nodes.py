"""NAS parser node extraction.

Provides streaming node parsing functionality for Nastran format files.
"""

import re
import numpy as np
from loguru import logger

from .structures import NodeArray
from .nas_parser_exceptions import NASParseError
from .nas_parser_utils import parse_nastran_float


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
                                    x_coords.append(x)
                                    y_coords.append(y)
                                    z_coords.append(z)
                                    node_id_to_index[node_id] = node_count
                                    node_count += 1
                                    parsed = True
                                    
                                    if node_count % 10000 == 0:
                                        logger.debug(f"Parsed {node_count:,} nodes...")
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
                                        x_coords.append(x)
                                        y_coords.append(y)
                                        z_coords.append(z)
                                        node_id_to_index[node_id] = node_count
                                        node_count += 1
                                        parsed = True
                                except (ValueError, IndexError):
                                    pass
                        
                        if not parsed:
                            parts = line_stripped[4:].split()
                            
                            if len(parts) >= 4:
                                try:
                                    node_id = int(parts[0])
                                    x = parse_nastran_float(parts[1])
                                    y = parse_nastran_float(parts[2])
                                    z = parse_nastran_float(parts[3])
                                    
                                    if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                                        x_coords.append(x)
                                        y_coords.append(y)
                                        z_coords.append(z)
                                        node_id_to_index[node_id] = node_count
                                        node_count += 1
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
    
    if parse_errors > 0:
        logger.warning(f"Encountered {parse_errors} parsing errors")
    if skipped_lines > 0:
        logger.info(f"Skipped {skipped_lines} GRID lines")
    
    x_array = np.array(x_coords, dtype=np.float64)
    y_array = np.array(y_coords, dtype=np.float64)
    z_array = np.array(z_coords, dtype=np.float64)
    
    logger.info(f"Successfully parsed {node_count:,} nodes")
    
    return (NodeArray(x=x_array, y=y_array, z=z_array), node_id_to_index)
