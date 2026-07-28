"""NAS parser cell extraction.

Provides streaming cell parsing functionality for Nastran format files,
specifically CTRIA3 triangular surface elements.
"""

import re
import numpy as np
from loguru import logger

from .structures import CellArray
from .nas_parser_exceptions import NASParseError


def parse_cells_from_nas(
    file_path: str,
    node_id_to_index: dict,
    encoding: str = 'UTF-8'
) -> CellArray:
    """解析单元数据(流式)
    
    Parses CTRIA3 cards from NAS file using streaming approach.
    Supports both comma-separated and fixed-format Nastran CTRIA3 cards.
    
    Args:
        file_path: Path to NAS file
        node_id_to_index: Mapping from NAS node IDs to array indices
        encoding: File encoding
        
    Returns:
        CellArray: Parsed cell connectivity and types
        
    Raises:
        NASParseError: If cell parsing fails
    """
    connectivity_list = []
    cell_types = []
    cell_count = 0
    parse_errors = 0
    skipped_cells = 0
    skipped_lines = 0
    
    ctria3_pattern_comma = re.compile(
        r'^\s*CTRIA3\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)',
        re.IGNORECASE
    )
    ctria3_pattern_fixed = re.compile(
        r'^\s*CTRIA3\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)',
        re.IGNORECASE
    )
    
    try:
        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
            for line in f:
                line_stripped = line.strip()
                
                if not line_stripped or line_stripped.startswith('$') or line_stripped.startswith('#'):
                    continue
                
                if not line_stripped.upper().startswith('CTRIA3'):
                    continue
                
                try:
                    # Try comma-separated format
                    match = ctria3_pattern_comma.match(line_stripped)
                    
                    if match:
                        n1 = int(match.group(3))
                        n2 = int(match.group(4))
                        n3 = int(match.group(5))
                        
                        if n1 in node_id_to_index and n2 in node_id_to_index and n3 in node_id_to_index:
                            idx1 = node_id_to_index[n1]
                            idx2 = node_id_to_index[n2]
                            idx3 = node_id_to_index[n3]
                            connectivity_list.append([idx1, idx2, idx3])
                            cell_types.append(0)
                            cell_count += 1
                            
                            if cell_count % 10000 == 0:
                                logger.debug(f"Parsed {cell_count:,} cells...")
                            continue
                        else:
                            skipped_cells += 1
                            continue
                    
                    # Try fixed-format
                    match = ctria3_pattern_fixed.match(line_stripped)
                    
                    if match:
                        n1 = int(match.group(3))
                        n2 = int(match.group(4))
                        n3 = int(match.group(5))
                        
                        if n1 in node_id_to_index and n2 in node_id_to_index and n3 in node_id_to_index:
                            idx1 = node_id_to_index[n1]
                            idx2 = node_id_to_index[n2]
                            idx3 = node_id_to_index[n3]
                            connectivity_list.append([idx1, idx2, idx3])
                            cell_types.append(0)
                            cell_count += 1
                            
                            if cell_count % 10000 == 0:
                                logger.debug(f"Parsed {cell_count:,} cells...")
                            continue
                        else:
                            skipped_cells += 1
                            continue
                    
                    # Flexible parsing
                    parts = line_stripped[6:].split()
                    
                    if len(parts) >= 5:
                        try:
                            n1 = int(parts[2])
                            n2 = int(parts[3])
                            n3 = int(parts[4])
                            
                            if n1 in node_id_to_index and n2 in node_id_to_index and n3 in node_id_to_index:
                                idx1 = node_id_to_index[n1]
                                idx2 = node_id_to_index[n2]
                                idx3 = node_id_to_index[n3]
                                connectivity_list.append([idx1, idx2, idx3])
                                cell_types.append(0)
                                cell_count += 1
                                
                                if cell_count % 10000 == 0:
                                    logger.debug(f"Parsed {cell_count:,} cells...")
                                continue
                            else:
                                skipped_cells += 1
                                continue
                        except (ValueError, IndexError):
                            parse_errors += 1
                            continue
                    else:
                        skipped_lines += 1
                        continue
                    
                except ValueError as e:
                    logger.debug(f"Invalid node ID: {line_stripped} - {e}")
                    parse_errors += 1
                except Exception as e:
                    logger.debug(f"Error parsing CTRIA3: {line_stripped} - {e}")
                    parse_errors += 1
    
    except Exception as e:
        raise NASParseError(f"Failed to read cells: {str(e)}") from e
    
    if cell_count == 0:
        logger.error("No valid CTRIA3 cards found")
        return CellArray(
            connectivity=np.array([], dtype=np.int32).reshape(0, 3),
            cell_type=np.array([], dtype=np.int32)
        )
    
    if parse_errors > 0:
        logger.warning(f"Encountered {parse_errors} parsing errors")
    if skipped_lines > 0:
        logger.info(f"Skipped {skipped_lines} CTRIA3 lines")
    if skipped_cells > 0:
        logger.info(f"Skipped {skipped_cells} CTRIA3 cells due to missing nodes")
    
    connectivity_array = np.array(connectivity_list, dtype=np.int32)
    cell_type_array = np.array(cell_types, dtype=np.int32)
    
    logger.info(f"Successfully parsed {cell_count:,} cells")
    
    return CellArray(connectivity=connectivity_array, cell_type=cell_type_array)
