"""NAS parser boundary condition extraction.

Provides specialized functions for parsing boundary conditions from NAS files,
including Property Name detection and boundary group mapping.
"""

import re
from typing import Dict, List, Optional, Tuple
import numpy as np
from loguru import logger

from ..structures import BoundaryMap
from .nas_parser_exceptions import NASParseError


def parse_boundary_properties(
    file_path: str,
    encoding: str = 'UTF-8',
    cell_count: int = 0,
    node_id_to_index: dict = None,
    cells_data: list = None
) -> BoundaryMap:
    """解析边界条件（v2.0：支持Properties Name识别）
    
    Parses boundary condition information from NAS file using Properties (PSHELL) Name.
    Supports three modes: auto, manual, hybrid.
    
    Args:
        file_path: Path to NAS file
        encoding: File encoding
        cell_count: Total number of cells parsed from the file
        node_id_to_index: Mapping from NAS node IDs to array indices
        cells_data: List of [cell_index, pid] pairs for mapping cells to properties
        
    Returns:
        BoundaryMap: Parsed boundary groups and BC types with Property information
        
    Raises:
        NASParseError: If boundary parsing fails
    """
    groups: Dict[str, np.ndarray] = {}
    bc_types: Dict[str, str] = {}
    property_ids: Dict[str, int] = {}
    property_names: Dict[int, str] = {}
    
    logger.info("Parsing boundary conditions from Properties...")
    
    try:
        # Step 1: Parse $ANSA_NAME_COMMENT cards to extract PID to Name mapping
        pid_to_name = _parse_property_names(file_path, encoding)
        
        # If no ANSA_NAME_COMMENT found, try parsing PSHELL cards directly
        if not pid_to_name:
            logger.warning("No $ANSA_NAME_COMMENT cards found. Trying alternative parsing...")
            pid_to_name = _parse_pshell_names(file_path, encoding)
        
        logger.info(f"Found {len(pid_to_name)} Properties with names")
        
        # Step 2: Parse CTRIA3 cards again to build cell_index to PID mapping
        cell_to_pid = _parse_cell_to_pid_mapping(
            file_path, encoding, node_id_to_index, cells_data
        )
        
        # Step 3: Group cells by Property ID
        pid_to_cells = _group_cells_by_pid(cell_to_pid)
        
        # Step 4: Map Property Names to boundary groups
        groups, bc_types, property_ids = _map_properties_to_boundaries(
            pid_to_name, pid_to_cells
        )
        
        # Convert lists to numpy arrays
        for name in groups:
            groups[name] = np.array(groups[name], dtype=np.int32)
        
        logger.info(f"Parsed {len(groups)} boundary groups")
        
        return BoundaryMap(
            groups=groups,
            bc_types=bc_types,
            property_ids=property_ids,
            property_names=property_names,
            detection_mode="auto",
            parameters={}
        )
        
    except Exception as e:
        raise NASParseError(f"Failed to parse boundaries: {str(e)}") from e


def _parse_property_names(file_path: str, encoding: str) -> Dict[int, str]:
    """Parse $ANSA_NAME_COMMENT cards to extract PID to Name mapping."""
    pid_to_name = {}
    
    with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
        for line in f:
            line_stripped = line.strip()
            
            if not line_stripped or not line_stripped.startswith('$'):
                continue
            
            # Check for ANSA_NAME_COMMENT card
            # Format: $ANSA_NAME_COMMENT;PID;PSHELL;name;;NO;NO;NO;NO;
            if line_stripped.upper().startswith('$ANSA_NAME_COMMENT'):
                parts = line_stripped.split(';')
                if len(parts) >= 5:
                    try:
                        pid = int(parts[1])
                        prop_type = parts[2].strip().upper()
                        prop_name = parts[3].strip()
                        
                        # Only process PSHELL properties
                        if prop_type == 'PSHELL' and prop_name:
                            pid_to_name[pid] = prop_name
                            logger.debug(f"Found Property: PID={pid}, Name='{prop_name}'")
                    except (ValueError, IndexError):
                        pass
    
    return pid_to_name


def _parse_pshell_names(file_path: str, encoding: str) -> Dict[int, str]:
    """Parse PSHELL cards with comment-based naming."""
    pid_to_name = {}
    
    with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
        prev_line = ""
        for line in f:
            line_stripped = line.strip()
            
            if not line_stripped or line_stripped.startswith('#'):
                continue
            
            # Check for PSHELL card
            if line_stripped.upper().startswith('PSHELL'):
                # Try to get name from previous comment line
                # Format: $ PROPERTY NAME: XXXX
                if prev_line.startswith('$'):
                    match = re.search(r'PROPERTY\s+NAME:\s*(\S+)', prev_line, re.IGNORECASE)
                    if match:
                        parts = [p.strip() for p in line_stripped.split(',') if p.strip()]
                        if len(parts) >= 2:
                            try:
                                pid = int(parts[1])
                                pid_to_name[pid] = match.group(1)
                            except ValueError:
                                pass
            
            prev_line = line_stripped
    
    return pid_to_name


def _parse_cell_to_pid_mapping(
    file_path: str,
    encoding: str,
    node_id_to_index: dict = None,
    cells_data: list = None
) -> Dict[int, int]:
    """Parse CTRIA3 cards to build cell_index to PID mapping."""
    cell_to_pid = {}
    
    if cells_data is not None:
        # Use pre-parsed data
        for cell_idx, pid in cells_data:
            cell_to_pid[cell_idx] = pid
        return cell_to_pid
    
    # Re-parse CTRIA3 cards to extract PID information
    cell_idx = 0
    
    # Support both comma-separated and fixed-format (space-separated)
    ctria3_pattern_comma = re.compile(
        r'^\s*CTRIA3\s*,\s*(\d+)\s*,\s*(\d+)\s*,',
        re.IGNORECASE
    )
    ctria3_pattern_fixed = re.compile(
        r'^\s*CTRIA3\s+(\d+)\s+(\d+)\s+',
        re.IGNORECASE
    )
    
    with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
        for line in f:
            line_stripped = line.strip()
            
            if not line_stripped or line_stripped.startswith('$') or line_stripped.startswith('#'):
                continue
            
            # Check if this is a CTRIA3 card
            if not line_stripped.upper().startswith('CTRIA3'):
                continue
            
            try:
                # Try comma-separated format first
                match = ctria3_pattern_comma.match(line_stripped)
                
                if match:
                    eid = int(match.group(1))
                    pid = int(match.group(2))
                    cell_to_pid[cell_idx] = pid
                    cell_idx += 1
                    continue
                
                # Try fixed-format
                match = ctria3_pattern_fixed.match(line_stripped)
                
                if match:
                    eid = int(match.group(1))
                    pid = int(match.group(2))
                    cell_to_pid[cell_idx] = pid
                    cell_idx += 1
                    continue
                    
            except (ValueError, IndexError):
                continue
    
    return cell_to_pid


def _group_cells_by_pid(cell_to_pid: Dict[int, int]) -> Dict[int, List[int]]:
    """Group cells by Property ID."""
    pid_to_cells = {}
    
    for cell_idx, pid in cell_to_pid.items():
        if pid not in pid_to_cells:
            pid_to_cells[pid] = []
        pid_to_cells[pid].append(cell_idx)
    
    return pid_to_cells


def _map_properties_to_boundaries(
    pid_to_name: Dict[int, str],
    pid_to_cells: Dict[int, List[int]]
) -> Tuple[Dict[str, List[int]], Dict[str, str], Dict[str, int]]:
    """Map Property Names to boundary groups with BC type detection."""
    groups = {}
    bc_types = {}
    property_ids = {}
    
    # Boundary keyword mapping for automatic detection. Order matters:
    # _detect_boundary_type returns the FIRST matching bc_type, so more
    # specific keyword sets are listed before the generic 'WALL' bucket -
    # otherwise a compound name like "TUNNEL_WALL" would match the plain
    # 'wall' substring before ever reaching the 'tunnel' keyword below.
    boundary_keywords = {
        'VELOCITY_INLET': ['inlet', 'inflow', 'entrance', '入口'],
        'PRESSURE_OUTLET': ['outlet', 'outflow', 'exit', '出口'],
        'SYMMETRY': ['symmetry', 'sym', '对称'],
        # A "tunnel" boundary is a frictionless duct wall (see
        # bc_handler.py's _classify: TUNNEL -> SYMMETRY/free-slip), not a
        # viscous no-slip wall - it must never get BL extrusion (there is
        # no velocity gradient at a slip wall to resolve). Previously
        # "tunnel" matched none of these keywords and silently fell
        # through to the 'WALL' default below, making it BL-extrude-
        # eligible - extruding a boundary layer on a domain-spanning
        # tunnel wall collapses almost immediately (hits the opposite
        # wall/body within 1-2 layers), producing hundreds of degenerate
        # tetrahedra and a non-manifold surface that crashes tetgen.
        'SLIP_WALL': ['slip', 'farfield', 'freestream', 'tunnel', '风洞', '洞壁'],
        'WALL': ['wall', 'body', 'surface', '车体', '车身', '壁面'],
    }
    
    for pid, name in pid_to_name.items():
        if pid not in pid_to_cells:
            continue
        
        # Determine boundary type based on name
        bc_type = _detect_boundary_type(name, boundary_keywords)
        
        # Add to groups
        groups[name] = pid_to_cells[pid]
        bc_types[name] = bc_type
        property_ids[name] = pid
        
        logger.debug(f"Mapped Property '{name}' (PID={pid}) to {bc_type}")
    
    return groups, bc_types, property_ids


def _detect_boundary_type(name: str, keywords: Dict[str, List[str]]) -> str:
    """Detect boundary type from Property Name using keyword matching."""
    name_lower = name.lower()
    
    for bc_type, keyword_list in keywords.items():
        for keyword in keyword_list:
            if keyword.lower() in name_lower:
                return bc_type
    
    # Default to WALL if no match
    return 'WALL'
