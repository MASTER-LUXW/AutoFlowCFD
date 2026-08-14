"""NAS 解析器：边界条件提取。

提供从 NAS 文件解析边界条件的专用函数，包括 Property Name 识别与边界分组。
"""

import re
from typing import Dict, List, Tuple
import numpy as np
from loguru import logger

from ..structures import BoundaryMap
from .nas_parser_exceptions import NASParseError

# Name used for the catch-all group holding cells whose Property ID could
# not be resolved to a name (no $ANSA_NAME_COMMENT and no parseable PSHELL
# comment). Without this bucket such cells silently vanished from every
# boundary group - they still existed in the mesh but had no boundary
# condition assigned at all.
_UNCLASSIFIED_GROUP = "UNCLASSIFIED"

# Shell-like property card types recognized for boundary naming. PCOMP/
# PCOMPG (composite/layered shells - common for painted or composite body
# panels) previously fell straight through to _UNCLASSIFIED_GROUP/WALL with
# no indication that it was a systematic property-type gap rather than a
# genuinely unnamed property, since only PSHELL was ever checked.
_SHELL_PROPERTY_TYPES = {'PSHELL', 'PCOMP', 'PCOMPG'}


def _leading_card_name(line_stripped: str) -> str:
    """Extract a Nastran card's leading keyword (e.g. "PSHELL" from both
    "PSHELL,1,1001,0.001" and "PSHELL       1      1      1.") - the card
    name is whatever comes before the first comma or whitespace, whichever
    is first."""
    return line_stripped.split(',', 1)[0].split()[0].upper() if line_stripped else ''


def parse_boundary_properties(
    file_path: str,
    encoding: str,
    cell_count: int,
    cells_data: list
) -> BoundaryMap:
    """解析边界条件（v2.0：支持Properties Name识别）

    Parses boundary condition information from NAS file using Properties (PSHELL) Name.
    Supports three modes: auto, manual, hybrid.

    Args:
        file_path: Path to NAS file
        encoding: File encoding
        cell_count: Total number of cells parsed from the file, used to warn
            when some cells could not be placed into any boundary group
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

        # Step 2: Build cell_index to PID mapping from the pre-parsed data
        cell_to_pid = _parse_cell_to_pid_mapping(cells_data)

        # Step 3: Group cells by Property ID
        pid_to_cells = _group_cells_by_pid(cell_to_pid)

        # Step 4: Map Property Names to boundary groups
        groups, bc_types, property_ids = _map_properties_to_boundaries(
            pid_to_name, pid_to_cells
        )

        # Cells whose PID never resolved to a name (or whose PID wasn't in
        # pid_to_name at all) would otherwise disappear from every boundary
        # group. Collect them into an UNCLASSIFIED bucket instead of
        # silently dropping them, and warn loudly - a solver that later
        # can't find a BC for some cells needs to know why.
        classified_cells = set()
        for indices in groups.values():
            classified_cells.update(indices)
        unclassified = sorted(
            idx for idx in cell_to_pid if idx not in classified_cells
        )
        if unclassified:
            groups[_UNCLASSIFIED_GROUP] = unclassified
            bc_types[_UNCLASSIFIED_GROUP] = 'WALL'
            logger.warning(
                f"{len(unclassified)} cells could not be mapped to a named "
                f"Property (missing $ANSA_NAME_COMMENT/PSHELL name) and were "
                f"placed in the '{_UNCLASSIFIED_GROUP}' group as WALL"
            )
        if cell_count and len(cell_to_pid) < cell_count:
            logger.warning(
                f"Only {len(cell_to_pid)}/{cell_count} cells resolved a "
                f"Property ID during boundary parsing; the remainder have "
                f"no boundary group at all"
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

                        # PSHELL, and composite/layered shells (PCOMP/
                        # PCOMPG) - common for painted or composite body
                        # panels, otherwise silently degraded to WALL via
                        # _UNCLASSIFIED_GROUP with no indication it was a
                        # property-TYPE gap rather than a genuinely unnamed
                        # property.
                        if prop_type in _SHELL_PROPERTY_TYPES and prop_name:
                            pid_to_name[pid] = prop_name
                            logger.debug(f"Found Property: PID={pid}, Name='{prop_name}'")
                    except (ValueError, IndexError):
                        pass
    
    return pid_to_name


def _parse_pshell_names(file_path: str, encoding: str) -> Dict[int, str]:
    """Parse PSHELL/PCOMP/PCOMPG cards with comment-based naming.

    Handles both comma-separated free-field cards (``PSHELL,1,1001,0.001``)
    and fixed-width small-field cards (``PSHELL       1      1      1.``,
    no commas - the format this project's own nas_export.py writes when it
    isn't paired with an $ANSA_NAME_COMMENT). A comma-only split previously
    matched neither the fixed-width case nor a genuinely bare PID column,
    silently returning no names for any such file.
    """
    pid_to_name = {}

    with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
        prev_line = ""
        for line in f:
            line_stripped = line.strip()

            if not line_stripped or line_stripped.startswith('#'):
                continue

            # Check for a shell-like property card (PSHELL, or composite/
            # layered PCOMP/PCOMPG - same fallback naming convention).
            if _leading_card_name(line_stripped) in _SHELL_PROPERTY_TYPES:
                # Try to get name from previous comment line
                # Format: $ PROPERTY NAME: XXXX
                if prev_line.startswith('$'):
                    match = re.search(r'PROPERTY\s+NAME:\s*(\S+)', prev_line, re.IGNORECASE)
                    if match:
                        # Comma-separated free-field form first...
                        parts = [p.strip() for p in line_stripped.split(',') if p.strip()]
                        if len(parts) < 2:
                            # ...fall back to whitespace-split fixed-width form.
                            parts = line_stripped.split()
                        if len(parts) >= 2:
                            try:
                                pid = int(parts[1])
                                pid_to_name[pid] = match.group(1)
                            except ValueError:
                                pass

            prev_line = line_stripped

    return pid_to_name


def _parse_cell_to_pid_mapping(cells_data: list) -> Dict[int, int]:
    """把预先解析好的 [cell_index, pid] 列表转换成 cell_index -> pid 字典。

    唯一调用方 parser_core.py 总是传入已经和 parse_cells_from_nas 结果对齐的
    cells_data（见调用处注释：独立重新扫描 CTRIA3 卡片无法知道哪些 cell 被
    跳过，索引会错位），因此这里不再保留"文件里独立重新扫描 CTRIA3"的分支。
    """
    return {cell_idx: pid for cell_idx, pid in cells_data}


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
        # 周期边界（见 grid/face_connectivity.py::pair_periodic_boundary_faces）
        # 只能靠属性名关键字识别出"这是一对周期面"，无法从几何/NAS 文件本身
        # 反推出配对的另一侧组名与平移向量——这两项必须通过 YAML 手动/混合
        # 配置补齐（写入 BoundaryMap.parameters[name]['paired_with'/'translation']），
        # 纯 NAS 自动模式无法单独完成周期边界的完整配置。
        'PERIODIC': ['periodic', '周期'],
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

        # Merge into the group rather than overwrite it: ANSA exports
        # routinely split one logical boundary (e.g. "WALL") across several
        # PIDs that share the same Property Name. Assigning instead of
        # extending here used to silently drop every PID's cells except the
        # last one processed for a given name.
        if name in groups:
            groups[name].extend(pid_to_cells[pid])
            if property_ids.get(name) != pid:
                logger.debug(
                    f"Property '{name}' spans multiple PIDs "
                    f"({property_ids.get(name)}, {pid}); merging cells "
                    f"into one boundary group"
                )
        else:
            groups[name] = list(pid_to_cells[pid])
            property_ids[name] = pid
        bc_types[name] = bc_type

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
