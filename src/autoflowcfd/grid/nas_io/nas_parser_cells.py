"""NAS parser cell extraction.

Provides streaming cell parsing functionality for Nastran format files:
CTRIA3 triangular surface elements natively, and CQUAD4 quadrilateral
surface elements split into 2 triangles each (n1,n2,n3 + n1,n3,n4) - real
ANSA automotive exports routinely mix both element types on doubly-curved
body panels; a CQUAD4-blind parser silently produces a surface mesh with
holes wherever quads were used, with no warning (a CQUAD4 line never
matched any prior pattern here, so it was invisible even to the drop-
fraction safety net below, unlike a genuinely malformed CTRIA3 line).
"""

import re
from typing import Tuple
import numpy as np
from loguru import logger

from ..structures import CellArray
from .nas_parser_exceptions import NASParseError

# Above this fraction of element (CTRIA3/CQUAD4) LINES dropped (parse
# errors, unparseable lines, or dangling node references), treat it as a
# systematic format/encoding mismatch - or GRID/element sections out of
# sync - rather than incidental noise, and fail loudly instead of silently
# returning a surface mesh missing a large chunk of its true geometry while
# still reporting success. Only enforced once there's a meaningful sample
# size (MIN_LINES_FOR_DROP_CHECK) - see the matching constant in
# nas_parser_nodes.py for why. Measured in LINES, not cells: a dropped
# CQUAD4 line loses one card, not the two triangles it would have produced,
# so cell-count-based accounting would understate its weight.
MAX_DROP_FRACTION = 0.05
MIN_LINES_FOR_DROP_CHECK = 20


def parse_cells_from_nas(
    file_path: str,
    node_id_to_index: dict,
    encoding: str = 'UTF-8'
) -> Tuple[CellArray, np.ndarray]:
    """解析单元数据(流式)

    Parses CTRIA3 and CQUAD4 cards from NAS file using streaming approach.
    Supports comma-separated, fixed-format, and whitespace-flexible
    Nastran cards. CQUAD4 quads are split into 2 triangles each (diagonal
    n1-n3), since the rest of this project's surface/volume mesh pipeline
    is triangle-only.

    Args:
        file_path: Path to NAS file
        node_id_to_index: Mapping from NAS node IDs to array indices
        encoding: File encoding

    Returns:
        Tuple[CellArray, np.ndarray]: Parsed cell connectivity/types, and the
        Property ID (PID) for each surviving cell, in the same order and
        length as the CellArray. Cells skipped due to missing node references
        are excluded from both, so index i always refers to the same cell.

    Raises:
        NASParseError: If cell parsing fails
    """
    connectivity_list = []
    cell_types = []
    cell_pids = []
    cell_count = 0
    parse_errors = 0
    skipped_cells = 0
    skipped_lines = 0
    total_lines_seen = 0
    quad_lines_parsed = 0
    eid_to_cell_idx: dict = {}
    duplicate_eids = 0

    def _record_triangle(key, pid: int, idx1: int, idx2: int, idx3: int) -> None:
        """Store one parsed triangle (a CTRIA3 card, or one half of a split
        CQUAD4) under `key`, applying Nastran's documented "last element ID
        wins" convention on a repeated key instead of appending a second,
        coincident triangle (which previously happened silently - the EID
        was parsed but never tracked, so re-exported or duplicated element
        cards inflated cell_count and could leave a non-manifold surface
        for the volume mesher).

        `key` is the bare int EID for a CTRIA3, or an (eid, 0|1) tuple for
        a CQUAD4's two sub-triangles - Nastran EIDs are unique per element
        (not per triangle), so a re-issued CQUAD4 with the same EID must
        overwrite both of its own previous sub-triangles, not collide with
        an unrelated CTRIA3 that happens to share the bare int value.
        """
        nonlocal cell_count, duplicate_eids
        existing_idx = eid_to_cell_idx.get(key)
        if existing_idx is not None:
            connectivity_list[existing_idx] = [idx1, idx2, idx3]
            cell_pids[existing_idx] = pid
            duplicate_eids += 1
            return
        connectivity_list.append([idx1, idx2, idx3])
        cell_types.append(0)
        cell_pids.append(pid)
        eid_to_cell_idx[key] = cell_count
        cell_count += 1
        if cell_count % 10000 == 0:
            logger.debug(f"Parsed {cell_count:,} cells...")

    def _record_quad(eid: int, pid: int, n1: int, n2: int, n3: int, n4: int) -> bool:
        """Split a CQUAD4's 4 nodes into 2 triangles (n1,n2,n3) and
        (n1,n3,n4) and record both. Returns False (recording nothing) if
        any of the 4 nodes is missing - the whole quad is skipped as one
        unit, not partially recorded, to avoid a torn/self-overlapping
        surface."""
        if not (n1 in node_id_to_index and n2 in node_id_to_index
                and n3 in node_id_to_index and n4 in node_id_to_index):
            return False
        i1 = node_id_to_index[n1]
        i2 = node_id_to_index[n2]
        i3 = node_id_to_index[n3]
        i4 = node_id_to_index[n4]
        _record_triangle((eid, 0), pid, i1, i2, i3)
        _record_triangle((eid, 1), pid, i1, i3, i4)
        return True

    ctria3_pattern_comma = re.compile(
        r'^\s*CTRIA3\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)',
        re.IGNORECASE
    )
    ctria3_pattern_fixed = re.compile(
        r'^\s*CTRIA3\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)',
        re.IGNORECASE
    )
    cquad4_pattern_comma = re.compile(
        r'^\s*CQUAD4\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)',
        re.IGNORECASE
    )
    cquad4_pattern_fixed = re.compile(
        r'^\s*CQUAD4\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)',
        re.IGNORECASE
    )

    try:
        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
            for line in f:
                line_stripped = line.strip()

                if not line_stripped or line_stripped.startswith('$') or line_stripped.startswith('#'):
                    continue

                line_upper = line_stripped.upper()
                is_quad = line_upper.startswith('CQUAD4')
                if not is_quad and not line_upper.startswith('CTRIA3'):
                    continue

                total_lines_seen += 1

                try:
                    if is_quad:
                        match = cquad4_pattern_comma.match(line_stripped) \
                            or cquad4_pattern_fixed.match(line_stripped)
                        if match:
                            eid, pid, n1, n2, n3, n4 = (int(g) for g in match.groups())
                            if _record_quad(eid, pid, n1, n2, n3, n4):
                                quad_lines_parsed += 1
                            else:
                                skipped_cells += 1
                            continue

                        # Flexible parsing (whitespace-tokenized fallback)
                        parts = line_stripped[6:].split()
                        if len(parts) >= 6:
                            try:
                                eid, pid, n1, n2, n3, n4 = (int(p) for p in parts[:6])
                                if _record_quad(eid, pid, n1, n2, n3, n4):
                                    quad_lines_parsed += 1
                                else:
                                    skipped_cells += 1
                            except (ValueError, IndexError):
                                parse_errors += 1
                        else:
                            skipped_lines += 1
                        continue

                    # Try comma-separated format
                    match = ctria3_pattern_comma.match(line_stripped)

                    if match:
                        eid = int(match.group(1))
                        pid = int(match.group(2))
                        n1 = int(match.group(3))
                        n2 = int(match.group(4))
                        n3 = int(match.group(5))

                        if n1 in node_id_to_index and n2 in node_id_to_index and n3 in node_id_to_index:
                            idx1 = node_id_to_index[n1]
                            idx2 = node_id_to_index[n2]
                            idx3 = node_id_to_index[n3]
                            _record_triangle(eid, pid, idx1, idx2, idx3)
                            continue
                        else:
                            skipped_cells += 1
                            continue

                    # Try fixed-format
                    match = ctria3_pattern_fixed.match(line_stripped)

                    if match:
                        eid = int(match.group(1))
                        pid = int(match.group(2))
                        n1 = int(match.group(3))
                        n2 = int(match.group(4))
                        n3 = int(match.group(5))

                        if n1 in node_id_to_index and n2 in node_id_to_index and n3 in node_id_to_index:
                            idx1 = node_id_to_index[n1]
                            idx2 = node_id_to_index[n2]
                            idx3 = node_id_to_index[n3]
                            _record_triangle(eid, pid, idx1, idx2, idx3)
                            continue
                        else:
                            skipped_cells += 1
                            continue

                    # Flexible parsing
                    parts = line_stripped[6:].split()

                    if len(parts) >= 5:
                        try:
                            eid = int(parts[0])
                            pid = int(parts[1])
                            n1 = int(parts[2])
                            n2 = int(parts[3])
                            n3 = int(parts[4])

                            if n1 in node_id_to_index and n2 in node_id_to_index and n3 in node_id_to_index:
                                idx1 = node_id_to_index[n1]
                                idx2 = node_id_to_index[n2]
                                idx3 = node_id_to_index[n3]
                                _record_triangle(eid, pid, idx1, idx2, idx3)
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
                    logger.debug(f"Error parsing element card: {line_stripped} - {e}")
                    parse_errors += 1

    except Exception as e:
        raise NASParseError(f"Failed to read cells: {str(e)}") from e

    if cell_count == 0:
        logger.error("No valid CTRIA3/CQUAD4 cards found")
        return CellArray(
            connectivity=np.array([], dtype=np.int32).reshape(0, 3),
            cell_type=np.array([], dtype=np.int32)
        ), np.array([], dtype=np.int32)

    dropped = parse_errors + skipped_lines + skipped_cells
    drop_fraction = dropped / total_lines_seen if total_lines_seen else 0.0
    if total_lines_seen >= MIN_LINES_FOR_DROP_CHECK and drop_fraction > MAX_DROP_FRACTION:
        raise NASParseError(
            f"{dropped}/{total_lines_seen} ({drop_fraction:.1%}) CTRIA3/CQUAD4 "
            f"lines could not be parsed or reference missing nodes - this "
            f"exceeds the {MAX_DROP_FRACTION:.0%} threshold for incidental "
            f"noise. A large dangling-node-reference count ({skipped_cells} "
            f"cells skipped) usually means the GRID and element sections are "
            f"out of sync (e.g. nodes parsed with a different ID range/format "
            f"than the elements reference). Proceeding would silently produce "
            f"a surface mesh missing a large fraction of its true geometry "
            f"while still reporting 'success' - check the file's GRID/element "
            f"card layout and encoding."
        )

    if parse_errors > 0:
        logger.warning(f"Encountered {parse_errors} parsing errors")
    if skipped_lines > 0:
        logger.info(f"Skipped {skipped_lines} element lines")
    if skipped_cells > 0:
        logger.info(f"Skipped {skipped_cells} elements due to missing nodes")
    if quad_lines_parsed > 0:
        logger.info(f"Split {quad_lines_parsed} CQUAD4 quads into {quad_lines_parsed * 2} triangles")
    if duplicate_eids > 0:
        logger.warning(
            f"{duplicate_eids} element cards reused an already-seen element ID; "
            f"kept the last card's connectivity for each (Nastran convention) "
            f"instead of creating a duplicate coincident triangle"
        )

    connectivity_array = np.array(connectivity_list, dtype=np.int32)
    cell_type_array = np.array(cell_types, dtype=np.int32)
    pid_array = np.array(cell_pids, dtype=np.int32)

    logger.info(f"Successfully parsed {cell_count:,} cells")

    return CellArray(connectivity=connectivity_array, cell_type=cell_type_array), pid_array
