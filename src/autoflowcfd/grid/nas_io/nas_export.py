"""NAS file export module.

Exports VolumeMeshData to Nastran (.nas) format for visualization and post-processing.

Key Components:
    - export_volume_mesh_to_nas: Main export function
    - _write_header: Write NAS file header
    - _write_nodes: Write GRID cards
    - _write_tetrahedra: Write CTETRA cards
    - _write_boundaries: Write boundary group information
"""

import math
import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path
from loguru import logger


def _format_nastran_compact_exponent(value: float, width: int = 8) -> str:
    """Format a float in Nastran's compact exponent notation (no 'e'),
    e.g. "-1.23+05" for -123000.0, guaranteed to fit within `width` chars.

    This is what nas_parser_utils.parse_nastran_float already knows how to
    read back, so a round-tripped file stays self-consistent. It exists as
    the last-resort fallback in format_coord_8char below, for coordinates
    where even 2-decimal fixed notation doesn't fit an 8-character Nastran
    Small Field - which Python's "%e" (used previously) does not fit
    either: "-5.0000e+04" is 11 characters, itself overflowing the field
    it was supposed to protect. Automotive external-aero domains routinely
    have +/-tens-of-meters extents, i.e. +/-tens-of-thousands of mm once
    exported at scale_factor=1000, so this path is realistically reachable,
    not just a theoretical edge case.
    """
    if value == 0.0:
        return "0.0"

    sign = '-' if value < 0 else ''
    abs_value = abs(value)
    exponent = int(math.floor(math.log10(abs_value)))
    mantissa = abs_value / (10.0 ** exponent)
    # Guard against log10 rounding landing exactly on a power-of-ten boundary.
    if mantissa >= 10.0:
        mantissa /= 10.0
        exponent += 1
    elif mantissa < 1.0:
        mantissa *= 10.0
        exponent -= 1

    def _render(mantissa: float, exponent: int) -> str:
        exp_sign = '+' if exponent >= 0 else '-'
        exp_str = f"{abs(exponent):02d}"
        avail = width - len(sign) - 1 - len(exp_str)  # 1 for exp_sign
        decimals = max(avail - 2, 0)  # 2 = one leading digit + '.'
        mantissa_str = f"{mantissa:.{decimals}f}" if decimals > 0 else f"{mantissa:.0f}"
        return mantissa_str, exp_sign, exp_str

    mantissa_str, exp_sign, exp_str = _render(mantissa, exponent)
    if float(mantissa_str) >= 10.0:
        # Rounding pushed the mantissa back up to two digits; re-render one
        # exponent higher so the field width budget stays correct.
        exponent += 1
        mantissa /= 10.0
        mantissa_str, exp_sign, exp_str = _render(mantissa, exponent)

    result = f"{sign}{mantissa_str}{exp_sign}{exp_str}"
    if len(result) > width:
        # Only reachable for 3+ digit exponents (|value| >= 1e100 or
        # <= 1e-100) - nonsensical for physical mesh coordinates, but clip
        # rather than silently overflow the fixed-width field.
        result = result[:width]
    return result


def export_volume_mesh_to_nas(
    volume_mesh,
    output_path: str,
    include_boundaries: bool = True,
    scale_factor: float = 1.0
) -> str:
    """Export VolumeMeshData to Nastran (.nas) format.
    
    Converts tetrahedral volume mesh to Nastran format with:
    - GRID cards for nodes (default: meters, same as input)
    - CTETRA cards for tetrahedral elements
    - PSHELL/PSET cards for boundary groups (optional)
    
    Args:
        volume_mesh: VolumeMeshData object with nodes, cells, boundaries
        output_path: Output file path (.nas extension)
        include_boundaries: Whether to include boundary group info
        scale_factor: Coordinate scaling factor (default 1.0 for meters)
                      Use 1000.0 to convert to millimeters if needed
        
    Returns:
        str: Path to exported file
        
    Example:
        >>> from autoflowcfd.grid import NASParser
        >>> parser = NASParser('surface.nas')
        >>> volume_mesh = parser.parse(generate_volume_mesh=True)
        >>> export_volume_mesh_to_nas(volume_mesh, 'volume_mesh.nas')
    """
    output_path = Path(output_path)

    # Ensure .nas extension
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
    # PSHELL PIDs 1..n_boundary_groups are used for boundary groups below, so the
    # PSOLID property for the volume mesh must live past that range - otherwise
    # it collides with a boundary's PSHELL PID as soon as there are >= 4 groups
    # (a very common case: inlet/outlet/wall/symmetry/ground).
    solid_pid = n_boundary_groups + 1

    try:
        with open(output_path, 'w') as f:
            # Write header
            _write_header(f, volume_mesh)

            # Write nodes (GRID cards)
            logger.info("Writing nodes...")
            _write_nodes(f, volume_mesh.nodes, scale_factor)

            # Write tetrahedral elements (CTETRA cards)
            logger.info("Writing tetrahedral elements...")
            n_tets = _write_tetrahedra(f, volume_mesh.cells.connectivity, solid_pid)

            # Write boundary information (optional): boundary faces as CTRIA3
            # elements referencing per-group PSHELL properties, so the groups
            # are actually selectable in ANSA/Nastran instead of being empty
            # property definitions.
            if write_boundaries:
                logger.info("Writing boundary groups...")
                _write_boundaries(
                    f, volume_mesh.boundaries, volume_mesh.cells.connectivity,
                    solid_pid=solid_pid, start_eid=n_tets + 1
                )

            # Every Bulk Data deck must end with ENDDATA regardless of whether
            # boundary groups were written.
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
    """Write NAS file header with metadata.
    
    Args:
        f: File handle
        volume_mesh: VolumeMeshData object
    """
    from datetime import datetime
    
    # ANSA-style header
    f.write("$ANSA_VERSION;21.0.1;\n")
    f.write("$\n")
    f.write("$\n")
    timestamp = datetime.now().strftime("%a %b %d %H:%M:%S %Y")
    f.write(f"$ file created by  A N S A  {timestamp}\n")
    f.write("$\n")
    f.write("$ output from :\n")
    f.write("$\n")
    f.write("$ AutoFlowCFD Volume Mesh Export\n")
    f.write(f"$ Nodes: {volume_mesh.node_count:,}\n")
    f.write(f"$ Elements: {volume_mesh.cell_count:,}\n")
    f.write(f"$ Total Volume: {volume_mesh.total_volume:.6e} m^3\n")
    f.write("$\n")
    f.write("$\n")
    f.write("$\n")
    f.write("BEGIN BULK                                                                      \n")


def _write_nodes(f, nodes, scale_factor: float) -> None:
    """Write GRID cards for all nodes.

    Nastran Small Field format (what this function actually writes below -
    see the Field 1-6 comments inline for the authoritative column layout):
    Columns 1-8:   "GRID" keyword
    Columns 9-16:  Node ID (right-aligned, 8 chars)
    Columns 17-24: Coordinate system ID (right-aligned, 8 chars)
    Columns 25-32: X coordinate (right-aligned, 8 chars)
    Columns 33-40: Y coordinate (right-aligned, 8 chars)
    Columns 41-48: Z coordinate (right-aligned, 8 chars)

    Args:
        f: File handle
        nodes: NodeArray with x, y, z coordinates
        scale_factor: Scaling factor for coordinates
    """
    n_nodes = len(nodes.x)
    
    # Batch write for performance (1000 nodes per batch)
    batch_size = 1000
    
    for start_idx in range(0, n_nodes, batch_size):
        end_idx = min(start_idx + batch_size, n_nodes)
        
        for i in range(start_idx, end_idx):
            node_id = i + 1  # Nastran IDs start from 1
            x = nodes.x[i] * scale_factor
            y = nodes.y[i] * scale_factor
            z = nodes.z[i] * scale_factor
            
            # Format coordinates with controlled precision to fit in 8-char fields
            # Maximum format: "-X.XXXXXX" (9 chars) or "XX.XXXXX" (8 chars)
            # Strategy: Use dynamic precision based on magnitude
            
            def format_coord_8char(value: float) -> str:
                """Format coordinate to fit in 8-character field.
                
                Args:
                    value: Coordinate value
                    
                Returns:
                    Formatted string <= 8 characters
                """
                # Try different precisions until we find one that fits
                for precision in [6, 5, 4, 3, 2]:
                    formatted = f"{value:.{precision}f}"
                    if len(formatted) <= 8:
                        return formatted

                # Fallback: Nastran compact exponent notation (no 'e', so
                # it actually fits 8 chars - Python's "%e"/.4e is itself
                # 10-11 characters and would silently overflow the field).
                return _format_nastran_compact_exponent(value, width=8)
            
            x_str = format_coord_8char(x)
            y_str = format_coord_8char(y)
            z_str = format_coord_8char(z)
            
            # Small Field Format: each field is exactly 8 characters
            # Field 1 (cols 1-8):   "GRID" keyword
            # Field 2 (cols 9-16):  Node ID (right-aligned)
            # Field 3 (cols 17-24): Coordinate system ID (0 = global, explicitly set)
            # Field 4 (cols 25-32): X coordinate (right-aligned, max 8 chars)
            # Field 5 (cols 33-40): Y coordinate (right-aligned, max 8 chars)
            # Field 6 (cols 41-48): Z coordinate (right-aligned, max 8 chars)
            # Fields 7-9: Omitted (trailing fields can be truncated)
            
            line = f"GRID    {node_id:>8}{0:>8}{x_str:>8}{y_str:>8}{z_str:>8}\n"
            f.write(line)
        
        if (start_idx + batch_size) % 10000 == 0:
            logger.debug(f"  Written {start_idx + batch_size}/{n_nodes} nodes")
    
    logger.info(f"  Total nodes written: {n_nodes:,}")


def _write_tetrahedra(f, connectivity: np.ndarray, solid_pid: int) -> int:
    """Write CTETRA cards for tetrahedral elements.

    ANSA Nastran CTETRA card format (fixed-width fields):
    CTETRA      EID       PID      G1       G2       G3       G4

    Field widths: 8-8-8-8-8-8 characters

    Args:
        f: File handle
        connectivity: Tetrahedral connectivity array, shape=(n_tets, 4)
        solid_pid: PSOLID property ID for the volume elements. Must match the
            PSOLID card written by _write_boundaries (or be free of any
            PSHELL PID) to avoid a duplicate-PID Bulk Data entry.

    Returns:
        int: Number of tetrahedra written (elem IDs used are 1..n_tets), so
        callers can continue element numbering (e.g. boundary CTRIA3 cards)
        without colliding with these element IDs.
    """
    n_tets = len(connectivity)

    # Batch write for performance
    batch_size = 1000

    for start_idx in range(0, n_tets, batch_size):
        end_idx = min(start_idx + batch_size, n_tets)

        for i in range(start_idx, end_idx):
            elem_id = i + 1  # Nastran IDs start from 1
            g1 = int(connectivity[i, 0]) + 1  # Convert 0-indexed to 1-indexed
            g2 = int(connectivity[i, 1]) + 1
            g3 = int(connectivity[i, 2]) + 1
            g4 = int(connectivity[i, 3]) + 1

            # ANSA format: fixed-width fields
            line = f"CTETRA{elem_id:>10}{solid_pid:>8}{g1:>8}{g2:>8}{g3:>8}{g4:>8}\n"
            f.write(line)

        if (start_idx + batch_size) % 10000 == 0:
            logger.debug(f"  Written {start_idx + batch_size}/{n_tets} elements")

    logger.info(f"  Total elements written: {n_tets:,}")
    return n_tets


def _extract_boundary_faces_by_group(
    connectivity: np.ndarray,
    boundary_groups: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """Recover each boundary group's actual exterior triangular faces.

    ``boundary_groups`` maps a boundary name to indices of the *owning
    tetrahedra* (see mesh_boundary.identify_boundaries_from_surface), not
    face geometry. To write a CTRIA3 element that a PSHELL property can
    actually reference, we need the specific 3 nodes of each such cell's
    exterior face - found the same way FaceExtractor/identify_boundaries_from_surface
    do: a tet face that occurs exactly once across the whole mesh is a
    boundary face.

    Args:
        connectivity: Tetrahedral connectivity, shape=(n_cells, 4)
        boundary_groups: boundary name -> owning cell indices

    Returns:
        Dict[str, np.ndarray]: boundary name -> face node indices (0-indexed),
        shape=(n_faces_in_group, 3)
    """
    n_cells = len(connectivity)
    face_templates = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3]
    ])
    all_faces = connectivity[:, face_templates].reshape(-1, 3)
    owner_cells = np.repeat(np.arange(n_cells), 4)

    sorted_faces = np.sort(all_faces, axis=1)
    face_dtype = np.dtype((np.void, sorted_faces.dtype.itemsize * 3))
    face_voids = np.ascontiguousarray(sorted_faces).view(face_dtype).reshape(-1)
    _, inverse, counts = np.unique(face_voids, return_inverse=True, return_counts=True)
    is_boundary_face = counts[inverse] == 1

    boundary_faces = all_faces[is_boundary_face]
    boundary_owners = owner_cells[is_boundary_face]

    faces_by_group = {}
    for name, cell_indices in boundary_groups.items():
        owner_in_group = np.zeros(n_cells, dtype=bool)
        owner_in_group[cell_indices] = True
        faces_by_group[name] = boundary_faces[owner_in_group[boundary_owners]]

    return faces_by_group


def _write_boundaries(
    f, boundaries, connectivity: np.ndarray, solid_pid: int, start_eid: int
) -> None:
    """Write boundary groups as PSHELL properties with real CTRIA3 face
    elements, plus the PSOLID card for the volume mesh.

    Args:
        f: File handle
        boundaries: BoundaryMap with groups and bc_types
        connectivity: Tetrahedral connectivity, used to recover each
            boundary group's actual exterior triangular faces
        solid_pid: PSOLID property ID already used for the CTETRA elements
            (reserved by the caller so it can't collide with a PSHELL PID)
        start_eid: First free Nastran element ID (n_tets + 1), so boundary
            CTRIA3 elements don't collide with CTETRA element IDs
    """
    if not boundaries.groups:
        logger.warning("No boundary groups found, skipping boundary export")
        return

    faces_by_group = _extract_boundary_faces_by_group(connectivity, boundaries.groups)

    pid_counter = 1
    mid_counter = 1
    eid_counter = start_eid

    for group_name, cell_indices in boundaries.groups.items():
        bc_type = boundaries.bc_types.get(group_name, "WALL")

        # Map boundary type to ANSA-compatible name
        ansa_name = bc_type.lower()

        # Write PSHELL card (8-character fields with continuation)
        f.write(f"PSHELL{pid_counter:>8}{mid_counter:>7}      1.{mid_counter:>7}      1.{mid_counter:>7}                +{pid_counter:07d}\n")
        f.write(f"+{pid_counter:07d}\n")

        # Write ANSA name comment
        f.write(f"$ANSA_NAME_COMMENT;{pid_counter};PSHELL;{ansa_name};;NO;NO;NO;NO;\n")

        # Write the group's actual boundary faces so the property above
        # references real geometry instead of being an empty definition.
        for face in faces_by_group.get(group_name, ()):
            n1, n2, n3 = int(face[0]) + 1, int(face[1]) + 1, int(face[2]) + 1
            f.write(f"CTRIA3{eid_counter:>10}{pid_counter:>8}{n1:>8}{n2:>8}{n3:>8}\n")
            eid_counter += 1

        pid_counter += 1
        mid_counter += 1

    logger.info(f"  Boundary face elements written: {eid_counter - start_eid:,}")

    # Write PSOLID card for volume mesh (PID reserved by the caller so it
    # never collides with the PSHELL PIDs written above)
    solid_mid = mid_counter
    f.write(f"PSOLID{solid_pid:>8}{solid_mid:>8}\n")
    f.write(f"$ANSA_NAME_COMMENT;{solid_pid};PSOLID;Auto Detected Volume;;NO;NO;NO;NO;\n")

    # Write MAT1 material cards
    for i in range(1, mid_counter + 1):
        f.write(f"$ANSA_COLOR;{i};MAT1;.725490212440491;.035294119268656;0.20392157137394;1.;\n")

    f.write(f"$ANSA_COLOR;{solid_mid};MAT1;.635294139385223;0.34901961684227;.341176480054855;1.;\n")

    # Write PART definitions
    f.write("$ANSA_PART;GROUP;ID;2;NAME;Auto Detected Volumes Group;BELONGS_HERE;YES;PID_OFFS\n")
    f.write("$ET;0;COLOR;137;211;69;0;IS_COLOR_ACTIVE;0;PART_TYPE;Undefined;ATTRIBUTES;2;DM/F\n")
    f.write("$ile Type;ANSA;DM/Status;WIP;CONTAINS;ANSAPART;3;\n")

    # Surface parts
    if pid_counter > 1:
        shell_range = f"1-{pid_counter-1}" if pid_counter > 2 else "1"
        f.write(f"$ANSA_PART;PART;ID;1;NAME;Untitled;BELONGS_HERE;YES;STUDY_VERSION;0;PID_OFFSET;0\n")
        f.write("$;COLOR;185;9;52;0;IS_COLOR_ACTIVE;1;PART_TYPE;Undefined;ATTRIBUTES;2;DM/File Ty\n")
        f.write(f"$pe;ANSA;DM/Status;WIP;CONTAINS;PSHELL;{shell_range};\n")

    # Volume part
    f.write(f"$ANSA_PART;PART;ID;3;NAME;Untitled_Volume_1;BELONGS_HERE;YES;STUDY_VERSION;0;PID\n")
    f.write("$_OFFSET;0;COLOR;215;68;166;0;IS_COLOR_ACTIVE;1;PART_TYPE;Undefined;ATTRIBUTES;2\n")
    f.write(f"$;DM/File Type;ANSA;DM/Status;WIP;CONTAINS;PSOLID;{solid_pid};\n")
    f.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
    f.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")

    logger.info(f"  Boundary groups written: {len(boundaries.groups)}")
