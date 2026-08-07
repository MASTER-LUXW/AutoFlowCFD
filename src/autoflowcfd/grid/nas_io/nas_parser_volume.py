"""Parser for externally-generated volume-mesh NAS files.

Unlike parser_core.NASParser (which reads a SURFACE mesh - CTRIA3 cards -
and this project's own generate-volume pipeline builds the volume mesh
from scratch), this module reads an ALREADY-COMPLETE volume mesh someone
else's tool produced (e.g. ANSA's own volume export: GRID + CTETRA +
CPENTA cards, fixed-width Nastran small-field format - the same format
nas_export.py itself writes, confirmed directly against a real ANSA
export this project was handed).

The parsed VolumeMeshData has an EMPTY BoundaryMap - an externally-built
volume mesh typically carries no boundary-condition information at all
(ANSA's own volume export only tags PSOLID material regions, not surface
BCs), unlike this project's own generation pipeline which tracks boundary
provenance as it builds the mesh. Attributing boundary groups (inlet/
outlet/wall/...) from a companion surface mesh file is a separate step -
see mesh_gen.mesh_boundary.map_boundaries_by_geometry, which has to match
by position (KD-tree nearest-centroid) rather than node index, since an
externally-generated mesh's own node numbering has no relationship to any
other file's.
"""

import numpy as np
from typing import Tuple
from loguru import logger

from ..structures import (
    NodeArray, TetrahedralCells, PrismCells, GridMetadata, VolumeMeshData, BoundaryMap,
)
from .nas_parser_utils import parse_nastran_float


def _parse_cards(path: str) -> Tuple[np.ndarray, np.ndarray, list, list]:
    """Single streaming pass over the file: collect GRID node id/xyz,
    CTETRA node-id rows, and CPENTA node-id rows. Fixed-width 8-char
    Nastran small-field cards throughout (matches nas_export.py's own
    CTETRA/CPENTA writers exactly, and ANSA's own volume export uses the
    same convention)."""
    node_ids = []
    node_xyz = []
    tet_rows = []
    prism_rows = []

    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            card = line[:8].strip()
            if card == "GRID":
                nid = int(line[8:16])
                x = parse_nastran_float(line[24:32])
                y = parse_nastran_float(line[32:40])
                z = parse_nastran_float(line[40:48])
                node_ids.append(nid)
                node_xyz.append((x, y, z))
            elif card == "CTETRA":
                tet_rows.append([int(line[24 + 8 * k:32 + 8 * k]) for k in range(4)])
            elif card == "CPENTA":
                prism_rows.append([int(line[24 + 8 * k:32 + 8 * k]) for k in range(6)])

    if not node_ids:
        raise ValueError(f"No GRID cards found in {path} - not a valid volume-mesh NAS file")
    if not tet_rows and not prism_rows:
        raise ValueError(
            f"No CTETRA/CPENTA cards found in {path} - this looks like a surface mesh "
            f"(CTRIA3-only); use NASParser instead"
        )

    node_ids_arr = np.array(node_ids, dtype=np.int64)
    node_xyz_arr = np.array(node_xyz, dtype=np.float64)
    return node_ids_arr, node_xyz_arr, tet_rows, prism_rows


# Matches NASParser.AUTO_UNITS_MM_THRESHOLD exactly - see that class's own
# comment for the reasoning (an automotive external-aero domain reads in
# the thousands in mm, a few to a few tens in metres).
_AUTO_UNITS_MM_THRESHOLD = 50.0


def parse_volume_mesh_nas(path: str, units: str = 'mm') -> VolumeMeshData:
    """Parse an externally-generated volume-mesh NAS file (GRID + CTETRA
    + CPENTA) into VolumeMeshData.

    Args:
        path: Path to the volume-mesh .nas file.
        units: Length unit of the coordinates in the file - 'mm' (default,
            matches NASParser's own default and ANSA's typical export
            convention), 'm' (no scaling), or 'auto' (detect from the raw
            bounding-box extent, same threshold/logic as NASParser's own
            units='auto'). Getting this wrong doesn't just distort the
            mesh - it silently breaks mesh_boundary.map_boundaries_by_
            geometry's nearest-centroid matching against the companion
            surface mesh (which NASParser always scales to metres),
            since every volume-mesh face would then sit ~1000x farther
            from its true position than any surface boundary face,
            comfortably outside even a generous tolerance. Confirmed
            directly: omitting this scaling on a real case matched 0 of
            39,352 exterior-face-owning cells to any surface boundary
            group.

    Returns:
        VolumeMeshData with an EMPTY BoundaryMap (groups={}, bc_types={})
        - see this module's own docstring for why boundary attribution is
        a separate step. Tets are re-oriented to positive volume the same
        way this project's own generation pipeline does; any exactly-
        degenerate (near-zero-volume) cell is dropped, matching the same
        cleanup this project applies to its own generated meshes.

    Raises:
        ValueError: no GRID cards, no CTETRA/CPENTA cards (e.g. a
            surface-only file was passed by mistake), or an invalid
            `units` value.
    """
    from ..mesh_gen.mesh_prism_to_tet import orient_tetrahedra
    from ..validation.quality_metrics import compute_prism_volumes

    if units not in ('mm', 'm', 'auto'):
        raise ValueError(f"units must be 'mm', 'm', or 'auto', got {units!r}")

    logger.info(f"Parsing external volume mesh: {path}")
    node_ids, node_xyz, tet_rows, prism_rows = _parse_cards(path)
    logger.info(
        f"Parsed {len(node_ids)} nodes, {len(tet_rows)} CTETRA, {len(prism_rows)} CPENTA"
    )

    raw_extent = float(np.max(node_xyz.max(axis=0) - node_xyz.min(axis=0)))
    if units == 'mm':
        scale_factor = 1e-3
    elif units == 'm':
        scale_factor = 1.0
    else:  # 'auto'
        if raw_extent > _AUTO_UNITS_MM_THRESHOLD:
            scale_factor = 1e-3
            logger.info(
                f"units='auto': raw bounding-box max extent={raw_extent:.4g} > "
                f"{_AUTO_UNITS_MM_THRESHOLD:g} -> assuming millimeters (scaling by 1e-3)"
            )
        else:
            scale_factor = 1.0
            logger.info(
                f"units='auto': raw bounding-box max extent={raw_extent:.4g} <= "
                f"{_AUTO_UNITS_MM_THRESHOLD:g} -> assuming the file is already in "
                f"meters (no scaling)"
            )
    node_xyz = node_xyz * scale_factor

    id_to_idx = np.full(int(node_ids.max()) + 1, -1, dtype=np.int64)
    id_to_idx[node_ids] = np.arange(len(node_ids))

    nodes_obj = NodeArray(
        x=np.ascontiguousarray(node_xyz[:, 0]),
        y=np.ascontiguousarray(node_xyz[:, 1]),
        z=np.ascontiguousarray(node_xyz[:, 2]),
    )

    tet_conn = np.zeros((0, 4), dtype=np.int64)
    if tet_rows:
        tet_conn = id_to_idx[np.array(tet_rows, dtype=np.int64)]
        if tet_conn.min() < 0:
            raise ValueError(f"{path}: CTETRA references a node id not defined by any GRID card")
        tet_conn = orient_tetrahedra(node_xyz, tet_conn.copy())
        tet_vol = TetrahedralCells.compute_volumes(nodes_obj, tet_conn)
        degenerate = np.abs(tet_vol) < 1e-20
        if np.any(degenerate):
            logger.warning(f"Dropping {int(degenerate.sum())} exactly-degenerate CTETRA cell(s)")
            tet_conn = tet_conn[~degenerate]
            tet_vol = tet_vol[~degenerate]
        neg = tet_vol < 0
        if np.any(neg):
            logger.warning(
                f"Dropping {int(neg.sum())} CTETRA cell(s) still negative-volume after "
                f"re-orientation (likely genuinely degenerate, not just misoriented)"
            )
            tet_conn = tet_conn[~neg]
            tet_vol = tet_vol[~neg]
    else:
        tet_vol = np.zeros(0, dtype=np.float64)

    prism_conn = np.zeros((0, 6), dtype=np.int64)
    prism_vol = np.zeros(0, dtype=np.float64)
    if prism_rows:
        prism_conn = id_to_idx[np.array(prism_rows, dtype=np.int64)]
        if prism_conn.min() < 0:
            raise ValueError(f"{path}: CPENTA references a node id not defined by any GRID card")
        prism_vol = compute_prism_volumes(node_xyz, prism_conn)
        degenerate_p = prism_vol < 1e-20
        if np.any(degenerate_p):
            logger.warning(f"Dropping {int(degenerate_p.sum())} exactly-degenerate CPENTA cell(s)")
            prism_conn = prism_conn[~degenerate_p]
            prism_vol = prism_vol[~degenerate_p]

    cells_obj = TetrahedralCells(
        connectivity=tet_conn.astype(np.int32), volumes=tet_vol.astype(np.float64)
    )
    prism_obj = (
        PrismCells(connectivity=prism_conn.astype(np.int32), volumes=prism_vol.astype(np.float64))
        if len(prism_conn) else None
    )

    boundaries_obj = BoundaryMap(groups={}, bc_types={})
    metadata = GridMetadata(
        node_count=len(node_xyz),
        cell_count=cells_obj.count + (prism_obj.count if prism_obj else 0),
        boundary_groups=[],
        file_format="external_volume_mesh",
    )
    volume_mesh = VolumeMeshData(
        nodes=nodes_obj, cells=cells_obj, boundaries=boundaries_obj,
        metadata=metadata, prism_cells=prism_obj,
    )
    logger.success(
        f"External volume mesh parsed: {volume_mesh.node_count} nodes, "
        f"{volume_mesh.cell_count} cells "
        f"({prism_obj.count if prism_obj else 0} prisms + {cells_obj.count} tets), "
        f"total volume {volume_mesh.total_volume:.6e} m^3"
    )
    return volume_mesh
