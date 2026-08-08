"""NAS 导出：边界组写入模块。

从 nas_export.py 拆分出来，专门负责把体网格的边界分组写成 ANSA 风格的
PSHELL 属性 + 真实 CTRIA3 面单元（外加 PSOLID/ANSA_PART 元数据），与
nas_export.py 里节点/单元几何写入的部分区分开。

Key Components:
    - extract_boundary_faces_by_group: 从体网格里还原每个边界组的实际外表面三角面
    - write_boundaries: 写出 PSHELL/CTRIA3/PSOLID/ANSA_PART 等边界元数据卡片
"""

from typing import Dict

import numpy as np
from loguru import logger


def extract_boundary_faces_by_group(
    volume_mesh,
    boundary_groups: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """Recover each boundary group's actual exterior triangular faces.

    ``boundary_groups`` maps a boundary name to *owning cell* indices (see
    mesh_boundary.identify_boundaries_from_surface) in this mesh's GLOBAL
    cell-index convention - prisms [0, n_prism), tets [n_prism,
    n_prism+n_tet) whenever volume_mesh has a BL prism region (see
    PrismCells/face_extractor.extract_faces_mixed) - NOT bare 0-based
    indices into a single tet-only connectivity array. Deriving boundary
    faces from volume_mesh.ensure_faces_exist() (rather than re-deriving
    "occurs exactly once" face dedup from a raw tet-only connectivity array
    here, as an earlier version of this function did) gets this right for
    free, reusing the SAME face graph the mesh's own quality validation
    computes and already gets right in both the tet-only and mixed cases.

    That earlier version was a real, confirmed bug for any mesh with a BL
    prism region: it built cell adjacency purely from `cells.connectivity`
    (tet-only) but was handed `boundary_groups`' GLOBAL indices, most of
    which (the wall/body group especially - it's now owned by prisms, not
    tets) point at prism cells entirely outside that tet-only array. Wrong
    faces got attributed to each boundary group (silently, when a prism
    index happened to be < n_tets - no crash to notice it by), which is why
    an exported volume mesh's boundary surface stopped matching the
    original input surface once the BL region became true prisms.

    Args:
        volume_mesh: VolumeMeshData - faces are extracted (or reused, if
            already cached) from this directly
        boundary_groups: boundary name -> owning cell indices, in
            volume_mesh's own global cell-index convention

    Returns:
        Dict[str, np.ndarray]: boundary name -> face node indices (0-indexed),
        shape=(n_faces_in_group, 3)
    """
    faces = volume_mesh.ensure_faces_exist()
    n_cells = volume_mesh.cell_count

    boundary_face_idx = faces.get_boundary_face_indices()
    boundary_owners = faces.connectivity[boundary_face_idx, 0]
    boundary_faces = faces.node_connectivity[boundary_face_idx]

    faces_by_group = {}
    for name, cell_indices in boundary_groups.items():
        owner_in_group = np.zeros(n_cells, dtype=bool)
        owner_in_group[cell_indices] = True
        faces_by_group[name] = boundary_faces[owner_in_group[boundary_owners]]

    return faces_by_group


def write_boundaries(
    f, volume_mesh, solid_pid: int, start_eid: int
) -> None:
    """Write boundary groups as PSHELL properties with real CTRIA3 face
    elements, plus the PSOLID card for the volume mesh.

    Args:
        f: File handle
        volume_mesh: VolumeMeshData - supplies both `boundaries` (BoundaryMap
            with groups and bc_types) and the face data needed to recover
            each group's actual exterior triangular faces (see
            extract_boundary_faces_by_group)
        solid_pid: PSOLID property ID already used for the CTETRA/CPENTA
            elements (reserved by the caller so it can't collide with a
            PSHELL PID)
        start_eid: First free Nastran element ID (n_prism + n_tets + 1), so
            boundary CTRIA3 elements don't collide with CPENTA/CTETRA
            element IDs
    """
    boundaries = volume_mesh.boundaries
    if not boundaries.groups:
        logger.warning("No boundary groups found, skipping boundary export")
        return

    faces_by_group = extract_boundary_faces_by_group(volume_mesh, boundaries.groups)

    pid_counter = 1
    mid_counter = 1
    eid_counter = start_eid

    for group_name, cell_indices in boundaries.groups.items():
        bc_type = boundaries.bc_types.get(group_name, "WALL")

        # Map boundary type to ANSA-compatible name
        ansa_name = bc_type.lower()

        # PSHELL Small Field Format, 8-char fields:
        # Field 1: "PSHELL  "  Field 2: PID  Field 3: MID1  Field 4: T
        # Field 5: MID2  Field 6: 12I/T^3  Field 7: MID3  Field 8: TS/T
        f.write(
            f"PSHELL  {pid_counter:>8}{mid_counter:>8}{1.0:>8.1f}"
            f"{mid_counter:>8}{1.0:>8.1f}{mid_counter:>8}{0.8333:>8.4f}\n"
        )

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

    # Write $ANSA_COLOR display-color comments, one per shell MID [1,
    # solid_mid) - NOT including solid_mid itself, which gets its own
    # (different) volume color right below; range(1, mid_counter + 1) used
    # to double-count it (solid_mid == mid_counter here), emitting two
    # conflicting color entries for the same MID.
    for i in range(1, mid_counter):
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
