"""VTK field data export module.

This module provides tools for exporting CFD simulation results to VTK format
for visualization in ParaView and other VTK-compatible viewers.

Key Components:
    - VTKExporter: Main exporter for VTK file generation
    - Supports velocity, pressure, turbulence variables export

Fidelity notes (why this isn't just the old simplified exporter):
    - Writes BOTH CELL_DATA (the raw, un-interpolated cell-centered value
      the solver actually produced) and POINT_DATA (volume-weighted
      node interpolation, for smooth contour plots) for every field - not
      POINT_DATA only. Mainstream solvers (Fluent/OpenFOAM/STAR-CCM+) are
      finite-volume/cell-centered and always preserve the un-smoothed
      per-cell value in their VTK-family output; a POINT_DATA-only export
      silently discards local extrema (e.g. peak wall pressure/shear).
    - Legacy .vtk supports a `binary=True` option (proper big-endian
      binary payloads per the VTK legacy spec) - required for real
      (100k+ cell) industrial meshes, where ASCII text is both far larger
      on disk and far slower to write/parse.
    - `format='xml'` is a real writer (delegates to pyvista/VTK's own
      vtkXMLUnstructuredGridWriter), not the previous stub that silently
      fell back to legacy - XML VTU with binary+zlib compression
      (`binary=True`, the default for xml) is the modern standard most
      current CFD post tools (OpenFOAM's foamToVTK, ParaView-native
      writers) actually emit.
    - `mu_t` (turbulent dynamic viscosity), if supplied, is the solver's
      own SST-blended value (see fvm_viscous_residual.py's
      `_eddy_viscosity`) persisted via CheckpointManager's extra_fields -
      not the simplified nu_t = k/omega estimate used when it's
      unavailable (e.g. older checkpoints saved before this was added).
    - `export_boundaries()` exports just the named boundary patches
      (WALL/INLET/OUTLET/... surface triangles) tagged with a stable
      integer BoundaryID/BoundaryTypeID (plus a name legend embedded as
      field data) - the patch-based workflow Fluent/OpenFOAM/STAR-CCM+
      use, instead of only ever exposing the whole volume mesh with no
      boundary identity.

Example:
    >>> from autoflowcfd.postprocess import VTKExporter
    >>> exporter = VTKExporter(grid_data, solution, mu_t=mu_t)
    >>> exporter.export("output.vtk", binary=True)
"""

import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
from loguru import logger

from ..grid.structures import GridData
from ..core.backend.base import SolutionVector
from ..core.bc_handler import BoundaryConditionHandler
from ._field_utils import cell_to_node

# VTK legacy cell-type codes (see VTK file format spec).
_VTK_TRIANGLE = 5
_VTK_TETRA = 10
_VTK_WEDGE = 13  # triangular prism - VTK's own node order matches this
                 # project's (v0,v1,v2,w0,w1,w2) convention directly (two
                 # triangle "caps" listed consecutively), no reordering needed

_VALID_FIELDS = {'velocity', 'pressure', 'k', 'omega', 'nut', 'turbulence'}


class VTKExporter:
    """VTK field data exporter

    Exports flow field data to VTK format for visualization in ParaView.
    Supports both legacy VTK (ASCII or binary) and XML-based VTK (.vtu,
    ASCII or binary+compressed) formats, each carrying both the raw
    cell-centered solver values (CELL_DATA) and node-interpolated values
    (POINT_DATA) for every field.

    Attributes:
        grid_data: Grid data object
        solution: Flow field solution vector
        mu_t: Optional (n_cells,) turbulent dynamic viscosity as actually
            computed by the solver (from CheckpointManager's extra_fields).
            When absent, 'nut' export falls back to a simplified k/omega
            estimate and logs a warning.

    Example:
        >>> exporter = VTKExporter(grid_data, solution, mu_t=mu_t)
        >>> exporter.export("result.vtk", fields=['velocity', 'pressure'], binary=True)
    """

    def __init__(
        self,
        grid_data: GridData,
        solution: SolutionVector,
        mu_t: Optional[np.ndarray] = None,
    ):
        """Initialize VTK exporter

        Args:
            grid_data: Grid data object
            solution: Flow field solution vector
            mu_t: Optional exact per-cell turbulent dynamic viscosity
                (Pa.s) from the solver, shape (n_cells,)

        Raises:
            ValueError: Invalid grid or solution data
        """
        self.grid_data = grid_data
        self.solution = solution
        self.mu_t = np.asarray(mu_t, dtype=np.float64) if mu_t is not None else None

        logger.info(
            f"VTKExporter initialized:\n"
            f"  Nodes:  {grid_data.metadata.node_count}\n"
            f"  Cells:  {grid_data.metadata.cell_count}"
        )

    def export(
        self,
        output_path: str,
        fields: Optional[List[str]] = None,
        format: str = 'legacy',
        binary: Optional[bool] = None,
    ) -> Path:
        """Export flow field to VTK file

        Args:
            output_path: Output file path (.vtk or .vtu)
            fields: Fields to export (default: all available fields)
                   Options: ['velocity', 'pressure', 'k', 'omega', 'nut']
            format: VTK format ('legacy' or 'xml')
            binary: Write binary payloads instead of ASCII text. Defaults
                to False for 'legacy' (matches prior behaviour) and True
                for 'xml' (binary+zlib-compressed .vtu is the standard
                choice for real mesh sizes; pass False to force ASCII XML).

        Returns:
            Path: Path to exported file

        Raises:
            ValueError: Invalid format or fields
            IOError: File write error

        Example:
            >>> path = exporter.export("result.vtk", binary=True)
            >>> print(f"Exported to: {path}")
        """
        if fields is None:
            fields = ['velocity', 'pressure']

        invalid_fields = set(fields) - _VALID_FIELDS
        if invalid_fields:
            raise ValueError(
                f"Invalid fields: {invalid_fields}. "
                f"Valid fields: {_VALID_FIELDS}"
            )

        output_path = Path(output_path)

        if format == 'legacy':
            if not output_path.suffix:
                output_path = output_path.with_suffix('.vtk')
            self._export_legacy(output_path, fields, binary=bool(binary))
        elif format == 'xml':
            if not output_path.suffix:
                output_path = output_path.with_suffix('.vtu')
            self._export_xml(output_path, fields, binary=(True if binary is None else binary))
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'legacy' or 'xml'")

        logger.success(f"VTK file exported: {output_path}")
        return output_path

    def export_boundaries(
        self,
        output_path: str,
        fields: Optional[List[str]] = None,
        format: str = 'legacy',
        binary: Optional[bool] = None,
    ) -> Path:
        """Export just the named boundary patches (WALL/INLET/OUTLET/...
        surface triangles), each tagged with:

          - BoundaryID (CELL_DATA, int32): a stable per-boundary-name zone
            id. The id->name legend is embedded as field data - a
            "<id>=<name>" string array named 'BoundaryID_to_Name' (legacy
            ASCII: a FIELD FieldData block right after DATASET; xml:
            field_data, both readable in ParaView's Field Data inspector)
            - not the name itself as a per-cell field, since string-typed
            CELL_DATA does not reliably round-trip through VTK's own
            readers (verified empirically; the array shows up but reads
            back NULL). An integer id + a legend is exactly the zone_id +
            name-table pattern OpenFOAM/Fluent use internally. Exception:
            legacy format='legacy' with binary=True has no embedded
            legend - VTK 9.3's own legacy reader fails to open *any*
            binary .vtk containing a string FIELD block (verified with a
            minimal repro independent of this writer); the legend is
            logged instead and BoundaryID itself is unaffected. Prefer
            format='xml' (the default binary+legend combination that
            works) if you need both binary and the embedded legend.
          - BoundaryTypeID (CELL_DATA, int32): the coarser physics-role
            bucket (WALL/GROUND/INLET/OUTLET/SYMMETRY/FARFIELD), via the
            *same* classification the live solve path uses
            (BoundaryConditionHandler._classify) - not a re-derived
            guess that could silently drift from what the solver actually
            treated that boundary as. Legend: 'BoundaryTypeID_to_Name'.
          - the requested flow fields, taken directly from each
            triangle's owner cell (raw, un-interpolated; there is no
            point-data pass here - a boundary-only node average would be
            ambiguous where a node is shared with the interior mesh, so
            this export deliberately only carries the exact per-face
            value).

        Lets you open just the surface patches in ParaView and color/
        filter by exact named zone or by physics role - the patch-based
        workflow Fluent/OpenFOAM/STAR-CCM+ use - instead of only ever
        seeing the whole volume mesh with no boundary identity.

        Args:
            output_path: Output file path (.vtk or .vtu)
            fields: Fields to export (default: ['velocity', 'pressure'])
            format: VTK format ('legacy' or 'xml')
            binary: See export(); same defaults.

        Returns:
            Path: Path to exported file

        Raises:
            ValueError: Invalid format/fields, or grid_data is a bare
                surface GridData rather than a VolumeMeshData (no
                per-tetrahedron boundary groups / face extraction to
                derive patches from)
        """
        if not hasattr(self.grid_data, 'ensure_faces_exist'):
            raise ValueError(
                "export_boundaries requires a VolumeMeshData grid (named "
                "boundary groups over tetrahedra + face extraction), not "
                "a bare surface GridData."
            )
        if fields is None:
            fields = ['velocity', 'pressure']
        invalid_fields = set(fields) - _VALID_FIELDS
        if invalid_fields:
            raise ValueError(
                f"Invalid fields: {invalid_fields}. "
                f"Valid fields: {_VALID_FIELDS}"
            )

        faces = self.grid_data.ensure_faces_exist()
        if faces.node_connectivity is None:
            raise RuntimeError(
                "Face data has no node_connectivity - if this grid came "
                "from a cached volume_mesh.pkl built before boundary "
                "export support was added, regenerate the volume mesh."
            )

        bidx = faces.get_boundary_face_indices()
        owner_cells = faces.connectivity[bidx, 0].astype(np.int64)
        tri_conn = faces.node_connectivity[bidx]

        boundary_id, type_id, id_legend, type_legend = self._boundary_zone_ids(owner_cells)

        full_cell_fields = self._cell_fields(fields)
        boundary_fields = {k: v[owner_cells] for k, v in full_cell_fields.items()}

        output_path = Path(output_path)
        if format == 'legacy':
            if not output_path.suffix:
                output_path = output_path.with_suffix('.vtk')
            self._export_boundaries_legacy(
                output_path, fields, tri_conn, boundary_fields,
                boundary_id, type_id, id_legend, type_legend, binary=bool(binary),
            )
        elif format == 'xml':
            if not output_path.suffix:
                output_path = output_path.with_suffix('.vtu')
            self._export_boundaries_xml(
                output_path, fields, tri_conn, boundary_fields,
                boundary_id, type_id, id_legend, type_legend,
                binary=(True if binary is None else binary),
            )
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'legacy' or 'xml'")

        logger.success(f"VTK boundary patches exported: {output_path}")
        return output_path

    _BC_TYPE_NAMES = ['WALL', 'GROUND', 'INLET', 'OUTLET', 'SYMMETRY', 'FARFIELD']

    def _boundary_zone_ids(self, owner_cells: np.ndarray):
        """Map each boundary face's owner tetrahedron to a BoundaryID
        (per boundary-group name) and BoundaryTypeID (per
        BoundaryConditionHandler._classify bucket), using the exact same
        "owner cell membership in BoundaryMap.groups" lookup
        bc_handler.py's _precompute_face_types uses for the live solve -
        so a face this exporter tags 'WALL' is guaranteed to be one the
        solver actually applied a no-slip wall BC to, not a name-pattern
        guess re-derived independently here.

        Returns:
            (boundary_id, type_id, id_legend, type_legend) - the first two
            are (n_boundary_faces,) int32 arrays, the legends are
            List[str] of "<id>=<name>" entries.
        """
        boundary_names = self.grid_data.boundaries.boundary_names
        name_to_id = {name: i for i, name in enumerate(boundary_names)}
        type_to_id = {t: i for i, t in enumerate(self._BC_TYPE_NAMES)}
        unclassified_id = len(boundary_names)

        cell_to_name_id: Dict[int, int] = {}
        cell_to_type_id: Dict[int, int] = {}
        for name in boundary_names:
            btype = BoundaryConditionHandler._classify(name.upper())
            nid = name_to_id[name]
            tid = type_to_id.get(btype, type_to_id['WALL'])
            for c in self.grid_data.boundaries.get_cell_indices(name):
                cell_to_name_id[int(c)] = nid
                cell_to_type_id[int(c)] = tid

        boundary_id = np.array(
            [cell_to_name_id.get(int(c), unclassified_id) for c in owner_cells], dtype=np.int32
        )
        type_id = np.array(
            [cell_to_type_id.get(int(c), type_to_id['WALL']) for c in owner_cells], dtype=np.int32
        )

        id_legend = [f"{i}={name}" for name, i in sorted(name_to_id.items(), key=lambda kv: kv[1])]
        if np.any(boundary_id == unclassified_id):
            n_unclassified = int(np.sum(boundary_id == unclassified_id))
            logger.warning(
                f"{n_unclassified} boundary faces have no matching boundary "
                f"group; tagged BoundaryID={unclassified_id} (<UNCLASSIFIED>)"
            )
            id_legend.append(f"{unclassified_id}=<UNCLASSIFIED>")
        type_legend = [f"{i}={name}" for name, i in sorted(type_to_id.items(), key=lambda kv: kv[1])]

        return boundary_id, type_id, id_legend, type_legend

    # ------------------------------------------------------------------
    # Shared field computation - single source of truth for both the
    # raw CELL_DATA values and the node-interpolated POINT_DATA values,
    # so the two representations of a field can never silently diverge.
    # ------------------------------------------------------------------

    def _cell_fields(self, fields: List[str]) -> Dict[str, np.ndarray]:
        """Compute every requested field at cell-center resolution
        (n_cells,) or (n_cells, 3) for vectors), applying the same
        fallback constants the old point-only writer used when solution
        data is unavailable (e.g. an empty SolutionVector).

        Returns:
            Dict mapping field name ('velocity', 'pressure', 'k', 'omega',
            'nut') to its raw per-cell array - exactly what CELL_DATA
            writes, and what POINT_DATA interpolates from.
        """
        n_cells = self.grid_data.cell_count
        has_data = self.solution.data is not None and self.solution.n_cells > 0
        out: Dict[str, np.ndarray] = {}

        if 'velocity' in fields:
            if has_data:
                u, v, w = self.solution.get_velocity()
                out['velocity'] = np.column_stack([u, v, w])
            else:
                logger.warning("Solution data not available. Using zero velocity.")
                out['velocity'] = np.zeros((n_cells, 3))

        if 'pressure' in fields:
            if has_data:
                out['pressure'] = self.solution.get_pressure()
            else:
                logger.warning("Solution data not available. Using uniform pressure.")
                out['pressure'] = np.full(n_cells, 101325.0)

        need_turb = 'k' in fields or 'omega' in fields or 'nut' in fields
        if need_turb:
            k = omega = np.array([])
            if has_data:
                k, omega = self.solution.get_turbulence()
                if len(k) == 0:
                    logger.warning(
                        "Solution has no turbulence columns (need >=7 variables); "
                        "writing zero for k/omega/nut"
                    )
            k_out = k if len(k) == n_cells else np.full(n_cells, 0.0)
            omega_out = omega if len(omega) == n_cells else np.full(n_cells, 0.0)

            if 'k' in fields:
                out['k'] = k_out
            if 'omega' in fields:
                out['omega'] = omega_out
            if 'nut' in fields:
                if self.mu_t is not None and len(self.mu_t) == n_cells and has_data:
                    rho = np.maximum(self.solution.get_density(), 1e-10)
                    out['nut'] = self.mu_t / rho
                elif len(k_out) > 0 and np.any(omega_out > 0):
                    logger.warning(
                        "Exact solver mu_t not available (checkpoint predates "
                        "extra_fields support, or turbulence disabled); "
                        "'nut' is the simplified nu_t = k/omega estimate, "
                        "not the actual SST-blended, a1-limited eddy "
                        "viscosity the solver used."
                    )
                    out['nut'] = k_out / np.maximum(omega_out, 1e-10)
                else:
                    out['nut'] = np.zeros(n_cells)

        return out

    def _cell_to_node(self, cell_values: np.ndarray, n_points: int, fallback: float = 0.0) -> np.ndarray:
        """Interpolate a per-cell scalar field to per-node values (volume-
        weighted average over each node's connected cells - see
        _field_utils.cell_to_node)."""
        conn = np.asarray(self.grid_data.cells.connectivity)
        volumes = getattr(self.grid_data.cells, "volumes", None)
        return cell_to_node(conn, cell_values, n_points, volumes=volumes, fallback=fallback)

    def _point_fields(self, cell_fields: Dict[str, np.ndarray], n_points: int) -> Dict[str, np.ndarray]:
        """Interpolate every cell-centered field in `cell_fields` to nodes."""
        out: Dict[str, np.ndarray] = {}
        for name, arr in cell_fields.items():
            if arr.ndim == 2:
                fallback = 0.0 if name == 'velocity' else 0.0
                out[name] = np.column_stack([
                    self._cell_to_node(arr[:, i], n_points, fallback=float(np.mean(arr[:, i])) if len(arr) else 0.0)
                    for i in range(arr.shape[1])
                ])
            else:
                fallback = 101325.0 if name == 'pressure' else 0.0
                out[name] = self._cell_to_node(arr, n_points, fallback=fallback)
        return out

    # ------------------------------------------------------------------
    # Legacy VTK (.vtk) - ASCII or binary
    # ------------------------------------------------------------------

    _FIELD_LABELS = {
        'velocity': 'Velocity',
        'pressure': 'Pressure',
        'k': 'TurbulentKineticEnergy',
        'omega': 'SpecificDissipationRate',
        'nut': 'TurbulentViscosity',
    }

    def _export_legacy(self, output_path: Path, fields: List[str], binary: bool) -> None:
        """Export to legacy VTK format (VTK Legacy spec, DataFile Version 3.0
        - the classic CELLS/CELL_TYPES layout, not VTK 9's newer OFFSETS/
        CONNECTIVITY variant, for maximum compatibility with older readers)."""
        logger.info(f"Exporting to legacy VTK format ({'binary' if binary else 'ASCII'}): {output_path}")

        n_points = self.grid_data.nodes.count
        n_cells = self.grid_data.cell_count
        cell_fields = self._cell_fields(fields)
        point_fields = self._point_fields(cell_fields, n_points)

        try:
            mode = 'wb' if binary else 'w'
            with open(output_path, mode) as f:
                self._wl(f, "# vtk DataFile Version 3.0\n", binary)
                self._wl(f, f"AutoFlowCFD Export - {output_path.name}\n", binary)
                self._wl(f, ("BINARY\n" if binary else "ASCII\n"), binary)
                self._wl(f, "\n", binary)
                self._wl(f, "DATASET UNSTRUCTURED_GRID\n", binary)
                self._wl(f, "\n", binary)

                self._write_points(f, binary)
                self._write_cells(f, binary)

                # CELL_DATA: raw, un-interpolated solver values.
                self._wl(f, f"CELL_DATA {n_cells}\n", binary)
                self._write_field_block(f, fields, cell_fields, binary)

                # POINT_DATA: volume-weighted interpolation, for smooth
                # contour rendering.
                self._wl(f, f"POINT_DATA {n_points}\n", binary)
                self._write_field_block(f, fields, point_fields, binary)

            logger.info("Legacy VTK file written successfully")

        except IOError as e:
            logger.error(f"Failed to write VTK file: {e}")
            raise

    @staticmethod
    def _wl(f, text: str, binary: bool) -> None:
        """Write a header/keyword line, encoding to bytes in binary mode."""
        f.write(text.encode('ascii') if binary else text)

    def _write_points(self, f, binary: bool) -> None:
        nodes = self.grid_data.nodes
        n_points = nodes.count
        coords = np.column_stack([nodes.x, nodes.y, nodes.z]).astype(np.float64)

        self._wl(f, f"POINTS {n_points} double\n", binary)
        if binary:
            f.write(coords.astype('>f8').tobytes())
            f.write(b'\n')
        else:
            np.savetxt(f, coords, fmt="%.6e")
        self._wl(f, "\n", binary)

    def _write_cells(self, f, binary: bool) -> None:
        """Write cell connectivity to VTK file.

        Detects the actual node count per cell from the connectivity
        array's own shape (3 = triangle, 4 = tetrahedron) - or, when
        grid_data.prism_cells is set, writes prisms (6-node wedge, global
        indices [0, n_prism)) followed by tets ([n_prism, n_prism+n_tet)),
        matching this project's global cell-index convention (see
        PrismCells/face_extractor.extract_faces_mixed).
        """
        prism_cells_obj = getattr(self.grid_data, 'prism_cells', None)
        if prism_cells_obj is not None:
            self._write_cells_mixed(f, prism_cells_obj.connectivity, self.grid_data.cells.connectivity, binary)
        else:
            self._write_cells_from(f, self.grid_data.cells.connectivity, binary)

    def _write_cells_mixed(self, f, prism_conn: np.ndarray, tet_conn: np.ndarray, binary: bool) -> None:
        """Write CELLS/CELL_TYPES for a mixed prism(wedge) + tetrahedron
        mesh - each row can have a different vertex count in legacy VTK's
        CELLS format (the leading integer per row IS that row's count), so
        prisms and tets simply concatenate into one block; CELL_TYPES
        carries the per-row type code (_VTK_WEDGE vs _VTK_TETRA)."""
        prism_conn = np.asarray(prism_conn, dtype=np.int32)
        tet_conn = np.asarray(tet_conn, dtype=np.int32)
        n_prism = len(prism_conn)
        n_tet = len(tet_conn)
        n_cells = n_prism + n_tet
        total_ints = n_prism * 7 + n_tet * 5  # (1 count + 6 verts) or (1 count + 4 verts)

        self._wl(f, f"CELLS {n_cells} {total_ints}\n", binary)
        if binary:
            if n_prism:
                prism_lines = np.hstack([np.full((n_prism, 1), 6, dtype=np.int32), prism_conn])
                f.write(prism_lines.astype('>i4').tobytes())
            if n_tet:
                tet_lines = np.hstack([np.full((n_tet, 1), 4, dtype=np.int32), tet_conn])
                f.write(tet_lines.astype('>i4').tobytes())
            f.write(b'\n')
        else:
            if n_prism:
                prism_lines = np.hstack([np.full((n_prism, 1), 6, dtype=np.int32), prism_conn])
                np.savetxt(f, prism_lines, fmt="%d")
            if n_tet:
                tet_lines = np.hstack([np.full((n_tet, 1), 4, dtype=np.int32), tet_conn])
                np.savetxt(f, tet_lines, fmt="%d")
        self._wl(f, "\n", binary)

        cell_types = np.concatenate([
            np.full(n_prism, _VTK_WEDGE, dtype=np.int32),
            np.full(n_tet, _VTK_TETRA, dtype=np.int32),
        ])
        self._wl(f, f"CELL_TYPES {n_cells}\n", binary)
        if binary:
            f.write(cell_types.astype('>i4').tobytes())
            f.write(b'\n')
        else:
            np.savetxt(f, cell_types.reshape(-1, 1), fmt="%d")
        self._wl(f, "\n", binary)

    def _write_cells_from(self, f, conn: np.ndarray, binary: bool) -> None:
        """Write CELLS/CELL_TYPES from an explicit connectivity array -
        shared by both the full-volume export (_write_cells) and the
        boundary-surface export (a different, smaller triangle set over
        the same node array)."""
        conn = np.asarray(conn, dtype=np.int32)
        n_cells = conn.shape[0]
        nodes_per_cell = conn.shape[1]

        vtk_type = {3: _VTK_TRIANGLE, 4: _VTK_TETRA}.get(nodes_per_cell)
        if vtk_type is None:
            raise ValueError(
                f"Unsupported cell connectivity width {nodes_per_cell} "
                f"(expected 3 for triangles or 4 for tetrahedra)"
            )

        counts = np.full((n_cells, 1), nodes_per_cell, dtype=np.int32)
        cell_lines = np.hstack([counts, conn])

        self._wl(f, f"CELLS {n_cells} {n_cells * (nodes_per_cell + 1)}\n", binary)
        if binary:
            f.write(cell_lines.astype('>i4').tobytes())
            f.write(b'\n')
        else:
            np.savetxt(f, cell_lines, fmt="%d")
        self._wl(f, "\n", binary)

        self._wl(f, f"CELL_TYPES {n_cells}\n", binary)
        types_arr = np.full(n_cells, vtk_type, dtype=np.int32)
        if binary:
            f.write(types_arr.astype('>i4').tobytes())
            f.write(b'\n')
        else:
            np.savetxt(f, types_arr, fmt="%d")
        self._wl(f, "\n", binary)

    def _write_field_block(self, f, fields: List[str], values: Dict[str, np.ndarray], binary: bool) -> None:
        if 'velocity' in fields and 'velocity' in values:
            self._write_vector(f, "Velocity", values['velocity'], binary)
        if 'pressure' in fields and 'pressure' in values:
            self._write_scalar(f, "Pressure", values['pressure'], binary)
        if 'k' in fields and 'k' in values:
            self._write_scalar(f, "TurbulentKineticEnergy", values['k'], binary)
        if 'omega' in fields and 'omega' in values:
            self._write_scalar(f, "SpecificDissipationRate", values['omega'], binary)
        if 'nut' in fields and 'nut' in values:
            self._write_scalar(f, "TurbulentViscosity", values['nut'], binary)

    def _write_scalar(self, f, name: str, values: np.ndarray, binary: bool, int_type: bool = False) -> None:
        vtk_type_name = "int" if int_type else "double"
        np_dtype = '>i4' if int_type else '>f8'
        self._wl(f, f"SCALARS {name} {vtk_type_name} 1\n", binary)
        self._wl(f, "LOOKUP_TABLE default\n", binary)
        if binary:
            f.write(np.ascontiguousarray(values).astype(np_dtype).tobytes())
            f.write(b'\n')
        else:
            np.savetxt(f, values, fmt="%d" if int_type else "%.6e")
        self._wl(f, "\n", binary)

    def _write_field_data_legacy(self, f, entries: Dict[str, List[str]], binary: bool) -> None:
        """Write a FIELD FieldData block (global metadata, e.g. the
        BoundaryID->name legend).

        Only emitted in ASCII mode: empirically, VTK 9.3's own
        vtkUnstructuredGridReader fails to parse *any* legacy file
        containing a string-typed FIELD block once the file's data mode
        is BINARY - confirmed with a minimal hand-built repro independent
        of this writer (field block before OR after the binary payload,
        both fail the same way; the same block in an ASCII-mode file
        reads back correctly). Rather than emit a binary .vtk that VTK's
        own reader can't open, the legend is logged instead when
        binary=True - the numeric BoundaryID/BoundaryTypeID CELL_DATA is
        unaffected either way. XML (.vtu) has no such issue (see
        _export_boundaries_xml) and is the recommended format for this
        export.
        """
        if not entries:
            return
        if binary:
            for name, values in entries.items():
                logger.info(f"{name}: " + ", ".join(values))
            return
        self._wl(f, f"FIELD FieldData {len(entries)}\n", binary)
        for name, values in entries.items():
            self._wl(f, f"{name} 1 {len(values)} string\n", binary)
            for v in values:
                self._wl(f, f"{v}\n", binary)
        self._wl(f, "\n", binary)

    def _write_vector(self, f, name: str, values: np.ndarray, binary: bool) -> None:
        self._wl(f, f"VECTORS {name} double\n", binary)
        if binary:
            f.write(np.ascontiguousarray(values, dtype=np.float64).astype('>f8').tobytes())
            f.write(b'\n')
        else:
            np.savetxt(f, values, fmt="%.6e")
        self._wl(f, "\n", binary)

    def _export_boundaries_legacy(
        self, output_path: Path, fields: List[str], tri_conn: np.ndarray,
        boundary_fields: Dict[str, np.ndarray], boundary_id: np.ndarray, type_id: np.ndarray,
        id_legend: List[str], type_legend: List[str], binary: bool,
    ) -> None:
        logger.info(f"Exporting boundary patches to legacy VTK format ({'binary' if binary else 'ASCII'}): {output_path}")
        n_tri = tri_conn.shape[0]

        try:
            mode = 'wb' if binary else 'w'
            with open(output_path, mode) as f:
                self._wl(f, "# vtk DataFile Version 3.0\n", binary)
                self._wl(f, f"AutoFlowCFD Boundary Export - {output_path.name}\n", binary)
                self._wl(f, ("BINARY\n" if binary else "ASCII\n"), binary)
                self._wl(f, "\n", binary)
                self._wl(f, "DATASET UNSTRUCTURED_GRID\n", binary)
                self._write_field_data_legacy(f, {
                    'BoundaryID_to_Name': id_legend,
                    'BoundaryTypeID_to_Name': type_legend,
                }, binary)
                self._wl(f, "\n", binary)

                self._write_points(f, binary)
                self._write_cells_from(f, tri_conn, binary)

                self._wl(f, f"CELL_DATA {n_tri}\n", binary)
                self._write_scalar(f, "BoundaryID", boundary_id, binary, int_type=True)
                self._write_scalar(f, "BoundaryTypeID", type_id, binary, int_type=True)
                self._write_field_block(f, fields, boundary_fields, binary)

            logger.info("Legacy VTK boundary file written successfully")

        except IOError as e:
            logger.error(f"Failed to write VTK boundary file: {e}")
            raise

    # ------------------------------------------------------------------
    # XML VTK (.vtu) - delegates to pyvista/VTK's own writer
    # ------------------------------------------------------------------

    def _export_xml(self, output_path: Path, fields: List[str], binary: bool) -> None:
        """Export to XML-based VTK format (.vtu), the modern standard most
        current CFD post tools emit. Builds a pyvista.UnstructuredGrid from
        the same cell/point fields the legacy writer uses and lets VTK's
        own vtkXMLUnstructuredGridWriter serialize it (binary+zlib
        compression when binary=True) - this avoids hand-rolling the XML
        appended-data binary encoding, which pyvista/VTK already implement
        correctly and is what ParaView itself both reads and writes.
        """
        import pyvista as pv

        logger.info(f"Exporting to XML VTK format ({'binary' if binary else 'ASCII'}): {output_path}")

        nodes = self.grid_data.nodes
        n_points = nodes.count
        points = np.column_stack([nodes.x, nodes.y, nodes.z]).astype(np.float64)

        conn = np.asarray(self.grid_data.cells.connectivity, dtype=np.int64)
        nodes_per_cell = conn.shape[1]
        cell_type = {3: pv.CellType.TRIANGLE, 4: pv.CellType.TETRA}.get(nodes_per_cell)
        if cell_type is None:
            raise ValueError(
                f"Unsupported cell connectivity width {nodes_per_cell} "
                f"(expected 3 for triangles or 4 for tetrahedra)"
            )

        grid = pv.UnstructuredGrid({cell_type: conn}, points)

        cell_fields = self._cell_fields(fields)
        point_fields = self._point_fields(cell_fields, n_points)
        for key, arr in cell_fields.items():
            grid.cell_data[self._FIELD_LABELS[key]] = arr
        for key, arr in point_fields.items():
            grid.point_data[self._FIELD_LABELS[key]] = arr

        grid.save(str(output_path), binary=binary)
        logger.info("XML VTK file written successfully")

    def _export_boundaries_xml(
        self, output_path: Path, fields: List[str], tri_conn: np.ndarray,
        boundary_fields: Dict[str, np.ndarray], boundary_id: np.ndarray, type_id: np.ndarray,
        id_legend: List[str], type_legend: List[str], binary: bool,
    ) -> None:
        """Export boundary patches to .vtu - see export_boundaries. The
        BoundaryID/BoundaryTypeID -> name legends go in field_data (global
        metadata, not per-cell): verified empirically that a per-cell
        *string* CELL_DATA array does not survive a VTK XML writer/reader
        round-trip (the array is listed but reads back as a null pointer),
        while field_data string arrays round-trip correctly as
        vtkStringArray - both through vtkXMLUnstructuredGridReader
        directly and through pyvista.read().
        """
        import pyvista as pv

        logger.info(f"Exporting boundary patches to XML VTK format ({'binary' if binary else 'ASCII'}): {output_path}")

        nodes = self.grid_data.nodes
        points = np.column_stack([nodes.x, nodes.y, nodes.z]).astype(np.float64)

        grid = pv.UnstructuredGrid({pv.CellType.TRIANGLE: np.asarray(tri_conn, dtype=np.int64)}, points)
        grid.cell_data['BoundaryID'] = boundary_id
        grid.cell_data['BoundaryTypeID'] = type_id
        for key, arr in boundary_fields.items():
            grid.cell_data[self._FIELD_LABELS[key]] = arr
        grid.field_data['BoundaryID_to_Name'] = np.array(id_legend)
        grid.field_data['BoundaryTypeID_to_Name'] = np.array(type_legend)

        grid.save(str(output_path), binary=binary)
        logger.info("XML VTK boundary file written successfully")
