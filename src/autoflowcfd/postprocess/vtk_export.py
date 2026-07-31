"""VTK field data export module.

This module provides tools for exporting CFD simulation results to VTK format
for visualization in ParaView and other VTK-compatible viewers.

Key Components:
    - VTKExporter: Main exporter for VTK file generation
    - Supports velocity, pressure, turbulence variables export

Example:
    >>> from autoflowcfd.postprocess import VTKExporter
    >>> exporter = VTKExporter(grid_data, solution)
    >>> exporter.export("output.vtk")
"""

import numpy as np
from pathlib import Path
from typing import Optional, List
from loguru import logger

from ..grid.structures import GridData
from ..core.backend.base import SolutionVector
from ._field_utils import cell_to_node

# VTK legacy cell-type codes (see VTK file format spec).
_VTK_TRIANGLE = 5
_VTK_TETRA = 10


class VTKExporter:
    """VTK field data exporter

    Exports flow field data to VTK format for visualization in ParaView.
    Supports both legacy VTK and XML-based VTK formats.

    Attributes:
        grid_data: Grid data object
        solution: Flow field solution vector

    Example:
        >>> exporter = VTKExporter(grid_data, solution)
        >>> exporter.export("result.vtk", fields=['velocity', 'pressure'])
    """

    def __init__(
        self,
        grid_data: GridData,
        solution: SolutionVector
    ):
        """Initialize VTK exporter

        Args:
            grid_data: Grid data object
            solution: Flow field solution vector

        Raises:
            ValueError: Invalid grid or solution data
        """
        self.grid_data = grid_data
        self.solution = solution

        logger.info(
            f"VTKExporter initialized:\n"
            f"  Nodes:  {grid_data.metadata.node_count}\n"
            f"  Cells:  {grid_data.metadata.cell_count}"
        )

    def export(
        self,
        output_path: str,
        fields: Optional[List[str]] = None,
        format: str = 'legacy'
    ) -> Path:
        """Export flow field to VTK file

        Args:
            output_path: Output file path (.vtk or .vtu)
            fields: Fields to export (default: all available fields)
                   Options: ['velocity', 'pressure', 'k', 'omega', 'nut']
            format: VTK format ('legacy' or 'xml')

        Returns:
            Path: Path to exported file

        Raises:
            ValueError: Invalid format or fields
            IOError: File write error

        Example:
            >>> path = exporter.export("result.vtk")
            >>> print(f"Exported to: {path}")
        """
        if fields is None:
            fields = ['velocity', 'pressure']

        # Validate fields
        valid_fields = {'velocity', 'pressure', 'k', 'omega', 'nut', 'turbulence'}
        invalid_fields = set(fields) - valid_fields
        if invalid_fields:
            raise ValueError(
                f"Invalid fields: {invalid_fields}. "
                f"Valid fields: {valid_fields}"
            )

        output_path = Path(output_path)

        if format == 'legacy':
            if not output_path.suffix:
                output_path = output_path.with_suffix('.vtk')
            self._export_legacy(output_path, fields)
        elif format == 'xml':
            if not output_path.suffix:
                output_path = output_path.with_suffix('.vtu')
            self._export_xml(output_path, fields)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'legacy' or 'xml'")

        logger.success(f"VTK file exported: {output_path}")
        return output_path

    def _export_legacy(self, output_path: Path, fields: List[str]) -> None:
        """Export to legacy VTK format

        Args:
            output_path: Output file path
            fields: Fields to export
        """
        logger.info(f"Exporting to legacy VTK format: {output_path}")

        try:
            with open(output_path, 'w') as f:
                # Write header
                f.write("# vtk DataFile Version 3.0\n")
                f.write(f"AutoFlowCFD Export - {output_path.name}\n")
                f.write("ASCII\n")
                f.write("\n")

                # Write dataset structure
                f.write("DATASET UNSTRUCTURED_GRID\n")
                f.write("\n")

                # Write points
                self._write_points(f)

                # Write cells
                self._write_cells(f)

                # Write point data
                self._write_point_data(f, fields)

            logger.info(f"Legacy VTK file written successfully")

        except IOError as e:
            logger.error(f"Failed to write VTK file: {e}")
            raise

    def _write_points(self, f) -> None:
        """Write node coordinates to VTK file (vectorized)."""
        nodes = self.grid_data.nodes
        n_points = nodes.count

        f.write(f"POINTS {n_points} float\n")
        coords = np.column_stack([nodes.x, nodes.y, nodes.z])
        np.savetxt(f, coords, fmt="%.6e")
        f.write("\n")

    def _write_cells(self, f) -> None:
        """Write cell connectivity to VTK file (vectorized).

        Detects the actual node count per cell from the connectivity
        array's own shape (3 = triangle, 4 = tetrahedron) instead of
        assuming triangles unconditionally - the real production path
        (post_commands.py export-vtk, loading a volume_mesh.pkl) always
        has 4-node tetrahedra, so hardcoding 3 silently discarded every
        cell's 4th vertex and mislabeled it as a triangle (VTK type 5)
        instead of a tetrahedron (type 10).
        """
        cells = self.grid_data.cells
        n_cells = cells.count
        conn = np.asarray(cells.connectivity)
        nodes_per_cell = conn.shape[1]

        vtk_type = {3: _VTK_TRIANGLE, 4: _VTK_TETRA}.get(nodes_per_cell)
        if vtk_type is None:
            raise ValueError(
                f"Unsupported cell connectivity width {nodes_per_cell} "
                f"(expected 3 for triangles or 4 for tetrahedra)"
            )

        f.write(f"CELLS {n_cells} {n_cells * (nodes_per_cell + 1)}\n")
        counts = np.full((n_cells, 1), nodes_per_cell, dtype=conn.dtype)
        cell_lines = np.hstack([counts, conn])
        np.savetxt(f, cell_lines, fmt="%d")
        f.write("\n")

        f.write(f"CELL_TYPES {n_cells}\n")
        np.savetxt(f, np.full(n_cells, vtk_type, dtype=int), fmt="%d")
        f.write("\n")

    def _write_point_data(self, f, fields: List[str]) -> None:
        """Write point data (velocity, pressure, etc.) to VTK file

        Args:
            f: File handle
            fields: Fields to export
        """
        n_points = self.grid_data.nodes.count

        f.write(f"POINT_DATA {n_points}\n")

        # Export velocity if requested
        if 'velocity' in fields:
            self._write_velocity(f, n_points)

        # Export pressure if requested
        if 'pressure' in fields:
            self._write_pressure(f, n_points)

        # Export turbulence variables if requested
        if 'k' in fields or 'omega' in fields or 'nut' in fields:
            self._write_turbulence(f, n_points, fields)

    def _cell_to_node(self, cell_values: np.ndarray, n_points: int, fallback: float = 0.0) -> np.ndarray:
        """Interpolate a per-cell scalar field to per-node values (volume-
        weighted average over each node's connected cells - see
        _field_utils.cell_to_node). Replaces the previous Python dict/list-
        comprehension implementation, which was O(n_cells) in pure Python
        per field and became the dominant cost for real (100k+ cell)
        meshes, and previously used a plain unweighted mean that let a
        node's value be pulled toward whichever neighboring cell happened
        to be largest/smallest instead of respecting cell size.
        """
        conn = np.asarray(self.grid_data.cells.connectivity)
        volumes = getattr(self.grid_data.cells, "volumes", None)
        return cell_to_node(conn, cell_values, n_points, volumes=volumes, fallback=fallback)

    def _write_scalar_field(self, f, name: str, cell_values: np.ndarray, n_points: int, fallback: float) -> None:
        """Write one SCALARS field, interpolating cell data to nodes if needed."""
        f.write(f"SCALARS {name} float 1\n")
        f.write("LOOKUP_TABLE default\n")
        if len(cell_values) == n_points:
            node_values = cell_values
        elif len(cell_values) > 0:
            node_values = self._cell_to_node(cell_values, n_points, fallback)
        else:
            node_values = np.full(n_points, fallback)
        np.savetxt(f, node_values, fmt="%.6e")
        f.write("\n")

    def _write_velocity(self, f, n_points: int) -> None:
        """Write velocity vector field (cell-centered solution interpolated
        to nodes with volume weighting - see _cell_to_node)."""
        f.write("VECTORS Velocity float\n")

        if self.solution.data is not None and self.solution.n_cells > 0:
            u, v, w = self.solution.get_velocity()
            n_cells = self.solution.n_cells

            if n_cells == n_points:
                node_vel = np.column_stack([u, v, w])
            else:
                logger.info(f"Interpolating cell-centered velocity ({n_cells} cells) to nodes ({n_points})...")
                node_vel = np.column_stack([
                    self._cell_to_node(u, n_points, fallback=float(np.mean(u)) if len(u) else 0.0),
                    self._cell_to_node(v, n_points, fallback=float(np.mean(v)) if len(v) else 0.0),
                    self._cell_to_node(w, n_points, fallback=float(np.mean(w)) if len(w) else 0.0),
                ])
        else:
            logger.warning("Solution data not available. Using zero velocity.")
            node_vel = np.zeros((n_points, 3))

        np.savetxt(f, node_vel, fmt="%.6e")
        f.write("\n")

    def _write_pressure(self, f, n_points: int) -> None:
        """Write pressure scalar field (cell-centered solution interpolated
        to nodes with volume weighting - see _cell_to_node)."""
        if self.solution.data is not None and self.solution.n_cells > 0:
            pressure = self.solution.get_pressure()
            self._write_scalar_field(f, "Pressure", pressure, n_points, fallback=101325.0)
        else:
            logger.warning("Solution data not available. Using uniform pressure.")
            self._write_scalar_field(f, "Pressure", np.array([]), n_points, fallback=101325.0)

    def _write_turbulence(self, f, n_points: int, fields: List[str]) -> None:
        """Write turbulence variable fields, actually reading them off the
        solution vector - this used to hardcode 0.0 for every node
        regardless of the real k/omega field."""
        k = omega = np.array([])
        if self.solution.data is not None and self.solution.n_cells > 0:
            k, omega = self.solution.get_turbulence()
            if len(k) == 0:
                logger.warning(
                    "Solution has no turbulence columns (need >=7 variables); "
                    "writing zero for k/omega/nut"
                )

        if 'k' in fields:
            self._write_scalar_field(f, "TurbulentKineticEnergy", k, n_points, fallback=0.0)

        if 'omega' in fields:
            self._write_scalar_field(f, "SpecificDissipationRate", omega, n_points, fallback=0.0)

        if 'nut' in fields:
            if len(k) > 0 and len(omega) > 0:
                # Standard k-omega eddy-viscosity relation (nu_t = k/omega).
                # This is a simplified estimate, not the SST-blended,
                # a1-limited eddy viscosity the solver actually uses
                # internally (core/fvm_viscous_residual.py) - good enough
                # for visualization, not for re-deriving exact wall shear.
                nut = k / np.maximum(omega, 1e-10)
            else:
                nut = np.array([])
            self._write_scalar_field(f, "TurbulentViscosity", nut, n_points, fallback=0.0)

    def _export_xml(self, output_path: Path, fields: List[str]) -> None:
        """Export to XML-based VTK format (VTU)

        Args:
            output_path: Output file path
            fields: Fields to export

        Note:
            XML format requires more complex structure.
            This is a simplified implementation.
        """
        logger.warning("XML VTK format not fully implemented. Using legacy format instead.")
        # For now, fall back to legacy format
        legacy_path = output_path.with_suffix('.vtk')
        self._export_legacy(legacy_path, fields)
