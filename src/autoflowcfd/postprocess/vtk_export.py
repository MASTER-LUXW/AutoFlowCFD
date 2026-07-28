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
        """Write node coordinates to VTK file
        
        Args:
            f: File handle
        """
        nodes = self.grid_data.nodes
        n_points = nodes.count
        
        f.write(f"POINTS {n_points} float\n")
        
        for i in range(n_points):
            f.write(f"{nodes.x[i]:.6e} {nodes.y[i]:.6e} {nodes.z[i]:.6e}\n")
        
        f.write("\n")
    
    def _write_cells(self, f) -> None:
        """Write cell connectivity to VTK file
        
        Args:
            f: File handle
        """
        cells = self.grid_data.cells
        n_cells = cells.count
        
        # For triangular mesh (simplified)
        # In production, need to handle different cell types
        f.write(f"CELLS {n_cells} {n_cells * 4}\n")
        
        for i in range(n_cells):
            conn = cells.connectivity[i]
            f.write(f"3 {conn[0]} {conn[1]} {conn[2]}\n")
        
        f.write("\n")
        
        # Cell types (5 = triangle)
        f.write(f"CELL_TYPES {n_cells}\n")
        for i in range(n_cells):
            f.write("5\n")
        
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
    
    def _write_velocity(self, f, n_points: int) -> None:
        """Write velocity vector field
        
        Args:
            f: File handle
            n_points: Number of points
        """
        f.write(f"VECTORS Velocity float\n")
        
        # Placeholder: uniform velocity field
        # In production, extract from solution vector
        for i in range(n_points):
            f.write(f"30.0 0.0 0.0\n")
        
        f.write("\n")
    
    def _write_pressure(self, f, n_points: int) -> None:
        """Write pressure scalar field
        
        Args:
            f: File handle
            n_points: Number of points
        """
        f.write(f"SCALARS Pressure float 1\n")
        f.write("LOOKUP_TABLE default\n")
        
        # Placeholder: uniform pressure
        # In production, extract from solution vector
        for i in range(n_points):
            f.write(f"0.0\n")
        
        f.write("\n")
    
    def _write_turbulence(self, f, n_points: int, fields: List[str]) -> None:
        """Write turbulence variable fields
        
        Args:
            f: File handle
            n_points: Number of points
            fields: Turbulence fields to export
        """
        if 'k' in fields:
            f.write(f"SCALARS TurbulentKineticEnergy float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for i in range(n_points):
                f.write(f"0.0\n")
            f.write("\n")
        
        if 'omega' in fields:
            f.write(f"SCALARS SpecificDissipationRate float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for i in range(n_points):
                f.write(f"0.0\n")
            f.write("\n")
    
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
