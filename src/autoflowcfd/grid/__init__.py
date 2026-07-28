"""Grid parsing and processing module.

This module handles ANSA .nas file parsing, grid data structures,
quality validation, and boundary condition mapping.

Key Components:
    - GridData: Main grid data structure with SoA layout
    - NodeArray: Node coordinates in Structure of Arrays format
    - CellArray: Cell connectivity and type information
    - BoundaryMap: Boundary condition mapping
    - NASParser: Parser for ANSA .nas files (v22/v23/v24)
    - GridValidator: Mesh quality checker

Example:
    >>> from autoflowcfd.grid import NASParser, GridValidator
    >>> parser = NASParser("car_model.nas")
    >>> grid = parser.parse()
    >>> validator = GridValidator(grid)
    >>> results = validator.validate()
    >>> print(f"Nodes: {grid.metadata.node_count}")
    >>> print(f"Quality check passed: {results['passed']}")
"""

# Re-export from modularized submodules
from .grid_nodes import NodeArray, CupyNodeArray
from .grid_cells import CellArray, CupyCellArray, TetrahedralCells
from .grid_boundaries import BoundaryMap
from .grid_metadata import GridMetadata
from .grid_faces import FaceData
from .grid_data import GridData, CupyGridData, VolumeMeshData

# Parser modules (modularized)
from .parser_core import NASParser
from .nas_parser_exceptions import NASParserError, NASFormatError, NASParseError

# Other modules
from .validator import GridValidator
from .volume_mesh_generator import VolumeMeshGenerator
from .quality_validator import MeshQualityValidator, MeshQualityReport
from .face_extractor import FaceExtractor, extract_faces_from_tetrahedra

# New mesh generation submodules (for internal use)
from .mesh_utils import (
    validate_surface_mesh,
    validate_bounding_box,
    compute_face_normals,
    check_reached_boundary,
    check_mesh_quality
)
from .mesh_extrusion import (
    extrude_layers,
    extrude_single_layer,
    convert_layers_to_tetrahedra
)
from .mesh_background import (
    generate_hybrid_mesh,
    generate_cartesian_grid,
    remove_overlapping_cells,
    merge_meshes
)
from .mesh_boundary import (
    identify_boundaries_from_surface,
    map_surface_boundaries
)

# NAS export module
from .nas_export import export_volume_mesh_to_nas

__all__ = [
    # Data structures (modularized)
    "GridData",
    "NodeArray",
    "CellArray",
    "BoundaryMap",
    "GridMetadata",
    "CupyNodeArray",
    "CupyCellArray",
    "CupyGridData",
    "TetrahedralCells",
    "VolumeMeshData",
    "FaceData",
    # Parsers (modularized)
    "NASParser",
    "NASParserError",
    "NASFormatError",
    "NASParseError",
    # Validators
    "GridValidator",
    # Generators
    "VolumeMeshGenerator",
    # Quality validator
    "MeshQualityValidator",
    "MeshQualityReport",
    # Face extraction
    "FaceExtractor",
    "extract_faces_from_tetrahedra",
    # Internal mesh generation utilities (not public API)
    # mesh_utils
    "validate_surface_mesh",
    "validate_bounding_box",
    "compute_face_normals",
    "check_reached_boundary",
    "check_mesh_quality",
    # mesh_extrusion
    "extrude_layers",
    "extrude_single_layer",
    "convert_layers_to_tetrahedra",
    # mesh_background
    "generate_hybrid_mesh",
    "generate_cartesian_grid",
    "remove_overlapping_cells",
    "merge_meshes",
    # mesh_boundary
    "identify_boundaries_from_surface",
    "map_surface_boundaries",
    # NAS export
    "export_volume_mesh_to_nas",
]
