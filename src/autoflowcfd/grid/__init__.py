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
from .schema.grid_nodes import NodeArray, CupyNodeArray
from .schema.grid_cells import CellArray, CupyCellArray, TetrahedralCells, PrismCells
from .schema.grid_boundaries import BoundaryMap
from .schema.grid_metadata import GridMetadata
from .schema.grid_faces import FaceData
from .schema.grid_data import GridData, CupyGridData, VolumeMeshData

# Parser modules (modularized)
from .nas_io.parser_core import NASParser
from .nas_io.nas_parser_exceptions import NASParserError, NASFormatError, NASParseError

# Other modules
from .validation.validator import GridValidator
from .mesh_gen.volume_mesh_generator import VolumeMeshGenerator
from .validation.quality_validator import MeshQualityValidator, MeshQualityReport
from .mesh_gen.face_extractor import FaceExtractor, extract_faces_from_tetrahedra

# New mesh generation submodules (for internal use)
from .mesh_gen.mesh_utils import (
    validate_surface_mesh,
    validate_bounding_box,
    compute_face_normals,
    check_reached_boundary,
    check_mesh_quality
)
from .mesh_gen.mesh_extrusion import extrude_layers
from .mesh_gen.mesh_layer_step import extrude_single_layer
from .mesh_gen.mesh_prism_to_tet import convert_layers_to_tetrahedra
from .mesh_gen.mesh_background import generate_hybrid_mesh
from .mesh_gen.mesh_boundary import (
    identify_boundaries_from_surface,
    map_surface_boundaries
)

# NAS export module
from .nas_io.nas_export import export_volume_mesh_to_nas

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
    "PrismCells",
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
    # mesh_boundary
    "identify_boundaries_from_surface",
    "map_surface_boundaries",
    # NAS export
    "export_volume_mesh_to_nas",
]
