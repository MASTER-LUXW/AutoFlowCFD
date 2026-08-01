"""Backward-compatibility shim for pickle deserialization.

See grid_nodes.py's module docstring - same reasoning, this module's
content moved to `autoflowcfd.grid.schema.grid_faces`.
"""

from .schema.grid_faces import FaceData

__all__ = ['FaceData']
