"""Backward-compatibility shim for pickle deserialization.

See grid_nodes.py's module docstring - same reasoning, this module's
content moved to `autoflowcfd.grid.schema.grid_metadata`.
"""

from .schema.grid_metadata import GridMetadata

__all__ = ['GridMetadata']
