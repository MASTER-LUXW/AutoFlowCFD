"""Backward-compatibility shim for pickle deserialization.

See grid_nodes.py's module docstring - same reasoning, this module's
content moved to `autoflowcfd.grid.schema.grid_boundaries`.
"""

from .schema.grid_boundaries import BoundaryMap

__all__ = ['BoundaryMap']
