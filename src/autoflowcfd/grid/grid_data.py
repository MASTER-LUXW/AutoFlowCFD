"""Backward-compatibility shim for pickle deserialization.

See grid_nodes.py's module docstring - same reasoning, this module's
content moved to `autoflowcfd.grid.schema.grid_data`.
"""

from .schema.grid_data import GridData, CupyGridData, VolumeMeshData

__all__ = ['GridData', 'CupyGridData', 'VolumeMeshData']
