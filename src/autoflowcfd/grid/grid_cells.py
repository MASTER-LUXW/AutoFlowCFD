"""Backward-compatibility shim for pickle deserialization.

See grid_nodes.py's module docstring - same reasoning, this module's
content moved to `autoflowcfd.grid.schema.grid_cells`.
"""

from .schema.grid_cells import CellArray, CupyCellArray, TetrahedralCells

__all__ = ['CellArray', 'CupyCellArray', 'TetrahedralCells']
