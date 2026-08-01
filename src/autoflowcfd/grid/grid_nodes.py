"""Backward-compatibility shim for pickle deserialization.

This module used to live here directly; its content moved to
`autoflowcfd.grid.schema.grid_nodes` when grid/ was split into
schema/nas_io/mesh_gen/validation subpackages. Python's pickle format
bakes in the exact module path a class was defined under at pickle time
(e.g. `volume_mesh.pkl` written by an older AutoFlowCFD version), so
existing pickled VolumeMeshData/GridData files still need
`autoflowcfd.grid.grid_nodes.NodeArray` to resolve - this module exists
purely so unpickling old files keeps working. New code should import
from `autoflowcfd.grid.schema.grid_nodes` (or `autoflowcfd.grid.structures`)
instead.
"""

from .schema.grid_nodes import NodeArray, CupyNodeArray

__all__ = ['NodeArray', 'CupyNodeArray']
