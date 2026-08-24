"""Quick diagnostic: trace boundary condition mapping through the pipeline."""
import sys
import logging
# Suppress loguru to reduce noise
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="ERROR")

from autoflowcfd.grid.nas_io.parser_core import NASParser
from autoflowcfd.grid.nas_io.nas_parser_volume import parse_volume_mesh_nas
from autoflowcfd.grid.mesh_gen.utils.mesh_boundary import map_boundaries_by_geometry

SURFACE = r"C:\Users\luxw_\Desktop\AutoFlowCFD\cube_demo\cube_demo.nas"
VOLUME  = r"C:\Users\luxw_\Desktop\AutoFlowCFD\cube_demo\cube_demo_volume.nas"

print("=== Step 1: Parse surface mesh ===")
parser = NASParser(SURFACE)
surface_grid = parser.parse(skip_validation=True)
print(f"  Surface boundary groups: {list(surface_grid.boundaries.groups.keys())}")
print(f"  Surface BC types: {dict(surface_grid.boundaries.bc_types)}")
for name, count in [(k, len(v)) for k, v in surface_grid.boundaries.groups.items()]:
    print(f"    {name}: {count} faces, bc_type={surface_grid.boundaries.bc_types.get(name)}")

print("\n=== Step 2: Parse volume mesh ===")
volume_mesh = parse_volume_mesh_nas(VOLUME, units='mm')
print(f"  Volume boundaries (before mapping): groups={list(volume_mesh.boundaries.groups.keys())}, bc_types={dict(volume_mesh.boundaries.bc_types)}")

print("\n=== Step 3: Map boundaries by geometry ===")
boundaries = map_boundaries_by_geometry(volume_mesh, surface_grid)
print(f"  Mapped boundary groups: {list(boundaries.groups.keys())}")
print(f"  Mapped BC types: {dict(boundaries.bc_types)}")
for name, count in [(k, len(v)) for k, v in boundaries.groups.items()]:
    print(f"    {name}: {count} cells, bc_type={boundaries.bc_types.get(name)}")

print("\n=== Diagnosis complete ===")
