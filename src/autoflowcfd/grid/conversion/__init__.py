"""网格格式转换：.nas（面网格或体网格）-> VTK / STL / CGNS（`autoflowcfd grid convert`）。"""

from .cgns import write_cgns
from .source import ConvertibleMesh, load_convertible_mesh
from .writers import write_stl, write_vtk

WRITERS = {"vtk": write_vtk, "stl": write_stl, "cgns": write_cgns}
EXTENSIONS = {"vtk": ".vtu", "stl": ".stl", "cgns": ".cgns"}

__all__ = ["ConvertibleMesh", "load_convertible_mesh", "write_vtk", "write_stl", "write_cgns", "WRITERS", "EXTENSIONS"]
