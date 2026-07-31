"""Simple VTK export test without full AutoFlowCFD dependencies.

This script demonstrates VTK file format and can be run independently.
For full functionality, use the main export_vtk_example.py after installing dependencies.
"""

import numpy as np
from pathlib import Path


def create_simple_vtk(output_path: str):
    """Create a simple VTK file for testing.
    
    Args:
        output_path: Output VTK file path
    """
    print(f"Creating simple VTK file: {output_path}")
    
    # Create a simple 2x2x2 grid (8 nodes, 1 hex cell)
    nodes = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ])
    
    # Hexahedral cell connectivity (8 nodes)
    # For simplicity, we'll split into triangles
    cells = np.array([
        [0, 1, 2],  # Triangle 1
        [0, 2, 3],  # Triangle 2
        [4, 5, 6],  # Triangle 3
        [4, 6, 7],  # Triangle 4
    ])
    
    n_nodes = len(nodes)
    n_cells = len(cells)
    
    # Write VTK file (Legacy format)
    with open(output_path, 'w') as f:
        # Header
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Simple Test Grid\n")
        f.write("ASCII\n\n")
        
        # Dataset structure
        f.write("DATASET UNSTRUCTURED_GRID\n\n")
        
        # Points
        f.write(f"POINTS {n_nodes} float\n")
        for node in nodes:
            f.write(f"{node[0]:.6f} {node[1]:.6f} {node[2]:.6f}\n")
        f.write("\n")
        
        # Cells
        f.write(f"CELLS {n_cells} {n_cells * 4}\n")
        for cell in cells:
            f.write(f"3 {cell[0]} {cell[1]} {cell[2]}\n")
        f.write("\n")
        
        # Cell types (5 = triangle)
        f.write(f"CELL_TYPES {n_cells}\n")
        for _ in range(n_cells):
            f.write("5\n")
        f.write("\n")
        
        # Point data
        f.write(f"POINT_DATA {n_nodes}\n\n")
        
        # Velocity vectors
        f.write("VECTORS Velocity float\n")
        velocities = np.array([
            [10.0, 0.0, 0.0],
            [12.0, 0.5, 0.0],
            [12.0, -0.5, 0.0],
            [10.0, 0.0, 0.0],
            [15.0, 0.0, 0.2],
            [18.0, 0.8, 0.2],
            [18.0, -0.8, 0.2],
            [15.0, 0.0, 0.2],
        ])
        for vel in velocities:
            f.write(f"{vel[0]:.6f} {vel[1]:.6f} {vel[2]:.6f}\n")
        f.write("\n")
        
        # Pressure scalars
        f.write("SCALARS Pressure float 1\n")
        f.write("LOOKUP_TABLE default\n")
        pressures = np.array([101325.0, 101300.0, 101300.0, 101325.0,
                             101250.0, 101200.0, 101200.0, 101250.0])
        for p in pressures:
            f.write(f"{p:.2f}\n")
        f.write("\n")
    
    print(f"✅ VTK file created: {output_path}")
    print(f"   Nodes: {n_nodes}")
    print(f"   Cells: {n_cells}")
    print(f"   Fields: Velocity, Pressure")


if __name__ == "__main__":
    output_dir = Path("./vtk_test_output")
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("Simple VTK Export Test")
    print("=" * 70)
    
    # Create test VTK file
    vtk_path = output_dir / "test_grid.vtk"
    create_simple_vtk(str(vtk_path))
    
    print("\n" + "=" * 70)
    print("Next Steps:")
    print("  1. Open ParaView")
    print(f"  2. File -> Open -> {vtk_path.absolute()}")
    print("  3. Click Apply")
    print("  4. View velocity and pressure fields")
    print("=" * 70)
