"""Example: Export VTK files for visualization in ParaView.

This example demonstrates how to export CFD simulation results to VTK format
for visualization and analysis in ParaView.

Key Features:
    - Export velocity field (vector)
    - Export pressure field (scalar)
    - Export turbulence variables (k, omega)
    - Support both legacy (.vtk) and XML (.vtu) formats

Usage:
    $ python examples/export_vtk_example.py
    
Output:
    - results/velocity_field.vtk
    - results/pressure_field.vtk
    - results/full_flow_field.vtk
"""

from pathlib import Path
import numpy as np
from loguru import logger

# Import AutoFlowCFD modules
from autoflowcfd.grid import GridData, NASParser
from autoflowcfd.core.backend.base import SolutionVector
from autoflowcfd.postprocess import VTKExporter


def create_sample_grid_and_solution():
    """Create sample grid data and solution for demonstration.
    
    In real usage, you would load actual grid and run simulation.
    This function creates placeholder data for testing VTK export.
    
    Returns:
        Tuple[GridData, SolutionVector]: Sample grid and solution
    """
    logger.info("Creating sample grid and solution for demonstration...")
    
    # Create a simple 3D box grid (10x5x5 = 250 cells)
    n_nodes_x, n_nodes_y, n_nodes_z = 11, 6, 6
    n_cells_x, n_cells_y, n_cells_z = 10, 5, 5
    
    # Generate node coordinates
    nodes_x = np.linspace(0, 5.0, n_nodes_x)  # 5m length
    nodes_y = np.linspace(-1.0, 1.0, n_nodes_y)  # 2m width
    nodes_z = np.linspace(0, 1.5, n_nodes_z)  # 1.5m height
    
    # Create 3D meshgrid
    xx, yy, zz = np.meshgrid(nodes_x, nodes_y, nodes_z, indexing='ij')
    
    # Flatten to 1D arrays
    x_coords = xx.flatten()
    y_coords = yy.flatten()
    z_coords = zz.flatten()
    
    n_nodes = len(x_coords)
    n_cells = n_cells_x * n_cells_y * n_cells_z
    
    logger.info(f"Grid size: {n_nodes} nodes, {n_cells} cells")
    
    # Create simple GridData structure (placeholder)
    # In real usage, load from .nas file using NASParser
    class SimpleNodes:
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z
            self.count = len(x)
    
    class SimpleCells:
        def __init__(self, connectivity, count):
            self.connectivity = connectivity
            self.count = count
    
    class SimpleMetadata:
        def __init__(self, node_count, cell_count):
            self.node_count = node_count
            self.cell_count = cell_count
    
    class SimpleBoundaries:
        groups = {}
        boundary_names = []
    
    # Create hexahedral cell connectivity
    connectivity = []
    for k in range(n_cells_z):
        for j in range(n_cells_y):
            for i in range(n_cells_x):
                # Calculate node indices for this cell
                n0 = i + j * n_nodes_x + k * n_nodes_x * n_nodes_y
                n1 = n0 + 1
                n2 = n0 + n_nodes_x
                n3 = n1 + n_nodes_x
                n4 = n0 + n_nodes_x * n_nodes_y
                n5 = n1 + n_nodes_x * n_nodes_y
                n6 = n2 + n_nodes_x * n_nodes_y
                n7 = n3 + n_nodes_x * n_nodes_y
                
                # For VTK triangle export, split hex into triangles
                # Simplified: just use first 3 nodes as triangle
                connectivity.append([n0, n1, n2])
    
    # Create grid data object
    grid_data = GridData.__new__(GridData)
    grid_data.nodes = SimpleNodes(x_coords, y_coords, z_coords)
    grid_data.cells = SimpleCells(np.array(connectivity), len(connectivity))
    grid_data.metadata = SimpleMetadata(n_nodes, len(connectivity))
    grid_data.boundaries = SimpleBoundaries()
    
    # Create sample solution vector
    # Solution format: [rho, rho*u, rho*v, rho*w, E]
    solution = SolutionVector.__new__(SolutionVector)
    solution.n_cells = n_cells
    
    # Initialize with uniform flow field
    rho_inf = 1.225  # kg/m³
    u_inf = 30.0     # m/s
    p_inf = 101325.0 # Pa
    
    # Conservative variables
    solution.rho = np.full(n_cells, rho_inf)
    solution.rhou = np.full(n_cells, rho_inf * u_inf)
    solution.rhov = np.zeros(n_cells)
    solution.rhow = np.zeros(n_cells)
    solution.E = np.full(n_cells, p_inf / 0.4 + 0.5 * rho_inf * u_inf**2)
    
    # Add some variation for visualization
    # Velocity increases linearly along x
    x_cell_centers = np.zeros(n_cells)
    idx = 0
    for k in range(n_cells_z):
        for j in range(n_cells_y):
            for i in range(n_cells_x):
                x_cell_centers[idx] = (i + 0.5) * (5.0 / n_cells_x)
                idx += 1
    
    # Modify velocity profile (boundary layer effect)
    velocity_factor = 1.0 + 0.2 * np.sin(x_cell_centers * np.pi / 5.0)
    solution.rhou = rho_inf * u_inf * velocity_factor
    
    # Add pressure gradient
    solution.E = p_inf / 0.4 + 0.5 * rho_inf * (u_inf * velocity_factor)**2
    
    logger.success("Sample grid and solution created successfully")
    
    return grid_data, solution


def export_vtk_files(output_dir: str = "./vtk_output"):
    """Export VTK files for visualization.
    
    Args:
        output_dir: Output directory for VTK files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("AutoFlowCFD - VTK Export Example")
    print("=" * 70)
    
    # Step 1: Create or load grid and solution
    print("\n[Step 1] Preparing grid and solution data...")
    grid_data, solution = create_sample_grid_and_solution()
    
    # Step 2: Create VTK exporter
    print("\n[Step 2] Creating VTK exporter...")
    exporter = VTKExporter(
        grid_data=grid_data,
        solution=solution
    )
    
    # Step 3: Export velocity field only
    print("\n[Step 3] Exporting velocity field...")
    vel_vtk = exporter.export(
        output_path=str(output_path / "velocity_field.vtk"),
        fields=['velocity'],
        format='legacy'
    )
    print(f"  ✅ Velocity field exported: {vel_vtk}")
    
    # Step 4: Export pressure field only
    print("\n[Step 4] Exporting pressure field...")
    pres_vtk = exporter.export(
        output_path=str(output_path / "pressure_field.vtk"),
        fields=['pressure'],
        format='legacy'
    )
    print(f"  ✅ Pressure field exported: {pres_vtk}")
    
    # Step 5: Export complete flow field
    print("\n[Step 5] Exporting complete flow field...")
    full_vtk = exporter.export(
        output_path=str(output_path / "full_flow_field.vtk"),
        fields=['velocity', 'pressure'],
        format='legacy'
    )
    print(f"  ✅ Full flow field exported: {full_vtk}")
    
    # Step 6: Generate README
    readme_path = output_path / "README.txt"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("AutoFlowCFD VTK Export - Visualization Guide\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("Generated VTK Files:\n")
        f.write(f"  1. {vel_vtk.name} - Velocity field only\n")
        f.write(f"  2. {pres_vtk.name} - Pressure field only\n")
        f.write(f"  3. {full_vtk.name} - Complete flow field (velocity + pressure)\n\n")
        
        f.write("How to Visualize in ParaView:\n")
        f.write("-" * 70 + "\n\n")
        
        f.write("1. Install ParaView:\n")
        f.write("   Download from: https://www.paraview.org/download/\n\n")
        
        f.write("2. Open VTK File:\n")
        f.write("   - Launch ParaView\n")
        f.write("   - File -> Open -> Select .vtk file\n")
        f.write("   - Click 'Apply' button to load data\n\n")
        
        f.write("3. View Velocity Cloud Map:\n")
        f.write("   - Select the dataset in Pipeline Browser\n")
        f.write("   - Apply 'Slice' filter to create cross-section\n")
        f.write("     * Slice Type: Plane\n")
        f.write("     * Origin: (2.5, 0, 0.75) - center of domain\n")
        f.write("     * Normal: (0, 1, 0) - YZ plane\n")
        f.write("   - In Coloring dropdown, select 'Velocity' -> 'Magnitude'\n")
        f.write("   - Adjust color map in Edit -> Color Map Editor\n\n")
        
        f.write("4. View Surface Pressure Distribution:\n")
        f.write("   - Select original dataset (not slice)\n")
        f.write("   - Ensure display mode is 'Surface'\n")
        f.write("   - In Coloring dropdown, select 'Pressure'\n")
        f.write("   - Enable Scalar Bar: View -> Scalar Bar Visibility\n")
        f.write("   - Adjust pressure range if needed\n\n")
        
        f.write("5. Advanced Visualization:\n")
        f.write("   - Streamlines: Filters -> Stream Tracer\n")
        f.write("   - Vector arrows: Apply 'Glyph' filter\n")
        f.write("   - Contours: Filters -> Contour (for iso-surfaces)\n\n")
        
        f.write("Tips:\n")
        f.write("  - Use 'Rescale to Data Range' for automatic color scaling\n")
        f.write("  - Try different color schemes (Jet, Rainbow, Cool to Warm)\n")
        f.write("  - Save screenshots: File -> Save Screenshot\n")
        f.write("  - Export animations: File -> Save Animation\n\n")
        
        f.write("Note:\n")
        f.write("  This example uses sample data. For real simulations,\n")
        f.write("  load your .nas grid file and run the solver first.\n")
    
    print(f"\n📖 Visualization guide saved: {readme_path}")
    
    print("\n" + "=" * 70)
    print("✅ VTK Export Complete!")
    print("=" * 70)
    print(f"\nOutput directory: {output_path.absolute()}")
    print("\nNext Steps:")
    print("  1. Open ParaView")
    print("  2. Load one of the .vtk files")
    print("  3. Follow the instructions in README.txt")
    print("=" * 70)
    
    return {
        'velocity': vel_vtk,
        'pressure': pres_vtk,
        'full_field': full_vtk,
        'readme': readme_path
    }


if __name__ == "__main__":
    # Run the example
    results = export_vtk_files(output_dir="./vtk_output")
    
    print("\nExported files:")
    for key, path in results.items():
        print(f"  {key}: {path}")
