"""Example script demonstrating grid parsing workflow.

This example shows how to:
1. Parse an ANSA .nas mesh file
2. Validate mesh quality
3. Save/load grid data via HDF5
4. Access grid statistics

Usage:
    python examples/grid_parsing_example.py
"""

import sys
from pathlib import Path

# Add src to path for development testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from autoflowcfd.grid import NASParser, GridValidator


def main():
    """Demonstrate complete grid parsing workflow."""
    
    print("=" * 70)
    print("AutoFlowCFD - Grid Parsing Example")
    print("=" * 70)
    
    # Get the demo NAS file
    demo_file = Path(__file__).parent / "ahmed_body_demo.nas"
    
    if not demo_file.exists():
        print(f"Error: Demo file not found: {demo_file}")
        print("Please ensure examples/ahmed_body_demo.nas exists")
        return
    
    print(f"\n[1/4] Parsing NAS file: {demo_file.name}")
    print("-" * 70)
    
    # Step 1: Parse the NAS file
    parser = NASParser(str(demo_file))
    
    # Get file info before parsing
    file_info = parser.get_file_info()
    print(f"File size: {file_info['file_size_mb']:.2f} MB")
    print(f"Estimated nodes: {file_info['estimated_nodes']}")
    print(f"Estimated cells: {file_info['estimated_cells']}")
    print(f"Detected format: {file_info['version']}")
    
    print("\nParsing...")
    grid = parser.parse()
    
    print(f"\n✓ Successfully parsed!")
    print(f"  Nodes: {grid.metadata.node_count:,}")
    print(f"  Cells: {grid.metadata.cell_count:,}")
    print(f"  Format: {grid.metadata.file_format}")
    
    # Step 2: Display metadata
    print("\n[2/4] Grid Metadata")
    print("-" * 70)
    print(grid.metadata.summary())
    
    # Step 3: Validate mesh quality
    print("\n[3/4] Validating Mesh Quality")
    print("-" * 70)
    
    validator = GridValidator(grid)
    results = validator.validate()
    
    print(f"\nValidation Result: {'✓ PASSED' if results['passed'] else '✗ FAILED'}")
    
    if not results['passed']:
        print("\nQuality issues detected:")
        ar = results['aspect_ratio']
        sk = results['skewness']
        jac = results['jacobian']
        
        if ar['max'] > validator.thresholds['aspect_ratio_max']:
            print(f"  - High aspect ratio: {ar['max']:.2f}")
        if sk['max'] > validator.thresholds['skewness_max']:
            print(f"  - High skewness: {sk['max']:.3f}")
        if jac['min'] < validator.thresholds['jacobian_min']:
            print(f"  - Low Jacobian: {jac['min']:.2e}")
    
    # Step 4: Demonstrate HDF5 save/load
    print("\n[4/4] Testing HDF5 Serialization")
    print("-" * 70)
    
    try:
        import h5py
        
        h5_file = Path(__file__).parent / "ahmed_body_demo.h5"
        
        print(f"Saving to: {h5_file.name}")
        grid.save_hdf5(str(h5_file))
        print("✓ Save successful")
        
        print(f"Loading from: {h5_file.name}")
        loaded_grid = type(grid).load_hdf5(str(h5_file))
        print("✓ Load successful")
        
        # Verify data integrity
        assert loaded_grid.node_count == grid.node_count
        assert loaded_grid.cell_count == grid.cell_count
        print("✓ Data integrity verified")
        
        # Clean up
        h5_file.unlink()
        print("✓ Temporary file cleaned up")
        
    except ImportError:
        print("⚠ h5py not installed. Skipping HDF5 test.")
        print("  Install with: pip install h5py")
    
    # Summary
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)
    
    print("\nNext steps:")
    print("  - Modify the mesh in ANSA and re-parse")
    print("  - Adjust quality thresholds in GridValidator")
    print("  - Integrate with solver (Iteration 3)")
    print("  - Explore GPU acceleration with grid.to_gpu()")


if __name__ == "__main__":
    main()
