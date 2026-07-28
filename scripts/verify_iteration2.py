"""Quick verification script for Iteration 2 deliverables."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from autoflowcfd.grid import (
            NASParser,
            GridValidator,
            GridData,
            NodeArray,
            CellArray,
            BoundaryMap,
            GridMetadata,
        )
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality without file I/O."""
    print("\nTesting basic functionality...")
    
    try:
        import numpy as np
        from autoflowcfd.grid import (
            NodeArray,
            CellArray,
            BoundaryMap,
            GridMetadata,
            GridData,
            GridValidator,
        )
        
        # Create minimal grid
        nodes = NodeArray(
            x=np.array([0.0, 1.0, 0.5], dtype=np.float64),
            y=np.array([0.0, 0.0, 0.866], dtype=np.float64),
            z=np.array([0.0, 0.0, 0.0], dtype=np.float64)
        )
        
        cells = CellArray(
            connectivity=np.array([[0, 1, 2]], dtype=np.int32),
            cell_type=np.array([0], dtype=np.int32)
        )
        
        boundaries = BoundaryMap(
            groups={"wall": np.array([0, 1, 2], dtype=np.int32)},
            bc_types={"wall": "WALL"}
        )
        
        metadata = GridMetadata(
            node_count=3,
            cell_count=1,
            boundary_groups=["wall"],
            file_format="v24"
        )
        
        grid = GridData(
            nodes=nodes,
            cells=cells,
            boundaries=boundaries,
            metadata=metadata
        )
        
        # Validate
        validator = GridValidator(grid)
        results = validator.validate()
        
        assert results['passed'] is True
        assert 'aspect_ratio' in results
        assert 'skewness' in results
        assert 'jacobian' in results
        
        print("✅ Basic functionality test passed")
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_nas_parser_exists():
    """Test that NAS parser class exists and has required methods."""
    print("\nTesting NAS parser structure...")
    
    try:
        from autoflowcfd.grid import NASParser
        
        # Check required methods exist
        required_methods = [
            'parse',
            '_detect_version',
            '_parse_nodes',
            '_parse_cells',
            '_parse_boundaries',
            'get_file_info',
        ]
        
        for method in required_methods:
            assert hasattr(NASParser, method), f"Missing method: {method}"
        
        print("✅ NAS parser structure verified")
        return True
        
    except Exception as e:
        print(f"❌ NAS parser check failed: {e}")
        return False


def test_file_structure():
    """Test that all expected files exist."""
    print("\nTesting file structure...")
    
    root = Path(__file__).parent.parent
    
    expected_files = [
        root / "src" / "autoflowcfd" / "grid" / "__init__.py",
        root / "src" / "autoflowcfd" / "grid" / "structures.py",
        root / "src" / "autoflowcfd" / "grid" / "parser.py",
        root / "src" / "autoflowcfd" / "grid" / "validator.py",
        root / "tests" / "unit" / "test_grid_structures.py",
        root / "tests" / "unit" / "test_nas_parser.py",
        root / "tests" / "unit" / "test_grid_validator.py",
        root / "tests" / "integration" / "test_grid_parsing.py",
        root / "examples" / "ahmed_body_demo.nas",
        root / "examples" / "grid_parsing_example.py",
    ]
    
    missing = []
    for filepath in expected_files:
        if not filepath.exists():
            missing.append(filepath)
    
    if missing:
        print(f"❌ Missing files:")
        for f in missing:
            print(f"   - {f}")
        return False
    else:
        print("✅ All expected files present")
        return True


def main():
    """Run all verification tests."""
    print("=" * 70)
    print("AutoFlowCFD Iteration 2 Verification")
    print("=" * 70)
    
    results = []
    
    results.append(("File Structure", test_file_structure()))
    results.append(("Imports", test_imports()))
    results.append(("NAS Parser Structure", test_nas_parser_exists()))
    results.append(("Basic Functionality", test_basic_functionality()))
    
    print("\n" + "=" * 70)
    print("Verification Summary")
    print("=" * 70)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:.<50} {status}")
    
    all_passed = all(r for _, r in results)
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL TESTS PASSED - Iteration 2 is ready!")
    else:
        print("❌ SOME TESTS FAILED - Please review")
    print("=" * 70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
