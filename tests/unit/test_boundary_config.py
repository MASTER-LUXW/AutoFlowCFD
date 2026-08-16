"""Test script for boundary condition configuration (v2.0).

This script demonstrates the three configuration modes:
- Auto mode: Automatic boundary detection
- Manual mode: Full YAML configuration
- Hybrid mode: YAML priority + automatic fallback
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import only what we need to avoid full module initialization
import numpy as np


def test_boundary_map_v2():
    """Test BoundaryMap v2.0 data structure."""
    print("=" * 80)
    print("Testing BoundaryMap v2.0 Data Structure")
    print("=" * 80)
    
    from autoflowcfd.grid.structures import BoundaryMap
    
    # Create a v2.0 boundary map with all new fields
    boundary_map = BoundaryMap(
        groups={
            "inlet": np.array([0, 1, 2], dtype=np.int32),
            "outlet": np.array([3, 4, 5], dtype=np.int32),
            "car_body": np.array(list(range(6, 100)), dtype=np.int32),
            "symmetry": np.array([100, 101], dtype=np.int32),
        },
        bc_types={
            "inlet": "VELOCITY_INLET",
            "outlet": "PRESSURE_OUTLET",
            "car_body": "WALL",
            "symmetry": "SYMMETRY",
        },
        property_ids={
            "inlet": 10,
            "outlet": 20,
            "car_body": 30,
            "symmetry": 40,
        },
        property_names={
            10: "INLET",
            20: "OUTLET",
            30: "CAR_BODY",
            40: "SYMMETRY",
        },
        detection_mode="auto",
        parameters={
            "inlet": {"velocity": [33.33, 0.0, 0.0]},
            "car_body": {"wall_function": "standard"},
        }
    )
    
    # Test new methods
    print("\nTesting new BoundaryMap methods:")
    print(f"  Boundary count: {boundary_map.boundary_count}")
    print(f"  Detection mode: {boundary_map.detection_mode}")
    
    for name in boundary_map.boundary_names:
        print(f"\n  {name}:")
        print(f"    Type: {boundary_map.get_boundary_type(name)}")
        print(f"    Property ID: {boundary_map.get_property_id(name)}")
        print(f"    Property Name: {boundary_map.get_property_name(name)}")
        print(f"    Parameters: {boundary_map.get_parameters(name)}")
        print(f"    Cell count: {len(boundary_map.get_cell_indices(name))}")
    
    # Test summary
    summary = boundary_map.get_summary()
    print(f"\nSummary:")
    print(f"  Total boundaries: {summary['total_boundaries']}")
    print(f"  Mode: {summary['detection_mode']}")
    
    print("\n✓ BoundaryMap v2.0 test passed!\n")


def test_boundary_type_mapper():
    """Test BoundaryTypeMapper."""
    print("=" * 80)
    print("Testing BoundaryTypeMapper")
    print("=" * 80)
    
    from autoflowcfd.boundary.config import BoundaryTypeMapper
    
    mapper = BoundaryTypeMapper()
    
    test_cases = [
        ("INLET", "VELOCITY_INLET"),
        ("Velocity_Inlet", "VELOCITY_INLET"),
        ("OUTLET", "PRESSURE_OUTLET"),
        ("Pressure_Outlet", "PRESSURE_OUTLET"),
        ("SYMMETRY", "SYMMETRY"),
        ("Symm_Plane", "SYMMETRY"),
        ("TUNNEL_WALL", "SLIP_WALL"),
        ("FARFIELD", "SLIP_WALL"),
        ("CAR_BODY", "WALL"),
        ("GROUND", "WALL"),
        ("UNKNOWN_PROP", "WALL"),  # Default to WALL
    ]
    
    print("\nTesting keyword mapping:")
    for prop_name, expected_type in test_cases:
        result = mapper.map(prop_name)
        status = "✓" if result == expected_type else "✗"
        print(f"  {status} {prop_name:20s} -> {result:20s} (expected: {expected_type})")
        assert result == expected_type, f"Expected {expected_type}, got {result}"
    
    print("\n✓ BoundaryTypeMapper test passed!\n")


def test_parameter_validator():
    """Test ParameterValidator."""
    print("=" * 80)
    print("Testing ParameterValidator")
    print("=" * 80)
    
    from autoflowcfd.boundary.config import ParameterValidator
    
    validator = ParameterValidator()
    
    # Test velocity validation
    print("\nTesting velocity validation:")
    try:
        validator.validate_velocity([30.0, 0.0, 0.0])
        print("  ✓ Valid velocity accepted")
    except Exception as e:
        print(f"  ✗ Valid velocity rejected: {e}")
        raise
    
    try:
        validator.validate_velocity([30.0, 0.0])  # Wrong length
        print("  ✗ Invalid velocity accepted")
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        print("  ✓ Invalid velocity rejected")
    
    # Test pressure validation
    print("\nTesting pressure validation:")
    try:
        validator.validate_pressure(0.0)
        print("  ✓ Zero pressure accepted")
    except Exception as e:
        print(f"  ✗ Zero pressure rejected: {e}")
        raise
    
    try:
        validator.validate_pressure(-1000.0)  # Negative (vacuum)
        print("  ✓ Negative pressure (vacuum) accepted")
    except Exception as e:
        print(f"  ✗ Negative pressure rejected: {e}")
        raise
    
    # Test turbulence intensity validation
    print("\nTesting turbulence intensity validation:")
    try:
        validator.validate_turbulence_intensity(0.05)
        print("  ✓ Normal TI accepted")
    except Exception as e:
        print(f"  ✗ Normal TI rejected: {e}")
        raise
    
    try:
        validator.validate_turbulence_intensity(1.5)  # Out of range
        print("  ✗ Invalid TI accepted")
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        print("  ✓ Invalid TI rejected")
    
    # Test roughness height validation
    print("\nTesting roughness height validation:")
    try:
        validator.validate_roughness_height(0.0001)
        print("  ✓ Valid roughness accepted")
    except Exception as e:
        print(f"  ✗ Valid roughness rejected: {e}")
        raise
    
    try:
        validator.validate_roughness_height(-0.0001)  # Negative
        print("  ✗ Negative roughness accepted")
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        print("  ✓ Negative roughness rejected")
    
    print("\n✓ ParameterValidator test passed!\n")


def test_yaml_config_loader():
    """Test YAMLConfigLoader."""
    print("=" * 80)
    print("Testing YAMLConfigLoader")
    print("=" * 80)
    
    try:
        import yaml
    except ImportError:
        print("\n⚠ PyYAML not installed. Skipping YAML tests.")
        print("Install with: pip install pyyaml\n")
        return
    
    from autoflowcfd.boundary.config import YAMLConfigLoader
    
    loader = YAMLConfigLoader()
    
    # Test loading example config
    config_path = Path(__file__).parent.parent / "examples" / "boundary_config_example.yaml"
    
    if not config_path.exists():
        # Try alternative path
        config_path = Path(__file__).parent.parent.parent / "examples" / "boundary_config_example.yaml"
    
    if config_path.exists():
        print(f"\nLoading config from: {config_path}")
        try:
            config = loader.load(str(config_path))
            
            print(f"\nConfiguration loaded successfully:")
            print(f"  Mode: {loader.get_mode()}")
            
            properties = loader.get_properties_mapping()
            print(f"  Properties configured: {len(properties)}")
            for prop_name, prop_config in properties.items():
                print(f"    - {prop_name}: {prop_config['type']}")
            
            defaults = loader.get_defaults()
            print(f"  Default sections: {list(defaults.keys())}")
            
            print("\n✓ YAMLConfigLoader test passed!\n")
        except Exception as e:
            print(f"\n✗ YAMLConfigLoader test failed: {e}\n")
            raise
    else:
        print(f"\n⚠ Config file not found: {config_path}")
        print("Skipping YAML loader test\n")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("AutoFlowCFD Boundary Condition Configuration Tests (v2.0)")
    print("=" * 80 + "\n")
    
    try:
        test_boundary_map_v2()
        test_boundary_type_mapper()
        test_parameter_validator()
        test_yaml_config_loader()
        
        print("=" * 80)
        print("All tests completed successfully! ✓")
        print("=" * 80)
    except Exception as e:
        print("=" * 80)
        print(f"Tests failed with error: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        sys.exit(1)
