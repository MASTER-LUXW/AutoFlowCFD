#!/usr/bin/env python
"""Quick verification script for Iteration 3 components.

This script verifies that all core modules can be imported successfully
and basic functionality works as expected.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def verify_imports():
    """Verify all module imports."""
    print("=" * 70)
    print("AutoFlowCFD Iteration 3 - Module Import Verification")
    print("=" * 70)
    
    modules_to_test = [
        ("autoflowcfd.fr", ["FROperators", "generate_fr_operators"]),
        ("autoflowcfd.core.fr_state", ["FRState"]),
        ("autoflowcfd.core.fr_kernels", ["compute_ausm_up_flux"]),
        ("autoflowcfd.core.fr_solver", ["FRSolver"]),
        ("autoflowcfd.core.time_integration", ["TimeIntegrator", "TimeIntegrationScheme"]),
        ("autoflowcfd.core.backend", ["create_backend", "get_available_backends"]),
    ]
    
    failed = []
    
    for module_name, classes in modules_to_test:
        try:
            module = __import__(module_name, fromlist=classes)
            
            # Verify classes exist
            for class_name in classes:
                if not hasattr(module, class_name):
                    raise AttributeError(f"Class {class_name} not found in {module_name}")
            
            print(f"✅ {module_name:<50} OK")
        
        except Exception as e:
            print(f"❌ {module_name:<50} FAILED: {e}")
            failed.append(module_name)
    
    print()
    
    if failed:
        print(f"❌ {len(failed)} module(s) failed to import")
        return False
    else:
        print("✅ All modules imported successfully!")
        return True


def verify_basic_functionality():
    """Verify basic functionality of key components."""
    print("\n" + "=" * 70)
    print("Basic Functionality Verification")
    print("=" * 70)
    
    import numpy as np
    
    # Test FR Operators
    print("\n1. Testing FR Operators...")
    try:
        from autoflowcfd.fr import FROperators, generate_fr_operators
        
        ops = generate_fr_operators(order=2)
        assert isinstance(ops, FROperators)
        assert ops.correction_matrix.shape[0] > 0
        assert ops.interpolation_matrix.shape[0] > 0
        
        print("   ✅ FR Operators: OK")
    
    except Exception as e:
        print(f"   ❌ FR Operators: FAILED - {e}")
        return False
    
    # Test FR State
    print("\n2. Testing FR State...")
    try:
        from autoflowcfd.core.fr_state import FRState
        
        state = FRState(n_cells=100, n_solution_points=5)
        assert state.density.shape == (100,)
        assert state.velocity.shape == (100, 3)
        assert state.pressure.shape == (100,)
        assert np.all(state.density == 1.225)  # Default air density
        
        print("   ✅ FR State: OK")
    
    except Exception as e:
        print(f"   ❌ FR State: FAILED - {e}")
        return False
    
    # Test AUSM+ Up Flux
    print("\n3. Testing AUSM+ Up Flux...")
    try:
        from autoflowcfd.core.fr_kernels import compute_ausm_up_flux
        
        left_state = np.array([1.225, 36.75, 0.0, 0.0, 100000.0])
        right_state = np.array([1.225, 30.0, 0.0, 0.0, 98000.0])
        normal = np.array([1.0, 0.0, 0.0])
        
        flux = compute_ausm_up_flux(left_state, right_state, normal)
        assert flux.shape == (5,)
        assert np.all(np.isfinite(flux))
        
        print("   ✅ AUSM+ Up Flux: OK")
    
    except Exception as e:
        print(f"   ❌ AUSM+ Up Flux: FAILED - {e}")
        return False
    
    # Test FR Solver
    print("\n4. Testing FR Solver...")
    try:
        from autoflowcfd.core.fr_solver import FRSolver
        from autoflowcfd.core.backend import create_backend
        
        backend = create_backend("cpu")
        solver = FRSolver(backend=backend, polynomial_order=2)
        assert solver.polynomial_order == 2
        assert solver.backend is not None
        
        print("   ✅ FR Solver: OK")
    
    except Exception as e:
        print(f"   ❌ FR Solver: FAILED - {e}")
        return False
    
    # Test Time Integrator
    print("\n5. Testing Time Integrator...")
    try:
        from autoflowcfd.core.time_integration import TimeIntegrator, TimeIntegrationScheme
        
        integrator = TimeIntegrator(
            scheme=TimeIntegrationScheme.BACKWARD_EULER,
            dt=1e-5
        )
        
        assert integrator.scheme == TimeIntegrationScheme.BACKWARD_EULER
        assert integrator.dt == 1e-5
        
        print("   ✅ Time Integrator: OK")
    
    except Exception as e:
        print(f"   ❌ Time Integrator: FAILED - {e}")
        return False
    
    # Test Backend
    print("\n6. Testing Backend...")
    try:
        from autoflowcfd.core.backend import create_backend
        
        backend = create_backend("cpu", n_threads=2)
        assert backend.backend_type == "cpu"
        
        backend.initialize(n_cells=100, n_nodes=50)
        assert backend.n_cells == 100
        
        info = backend.get_device_info()
        assert "backend" in info
        
        print("   ✅ Backend: OK")
    
    except Exception as e:
        print(f"   ❌ Backend: FAILED - {e}")
        return False
    
    print("\n" + "=" * 70)
    print("✅ ALL BASIC FUNCTIONALITY TESTS PASSED!")
    print("=" * 70)
    
    return True


def main():
    """Run all verification tests."""
    print()
    
    # Verify imports
    imports_ok = verify_imports()
    
    if not imports_ok:
        print("\n❌ Import verification failed. Aborting.")
        return 1
    
    # Verify functionality
    func_ok = verify_basic_functionality()
    
    if not func_ok:
        print("\n❌ Functionality verification failed.")
        return 1
    
    print("\n🎉 Iteration 3 verification completed successfully!")
    print("\nNext steps:")
    print("  1. Run unit tests: pytest tests/unit/ -v")
    print("  2. Run integration test: python tests/integration/test_iteration3_solver.py")
    print("  3. Run example: python examples/steady_rans_example.py")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
