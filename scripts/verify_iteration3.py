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
        ("autoflowcfd.core.fr_scheme", ["FRScheme", "FROrder"]),
        ("autoflowcfd.core.turbulence", ["SSTKOmegaModel"]),
        ("autoflowcfd.core.wall_functions", ["WallFunctionModel"]),
        ("autoflowcfd.core.time_integration", ["TimeIntegrator", "TimeIntegrationScheme"]),
        ("autoflowcfd.core.convergence", ["ConvergenceMonitor", "ConvergenceHistory"]),
        ("autoflowcfd.core.solver_transient", ["TransientSolver", "TransientResult"]),
        ("autoflowcfd.core.coupling", ["SteadyTransientCoupler", "SyntheticTurbulenceGenerator"]),
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
    
    # Test FR Scheme
    print("\n1. Testing FR Scheme...")
    try:
        from autoflowcfd.core import FRScheme, FROrder
        
        fr = FRScheme(FROrder.SECOND)
        assert fr.order == FROrder.SECOND
        assert fr.num_correction_points == 3
        
        sol_left = np.array([1.225, 36.75, 0.0, 0.0, 100000.0])
        sol_right = np.array([1.225, 30.0, 0.0, 0.0, 98000.0])
        normal = np.array([1.0, 0.0, 0.0])
        
        flux = fr.compute_flux(sol_left, sol_right, normal)
        assert flux.shape == (5,)
        assert np.all(np.isfinite(flux))
        
        print("   ✅ FR Scheme: OK")
    
    except Exception as e:
        print(f"   ❌ FR Scheme: FAILED - {e}")
        return False
    
    # Test Backend
    print("\n2. Testing Backend...")
    try:
        from autoflowcfd.core import create_backend
        
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
    
    # Test Turbulence Model
    print("\n3. Testing SST k-ω Model...")
    try:
        from autoflowcfd.core import SSTKOmegaModel
        
        model = SSTKOmegaModel()
        k, omega = model.initialize_turbulence_fields(100)
        
        assert k.shape == (100,)
        assert omega.shape == (100,)
        assert np.all(k > 0)
        assert np.all(omega > 0)
        
        print("   ✅ SST k-ω Model: OK")
    
    except Exception as e:
        print(f"   ❌ SST k-ω Model: FAILED - {e}")
        return False
    
    # Test Time Integrator
    print("\n4. Testing Time Integrator...")
    try:
        from autoflowcfd.core import TimeIntegrator, TimeIntegrationScheme
        
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
    
    # Test Wall Function
    print("\n5. Testing Wall Function...")
    try:
        from autoflowcfd.core import WallFunctionModel
        
        wf = WallFunctionModel()
        y_plus = wf.compute_y_plus(
            u_tau=np.array([1.0]),
            y_distance=np.array([0.001]),
            nu=1.5e-5
        )
        
        assert y_plus.shape == (1,)
        assert y_plus[0] > 0
        
        print("   ✅ Wall Function: OK")
    
    except Exception as e:
        print(f"   ❌ Wall Function: FAILED - {e}")
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
