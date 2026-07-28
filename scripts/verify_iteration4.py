#!/usr/bin/env python
"""Verification script for Iteration 4 completion.

This script verifies that all Iteration 4 components are properly implemented
and can be imported without errors.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_boundary_module():
    """Test boundary condition module."""
    print("=" * 60)
    print("Testing Boundary Condition Module...")
    print("=" * 60)
    
    try:
        from autoflowcfd.boundary import (
            InletBC, OutletBC, WallBC, GroundBC,
            FarfieldBC, SymmetryBC, BodyBC,
            BoundaryManager, register_boundary_condition
        )
        print("✓ All boundary condition classes imported successfully")
        
        # Test BC creation
        inlet = InletBC(velocity_x=30.0, pressure=101325.0)
        assert inlet.validate() is True
        print("✓ InletBC creation and validation works")
        
        outlet = OutletBC(pressure=101325.0)
        assert outlet.validate() is True
        print("✓ OutletBC creation and validation works")
        
        wall = WallBC(wall_function='standard')
        assert wall.validate() is True
        print("✓ WallBC creation and validation works")
        
        ground = GroundBC(moving=False)
        assert ground.validate() is True
        print("✓ GroundBC creation and validation works")
        
        farfield = FarfieldBC()
        assert farfield.validate() is True
        print("✓ FarfieldBC creation and validation works")
        
        symmetry = SymmetryBC()
        assert symmetry.validate() is True
        print("✓ SymmetryBC creation and validation works")
        
        body = BodyBC()
        assert body.validate() is True
        print("✓ BodyBC creation and validation works")
        
        print("\n✅ Boundary module: ALL TESTS PASSED\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Boundary module test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_cli_module():
    """Test CLI module."""
    print("=" * 60)
    print("Testing CLI Module...")
    print("=" * 60)
    
    try:
        from autoflowcfd.cli.main import cli
        from click.testing import CliRunner
        
        runner = CliRunner()
        
        # Test main help
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "grid" in result.output
        assert "solve" in result.output
        assert "post" in result.output
        assert "config" in result.output
        assert "utils" in result.output
        print("✓ Main CLI help works")
        
        # Test version
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "AutoFlowCFD" in result.output
        print("✓ Version command works")
        
        # Test grid subcommand
        result = runner.invoke(cli, ["grid", "--help"])
        assert result.exit_code == 0
        assert "parse" in result.output
        print("✓ Grid subcommand group works")
        
        # Test solve subcommand
        result = runner.invoke(cli, ["solve", "--help"])
        assert result.exit_code == 0
        assert "run" in result.output
        print("✓ Solve subcommand group works")
        
        # Test post subcommand
        result = runner.invoke(cli, ["post", "--help"])
        assert result.exit_code == 0
        print("✓ Post subcommand group works")
        
        # Test config subcommand
        result = runner.invoke(cli, ["config", "--help"])
        assert result.exit_code == 0
        print("✓ Config subcommand group works")
        
        # Test utils subcommand
        result = runner.invoke(cli, ["utils", "--help"])
        assert result.exit_code == 0
        print("✓ Utils subcommand group works")
        
        # Test utils version
        result = runner.invoke(cli, ["utils", "version"])
        assert result.exit_code == 0
        print("✓ Utils version command works")
        
        # Test utils doctor
        result = runner.invoke(cli, ["utils", "doctor"])
        assert result.exit_code == 0
        print("✓ Utils doctor command works")
        
        print("\n✅ CLI module: ALL TESTS PASSED\n")
        return True
        
    except Exception as e:
        print(f"\n❌ CLI module test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_api_module():
    """Test Python API module."""
    print("=" * 60)
    print("Testing Python API Module...")
    print("=" * 60)
    
    try:
        from autoflowcfd import AutoFlowCFDAPI, create_api, get_version
        
        # Test version
        version = get_version()
        assert version == "0.1.0"
        print(f"✓ Version: {version}")
        
        # Test API creation
        api = AutoFlowCFDAPI()
        assert api.verbose is False
        print("✓ API initialization works")
        
        # Test verbose API
        api_verbose = AutoFlowCFDAPI(verbose=True)
        assert api_verbose.verbose is True
        print("✓ Verbose API initialization works")
        
        # Test convenience function
        api2 = create_api()
        assert isinstance(api2, AutoFlowCFDAPI)
        print("✓ create_api() convenience function works")
        
        # Test environment check
        env_info = api.check_environment()
        assert "python_version" in env_info
        assert "autoflowcfd_version" in env_info
        print("✓ Environment check works")
        
        # Test config creation
        steady_config = api.create_steady_config(backend="gpu", order=3)
        assert steady_config.backend.value == "gpu"
        assert steady_config.order == 3
        print("✓ Steady config creation works")
        
        transient_config = api.create_transient_config(mode="des", dt=1e-4)
        assert transient_config.dt == 1e-4
        print("✓ Transient config creation works")
        
        print("\n✅ API module: ALL TESTS PASSED\n")
        return True
        
    except Exception as e:
        print(f"\n❌ API module test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_config_module():
    """Test configuration module."""
    print("=" * 60)
    print("Testing Configuration Module...")
    print("=" * 60)
    
    try:
        from autoflowcfd.config import (
            SteadyConfig, TransientConfig,
            BackendType, TurbulenceModel, TimeIntegrationScheme,
            ConfigLoader
        )
        print("✓ Config classes imported successfully")
        
        # Test enum values
        assert BackendType.CPU.value == "cpu"
        assert BackendType.GPU.value == "gpu"
        print("✓ BackendType enum works")
        
        assert TurbulenceModel.SST_KW.value == "sst_kw"
        assert TurbulenceModel.DES.value == "des"
        print("✓ TurbulenceModel enum works")
        
        assert TimeIntegrationScheme.BACKWARD_EULER.value == "backward_euler"
        print("✓ TimeIntegrationScheme enum works")
        
        # Test config creation
        steady = SteadyConfig(backend=BackendType.GPU, order=3)
        assert steady.backend == BackendType.GPU
        assert steady.order == 3
        print("✓ SteadyConfig creation works")
        
        transient = TransientConfig(dt=1e-4, total_time=0.3)
        assert transient.dt == 1e-4
        assert transient.total_time == 0.3
        print("✓ TransientConfig creation works")
        
        print("\n✅ Config module: ALL TESTS PASSED\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Config module test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification tests."""
    print("\n" + "=" * 60)
    print("AutoFlowCFD Iteration 4 Verification")
    print("=" * 60 + "\n")
    
    results = []
    
    # Run tests
    results.append(("Boundary Module", test_boundary_module()))
    results.append(("CLI Module", test_cli_module()))
    results.append(("API Module", test_api_module()))
    results.append(("Config Module", test_config_module()))
    
    # Summary
    print("=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:<30} {status}")
    
    print("=" * 60)
    print(f"Total: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("\n🎉 All verification tests passed! Iteration 4 is complete.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
