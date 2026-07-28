"""Quick environment verification script.

This script checks if all dependencies are correctly installed.

Usage:
    poetry run python scripts/quick_check.py
"""

import sys
from pathlib import Path

# Add src to path so we can import autoflowcfd during testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def check_python_version():
    """Check Python version"""
    print(f"Python version: {sys.version}")
    if sys.version_info < (3, 10):
        print("❌ Python version must be >= 3.10")
        return False
    print("✅ Python version OK")
    return True


def check_core_dependencies():
    """Check core dependencies"""
    print("\nChecking core dependencies...")
    
    deps = {
        'numpy': 'NumPy',
        'click': 'Click',
        'yaml': 'PyYAML',
        'h5py': 'H5Py',
        'loguru': 'Loguru',
        'numba': 'Numba',
        'llvmlite': 'LLVMLite',
    }
    
    all_ok = True
    for module, name in deps.items():
        try:
            mod = __import__(module)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✅ {name}: {version}")
        except ImportError as e:
            print(f"❌ {name}: NOT INSTALLED ({e})")
            all_ok = False
    
    return all_ok


def check_autoflowcfd_module():
    """Check autoflowcfd module"""
    print("\nChecking AutoFlowCFD module...")
    try:
        import autoflowcfd
        print(f"✅ AutoFlowCFD imported successfully")
        if hasattr(autoflowcfd, '__version__'):
            print(f"   Version: {autoflowcfd.__version__}")
        return True
    except ImportError as e:
        print(f"❌ AutoFlowCFD import failed: {e}")
        return False


def main():
    """Main verification function"""
    print("=" * 60)
    print("AutoFlowCFD Environment Verification")
    print("=" * 60)
    
    results = []
    
    # Check Python version
    results.append(check_python_version())
    
    # Check dependencies
    results.append(check_core_dependencies())
    
    # Check module
    results.append(check_autoflowcfd_module())
    
    # Summary
    print("\n" + "=" * 60)
    if all(results):
        print("✅ All checks passed! Environment is ready.")
        print("=" * 60)
        return 0
    else:
        print("❌ Some checks failed. Please review the errors above.")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
