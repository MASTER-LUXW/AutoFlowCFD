#!/usr/bin/env python3
"""
AutoFlowCFD Dependency Checker and Installer

This script checks for required dependencies and installs them if missing.
"""

import sys
import subprocess
from typing import List, Tuple


# Core dependencies required for Iteration 2
CORE_DEPENDENCIES = [
    ("numpy", "1.24.0"),
    ("click", "8.1.0"),
    ("pyyaml", "6.0.0"),
    ("h5py", "3.9.0"),
    ("loguru", "0.7.0"),
]

# Development dependencies
DEV_DEPENDENCIES = [
    ("pytest", "7.4.0"),
    ("pytest-cov", "4.1.0"),
    ("black", "23.7.0"),
    ("isort", "5.12.0"),
    ("flake8", "6.1.0"),
    ("mypy", "1.5.0"),
]

# Optional dependencies
OPTIONAL_DEPENDENCIES = [
    ("cupy", "12.2.0", "cupy-cuda12x"),  # GPU support
    ("pyvista", "0.42.0", None),  # Visualization
]


def check_package(package_name: str) -> bool:
    """Check if a package is installed."""
    try:
        __import__(package_name.replace("-", "_"))
        return True
    except ImportError:
        return False


def get_installed_version(package_name: str) -> str:
    """Get the installed version of a package."""
    try:
        import importlib
        module = importlib.import_module(package_name.replace("-", "_"))
        return getattr(module, "__version__", "unknown")
    except (ImportError, AttributeError):
        return "not installed"


def install_package(package_name: str, min_version: str) -> bool:
    """Install a package with minimum version requirement."""
    try:
        print(f"  Installing {package_name}>={min_version}...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", f"{package_name}>={min_version}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed to install {package_name}: {e}")
        return False


def check_dependencies(dependencies: List[Tuple[str, str]], category: str) -> Tuple[List[str], List[str]]:
    """Check a list of dependencies and return missing ones."""
    print(f"\n{'='*60}")
    print(f"Checking {category} Dependencies")
    print(f"{'='*60}")
    
    missing = []
    installed = []
    
    for package, min_version in dependencies:
        is_installed = check_package(package)
        version = get_installed_version(package)
        
        if is_installed:
            print(f"  ✓ {package:20s} {version:15s} (required: >={min_version})")
            installed.append(package)
        else:
            print(f"  ✗ {package:20s} NOT INSTALLED   (required: >={min_version})")
            missing.append((package, min_version))
    
    return missing, installed


def install_missing(missing: List[Tuple[str, str]]) -> bool:
    """Install all missing dependencies."""
    if not missing:
        print("\n✓ All dependencies are already installed!")
        return True
    
    print(f"\n{'='*60}")
    print(f"Installing {len(missing)} missing dependencies...")
    print(f"{'='*60}\n")
    
    success_count = 0
    for package, min_version in missing:
        if install_package(package, min_version):
            success_count += 1
    
    print(f"\nInstalled {success_count}/{len(missing)} packages")
    return success_count == len(missing)


def main():
    """Main function to check and install dependencies."""
    print("="*60)
    print("AutoFlowCFD - Dependency Checker")
    print("="*60)
    
    # Check core dependencies
    core_missing, core_installed = check_dependencies(CORE_DEPENDENCIES, "Core")
    
    # Check dev dependencies
    dev_missing, dev_installed = check_dependencies(DEV_DEPENDENCIES, "Development")
    
    # Check optional dependencies (don't fail if missing)
    print(f"\n{'='*60}")
    print("Checking Optional Dependencies")
    print(f"{'='*60}")
    for package, min_version, pip_name in OPTIONAL_DEPENDENCIES:
        is_installed = check_package(package)
        version = get_installed_version(package)
        status = "✓" if is_installed else "○"
        print(f"  {status} {package:20s} {version:15s} (optional)")
    
    # Install missing dependencies
    all_missing = core_missing + dev_missing
    
    if all_missing:
        print(f"\n{'='*60}")
        print(f"Found {len(all_missing)} missing dependencies")
        print(f"{'='*60}")
        
        response = input("\nDo you want to install missing dependencies? (y/n): ")
        if response.lower() == 'y':
            success = install_missing(all_missing)
            if success:
                print("\n✓ All dependencies installed successfully!")
                print("\nYou can now run:")
                print("  pytest tests/                     # Run tests")
                print("  python examples/grid_parsing_example.py  # Run example")
                print("  python scripts/verify_iteration2.py      # Verify Iteration 2")
            else:
                print("\n✗ Some installations failed. Please check the errors above.")
                print("You may need to install some packages manually.")
                return 1
        else:
            print("\nInstallation cancelled.")
            print("\nTo install manually, run:")
            for package, min_version in all_missing:
                print(f"  pip install {package}>={min_version}")
            return 1
    else:
        print("\n✓ All required dependencies are installed!")
        print("\nYou can now run:")
        print("  pytest tests/                     # Run tests")
        print("  python examples/grid_parsing_example.py  # Run example")
        print("  python scripts/verify_iteration2.py      # Verify Iteration 2")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
