"""Fix and verify loguru installation."""

import subprocess
import sys


def install_loguru():
    """Install loguru package."""
    print("Installing loguru...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "loguru>=0.7.0"
        ])
        print("✓ loguru installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install loguru: {e}")
        
        # Try with mirror
        print("\nTrying with Tsinghua mirror...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "-i", "https://pypi.tuna.tsinghua.edu.cn/simple",
                "loguru>=0.7.0"
            ])
            print("✓ loguru installed successfully with mirror")
            return True
        except subprocess.CalledProcessError:
            print("✗ Still failed. Please check your network connection.")
            return False


def verify_loguru():
    """Verify loguru can be imported and used."""
    print("\nVerifying loguru...")
    try:
        from loguru import logger
        logger.info("Test message")
        print("✓ loguru works correctly")
        return True
    except ImportError as e:
        print(f"✗ Cannot import loguru: {e}")
        return False
    except Exception as e:
        print(f"✗ loguru error: {e}")
        return False


def test_project_modules():
    """Test importing project modules that use loguru."""
    print("\nTesting project modules...")
    
    modules = [
        "autoflowcfd.grid.structures",
        "autoflowcfd.grid.parser",
        "autoflowcfd.grid.validator",
    ]
    
    all_ok = True
    for module in modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module}: {e}")
            all_ok = False
    
    return all_ok


def main():
    """Main function."""
    print("="*60)
    print("AutoFlowCFD - Loguru Fix Script")
    print("="*60)
    print()
    
    # Step 1: Verify current state
    if verify_loguru():
        print("\n✅ loguru is already working!")
    else:
        print("\n⚠ loguru needs to be installed/fixed")
        if not install_loguru():
            print("\n❌ Installation failed")
            return 1
    
    # Step 2: Test project imports
    print()
    if test_project_modules():
        print("\n✅ All modules working correctly!")
        return 0
    else:
        print("\n⚠ Some modules have issues (may be path-related)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
