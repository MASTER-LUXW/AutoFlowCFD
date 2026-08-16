"""Test utilities for reliable command execution.

This module provides utility functions for running tests and commands
with reliable output capture, avoiding terminal output issues.

Example:
    >>> from tests.test_utils import run_pytest, run_command
    >>> output = run_pytest("tests/unit/test_boundary.py")
    >>> print(output)
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional


def run_pytest(test_path: str, verbose: bool = True) -> str:
    """Run pytest and return output reliably.
    
    Args:
        test_path: Path to test file or directory
        verbose: Enable verbose output
        
    Returns:
        str: Test output
        
    Example:
        >>> output = run_pytest("tests/unit/test_boundary.py")
        >>> if "passed" in output:
        ...     print("Tests passed!")
    """
    cmd = [
        sys.executable, "-m", "pytest",
        test_path,
        "-v" if verbose else "-q",
        "--tb=short",
        "--color=no",  # Disable color for cleaner output
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
            timeout=300  # 5 minute timeout
        )
        
        output = result.stdout + result.stderr
        return output
        
    except subprocess.TimeoutExpired:
        return "ERROR: Test execution timed out (5 minutes)"
    except Exception as e:
        return f"ERROR: Failed to run tests: {e}"


def run_command(command: str, timeout: int = 60) -> dict:
    """Run arbitrary command and capture output reliably.
    
    Args:
        command: Command to execute
        timeout: Timeout in seconds
        
    Returns:
        dict: {'success': bool, 'stdout': str, 'stderr': str, 'returncode': int}
        
    Example:
        >>> result = run_command("python scripts/verify_iteration4.py")
        >>> if result['success']:
        ...     print(result['stdout'])
    """
    try:
        # Use shell=True for complex commands
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
            timeout=timeout,
            encoding='utf-8',
            errors='replace'  # Handle encoding errors gracefully
        )
        
        return {
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode,
        }
        
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'stdout': '',
            'stderr': f'Command timed out after {timeout} seconds',
            'returncode': -1,
        }
    except Exception as e:
        return {
            'success': False,
            'stdout': '',
            'stderr': str(e),
            'returncode': -1,
        }


def verify_module_import(module_name: str) -> dict:
    """Verify that a module can be imported without errors.
    
    Args:
        module_name: Module name to import
        
    Returns:
        dict: {'success': bool, 'error': str or None}
        
    Example:
        >>> result = verify_module_import("autoflowcfd.boundary")
        >>> if result['success']:
        ...     print("Module imports successfully")
    """
    try:
        __import__(module_name)
        return {
            'success': True,
            'error': None,
        }
    except ImportError as e:
        return {
            'success': False,
            'error': f"ImportError: {e}",
        }
    except Exception as e:
        return {
            'success': False,
            'error': f"{type(e).__name__}: {e}",
        }


def check_code_syntax(file_path: str) -> dict:
    """Check Python file for syntax errors.
    
    Args:
        file_path: Path to Python file
        
    Returns:
        dict: {'valid': bool, 'errors': list}
        
    Example:
        >>> result = check_code_syntax("src/autoflowcfd/api.py")
        >>> if result['valid']:
        ...     print("No syntax errors")
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        compile(code, file_path, 'exec')
        return {
            'valid': True,
            'errors': [],
        }
        
    except SyntaxError as e:
        return {
            'valid': False,
            'errors': [str(e)],
        }
    except Exception as e:
        return {
            'valid': False,
            'errors': [f"Failed to read file: {e}"],
        }


def run_all_unit_tests() -> str:
    """Run all unit tests and return summary.
    
    Returns:
        str: Test summary output
        
    Example:
        >>> summary = run_all_unit_tests()
        >>> print(summary)
    """
    return run_pytest("tests/unit/", verbose=True)


def run_integration_tests() -> str:
    """Run all integration tests and return summary.
    
    Returns:
        str: Test summary output
    """
    return run_pytest("tests/integration/", verbose=True)


if __name__ == "__main__":
    # Quick self-test
    print("Testing test utilities...")
    
    # Test 1: Module import verification
    result = verify_module_import("autoflowcfd")
    print(f"✓ Module import: {'PASS' if result['success'] else 'FAIL'}")
    
    # Test 2: Run a simple test
    output = run_pytest("tests/unit/test_boundary.py::TestInletBC::test_creation_with_defaults")
    if "passed" in output.lower() or "PASSED" in output:
        print("✓ Test execution: PASS")
    else:
        print("✗ Test execution: Check output below")
        print(output)
