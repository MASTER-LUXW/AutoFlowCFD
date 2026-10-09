"""可靠执行命令的测试工具。

本模块提供运行测试与命令的工具函数，可靠地捕获输出，避开终端输出的问题。

示例:
    >>> from tests.test_utils import run_pytest, run_command
    >>> output = run_pytest("tests/unit/test_fr_residual_inviscid.py")
    >>> print(output)
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional


def run_pytest(test_path: str, verbose: bool = True) -> str:
    """运行 pytest 并可靠地返回输出。

    Args:
        test_path: 测试文件或目录的路径
        verbose: 是否输出详细信息

    Returns:
        str: 测试输出

    示例:
        >>> output = run_pytest("tests/unit/test_fr_residual_inviscid.py")
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
    """运行任意命令并可靠地捕获输出。

    Args:
        command: 要执行的命令
        timeout: 超时（秒）

    Returns:
        dict: {'success': bool, 'stdout': str, 'stderr': str, 'returncode': int}

    示例:
        >>> result = run_command("python scripts/verify_iteration4.py")
        >>> if result['success']:
        ...     print(result['stdout'])
    """
    try:
        # 复杂命令用 shell=True 执行
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
    """验证模块可以无错误地导入。

    Args:
        module_name: 要导入的模块名

    Returns:
        dict: {'success': bool, 'error': str or None}

    示例:
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
    """检查 Python 文件的语法错误。

    Args:
        file_path: Python 文件路径

    Returns:
        dict: {'valid': bool, 'errors': list}

    示例:
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
    """运行全部单元测试并返回摘要。

    Returns:
        str: 测试摘要输出

    示例:
        >>> summary = run_all_unit_tests()
        >>> print(summary)
    """
    return run_pytest("tests/unit/", verbose=True)


def run_integration_tests() -> str:
    """运行全部集成测试并返回摘要。

    Returns:
        str: 测试摘要输出
    """
    return run_pytest("tests/integration/", verbose=True)


if __name__ == "__main__":
    # Quick self-test
    print("Testing test utilities...")
    
    # Test 1: Module import verification
    result = verify_module_import("autoflowcfd")
    print(f"✓ Module import: {'PASS' if result['success'] else 'FAIL'}")
    
    # Test 2: Run a simple test
    output = run_pytest("tests/unit/test_fr_residual_inviscid.py::TestAusmUpConsistency::test_flux_consistency_random_states")
    if "passed" in output.lower() or "PASSED" in output:
        print("✓ Test execution: PASS")
    else:
        print("✗ Test execution: Check output below")
        print(output)
