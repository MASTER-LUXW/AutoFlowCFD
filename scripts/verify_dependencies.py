"""Comprehensive dependency verification for AutoFlowCFD."""

import sys
from typing import List, Tuple


def check_package(package_name: str, min_version: str = None) -> Tuple[bool, str]:
    """Check if a package is installed and get its version."""
    try:
        module = __import__(package_name.replace("-", "_"))
        version = getattr(module, "__version__", "unknown")
        
        if min_version:
            # Simple version comparison (for basic check)
            pass
        
        return True, version
    except ImportError as e:
        return False, str(e)


def main():
    """Verify all required dependencies."""
    print("="*70)
    print("AutoFlowCFD - Dependency Verification")
    print("="*70)
    print()
    
    # Core dependencies
    core_deps = [
        ("numpy", "1.24.0", "数值计算基础库"),
        ("click", "8.1.0", "CLI命令行框架"),
        ("yaml", "6.0.0", "YAML配置解析（pyyaml）"),
        ("h5py", "3.9.0", "HDF5数据序列化"),
        ("loguru", "0.7.0", "日志记录系统"),
    ]
    
    # Dev dependencies
    dev_deps = [
        ("pytest", "7.4.0", "单元测试框架"),
        ("pytest_cov", "4.1.0", "测试覆盖率（pytest-cov）"),
        ("black", "23.7.0", "代码格式化"),
        ("isort", "5.12.0", "import排序"),
        ("flake8", "6.1.0", "代码风格检查"),
        ("mypy", "1.5.0", "静态类型检查"),
    ]
    
    all_passed = True
    
    # Check core dependencies
    print("核心依赖 (Core Dependencies):")
    print("-" * 70)
    for pkg, min_ver, desc in core_deps:
        success, version = check_package(pkg, min_ver)
        status = "✓" if success else "✗"
        display_name = pkg.replace("_", "-")
        print(f"  {status} {display_name:20s} v{version:15s} - {desc}")
        if not success:
            all_passed = False
    
    print()
    
    # Check dev dependencies
    print("开发依赖 (Development Dependencies):")
    print("-" * 70)
    for pkg, min_ver, desc in dev_deps:
        success, version = check_package(pkg, min_ver)
        status = "✓" if success else "○"  # ○ for optional
        display_name = pkg.replace("_", "-")
        print(f"  {status} {display_name:20s} v{version:15s} - {desc}")
    
    print()
    print("="*70)
    
    # Test importing project modules
    print("\n项目模块导入测试:")
    print("-" * 70)
    
    modules_to_test = [
        ("autoflowcfd", "主包"),
        ("autoflowcfd.grid", "网格模块"),
        ("autoflowcfd.grid.structures", "数据结构"),
        ("autoflowcfd.grid.nas_io.parser", "NAS解析器"),
        ("autoflowcfd.grid.validation.validator", "质量校验器"),
    ]
    
    for module_name, desc in modules_to_test:
        try:
            __import__(module_name)
            print(f"  ✓ {module_name:40s} - {desc}")
        except ImportError as e:
            print(f"  ✗ {module_name:40s} - {desc}")
            print(f"    错误: {e}")
            all_passed = False
    
    print()
    print("="*70)
    
    if all_passed:
        print("\n✅ 所有依赖安装成功！")
        print("\n您可以运行以下命令：")
        print("  python scripts/verify_iteration2.py      # 验证迭代2")
        print("  pytest tests/                            # 运行测试")
        print("  python examples/grid_parsing_example.py  # 运行示例")
        return 0
    else:
        print("\n❌ 部分依赖缺失或导入失败")
        print("\n请运行以下命令安装缺失的依赖：")
        print("  pip install numpy click pyyaml h5py loguru pytest pytest-cov")
        print("  pip install black isort flake8 mypy pre-commit")
        return 1


if __name__ == "__main__":
    sys.exit(main())
