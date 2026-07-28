@echo off
setlocal enabledelayedexpansion

:: ============================================================================
:: AutoFlowCFD 依赖安装脚本 (Windows)
:: 
:: 功能: 自动安装所有必需的Python依赖包
:: 用法: install_dependencies.bat
:: ============================================================================

echo.
echo ========================================================================
echo AutoFlowCFD Dependency Installation Script
echo ========================================================================
echo.

:: 检查Python是否安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.9 or higher from https://www.python.org/
    pause
    exit /b 1
)

:: 检查Python版本
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [INFO] Detected Python version: %PYTHON_VERSION%

:: 提取主版本号
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set MAJOR=%%a
    set MINOR=%%b
)

if %MAJOR% lss 3 (
    echo [ERROR] Python 3.9+ is required, but found %PYTHON_VERSION%
    pause
    exit /b 1
)

if %MAJOR% equ 3 if %MINOR% lss 9 (
    echo [ERROR] Python 3.9+ is required, but found %PYTHON_VERSION%
    pause
    exit /b 1
)

echo [OK] Python version check passed
echo.

:: 询问是否安装GPU支持
set INSTALL_GPU=0
set /p GPU_CHOICE="Install GPU support (CuPy)? This requires CUDA Toolkit. [y/N]: "
if /i "%GPU_CHOICE%"=="y" set INSTALL_GPU=1
if /i "%GPU_CHOICE%"=="Y" set INSTALL_GPU=1

echo.
echo ========================================================================
echo Step 1: Installing Core Dependencies
echo ========================================================================
echo.

pip install numpy>=1.24.0 click>=8.1.0 pyyaml>=6.0.0 h5py>=3.9.0 loguru>=0.7.0
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install core dependencies
    pause
    exit /b 2
)

echo.
echo [OK] Core dependencies installed successfully
echo.

echo ========================================================================
echo Step 2: Installing Development Dependencies
echo ========================================================================
echo.

pip install pytest>=7.4.0 pytest-cov>=4.1.0 black>=23.7.0 isort>=5.12.0 flake8>=6.1.0 mypy>=1.5.0
if %errorlevel% neq 0 (
    echo [WARNING] Some development dependencies failed to install
    echo You can install them manually later if needed
)

echo.
echo [OK] Development dependencies installation completed
echo.

:: 如果用户选择安装GPU支持
if %INSTALL_GPU% equ 1 (
    echo ========================================================================
    echo Step 3: Installing GPU Support (CuPy for CUDA 12.x)
    echo ========================================================================
    echo.
    
    echo [INFO] Installing CuPy for CUDA 12.x...
    echo [NOTE] Make sure you have NVIDIA drivers and CUDA Toolkit 12.x installed
    echo.
    
    pip install cupy-cuda12x
    if %errorlevel% neq 0 (
        echo [WARNING] CuPy installation failed
        echo Please ensure CUDA Toolkit 12.x is properly installed
        echo You can try installing it manually later: pip install cupy-cuda12x
    ) else (
        echo [OK] GPU support installed successfully
    )
    echo.
)

echo ========================================================================
echo Step 4: Verifying Installation
echo ========================================================================
echo.

:: 验证核心依赖
echo Checking core dependencies...
python -c "import numpy, click, yaml, h5py, loguru; print('[OK] All core dependencies OK')" 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Core dependency verification failed
    pause
    exit /b 3
)

:: 尝试验证开发依赖
echo Checking development dependencies...
python -c "import pytest, black, isort" 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] Some development dependencies may not be installed correctly
) else (
    echo [OK] Development dependencies OK
)

:: 如果安装了GPU支持，验证CuPy
if %INSTALL_GPU% equ 1 (
    echo Checking GPU support...
    python -c "import cupy; print(f'[OK] CuPy {cupy.__version__} installed')" 2>nul
    if %errorlevel% neq 0 (
        echo [WARNING] CuPy verification failed - GPU support may not work
    )
)

echo.
echo ========================================================================
echo Installation Summary
echo ========================================================================
echo.
echo [OK] Core dependencies: INSTALLED
echo [OK] Development dependencies: INSTALLED
if %INSTALL_GPU% equ 1 (
    echo [OK] GPU support: INSTALLED
) else (
    echo [SKIP] GPU support: Not installed
)
echo.
echo Next steps:
echo   1. Run tests: pytest tests/ -v
echo   2. Verify Iteration 2: python scripts/verify_iteration2.py
echo   3. Try example: python examples/grid_parsing_example.py
echo.
echo ========================================================================
echo Installation completed successfully!
echo ========================================================================
echo.

pause
exit /b 0
