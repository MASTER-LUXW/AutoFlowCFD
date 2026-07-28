# AutoFlowCFD 依赖安装指南

## 快速安装

### 方法1: 使用pip直接安装（推荐）

```bash
# 安装核心依赖（Iteration 2必需）
pip install numpy>=1.24.0 click>=8.1.0 pyyaml>=6.0.0 h5py>=3.9.0 loguru>=0.7.0

# 安装开发依赖（测试和代码检查）
pip install pytest>=7.4.0 pytest-cov>=4.1.0 black>=23.7.0 isort>=5.12.0 flake8>=6.1.0 mypy>=1.5.0
```

### 方法2: 使用自动化脚本

**Windows:**
```bash
install_dependencies.bat
```

**Linux/Mac:**
```bash
python scripts/check_dependencies.py
```

### 方法3: 使用Poetry（项目推荐）

```bash
# 安装Poetry（如果还没有）
pip install poetry

# 安装所有依赖
poetry install

# 安装包含可选依赖
poetry install --extras "viz gpu"
```

---

## 依赖清单

### 核心依赖（必需）

|包名|最低版本|用途|
|---|---|---|
|numpy|1.24.0|数值计算基础库|
|click|8.1.0|CLI命令行框架|
|pyyaml|6.0.0|YAML配置文件解析|
|h5py|3.9.0|HDF5数据序列化|
|loguru|0.7.0|日志记录系统|

### 开发依赖（测试/检查）

|包名|最低版本|用途|
|---|---|---|
|pytest|7.4.0|单元测试框架|
|pytest-cov|4.1.0|测试覆盖率报告|
|black|23.7.0|代码格式化工具|
|isort|5.12.0|import排序工具|
|flake8|6.1.0|代码风格检查|
|mypy|1.5.0|静态类型检查|

### 可选依赖

|包名|最低版本|用途|安装命令|
|---|---|---|---|
|cupy-cuda12x|12.2.0|GPU加速（NVIDIA）|`pip install cupy-cuda12x`|
|pyvista|0.42.0|3D可视化|`pip install pyvista`|

---

## 验证安装

### 1. 快速验证核心依赖

```bash
python -c "import numpy, click, yaml, h5py, loguru; print('✓ All core dependencies OK')"
```

### 2. 运行完整验证脚本

```bash
python scripts/verify_iteration2.py
```

预期输出：
```
======================================================================
AutoFlowCFD Iteration 2 Verification
======================================================================
Testing file structure...
✅ All expected files present
Testing imports...
✅ All imports successful
...
✅ ALL TESTS PASSED - Iteration 2 is ready!
```

### 3. 运行单元测试

```bash
pytest tests/unit/test_grid_structures.py -v
pytest tests/unit/test_nas_parser.py -v
pytest tests/unit/test_grid_validator.py -v
```

### 4. 运行示例

```bash
python examples/grid_parsing_example.py
```

---

## 常见问题

### Q1: 提示"No module named 'numpy'"

**解决方案**:
```bash
pip install numpy>=1.24.0
```

### Q2: h5py安装失败

**原因**: 可能需要编译环境  
**解决方案**:
```bash
# Windows
pip install h5py

# Linux (需要先安装HDF5库)
sudo apt-get install libhdf5-dev
pip install h5py
```

### Q3: mypy类型检查报错

**原因**: 某些包缺少类型存根  
**解决方案**:
```bash
pip install types-PyYAML
```

### Q4: CuPy安装失败

**原因**: 需要CUDA Toolkit  
**解决方案**:
1. 安装NVIDIA驱动和CUDA Toolkit 12.x
2. 确认CUDA路径在环境变量中
3. 运行: `pip install cupy-cuda12x`

### Q5: Poetry安装慢或失败

**解决方案**:
```bash
# 使用国内镜像源
poetry config repositories.tuna https://pypi.tuna.tsinghua.edu.cn/simple
poetry install
```

---

## GPU支持（可选）

如果需要GPU加速功能，需要额外安装：

### 1. 安装NVIDIA驱动

- 下载: https://www.nvidia.com/drivers
- 要求: Driver ≥ 470.x

### 2. 安装CUDA Toolkit

- 下载: https://developer.nvidia.com/cuda-downloads
- 推荐版本: CUDA 12.x

### 3. 安装CuPy

```bash
# CUDA 12.x
pip install cupy-cuda12x

# CUDA 11.x
pip install cupy-cuda11x
```

### 4. 验证GPU支持

```python
import cupy as cp
print(f"CuPy version: {cp.__version__}")
print(f"CUDA available: {cp.cuda.runtime.getDeviceCount() > 0}")
```

---

## 下一步

依赖安装完成后，您可以：

1. **运行测试**: `pytest tests/ -v`
2. **查看示例**: `python examples/grid_parsing_example.py`
3. **验证迭代2**: `python scripts/verify_iteration2.py`
4. **开始开发**: 进入Iteration 3 - FR求解器开发

---

**最后更新**: 2026-07-23  
**维护者**: AutoFlowCFD Team
