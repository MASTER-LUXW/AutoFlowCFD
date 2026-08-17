# AutoFlowCFD 依赖安装完整指南

## 🚀 快速开始（推荐）

### 方法1：一键安装脚本（Windows）

双击运行或在命令行执行：
```bash
install_all_deps.bat
```

该脚本会自动：
1. ✅ 安装所有核心依赖
2. ✅ 安装开发工具
3. ✅ 验证安装结果

---

## 📦 依赖清单

### 核心依赖（必需）

这些是运行AutoFlowCFD所必需的包：

| 包名 | 最低版本 | 用途 | 安装命令 |
|------|---------|------|---------|
| numpy | 1.26.0 | 数值计算基础库 | `pip install numpy>=1.26.0` |
| click | 8.1.0 | CLI命令行框架 | `pip install click>=8.1.0` |
| pyyaml | 6.0.0 | YAML配置文件解析 | `pip install pyyaml>=6.0.0` |
| h5py | 3.9.0 | HDF5数据序列化 | `pip install h5py>=3.9.0` |
| loguru | 0.7.0 | 日志记录系统 | `pip install loguru>=0.7.0` |

**一键安装命令：**
```bash
pip install numpy>=1.26.0 click>=8.1.0 pyyaml>=6.0.0 h5py>=3.9.0 loguru>=0.7.0
```

### 开发依赖（测试和代码质量）

这些是开发和测试所需的工具：

| 包名 | 最低版本 | 用途 | 安装命令 |
|------|---------|------|---------|
| pytest | 7.4.0 | 单元测试框架 | `pip install pytest>=7.4.0` |
| pytest-cov | 4.1.0 | 测试覆盖率报告 | `pip install pytest-cov>=4.1.0` |
| black | 23.7.0 | 代码格式化工具 | `pip install black>=23.7.0` |
| isort | 5.12.0 | import排序工具 | `pip install isort>=5.12.0` |
| flake8 | 6.1.0 | 代码风格检查 | `pip install flake8>=6.1.0` |
| mypy | 1.5.0 | 静态类型检查 | `pip install mypy>=1.5.0` |

**一键安装命令：**
```bash
pip install black>=23.7.0 isort>=5.12.0 flake8>=6.1.0 mypy>=1.5.0 pytest>=7.4.0 pytest-cov>=4.1.0
```

### 可选依赖

#### GPU加速支持
```bash
# CUDA 12.x
pip install cupy-cuda12x>=12.2.0

# CUDA 11.x
pip install cupy-cuda11x>=11.0.0
```

#### 可视化支持
```bash
pip install pyvista>=0.42.0
```

---

## ✅ 验证安装

### 方法1：快速验证

```bash
python -c "import numpy, click, yaml, h5py, loguru; print('✓ 所有核心依赖安装成功')"
```

### 方法2：完整验证脚本

```bash
python scripts\verify_dependencies.py
```

预期输出：
```
======================================================================
AutoFlowCFD - Dependency Verification
======================================================================

核心依赖 (Core Dependencies):
----------------------------------------------------------------------
  ✓ numpy                v1.24.x          - 数值计算基础库
  ✓ click                v8.1.x           - CLI命令行框架
  ✓ yaml                 v6.0.x           - YAML配置解析
  ✓ h5py                 v3.9.x           - HDF5数据序列化
  ✓ loguru               v0.7.x           - 日志记录系统

✅ 所有依赖安装成功！
```

### 方法3：运行测试

```bash
# 运行单元测试
poetry run pytest tests/unit/ -v

# 运行完整验证
poetry run pytest tests/ -v
```

---

## 🔧 常见问题解决

### 问题1：pip命令找不到

**解决方案：**
```bash
# Windows
python -m pip install numpy

# 或者添加Python Scripts到PATH
# C:\Users\YourName\AppData\Local\Programs\Python\Python310\Scripts
```

### 问题2：h5py安装失败

**原因：** 需要C编译器或HDF5库

**解决方案：**
```bash
# Windows - 使用预编译wheel
pip install h5py --only-binary=h5py

# 如果仍然失败，下载预编译的whl文件
# 从 https://www.lfd.uci.edu/~gohlke/pythonlibs/ 下载
```

### 问题3：权限错误

**错误信息：** `PermissionError: [WinError 5]`

**解决方案：**
```bash
# 使用--user标志安装到用户目录
pip install --user numpy click pyyaml h5py loguru

# 或者以管理员身份运行命令提示符
```

### 问题4：版本冲突

**错误信息：** `ERROR: Cannot install ... because these package versions have conflicting dependencies`

**解决方案：**
```bash
# 创建虚拟环境
python -m venv venv
venv\Scripts\activate

# 然后在虚拟环境中安装
pip install numpy>=1.24.0 click>=8.1.0 pyyaml>=6.0.0 h5py>=3.9.0 loguru>=0.7.0
```

### 问题5：网络连接慢

**解决方案：使用国内镜像源**
```bash
# 清华大学镜像
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy click pyyaml h5py loguru

# 阿里云镜像
pip install -i https://mirrors.aliyun.com/pypi/simple/ numpy click pyyaml h5py loguru
```

---

## 🎯 使用Poetry（项目推荐方式）

如果您想使用Poetry进行依赖管理：

### 1. 安装Poetry
```bash
pip install poetry
```

### 2. 安装项目依赖
```bash
cd d:\myWorkspace\AutoFlowCFD
poetry install
```

### 3. 激活虚拟环境
```bash
poetry shell
```

### 4. 运行项目
```bash
# 在虚拟环境中
poetry run autoflowcfd --version
```

---

## 📝 下一步

依赖安装完成后，您可以：

1. **运行测试套件**
   ```bash
   poetry run pytest tests/ -v
   ```

2. **快速入门**
   ```bash
   poetry run autoflowcfd --help
   ```

3. **查看文档**
   ```bash
   # 查看快速入门指南
   type QUICKSTART.md
   ```

---

## 🆘 获取帮助

如果遇到问题：

1. 检查Python版本：`python --version`（需要≥3.10）
2. 检查pip版本：`pip --version`
3. 查看完整日志：运行安装脚本时注意错误信息
4. 查阅[INSTALL.md](INSTALL.md)获取更多细节

---

**最后更新**: 2026-08-17  
**维护者**: AutoFlowCFD Team
