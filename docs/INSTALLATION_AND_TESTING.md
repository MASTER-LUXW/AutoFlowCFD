# 安装与测试指南

本文档提供AutoFlowCFD的详细安装步骤和测试方法。

## 系统要求

### 最低配置

- **操作系统**: Linux / Windows 10+ / macOS 10.15+
- **Python**: 3.10 或更高版本
- **内存**: 8GB RAM
- **磁盘空间**: 2GB（不含网格文件）

### 推荐配置（CPU计算）

- **CPU**: 8核心以上（Intel i7 / AMD Ryzen 7或更高）
- **内存**: 32GB RAM
- **磁盘**: SSD，10GB可用空间

### 推荐配置（GPU计算）

- **GPU**: NVIDIA GPU with CUDA支持（RTX 3090 / A100或更高）
- **显存**: ≥10GB（百万级网格），≥40GB（千万级网格）
- **CUDA版本**: 12.x
- **驱动**: 最新NVIDIA显卡驱动

---

## 安装步骤

### 方法1: 使用Poetry（推荐）

#### 1. 安装Poetry

```bash
# Linux/macOS
curl -sSL https://install.python-poetry.org | python3 -

# Windows (PowerShell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

验证安装：
```bash
poetry --version
# 应输出: Poetry (version 1.x.x)
```

#### 2. 克隆仓库

```bash
git clone https://github.com/AutoFlowCFD/AutoFlowCFD.git
cd AutoFlowCFD
```

#### 3. 安装依赖

```bash
# 安装核心依赖
poetry install

# 如果需要GPU支持
poetry install -E gpu

# 如果需要可视化支持
poetry install -E viz
```

#### 4. 激活虚拟环境

```bash
poetry shell
```

#### 5. 安装Pre-commit钩子

```bash
pre-commit install
```

#### 6. 验证安装

```bash
# 查看版本
autoflowcfd --version
# 应输出: AutoFlowCFD, version 0.1.0

# 查看帮助
autoflowcfd --help
```

### 方法2: 使用pip（传统方式）

```bash
# 克隆仓库
git clone https://github.com/AutoFlowCFD/AutoFlowCFD.git
cd AutoFlowCFD

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# 安装依赖
pip install -e .

# 安装开发依赖
pip install black isort flake8 mypy pytest pytest-cov
```

---

## 运行测试

### 单元测试

```bash
# 运行所有单元测试
poetry run pytest tests/unit -v

# 运行特定测试文件
poetry run pytest tests/unit/test_version.py -v

# 带覆盖率报告
poetry run pytest tests/unit --cov=autoflowcfd --cov-report=html

# 查看HTML覆盖率报告
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

### 代码质量检查

```bash
# Black代码格式化检查
poetry run black --check src/ tests/

# 自动格式化（如果有错误）
poetry run black src/ tests/

# isort import排序检查
poetry run isort --check-only src/ tests/

# 自动排序
poetry run isort src/ tests/

# flake8代码风格检查
poetry run flake8 src/ tests/

# mypy类型检查
poetry run mypy src/
```

### Pre-commit手动触发

```bash
# 对所有文件运行pre-commit
pre-commit run --all-files

# 仅对暂存的文件运行
pre-commit run
```

---

## 常见问题

### Q1: Poetry安装失败

**问题**: `poetry install` 时报错

**解决方案**:
```bash
# 清除缓存
poetry cache clear --all .

# 重新安装
poetry install

# 如果仍然失败，尝试更新Poetry
pip install --upgrade poetry
```

### Q2: 依赖冲突

**问题**: 某些依赖版本冲突

**解决方案**:
```bash
# 删除锁定文件
rm poetry.lock  # Linux/macOS
del poetry.lock  # Windows

# 重新生成
poetry install
```

### Q3: CUDA/GPU支持问题

**问题**: `import cupy` 失败

**解决方案**:
1. 确认已安装NVIDIA显卡驱动
2. 确认已安装CUDA Toolkit 12.x
3. 检查CUDA版本：
   ```bash
   nvcc --version
   ```
4. 重新安装CuPy：
   ```bash
   poetry install -E gpu
   ```

### Q4: Pre-commit钩子失败

**问题**: `pre-commit install` 后提交代码时检查失败

**解决方案**:
```bash
# 查看具体错误信息
pre-commit run --all-files

# 如果需要跳过检查（不推荐）
git commit -m "message" --no-verify

# 修复问题后重新提交
```

### Q5: 测试覆盖率不足

**问题**: 测试覆盖率低于80%

**解决方案**:
```bash
# 查看哪些行未覆盖
poetry run pytest --cov=autoflowcfd --cov-report=term-missing

# 为未覆盖的代码添加测试
# 编辑 tests/unit/test_*.py 文件
```

---

## 性能基准测试

### CPU后端基准

```bash
# 运行基准测试（需要实际网格文件）
autoflowcfd benchmark --grid test.nas --backend cpu --iterations 100
```

### GPU后端基准

```bash
# GPU基准测试
autoflowcfd benchmark --grid test.nas --backend gpu --iterations 100
```

**预期性能**（参考值）：

|网格规模|CPU时间/步|GPU时间/步|加速比|
|---|---|---|---|
|100K cells|~0.5s|~0.1s|5x|
|1M cells|~2.5s|~0.6s|4x|
|10M cells|~25s|~5s|5x|

*测试环境: Intel Xeon Gold 6248 / NVIDIA A100 40GB*

---

## 下一步

完成安装和测试后，你可以：

1. 📖 阅读 [快速开始指南](QUICKSTART.md)
2. 🔧 查看 [配置示例](examples/config_example.yaml)
3. 💻 开始 [Iteration 2开发](docs/ITERATION_1_COMPLETION_REPORT.md)
4. 🤝 参与 [社区贡献](CONTRIBUTING.md)

---

## 获取帮助

- **文档**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/AutoFlowCFD/AutoFlowCFD/issues)
- **讨论**: [GitHub Discussions](https://github.com/AutoFlowCFD/AutoFlowCFD/discussions)
- **邮件**: contact@autoflowcfd.org

---

**祝你使用愉快！** 🚀
