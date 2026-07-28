# AutoFlowCFD 快速开始指南

本指南帮助你在5分钟内完成AutoFlowCFD的安装和首次运行。

## 📋 目录

- [前置要求](#前置要求)
- [安装步骤](#安装步骤)
- [首次运行](#首次运行)
- [使用方式](#使用方式)
- [输出说明](#输出说明)
- [常见问题](#常见问题)
- [下一步](#下一步)

---

## 前置要求

### 必需软件

- **Python**: 3.10 或更高版本
- **操作系统**: Linux / Windows / macOS
- **Poetry**: Python依赖管理工具

### 可选软件（GPU加速）

- **NVIDIA GPU**: 支持CUDA的显卡
- **CUDA Toolkit**: 12.x 版本
- **NVIDIA驱动**: 最新版本

### 网格准备

- **ANSA前处理器**: 用于生成 `.nas` 格式网格文件（v22/v23/v24版本）

---

## 安装步骤

### 1. 克隆仓库

```bash
git clone https://github.com/AutoFlowCFD/AutoFlowCFD.git
cd AutoFlowCFD
```

### 2. 安装Poetry（如果尚未安装）

```bash
pip install poetry
```

### 3. 安装项目依赖

```bash
poetry install
```

这将安装所有核心依赖和开发工具。如需GPU支持，请额外安装：

```bash
poetry install -E gpu
```

### 4. 激活虚拟环境

**Poetry 2.0+ 推荐使用以下两种方式之一：**

#### 方式一：使用 `poetry run`（推荐）

无需激活环境，直接运行命令：

```bash
# 验证安装
poetry run autoflowcfd --version

# 运行仿真
poetry run autoflowcfd solve run car_model.nas --mode steady
```

#### 方式二：使用 `poetry env activate`

```bash
# 激活虚拟环境（会输出激活命令）
poetry env activate

# 执行输出的命令，例如：
# source .venv/Scripts/activate  (Linux/macOS)
# .venv\Scripts\activate.bat     (Windows)
```

### 5. 验证安装

```bash
# 查看版本信息
poetry run autoflowcfd --version

# 应输出: AutoFlowCFD, version 0.1.0

# 检查可用后端
python -c "from autoflowcfd.core import get_available_backends; print(get_available_backends())"
```

---

## 首次运行

### 准备网格文件

你需要一个ANSA生成的 `.nas` 格式网格文件。如果没有，可以使用项目自带的示例：

```bash
# 使用Ahmed Body示例网格
ls examples/ahmed_body/
```

### 方法1：使用CLI命令行接口（推荐）

#### 稳态RANS仿真

```bash
# 基本命令
poetry run autoflowcfd solve run examples/ahmed_body_demo.nas \
    --mode steady \
    --turbulence sst_kw \
    --backend cpu

# 指定输出目录和迭代次数
poetry run autoflowcfd solve run sedan.nas \
    --output ./results/steady \
    --max-iter 5000 \
    --order 2

# GPU加速（需要CUDA）
poetry run autoflowcfd solve run sedan.nas \
    --backend gpu \
    --gpu-device 0 \
    --order 3 \
    --max-iter 5000
```

#### 瞬态DES/DDES仿真

```bash
# DES瞬态仿真
poetry run autoflowcfd solve run car_model.nas \
    --mode transient \
    --turbulence des \
    --dt 1e-4 \
    --total-time 0.1 \
    --backend gpu

# DDES仿真
poetry run autoflowcfd solve run car_model.nas \
    --mode transient \
    --turbulence ddes \
    --dt 5e-5 \
    --total-time 0.05
```

#### 使用配置文件

创建 `config.yaml`:

```yaml
solver:
  backend: cpu
  order: 2
  turbulence: sst_kw
  max_iter: 5000
  cfl_init: 0.1
  cfl_max: 5.0
  convergence_tol: 1.0e-6
  checkpoint_interval: 100
  output_dir: ./results
  
boundary_conditions:
  inlet:
    type: INLET
    velocity_x: 30.0
    pressure: 101325.0
    
  outlet:
    type: OUTLET
    pressure: 101325.0
    
  body:
    type: WALL
    wall_function: enhanced
```

运行：

```bash
poetry run autoflowcfd solve run sedan.nas -c config.yaml
```

#### 后处理结果

```bash
# 导出所有格式
poetry run autoflowcfd post export --case case_001 --output all

# 仅导出VTK（用于ParaView可视化）
poetry run autoflowcfd post export-vtk --case case_001

# 计算气动系数
poetry run autoflowcfd post coefficients --case case_001

# 生成收敛报告
poetry run autoflowcfd post convergence --case case_001

# 瞬态统计（均值、RMS、PSD）
poetry run autoflowcfd post transient-mean --case case_001
poetry run autoflowcfd post transient-rms --case case_001
poetry run autoflowcfd post transient-psd --case case_001
```

#### 网格工具

```bash
# 解析网格并显示信息
poetry run autoflowcfd grid parse sedan.nas

# 校验网格质量
poetry run autoflowcfd grid validate sedan.nas

# 转换网格格式
poetry run autoflowcfd grid convert sedan.nas --format vtk
```

### 方法2：使用Python API

#### 基本用法

```python
from autoflowcfd.grid import NASParser
from autoflowcfd.core import FRSolver
from autoflowcfd.config import SteadyConfig, BackendType, TurbulenceModel

# 加载网格
print("Loading grid...")
parser = NASParser("examples/ahmed_body_demo.nas")
grid_data = parser.parse()
print(f"Grid: {grid_data.node_count} nodes, {grid_data.cell_count} cells")

# 创建配置
config = SteadyConfig(
    backend=BackendType.CPU,
    order=2,
    turbulence=TurbulenceModel.SST_KW,
    max_iter=1000,
    convergence_tol=1.0e-6,
    output_dir="./quick_test_output"
)

# 运行求解器
print("Creating solver...")
solver = FRSolver(grid_data, config)

print("Running simulation...")
result = solver.solve()

# 输出结果
print("\n" + "="*60)
print("Simulation Results")
print("="*60)
print(f"Converged: {result.converged}")
print(f"Iterations: {result.iterations}")
print(f"Final Residual: {result.final_residual:.6e}")
coeffs = result.get_mean_coefficients()
print(f"Cd: {coeffs['Cd']:.4f}")
print(f"Cl: {coeffs['Cl']:.4f}")
print("="*60)
```

#### 高级API用法

```python
from autoflowcfd import GridParser, Solver, Config

# 解析网格文件
parser = GridParser("car_model.nas")
grid = parser.parse()

# 配置求解器
config = Config(
    mode="steady",
    turbulence="sst_kw",
    backend="cpu",
    order=2
)

# 创建求解器并运行
solver = Solver(grid, config)
results = solver.run()

# 获取风阻系数
print(f"Drag Coefficient (Cd): {results.cd:.4f}")
print(f"Lift Coefficient (Cl): {results.cl:.4f}")
```

---

## 输出说明

### 目录结构

求解器会在输出目录生成以下文件：

```
results/
├── checkpoints/              # 检查点目录
│   ├── checkpoint_iter_000100.h5
│   ├── checkpoint_iter_000200.h5
│   └── latest_checkpoint.h5  # 最新检查点符号链接
├── fields/                   # 流场数据
│   ├── field_iter_000100.vtu
│   └── field_iter_000200.vtu
├── convergence.csv           # 收敛历史（残差、Cd、Cl）
├── coefficients.csv          # 气动系数历史
├── config_copy.yaml          # 配置文件副本
└── summary.json              # 输出摘要
```

### 瞬态仿真额外输出

```
results/
├── time_mean/                # 时间平均流场
│   └── time_mean.vtu
├── rms/                      # RMS脉动量
│   └── rms.vtu
└── psd/                      # 功率谱密度
    └── psd_data.csv
```

### convergence.csv 格式

```csv
iteration,residual,Cd,Cl
1,1.234567e+02,0.3500,0.0500
2,9.876543e+01,0.3450,0.0495
...
```

---

## 常见问题

### Q1: 安装时遇到依赖冲突怎么办？

A: 尝试更新Poetry并清理缓存：
```bash
pip install --upgrade poetry
poetry cache clear --all .
poetry install
```

### Q2: 导入错误 "cannot import name 'FRSolver'"

A: 确保已重新安装模块：
```bash
poetry install
```

### Q3: GPU后端无法使用？

A: 确保已安装：
1. NVIDIA显卡驱动（最新版本）
2. CUDA Toolkit 12.x
3. CuPy: `poetry install -E gpu`

验证GPU可用性：
```python
import cupy as cp
print(cp.cuda.runtime.getDeviceCount())  # 应输出 > 0
```

或使用AutoFlowCFD内置检查：
```bash
python -c "from autoflowcfd.core import get_available_backends; print(get_available_backends())"
```

### Q4: 网格文件找不到

A: 确保网格文件路径正确：
```bash
# 使用绝对路径
poetry run autoflowcfd solve run C:/path/to/grid.nas

# 或使用相对路径（从项目根目录）
poetry run autoflowcfd solve run examples/ahmed_body_demo.nas
```

### Q5: 如何查看详细的日志输出？

A: 使用 `-v` 标志：
```bash
poetry run autoflowcfd solve run car.nas -v
```

### Q6: 计算结果保存在哪里？

A: 默认保存在 `./results` 目录，可通过 `--output` 参数指定。

### Q7: 内存不足怎么办？

A: 对于大网格，减少线程数或使用GPU：
```bash
# 限制CPU线程数
poetry run autoflowcfd solve run large_grid.nas --threads 4

# 或使用GPU
poetry run autoflowcfd solve run large_grid.nas --backend gpu
```

---

## 性能参考

基于不同硬件配置的预期性能：

| 网格规模 | CPU (4线程) | GPU (CUDA) |
|---------|------------|-----------|
| 10万单元 | ~200 迭代/分钟 | ~800 迭代/分钟 |
| 100万单元 | ~50 迭代/分钟 | ~200 迭代/分钟 |
| 1000万单元 | ~5 迭代/分钟 | ~20 迭代/分钟 |

实际性能取决于：
- FR阶数（1/2/3阶，阶数越高计算量越大）
- 湍流模型复杂度（RANS < DES < DDES）
- 硬件配置（CPU核心数/GPU型号）
- 内存带宽

---

## 下一步

### 学习资源

- 📖 [完整文档](docs/) - 详细的技术文档
- 🔧 [配置示例](examples/config_example.yaml) - YAML配置模板
- 💡 [API参考](docs/API.md) - Python API详细说明
- 🎯 [算例教程](examples/) - Ahmed Body等完整算例
- 🤝 [贡献指南](CONTRIBUTING.md) - 参与项目开发

### 进阶主题

1. **自定义边界条件**: 扩展BoundaryManager实现特殊边界
2. **湍流模型选择**: 根据仿真需求选择合适的湍流模型
3. **网格质量优化**: 使用网格校验器提升仿真精度
4. **并行计算**: 配置多核CPU或多GPU加速
5. **AI集成**: 与AI Agent结合实现自动化优化

### 获取帮助

- **问题报告**: [GitHub Issues](https://github.com/AutoFlowCFD/AutoFlowCFD/issues)
- **讨论交流**: [GitHub Discussions](https://github.com/AutoFlowCFD/AutoFlowCFD/discussions)
- **邮件联系**: contact@autoflowcfd.org

---

**祝你使用愉快！** 🚀

**最后更新**: 2026-07-25  
**版本**: AutoFlowCFD v0.1.0
