# AutoFlowCFD 快速开始指南

<div align="center">

**5 分钟完成安装与首次 CFD 仿真**

</div>

---

## 📋 目录

- [前置要求](#前置要求)
- [安装步骤](#安装步骤)
- [首次运行](#首次运行)
- [结果可视化](#结果可视化)
- [常见问题](#常见问题)
- [下一步学习](#下一步学习)

---

## 前置要求

### 必需软件

| 软件 | 版本要求 | 说明 |
|------|---------|------|
| **Python** | 3.10+ | 推荐使用 Anaconda/Miniconda 管理环境 |
| **Poetry** | 1.6+ | Python 依赖管理工具 |
| **Git** | 2.0+ | 代码版本控制 |

### 可选软件（GPU 加速）

| 软件 | 版本要求 | 说明 |
|------|---------|------|
| **NVIDIA GPU** | CUDA Compute Capability 5.0+ | GTX 10 系列及以上推荐 |
| **CUDA Toolkit** | 12.x | [下载地址](https://developer.nvidia.com/cuda-downloads) |
| **NVIDIA 驱动** | 最新版本 | 确保与 CUDA 版本兼容 |

### 网格准备

- **ANSA 前处理器**：用于生成 `.nas` 格式网格文件（v22/v23/v24 版本）
- **示例网格**：项目自带 Ahmed Body 标准算例，可直接体验

---

## 安装步骤

### 1️⃣ 克隆仓库

```bash
git clone https://github.com/AutoFlowCFD/AutoFlowCFD.git
cd AutoFlowCFD
```

### 2️⃣ 安装 Poetry

如果尚未安装 Poetry：

```bash
# 方式一：使用 pip（推荐）
pip install poetry

# 方式二：使用官方安装脚本
curl -sSL https://install.python-poetry.org | python3 -
```

验证安装：
```bash
poetry --version
# 应输出: Poetry (version 1.x.x)
```

### 3️⃣ 安装项目依赖

```bash
# 安装核心依赖（CPU 计算）
poetry install

# （可选）启用 GPU 支持
poetry install -E gpu
```

> **提示**：首次安装可能需要 3-5 分钟，取决于网络速度。Poetry 会自动创建虚拟环境并安装所有依赖。

### 4️⃣ 验证安装

```bash
# 查看版本信息
poetry run autoflowcfd --version
# 应输出: AutoFlowCFD, version 0.1.0

# 检查可用后端
poetry run python -c "from autoflowcfd.core import get_available_backends; print(get_available_backends())"
# 应输出: ['cpu'] 或 ['cpu', 'gpu']（如果已安装 GPU 支持）
```

---

## 首次运行

### 准备网格文件

项目自带 Ahmed Body 标准算例，位于 `examples/ahmed_demo/` 目录：

```bash
ls examples/ahmed_demo/
# 应看到: car_model.nas（或其他 .nas 文件）
```

如果没有现成网格，可以使用我们提供的示例文件继续测试。

### 方式一：CLI 命令行（推荐新手）

#### 🎯 稳态 RANS 仿真（CPU）

最简单的入门命令：

```bash
poetry run autoflowcfd solve run examples/ahmed_demo/car_model.nas \
    --mode steady \
    --turbulence sst_kw \
    --backend cpu \
    --order 2 \
    --max-iter 1000
```

**参数说明**：
- `--mode steady`：稳态求解模式
- `--turbulence sst_kw`：SST k-ω 湍流模型
- `--backend cpu`：使用 CPU 计算
- `--order 2`：FR 二阶精度
- `--max-iter 1000`：最大迭代次数

#### ⚡ GPU 加速仿真

如果您的系统支持 GPU：

```bash
poetry run autoflowcfd solve run examples/ahmed_demo/car_model.nas \
    --mode steady \
    --turbulence sst_kw \
    --backend gpu \
    --order 2 \
    --max-iter 1000
```

#### 🌊 瞬态 DES 仿真

捕捉非定常流动特征：

```bash
poetry run autoflowcfd solve run examples/ahmed_demo/car_model.nas \
    --mode transient \
    --turbulence des \
    --backend gpu \
    --dt 1e-4 \
    --total-time 0.05
```

**参数说明**：
- `--mode transient`：瞬态求解模式
- `--turbulence des`：DES 混合湍流模型
- `--dt 1e-4`：时间步长（秒）
- `--total-time 0.05`：总仿真时间（秒）

#### 📝 使用配置文件（进阶）

创建 `config.yaml` 文件：

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
    velocity_x: 30.0  # m/s
    pressure: 101325.0  # Pa
    
  outlet:
    type: OUTLET
    pressure: 101325.0
    
  body:
    type: WALL
    wall_function: enhanced
    
  ground:
    type: WALL
    moving: true  # 移动地面
    velocity_x: 30.0
```

运行仿真：

```bash
poetry run autoflowcfd solve run car_model.nas -c config.yaml
```

### 方式二：Python API（推荐开发者）

创建 `run_simulation.py` 脚本：

```python
from autoflowcfd import AutoFlowCFDAPI

# 创建 API 实例
api = AutoFlowCFDAPI(verbose=True)

# 加载网格
print("📂 加载网格文件...")
grid = api.load_grid("examples/ahmed_demo/car_model.nas")
print(f"✅ 网格加载成功: {grid.node_count} 节点, {grid.cell_count} 单元")

# 配置并运行稳态仿真
print("\n🚀 启动稳态仿真...")
result = api.run_steady(
    grid,
    backend="cpu",
    turbulence="sst_kw",
    order=2,
    max_iter=1000,
    convergence_tol=1.0e-6,
    output_dir="./quick_test_output"
)

# 输出结果
print("\n" + "="*60)
print("📊 仿真结果摘要")
print("="*60)
print(f"收敛状态: {'✅ 已收敛' if result.converged else '❌ 未收敛'}")
print(f"迭代次数: {result.iterations}")
print(f"最终残差: {result.final_residual:.6e}")

coeffs = api.calculate_coefficients(result)
print(f"\n风阻系数 Cd: {coeffs['Cd']:.4f}")
print(f"升力系数 Cl: {coeffs['Cl']:.4f}")
print(f"侧力系数 Cs: {coeffs['Cs']:.4f}")
print("="*60)

# 导出 VTK 可视化文件
print("\n💾 导出可视化数据...")
api.export_vtk(result, output_file="./quick_test_output/result.vtu")
print("✅ 导出完成！可使用 ParaView 打开 result.vtu 查看流场")
```

运行脚本：

```bash
poetry run python run_simulation.py
```

---

## 结果可视化

### 📁 输出文件结构

仿真完成后，输出目录（默认 `./results` 或 `./quick_test_output`）包含：

```
results/
├── checkpoints/              # 检查点文件（支持断点续算）
│   ├── checkpoint_iter_000100.h5
│   ├── checkpoint_iter_000200.h5
│   └── latest_checkpoint.h5  # 最新检查点符号链接
├── fields/                   # 流场数据（VTK 格式）
│   ├── field_iter_000100.vtu
│   └── field_iter_000200.vtu
├── convergence.csv           # 收敛历史（残差、Cd、Cl）
├── coefficients.csv          # 气动系数历史
├── config_copy.yaml          # 配置文件副本
└── summary.json              # 输出摘要（JSON 格式）
```

### 🎨 使用 ParaView 可视化

1. **安装 ParaView**：[下载地址](https://www.paraview.org/download/)

2. **打开 VTK 文件**：
   ```bash
   paraview results/fields/field_iter_001000.vtu
   ```

3. **可视化建议**：
   - **压力云图**：显示 `pressure` 字段，调整颜色映射
   - **速度矢量**：显示 `velocity` 矢量箭头
   - **涡量等值面**：显示 `vorticity` 识别涡系结构
   - **流线追踪**：从车头上游释放流线，观察流动分离

### 📈 收敛曲线分析

查看收敛历史：

```bash
# CLI 方式
poetry run autoflowcfd post convergence --case results

# 或直接查看 CSV 文件
cat results/convergence.csv
```

使用 Python 绘制收敛曲线：

```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取收敛数据
df = pd.read_csv("results/convergence.csv")

# 绘制残差曲线
plt.figure(figsize=(10, 6))
plt.semilogy(df['iteration'], df['residual'], label='Residual')
plt.xlabel('Iteration')
plt.ylabel('Residual')
plt.title('Convergence History')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig("convergence.png", dpi=150)
plt.show()

# 绘制气动系数曲线
plt.figure(figsize=(10, 6))
plt.plot(df['iteration'], df['Cd'], label='Cd (Drag)')
plt.plot(df['iteration'], df['Cl'], label='Cl (Lift)')
plt.xlabel('Iteration')
plt.ylabel('Coefficient')
plt.title('Aerodynamic Coefficients')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig("coefficients.png", dpi=150)
plt.show()
```

---

## 常见问题

### ❓ Q1: 安装时遇到依赖冲突怎么办？

**A**: 尝试以下步骤：

```bash
# 1. 更新 Poetry 到最新版本
pip install --upgrade poetry

# 2. 清理缓存并重新安装
poetry cache clear --all .
rm -rf .venv  # 删除现有虚拟环境
poetry install
```

### ❓ Q2: 导入错误 "cannot import name 'AutoFlowCFDAPI'"

**A**: 确保已正确安装模块：

```bash
# 重新安装
poetry install

# 验证安装
poetry run python -c "from autoflowcfd import AutoFlowCFDAPI; print('OK')"
```

### ❓ Q3: GPU 后端无法使用？

**A**: 按以下步骤排查：

```bash
# 1. 确认已安装 GPU 支持
poetry install -E gpu

# 2. 检查 CUDA 可用性
poetry run python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceCount())"
# 应输出 > 0

# 3. 检查 AutoFlowCFD 后端检测
poetry run python -c "from autoflowcfd.core import get_available_backends; print(get_available_backends())"
# 应输出 ['cpu', 'gpu']

# 4. 确认 NVIDIA 驱动和 CUDA 版本兼容
nvidia-smi  # 查看驱动版本
nvcc --version  # 查看 CUDA 版本
```

### ❓ Q4: 网格文件找不到或解析失败？

**A**: 检查以下几点：

```bash
# 1. 确认文件路径正确（使用绝对路径更可靠）
poetry run autoflowcfd solve run /absolute/path/to/car_model.nas

# 2. 验证网格文件格式
poetry run autoflowcfd grid validate car_model.nas

# 3. 查看网格详细信息
poetry run autoflowcfd grid parse car_model.nas
```

### ❓ Q5: 如何查看详细的日志输出？

**A**: 使用 `-v` 或 `--verbose` 标志：

```bash
poetry run autoflowcfd solve run car_model.nas -v
```

### ❓ Q6: 计算结果保存在哪里？

**A**: 
- 默认保存在 `./results` 目录
- 可通过 `--output` 参数指定：
  ```bash
  poetry run autoflowcfd solve run car_model.nas --output ./my_results
  ```

### ❓ Q7: 内存不足怎么办？

**A**: 针对大网格（>1000 万单元）：

```bash
# 方式一：限制 CPU 线程数
poetry run autoflowcfd solve run large_grid.nas --threads 4

# 方式二：使用 GPU（显存通常更大）
poetry run autoflowcfd solve run large_grid.nas --backend gpu

# 方式三：降低 FR 阶数（减少内存占用）
poetry run autoflowcfd solve run large_grid.nas --order 1
```

### ❓ Q8: 仿真不收敛怎么办？

**A**: 尝试以下调整：

```bash
# 1. 降低初始 CFL 数
poetry run autoflowcfd solve run car_model.nas --cfl-init 0.05

# 2. 增加最大迭代次数
poetry run autoflowcfd solve run car_model.nas --max-iter 10000

# 3. 放宽收敛容差
poetry run autoflowcfd solve run car_model.nas --convergence-tol 1.0e-5

# 4. 检查网格质量
poetry run autoflowcfd grid validate car_model.nas
```

---

## 性能参考

基于不同硬件配置的预期性能（Ahmed Body 100 万单元网格）：

| 硬件配置 | FR 阶数 | 每步耗时 | 1000 步耗时 | 推荐场景 |
|---------|:-------:|:--------:|:-----------:|:--------:|
| CPU (4 线程) | 2nd | ~2.5s | ~42 分钟 | 小型笔记本 |
| CPU (16 线程) | 2nd | ~0.8s | ~13 分钟 | 工作站 |
| GPU (RTX 3090) | 2nd | ~0.4s | ~7 分钟 | 高性能桌面 |
| GPU (A100 40GB) | 2nd | ~0.3s | ~5 分钟 | 服务器/HPC |
| GPU (A100 40GB) | 3rd | ~0.5s | ~8 分钟 | 高精度需求 |

> **注**：实际性能受网格质量、边界层分辨率、湍流模型复杂度影响。GPU 加速在大规模网格（>500 万单元）下优势更显著。

---

## 下一步学习

### 📖 深入学习资源

- **[完整文档](docs/)** - 详细的技术文档与教程
- **[架构设计](ARCHITECTURE.md)** - 系统架构与模块划分详解
- **[API 参考](docs/API.md)** - Python API 完整说明
- **[配置示例](examples/config_example.yaml)** - YAML 配置模板库
- **[算例教程](examples/)** - Ahmed Body、立方体绕流等完整案例

### 🎯 进阶主题

1. **自定义边界条件**：扩展 BoundaryManager 实现特殊边界（如旋转车轮）
2. **湍流模型选择**：根据仿真需求选择合适的模型（RANS vs DES vs LES）
3. **网格质量优化**：使用网格校验器提升仿真精度与稳定性
4. **并行计算调优**：配置多核 CPU 或多 GPU 加速策略
5. **AI 集成开发**：与 AI Agent 结合实现自动化参数优化

### 🤝 参与社区

- **问题报告**：[GitHub Issues](https://github.com/AutoFlowCFD/AutoFlowCFD/issues)
- **讨论交流**：[GitHub Discussions](https://github.com/AutoFlowCFD/AutoFlowCFD/discussions)
- **贡献代码**：阅读 [CONTRIBUTING.md](CONTRIBUTING.md)
- **邮件联系**：contact@autoflowcfd.org

### 💡 实践建议

1. **从简单算例开始**：先运行 Ahmed Body 标准算例，熟悉工作流程
2. **逐步增加复杂度**：稳态 → 瞬态，低阶 → 高阶，CPU → GPU
3. **验证与对比**：与实验数据或商业软件结果对比，验证仿真精度
4. **加入社区**：分享您的算例与经验，帮助他人快速上手

---

<div align="center">

**祝你使用愉快！** 🚀

遇到问题不要犹豫，随时通过 GitHub Issues 或社区讨论寻求帮助。

*AutoFlowCFD 团队期待与您共同打造更好的开源 CFD 工具*

</div>

---

## 📬 联系方式

- **问题反馈**：[GitHub Issues](https://github.com/AutoFlowCFD/AutoFlowCFD/issues)
- **讨论交流**：[GitHub Discussions](https://github.com/AutoFlowCFD/AutoFlowCFD/discussions)
- **项目联系人**：Mr Lu
- **邮箱联系**：luxw_chd@126.com

---

**最后更新**: 2026-07-25  
**版本**: AutoFlowCFD v0.1.0
