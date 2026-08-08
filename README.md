# AutoFlowCFD

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](docs/)

<div align="center">
**面向汽车空气动力学的高性能开源 CFD 软件**

*Python 全栈开发 · 通量重构高阶格式 · CPU/GPU 异构加速 · AI Agent 原生集成*


---

## 🚀 项目愿景

AutoFlowCFD 旨在填补工业级高精度 CFD 与低门槛二次开发之间的鸿沟。我们打造全球首款**基于 Python 全栈顶层开发、原生支持 ANSA .nas 网格、采用通量重构（FR）高阶算法、CPU/GPU 混合调度、AI Agent 友好**的汽车外流场仿真工具。

### 💡 为什么选择 AutoFlowCFD？

| 传统商业 CFD | 现有开源 CFD | **AutoFlowCFD** |
|:-----------:|:-----------:|:--------------:|
| ❌ 闭源高价，授权费用昂贵 | ❌ C++ 底层，二次开发门槛极高 | ✅ **Apache 2.0 开源，Python 低门槛开发** |
| ❌ 无法自定义核心算法 | ❌ 传统二阶 FVM，精度有限 | ✅ **FR 高阶格式，精度提升显著** |
| ❌ GPU/高阶格式单独收费 | ❌ 无原生 NAS 网格支持 | ✅ **原生解析 ANSA .nas，零成本适配** |
| ❌ 算力无法与 AI 共享 | ❌ 缺少标准化 API/CLI | ✅ **CPU/GPU 动态调度，AI Agent 原生集成** |

---

## ✨ 核心特性

### 🔬 高精度数值算法
- **通量重构（Flux Reconstruction）高阶格式**：支持 1-3 阶精度，显著提升边界层、尾流场仿真精度
- **先进湍流模型体系**：SST k-ω RANS、DES/DDES 混合、LES 大涡模拟（插件化扩展）
- **自适应 CFL 策略**：智能收敛监控，稳态-瞬态无缝耦合

### ⚡ 高性能计算引擎
- **CPU 并行加速**：Numba JIT 编译 + 多线程向量化，4-5x 加速比
- **GPU CUDA 加速**：CuPy 封装 FR 核心算子，10-20x 加速比（相比单核 CPU）
- **SoA 内存布局**：结构体数组优化，最大化缓存命中率

### 🔧 工业级工作流适配
- **原生 NAS 网格解析**：直接读取 ANSA v22/v23/v24 生成的 `.nas` 文件，无需转换
- **智能边界识别**：自动映射边界条件组（INLET/OUTLET/WALL/SYMMETRY）
- **网格质量校验**：长宽比、扭曲度、雅可比行列式全自动检测

### 🤖 AI Agent 友好设计
- **双接口架构**：CLI 命令行 + Python API，无缝嵌入自动化流水线
- **结构化输出**：JSON/CSV/VTK 多格式结果，便于后处理与数据驱动优化
- **算力分时复用**：与大模型训练共享 GPU 资源池，降低硬件成本

### 🧩 模块化扩展架构
- **插件化湍流模型**：新增模型仅需实现 Python 接口，无底层代码侵入
- **清晰模块划分**：Grid/Core/Boundary/Postprocess 职责明确
- **类型注解完备**：MyPy 严格检查，二次开发上手简单

---

## 🎯 应用场景

### 汽车外流场仿真
- **风阻系数（Cd）预测**：误差 ≤1.5%，满足工程开发精度要求
- **气动升力/侧力分析**：高速稳定性评估与优化
- **压力分布可视化**：车身表面 Cp 云图，指导造型改进

### 尾流与涡脱落研究
- **瞬态 DES/DDES 仿真**：捕捉非定常流动特征
- **涡量场分析**：A 柱、后视镜、尾部涡系结构识别
- **气动噪声源定位**：为 NVH 优化提供依据

### 参数化优化与 AI 耦合
- **批量仿真调度**：车身几何参数自动遍历（前倾角、离地间隙等）
- **代理模型训练**：结合机器学习构建快速预测模型
- **数字孪生集成**：实时仿真与物理测试数据融合

---

## 📦 快速开始

### 前置要求

- **Python**: 3.10+
- **操作系统**: Linux / Windows / macOS
- **可选**: NVIDIA GPU + CUDA Toolkit 12.x（用于 GPU 加速）
- **网格准备**: ANSA 前处理器（生成 `.nas` 格式网格）

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/AutoFlowCFD/AutoFlowCFD.git
cd AutoFlowCFD

# 2. 使用 Poetry 安装依赖
pip install poetry
poetry install

# 3. （可选）启用 GPU 支持
poetry install -E gpu
```

### 首次运行

#### 方式一：CLI 命令行（推荐）

```bash
# 稳态 RANS 仿真（CPU）
poetry run autoflowcfd solve run examples/ahmed_demo/car_model.nas \
    --mode steady \
    --turbulence sst_kw \
    --backend cpu \
    --order 2

# 瞬态 DES 仿真（GPU 加速）
poetry run autoflowcfd solve run car_model.nas \
    --mode transient \
    --turbulence des \
    --backend gpu \
    --dt 1e-4 \
    --total-time 0.1
```

#### 方式二：Python API

```python
from autoflowcfd import AutoFlowCFDAPI

# 创建 API 实例
api = AutoFlowCFDAPI()

# 加载网格
grid = api.load_grid("car_model.nas")

# 运行稳态仿真
result = api.run_steady(
    grid,
    backend="gpu",
    turbulence="sst_kw",
    order=2,
    max_iter=5000
)

# 获取气动系数
coeffs = api.calculate_coefficients(result)
print(f"风阻系数 Cd: {coeffs['Cd']:.4f}")
print(f"升力系数 Cl: {coeffs['Cl']:.4f}")
```

### 后处理与可视化

```bash
# 导出 VTK 格式（ParaView 可视化）
poetry run autoflowcfd post export-vtk --case case_001

# 计算气动系数历史
poetry run autoflowcfd post coefficients --case case_001

# 生成收敛报告
poetry run autoflowcfd post convergence --case case_001
```

---

## 📊 性能基准

基于 Ahmed Body 标准算例（100 万六面体网格单元）的测试结果：

| 后端配置 | FR 阶数 | 每步耗时 | 内存占用 | 相对加速比 |
|:--------:|:-------:|:--------:|:--------:|:----------:|
| CPU (4 线程) | 2nd | ~2.5s | 4 GB | 1.0x |
| CPU (16 线程) | 2nd | ~0.8s | 4 GB | 3.1x |
| GPU (NVIDIA A100) | 2nd | ~0.3s | 6 GB | 8.3x |
| GPU (NVIDIA A100) | 3rd | ~0.5s | 8 GB | 5.0x |

*测试环境: Intel Xeon Gold 6248 (16C/32T) / NVIDIA A100 40GB / CUDA 12.2*

> **注**：实际性能受网格质量、边界层分辨率、湍流模型复杂度影响。GPU 加速在大规模网格（>500 万单元）下优势更显著。

---

## 🏗️ 项目架构

```
AutoFlowCFD/
├── src/autoflowcfd/          # 核心源代码
│   ├── cli/                  # CLI 命令行接口（Click 框架）
│   ├── api.py                # Python API 统一入口
│   ├── core/                 # 求解器引擎
│   │   ├── backend/          # CPU/GPU 后端实现
│   │   ├── solver_steady.py  # 稳态求解器主循环
│   │   ├── transient_solver_loop.py  # 瞬态求解器
│   │   ├── aero_coeffs.py    # 气动系数计算
│   │   └── checkpoint.py     # 检查点管理
│   ├── grid/                 # 网格解析与处理
│   │   ├── parser.py         # NAS 文件解析器
│   │   ├── structures.py     # SoA 数据结构
│   │   └── validator.py      # 网格质量校验
│   ├── boundary/             # 边界条件管理
│   ├── config/               # 配置管理（YAML）
│   ├── postprocess/          # 后处理工具
│   └── utils/                # 工具函数
├── tests/                    # 测试套件（pytest）
│   ├── unit/                 # 单元测试
│   └── integration/          # 集成测试
├── examples/                 # 示例算例
│   ├── ahmed_demo/           # Ahmed Body 标准算例
│   ├── cube_demo/            # 立方体绕流验证
│   └── plate_demo/           # 平板边界层案例
├── docs/                     # 技术文档
├── ProjectFiles/             # 项目规划文档
└── pyproject.toml            # Poetry 依赖管理
```

---

## 📚 文档导航

### 入门指南
- [📘 快速开始](QUICKSTART.md) - 5 分钟完成安装与首次仿真
- [🔧 安装详解](INSTALL.md) - 多平台安装与依赖管理
- [💻 使用示例](examples/) - Ahmed Body 等完整算例教程

### 技术文档
- [🏛️ 架构设计](ARCHITECTURE.md) - 系统架构与模块划分
- [📐 数据结构](ProjectFiles/2-3_数据结构设计文档-Part1.md) - SoA 布局与内存优化
- [🔌 API 参考](docs/API.md) - Python API 详细说明
- [⚙️ 配置指南](docs/configuration.md) - YAML 配置文件模板

### 进阶主题
- [🚀 性能优化](docs/CFL_ADAPTIVE_OPTIMIZATION.md) - CFL 自适应与收敛加速
- [🎨 VTK 导出](docs/VTK_EXPORT_GUIDE.md) - 可视化数据导出指南
- [🔬 边界条件](docs/boundary_configuration_guide.md) - 边界配置详解
- [🛠️ 二次开发](CONTRIBUTING.md) - 贡献指南与代码规范

### 社区与维护
- [🗺️ 项目路线图](ROADMAP.md) - 迭代规划与功能预告
- [📝 变更日志](CHANGELOG.md) - 版本更新记录
- [🤝 行为准则](CODE_OF_CONDUCT.md) - 社区协作规范
- [🔒 安全策略](SECURITY.md) - 漏洞报告与安全规范

---

## 🚧 当前开发状态

### ✅ 已完成功能（Iteration 1-3）

**Iteration 1: 工程基础设施**
- ✅ Poetry 依赖管理与虚拟环境
- ✅ CI/CD 自动化流水线（GitHub Actions）
- ✅ 代码质量工具链（Black/Isort/MyPy/Pylint）
- ✅ pytest 单元测试框架（覆盖率 ≥80%）

**Iteration 2: 网格解析模块**
- ✅ NAS 文件解析器（v22/v23/v24 格式）
- ✅ SoA 内存布局（NodeArray/CellArray/BoundaryMap）
- ✅ 网格质量校验器（长宽比/扭曲度/雅可比）
- ✅ 边界条件自动识别与映射

**Iteration 3: FR 求解器核心**
- ✅ FR 离散格式（1st/2nd/3rd order）
- ✅ CPU 后端（Numba JIT + 多线程，4.2x 加速）
- ✅ GPU 后端（CuPy 封装，10-20x 加速）
- ✅ SST k-ω 湍流模型 + 壁面函数（y+=30-100）
- ✅ 时间离散（Backward Euler/RK2/AB3）
- ✅ 收敛监控与自适应 CFL 策略
- ✅ 瞬态求解器主循环 + STG 合成湍流
- ✅ 检查点机制（HDF5 存储，支持断点续算）

### 🚧 进行中（Iteration 4）

- 🔄 CLI 命令行接口完整实现（solve/post/grid 子命令）
- 🔄 Python API 高层封装（AutoFlowCFDAPI 类）
- 🔄 配置管理系统（YAML 解析 + 验证）
- 🔄 边界条件模块增强（速度入口/压力出口/滑移壁面）

### 📅 规划中（Iteration 5-6）

- 📋 多 GPU 分布式计算（MPI + NCCL）
- 📋 气动噪声模块（FW-H声类比）
- 📋 LES 大涡模拟插件
- 📋 AI Agent 集成示例（参数优化流水线）
- 📋 Docker 容器化部署
- 📋 Web 可视化界面（可选）

---

## 🤝 贡献指南

我们欢迎社区贡献！无论是 Bug 修复、功能开发、文档完善还是算例分享，每一份贡献都至关重要。

### 快速参与

1. **Fork 本仓库** → 创建功能分支 → 提交更改 → 发起 Pull Request
2. **报告问题**：通过 [GitHub Issues](https://github.com/AutoFlowCFD/AutoFlowCFD/issues) 反馈 Bug 或提出新功能建议
3. **参与讨论**：在 [GitHub Discussions](https://github.com/AutoFlowCFD/AutoFlowCFD/discussions) 交流技术问题与使用心得
4. **分享算例**：贡献您的仿真案例至 `examples/` 目录，帮助他人快速上手

### 代码质量标准

- ✅ 遵循 PEP 8 风格指南（Black 格式化强制检查）
- ✅ 为所有公共 API 添加类型注解（MyPy 严格模式）
- ✅ 编写单元测试，覆盖率 ≥80%
- ✅ API 变更时同步更新文档与示例代码
- ✅ Git 提交遵循 Conventional Commits 规范（feat/fix/docs/refactor/test）

详细贡献流程请参阅 [CONTRIBUTING.md](CONTRIBUTING.md)。

---

## 📄 许可证

本项目采用 **Apache License 2.0** 开源许可证 - 详见 [LICENSE](LICENSE) 文件。

**商业友好**：允许私有化修改、商用部署、无 GPL 传染风险。  
**署名要求**：衍生作品需保留原始版权声明与许可证副本。

---

## 🙏 致谢

AutoFlowCFD 站在巨人的肩膀上，感谢以下开源项目的卓越贡献：

- **ANSA**：业界领先的 CAE 前处理工具（网格生成）
- **NumPy/SciPy**：Python 科学计算基石
- **Numba**：CPU 并行加速利器
- **CuPy**：GPU 数值计算库
- **Click**：优雅的 CLI 框架
- **VTK/PyVista**：可视化数据处理
- **HDF5/h5py**：高性能数据序列化
- **Loguru**：现代化日志记录

特别感谢流体力学社区对通量重构（FR）算法的持续研究与开源分享。

---

## 📬 联系方式

- **问题反馈**：[GitHub Issues](https://github.com/AutoFlowCFD/AutoFlowCFD/issues)
- **社区讨论**：[GitHub Discussions](https://github.com/AutoFlowCFD/AutoFlowCFD/discussions)
- **项目联系人**：Mr Lu
- **邮箱联系**：luxw_chd@126.com
- **项目官网**：https://autoflowcfd.github.io（建设中）

---

<div align="center">

**AutoFlowCFD** - 让高精度 CFD 触手可及 🚀

*赋能汽车空气动力学创新，拥抱开源与 AI 时代*

[⭐ Star this repo](https://github.com/AutoFlowCFD/AutoFlowCFD) · [🍴 Fork this repo](https://github.com/AutoFlowCFD/AutoFlowCFD/fork) · [📖 Read the docs](docs/)

