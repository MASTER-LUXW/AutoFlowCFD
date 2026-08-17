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
- **先进湍流模型体系**：SST k-ω RANS、DDES 延迟分离涡模拟、WMLES 壁面模型大涡模拟、WALE 亚格子模型
- **AUSM+up 黎曼求解器**：含低马赫数修正，保持反对称性，保证全速域精度
- **Order Continuation**：P0→P1→...→目标阶数逐步提升策略，加速高阶格式收敛
- **Q-Criterion 涡识别准则**：基于 Green-Gauss 速度梯度重建，供 VTK 导出使用

### ⚡ 高性能计算引擎
- **CPU 并行加速**：Numba JIT 编译 + `prange` 多线程向量化，界面项/体积项全并行
- **MPI 域分解并行**：METIS 网格分区 + Halo 层交换 + 分布式求解器，支持跨节点 HPC 集群扩展
- **面图着色优化**：消除 scatter-add 写冲突，内存从 O(n_threads × N) 降至 O(N)
- **GPU CUDA 加速**：CuPy 统一框架，覆盖 P0/P≥1 无粘残差、粘性残差、物理梯度、时间积分全流程
- **多 GPU + MPI 分布式**：每个 rank 绑定一块 GPU，Halo 交换（CUDA-aware / staging buffer）+ GPU 残差 + 全局归约
- **高级时间推进方案**：SSP-RK2/RK3、IMEX_EULER（隐式-显式分裂）、DUAL_TIME（BDF1/BDF2 + 伪时间迭代）
- **体积项去混叠（Over-integration）**：fine 几何积分 + 插值 + 限制回 coarse，消除高阶格式混叠误差

### 🔧 工业级工作流适配
- **原生 NAS 网格解析**：直接读取 ANSA v22/v23/v24 生成的 `.nas` 文件，无需转换
- **智能边界识别**：自动映射边界条件组（INLET/OUTLET/WALL/SYMMETRY）
- **网格质量校验**：长宽比、扭曲度、雅可比行列式全自动检测

### 🤖 AI Agent 友好设计
- **CLI 命令行接口**：Click 框架，`solve`/`post`/`grid`/`config`/`utils` 子命令体系，无缝嵌入自动化流水线
- **结构化输出**：JSON/CSV/VTK 多格式结果，便于后处理与数据驱动优化
- **检查点断点续算**：单机/分布式 GPU Checkpoint（HDF5），支持容错恢复与变 rank 数加载

### 🧩 模块化扩展架构
- **插件化湍流模型**：新增模型仅需实现 Python 接口，无底层代码侵入
- **清晰模块划分**：FR/Core/Grid/Boundary/Postprocess/CLI/Config 职责明确
- **幽灵态边界框架**：统一处理 WALL/FARFIELD/INLET/OUTLET/SYMMETRY 边界，含 SEM 合成湍流入口

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
# 1. 从面网格生成体网格
poetry run autoflowcfd grid generate-volume car_model.nas -o car_volume.nas

# 2. 稳态 RANS 仿真（CPU 单机）
poetry run autoflowcfd solve steady car_volume.pkl \
    --backend cpu \
    --order 2 \
    --turbulence-model sst \
    --max-iter 5000 \
    --reference-area 2.2

# 3. 稳态 RANS 仿真（MPI 并行，域分解）
mpirun -np 8 poetry run autoflowcfd solve steady car_volume.pkl \
    --backend cpu \
    --order 2 \
    --turbulence-model sst \
    --n-ranks 8 \
    --reference-area 2.2

# 4. 稳态 RANS 仿真（GPU 单机）
poetry run autoflowcfd solve steady car_volume.pkl \
    --backend gpu \
    --gpu-device 0 \
    --order 2 \
    --turbulence-model sst \
    --max-iter 5000 \
    --reference-area 2.2

# 5. 瞬态 DES 仿真（GPU + DUAL_TIME）
poetry run autoflowcfd solve transient car_volume.pkl \
    --backend gpu \
    --time-method dual-time \
    --turbulence-model ddes \
    --dt 1e-4 \
    --physical-time 0.1 \
    --reference-area 2.2

# 6. 多 GPU 分布式仿真（MPI + GPU）
mpirun -np 4 poetry run autoflowcfd solve steady car_volume.pkl \
    --backend gpu \
    --multi-gpu \
    --n-ranks 4 \
    --save-checkpoint checkpoint.h5

# 5. 后处理：导出 VTK 可视化（含 Q-Criterion 涡识别）
poetry run autoflowcfd post export-vtk \
    --case ./results/steady/ \
    --variables velocity pressure q_criterion \
    --output flow_field.vtu
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

基于 cube_demo 生产网格（545,597 单元，1,326,110 面）的 P2 阶数测试结果：

| 后端配置 | 单次迭代耗时 | 相对加速比 | 说明 |
|:--------:|:----------:|:----------:|:-----|
| CPU (1 线程) | 87.63s | 1.00x | 基准 |
| **CPU (4 线程)** | **57.35s** | **1.53x** | **实测甜点** |
| CPU (8 线程) | 61.46s | 1.43x | 开始倒退 |
| CPU (16 线程) | 67.19s | 1.30x | per-thread buffer 内存压力 |

*测试环境: 16 物理核 CPU / 64GB RAM*

> **注**：超过 4 线程后性能倒退的根因是界面 kernel 的 scatter-add 需要私有缓冲区归约。已通过面图着色方案（`face_coloring.py`）提供替代方案，以及 MPI 域分解实现跨节点扩展。

---

## 🏗️ 项目架构

```
AutoFlowCFD/
├── src/autoflowcfd/          # 核心源代码
│   ├── fr/                   # 通量重构高阶格式（算子、模态基、积分点）
│   ├── core/                 # 求解器引擎
│   │   ├── backend/          # CPU/GPU 后端实现
│   │   ├── mpi/              # MPI 域分解并行（METIS 分区 + Halo 交换 + 分布式求解器）
│   │   ├── fr_solver.py      # FR 求解器主类
│   │   ├── fr_residual_inviscid.py  # 无粘残差（AUSM+up 黎曼求解器）
│   │   ├── fr_viscous_flux.py       # 粘性残差（BR1 格式）
│   │   ├── turbulence_sst.py        # SST k-ω 湍流模型
│   │   ├── turbulence_transport_kernel.py  # 湍流输运 Numba kernel
│   │   ├── face_coloring.py         # 面图着色（scatter-add 冲突消除）
│   │   ├── turbulence_des.py        # DDES 混合模型
│   │   ├── turbulence_wmles.py      # WMLES 壁面模型
│   │   ├── time_integration*.py     # 时间积分（SSP-RK/IMEX/Dual-Time）
│   │   ├── order_continuation.py    # 阶数延续策略
│   │   └── checkpoint.py            # 检查点管理
│   ├── grid/                 # 网格解析与处理
│   │   ├── high_order_mesh.py       # 高阶网格（Duffy 坍缩坐标映射）
│   │   ├── face_connectivity.py     # FR 面连接关系
│   │   └── mesh_gen/                # 体网格生成（BL  extrusion + tetgen）
│   ├── boundary/             # 边界条件（幽灵态框架 + SEM 合成湍流入口）
│   ├── cli/                  # CLI 命令行接口（Click 框架）
│   ├── config/               # 配置管理（YAML）
│   ├── postprocess/          # 后处理（气动系数、VTK 导出、Q-Criterion）
│   └── utils/                # 工具函数
├── tests/                    # 测试套件（pytest）
│   ├── unit/                 # 单元测试
│   ├── integration/          # 集成测试
│   └── validation/           # 验证算例
├── examples/                 # 示例算例
│   ├── ahmed_demo/           # Ahmed Body 标准算例
│   ├── cube_demo/            # 立方体绕流验证
│   └── plate_demo/           # 平板边界层案例
├── docs/                     # 技术文档
├── ProjectFiles/             # 项目规划文档（V1.0/V2.0）
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
- [🔌 API 参考](docs/API.md) - Python API 详细说明
- [⚙️ 配置指南](docs/CONFIGURATION_GUIDE.md) - YAML 配置文件说明
- [📐 V2.0 实施路径](ProjectFiles/V2.0/0_项目实施路径.md) - V2.0 改造规划
- [📋 V2.0 功能点清单](ProjectFiles/V2.0/1_系统改造功能点.md) - 25 项功能详细说明

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

### ✅ V2.0 已完成功能

**数值算法核心**
- ✅ FR 高阶离散格式（P1/P2/P3），含 Duffy 坍缩坐标四面体/棱柱映射
- ✅ AUSM+up 黎曼求解器（含低马赫数 Mp/pu 修正，保持反对称性）
- ✅ BR1 粘性界面耦合（真实边界幽灵态，温度梯度完整计算）
- ✅ 体积项去混叠（Over-integration：fine 几何 + 插值 + 限制回 coarse）
- ✅ 问题单元检测机制（`suppress_residual_outliers` 残差异常抑制）

**时间积分**
- ✅ SSP-RK2/RK3（Shu-Osher 形式，每 stage 重新计算残差）
- ✅ IMEX Euler（显式对流 + 隐式粘性，阻尼 Picard 子迭代）
- ✅ Dual-Time Stepping（BDF1/BDF2 + SSP-RK3 伪时间 + CFL 自适应）

**湍流模型体系**
- ✅ SST k-ω RANS（F1/F2 混合函数标准 Menter 1994 公式、正性限制器）
- ✅ DDES 延迟分离涡模拟（屏蔽函数 + 有效长度尺度替换）
- ✅ WMLES 壁面模型大涡模拟（Spalding 律 + Newton-Raphson 迭代）
- ✅ WALE 亚格子应力模型

**网格与边界**
- ✅ 原生 NAS 网格解析（ANSA v22/v23/v24）+ 自动体网格生成（BL extrusion + tetgen）
- ✅ 高阶网格初始化（Duffy 映射、解析雅可比、面通量点定位/合并）
- ✅ 幽灵态边界框架（WALL/FARFIELD/INLET/OUTLET/SYMMETRY）
- ✅ SEM 合成湍流入口（Cholesky 分解雷诺应力、涡核对流+再生）
- ✅ 壁面距离场（KD-Tree + Eikonal Dijkstra 近似）

**工程工作流**
- ✅ CLI 完整命令体系（`grid`/`solve`/`post`/`config`/`utils`）
- ✅ 检查点机制（HDF5 存储完整 (n_cells,n_sps,n_vars) 状态，支持 `solve resume`）
- ✅ 气动系数积分（直接在 FR 面通量点上积分压力+粘性力）
- ✅ Q-Criterion 涡识别准则（Green-Gauss 速度梯度重建）
- ✅ 力系数时间平均统计（Welford 在线算法）
- ✅ Order Continuation（P0→P1→...→目标阶数，残差下降触发判据）
- ✅ VTK 导出（legacy + XML VTU，含边界分区、Q-Criterion）

**HPC 并行计算**
- ✅ 面图着色算法 + kernel 完整接入（无粘/粘性/湍流输运，内存 O(N)）
- ✅ 湍流输运 Numba 化（消除 SST k-ω 输运方程串行瓶颈）
- ✅ MPI 域分解基础设施（METIS 分区 + Halo 交换 + 分布式状态/面几何）
- ✅ 分布式残差计算完整接入（`DistributedMeshAdapter` + 分布式无粘/粘性/梯度/湍流残差）
- ✅ 分布式 Checkpoint 保存/加载 + 结果保存（与单机格式兼容，支持变 rank 数恢复）
- ✅ 分区优化（Root rank 执行 METIS，广播结果，非 root rank 不再运行分区算法）
- ✅ 完全分布式网格加载（只有 root 加载完整网格，非 root rank 内存占用降为 1/n_ranks）
- ✅ CLI `--n-ranks` 选项实际触发分布式路径（`mpirun -np N autoflowcfd solve steady ... --n-ranks N`）

**GPU 大规模并行计算**
- ✅ 统一 CuPy 框架（移除 Numba CUDA），覆盖张量运算 + 自定义 kernel
- ✅ GPU 直接 Halo 交换（CUDA-aware MPI 零拷贝 / staging buffer 优化）
- ✅ 分布式 GPU SSP-RK 时间推进（每 stage 重新 halo 交换 + 残差评估）
- ✅ 湍流模型源项 GPU 化（SST k-ω 全程 GPU：F1/F2 blending、涡粘系数、Sk/S_omega）
- ✅ GPU 版 DUAL_TIME（BDF1/BDF2 + SSP-RK3 伪时间迭代 + CFL 自适应）
- ✅ GPU 版 IMEX_EULER（阻尼 Picard 子迭代 + 自适应阻尼因子）
- ✅ 单机/分布式 GPU Checkpoint（HDF5 格式，支持 U/Q 场、湍流场、残差历史、DUAL_TIME 历史）

### 🚧 规划中

- 📋 NCCL 直接 GPU↔GPU 通信（替代当前 halo 交换的 CPU 中转）
- 📋 动态重分区（AMR 场景负载均衡）
- 📋 气动噪声模块（FW-H 声类比）
- 📋 Docker 容器化部署
- 📋 AI Agent 集成示例（参数优化流水线）

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

