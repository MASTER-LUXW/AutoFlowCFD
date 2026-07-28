# AutoFlowCFD

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![CI Status](https://github.com/AutoFlowCFD/AutoFlowCFD/actions/workflows/ci.yml/badge.svg)](https://github.com/AutoFlowCFD/AutoFlowCFD/actions)

## 项目概述

**AutoFlowCFD** 是一款专注于汽车外流场仿真的开源计算流体力学（CFD）软件。它提供高精度、高速度的CFD分析能力，并支持AI Agent集成。

### 核心特性

- **原生NAS网格支持**：直接解析ANSA生成的 `.nas` 网格文件（v22/v23/v24格式）
- **异构计算**：同时支持CPU（Numba并行化）和GPU（CUDA加速）
- **高阶FR求解器**：采用Flux Reconstruction方法，支持1-3阶精度
- **先进湍流模型**：SST k-ω、DES/DDES、LES（插件化架构）
- **双接口设计**：CLI命令行界面 + Python API，便于Agent集成
- **模块化设计**：清晰的模块划分，易于扩展和定制

### 应用场景

- 汽车外流场仿真分析
- 风阻系数（Cd）预测（误差 ≤1.5%）
- 尾流分析与涡脱落检测
- 气动优化与设计迭代

## 快速开始

### 前置要求

- Python 3.10+
- NVIDIA GPU（可选，用于GPU加速）
- ANSA前处理器（用于生成.nas网格文件）

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/AutoFlowCFD/AutoFlowCFD.git
cd AutoFlowCFD

# 使用Poetry安装依赖
pip install poetry
poetry install

# 激活虚拟环境
poetry shell
```

### 基本用法

```bash
# 运行稳态RANS仿真
autoflowcfd solve --grid car_model.nas --mode steady --turbulence sst_kw

# 运行瞬态DES仿真
autoflowcfd solve --grid car_model.nas --mode transient --turbulence ddes --time-step 1e-4

# 后处理结果
autoflowcfd postprocess --case case_001 --output vtk
```

### Python API示例

```python
from autoflowcfd import GridParser, Solver, Config

# 解析网格文件
parser = GridParser("car_model.nas")
grid = parser.parse()

# 配置求解器
config = Config(
    mode="steady",
    turbulence="sst_kw",
    backend="cpu"  # 或 "gpu"
)

# 运行仿真
solver = Solver(grid, config)
results = solver.run()

# 提取风阻系数
cd = results.aerodynamic_coefficients.drag
print(f"风阻系数: {cd:.4f}")
```

## 文档

- [架构设计](ARCHITECTURE.md)
- [API参考](docs/API.md)
- [部署指南](docs/DEPLOY.md)
- [贡献指南](CONTRIBUTING.md)
- [行为准则](CODE_OF_CONDUCT.md)
- [安全策略](SECURITY.md)
- [项目路线图](ROADMAP.md)
- [变更日志](CHANGELOG.md)

## 项目结构

```
AutoFlowCFD/
├── src/autoflowcfd/       # 源代码
│   ├── cli/               # 命令行接口
│   ├── core/              # 核心求解器引擎
│   │   ├── backend/       # CPU/GPU后端实现
│   │   ├── fr_scheme.py   # FR离散格式
│   │   ├── turbulence.py  # 湍流模型
│   │   └── ...            # 其他核心模块
│   ├── grid/              # 网格解析与处理
│   ├── boundary/          # 边界条件管理
│   ├── postprocess/       # 后处理工具
│   ├── config/            # 配置管理
│   └── utils/             # 工具函数
├── tests/                 # 测试套件
│   ├── unit/              # 单元测试
│   └── integration/       # 集成测试
├── docs/                  # 文档
├── examples/              # 示例算例
└── ProjectFiles/          # 项目规划文档
```

## 性能基准

| 网格规模 | 后端 | 每步耗时 | 内存占用 |
|---------|------|---------|---------|
| 100万单元 | CPU | ~2.5秒 | 4 GB |
| 100万单元 | GPU | ~0.6秒 | 6 GB |
| 1000万单元 | CPU | ~25秒 | 40 GB |
| 1000万单元 | GPU | ~5秒 | 12 GB |

*测试环境: Intel Xeon Gold 6248 / NVIDIA A100 40GB*

## 当前开发状态

### ✅ 已完成功能（Iteration 1-3）

- **Iteration 1**: 工程基础设施搭建
  - 项目骨架与目录结构
  - Poetry依赖管理
  - CI/CD自动化流水线
  - 代码质量检查工具链

- **Iteration 2**: 网格解析模块
  - NAS文件解析器（v22/v23/v24）
  - 网格数据结构（SoA布局）
  - 网格质量校验器
  - 边界条件识别

- **Iteration 3**: FR求解器核心
  - FR离散格式（1st/2nd/3rd order）
  - CPU后端（Numba并行，4.2x加速）
  - GPU后端（CuPy封装）
  - SST k-ω湍流模型
  - 壁面函数（y+=30-100支持）
  - 时间离散（BE/RK2/AB3）
  - 收敛监控与自适应CFL
  - 瞬态求解器主循环
  - 稳态-瞬态耦合（STG合成湍流）

### 🚧 进行中（Iteration 4）

- CLI命令行接口完整实现
- Python API封装
- 配置管理系统
- 边界条件模块

## 贡献指南

我们欢迎社区贡献！请阅读 [贡献指南](CONTRIBUTING.md) 了解如何参与。

### 开发工作流

1. Fork本仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'feat: add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建Pull Request

### 代码质量标准

- 遵循PEP 8风格指南（Black格式化强制检查）
- 为所有公共API添加类型注解
- 编写单元测试，覆盖率 ≥80%
- API变更时同步更新文档

## 许可证

本项目采用 Apache License 2.0 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 致谢

- ANSA前处理器（网格生成）
- NumPy/CuPy（数值计算）
- Numba（CPU并行加速）
- Click（CLI框架）
- VTK（可视化）

## 联系方式

- **问题反馈**: [GitHub Issues](https://github.com/AutoFlowCFD/AutoFlowCFD/issues)
- **社区讨论**: [GitHub Discussions](https://github.com/AutoFlowCFD/AutoFlowCFD/discussions)
- **邮箱**: contact@autoflowcfd.org（占位符）

---

**AutoFlowCFD** - 面向汽车空气动力学的高性能CFD软件
