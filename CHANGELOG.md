# 变更日志

本文档记录 AutoFlowCFD 的所有重要变更。

格式遵循 [Keep a Changelog](https://keepachangelog.com/) 规范。

---

## [未发布]

### 计划中

- 多 GPU 分布式计算支持（MPI + NCCL）
- LES 大涡模拟插件
- 气动噪声模块（FW-H 声类比）
- Web 可视化界面（可选）
- Docker 容器化部署
- AI Agent 集成示例（参数优化流水线）

---

## [0.1.0] - 2026-07-25

### ✨ 新增功能

#### Iteration 1: 工程基础设施
- ✅ 项目骨架与目录结构
- ✅ Poetry 依赖管理系统
- ✅ CI/CD 自动化流水线（GitHub Actions）
- ✅ 代码质量工具链（Black/Isort/MyPy/Pylint）
- ✅ pytest 单元测试框架
- ✅ pre-commit 钩子配置

#### Iteration 2: 网格解析模块
- ✅ NAS 文件解析器（支持 v22/v23/v24 格式）
- ✅ SoA 内存布局（NodeArray/CellArray/BoundaryMap）
- ✅ 网格质量校验器（长宽比/扭曲度/雅可比行列式）
- ✅ 边界条件自动识别与映射
- ✅ 流式解析大文件（>1GB）支持

#### Iteration 3: FR 求解器核心
- ✅ FR 离散格式（1st/2nd/3rd order）
- ✅ CPU 后端（Numba JIT + 多线程，4.2x 加速）
- ✅ GPU 后端（CuPy 封装，10-20x 加速）
- ✅ SST k-ω 湍流模型
- ✅ 壁面函数（y+=30-100 支持）
- ✅ 时间离散方案（Backward Euler/RK2/AB3）
- ✅ 收敛监控与自适应 CFL 策略
- ✅ 瞬态求解器主循环
- ✅ 稳态-瞬态耦合（STG 合成湍流）
- ✅ 检查点机制（HDF5 存储，支持断点续算）
- ✅ 气动系数计算（Cd/Cl/Cs/Cm）

#### 文档与示例
- ✅ README.md 项目概述
- ✅ QUICKSTART.md 快速开始指南
- ✅ ARCHITECTURE.md 架构设计文档
- ✅ CONTRIBUTING.md 贡献指南
- ✅ CODE_OF_CONDUCT.md 社区行为准则
- ✅ SECURITY.md 安全策略
- ✅ ROADMAP.md 项目路线图
- ✅ API.md Python API 参考
- ✅ CONFIGURATION_GUIDE.md 配置指南
- ✅ DEVELOPER_GUIDE.md 开发者指南
- ✅ TUTORIALS.md 算例教程
- ✅ PERFORMANCE_OPTIMIZATION.md 性能优化指南
- ✅ INDEX.md 文档中心索引
- ✅ Ahmed Body 标准算例
- ✅ 立方体绕流验证算例
- ✅ 平板边界层案例

### 🎯 核心特性

- **原生 NAS 网格支持**: 直接解析 ANSA 生成的 `.nas` 文件（v22/v23/v24）
- **异构计算**: 同时支持 CPU（Numba 并行化）和 GPU（CUDA 加速）
- **高阶 FR 求解器**: 采用 Flux Reconstruction 方法，支持 1-3 阶精度
- **先进湍流模型**: SST k-ω、DES/DDES（插件化架构）
- **双接口设计**: CLI 命令行界面 + Python API，便于 Agent 集成
- **模块化设计**: 清晰的模块划分，易于扩展和定制

### 📊 性能指标

基于 Ahmed Body 算例（100 万六面体单元）：

| 后端 | FR 阶数 | 每步耗时 | 加速比 |
|------|---------|---------|--------|
| CPU (4 线程) | 2nd | ~2.5s | 1.0x |
| CPU (16 线程) | 2nd | ~0.8s | 3.1x |
| GPU (A100) | 2nd | ~0.3s | 8.3x |
| GPU (A100) | 3rd | ~0.5s | 5.0x |

### 🔧 技术栈

- **语言**: Python 3.10+
- **数值计算**: NumPy/CuPy
- **并行加速**: Numba/CUDA
- **CLI 框架**: Click
- **配置管理**: PyYAML
- **数据序列化**: HDF5/h5py
- **可视化**: VTK/pyvista
- **日志**: loguru
- **测试**: pytest

### 📝 已知限制

- 仅支持单 GPU 计算（多 GPU 分布式计算在规划中）
- LES 大涡模拟尚未实现
- 气动噪声模块尚未实现
- 网格变形功能需要额外工具支持
- Web 可视化界面尚未开发

### 🐛 已知问题

详见 [GitHub Issues](https://github.com/AutoFlowCFD/AutoFlowCFD/issues)

---

## 版本说明

### 版本号规则

AutoFlowCFD 遵循 [语义化版本 2.0.0](https://semver.org/lang/zh-CN/)：

- **主版本号**: 不兼容的 API 修改
- **次版本号**: 向下兼容的功能性新增
- **修订号**: 向下兼容的问题修正

### 阶段标识

- **Alpha** (0.0.x): 早期开发，API 可能频繁变化
- **Beta** (0.x.x): 功能基本稳定，API 可能有小幅调整
- **Stable** (1.x.x+): API 稳定，向后兼容

当前版本 **0.1.0** 处于 **Beta** 阶段。

---

## 升级指南

### 从 0.0.x 升级到 0.1.0

由于 0.1.0 是首个公开发布版本，无需升级。

### 未来升级注意事项

在 1.0 版本之前，API 可能会有不兼容的变更。建议在 `pyproject.toml` 中锁定版本：

```toml
[tool.poetry.dependencies]
autoflowcfd = "==0.1.0"
```

---

## 贡献者

感谢以下贡献者对 AutoFlowCFD 的支持：

- **AutoFlowCFD Team**: 核心开发团队
- **社区贡献者**: [查看完整列表](CONTRIBUTORS.md)（待创建）

---

## 参考链接

- [GitHub Releases](https://github.com/AutoFlowCFD/AutoFlowCFD/releases)
- [Keep a Changelog](https://keepachangelog.com/)
- [语义化版本](https://semver.org/lang/zh-CN/)

---

**最后更新**: 2026-07-25
