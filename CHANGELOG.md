# 变更日志

本文档记录 AutoFlowCFD 的所有重要变更。

格式遵循 [Keep a Changelog](https://keepachangelog.com/) 规范。

---

## [未发布]

包版本号以 `pyproject.toml` 为准（当前 0.1.0）；下面的 `[2.0.0]` 是 V2.0 改造的内部里程碑，没有作为该版本号发布。
逐项的缺陷、根因与测试见 git 提交记录（`V2 - ...`）。

### 数值方法
- 四面体与三棱柱改用原生基（Warp & Blend 解点）；删除坍缩坐标基。离散守恒（投影与迹并进体积算子）。
- 隐式稳态 Newton-Krylov：矩阵自由 GMRES、单元块 ILU 与多层粗校正预处理、伪瞬态延拓与自适应 CFL、平均流与湍流紧耦合，
  四个后端同一实现。
- 双时间步内迭代改为隐式 Newton（BDF2）；显式瞬态改为全场同一物理步长的时间精确推进。删除 IMEX_EULER。
- SA-neg 湍流模型；SST-IDDES；RANS 自动近壁处理（平衡壁面模型，`--wall-treatment auto` 为缺省）。
- 守恒的正性限制器取代硬钳位；低马赫伪时间预处理只作用于稳态伪时间推进。
- 修复的一阶物理缺陷包括：热传导符号、AUSM+up P5 系数、壁面内罚项符号、P0 粘性度量、壁面外插负涡粘注入动量、
  壁距按单元索引、湍流输运 neighbor 侧通量点顺序等。

### 后端与求解流程
- 单 GPU 统一到 `GPUFRSolver`；四个后端共用一个求解循环与 Order Continuation 循环、一份 checkpoint 格式。
- `solve steady/transient` 按后端执行只有一份（`cli/solve/run.py`），分布式四种组合只有一个构造入口；多 GPU 收尾写
  `final_state.pkl`；分布式 numba 线程按 `--threads` 设置。
- `--init-from` 从低阶解精确延拓；续算时决定物理算例的参数（来流、粘度、Tu/VR、人工粘性、近壁处理、合成湍流入口
  涡核数、是否时间精确）从 checkpoint 恢复，缺失即报错。
- 物理时间随时间精确推进累计并写进 checkpoint；求解器侧探针（`solve transient --probe`）每个物理步采样。

### 网格
- `.nas` 坐标单位（`--units mm/m/auto`）贯穿解析、生成、导入、求解与转换；求解读 `.nas` 体网格时转存米制 `volume_mesh.pkl`，
  不再静默修改用户网格。
- 边界类型按组名整词识别，识别不了或有歧义时用 `--bc-type` 显式指定；不再把未知组当成固壁或远场。
- 体网格生成输出做硬检查（退化单元、流形性、退化面、边界覆盖）；棱柱四边形对角线全局一致；碰撞检测与重叠检查的
  广相位完备。
- 新增 CGNS / STL 导出。

### 后处理
- 气动系数按组积分、表压、WMLES 壁面应力，力矩可给参考长度与力矩中心；参考面积只取物理无滑移固壁。
- 高阶 VTK 导出到 P3（四面体与三棱柱），含湍流量与 Q 准则。
- 压力谱改用 Welch 法、读探针时间序列；瞬态统计按物理时间取样；报告的收敛结论取求解器自己的 `solve_status`。

### 工程
- `utils doctor` 按 `pyproject.toml` 的依赖声明核对；`utils benchmark` 计时完整求解步。
- 用户文档按实现重写，删除描述旧实现或重复的文档。

---

## [2.0.0] - 2026-08-17

### ✨ 重大更新：V2.0 系统改造

V2.0 是一次全面的系统改造，重点修复了 V1.0 专家评审发现的全部问题，并实现了完整的工业级计算功能。

#### 数值算法核心
- ✅ FR 高阶离散格式（P1/P2/P3），含 Duffy 坍缩坐标四面体/棱柱映射
- ✅ AUSM+up 黎曼求解器（含低马赫数 Mp/pu 修正，保持反对称性）
- ✅ BR1 粘性界面耦合（真实边界幽灵态，温度梯度完整计算）
- ✅ 体积项去混叠（Over-integration：fine 几何 + 插值 + 限制回 coarse）
- ✅ 问题单元检测机制（残差异常抑制）

#### 时间积分
- ✅ SSP-RK2/RK3（Shu-Osher 形式，每 stage 重新计算残差）
- ✅ IMEX Euler（显式对流 + 隐式粘性，阻尼 Picard 子迭代）
- ✅ Dual-Time Stepping（BDF1/BDF2 + SSP-RK3 伪时间 + CFL 自适应）

#### 湍流模型体系
- ✅ SST k-ω RANS（F1/F2 混合函数标准 Menter 1994 公式、正性限制器）
- ✅ DDES 延迟分离涡模拟（屏蔽函数 + 有效长度尺度替换）
- ✅ WMLES 壁面模型大涡模拟（Spalding 律 + Newton-Raphson 迭代）
- ✅ WALE 亚格子应力模型

#### 网格与边界
- ✅ 原生 NAS 网格解析 + 自动体网格生成（BL extrusion + tetgen）
- ✅ 高阶网格初始化（Duffy 映射、解析雅可比、面通量点定位/合并）
- ✅ 幽灵态边界框架（WALL/FARFIELD/INLET/OUTLET/SYMMETRY）
- ✅ SEM 合成湍流入口（Cholesky 分解雷诺应力、涡核对流+再生）
- ✅ 壁面距离场（KD-Tree + Eikonal Dijkstra 近似）

#### 工程工作流
- ✅ CLI 完整命令体系（`grid`/`solve`/`post`/`config`/`utils`）
- ✅ 检查点机制（HDF5 存储完整状态，支持 `solve resume` 断点续算）
- ✅ 气动系数积分（直接在 FR 面通量点上积分压力+粘性力）
- ✅ Q-Criterion 涡识别准则（Green-Gauss 速度梯度重建）
- ✅ 力系数时间平均统计（Welford 在线算法）
- ✅ Order Continuation（P0→P1→...→目标阶数，残差下降触发判据）
- ✅ VTK 导出（legacy + XML VTU，含边界分区、Q-Criterion）
- ✅ CPU 性能优化（界面项 numba 化 4.38x 加速 + prange 多核并行 + 体积项 einsum→matmul/tensordot）
- ✅ CPU 性能优化续（湍流输运 numba 化 + 面图着色 + MPI 域分解基础设施）

### 🐛 重大修复

- 修复 AUSM+up 熵修正破坏反对称性（替换为 Mp/pu 低马赫数修正）
- 修复 SST F1/F2 混合函数公式倒置（改为标准 Menter 1994）
- 修复 IMEX 时间积分符号写反
- 修复壁面距离 Eikonal max_iter=500 导致 99%+ 节点为 inf
- 修复 `solve resume` 死路径（重建 FRSolver 并恢复完整状态）
- 修复气动力系数占位实现（改为直接在 FR 面通量点上积分）
- 修复粘性残差不施加边界条件（改为 BR1 + 幽灵态）
- 修复 SEM 合成湍流入口死代码（重写并接入 InletSEMGhostState）
- 修复 WMLES 壁面应力修正缺少 return 语句（T-05 功能完全失效）
- 修复 Order Continuation 触发条件（改为残差下降判据）

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

**最后更新**: 2026-08-17
