# 变更日志

本文档记录 AutoFlowCFD 的所有重要变更。

格式遵循 [Keep a Changelog](https://keepachangelog.com/) 规范。

---

## [未发布]

### 计划中

- 气动噪声模块（FW-H 声类比）
- Docker 容器化部署
- AI Agent 集成示例（参数优化流水线）

### 已实现（GPU 大规模并行计算）

- ✅ GPU 加速基础设施（`core/gpu/`）
  - `GPUArrayManager`: GPU 内存管理、设备选择、数据上传/下载
  - GPU 可用性检测统一为 CuPy（移除 Numba CUDA 依赖）
- ✅ P0 无粘残差 CuPy RawKernel（`gpu_p0_inviscid.py`）
  - 从 numba.cuda 迁移到 CuPy，AUSM+up CUDA C kernel + atomicAdd
  - GPU 常驻版本（避免 CPU↔GPU 传输）
- ✅ 高阶 FR 体积项 GPU 化（`gpu_volume_contract.py` + `gpu_flux.py` + `gpu_inviscid.py`）
  - CuPy 张量收缩（底层 cuBLAS gemm）
  - 欧拉/粘性物理通量 CuPy 向量化
  - Over-integration 去混叠 GPU 路径
- ✅ 高阶 FR 界面项 GPU 化（`gpu_face_geometry.py` + `gpu_inviscid.py`）
  - GPU 版面几何缓存（面图着色直接复用）
  - AUSM+up 批量通量计算 + 图着色逐色校正分配
- ✅ 粘性残差 + 物理梯度 GPU 化（`gpu_viscous.py` + `gpu_gradients.py`）
  - BR1 粘性耦合、应力张量、热传导 GPU 实现
  - 物理空间梯度（参考空间梯度 + 链式法则）
- ✅ GPU 时间积分（`gpu_time_integration.py`）
  - SSP-RK2/RK3 Shu-Osher stage GPU 推进
  - GPU 正定性强制（rho>0, p>0）
  - GPU 局部 CFL 步长计算
- ✅ GPU FRSolver（`gpu_solver.py`）
  - 完整 GPU 求解器（solve/step/compute_*_residual）
  - 数据常驻 GPU，只在 I/O 时传输
- ✅ 多 GPU + MPI 分布式求解器（`gpu_distributed.py`）
  - 每个 MPI rank 绑定一块 GPU
  - Halo 交换 + GPU 残差计算 + 全局归约
- ✅ GPU 直接 Halo 交换（`gpu_halo_exchange.py`）
  - CUDA-aware MPI 零拷贝模式（GPU buffer 直接通信）
  - Staging buffer 优化模式（只传输 send/recv 列表数据）
  - 自动检测 CUDA-aware MPI 支持
- ✅ 分布式 GPU SSP-RK 时间推进
  - SSP-RK2/RK3 多 stage 时间积分（每 stage 重新 halo 交换 + 残差评估）
- ✅ GPU 版 DUAL_TIME / IMEX_EULER 时间推进
  - `gpu_time_integration_dual.py`: BDF1/BDF2 隐式时间离散 + 伪时间迭代
  - `gpu_time_integration_imex.py`: 阻尼 Picard 子迭代求解隐式方程
  - CFL 自适应、模态滤波、增广伪残差含物理时间导数项
- ✅ GPU Checkpoint 功能
  - `gpu_solver.py` 新增 `save_checkpoint()` / `load_checkpoint()` 方法
  - `gpu_distributed.py` 新增 `save_checkpoint_distributed()` / `load_checkpoint_distributed()` 方法
  - 保存 U/Q 场、湍流场、残差历史、DUAL_TIME 历史到 HDF5 文件
  - 正定性强制在每个 stage 完成后执行
- ✅ 湍流模型源项 GPU 化（`gpu_turbulence_sst.py`）
  - SST k-ω 完整源项计算全程 GPU
  - Blending functions F1/F2、涡粘系数、源项 Sk/S_omega
  - GPU 正性保持限制器
- ✅ CLI 集成（`--backend gpu` + `--gpu-device` + `--multi-gpu`）
- ✅ GPU 基准测试套件（`benchmarks/benchmark_gpu.py`）

### 已实现（HPC 并行计算优化）

- ✅ 分布式残差计算完整接入（`core/mpi/distributed_compute.py`）
  - `DistributedMeshAdapter`: 将分布式数据包装为 mesh 接口，复用现有残差函数
  - 分布式无粘/粘性/梯度/湍流输运残差计算
  - `DistributedFRSolver.step()` / `solve()` 时间推进循环
  - CLI `--n-ranks` 选项实际触发分布式路径
- ✅ 分布式 Checkpoint 保存/加载 + 结果保存（`core/mpi/distributed_checkpoint.py`）
  - Root rank 收集全局数据后保存为单文件（与单机格式兼容）
  - 支持变 rank 数恢复（4 ranks 保存 → 8 ranks 恢复）
- ✅ 分区优化（Root rank 执行 METIS，广播结果到其他 rank，非 root rank 不再运行分区算法）
- ✅ 完全分布式网格加载（`core/mpi/distributed_mesh_loader.py`）
  - 只有 root rank 加载完整网格，通过 MPI 分发各 rank 的局部数据
  - 非 root rank 不再持有完整网格，内存占用降为 1/n_ranks
- ✅ 湍流输运 Numba 化（`turbulence_transport_kernel.py`，消除 SST k-ω 输运方程串行瓶颈）
- ✅ 面图着色算法 + kernel 完整接入（所有界面 kernel 均支持图着色方案）
  - 无粘界面 kernel：`compute_inviscid_interface_correction_kernel_colored`
  - 粘性界面 kernel：`compute_viscous_interface_correction_kernel_colored`
  - 湍流输运 kernel：`distribute_corrections_to_cells_kernel_colored`
  - 内存从 O(n_threads × N) 降至 O(N)
  - 环境变量 `AFCFD_USE_COLORING` 控制（默认启用）
- ✅ 面图着色缓存（着色结果在 `FlatFaceGeometry` 构建时一次性计算并缓存，避免每次残差调用重复着色）
- ✅ MPI 域分解基础设施（`core/mpi/`，6 模块 ~1200 行）
  - METIS 网格分区（pymetis 接口，不可用时降级为 block 分区）
  - Halo 层管理与非阻塞数据交换（Isend/Irecv + 预分配 buffer）
  - 分布式状态 / 面几何 / 求解器骨架
  - CLI 新增 `--n-ranks`/`--np` 选项
- ✅ 完整 SST k-ω 输运方程（对流+扩散，FR 高阶离散）

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
