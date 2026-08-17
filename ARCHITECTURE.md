# 系统架构文档

## 1. 概述

AutoFlowCFD是一款专注于汽车外流场仿真的开源CFD软件，采用模块化设计，支持CPU/GPU异构计算。本文档描述系统的整体架构、模块划分和技术选型。

### 1.1 设计目标

- **高性能**：通过Numba CPU并行和CUDA GPU加速实现快速求解
- **高精度**：采用FR（Flux Reconstruction）高阶格式，支持1-3阶精度
- **易用性**：提供CLI命令行和Python API双接口
- **可扩展**：插件化架构，易于新增湍流模型和后处理功能
- **AI友好**：便于Agent工具化集成和自动化流水线

### 1.2 技术栈

|层级|技术选型|说明|
|---|---|---|
|语言|Python 3.10+|顶层业务逻辑|
|数值计算|NumPy/CuPy|CPU/GPU数组计算|
|并行加速|Numba/CUDA|CPU多线程/GPU kernel|
|CLI框架|Click|命令行接口|
|配置管理|PyYAML|YAML配置文件|
|数据序列化|HDF5/h5py|检查点存储|
|可视化|VTK/pyvista|场数据导出|
|日志|loguru|结构化日志|
|测试|pytest|单元测试框架|

---

## 2. 系统架构图

```
+-----------------------------------------------------------------------+
|                        User Interface Layer                           |
+---------------------------+-------------------------------------------+
|   CLI (Click)             |   Python API                              |
|   autoflowcfd solve       |   from autoflowcfd import Solver          |
|   autoflowcfd postprocess |   solver.run()                            |
+------------+--------------+------------------+------------------------+
             |                                 |
             v                                 v
+-----------------------------------------------------------------------+
|                      Application Core Layer                           |
+---------------------------+-------------------------------------------+
|   Config Manager          |   Grid Parser      |   Postprocessor      |
|   - YAML parsing          |   - NAS reader     |   - Cd calculation   |
|   - Validation            |   - Quality check  |   - VTK export       |
|   - Defaults              |   - BC mapping     |   - Convergence      |
+------------+--------------+---------+----------+---------+------------+
             |                          |                    |
             v                          v                    v
+-----------------------------------------------------------------------+
|                       Solver Engine Layer                             |
+---------------------------+-------------------------------------------+
|   FR Discretization       |   Turbulence Models  |   Time Integration |
|   - 1st/2nd/3rd order     |   - SST k-omega      |   - Backward Euler |
|   - Riemann solver        |   - DES/DDES         |   - Runge-Kutta    |
|   - Boundary treatment    |   - LES (plugin)     |                    |
+------------+----------------------------------------------------------+
             |
             v
+-----------------------------------------------------------------------+
|                     Compute Backend Layer                             |
+---------------------------+-------------------------------------------+
|   CPU Backend (Numba)     |   GPU Backend (CUDA)                      |
|   - Multi-threading       |   - Kernel functions                      |
|   - Vectorization         |   - Memory management                     |
|   - Cache optimization    |   - Stream processing                     |
+------------+--------------+------------------+------------------------+
             |                                 |
             v                                 v
+-----------------------------------------------------------------------+
|                         Data Storage Layer                            |
+---------------------------+-------------------------------------------+
|   SoA Memory Layout       |   HDF5 Checkpoints   |   Output Files     |
|   - NodeArray             |   - Solution state   |   - JSON (Cd)      |
|   - CellArray             |   - Restart support  |   - CSV (history)  |
|   - BoundaryMap           |   - Cross-backend    |   - VTK (fields)   |
+-----------------------------------------------------------------------+
```

---

## 3. 模块详细说明

### 3.1 网格解析模块（grid/）

**职责**：解析ANSA生成的.nas网格文件，构建内存数据结构

**核心组件**：
- `parser.py`：NAS文件解析器，支持v22/v23/v24格式
- `structures.py`：网格数据结构（GridData、NodeArray、CellArray、BoundaryMap）
- `validator.py`：网格质量校验器（长宽比、扭曲度、雅可比行列式）

**关键特性**：
- SoA（Structure of Arrays）内存布局，优化缓存命中率
- 流式解析大文件（>1GB），避免内存溢出
- 自动识别边界条件组并映射为标准BC类型

**输入**：ANSA .nas文件  
**输出**：GridData对象

### 3.2 求解器核心模块（core/）

**职责**：实现 FR 离散格式、黎曼求解器、粘性格式、湍流模型和时间积分算法

**核心组件**：
- `fr_solver.py`：FR 求解器主类（状态初始化、算子预计算、求解主循环）
- `fr_residual_inviscid.py`：无粘残差（体积项 over-integration 去混叠 + 界面项 AUSM+up）
- `fr_viscous_flux.py`：粘性残差（BR1 界面耦合 + 真实边界幽灵态）
- `fr_kernels.py`：AUSM+up 黎曼求解器（含低马赫数 Mp/pu 修正）
- `turbulence_sst.py`：SST k-ω 模型（F1/F2 混合函数、正性限制器、DES 长度尺度替换）
- `turbulence_transport_kernel.py`：湍流输运 Numba kernel（标量场外插 + 校正量分配）
- `face_coloring.py`：面图着色（消除 scatter-add 写冲突，替代 per-thread buffer）
- `turbulence_des.py`：DDES 延迟分离涡模拟（屏蔽函数 + 有效长度尺度）
- `turbulence_wmles.py`：WMLES 壁面模型（Spalding 律 + Newton-Raphson 迭代）
- `turbulence_sgs.py`：WALE 亚格子应力模型
- `time_integration.py`：SSP-RK2/RK3（Shu-Osher 形式）
- `time_integration_imex.py`：IMEX Euler（显式对流 + 隐式粘性）
- `time_integration_dual.py`：Dual-Time Stepping（BDF1/BDF2 + SSP-RK3 + CFL 自适应）
- `order_continuation.py`：Order Continuation 策略（P0→P1→...→目标阶数）
- `wall_distance.py`：壁面距离场（KD-Tree + Eikonal Dijkstra）
- `mpi/`：MPI 域分解并行模块
  - `__init__.py`：MPI 可选依赖检测（mpi4py 不可用时优雅降级）
  - `comm.py`：通信封装（Allreduce/Allgather/Isend/Irecv + MPITimer）
  - `partition.py`：METIS 网格分区 + DistributedPartition 数据结构
  - `halo.py`：Halo 层管理与非阻塞数据交换
  - `distributed_state.py`：分布式 FRState（local + halo 扩展数组）
  - `distributed_flat_face.py`：分布式面几何（面分类 + 扩展索引）
  - `distributed_solver.py`：分布式 FRSolver（组合模式）
- `backend/`：计算后端抽象
  - `fr_gpu_p0.py`：P0 有限体积 GPU kernel（P≥1 阶仍运行 CPU）

**关键特性**：
- 统一的求解器框架，支持稳态/瞬态无缝切换
- 插件化湍流模型，通过注册机制扩展
- Numba `prange` 多核并行，界面项/体积项全并行
- 面图着色消除 scatter-add 冲突，内存从 O(n_threads × N) 降至 O(N)
- MPI 域分解支持跨节点 HPC 集群扩展（METIS 分区 + Halo 交换）
- 问题单元检测机制（残差异常抑制）

**输入**：GridData + SolverConfig  
**输出**：SolutionVector（流场变量）

### 3.3 边界条件模块（boundary/）

**职责**：实现幽灵态边界条件框架和合成湍流入口

**核心组件**：
- `fr_ghost_state.py`：幽灵态边界条件框架
  - WALL：镜像构造（无滑移/滑移）
  - FARFIELD/INLET/OUTLET：特征边界条件
  - SYMMETRY：对称面镜像
- `synthetic_inlet.py`：SEM 合成湍流入口（Cholesky 分解雷诺应力、涡核对流+再生）
- `manager.py`：边界条件管理器
- `config.py`：边界配置解析

**关键特性**：
- 基于边界组名称自动匹配幽灵态构造方式
- InletSEMGhostState：SEM 与幽灵态的粘合层
- 壁面无滑移通过镜像构造实现，无需壁面函数

**输入**：BoundaryMap + 边界参数  
**输出**：边界通量修正

### 3.4 配置管理模块（config/）

**职责**：解析和验证求解器配置

**核心组件**：
- `solver_config.py`：SolverConfig数据类
- `yaml_parser.py`：YAML配置文件解析
- `validator.py`：配置合法性校验

**配置示例**：
```yaml
solver:
  mode: steady          # steady / transient
  turbulence: sst_kw    # sst_kw / des / ddes
  order: 2              # FR格式阶数 1/2/3
  backend: cpu          # cpu / gpu
  
mesh:
  file: car_model.nas
  scale: 1.0            # 几何缩放因子
  
numerics:
  max_iterations: 10000
  convergence_tol: 1e-6
  cfl: 1.0              # CFL数
  
output:
  format: vtk
  interval: 100         # 输出间隔
```

### 3.5 后处理模块（postprocess/）

**职责**：计算气动系数、导出可视化数据、分析收敛性

**核心组件**：
- `fr_coefficients.py`：FR 原生气动系数计算（直接在面通量点上积分压力+粘性力）
- `vtk_export.py`：VTK 场数据导出（legacy + XML VTU，含边界分区）
- `q_criterion.py`：Q-Criterion 涡识别准则（Green-Gauss 速度梯度重建）
- `transient_stats.py`：瞬态统计（时均场、RMS、力系数时间平均）
- `report.py`：收敛曲线分析

**关键特性**：
- 直接在 FR 求解器原生数据上积分，不经过单元中心近似
- 支持 Q-Criterion 涡识别准则导出
- 力系数时间平均统计（Welford 在线算法）
- 同时输出 CELL_DATA（原始值）和 POINT_DATA（节点插值）

**输入**：SolutionVector + 参考参数  
**输出**：JSON/CSV/VTK文件

### 3.6 CLI交互层（cli/）

**职责**：提供命令行接口，便于Agent工具化调用

**核心命令**：
```bash
# 网格处理
autoflowcfd grid generate-volume <surface.nas> -o <volume.nas>
autoflowcfd grid import-volume <volume.nas>

# 求解
autoflowcfd solve steady <volume.pkl> --order 2 --turbulence-model sst
autoflowcfd solve transient <volume.pkl> --time-method dual-time --physical-time 0.1
autoflowcfd solve resume <checkpoint.h5>

# 后处理
autoflowcfd post export-vtk --case <case_dir> --variables velocity pressure q_criterion
autoflowcfd post coefficients --case <case_dir>
autoflowcfd post report --case <case_dir>
```

**关键特性**：
- Click框架实现，自动生成帮助文档
- 结构化JSON输出，便于程序解析
- 统一的退出码体系（0成功，非0失败）

### 3.7 工具模块（utils/）

**职责**：提供通用工具函数

**核心组件**：
- `logger.py`：loguru日志配置
- `exceptions.py`：自定义异常层次结构
- `performance.py`：性能计时和基准测试
- `io_helpers.py`：文件I/O辅助函数

---

## 4. 数据流图

### 4.1 稳态求解流程

```
[NAS File] 
    |
    v
+-------------+
| Grid Parser |----> GridData (SoA layout)
+-------------+         |
                        v
                  +------------+
                  | Validator  |----> Quality Report
                  +------------+         |
                                         v
                                   +------------+
                                   | Config     |----> SolverConfig
                                   +------------+         |
                                                          v
                                                    +------------+
                                                    | FR Solver  |
                                                    | (CPU/GPU)  |
                                                    +------------+
                                                          |
                                                          v
                                                  +---------------+
                                                  | SolutionVector|
                                                  +---------------+
                                                          |
                                                          v
                                                  +---------------+
                                                  | Postprocessor |
                                                  +---------------+
                                                          |
                                          +---------------+---------------+
                                          v               v               v
                                      [Cd.json]      [history.csv]    [fields.vtk]
```

### 4.2 瞬态求解流程

```
[NAS File] --> [Grid Parser] --> [GridData]
                                      |
                                      v
                               [Time Loop Start]
                                      |
                                      v
                              +----------------+
                              | Time Step n    |
                              | - FR Update    |
                              | - BC Apply     |
                              | - Turbulence   |
                              +----------------+
                                      |
                                      v
                              [Convergence Check]
                                /            \
                          Not Converged    Converged
                              |                |
                              v                v
                         [Next Step]    [Sample Data]
                                              |
                                              v
                                       [Time Loop End]
                                              |
                                              v
                                      [Transient Stats]
                                              |
                                    +---------+---------+
                                    v         v         v
                                [Mean.vtk] [RMS.vtk] [PSD.csv]
```

---

## 5. 扩展机制

### 5.1 湍流模型插件

新增湍流模型只需实现统一接口并注册：

```python
from autoflowcfd.core.turbulence import register_turbulence_model

@register_turbulence_model("my_model")
class MyTurbulenceModel:
    def __init__(self, config):
        pass
    
    def compute_source_terms(self, solution):
        # 实现源项计算
        pass
```

### 5.2 自定义边界条件

```python
from autoflowcfd.boundary import register_boundary_condition

@register_boundary_condition("CUSTOM_INLET")
class CustomInletBC:
    def apply(self, boundary_nodes, time):
        # 实现自定义边界逻辑
        pass
```

### 5.3 后处理插件

```python
from autoflowcfd.postprocess import register_postprocessor

@register_postprocessor("force_breakdown")
class ForceBreakdownAnalyzer:
    def analyze(self, solution, grid):
        # 实现力分解分析
        pass
```

---

## 6. 性能优化策略

### 6.1 内存优化

- **SoA布局**：提升CPU缓存命中率
- **预分配内存池**：避免运行时频繁分配
- **零拷贝传输**：CPU-GPU数据传输优化
- **面图着色**：消除 scatter-add 的 per-thread buffer，内存从 O(n_threads × N) 降至 O(N)

### 6.2 计算优化

- **Numba JIT编译**：Python代码接近C性能（界面项、体积项、湍流输运全 kernel 化）
- **CUDA Kernel融合**：减少kernel launch开销
- **向量化操作**：充分利用SIMD指令
- **prange 多核并行**：界面 kernel 采用 per-thread buffer + sum 归约

### 6.3 MPI 并行计算

- **METIS 网格分区**：最小化分区间切割边数（= 跨 rank halo 交换量）
- **非阻塞通信**：Isend/Irecv 异步数据交换，可与体积项计算重叠
- **预分配 buffer**：Halo 交换管理器预分配固定大小 send/recv buffer
- **两级并行**：MPI 跨节点 + Numba 多线程，总并行度 = n_ranks × n_threads

### 6.4 I/O优化

- **异步写入**：不阻塞主计算线程
- **压缩存储**：HDF5内置压缩
- **增量输出**：仅输出变化数据

---

## 7. 部署架构

### 7.1 单机部署

```
+---------------------+
|  AutoFlowCFD App    |
|  + CLI Interface    |
|  + Solver Engine    |
|  + Postprocessor    |
+----------+----------+
           |
           v
+---------------------+
|  Local Filesystem   |
|  - Input grids      |
|  - Output results   |
+---------------------+
```

### 7.2 集群部署（MPI 域分解）

```
+-------------------+     +-------------------+
|  Head Node        |     |  Worker Node 1    |
|  - mpirun 启动    |---->|  - MPI Rank 0-3   |
|  - 结果收集       |     |  - 4 threads/rank |
+-------------------+     +-------------------+
                          |  Worker Node 2    |
                          |  - MPI Rank 4-7   |
                          |  - 4 threads/rank |
                          +-------------------+

使用方式:
  mpirun -np 8 autoflowcfd solve steady <grid_file> --n-ranks 8
```

**进程模型**：
- 每个 MPI rank 是独立进程，内部用 Numba 多线程
- 总并行度 = n_ranks × n_threads_per_rank
- 典型配置：4 nodes × 16 ranks/node × 4 threads = 256 总并行度

---

## 8. 安全考虑

- **输入验证**：严格校验网格文件格式和内容
- **沙箱执行**：在隔离环境中运行用户提供的网格
- **资源限制**：限制最大内存使用和计算时间
- **依赖审计**：定期扫描第三方库漏洞

详见 [SECURITY.md](SECURITY.md)

---

## 9. 相关文档

- [需求规格说明书](ProjectFiles/2-1_需求规格说明书-Part1.md)
- [数据结构设计](ProjectFiles/2-3_数据结构设计文档-Part1.md)
- [接口文档](ProjectFiles/2-4_接口文档-Part1.md)
- [部署文档](ProjectFiles/2-5_部署文档-Part1.md)
- [编码规范](ProjectFiles/2-7_编码规范-Part1.md)

---

**文档版本**：v2.0  
**最后更新**：2026-08-17（HPC 并行计算优化更新）
