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

**职责**：实现FR离散格式、湍流模型和时间积分算法

**核心组件**：
- `discretization.py`：FR空间离散格式（1-3阶）
- `turbulence/`：湍流模型插件
  - `sst_kw.py`：SST k-ω模型
  - `des.py`：DES/DDES混合模型
  - `les_plugin.py`：LES插件接口（v2.0）
- `time_integration.py`：时间离散格式
  - Backward Euler（稳态）
  - Runge-Kutta（瞬态，2-3阶）
- `backend/`：计算后端抽象
  - `cpu_backend.py`：Numba并行实现
  - `gpu_backend.py`：CUDA kernel实现

**关键特性**：
- 统一的Backend接口，支持CPU/GPU无缝切换
- 插件化湍流模型，通过注册机制扩展
- 预分配内存池，避免频繁malloc/free

**输入**：GridData + SolverConfig  
**输出**：SolutionVector（流场变量）

### 3.3 边界条件模块（boundary/）

**职责**：管理边界条件的应用和更新

**核心组件**：
- `manager.py`：边界条件管理器
- `conditions.py`：内置BC类型
  - Inlet（速度入口）
  - Outlet（压力出口）
  - Wall（壁面，含壁面函数）
  - Symmetry（对称面）
  - Farfield（远场）
- `custom_bc.py`：自定义BC扩展接口

**关键特性**：
- 基于边界组名称自动匹配BC类型
- 支持用户自定义边界条件插件
- 壁面函数自动适配y+范围

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
- `aerodynamics.py`：气动系数计算（Cd、Cl、Cm）
- `vtk_export.py`：VTK场数据导出
- `convergence.py`：收敛曲线分析
- `transient_stats.py`：瞬态统计（时均场、RMS、频谱）

**关键特性**：
- 实时计算气动力系数
- 支持ParaView兼容的VTK格式
- 自动检测收敛并提供建议

**输入**：SolutionVector + 参考参数  
**输出**：JSON/CSV/VTK文件

### 3.6 CLI交互层（cli/）

**职责**：提供命令行接口，便于Agent工具化调用

**核心命令**：
```bash
autoflowcfd solve --grid <file> --mode <steady/transient> [options]
autoflowcfd postprocess --case <case_id> --output <format>
autoflowcfd validate-grid --grid <file>
autoflowcfd benchmark --grid <file> --backend <cpu/gpu>
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

### 6.2 计算优化

- **Numba JIT编译**：Python代码接近C性能
- **CUDA Kernel融合**：减少kernel launch开销
- **向量化操作**：充分利用SIMD指令

### 6.3 I/O优化

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

### 7.2 集群部署（v2.0规划）

```
+-------------------+     +-------------------+
|  Head Node        |     |  Worker Node 1    |
|  - Task Scheduler |---->|  - GPU Solver     |
|  - Result Aggreg. |     +-------------------+
+-------------------+     |  Worker Node 2    |
                          |  - GPU Solver     |
                          +-------------------+
```

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

**文档版本**：v0.1  
**最后更新**：2026-07-23
