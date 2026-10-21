# AutoFlowCFD 架构

本文说明各模块的职责、一次求解的数据流，以及贯穿全仓库的几个设计约束。目录树与扩展步骤见
[docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md)；算法细节以各模块的文档字符串为准。

---

## 1. 分层

```
命令行 cli/            Python API api/            配置 config/（字段即 YAML 键，命令行默认值读它）
        \                  |                       /
         `----- cli/solve/run.py（SolveRun -> 构造 -> 求解 -> 落盘）-----'
                           |
         求解器（四个后端，同一个求解循环 SolveLoopMixin）
           FRSolver（CPU） GPUFRSolver（单 GPU） DistributedFRSolver（CPU MPI） MultiGPUDistributedSolver
                           |
   时间推进 core/time_integration   残差 core/fr_residual + core/fr_operators   湍流 core/turbulence
                           |
   FR 算子 fr/（基、解点、面通量点）     高阶网格 grid/high_order     边界 core/fr_solver/boundary + boundary/
                           |
   网格 grid/（.nas 解析、体网格生成、质量校验、连接关系）     后处理 postprocess/     checkpoint core/utils
```

---

## 2. 一次求解的数据流

1. **网格输入**（`cli/solve/mesh_loader.py`）：`.pkl` 体网格直接读；`.nas` 体网格按 `--units` 与 `--surface-mesh` 导入，
   转存为输出目录下的米制 `volume_mesh.pkl`。求解前质量门（`grid/validation`）拦截畸形网格。
2. **高阶网格**（`grid/high_order`）：按阶数生成解点物理坐标、度量（雅可比及其逆、细点度量）、面连接关系与面通量点
   几何；直边四面体 / 三棱柱，原生基。
3. **构造求解器**：单机 `cli/solve/solver_factory.py`，分布式 `cli/solve/distributed_build.py`（METIS 分区、halo、
   传统模式或完全分布式加载）。来流、湍流模型、近壁处理、时间格式、CFL 策略（`adaptive_cfl/policy.py`）、低马赫预处理
   在构造时确定；边界幽灵态提供者按网格的组与类型构造（`core/fr_solver/boundary/ghost.py`）。
4. **初场**：均匀来流，或 `--init-from` 从 checkpoint 精确延拓；`solve resume` 从 checkpoint 恢复全部状态与物理参数。
5. **求解循环**（`core/fr_solver/solver/solve_loop.py`）：目标阶数 >= 1 且从来流起步时走 Order Continuation
   （`core/utils/order_continuation/run.py`，P0 起逐阶升阶），否则定阶循环。每步 `advance()`：
   - 稳态显式：局部时间步 SSP-RK3；稳态隐式：一个 Newton-Krylov 步（`time_integration/implicit/newton/`：矩阵自由
     GMRES、单元块 ILU 与多层粗校正预处理、伪时间步与线搜索、平均流与湍流紧耦合）；
   - 瞬态 dual-time：每个物理步若干个带 BDF2 项的 Newton 内迭代（`newton/physical_time.py`）；瞬态显式：全场同一
     物理步长的子步（`time_integration/time_accurate.py`），物理时间随之累计。
   - 残差：无粘（AUSM+up，体积项过积分）+ 粘性（BR1）+ 湍流源项与输运；守恒的正性限制器。
   - 每步记录收敛历史、可选的力系数与探针（`convergence_history.record_step`），按间隔写 checkpoint。
6. **落盘**：`final_state.pkl`、结束时的 checkpoint、气动系数与壁面 y+ 报告；分布式由 root 汇总写出。
7. **后处理**（`cli/post`、`postprocess/`）：从 checkpoint 读取或重建求解器，计算系数、导出 VTK、统计与频谱。

---

## 3. 模块职责

| 模块 | 职责 |
|------|------|
| `fr/` | 与求解器状态无关的 FR 离散：原生四面体（Warp & Blend 节点、单纯形模态基）与原生三棱柱（三角形 × 线段）基、解点与面通量点、过积分阶数、阶数间延拓、任意参考点插值 |
| `grid/nas_io` | `.nas` 解析与导出、单位、边界类型识别与显式指定 |
| `grid/mesh_gen` | 体网格生成：边界层棱柱挤出（碰撞冻结、厚度限制、塌缩降级）、tetgen 核心区填充、合并与修复、面提取 |
| `grid/high_order`、`grid/connectivity` | 高阶网格与度量、面连接关系（含周期面配对） |
| `grid/validation` | 面网格与体网格质量校验、求解前质量门、退化单元判定 |
| `core/fr_solver` | CPU 求解器、求解循环、单步推进、边界幽灵态、CFL 与局部时间步、收敛历史、伪时间预算诊断 |
| `core/fr_residual`、`core/fr_operators` | 无粘 / 粘性残差（含 P0 专用核）与解析 Jacobian、numba 计算核、梯度、人工粘性 |
| `core/time_integration` | 显式 RK、时间精确推进、隐式 Newton-Krylov 全套（Krylov、块预处理、粗校正、全局化、物理性限幅）、自适应 CFL、正性限制器、残差范数 |
| `core/turbulence` | 模型族定义（`registry.py`）、SST k-ln(omega)、SA-neg、DDES / IDDES、WALE / WMLES、平衡壁面模型与近壁处理、湍流输运与湍流 Jacobian、输运模型统一接口（`transported.py`） |
| `core/gpu` | 单 GPU 与多 GPU 求解器、CuPy 核、GPU halo 交换 |
| `core/mpi` | 分区、halo、分布式状态与面几何、分布式求解器、完全分布式加载与换阶重新分发、分布式 checkpoint、全局归约 |
| `core/utils` | checkpoint 读写与元数据（物理参数、时间状态、求解状态）、Order Continuation、壁距、面着色 |
| `core/probes` | 探针定位、每步采样与时间序列文件 |
| `boundary/` | 幽灵态函数（numpy / cupy 通用）、合成湍流入口（SEM）、物理无滑移固壁判定 |
| `postprocess/` | 气动系数（按组、风轴系 / 体轴系）、VTK（低阶 / 高阶）、Q 准则、y+、瞬态统计、Welch 压力谱、报告 |
| `cli/` | 命令行；`cli/solve/run.py` 是按后端执行的唯一流程，`distributed_build.py` 是分布式构造的唯一入口 |
| `api/` | Python API，与命令行共用构造参数映射与写出函数 |
| `config/` | 配置类与 YAML 加载器（未知字段报错） |

---

## 4. 设计约束

- **一个事实来源**：默认值、判据、模型族、边界类型、单位、插值矩阵等只定义一处；命令行默认值读配置类，四个后端共用
  一个求解循环、一个 checkpoint 格式。
- **四后端同一算法**：后端差异只经少数钩子进入求解循环（`step`、`host_view`、`pull_host_state` / `push_host_state`、
  换阶），跨后端对照测试保证一致。
- **不静默兜底**：缺失的输入与元数据报错而不是取缺省；决定物理算例的参数随 checkpoint 持久化，续算原样恢复。
- **checkpoint 是唯一的结果载体**：解点状态、湍流场、收敛历史、物理与求解参数、时间状态、探针坐标都在其中；后处理与
  续算只依赖 checkpoint 与 `volume_mesh.pkl`。
- **数值缓存可信**：numba 磁盘缓存按全部核源码的内容哈希分目录，改动被内联的核后不会继续执行旧机器码。
