# AutoFlowCFD 开发者指南

给修改或扩展求解器的人：环境、代码组织、必须遵守的约定、测试方法，以及新增湍流模型 / 边界类型 / 后处理功能时
要改哪些地方。模块职责的总览见 [ARCHITECTURE.md](../ARCHITECTURE.md)，贡献流程见 [CONTRIBUTING.md](../CONTRIBUTING.md)。

---

## 环境

```bash
git clone https://github.com/AutoFlowCFD/AutoFlowCFD.git && cd AutoFlowCFD
poetry install -E gpu -E mpi -E perf      # 按需选可选组
poetry run autoflowcfd utils doctor
poetry run pytest tests/unit -q -o addopts=
```

没有 GPU 的机器也能跑 GPU 后端的单元测试：`tests/unit/_numpy_as_cupy.py` 用 numpy 充当 cupy（`_gpu_cupy_shim.py`
替换各 GPU 模块的 `get_cupy`）；多 rank 通信在单元测试里用单进程替身（`_thread_comm.py`、`_fake_halo.py`）。

---

## 代码组织

```
src/autoflowcfd/
├── fr/                  FR 算子（与求解器状态无关）：原生四面体 / 棱柱基、解点与面通量点、过积分、任意点插值
├── core/
│   ├── fr_solver/       CPU 单机求解器 FRSolver；SolveLoopMixin（四后端共用的求解循环）；边界幽灵态；收敛历史
│   ├── fr_residual/     无粘（AUSM+up）/ 粘性（BR1）残差与解析 Jacobian
│   ├── fr_operators/    numba 计算核、梯度、人工粘性
│   ├── time_integration/ 显式 RK、时间精确推进、隐式 Newton-Krylov（krylov / blocks / coarse / newton）、自适应 CFL、正性限制器
│   ├── turbulence/      SST、SA-neg、DES、LES/WMLES、壁面模型、输运与湍流 Jacobian；registry.py 是模型族的唯一定义
│   ├── gpu/             单 GPU（solver/）与多 GPU（distributed/）求解器及其核
│   ├── mpi/             分区、halo、分布式求解器、完全分布式加载、分布式 checkpoint
│   ├── probes/          求解器侧探针
│   └── utils/           checkpoint 读写与元数据、Order Continuation、壁距、面着色
├── grid/                .nas 解析（nas_io）、体网格生成（mesh_gen）、高阶网格（high_order）、连接关系、质量校验、格式转换
├── boundary/            幽灵态（fr_ghost_state.py）、合成湍流入口（synthetic_inlet.py）、固壁判定（wall_kind.py）
├── postprocess/         气动系数、VTK（低阶 / 高阶）、Q 准则、y+、瞬态统计、压力谱、报告
├── cli/                 grid / solve / post / config / utils；solve/run.py 是按后端执行的唯一流程
├── api/                 Python API（与 CLI 共用构造与写出函数）
└── config/              配置类（字段即 YAML 键）与加载器
```

四个后端的关系：`FRSolver`（CPU）、`GPUFRSolver`、`DistributedFRSolver`（CPU MPI）、`MultiGPUDistributedSolver` 构造参数
同名，都继承 `SolveLoopMixin`，后端差异只经少数钩子进入（`step()`、`host_view()`、`pull_host_state()` /
`push_host_state()`、`_interpolate_to_new_order()` 等，见 `core/fr_solver/solver/solve_loop.py` 模块文档）。命令行的构造入口：
单机 `cli/solve/solver_factory.py::build_single_node_solver`，分布式 `cli/solve/distributed_build.py::build_distributed_solver`。

---

## 约定（评审按这些检查）

- **同一语义只有一个事实来源**：常量、默认值、判据、枚举只定义一处，其余地方去读它（例如湍流模型族在
  `turbulence/registry.py`、边界类型在 `grid/nas_io/boundary_types.py`、单位在 `grid/nas_io/units.py`、命令行默认值读配置类）。
  发现两份时合并、删掉多余的那份。
- **不静默兜底**：缺输入、缺键、读不到的元数据一律报错并说明原因，不用缺省值继续算（`dict.get(键, 缺省)` 只用于
  真正可选的量）；决定物理算例的参数写进 checkpoint，续算缺失即报错（`cli/solve/checkpoint_io/rebuild.py`）。
- **四后端一致**：改一个后端的算法时，其余三个同步，并有跨后端对照测试。
- **不简化**：调试中不为让测试通过而降低算法（去掉项、放宽判据、换低阶）；找到根因再改。
- **单文件不超过 500 行**，超过按功能拆成子包。
- **注释与文档字符串用中文**（术语、公式、代码标识符除外）；写"为什么"，开发史只在必要处一句话说明此前的缺陷。
- **numba 核**：`@njit(cache=True)`；缓存目录按全部核相关源文件的内容哈希区分（`_numba_cache.py`），改动被内联的
  核后无需手动清缓存。验证改动时用空的 `AFCFD_NUMBA_CACHE_ROOT` 再跑一遍，确认不是旧机器码。
- **输出目录不放在项目树里**：测试用 `tmp_path`，手工试算放到仓库之外。

---

## 测试

| 目录 | 内容 |
|------|------|
| `tests/unit/` | 单元测试（约 4000 项）；GPU 用 numpy 替身，分布式用单进程替身或 `n_ranks=1` |
| `tests/validation/` | 有精确解 / 关联式 / 守恒量可定量核对的算例（Blasius、Couette、湍流平板、等熵涡、TGV……），较慢 |

```bash
poetry run pytest tests/unit -q -o addopts=                    # 全量单元测试（-o addopts= 去掉缺省的覆盖率统计）
poetry run pytest tests/unit/test_solver_probes.py -q -o addopts=
poetry run pytest tests/validation -q -o addopts=
```

修缺陷的测试要先在修改前的代码上失败（fail-before），修改后通过；行为测试优先于"源码里出现某字符串"式的文本检查。

---

## 扩展

### 新增湍流模型

带输运方程的模型实现 `core/turbulence/transported.py::TransportedTurbulence` 接口（输运场名、Newton 未知量映射、
未知量尺度、显式更新与限幅、来流值与上界），参照 `turbulence/sa/`（一个未知量）与 `turbulence/sst/`（两个未知量）。
要动的地方：

1. `turbulence/registry.py`：加入模型族（是否输运、是否需要壁距、是否代数亚格子、是否用合成湍流入口）。
2. 模型本体与源项、点隐式线性化：`turbulence/<model>/`；湍流解析 Jacobian：`turbulence/jacobian/`。
3. 构造与初值：`core/fr_solver/turbulence/init.py`（四个后端共用）；输出键：`turbulence/output.py`。
4. GPU：`core/gpu/turbulence/`；分布式：`core/mpi/distributed_turbulence.py` 与完全分布式打包
   （`core/mpi/distributed_mesh_loader/`）。
5. 命令行 `--turbulence-model` 的取值与配置类 `TurbulenceModel`（`config/`）。
6. 测试：源项对照解析式、Jacobian 对照差分、四后端一致、checkpoint 往返与续算。

### 新增边界类型

1. `grid/nas_io/boundary_types.py`：`SOLVER_BC_TYPES` 与组名关键字。
2. `core/fr_solver/boundary/ghost.py` 的类型表（幽灵态配置）与 `boundary/fr_ghost_state.py` 的幽灵态函数（numpy / cupy
   通用）。
3. 粘性与湍流边界：`core/fr_residual/viscous_flux/`、`core/turbulence/transport/` 里按类型分派的部分；是否计入气动力
   与固壁判定：`boundary/wall_kind.py`。
4. 体网格生成里是否挤出边界层（`grid/mesh_gen/`）。

### 新增后处理功能

实现放在 `postprocess/`，读求解器的主机视图（`solver.host_view()`，GPU 状态拷回主机）或 checkpoint；命令行在
`cli/post/`（命令重建求解器用 `cli/solve/checkpoint_io.rebuild_solver_from_checkpoint`），API 方法在 `api/`。

---

## 调试

- 每步日志：残差、CFL、Newton 诊断、第 1 步与每 10 步的逐方程缩放残差与最大残差位置、累计伪时间。
- 残差非有限时立即中止并报出位置，不会写坏 checkpoint。
- 从 checkpoint 复现：`rebuild_solver_from_checkpoint(path)` 得到状态完整的求解器，可直接调用残差函数检查。
- 日志用 loguru；`logger.opt(exception=True)` 记录异常栈。

---

## 提交

- 一个提交只做一件事；提交信息首行 `V2 - <中文概述>（问题编号）`，正文写清缺陷现象与根因、改了什么、测试
  （含修改前失败的核对）。
- 改动涉及命令行选项、配置字段、输出文件或 checkpoint 格式时，同步更新 `docs/` 与 `--help`；checkpoint 新增必需键时
  说明旧 checkpoint 的处理。
