# AutoFlowCFD

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)](https://www.python.org/)

**面向汽车外流场的高阶开源 CFD 求解器**：通量重构（FR）高阶离散，Python 顶层 + Numba / CuPy 计算核，读取 ANSA
`.nas` 网格，CPU、单 GPU、CPU-MPI、多 GPU 四个后端共用同一套算法。

- 快速上手：[QUICKSTART.md](QUICKSTART.md)
- 文档索引：[docs/INDEX.md](docs/INDEX.md)

---

## 功能

### 数值方法
- **FR 高阶离散**：P0 ~ P3；四面体与三棱柱都用原生基（Warp & Blend 解点），直边混合网格；体积项过积分去混叠；
  离散守恒（投影与迹并进体积算子）。
- **无粘通量**：AUSM+up 黎曼求解器；低马赫伪时间预处理（只改伪时间推进、不改定常解）。
- **粘性通量**：BR1 界面耦合（界面处原始变量与梯度取两侧外插的平均）加内罚项；固壁绝热，壁面温度梯度按边界类型
  做法向镜像。
- **湍流模型**：SST k-ω、SA-neg（为高阶离散设计的负值鲁棒 Spalart-Allmaras）、SST-DDES / IDDES、LES（WALE）、
  WMLES（平衡壁面模型）；RANS 近壁处理 `auto`（Spalding 平衡壁面模型，第一层 y+ 从 1 到对数层）或 `resolved`；
  LES/DES 类模型在速度入口注入合成湍流（SEM）。
- **时间推进**：
  - 稳态：显式 SSP-RK3（局部时间步）或隐式 Newton-Krylov（矩阵自由 JFNK、GMRES、单元块 ILU 与多层预处理、
    伪瞬态延拓与自适应 CFL）；
  - 瞬态：双时间步（BDF2 物理时间 + 隐式 Newton 内迭代）或显式时间精确推进（全场同一物理步长）；
  - Order Continuation（P0 -> 目标阶数逐阶爬坡）、`--init-from` 从低阶解精确延拓起步、守恒的正性限制器。

### 计算后端
- **CPU**：Numba `prange` 并行核，界面项按面着色累加。
- **单 GPU**：CuPy。
- **CPU MPI**：METIS 分区 + halo 交换；传统模式（每个 rank 加载完整网格）或完全分布式加载（只有 root 加载）。
- **多 GPU**：每个 rank 一块 GPU，同样支持两种加载方式。
- 四个后端走同一个求解循环、同一份 checkpoint 格式，续算可在后端之间切换。

### 网格
- 解析 ANSA `.nas`（`CTRIA3` 面网格按 `PSHELL` 分组；`CTETRA` / `CPENTA` 体网格），坐标单位 mm / m / auto。
- 体网格生成：边界层棱柱挤出（可按目标 y+ 定第一层高度）+ tetgen 核心区四面体；导入外部体网格并修复。
- 求解前网格质量门；转换为 VTK / CGNS / STL。

### 后处理
- 气动系数：壁面压力与粘性力按组积分，力矩可给参考长度与力矩中心；壁面 y+ 统计。
- VTK 导出：体网格或命名边界面片，legacy `.vtk` / XML `.vtu`，可导出 FR 多项式的高阶 Lagrange 单元；Q 准则。
- 瞬态：时均与脉动均方根；求解器每个物理步记录探针时间序列，Welch 法压力谱。
- 收敛曲线、JSON 报告；HDF5 checkpoint 续算（决定物理算例的参数随 checkpoint 恢复）。

---

## 安装与第一个算例

```bash
git clone https://github.com/AutoFlowCFD/AutoFlowCFD.git
cd AutoFlowCFD
poetry install                 # 另有 -E gpu（CuPy）、-E mpi（mpi4py、pymetis）、-E perf（psutil）
poetry run autoflowcfd utils doctor

# 面网格 -> 体网格 -> 稳态求解 -> 后处理（examples/cube_demo 的坐标为毫米，即 .nas 的缺省单位）
poetry run autoflowcfd grid generate-volume examples/cube_demo/cube_demo.nas -o cube/cube_volume.nas
poetry run autoflowcfd solve steady cube/cube_volume.nas --surface-mesh examples/cube_demo/cube_demo.nas \
    --order 1 --turbulence-model sa --time-scheme newton-krylov --vel-inf 40 --max-iter 200 -o cube/steady
poetry run autoflowcfd post coefficients --case cube/steady --reference-area 1.0
poetry run autoflowcfd post export-vtk --case cube/steady -o cube/steady/flow.vtu

# 分布式与 GPU
mpirun -np 8 poetry run autoflowcfd solve steady cube/steady/volume_mesh.pkl --n-ranks 8 -o cube/steady_mpi
poetry run autoflowcfd solve steady cube/steady/volume_mesh.pkl --backend gpu -o cube/steady_gpu
mpirun -np 4 poetry run autoflowcfd solve steady cube/steady/volume_mesh.pkl --backend gpu --multi-gpu --n-ranks 4
```

Python API 与配置文件见 [QUICKSTART.md](QUICKSTART.md)、[docs/API.md](docs/API.md)、
[docs/CONFIGURATION_GUIDE.md](docs/CONFIGURATION_GUIDE.md)。

---

## 源码结构

```
src/autoflowcfd/
├── fr/              # FR 算子：原生四面体 / 棱柱基与解点、面通量点、过积分、任意点插值
├── core/
│   ├── fr_solver/       # 单机求解器（CPU）、求解循环（四后端共用）、边界幽灵态、收敛历史
│   ├── fr_residual/     # 无粘 / 粘性残差
│   ├── fr_operators/    # 计算核、人工粘性、问题单元判据
│   ├── time_integration/ # 显式 / 隐式（Newton-Krylov、预处理）/ 时间精确推进、自适应 CFL、正性限制器
│   ├── turbulence/      # SST、SA-neg、DES、LES/WMLES、壁面模型、湍流输运
│   ├── gpu/             # 单 GPU 与多 GPU 求解器
│   ├── mpi/             # 分区、halo、分布式求解器与 checkpoint
│   ├── probes/          # 求解器侧探针
│   └── utils/           # checkpoint、Order Continuation、壁距等
├── grid/            # .nas 解析、体网格生成、高阶网格、连接关系、质量校验、格式转换
├── boundary/        # 幽灵态边界与合成湍流入口
├── postprocess/     # 气动系数、VTK 导出、Q 准则、瞬态统计、压力谱、报告
├── cli/             # 命令行（grid / solve / post / config / utils）
├── api/             # Python API
└── config/          # 配置类与 YAML 加载
tests/               # unit / integration / validation
examples/            # cube_demo、ahmed_demo、plate_demo
```

更详细的模块说明见 [ARCHITECTURE.md](ARCHITECTURE.md)。

---

## 文档

- [QUICKSTART.md](QUICKSTART.md)：安装与第一个算例
- [INSTALL.md](INSTALL.md)：安装细节
- [docs/INDEX.md](docs/INDEX.md)：全部文档索引（教程、API、配置、后处理、VTK、边界条件、开发者指南）
- [CONTRIBUTING.md](CONTRIBUTING.md)：贡献流程与代码规范
- [CHANGELOG.md](CHANGELOG.md)、[ROADMAP.md](ROADMAP.md)

---

## 许可证

Apache License 2.0，见 [LICENSE](LICENSE)。

## 致谢

依赖的开源项目：NumPy、SciPy、Numba、CuPy、mpi4py、PyMetis、TetGen（tetgen Python 封装）、PyVista / VTK、h5py、
Click、Loguru。网格前处理使用 ANSA。

## 联系

- 问题反馈：[GitHub Issues](https://github.com/AutoFlowCFD/AutoFlowCFD/issues)
- 项目联系人：Mr Lu（luxw_chd@126.com）
