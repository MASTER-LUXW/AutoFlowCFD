# AutoFlowCFD 快速开始

从安装到跑通第一个算例：面网格 -> 体网格 -> 稳态 / 瞬态求解 -> 后处理。文中命令都可以原样运行（以仓库自带的
`examples/cube_demo` 立方体绕流为例）；每条命令的全部选项见 `autoflowcfd <命令组> <命令> --help`。

---

## 目录

- [前置要求](#前置要求)
- [安装](#安装)
- [第一个算例：命令行](#第一个算例命令行)
- [配置文件](#配置文件)
- [Python API](#python-api)
- [输出文件](#输出文件)
- [结果可视化与后处理](#结果可视化与后处理)
- [常见问题](#常见问题)
- [下一步](#下一步)

---

## 前置要求

| 软件 | 版本 | 说明 |
|------|------|------|
| Python | 3.10 或 3.11 | scipy 的依赖声明只覆盖 Python < 3.12（`pyproject.toml`） |
| Poetry | 1.6+ | 依赖管理 |
| CUDA 12.x + NVIDIA 驱动 | 可选 | 单 GPU / 多 GPU 后端（CuPy `cupy-cuda12x`） |
| MPI（OpenMPI / MS-MPI / MPICH） | 可选 | `--n-ranks > 1` 的分布式求解（mpi4py + pymetis） |

网格：ANSA 导出的 Nastran `.nas`（`CTRIA3` 面网格按 `PSHELL` 分组，或 `CTETRA` / `CPENTA` 体网格）。仓库自带三个算例：

| 目录 | 内容 |
|------|------|
| `examples/cube_demo/` | `cube_demo.nas`（圆角立方体 + 风洞的面网格，4 组：inlet / outlet / tunnel / body，坐标为毫米） |
| `examples/ahmed_demo/` | `ahmed_body_demo.nas`（Ahmed 体 + 地面 + 风洞的面网格，5 组），`config_example.yaml`（配置文件示例） |
| `examples/plate_demo/` | `plate_demo.nas`（平板面网格） |

---

## 安装

```bash
git clone https://github.com/AutoFlowCFD/AutoFlowCFD.git
cd AutoFlowCFD

poetry install                 # CPU
poetry install -E gpu          # 另装 GPU 后端（CuPy）
poetry install -E mpi          # 另装分布式求解（mpi4py、pymetis）
```

验证：

```bash
poetry run autoflowcfd --version          # AutoFlowCFD, version 0.1.0
poetry run autoflowcfd utils doctor       # 逐项核对 pyproject 声明的依赖与版本、GPU / MPI 是否可用
```

---

## 第一个算例：命令行

### 1. 面网格 -> 体网格

```bash
poetry run autoflowcfd grid info examples/cube_demo/cube_demo.nas
poetry run autoflowcfd grid generate-volume examples/cube_demo/cube_demo.nas \
    -o cube/cube_volume.nas
```

`.nas` 的缺省单位是毫米（`--units mm`），米制文件加 `--units m`，`grid info` 打印的包围盒（米）可用来核对。
`generate-volume` 生成边界层棱柱 + 四面体混合网格，导出为体网格 `.nas`（与输入同一单位），并打印质量报告。常用选项：
`--bl-layers`（边界层层数）、`--first-layer-height`（第一层高度，米）或 `--target-yplus 1 --ref-velocity 40
--ref-length 1`（按目标 y+ 推出第一层高度）、`--max-cell-size`、`--bc-type 组名=类型`（组名识别不了时指定边界类型）。

### 2. 稳态求解

```bash
poetry run autoflowcfd solve steady cube/cube_volume.nas --surface-mesh examples/cube_demo/cube_demo.nas \
    --order 1 --turbulence-model sa --time-scheme newton-krylov --vel-inf 40 --max-iter 200 -o cube/steady
```

- 输入是体网格 `.nas` 时必须给 `--surface-mesh`（反推边界分组）与 `--units`；入口处把它转存为输出目录下的米制
  `volume_mesh.pkl`，之后续算与后处理都用它（也可以直接把 `.pkl` 作为输入）。
- `--order`：FR 目标阶数 0~3；>=1 时从 P0 逐阶爬坡（Order Continuation）。
- `--time-scheme`：`rk3`（显式）或 `newton-krylov`（隐式，收敛步数由非线性程度决定，复杂网格推荐）。
- `--turbulence-model`：`none / sst / sa / ddes / iddes / wmles / les`；RANS 的近壁处理 `--wall-treatment auto`（缺省，
  平衡壁面模型，第一层 y+ 从 1 到对数层都可用）或 `resolved`（要求 y+ ~ 1）。
- 来流：`--vel-inf --rho-inf --p-inf --mu-molecular`，攻角 / 侧滑角 `--aoa --aos`（度）。
- `--reference-area`：气动系数参考面积 (m²)；不给时沿来流方向由物体面（物理无滑移固壁）的迎风投影自动估算。
- 收敛判据 `--tol`（残差相对初值下降的倍数，缺省 1e-6）；`--checkpoint-interval` 步写一次中间 checkpoint。
- GPU：`--backend gpu`；分布式：`mpirun -np 4 autoflowcfd solve steady ... --n-ranks 4`（多 GPU 另加
  `--backend gpu --multi-gpu`，只让 root 加载完整网格加 `--fully-distributed`）。

### 3. 续算

```bash
poetry run autoflowcfd solve resume cube/steady/checkpoints/checkpoint_iter_000200.h5 --max-iter 300
```

决定物理算例的参数（来流、粘度、湍流模型、近壁处理……）全部取自 checkpoint；CFL 等数值参数可重新给。缺省接着写进
checkpoint 所在算例目录，`--output` 可改到别处。

### 4. 瞬态求解

```bash
poetry run autoflowcfd solve transient cube/steady/volume_mesh.pkl --order 1 --turbulence-model ddes \
    --time-method dual-time --dt 1e-4 --physical-time 0.05 --vel-inf 40 \
    --init-from cube/steady/checkpoints/checkpoint_iter_000200.h5 \
    --probe 2.0 0.0 0.0 --probe 3.0 0.5 0.0 -o cube/transient
```

- `--time-method`：`dual-time`（缺省，每个物理步若干个带 BDF 物理时间项的隐式 Newton 步）或 `rk3`（显式，全场同一
  物理步长的时间精确推进）。
- `--init-from`：以稳态解为初场（阶数可低于 `--order`，精确延拓）。
- `--probe X Y Z`（米，可重复）：每个物理步在该点求 FR 解多项式的值，写进 `probes.csv`，供 `post transient-psd`。
- ddes / iddes / les / wmles 在速度入口注入合成湍流（`--sem-num-eddies` 指定涡核数，缺省按密度判据取）。

---

## 配置文件

YAML 是平铺的 `键: 值`，字段名与配置类一致，未知字段直接报错（全部字段见
[配置指南](docs/CONFIGURATION_GUIDE.md)）：

```bash
poetry run autoflowcfd config init --template steady -o case.yaml   # 生成模板
poetry run autoflowcfd config validate case.yaml                     # 校验
poetry run autoflowcfd solve steady cube/steady/volume_mesh.pkl --config case.yaml
```

```yaml
mode: steady
order: 2
turbulence: sa
time_scheme: newton_krylov
max_iter: 3000
vel_inf: 40.0
checkpoint_interval: 100
output_dir: ./results
```

命令行显式给出的选项优先于文件，文件里写出的字段优先于命令行缺省值。边界条件类型来自网格的组名（或
`grid generate-volume --bc-type`），不在配置文件里设置。

---

## Python API

```python
from autoflowcfd import AutoFlowCFDAPI

api = AutoFlowCFDAPI()
surface = api.load_grid("examples/cube_demo/cube_demo.nas", units="mm")
volume_mesh = api.generate_volume_mesh(surface)

# 参数名就是配置文件字段（docs/CONFIGURATION_GUIDE.md）；求解前跑与 CLI 同一道网格质量门
result = api.run_steady(volume_mesh, order=1, turbulence="sa", time_scheme="newton-krylov", vel_inf=40.0,
                        max_iter=200, output_dir="./cube_api")
print(result.converged, result.iterations, result.final_residual)

coeffs = api.calculate_coefficients()                 # 参考面积缺省按迎风投影估算
print(coeffs["Cd"], coeffs["Cl"], coeffs["Cs"])
api.export_vtk("./cube_api/flow.vtu")
```

各方法的参数以 `help(AutoFlowCFDAPI.run_steady)` 等为准，完整说明见 [API 文档](docs/API.md)。

---

## 输出文件

| 文件 | 何时写出 | 内容 |
|------|---------|------|
| `volume_mesh.pkl` | 输入是 `.nas` 体网格时 | 米制体网格（含边界分组），续算与后处理的网格来源 |
| `checkpoints/checkpoint_iter_XXXXXX.h5` | 每 `--checkpoint-interval` 步与结束时 | 全部解点状态、湍流场、收敛历史、物理与求解参数（续算 / 后处理用） |
| `final_state.pkl` | 结束时 | 解点状态（守恒量 U、原始变量 Q） |
| `probes.csv` / `probes.json` | 瞬态给了 `--probe` 时 | 每个物理步的探针 p、u、v、w；探针坐标与所在单元 |

求解不写 VTK、CSV 或报告，这些由下面的后处理命令从 checkpoint 生成。

---

## 结果可视化与后处理

```bash
# 流场 VTK（ParaView 打开）；--variables 可重复给出：velocity / pressure / turbulence / q_criterion / k / omega ...
poetry run autoflowcfd post export-vtk --case cube/steady -o cube/steady/flow.vtu
poetry run autoflowcfd post export-vtk --case cube/steady --high-order -o cube/steady/flow_ho.vtu   # FR 多项式高阶单元
poetry run autoflowcfd post export-vtk --case cube/steady --boundaries-only -o cube/steady/walls.vtu

# 气动系数（按组积分壁面压力与粘性力；力矩需给参考长度与力矩中心）
poetry run autoflowcfd post coefficients --case cube/steady --reference-area 1.0 \
    --reference-length 1.0 --moment-center 0 0 0

# 收敛曲线（PNG）与报告（JSON）
poetry run autoflowcfd post convergence --case cube/steady --variables residual --variables cfl -o conv.png
poetry run autoflowcfd post report --case cube/steady -o report.json

# 瞬态：时均 / 脉动均方根（VTK）、探针压力谱（CSV）
poetry run autoflowcfd post transient-mean --case cube/transient --warmup-time 0.01
poetry run autoflowcfd post transient-rms --case cube/transient --warmup-time 0.01
poetry run autoflowcfd post transient-psd --case cube/transient --warmup-time 0.01
```

详见 [后处理指南](docs/POSTPROCESSING_GUIDE.md) 与 [VTK 导出指南](docs/VTK_EXPORT_GUIDE.md)。

---

## 常见问题

**依赖装不上 / 版本冲突** —— `poetry run autoflowcfd utils doctor` 逐项列出缺失或版本不符的依赖；必需依赖缺失为问题，
可选组（gpu / mpi / perf）缺失为警告。

**`--backend gpu` 报 CuPy 不可用** —— `poetry install -E gpu`，再用 `python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"`
确认能看到设备。

**分布式报"需要 MPI"** —— `poetry install -E mpi`，并以 `mpirun -np <N> autoflowcfd solve ... --n-ranks <N>` 启动
（两个数必须一致）。

**网格读不进来 / 单位不对** —— `grid info` 打印包围盒（米），据此确认 `--units`；`grid validate` 检查网格质量。
`.nas` 缺省按毫米读入。

**求解前被质量门拦下** —— 质量报告指出畸变单元的位置；用 `grid generate-volume` 重新生成或 `grid import-volume`
修复后再算。`--skip-quality-check` 只用于临时诊断。

**不收敛** —— 先看每步日志里的逐方程缩放残差与最大残差位置；显式格式可调低 `--cfl-start`，复杂网格改用
`--time-scheme newton-krylov`；从低阶收敛解 `--init-from` 起步再升阶。

**内存不够** —— 内存随阶数快速增长（P2 约为 P1 的数倍）；先在 P1 收敛，或用 `--n-ranks` 分布式分摊到多台机器。

---

## 下一步

- [文档索引](docs/INDEX.md)
- [教程](docs/TUTORIALS.md)
- [配置指南](docs/CONFIGURATION_GUIDE.md)
- [API 文档](docs/API.md)
- [架构说明](ARCHITECTURE.md)
- 问题反馈：[GitHub Issues](https://github.com/AutoFlowCFD/AutoFlowCFD/issues)；联系人 Mr Lu（luxw_chd@126.com）
