# AutoFlowCFD 配置指南

本文档说明求解参数的三种给法（命令行、YAML 配置文件、Python API）、YAML 文件的全部字段，以及它们之间的优先级。
字段表与 `src/autoflowcfd/config/solver_config.py` 的 `SolverConfig` / `SteadyConfig` / `TransientConfig`
逐项对应；加载器 `config/loader.py::ConfigLoader` 对**未知字段直接报错**，所以本文档之外的写法都加载不了。

---

## 📋 目录

- [三种给法与优先级](#三种给法与优先级)
- [YAML 文件格式](#yaml-文件格式)
- [字段表](#字段表)
  - [公共字段](#公共字段)
  - [稳态字段（mode: steady）](#稳态字段mode-steady)
  - [瞬态字段（mode: transient）](#瞬态字段mode-transient)
- [不在配置文件里的设置](#不在配置文件里的设置)
- [配置示例](#配置示例)
- [生成、查看与校验](#生成查看与校验)
- [常见问题](#常见问题)

---

## 三种给法与优先级

| 给法 | 入口 | 适用 |
|---|---|---|
| 命令行 | `autoflowcfd solve steady/transient/resume ... --选项` | 日常运行 |
| YAML 文件 | `autoflowcfd solve steady/transient <网格> --config case.yaml` | 固定一套参数、纳入版本管理 |
| Python API | `AutoFlowCFDAPI().run_steady(volume_mesh, config=SteadyConfig(...))` / `run_transient(...)` | 程序化调用 |

命令行与 YAML 同时给出时（`cli/solve/config_file.py`）：

1. 命令行**显式写出**的选项；
2. YAML 里**显式写出**的字段；
3. 命令行选项的默认值。

三条入口的默认值是同一份：命令行选项的默认值就读配置类的字段默认值（`cli/solve/shared_options.py`），下表"默认"
一列对命令行、YAML 与 Python API 都成立。YAML 里出现当前命令没有对应参数的字段（例如把瞬态的 `dt` 写进稳态配置）
时报错，而不是静默忽略。

---

## YAML 文件格式

平铺的 `键: 值`，没有嵌套分组。第一行的 `mode` 决定按哪个配置类解析：

```yaml
mode: steady          # steady | transient；不写时为 steady
order: 2
turbulence: sa
max_iter: 3000
vel_inf: 40.0
```

枚举类字段写字符串（大小写不敏感）；`null` 表示"未给出"（例如 `cfl_init: null` = 按时间格式取 CFL 律的默认值）。

---

## 字段表

### 公共字段

| 字段 | 默认 | 对应命令行 | 说明 |
|---|---|---|---|
| `backend` | `cpu` | `--backend` | `cpu` / `gpu` / `auto`（有可用 GPU 时用 gpu；命令行只收 cpu/gpu） |
| `order` | `2` | `--order` | FR 目标阶数 0~3；>=1 时从 P0 逐阶爬坡（Order Continuation） |
| `turbulence` | `sst_kw`（瞬态 `ddes`） | `--turbulence-model` | `none` / `sst`（同 `sst_kw`）/ `sa` / `ddes` / `iddes` / `wmles` / `les` |
| `gpu_device` | `null` | `--gpu-device` | GPU 设备号；`null` = 单 GPU 用 0、多 GPU 按 rank % GPU 数分配 |
| `n_threads` | `-1` | `--threads` | CPU 线程数，`-1` = 运行时取本机核数 |
| `output_dir` | `./results`（瞬态 `./transient_results`） | `--output` | 结果目录 |
| `checkpoint_interval` | `100` | `--checkpoint-interval` | 中间 checkpoint 间隔（步） |
| `turbulence_intensity` | `0.01` | `--turbulence-intensity` | 来流湍流强度 Tu，(0, 1] |
| `viscosity_ratio` | `5.0` | `--viscosity-ratio` | 来流粘性比 nu_t/nu |
| `mu_molecular` | `1.8e-5` | `--mu-molecular` | 分子动力粘度 (Pa·s) |
| `artificial_viscosity_enabled` | `false` | `--artificial-viscosity` | 问题单元人工粘性（熵残差判据，P1 起有效） |
| `artificial_viscosity_alpha` | `1.0` | `--av-alpha` | 人工粘性强度标定常数 |
| `wall_treatment` | `auto` | `--wall-treatment` | RANS 近壁处理：`auto`（平衡壁面模型，第一层 y+ 从 1 到对数层都可用）/ `resolved`（要求 y+~1） |
| `phase_max_iter` | `null` | `--phase-max-iter` | Order Continuation 非最终阶段的步数上限；`null` = 按阶段数均分，目标阶数吃满剩余步数 |
| `residual_drop_threshold` | `100.0` | `--residual-drop-threshold` | 非最终阶段提前升阶所需的残差下降倍数 |

### 稳态字段（mode: steady）

| 字段 | 默认 | 对应命令行 | 说明 |
|---|---|---|---|
| `max_iter` | `1000` | `--max-iter` | 最大迭代步数 |
| `time_scheme` | `ssp_rk3` | `--time-scheme` | 稳态推进格式：`rk3`（显式）/ `newton-krylov`（隐式） |
| `cfl_init` / `cfl_max` / `cfl_min` | `null` | `--cfl-start` / `--cfl-max` / `--cfl-min` | 自适应 CFL 初值/上限/下限；`null` 按时间格式取默认（显式 0.03/0.06/0.01，隐式 SER 律起点 5） |
| `convergence_tol` | `1e-6` | `--tol` | 相对收敛容差（目标阶数的残差降到初值的 tol 倍） |
| `low_mach_precond` | `true` | `--low-mach-precond/--no-low-mach-precond` | 低马赫数伪时间预处理：不改变定常解，低马赫数下去掉伪时间推进里的声学刚性 |
| `rho_inf` / `vel_inf` / `p_inf` | `1.225` / `33.33` / `101325.0` | `--rho-inf` / `--vel-inf` / `--p-inf` | 来流密度、速度、静压：初场、远场/入口边界与气动系数归一化的唯一来源 |

### 瞬态字段（mode: transient）

| 字段 | 默认 | 对应命令行 | 说明 |
|---|---|---|---|
| `dt` | `1e-5` | `--dt` | 物理时间步长 (s) |
| `total_time` | `null` | `--physical-time` | 总物理时间 (s)；步数 = total_time/dt（非整数倍时向上取整并告警） |
| `max_iter` | `100` | `--max-iter` | `total_time` 为 `null` 时的物理时间步数 |
| `time_scheme` | `dual_time` | `--time-method` | `dual-time`（BDF + Newton-Krylov 内迭代）或显式格式（全场同一物理步长的时间精确推进） |
| `cfl_init` / `cfl_max` / `cfl_min` | `null` | `--cfl-start` / `--cfl-max` / `--cfl-min` | 显式：子步固定 CFL；dual-time：内迭代伪时间 CFL 的 SER 律参数 |
| `init_from_checkpoint` | `null` | `--init-from` | 以已有 checkpoint 的场为初场（阶数可低于 `order`，精确延拓） |
| `rho_inf` / `vel_inf` / `p_inf` | 同稳态 | 同稳态 | 同稳态 |

低马赫数预处理只用于稳态伪时间推进，瞬态没有这个字段（DUAL_TIME 内迭代的 BDF 项已让系统良态；显式时间精确推进
不能改变物理时间导数）。

---

## 不在配置文件里的设置

| 设置 | 给法 |
|---|---|
| 边界条件类型 | 生成/导入体网格时按边界组名识别，识别不了或有歧义的组用 `grid generate-volume/import-volume --bc-type 组名=类型` 指定（`WALL`/`SLIP_WALL`/`VELOCITY_INLET`/`PRESSURE_OUTLET`/`SYMMETRY`/`FARFIELD`），写进体网格文件 |
| 来流方向 | `solve steady/transient --aoa/--aos`（度） |
| 参考面积 | `--reference-area`，不给时按迎风投影自动估算 |
| 并行方式 | `--n-ranks`、`--multi-gpu`、`--fully-distributed` |
| 面网格（.nas 体网格反推边界分组用） | `--surface-mesh` |
| 跳过网格质量门 | `--skip-quality-check` |

运行环境变量只有两个，都不影响计算结果：`AFCFD_NUMBA_CACHE_ROOT`（numba 编译缓存目录）、
`AFCFD_NO_BLAS_THREAD_LIMIT`（不限制 BLAS 线程数）。

---

## 配置示例

### 稳态 RANS（SA，隐式）

```yaml
mode: steady
backend: cpu
order: 2
turbulence: sa
time_scheme: newton-krylov
max_iter: 400
convergence_tol: 1.0e-6
vel_inf: 40.0
wall_treatment: auto
output_dir: ./results/steady_sa
```

```bash
autoflowcfd solve steady volume_mesh.pkl --config steady_sa.yaml
```

### 瞬态 DDES（双时间步）

```yaml
mode: transient
backend: gpu
order: 2
turbulence: ddes
time_scheme: dual-time
dt: 1.0e-4
total_time: 0.2
init_from_checkpoint: ./results/steady_sa/checkpoints/checkpoint_iter_000400.h5
output_dir: ./results/ddes
```

```bash
autoflowcfd solve transient volume_mesh.pkl --config ddes.yaml
```

### 命令行覆盖配置文件

```bash
# YAML 写了 max_iter: 400，命令行显式给出的 --max-iter 优先
autoflowcfd solve steady volume_mesh.pkl --config steady_sa.yaml --max-iter 800
```

### Python API

```python
from autoflowcfd import AutoFlowCFDAPI
from autoflowcfd.config import SteadyConfig

api = AutoFlowCFDAPI()
volume_mesh = api.generate_volume_mesh(api.load_grid("car_surface.nas"))
config = SteadyConfig(order=2, turbulence="sa", time_scheme="newton-krylov", max_iter=400, vel_inf=40.0)
result = api.run_steady(volume_mesh, config=config)
```

配置对象在构造时校验（与 YAML 加载同一处）：非法取值直接抛 `ValueError`。`run_steady` 的显式参数优先于
`config` 里的同名字段。

---

## 生成、查看与校验

```bash
autoflowcfd config init --template steady -o steady.yaml       # 模板由配置类字段逐项生成，含全部字段与默认值
autoflowcfd config show steady.yaml                            # 解析后的完整配置（未写字段显示默认值）
autoflowcfd config validate steady.yaml                        # 只校验，不运行
```

---

## 常见问题

**续算（`solve resume`）读配置文件吗？** 不读。决定物理解的参数（来流、粘度、Tu/VR、人工粘性、近壁处理）与
时间格式、低马赫预处理开关都从 checkpoint 恢复；CFL、步数、容差等纯数值参数由 `resume` 的命令行选项重新给出，
`--time-scheme`、`--low-mach-precond/--no-low-mach-precond` 可覆盖 checkpoint 记录。

**CFL 该怎么设？** 一般不设（`null`）：显式与隐式格式的 CFL 律不同、默认值相差两个数量级，由
`core/time_integration/adaptive_cfl/policy.py` 按时间格式给出。

---

## 参考资源

- [API 参考](API.md)
- [快速开始](../QUICKSTART.md)

---

**最后更新**: 2026-10-19
