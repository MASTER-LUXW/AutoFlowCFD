# AutoFlowCFD Python API 参考

`autoflowcfd.AutoFlowCFDAPI` 提供网格处理、FR 求解与后处理的 Python 接口。它与命令行走同一套实现：

- 体网格生成：`grid generate-volume`
- 求解前质量门：`solve steady/transient`
- 求解器构造
- checkpoint 续算：`solve resume`
- 气动系数与 VTK 导出

参数名就是配置类的字段名，默认值与校验只在配置类里，字段表见 [配置指南](CONFIGURATION_GUIDE.md)。

---

## 📋 目录

- [快速开始](#快速开始)
- [网格](#网格)
- [配置](#配置)
- [求解](#求解)
- [续算](#续算)
- [后处理](#后处理)
- [异常](#异常)
- [包级函数](#包级函数)

---

## 快速开始

```python
from autoflowcfd import AutoFlowCFDAPI

api = AutoFlowCFDAPI()

surface = api.load_grid("car_surface.nas", units="mm")     # NAS 面网格（三角形）
volume_mesh = api.generate_volume_mesh(surface, bl_layers=6)

result = api.run_steady(volume_mesh, order=2, turbulence="sa", time_scheme="newton-krylov", vel_inf=40.0)
print(result.converged, result.iterations, result.final_residual)

coeffs = api.calculate_coefficients()                       # 参考面积缺省按来流方向的迎风投影估算
print(f"Cd = {coeffs['Cd']:.4f}")

api.export_vtk("result.vtu", high_order=True)               # ParaView 打开
```

---

## 网格

| 方法 | 说明 |
|---|---|
| `load_grid(grid_file, units="mm", bc_types=None, encoding="UTF-8", validate=True)` | 解析 NAS 面网格，返回 `GridData`。`units`：mm / m / auto（同 `--units`，内部一律米）；`bc_types`：`{组名: 类型}` 覆盖按组名识别的结果（同 `--bc-type`）；`validate` 时跑 `GridValidator`（与 `grid validate` 同一实现），不通过只记警告 |
| `validate_grid(grid_data)` | 面网格质量校验结果字典：`passed`、`summary`、长宽比/扭曲度/Jacobian 统计 |
| `get_grid_info(grid_data)` | 节点数、单元数、边界组数（`boundary_count`）与组名（`boundary_groups`） |
| `generate_volume_mesh(grid_data, *, first_layer_height=None, target_yplus=None, ref_velocity=None, ref_length=None, rho_inf=1.225, mu_molecular=1.8e-5, **kwargs)` | 由面网格生成体网格，返回 `VolumeMeshData`，与 `grid generate-volume` 同一入口（见下） |
| `validate_volume_mesh(volume_mesh)` | 体网格质量报告（`MeshQualityReport`，即求解前质量门的判据）：`.passed`、`.summary()` |
| `get_volume_mesh_info(volume_mesh)` | 节点数、单元数（`cell_count`）及其中四面体数（`tet_count`）与棱柱数（`prism_count`） |

`generate_volume_mesh` 的参数：
- 第一层高度直接给出（`first_layer_height`，米）或由目标 y+ 推出（`target_yplus` + `ref_velocity` + `ref_length`），二选一，与 CLI 同一个函数。
- 其余参数为 `growth_rate`、`min_cell_size`、`max_cell_size`、`bl_layers`、`bl_only`、`core_only`。
- 含义与 `grid generate-volume` 的同名选项一致。
- 传入未知参数时报 `TypeError`。

边界条件类型按面网格的属性名识别；名字识别不了的组在 `load_grid(..., bc_types={...})` 里指定（同 CLI `--bc-type`）。

---

## 配置

| 方法 | 说明 |
|---|---|
| `create_steady_config(**fields)` | `SteadyConfig(**fields)` |
| `create_transient_config(**fields)` | `TransientConfig(**fields)` |
| `load_config(config_file)` | 从 YAML 加载并校验，返回 `SteadyConfig` 或 `TransientConfig`（按 `mode`），与 CLI `--config` 同一个加载器 |

配置对象在构造时校验。非法取值（阶数越界、未知湍流模型、稳态里写了 `dual-time` 等）直接抛 `ValueError`；未知字段名抛 `TypeError`。

```python
from autoflowcfd.config import SteadyConfig

config = SteadyConfig(order=3, turbulence="sa", time_scheme="newton-krylov", max_iter=400)
```

---

## 求解

```python
run_steady(volume_mesh, config=None, *, aoa_deg=0.0, aos_deg=0.0, reference_area=None,
           skip_quality_check=False, **fields) -> SolverResult
run_transient(volume_mesh, config=None, *, aoa_deg=0.0, aos_deg=0.0, reference_area=None,
              skip_quality_check=False, probes=None, **fields) -> SolverResult
```

| 参数 | 说明 |
|---|---|
| `volume_mesh` | 体网格（`generate_volume_mesh` 的输出；面网格不能直接求解） |
| `config` | `SteadyConfig` / `TransientConfig`；不给时取配置类默认值 |
| `**fields` | 覆盖 `config` 的字段，名字同 YAML：`order`、`turbulence`、`time_scheme`、`max_iter`、`convergence_tol`、`vel_inf`、`wall_treatment`、`dt`、`total_time` 等。不会改动传入的 `config` 对象 |
| `aoa_deg` / `aos_deg` | 攻角 / 侧滑角（度），约定同 CLI `--aoa/--aos` |
| `reference_area` | 气动系数参考面积 (m²)，不给时按来流方向的迎风投影估算 |
| `skip_quality_check` | 跳过求解前质量门（同 CLI `--skip-quality-check`，仅用于临时诊断） |
| `probes` | 仅瞬态：探针位置 `(k, 3)`（米，同 CLI `--probe`），每个物理步在探针处求 FR 解多项式的值，写进 `output_dir/probes.csv` |

**求解行为**
- 求解前跑与 CLI 同一道质量门；不通过时抛 `MeshQualityGateError`。
- 后端为 CPU 或单 GPU（`backend` 字段）；多 rank 并行用命令行（`--n-ranks`、`--multi-gpu`）。
- 求解器和体网格会存到 `api.solver` / `api.volume_mesh`，后处理方法作用于它们。

**稳态**
- 最多迭代 `max_iter` 步。
- 目标阶数的残差相对初值下降到 `convergence_tol` 倍时收敛。
- 目标阶数 ≥1 时从 P0 逐阶爬坡。

**瞬态**
- 物理时间步数为 `config.n_steps`：给了 `total_time` 时取 `total_time/dt`，否则为 `max_iter`。
- 跑满全部物理步，不以残差下降为停止条件。
- 给了 `init_from_checkpoint` 时，以该 checkpoint 的场为初场。
- 物理时间随时间精确推进累计，写进 checkpoint（`post transient-*` 的时间轴）。

`SolverResult` 字段：`converged`、`iterations`、`final_residual`。解场在 `api.solver` 上。

---

## 续算

```python
resume_simulation(checkpoint_file, max_iter=0, *, dt=None, tol=1e-6, backend=None, surface_mesh=None,
                  threads=-1, gpu_device=None, skip_quality_check=False, cfl_start=None, cfl_max=None,
                  cfl_min=None, time_scheme=None, low_mach_precond=None, phase_max_iter=None,
                  residual_drop_threshold=100.0) -> SolverResult
```

与 CLI `solve resume` 的单机分支同一套重建逻辑。

- **从 checkpoint 恢复：** 来流、粘度、湍流参数、人工粘性、近壁处理、时间格式、低马赫预处理开关。
- **`dt`：** 不给时沿用 checkpoint 记录的物理时间步长。
- **dual-time 续算：** 跑满 `max_iter` 步，不按残差停止。
- **`max_iter=0`：** 只恢复状态，求解器与体网格存入 `api.solver` / `api.volume_mesh`，可以直接导出或计算气动系数。
- **`time_scheme`：** 只在稳态格式（`rk3` / `newton-krylov`）之间切换。

---

## 后处理

| 方法 | 说明 |
|---|---|
| `calculate_coefficients(reference_area=None, reference_length=1.0, moment_center=None, groups=None)` | 当前解的气动力 / 力矩系数字典 `Cd/Cl/Cs/Cm/Cy/Cr`。`reference_area` 缺省时按来流方向的迎风投影估算；力矩系数报在体轴系，`moment_center` 缺省为原点；`groups` 为参与积分的 WALL 组，缺省为全部 |
| `export_vtk(filename, high_order=False, fields=None)` | 导出当前解。低阶：单元平均，`.vtk` 写 legacy、`.vtu` 写 XML；`high_order=True`：按 FR 分段多项式导出拉格朗日高阶单元（只 `.vtu`）。`fields` 缺省为速度与压力，可选 `velocity`、`pressure`、`turbulence`、`k`、`omega`、`nu_tilde`、`nut`、`q_criterion`（高阶另可选 `density`） |
| `get_convergence_history()` | `{"iterations": [...], "residuals": [...]}`；还没有求解时两个列表为空 |

上述方法都需要先调用 `run_steady`、`run_transient` 或 `resume_simulation`，否则抛 `ValueError`。

---

## 异常

| 异常 | 场景 |
|---|---|
| `ValueError` | 配置取值非法；后处理前没有求解器；无法自动估算参考面积 |
| `TypeError` | 配置字段名或体网格生成参数不存在；`config` 类型与求解入口不符 |
| `autoflowcfd.grid.validation.solve_gate.MeshQualityGateError` | 体网格未通过求解前质量门，或构造高阶几何时检测到畸变单元（是 `ValueError` 的子类） |
| `FileNotFoundError` | checkpoint 或配置文件不存在 |

---

## 包级函数

| 函数 | 说明 |
|---|---|
| `autoflowcfd.create_api(verbose=False)` | 返回 `AutoFlowCFDAPI` 实例 |
| `autoflowcfd.get_version()` | 版本号 |
| `autoflowcfd.get_info()` | 版本、作者、许可证等 |

`AutoFlowCFDAPI` 实例上的方法：
- `get_version()`：版本号。
- `check_environment()`：平台、可用后端、CPU 核数、Python 版本。

---

**最后更新**: 2026-10-21
