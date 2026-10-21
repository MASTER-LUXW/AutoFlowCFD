# AutoFlowCFD 后处理指南

后处理全部从算例目录（`solve` 的 `-o` 目录）读取：`checkpoints/*.h5`（解点状态、湍流场、收敛历史、物理参数）、
`volume_mesh.pkl`（体网格），瞬态给了探针时另有 `probes.csv` / `probes.json`。命令行入口是 `autoflowcfd post`，
Python 入口是 `AutoFlowCFDAPI` 的后处理方法与 `autoflowcfd.postprocess` 模块。

---

## 目录

- [命令一览](#命令一览)
- [气动系数](#气动系数)
- [VTK 导出](#vtk-导出)
- [收敛曲线与报告](#收敛曲线与报告)
- [瞬态统计与压力谱](#瞬态统计与压力谱)
- [Python 接口](#python-接口)

---

## 命令一览

| 命令 | 输出 | 数据来源 |
|------|------|---------|
| `post coefficients` | 终端 + JSON（`-o`，缺省 coefficients.json） | 指定或最新的 checkpoint，在 CPU 上重建求解器 |
| `post export-vtk` | `.vtk` / `.vtu` | checkpoint（低阶取单元积分平均；`--high-order` 重建求解器取解点多项式） |
| `post convergence` | PNG | checkpoint 的收敛历史 |
| `post report` | JSON | checkpoint 的收敛历史与求解参数 |
| `post transient-mean` / `transient-rms` | `.vtk`（节点场） | 算例目录的全部 checkpoint（按物理时间取样） |
| `post transient-psd` | CSV | `probes.csv`（每个物理步一个样本） |

`--checkpoint` 指定 checkpoint，缺省取 `checkpoints/` 下迭代数最大的一个；`--grid` 指定体网格，缺省取算例目录的
`volume_mesh.pkl`，没有时取 checkpoint 记录的求解输入。原求解靠 `--skip-quality-check` 才跑得起来的网格，需要重建求解器的
命令同样要给 `--skip-quality-check`。

---

## 气动系数

```bash
autoflowcfd post coefficients --case results --reference-area 2.2 \
    --reference-length 2.7 --moment-center 1.4 0 0 --groups body --groups wheels
```

- 在 FR 面通量点上积分壁面压力与粘性力（WMLES 的壁面应力取壁面模型）；压力取相对来流静压的表压。
- 缺省积分全部物理无滑移固壁组；`--groups` 指定组（可重复），例如要排除地面时只给车身各组。
- `Cd / Cl / Cs` 按来流方向（`--aoa / --aos`）分解到风轴系；`Cm / Cy / Cr`（俯仰 / 偏航 / 滚转力矩）报在体轴系，
  对 `--moment-center` 取矩、用 `--reference-length` 无量纲化。
- `--reference-area` 必须给出（不同车辆的参考面积约定不同，不猜）；求解过程中的每步系数则用求解时的参考面积
  （求解命令不给时按物体面的迎风投影估算）。
- 求解结束时另打印壁面 y+ 统计（贴壁单元形心高度），超出所选近壁处理的适用范围时警告。

---

## VTK 导出

```bash
autoflowcfd post export-vtk --case results -o flow.vtu                                   # 速度、压力（缺省）
autoflowcfd post export-vtk --case results -o flow.vtu --variables velocity --variables pressure \
    --variables turbulence --variables q_criterion
autoflowcfd post export-vtk --case results -o walls.vtu --boundaries-only               # 命名边界面片
autoflowcfd post export-vtk --case results -o flow_ho.vtu --high-order                  # 高阶 Lagrange 单元
```

- 变量：`velocity`、`pressure`、`turbulence`（展开为模型的全部湍流量）、`k`、`omega`、`nu_tilde`、`nut`、`q_criterion`；
  `--high-order` 另可选 `density`。`--variables` 可重复给出。
- 格式按扩展名：`.vtk` 为 legacy（缺省 ASCII），`.vtu` 为 XML（缺省二进制 + 压缩）；`--binary/--ascii` 覆盖。
- 低阶导出是单元积分平均（每个单元一个值）；`--high-order` 按 FR 分段多项式在 VTK 高阶单元节点上取值（四面体与三棱柱
  到 P3），ParaView 里调高 `Nonlinear Subdivision Level` 可看到单元内部分布。
- `--boundaries-only` 只写边界面片，带 `BoundaryID` / `BoundaryTypeID` 与组名图例，可在 ParaView 里按组筛选着色。
- 瞬态某一时刻：`--checkpoint` 指定那一时刻的 checkpoint。

更多细节见 [VTK 导出指南](VTK_EXPORT_GUIDE.md)。

---

## 收敛曲线与报告

```bash
autoflowcfd post convergence --case results --variables residual --variables cfl --variables coefficients -o conv.png
autoflowcfd post report --case results -o report.json
```

- `--variables` 取 `residual`（监控残差与各方程残差）、`cfl`、`coefficients`（逐步压力积分力系数，只在求解时设置了参考
  面积才有记录；没有记录时报错）。
- 报告含求解参数、逐步残差 / CFL / 系数的统计与最终值，以及求解器自己的结论 `solve_status`（running / converged /
  not_converged / finished）。

---

## 瞬态统计与压力谱

```bash
autoflowcfd post transient-mean --case transient_results --warmup-time 0.05 -o mean.vtk
autoflowcfd post transient-rms  --case transient_results --warmup-time 0.05 -o rms.vtk
autoflowcfd post transient-psd  --case transient_results --warmup-time 0.05 -o psd.csv
```

- 时间轴是 checkpoint 记录的物理时间（时间精确推进时累计）；`--warmup-time` 之前的样本不计入。
- `transient-mean`：节点分辨率的 `MeanVelocity` / `MeanPressure`；`--coefficients`（缺省开）同时统计 Cd / Cl / Cs 的时均与
  标准差——用第一个计入统计的 checkpoint 重建一次求解器，之后每个 checkpoint 只替换状态，任何一个样本失败即报错；
  没有物体的算例用 `--no-coefficients`。
- `transient-rms`：`VelocityRMS` / `PressureRMS`。
- 两者的样本是各个 checkpoint（间隔 = `--checkpoint-interval` 个物理步），需要足够多的 checkpoint 才有统计意义。
- `transient-psd`：探针压力的单边功率谱密度（Pa²/Hz，Welch 法：Hann 窗、50% 重叠；`--nperseg` 指定段长），样本是求解器
  每个物理步在探针处求的 FR 解多项式值（`solve transient --probe X Y Z`）；输出各探针的主导频率。样本必须等间隔
  （固定物理时间步长）。

---

## Python 接口

已有求解器（`api.run_steady` / `run_transient` / `resume_simulation` 之后的 `api.solver`）时：

```python
coeffs = api.calculate_coefficients(reference_area=2.2, reference_length=2.7, moment_center=[1.4, 0, 0],
                                    groups=["body"])
api.export_vtk("flow.vtu", fields=["velocity", "pressure", "q_criterion"])
api.export_vtk("flow_ho.vtu", high_order=True)
history = api.get_convergence_history()
```

模块级（任意求解器对象）：

| 接口 | 说明 |
|------|------|
| `postprocess.fr_coefficients.compute_aerodynamic_coefficients_fr(solver, reference_area, reference_length=1.0, moment_center=None, include_viscous=True, groups=None)` | 返回 `AerodynamicCoefficients`（`Cd Cl Cs Cm Cy Cr`，`.to_dict()`） |
| `postprocess.vtk_export_highorder.export_solver_highorder_vtk(solver, path, fields=None, binary=True)` | 求解器当前状态的高阶导出 |
| `postprocess.VTKExporter(grid_data, solution, turbulence=None, q_criterion=None)` | 单元平均场导出：`.export(path, fields, format='legacy'|'xml', binary=None)`、`.export_boundaries(...)` |
| `postprocess.ConvergenceAnalyzer` | `add_iteration(iteration, residuals, cfl, coefficients)`、`export_csv(path)` |
| `postprocess.SimulationReport(config_summary, analyzer, solve_status)` | `.generate(path, metadata=None)` 写 JSON |
| `postprocess.TransientStatistics(grid_data)` | `accumulate(solution, time)`、`accumulate_coefficients(Cd, Cl, Cs)`、`compute_statistics()` |
| `postprocess.PressurePSD(monitor_points)` | `add_sample(time, pressures)`、`compute_psd(i, nperseg=None, overlap=0.5, window="hann")`、`find_dominant_frequency(i, min_freq, max_freq)` |
| `core.probes.read_probe_series(case_dir)` | `(探针坐标 (k,3), 时刻 (n,), {"p","u","v","w": (n,k)})` |
| `postprocess.wall_yplus.wall_y_plus(solver)` | 贴壁单元形心 y+ 统计 |
