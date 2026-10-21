# VTK 导出指南

把求解结果导出给 ParaView（或任何读 VTK 的工具）。实现：`postprocess/vtk_export.py`（低阶）、
`postprocess/vtk_export_highorder/`（高阶），命令行入口 `autoflowcfd post export-vtk`。

---

## 两种导出

| | 低阶（缺省） | 高阶（`--high-order`） |
|---|---|---|
| 数据来源 | checkpoint 里每个单元的积分平均 | 从 checkpoint 重建求解器，取全部解点 |
| 单元 | 线性四面体 / 三棱柱 | VTK Lagrange 高阶四面体（71）/ 三棱柱（73），节点值为 FR 解多项式的精确取值 |
| 写出的数组 | CELL_DATA（单元平均）+ POINT_DATA（按体积加权平均到节点） | POINT_DATA（高阶节点） |
| 格式 | `.vtk` legacy 或 `.vtu` XML | 只 `.vtu` |
| 阶数 | 任意 | 四面体与三棱柱到 P3 |
| 可选变量 | `velocity` `pressure` `turbulence` `k` `omega` `nu_tilde` `nut` `q_criterion` | 同左，另有 `density` |

低阶文件小、读得快，适合大网格总体浏览；高阶保留单元内部的真实分布（边界层、剪切层），ParaView 里把
`Nonlinear Subdivision Level` 调到 2~3 才能看到。

`turbulence` 展开为当前湍流模型的全部量（SST：k、omega、nut；SA：nu_tilde、nut）；层流算例请求它时报错。
`q_criterion` 用解点上的速度梯度计算（与湍流模型同一实现）。

---

## 命令行

```bash
autoflowcfd post export-vtk --case results -o flow.vtu
autoflowcfd post export-vtk --case results -o flow.vtk --variables velocity --variables pressure --variables turbulence
autoflowcfd post export-vtk --case results -o flow_ho.vtu --high-order --variables velocity --variables q_criterion
autoflowcfd post export-vtk --case results -o walls.vtu --boundaries-only
autoflowcfd post export-vtk --case results --checkpoint results/checkpoints/checkpoint_iter_000500.h5 -o t500.vtu
```

| 选项 | 说明 |
|---|---|
| `--case` | 算例目录（`solve` 的 `-o` 目录） |
| `--output / -o` | 输出文件；扩展名决定格式（`.vtk` legacy、`.vtu` XML） |
| `--variables` | 要导出的变量，可重复给出；缺省 velocity 与 pressure |
| `--checkpoint` | 指定 checkpoint（瞬态某一时刻）；缺省取最新 |
| `--grid / -g` | 体网格；缺省取算例目录的 `volume_mesh.pkl` 或 checkpoint 记录的求解输入 |
| `--binary / --ascii` | 缺省 legacy 写 ASCII、XML 写二进制 + 压缩 |
| `--boundaries-only` | 只写命名边界面片（不能与 `--high-order` 同用） |
| `--high-order` | 高阶 Lagrange 单元 |
| `--surface-mesh / -s` | 需要重建求解器（`q_criterion`、`--high-order`）且 checkpoint 记录的输入是 `.nas` 体网格、又没记录面网格时给出 |
| `--skip-quality-check` | 原求解靠它才跑得起来的网格，重建时同样跳过质量门 |

数组名：`Velocity`、`Pressure`、`Density`、`TurbulentKineticEnergy`、`SpecificDissipationRate`、`SANuTilde`、
`TurbulentViscosity`、`QCriterion`；边界面片另有 `BoundaryID` / `BoundaryTypeID` 与组名图例（field data）。

瞬态时均 / 脉动场由 `post transient-mean` / `transient-rms` 导出（节点场 `MeanVelocity`、`MeanPressure`、`VelocityRMS`、
`PressureRMS`），见 [后处理指南](POSTPROCESSING_GUIDE.md)。

---

## Python

```python
from autoflowcfd import AutoFlowCFDAPI

api = AutoFlowCFDAPI()
api.resume_simulation("results/checkpoints/checkpoint_iter_000500.h5")   # max_iter=0：只恢复状态
api.export_vtk("flow.vtu", fields=["velocity", "pressure"])
api.export_vtk("flow_ho.vtu", high_order=True, fields=["velocity", "density"])
```

直接使用导出器：

```python
from autoflowcfd.postprocess import VTKExporter
from autoflowcfd.postprocess.vtk_export_highorder import export_solver_highorder_vtk

VTKExporter(volume_mesh, solution, turbulence={"k": k_cell, "omega": w_cell}).export(
    "flow.vtu", fields=["velocity", "pressure", "k"], format="xml")
export_solver_highorder_vtk(solver, "flow_ho.vtu", fields=["velocity", "pressure"])
```

`solution` 是单元积分平均的守恒量（`SolutionVector`，单元顺序棱柱在前、四面体在后，与体网格一致）。

---

## 只导出网格

不带解的网格（检查生成的体网格、边界分组）用 `grid convert`：

```bash
autoflowcfd grid convert cube/cube_volume.nas -f vtk -o cube_mesh.vtu   # 体单元 + 外部面的组号
autoflowcfd grid convert examples/cube_demo/cube_demo.nas -f stl       # 每个边界组一个 solid
autoflowcfd grid convert cube/cube_volume.nas -f cgns                  # CGNS/HDF5，ZoneBC 每组一个 Family
```

---

## 在 ParaView 里

- 低阶文件同时有单元值与节点值：着色时选 `Pressure`（点）得到平滑云图，选单元值看每个单元的平均。
- 高阶文件：`Properties → Nonlinear Subdivision Level` 设 2~3；切片、等值面、流线照常使用。
- Q 准则：`Contour` 取 `QCriterion` 的正值（如 `(U_inf / L)^2` 的量级）看涡结构。
- 边界面片：`Threshold` 按 `BoundaryID` 过滤某一组，或按 `BoundaryTypeID` 只看固壁。

---

## 常见问题

**文件很大** —— 用 `.vtu`（二进制 + 压缩）；只导出需要的变量；大网格先用低阶浏览，局部再用高阶。

**高阶文件在 ParaView 里看起来和低阶一样** —— 没有调 `Nonlinear Subdivision Level`。

**报"网格与解的单元数不符"** —— 用了别的网格：后处理必须用求解时的体网格（算例目录的 `volume_mesh.pkl`），不能用面网格
重新生成。

**请求 `q_criterion` / `--high-order` 时报找不到面网格** —— 求解输入是 `.nas` 体网格的旧算例，加 `--surface-mesh`。
