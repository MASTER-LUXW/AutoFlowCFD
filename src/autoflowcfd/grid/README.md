# grid 网格子系统

从 ANSA `.nas` 到求解器可用的高阶网格：解析、体网格生成、导入与修复、质量校验、连接关系、格式转换。命令行入口
`autoflowcfd grid ...`（`cli/grid/`）。

## 子包

| 子包 | 内容 |
|------|------|
| `nas_io/` | `.nas` 解析（`NASParser`：GRID / CTRIA3 / CTETRA / CPENTA / PSHELL）与导出；坐标单位 `units.py`（mm / m / auto，内部一律米）；边界类型识别与显式指定 `boundary_types.py` |
| `schema/` | 数据结构：`NodeArray`、`TetrahedralCells` / `PrismCells`、`BoundaryMap`（组 -> 贴边界单元、组 -> 类型、周期参数）、`GridData`（面网格）、`VolumeMeshData`（体网格，棱柱在前、四面体在后） |
| `mesh_gen/` | 体网格生成：`extrusion/` 边界层棱柱挤出（第一层高度可按目标 y+ 推出，碰撞冻结、局部厚度限制）、`tetgen/` 核心区四面体填充与合并（`VolumeMeshGenerator`）、`repair/` 质量修复与非流形修补、`extraction/` 面提取、`background/` 合并与面几何 |
| `validation/` | 面网格校验 `GridValidator`、体网格质量 `MeshQualityValidator`（求解前质量门 `solve_gate.py`）、退化单元判定 `cell_validity.py` |
| `connectivity/` | 面连接关系（内部面 / 边界面、周期面配对、逐面边界组标记） |
| `high_order/` | `HighOrderMesh`：按阶数生成解点坐标、度量、面通量点几何；阶数切换 |
| `conversion/` | 导出 VTK / CGNS / STL（`grid convert`） |
| `curved_mapping/` | 参考单元到物理单元的映射与单元朝向（直边单元） |

## 典型用法

```python
from autoflowcfd.grid import NASParser, GridValidator

surface = NASParser("examples/cube_demo/cube_demo.nas", units="m").parse()
report = GridValidator(surface).validate()          # passed / summary / aspect_ratio / skewness / jacobian
print(surface.node_count, surface.cell_count, sorted(surface.boundaries.groups), report["passed"])
```

体网格生成与导入建议走命令行（同一套实现，参数更全）：

```bash
autoflowcfd grid generate-volume surface.nas -o volume.nas --units m --bl-layers 6 --target-yplus 1 \
    --ref-velocity 40 --ref-length 1
autoflowcfd grid import-volume ext_volume.nas --surface-mesh surface.nas -o volume.pkl --units mm
autoflowcfd grid validate volume.nas --units m
```

## 约定

- 坐标单位只在解析 / 导出时换算，内部一律米。
- 每个边界面必须属于某个组，每个组必须有边界条件类型；识别不了的组报错，由 `--bc-type` 指定。
- 体网格生成的输出要通过硬检查（退化单元、流形性、退化面、边界覆盖）；质量门不过的网格不能求解
  （`--skip-quality-check` 只用于临时诊断）。

## 测试

```bash
python -m pytest tests/unit -q -o addopts= -k "grid or nas or mesh"
```
