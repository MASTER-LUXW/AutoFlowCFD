# `.pkl` 体网格

`.pkl` 是 `VolumeMeshData` 对象（`src/autoflowcfd/grid/schema/grid_data.py`）的 Python pickle，求解器的原生体网格格式。

---

## 从哪里来

| 来源 | 文件 |
|------|------|
| `grid import-volume ext.nas --surface-mesh surf.nas -o mesh.pkl` | 外部体网格经质量检查（与可选修复）后写出 |
| `solve steady / transient` 的输入是 `.nas` 体网格时 | 按 `--units` 导入后写成 `<输出目录>/volume_mesh.pkl` |
| Python API `run_steady / run_transient` | 把传入的体网格写成 `<output_dir>/volume_mesh.pkl` |

坐标一律为米（`.nas` 导入时按单位换算）。checkpoint 的 `input_file` 记录这份 `.pkl`，`solve resume` 与 `post` 命令据此
重建网格——后处理必须用求解时的那张体网格（解的单元与它一一对应），不能由面网格重新生成。

## 内容

| 属性 | 内容 |
|------|------|
| `nodes` | 节点坐标 `x / y / z`（米） |
| `prism_cells` | 边界层三棱柱 `connectivity (N_prism, 6)`（可为 None） |
| `cells` | 四面体 `connectivity (N_tet, 4)` |
| `boundaries` | `BoundaryMap`：`groups`（组名 -> 贴边界的单元编号）、`bc_types`（组名 -> 边界条件类型）、`parameters`（周期组的配对与平移） |
| `surface_mesh` | 边界三角形与分组（逐面边界归属、参考面积估算用） |
| `metadata` | 网格元数据 |

全局单元编号约定棱柱在前 `[0, N_prism)`、四面体在后，与求解器状态的行顺序、VTK 导出的单元顺序一致。

## 读取

```python
import pickle

with open("results/volume_mesh.pkl", "rb") as f:
    mesh = pickle.load(f)
print(mesh.node_count, mesh.cell_count, sorted(mesh.boundaries.groups))
```

pickle 只能在兼容的 AutoFlowCFD 版本之间读取（类定义变化后旧文件可能读不进来）；需要交换给别的软件时用
`grid convert -f vtk / cgns / stl`。只加载可信来源的 `.pkl`（pickle 反序列化会执行文件里的代码）。
