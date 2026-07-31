# VTK文件数据基础详解

## 📊 核心概念

VTK文件本质上是**网格几何 + 物理场数据**的组合。要生成有意义的VTK文件，必须基于真实的CFD仿真结果。

---

## 🔑 **三大核心数据源**

### 1️⃣ **网格几何数据（Grid Geometry）**

#### **数据来源**
```python
# 从NAS文件解析得到
from autoflowcfd.grid import NASParser, GridData

parser = NASParser("sedan.nas")
grid_data = parser.parse()
```

#### **数据结构**
```python
grid_data = {
    # 节点坐标（定义空间位置）
    'nodes': {
        'x': np.array([x1, x2, ..., xn]),    # X坐标数组
        'y': np.array([y1, y2, ..., yn]),    # Y坐标数组
        'z': np.array([z1, z2, ..., zn]),    # Z坐标数组
        'count': n_nodes                      # 节点总数
    },
    
    # 单元连通性（定义拓扑关系）
    'cells': {
        'connectivity': np.array([
            [n0, n1, n2],   # 单元0的节点索引
            [n3, n4, n5],   # 单元1的节点索引
            ...
        ]),
        'types': np.array([5, 5, 9, ...]),   # 单元类型
        'count': n_cells                      # 单元总数
    }
}
```

#### **在VTK中的表示**
```vtk
POINTS 1000 float
0.0 0.0 0.0
1.0 0.0 0.0
...

CELLS 500 2000
3 0 1 2
3 3 4 5
...

CELL_TYPES 500
5
5
...
```

---

### 2️⃣ **流场解数据（Solution Vector）** ⭐最关键

#### **数据来源**
```python
# 从求解器获得
from autoflowcfd.core.backend.base import SolutionVector

# 方式1: 运行仿真后获取
result = api.run_steady(grid, backend="gpu")
solution = result.solution

# 方式2: 从检查点加载
from autoflowcfd.core.checkpoint import CheckpointManager
ckpt_manager = CheckpointManager("./checkpoints")
solution, _, _, _ = ckpt_manager.load("checkpoint_500.h5")
```

#### **数据结构（保守变量形式）**
```python
solution.data.shape = (n_cells, 5)

# 每行的5个变量：
# Column 0: rho       - 密度 (kg/m³)
# Column 1: rho*u     - X方向动量 OR u速度 (取决于实现)
# Column 2: rho*v     - Y方向动量 OR v速度
# Column 3: rho*w     - Z方向动量 OR w速度
# Column 4: E         - 总能量 (J/m³) OR 压力
```

**注意：** 根据 [`base.py`](d:\myWorkspace\AutoFlowCFD\src\autoflowcfd\core\backend\base.py) 的实现：
- `get_velocity()` 返回 `data[:, 1:4]`，**假设已经是原始速度u,v,w**
- `get_pressure()` 返回 `data[:, 4]`，**假设直接存储压力p**

#### **提取物理量**
```python
# 密度
rho = solution.get_density()          # shape: (n_cells,)

# 速度分量
u, v, w = solution.get_velocity()     # 每个shape: (n_cells,)

# 速度大小
velocity_mag = np.sqrt(u**2 + v**2 + w**2)

# 压力
pressure = solution.get_pressure()    # shape: (n_cells,)
```

#### **如果需要从保守变量转换**
```python
# 如果data[:, 1:4]存储的是动量(rho*u)，需要转换：
rho = solution.data[:, 0]
momentum_x = solution.data[:, 1]
momentum_y = solution.data[:, 2]
momentum_z = solution.data[:, 3]
E = solution.data[:, 4]

# 计算原始速度
u = momentum_x / rho
v = momentum_y / rho
w = momentum_z / rho

# 计算压力（理想气体状态方程）
gamma = 1.4  # 比热比
kinetic_energy = 0.5 * rho * (u**2 + v**2 + w**2)
pressure = (gamma - 1) * (E - kinetic_energy)
```

#### **在VTK中的表示**
```vtk
POINT_DATA 1000

VECTORS Velocity float
30.0 0.0 0.0
30.5 0.2 0.0
...

SCALARS Pressure float 1
LOOKUP_TABLE default
101325.0
101300.0
...
```

---

### 3️⃣ **湍流变量（Turbulence Fields，可选）**

#### **数据来源**
```python
# SST k-omega模型会额外存储
k_field = turbulence_model.k          # 湍流动能
omega_field = turbulence_model.omega  # 比耗散率
```

#### **数据结构**
```python
turbulence_data = {
    'k': np.array([k1, k2, ..., kn]),         # shape: (n_cells,)
    'omega': np.array([ω1, ω2, ..., ωn]),     # shape: (n_cells,)
    'nut': np.array([νt1, νt2, ..., νtn])     # 湍流粘度
}
```

#### **在VTK中的表示**
```vtk
SCALARS TurbulentKineticEnergy float 1
LOOKUP_TABLE default
0.5
0.6
...

SCALARS SpecificDissipationRate float 1
LOOKUP_TABLE default
100.0
120.0
...
```

---

## 🔄 **完整数据流程图**

```
┌─────────────────┐
│  NAS网格文件     │
│  (sedan.nas)    │
└────────┬────────┘
         │ NASParser.parse()
         ▼
┌─────────────────┐
│   GridData       │ ←── 网格几何数据
│  - nodes (xyz)   │
│  - cells (conn)  │
└────────┬────────┘
         │
         │  +  ┌──────────────────┐
         │     │  CFD求解器        │
         │     │  (FR Solver)      │
         │     └────────┬─────────┘
         │              │ solve()
         ▼              ▼
┌────────────────────────────────┐
│   SolutionVector               │ ←── 流场解数据
│   data.shape = (n_cells, 5)    │
│   [rho, u, v, w, p]            │
└────────┬───────────────────────┘
         │
         │ VTKExporter.export()
         ▼
┌─────────────────┐
│  VTK文件         │
│  (result.vtk)   │
│  - 网格几何      │
│  - 速度场        │
│  - 压力场        │
│  - 湍流量(可选)  │
└────────┬────────┘
         │ ParaView打开
         ▼
┌─────────────────┐
│  可视化结果      │
│  - 速度云图      │
│  - 压力分布      │
│  - 流线图        │
└─────────────────┘
```

---

## 💻 **实际代码示例**

### **完整的VTK导出流程**

```python
from autoflowcfd import AutoFlowCFDAPI
from autoflowcfd.postprocess import VTKExporter

# 步骤1: 加载网格
api = AutoFlowCFDAPI()
grid = api.load_grid("ahmed_body.nas")

print(f"网格信息:")
print(f"  节点数: {grid.metadata.node_count}")
print(f"  单元数: {grid.metadata.cell_count}")

# 步骤2: 运行仿真（获得SolutionVector）
result = api.run_steady(
    grid_data=grid,
    backend="gpu",
    order=3,
    turbulence="sst_kw",
    max_iter=1000
)

solution = result.solution
print(f"\n解向量信息:")
print(f"  形状: {solution.shape}")
print(f"  密度范围: [{solution.get_density().min():.3f}, {solution.get_density().max():.3f}]")
print(f"  压力范围: [{solution.get_pressure().min():.1f}, {solution.get_pressure().max():.1f}]")

# 步骤3: 创建VTK导出器
exporter = VTKExporter(
    grid_data=grid,
    solution=solution
)

# 步骤4: 导出VTK文件
vtk_path = exporter.export(
    output_path="results/flow_field.vtk",
    fields=['velocity', 'pressure'],
    format='legacy'
)

print(f"\n✅ VTK文件已导出: {vtk_path}")
print("💡 使用ParaView打开查看速度和压力分布")
```

### **从检查点恢复并导出**

```python
from autoflowcfd.core.checkpoint import CheckpointManager

# 加载检查点
ckpt_manager = CheckpointManager("./checkpoints")
solution, history, iteration, metadata = ckpt_manager.load(
    "checkpoint_500.h5",
    target_backend="cpu"
)

print(f"从迭代 {iteration} 恢复")

# 导出该时刻的流场
exporter = VTKExporter(grid_data=grid, solution=solution)
vtk_path = exporter.export(f"results/step_{iteration}.vtk")
```

### **瞬态仿真批量导出**

```python
# 在瞬态求解循环中
output_interval = 50  # 每50步导出一次

for step in range(total_steps):
    # ... 求解一步 ...
    solution = solver.step()
    
    if step % output_interval == 0:
        exporter = VTKExporter(grid_data=grid, solution=solution)
        vtk_path = f"transient/step_{step:04d}.vtk"
        exporter.export(vtk_path, fields=['velocity', 'pressure'])
        print(f"Exported: {vtk_path}")
```

---

## ⚠️ **常见问题与解决方案**

### **Q1: 节点数和单元数不匹配怎么办？**

**问题：** VTK要求POINT_DATA的数量与POINTS数量一致，但CFD解通常是cell-centered（单元中心）。

**解决方案：**
```python
# 方案1: 简单平均插值（当前实现）
if n_cells != n_points:
    u_mean = np.mean(u)
    # 所有节点使用相同值
    
# 方案2: 最近邻插值（更准确）
# 为每个节点找到最近的单元
from scipy.spatial import cKDTree

cell_centers = np.column_stack([cell_x, cell_y, cell_z])
node_coords = np.column_stack([nodes.x, nodes.y, nodes.z])

tree = cKDTree(cell_centers)
_, indices = tree.query(node_coords)

# 插值到节点
u_at_nodes = u[indices]
v_at_nodes = v[indices]
w_at_nodes = w[indices]
```

### **Q2: 如何验证导出的数据是正确的？**

**方法1: 数值检查**
```python
# 导出前验证
rho = solution.get_density()
assert np.all(rho > 0), "密度必须为正"

p = solution.get_pressure()
assert np.all(p > 0), "压力必须为正"

u, v, w = solution.get_velocity()
vel_mag = np.sqrt(u**2 + v**2 + w**2)
print(f"速度范围: [{vel_mag.min():.2f}, {vel_mag.max():.2f}] m/s")
```

**方法2: ParaView可视化验证**
```
1. 打开VTK文件
2. 检查速度矢量方向是否合理（应沿来流方向）
3. 检查压力分布是否符合物理预期
4. 对比气动力系数计算结果
```

### **Q3: 湍流变量如何存储和导出？**

**当前限制：** SolutionVector默认只有5个变量，不包含湍流量。

**解决方案：**
```python
# 方案1: 扩展SolutionVector
class ExtendedSolution(SolutionVector):
    def __init__(self, n_cells, n_turb_vars=2):
        super().__init__(n_cells=n_cells, n_variables=5+n_turb_vars)
        self.k = self.data[:, 5]      # 湍流动能
        self.omega = self.data[:, 6]  # 比耗散率

# 方案2: 单独传递湍流场
exporter = VTKExporter(grid_data=grid, solution=solution)
exporter.turbulence_fields = {
    'k': k_field,
    'omega': omega_field
}
exporter.export("full.vtk", fields=['velocity', 'pressure', 'k', 'omega'])
```

---

## 📚 **相关文档**

- **VTK文件格式规范**: [VTK File Formats](https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf)
- **SolutionVector定义**: [`src/autoflowcfd/core/backend/base.py`](d:\myWorkspace\AutoFlowCFD\src\autoflowcfd\core\backend\base.py)
- **VTK导出器**: [`src/autoflowcfd/postprocess/vtk_export.py`](d:\myWorkspace\AutoFlowCFD\src\autoflowcfd\postprocess\vtk_export.py)
- **完整指南**: [`docs/VTK_EXPORT_GUIDE.md`](d:\myWorkspace\AutoFlowCFD\docs\VTK_EXPORT_GUIDE.md)

---

## 🎯 **总结**

生成VTK文件需要的**核心数据**：

| 数据类型 | 来源 | 必需性 | 用途 |
|---------|------|--------|------|
| **网格节点坐标** | NASParser | ✅ 必需 | 定义几何形状 |
| **单元连通性** | NASParser | ✅ 必需 | 定义拓扑结构 |
| **速度场** | SolutionVector | ✅ 推荐 | 速度云图、流线 |
| **压力场** | SolutionVector | ✅ 推荐 | 压力分布、气动力 |
| **湍流量** | TurbulenceModel | ⭕ 可选 | 湍流分析 |

**关键原则：**
1. **网格数据**来自NAS文件解析
2. **流场数据**来自CFD求解器的SolutionVector
3. VTK导出器负责将两者结合并格式化为VTK格式
4. 当前实现已从SolutionVector提取真实数据（非占位符）
