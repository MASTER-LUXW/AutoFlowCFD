# AutoFlowCFD 边界条件配置指南 (v2.0)

## 概述

AutoFlowCFD v2.0引入了基于Properties Name的智能边界条件识别机制，支持三种配置模式：自动、手动和混合模式。本指南详细介绍如何使用这些功能。

## 核心特性

### 1. Properties-based边界识别

传统的CFD软件通常依赖SET卡片或手动指定边界，而AutoFlowCFD v2.0采用更符合工业标准的Properties (PSHELL) Name识别方式：

- **自动解析**：从NAS文件中提取PSHELL卡片的Property Name
- **关键词匹配**：根据Property Name中的关键词自动映射边界类型
- **灵活配置**：支持YAML配置文件精确控制

### 2. 三种配置模式

| 模式 | 描述 | 适用场景 | 配置工作量 |
|------|------|---------|-----------|
| **Auto** | 完全自动识别，基于关键词 | 快速原型、标准工况 | 零配置 |
| **Manual** | 完全依赖YAML配置 | 精确控制、非标准命名 | 全量配置 |
| **Hybrid** | YAML优先 + 自动兜底 | 大部分工程场景（推荐） | 部分配置 |

## 快速开始

### 方案一：自动模式（Auto Mode）

最简单的使用方式，无需任何配置文件：

```python
from autoflowcfd import AutoFlowCFDAPI
from autoflowcfd.boundary import BoundaryManager

# 加载网格
api = AutoFlowCFDAPI()
grid = api.load_grid("ahmed_body.nas")

# 创建边界管理器并自动配置
bc_manager = BoundaryManager(grid.boundaries)
bc_manager.auto_configure()

# 查看配置结果
summary = bc_manager.get_summary()
print(f"Detected {summary['total_boundaries']} boundaries")
for name, details in summary['bc_details'].items():
    print(f"  {name}: {details['type']} ({details['cell_count']} cells)")
```

**自动识别规则**：

| Property Name关键词 | 映射边界类型 | 示例 |
|-------------------|------------|------|
| INLET, INFLOW | VELOCITY_INLET | "INLET", "Velocity_Inlet" |
| OUTLET, OUTFLOW | PRESSURE_OUTLET | "OUTLET", "Pressure_Outlet" |
| SYMMETRY, SYMM | SYMMETRY | "SYMMETRY", "Symm_Plane" |
| TUNNEL, FARFIELD | SLIP_WALL | "TUNNEL", "FARFIELD" |
| 其他所有名称 | WALL | "CAR_BODY", "GROUND" |

### 方案二：手动模式（Manual Mode）

通过YAML配置文件精确控制每个边界的类型和参数：

**步骤1：创建YAML配置文件**

```yaml
# boundary_config.yaml
boundary_detection:
  mode: "manual"

properties_mapping:
  INLET:
    type: "VELOCITY_INLET"
    parameters:
      velocity: [40.0, 0.0, 0.0]  # m/s
      turbulence_intensity: 0.05
  
  OUTLET:
    type: "PRESSURE_OUTLET"
    parameters:
      pressure: 0.0  # Pa (gauge)
  
  CAR_BODY:
    type: "WALL"
    parameters:
      wall_function: "enhanced"
      roughness_height: 0.0001  # m
  
  GROUND_PLANE:
    type: "WALL"
    parameters:
      wall_function: "moving_wall"
      velocity: [40.0, 0.0, 0.0]  # m/s
  
  SYMMETRY:
    type: "SYMMETRY"

defaults:
  velocity_inlet:
    velocity: [33.33, 0.0, 0.0]
    turbulence_intensity: 0.05
  pressure_outlet:
    pressure: 0.0
  wall:
    wall_function: "standard"
```

**步骤2：应用配置**

```python
from autoflowcfd.boundary import BoundaryManager

# 创建边界管理器
bc_manager = BoundaryManager(grid.boundaries)

# 从YAML配置
bc_manager.configure_from_yaml("boundary_config.yaml")

# 验证配置
summary = bc_manager.get_summary()
print(f"Configured {summary['boundaries_with_bc']} boundaries")
```

**注意**：Manual模式下，YAML中未配置的边界会报错。

### 方案三：混合模式（Hybrid Mode - 推荐）

结合YAML配置的精确性和自动识别的便利性：

```python
from autoflowcfd.boundary import BoundaryManager

# 创建边界管理器
bc_manager = BoundaryManager(grid.boundaries)

# 混合模式配置
bc_manager.hybrid_configure("partial_config.yaml")
```

**partial_config.yaml示例**：

```yaml
boundary_detection:
  mode: "hybrid"

properties_mapping:
  # 只配置需要自定义的关键边界
  INLET:
    type: "VELOCITY_INLET"
    parameters:
      velocity: [40.0, 0.0, 0.0]  # 自定义速度
  
  CAR_BODY:
    type: "WALL"
    parameters:
      wall_function: "enhanced"

# 其他边界（如OUTLET、SYMMETRY等）未配置
# 系统将自动按关键词规则映射并使用默认参数

defaults:
  velocity_inlet:
    velocity: [33.33, 0.0, 0.0]
  wall:
    wall_function: "standard"
```

**工作流程**：
1. YAML中配置的边界 → 使用YAML配置
2. YAML中未配置的边界 → 自动识别 + 默认参数

## ANSA前处理最佳实践

### Property命名规范

为了获得最佳的自动识别效果，建议在ANSA中遵循以下命名规范：

**推荐命名**：
```
INLET / VELOCITY_INLET / INFLOW
OUTLET / PRESSURE_OUTLET / OUTFLOW
SYMMETRY / SYMMETRY_PLANE / SYMM
TUNNEL / WIND_TUNNEL / FARFIELD
CAR_BODY / VEHICLE_SURFACE / BODY
GROUND / GROUND_PLANE / ROAD
```

**避免的命名**：
- ❌ 使用特殊字符、空格或中文
- ❌ 无意义的名称（如"BC1", "BC2"）
- ❌ 过长的名称（建议<30字符）

### 在ANSA中设置Property Name

1. 打开ANSA，加载几何模型
2. 为不同区域创建不同的Property
3. 在Property卡片中设置Name字段：
   ```
   $ PROPERTY NAME: INLET
   PSHELL, 10, 1, 0.001
   ```
4. 将对应的面网格分配给相应的Property
5. 导出为.nas文件

## 高级用法

### 动态更新边界参数

```python
# 修改特定边界的参数
bc_manager.update_boundary_params(
    "CAR_BODY",
    wall_function="enhanced",
    roughness_height=0.0001
)

# 查看更新后的参数
params = bc_manager.boundary_map.get_parameters("CAR_BODY")
print(params)
```

### 生成配置模板

```python
# 基于自动识别结果生成可编辑的YAML模板
bc_manager.generate_template("template.yaml")

# 然后编辑template.yaml，补充/修改参数
# 最后使用hybrid模式加载
bc_manager.hybrid_configure("template.yaml")
```

### 导出验证文件

```python
# 导出边界统计信息到JSON
stats = bc_manager.export_to_json("boundary_stats.json")

# 导出VTK文件用于ParaView可视化验证
bc_manager.export_to_vtk("boundaries.vtk")
```

在ParaView中：
1. 打开`boundaries.vtk`
2. 按`BoundaryID`着色
3. 检查边界划分是否正确

### CLI命令行使用

```bash
# 自动模式（默认）
autoflowcfd solve run model.nas --backend gpu

# 指定YAML配置文件（混合模式）
autoflowcfd solve run model.nas \
  --backend gpu \
  --boundary-config boundary_config.yaml

# 查看边界检测结果
autoflowcfd grid boundaries model.nas --output boundaries.json

# 生成边界配置模板
autoflowcfd grid generate-bc-template model.nas --output template.yaml

# 导出边界VTK用于可视化验证
autoflowcfd grid export-boundaries model.nas --output boundaries.vtk
```

## 边界类型详解

### 1. VELOCITY_INLET（速度入口）

**物理含义**：指定来流速度和方向

**参数**：
- `velocity`: [vx, vy, vz] (m/s) - 速度矢量
- `turbulence_intensity`: float (0.0-1.0) - 湍流强度，默认0.05

**示例**：
```yaml
INLET:
  type: "VELOCITY_INLET"
  parameters:
    velocity: [40.0, 0.0, 0.0]  # 144 km/h沿X轴
    turbulence_intensity: 0.05
```

### 2. PRESSURE_OUTLET（压力出口）

**物理含义**：指定静压边界条件

**参数**：
- `pressure`: float (Pa) - 表压，默认0.0

**示例**：
```yaml
OUTLET:
  type: "PRESSURE_OUTLET"
  parameters:
    pressure: 0.0  # 大气压（表压）
```

### 3. WALL（无滑移壁面）

**物理含义**：固体壁面，无滑移条件

**参数**：
- `wall_function`: str - 壁面函数类型
  - `"standard"`: 标准壁面函数（默认）
  - `"enhanced"`: 增强壁面函数
  - `"moving_wall"`: 移动壁面
- `roughness_height`: float (m) - 粗糙度高度，默认0.0
- `velocity`: [vx, vy, vz] (m/s) - 仅用于moving_wall

**示例**：
```yaml
# 静止车身
CAR_BODY:
  type: "WALL"
  parameters:
    wall_function: "enhanced"
    roughness_height: 0.0001

# 移动地面
GROUND:
  type: "WALL"
  parameters:
    wall_function: "moving_wall"
    velocity: [40.0, 0.0, 0.0]
```

### 4. SYMMETRY（对称面）

**物理含义**：对称平面，法向速度为零

**参数**：无

**示例**：
```yaml
SYMMETRY:
  type: "SYMMETRY"
```

### 5. SLIP_WALL（滑移壁面）

**物理含义**：远场边界或自由滑移面，无摩擦切向速度

**参数**：无

**示例**：
```yaml
TUNNEL_WALL:
  type: "SLIP_WALL"
```

## 常见问题

### Q1: 如何检查边界识别是否正确？

**A**: 使用以下方法验证：

```python
# 方法1：打印摘要
summary = bc_manager.get_summary()
print(summary)

# 方法2：导出JSON
bc_manager.export_to_json("check.json")

# 方法3：导出VTK并在ParaView中可视化
bc_manager.export_to_vtk("boundaries.vtk")
```

### Q2: 自动识别的边界类型不正确怎么办？

**A**: 使用Hybrid模式，在YAML中覆盖自动识别结果：

```yaml
properties_mapping:
  MY_PROPERTY:
    type: "WALL"  # 强制指定类型
    parameters:
      wall_function: "standard"
```

### Q3: 如何添加自定义边界类型？

**A**: 扩展BoundaryTypeMapper：

```python
from autoflowcfd.boundary import BoundaryTypeMapper

mapper = BoundaryTypeMapper()
mapper.add_rule("CUSTOM_TYPE", ["KEYWORD1", "KEYWORD2"])
```

### Q4: 参数超出合理范围会怎样？

**A**: 系统会给出警告但不会中断执行。例如：

```
WARNING: High turbulence intensity: 0.250. Typical values are 0.01-0.1 for external flows.
```

### Q5: 支持哪些NAS文件格式？

**A**: 支持ANSA v22/v23/v24输出的标准.nas格式，包括：
- GRID卡片（节点坐标）
- CTRIA3卡片（三角形单元）
- PSHELL卡片（属性定义）
- SET卡片（向后兼容）

### Q6: 性能如何？

**A**: 
- 百万级网格解析时间 < 10秒
- 千万级网格解析时间 < 60秒
- 边界配置应用开销 < 1%总计算时间

### Q7: 如何处理复杂的边界命名？

**A**: 使用Hybrid模式，只需配置关键边界：

```yaml
properties_mapping:
  COMPLEX_NAME_123:
    type: "WALL"
    parameters:
      wall_function: "enhanced"

# 其他边界自动识别
```

## 故障排除

### 问题1：找不到边界

**症状**：`KeyError: Boundary 'XXX' not found`

**解决**：
1. 检查边界名称拼写（大小写敏感）
2. 使用`bc_manager.list_boundaries()`查看所有可用边界
3. 检查NAS文件中是否正确定义了Property Name

### 问题2：YAML配置错误

**症状**：`ConfigurationError: Missing required key...`

**解决**：
1. 检查YAML语法（使用在线YAML验证器）
2. 确保包含`boundary_detection.mode`字段
3. Manual/Hybrid模式必须包含`properties_mapping`

### 问题3：参数验证失败

**症状**：`ValueError: Invalid parameters...`

**解决**：
1. 检查参数类型和范围
2. 参考本文档"边界类型详解"章节
3. 使用默认参数作为起点

## 总结

AutoFlowCFD v2.0的边界条件配置系统提供了：

✅ **智能化**：基于Properties Name的自动识别  
✅ **灵活性**：三种配置模式适应不同场景  
✅ **易用性**：零配置即可开始使用  
✅ **精确性**：YAML配置实现精细控制  
✅ **兼容性**：向后兼容SET卡片  

**推荐工作流**：
1. ANSA中规范命名Property
2. 使用Auto模式快速测试
3. 根据需要切换到Hybrid模式微调
4. 导出VTK验证边界划分
5. 开始仿真计算

更多帮助请参考：
- `autoflowcfd grid --help`
- `autoflowcfd solve --help`
- 项目文档：`docs/`目录
