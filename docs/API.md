# AutoFlowCFD API 参考文档

本文档提供 AutoFlowCFD Python API 的完整参考说明，包括核心类、方法签名、参数说明和使用示例。

---

## 📋 目录

- [快速开始](#快速开始)
- [核心 API](#核心-api)
  - [AutoFlowCFDAPI](#autoflowcfdapi)
  - [网格处理](#网格处理)
  - [求解器配置](#求解器配置)
  - [仿真运行](#仿真运行)
  - [后处理工具](#后处理工具)
- [高级用法](#高级用法)
- [错误处理](#错误处理)
- [性能优化建议](#性能优化建议)

---

## 快速开始

```python
from autoflowcfd import AutoFlowCFDAPI

# 创建 API 实例
api = AutoFlowCFDAPI(verbose=True)

# 加载网格
grid = api.load_grid("car_model.nas")

# 运行稳态仿真
result = api.run_steady(
    grid,
    backend="gpu",
    turbulence="sst_kw",
    order=2
)

# 获取气动系数
coeffs = api.calculate_coefficients(result)
print(f"Cd: {coeffs['Cd']:.4f}")
```

---

## 核心 API

### AutoFlowCFDAPI

主 API 类，提供统一的接口访问所有功能。

#### 初始化

```python
from autoflowcfd import AutoFlowCFDAPI

api = AutoFlowCFDAPI(
    verbose=False,        # 是否启用详细日志
    log_level="INFO"      # 日志级别: DEBUG, INFO, WARNING, ERROR
)
```

**参数说明**：
- `verbose` (bool): 启用详细日志输出，默认 `False`
- `log_level` (str): 日志级别，默认 `"INFO"`

---

### 网格处理

#### load_grid()

加载并解析 NAS 网格文件。

```python
grid = api.load_grid(
    filepath="car_model.nas",
    validate=True,         # 是否进行网格质量校验
    show_info=True         # 是否显示网格统计信息
)
```

**参数说明**：
- `filepath` (str): NAS 文件路径
- `validate` (bool): 是否执行网格质量校验，默认 `True`
- `show_info` (bool): 是否打印网格统计信息，默认 `True`

**返回值**：
- `GridData`: 网格数据对象，包含节点、单元、边界信息

**示例**：
```python
grid = api.load_grid("examples/ahmed_demo/car_model.nas")
print(f"节点数: {grid.node_count}")
print(f"单元数: {grid.cell_count}")
print(f"边界组数: {len(grid.boundary_map)}")
```

**GridData 属性**：
- `node_count` (int): 节点总数
- `cell_count` (int): 单元总数
- `nodes` (NodeArray): 节点坐标数组（SoA 布局）
- `cells` (CellArray): 单元连接关系数组
- `boundary_map` (BoundaryMap): 边界条件映射

---

#### validate_grid()

独立校验网格质量。

```python
validation_result = api.validate_grid(grid)

if validation_result.is_valid:
    print("✅ 网格质量合格")
else:
    print("❌ 网格存在问题:")
    for issue in validation_result.issues:
        print(f"  - {issue}")
```

**返回值**：
- `ValidationResult`: 包含校验结果和问题列表

**ValidationResult 属性**：
- `is_valid` (bool): 网格是否通过校验
- `issues` (list[str]): 问题描述列表
- `statistics` (dict): 网格质量统计（长宽比、扭曲度等）

---

### 求解器配置

#### SteadyConfig

稳态仿真配置类。

```python
from autoflowcfd.config import SteadyConfig, BackendType, TurbulenceModel

config = SteadyConfig(
    backend=BackendType.CPU,          # 计算后端: CPU 或 GPU
    order=2,                          # FR 阶数: 1, 2, 3
    turbulence=TurbulenceModel.SST_KW, # 湍流模型
    max_iter=5000,                    # 最大迭代次数
    convergence_tol=1.0e-6,           # 收敛容差
    cfl_init=0.1,                     # 初始 CFL 数
    cfl_max=5.0,                      # 最大 CFL 数
    output_dir="./results",           # 输出目录
    checkpoint_interval=100           # 检查点保存间隔
)
```

**参数说明**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `backend` | BackendType | CPU | 计算后端（CPU/GPU） |
| `order` | int | 2 | FR 离散格式阶数（1/2/3） |
| `turbulence` | TurbulenceModel | SST_KW | 湍流模型 |
| `max_iter` | int | 5000 | 最大迭代次数 |
| `convergence_tol` | float | 1.0e-6 | 收敛容差（残差） |
| `cfl_init` | float | 0.1 | 初始 CFL 数 |
| `cfl_max` | float | 5.0 | 最大 CFL 数 |
| `output_dir` | str | "./results" | 输出目录路径 |
| `checkpoint_interval` | int | 100 | 检查点保存间隔（步数） |

**BackendType 枚举**：
- `BackendType.CPU`: CPU 后端（Numba 并行）
- `BackendType.GPU`: GPU 后端（CUDA 加速）

**TurbulenceModel 枚举**：
- `TurbulenceModel.SST_KW`: SST k-ω RANS 模型
- `TurbulenceModel.DES`: DES 混合模型
- `TurbulenceModel.DDES`: DDES 延迟分离涡模型

---

#### TransientConfig

瞬态仿真配置类。

```python
from autoflowcfd.config import TransientConfig

config = TransientConfig(
    backend=BackendType.GPU,
    order=2,
    turbulence=TurbulenceModel.DDES,
    dt=1.0e-4,              # 时间步长（秒）
    total_time=0.1,         # 总仿真时间（秒）
    output_dir="./results_transient"
)
```

**额外参数**：
- `dt` (float): 时间步长（秒）
- `total_time` (float): 总仿真时间（秒）
- `time_scheme` (str): 时间离散格式（"BE"/"RK2"/"AB3"）

---

### 仿真运行

#### run_steady()

运行稳态 RANS 仿真。

```python
result = api.run_steady(
    grid,
    backend="cpu",
    turbulence="sst_kw",
    order=2,
    max_iter=5000,
    convergence_tol=1.0e-6,
    cfl_init=0.1,
    cfl_max=5.0,
    output_dir="./results",
    config=None  # 或使用预定义的 SteadyConfig 对象
)
```

**参数说明**：
- `grid` (GridData): 网格数据对象
- `backend` (str): "cpu" 或 "gpu"
- `turbulence` (str): "sst_kw", "des", "ddes"
- `order` (int): FR 阶数（1/2/3）
- `max_iter` (int): 最大迭代次数
- `convergence_tol` (float): 收敛容差
- `cfl_init` (float): 初始 CFL 数
- `cfl_max` (float): 最大 CFL 数
- `output_dir` (str): 输出目录
- `config` (SteadyConfig): 可选，使用预定义配置对象

**返回值**：
- `SteadyResult`: 稳态仿真结果对象

**SteadyResult 属性**：
- `converged` (bool): 是否收敛
- `iterations` (int): 实际迭代次数
- `final_residual` (float): 最终残差
- `execution_time` (float): 计算耗时（秒）
- `output_dir` (str): 输出目录路径

**示例**：
```python
result = api.run_steady(grid, backend="gpu", order=2)

if result.converged:
    print(f"✅ 收敛于 {result.iterations} 步")
    print(f"最终残差: {result.final_residual:.6e}")
else:
    print(f"❌ 未收敛，最终残差: {result.final_residual:.6e}")
```

---

#### run_transient()

运行瞬态 DES/DDES 仿真。

```python
result = api.run_transient(
    grid,
    backend="gpu",
    turbulence="ddes",
    order=2,
    dt=1.0e-4,
    total_time=0.1,
    output_dir="./results_transient"
)
```

**参数说明**：
- `grid` (GridData): 网格数据对象
- `backend` (str): "cpu" 或 "gpu"
- `turbulence` (str): "des" 或 "ddes"
- `order` (int): FR 阶数（1/2/3）
- `dt` (float): 时间步长（秒）
- `total_time` (float): 总仿真时间（秒）
- `output_dir` (str): 输出目录

**返回值**：
- `TransientResult`: 瞬态仿真结果对象

**TransientResult 属性**：
- `num_steps` (int): 时间步总数
- `current_time` (float): 当前仿真时间（秒）
- `execution_time` (float): 计算耗时（秒）
- `output_dir` (str): 输出目录路径

---

#### resume_from_checkpoint()

从检查点恢复仿真。

```python
result = api.resume_from_checkpoint(
    checkpoint_file="./results/checkpoints/latest_checkpoint.h5",
    max_iter=10000  # 继续迭代的总次数
)
```

**参数说明**：
- `checkpoint_file` (str): 检查点文件路径
- `max_iter` (int): 继续迭代的目标次数

**返回值**：
- `SteadyResult`: 恢复后的仿真结果

---

### 后处理工具

#### calculate_coefficients()

计算气动系数（Cd, Cl, Cs）。

```python
coeffs = api.calculate_coefficients(result)

print(f"风阻系数 Cd: {coeffs['Cd']:.4f}")
print(f"升力系数 Cl: {coeffs['Cl']:.4f}")
print(f"侧力系数 Cs: {coeffs['Cs']:.4f}")
```

**参数说明**：
- `result` (SteadyResult 或 TransientResult): 仿真结果对象

**返回值**：
- `dict`: 包含气动系数的字典
  - `'Cd'` (float): 风阻系数
  - `'Cl'` (float): 升力系数
  - `'Cs'` (float): 侧力系数
  - `'Cm_pitch'` (float): 俯仰力矩系数
  - `'Cm_yaw'` (float): 偏航力矩系数
  - `'Cm_roll'` (float): 滚转力矩系数

---

#### export_vtk()

导出 VTK 格式可视化文件。

```python
# 导出单个时间步
api.export_vtk(
    result,
    output_file="./results/field.vtu",
    variables=["pressure", "velocity", "vorticity"]
)

# 导出所有时间步（瞬态仿真）
api.export_all_vtk(
    result,
    output_dir="./results/vtk_fields"
)
```

**参数说明**：
- `result` (SteadyResult 或 TransientResult): 仿真结果对象
- `output_file` (str): 输出文件路径（单步）
- `output_dir` (str): 输出目录（全部）
- `variables` (list[str]): 导出的变量列表

**可用变量**：
- `"pressure"`: 静压
- `"velocity"`: 速度矢量
- `"vorticity"`: 涡量
- `"turbulence_ke"`: 湍流动能
- `"wall_shear_stress"`: 壁面剪应力

---

#### export_convergence_history()

导出收敛历史数据。

```python
import pandas as pd

df = api.export_convergence_history(result)
print(df.head())

# 绘制收敛曲线
import matplotlib.pyplot as plt

plt.semilogy(df['iteration'], df['residual'])
plt.xlabel('Iteration')
plt.ylabel('Residual')
plt.title('Convergence History')
plt.grid(True, alpha=0.3)
plt.show()
```

**返回值**：
- `pd.DataFrame`: 包含收敛历史的 DataFrame
  - `iteration`: 迭代步数
  - `residual`: 残差
  - `Cd`, `Cl`, `Cs`: 各迭代步的气动系数

---

#### generate_report()

生成仿真报告（JSON 格式）。

```python
report = api.generate_report(result)

import json
with open("report.json", "w") as f:
    json.dump(report, f, indent=2)
```

**返回值**：
- `dict`: 包含完整仿真信息的字典
  - `simulation_info`: 仿真配置信息
  - `grid_info`: 网格统计信息
  - `results_summary`: 结果摘要
  - `performance_metrics`: 性能指标

---

## 高级用法

### 自定义边界条件

```python
from autoflowcfd.boundary import BoundaryCondition

# 定义速度入口
inlet_bc = BoundaryCondition(
    type="INLET",
    velocity_x=30.0,  # m/s
    velocity_y=0.0,
    velocity_z=0.0,
    pressure=101325.0  # Pa
)

# 定义移动地面（模拟车轮旋转）
moving_ground_bc = BoundaryCondition(
    type="WALL",
    moving=True,
    velocity_x=30.0,
    wall_function="enhanced"
)

# 应用边界条件
api.apply_boundary_conditions(grid, {
    "inlet_group": inlet_bc,
    "ground_group": moving_ground_bc
})
```

---

### 批量参数扫描

```python
import numpy as np

# 参数范围
angles = np.linspace(-5, 5, 11)  # 前倾角: -5° 到 5°
results = []

for angle in angles:
    print(f"\n运行角度: {angle}°")
    
    # 修改网格（需要额外的网格变形工具）
    modified_grid = api.morph_grid(grid, pitch_angle=angle)
    
    # 运行仿真
    result = api.run_steady(
        modified_grid,
        backend="gpu",
        order=2,
        max_iter=3000
    )
    
    # 记录结果
    coeffs = api.calculate_coefficients(result)
    results.append({
        'angle': angle,
        'Cd': coeffs['Cd'],
        'Cl': coeffs['Cl']
    })

# 分析结果
import pandas as pd
df_results = pd.DataFrame(results)
print(df_results)

# 绘制 Cd vs 角度曲线
plt.plot(df_results['angle'], df_results['Cd'], 'o-')
plt.xlabel('Pitch Angle (deg)')
plt.ylabel('Drag Coefficient (Cd)')
plt.title('Cd vs Pitch Angle')
plt.grid(True)
plt.show()
```

---

### 与 AI Agent 集成

```python
import json

def optimize_aerodynamics(api, grid, initial_params):
    """
    简单的参数优化循环（可替换为贝叶斯优化、遗传算法等）
    """
    best_cd = float('inf')
    best_params = initial_params
    
    for iteration in range(20):
        # 1. 根据参数修改网格
        modified_grid = api.morph_grid(grid, **initial_params)
        
        # 2. 运行仿真
        result = api.run_steady(modified_grid, backend="gpu")
        coeffs = api.calculate_coefficients(result)
        
        # 3. 记录结果
        cd = coeffs['Cd']
        print(f"Iteration {iteration}: Cd = {cd:.4f}")
        
        # 4. 更新最优解
        if cd < best_cd:
            best_cd = cd
            best_params = initial_params.copy()
        
        # 5. 调整参数（简化示例，实际应使用优化算法）
        initial_params['front_angle'] += np.random.randn() * 0.5
    
    return best_params, best_cd

# 使用示例
optimal_params, min_cd = optimize_aerodynamics(
    api, grid,
    initial_params={'front_angle': 0.0, 'rear_angle': 0.0}
)

print(f"\n最优参数: {optimal_params}")
print(f"最小 Cd: {min_cd:.4f}")

# 输出结构化结果（供 AI Agent 解析）
output = {
    "status": "success",
    "optimal_parameters": optimal_params,
    "minimum_drag_coefficient": min_cd,
    "optimization_iterations": 20
}

with open("optimization_result.json", "w") as f:
    json.dump(output, f, indent=2)
```

---

### 多 GPU 分布式计算（规划中）

```python
# 未来版本支持
result = api.run_steady_distributed(
    grid,
    num_gpus=4,
    backend="multi-gpu",
    order=3
)
```

---

## 错误处理

### 常见异常类型

```python
from autoflowcfd.exceptions import (
    GridParseError,
    SolverConvergenceError,
    BackendNotAvailableError,
    ConfigurationError
)

try:
    grid = api.load_grid("invalid_file.nas")
except GridParseError as e:
    print(f"网格解析失败: {e}")
    print("请检查文件格式是否正确")

try:
    result = api.run_steady(grid, backend="gpu")
except BackendNotAvailableError as e:
    print(f"GPU 后端不可用: {e}")
    print("回退到 CPU 计算")
    result = api.run_steady(grid, backend="cpu")

try:
    result = api.run_steady(grid, max_iter=100)
except SolverConvergenceError as e:
    print(f"仿真未收敛: {e}")
    print("尝试增加迭代次数或调整 CFL 数")
```

---

### 日志配置

```python
import logging

# 设置日志级别
api.set_log_level("DEBUG")  # 详细调试信息
api.set_log_level("INFO")   # 标准信息（默认）
api.set_log_level("WARNING") # 仅警告和错误

# 自定义日志处理器
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("autoflowcfd.log"),
        logging.StreamHandler()
    ]
)
```

---

## 性能优化建议

### 1. 选择合适的 FR 阶数

```python
# 低精度快速预览
result = api.run_steady(grid, order=1)  # 最快，精度最低

# 工程推荐精度
result = api.run_steady(grid, order=2)  # 平衡精度与速度

# 高精度研究
result = api.run_steady(grid, order=3)  # 最慢，精度最高
```

**建议**：
- 初步设计阶段：使用 1 阶快速评估
- 工程开发阶段：使用 2 阶平衡精度与效率
- 学术研究/最终验证：使用 3 阶获取最高精度

---

### 2. CPU/GPU 选择策略

```python
# 小网格（<100 万单元）：CPU 可能更快
if grid.cell_count < 1_000_000:
    backend = "cpu"
else:
    backend = "gpu"

result = api.run_steady(grid, backend=backend)
```

**经验法则**：
- CPU（16 线程）：适合 <500 万单元网格
- GPU（A100）：适合 >100 万单元网格，规模越大优势越明显

---

### 3. 内存优化

```python
# 限制检查点频率（减少磁盘 I/O）
config = SteadyConfig(
    checkpoint_interval=500,  # 每 500 步保存一次（默认 100）
    output_dir="./results"
)

# 降低输出变量数量
api.export_vtk(result, variables=["pressure", "velocity"])  # 仅导出必要变量
```

---

### 4. 收敛加速技巧

```python
# 策略 1: 从低阶开始，逐步提升
result_1st = api.run_steady(grid, order=1, max_iter=1000)
result_2nd = api.resume_and_upgrade_order(result_1st, order=2, max_iter=3000)

# 策略 2: 自适应 CFL 数（已内置）
config = SteadyConfig(
    cfl_init=0.1,   # 从小 CFL 开始稳定启动
    cfl_max=10.0    # 允许 CFL 自动增长加速收敛
)

# 策略 3: 多重网格（未来版本）
# result = api.run_multigrid(grid, levels=3)
```

---

### 5. 批量仿真并行化

```python
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def run_single_simulation(params):
    """单个仿真任务"""
    api_local = AutoFlowCFDAPI()
    grid = api_local.load_grid(params['grid_file'])
    result = api_local.run_steady(grid, **params['config'])
    coeffs = api_local.calculate_coefficients(result)
    return {'params': params, 'coeffs': coeffs}

# 并行执行多个仿真
param_list = [
    {'grid_file': 'case1.nas', 'config': {'backend': 'gpu'}},
    {'grid_file': 'case2.nas', 'config': {'backend': 'gpu'}},
    {'grid_file': 'case3.nas', 'config': {'backend': 'gpu'}},
]

num_workers = min(len(param_list), multiprocessing.cpu_count())
with ProcessPoolExecutor(max_workers=num_workers) as executor:
    results = list(executor.map(run_single_simulation, param_list))

for r in results:
    print(f"Case: {r['params']['grid_file']}, Cd: {r['coeffs']['Cd']:.4f}")
```

---

## 完整示例

### 示例 1: 标准工作流程

```python
from autoflowcfd import AutoFlowCFDAPI
import matplotlib.pyplot as plt

# 1. 初始化
api = AutoFlowCFDAPI(verbose=True)

# 2. 加载网格
print("加载网格...")
grid = api.load_grid("car_model.nas")

# 3. 运行仿真
print("运行稳态仿真...")
result = api.run_steady(
    grid,
    backend="gpu",
    turbulence="sst_kw",
    order=2,
    max_iter=5000
)

# 4. 检查结果
print(f"收敛: {result.converged}")
print(f"迭代次数: {result.iterations}")

# 5. 计算气动系数
coeffs = api.calculate_coefficients(result)
print(f"Cd: {coeffs['Cd']:.4f}")
print(f"Cl: {coeffs['Cl']:.4f}")

# 6. 导出结果
api.export_vtk(result, output_file="result.vtu")
df_conv = api.export_convergence_history(result)

# 7. 绘制收敛曲线
plt.figure(figsize=(10, 6))
plt.semilogy(df_conv['iteration'], df_conv['residual'])
plt.xlabel('Iteration')
plt.ylabel('Residual')
plt.title('Convergence History')
plt.grid(True, alpha=0.3)
plt.savefig("convergence.png")
plt.show()

# 8. 生成报告
report = api.generate_report(result)
import json
with open("report.json", "w") as f:
    json.dump(report, f, indent=2)

print("✅ 仿真完成！")
```

---

### 示例 2: 瞬态 DES 仿真

```python
from autoflowcfd import AutoFlowCFDAPI

api = AutoFlowCFDAPI()
grid = api.load_grid("car_model.nas")

# 运行瞬态仿真
print("运行瞬态 DES 仿真...")
result = api.run_transient(
    grid,
    backend="gpu",
    turbulence="des",
    order=2,
    dt=1.0e-4,       # 时间步长 0.1ms
    total_time=0.1   # 总时间 0.1s
)

print(f"完成 {result.num_steps} 个时间步")
print(f"仿真时间: {result.current_time:.4f} s")
print(f"计算耗时: {result.execution_time:.2f} s")

# 导出所有时间步
api.export_all_vtk(result, output_dir="./transient_fields")

# 计算时均气动系数
mean_coeffs = api.calculate_mean_coefficients(result)
print(f"时均 Cd: {mean_coeffs['Cd']:.4f}")
```

---

## API 版本兼容性

| AutoFlowCFD 版本 | Python 版本 | API 稳定性 |
|-----------------|------------|-----------|
| 0.1.x | 3.10+ | Beta（可能有 breaking changes） |
| 0.2.x | 3.10+ | V2.0 系统改造版（当前版本） |
| 1.0.x | 3.10+ | LTS（长期支持，规划中） |

**注意**：在 1.0 版本之前，API 可能会有不兼容的变更。建议在 `pyproject.toml` 中锁定版本：

```toml
[tool.poetry.dependencies]
autoflowcfd = "==0.1.0"
```

---

## 📬 获取帮助

- **文档**: [docs/](../docs/) 目录
- **示例**: [examples/](../examples/) 目录
- **Issues**: [GitHub Issues](https://github.com/AutoFlowCFD/AutoFlowCFD/issues)
- **讨论**: [GitHub Discussions](https://github.com/AutoFlowCFD/AutoFlowCFD/discussions)
- **项目联系人**: Mr Lu
- **邮箱**: luxw_chd@126.com

---

**最后更新**: 2026-08-17  
**版本**: AutoFlowCFD v0.2.0 (V2.0 系统改造版)
