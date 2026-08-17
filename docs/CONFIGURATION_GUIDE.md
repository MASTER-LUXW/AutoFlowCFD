# AutoFlowCFD 配置指南

本文档详细介绍 AutoFlowCFD 的配置系统，包括 YAML 配置文件格式、命令行参数说明以及最佳实践建议。

---

## 📋 目录

- [配置方式概述](#配置方式概述)
- [YAML 配置文件](#yaml-配置文件)
  - [基础配置](#基础配置)
  - [求解器配置](#求解器配置)
  - [边界条件配置](#边界条件配置)
  - [数值方法配置](#数值方法配置)
  - [输出配置](#输出配置)
- [命令行参数](#命令行参数)
- [配置优先级](#配置优先级)
- [配置示例](#配置示例)
- [最佳实践](#最佳实践)

---

## 配置方式概述

AutoFlowCFD 支持三种配置方式：

1. **YAML 配置文件**（推荐）：适合复杂配置，可版本管理
2. **命令行参数**：适合快速测试和简单场景
3. **Python API**：适合程序化配置和动态调整

**优先级**：命令行参数 > YAML 配置 > 默认值

---

## YAML 配置文件

### 基础配置

```yaml
# config.yaml - 基础配置示例

# 仿真基本信息
simulation:
  name: "Ahmed Body Steady RANS"
  description: "Steady-state RANS simulation of Ahmed body at 30 m/s"
  version: "1.0"

# 网格文件
grid:
  file: "car_model.nas"
  validate: true              # 是否校验网格质量
  unit: "mm"                  # 网格单位 (mm/m)
  
# 计算后端
compute:
  backend: "gpu"              # "cpu" 或 "gpu"
  device_id: 0                # GPU 设备 ID（多 GPU 时指定）
  threads: 8                  # CPU 线程数（仅 CPU 模式）
```

---

### 求解器配置

#### 稳态求解器

```yaml
solver:
  type: "steady"              # "steady" 或 "transient"
  
  # FR 离散格式
  fr_order: 2                 # FR 阶数: 1, 2, 3
  
  # 湍流模型
  turbulence_model: "sst_kw"  # "sst_kw", "des", "ddes"
  
  # 收敛控制
  max_iterations: 5000        # 最大迭代次数
  convergence_tolerance: 1.0e-6  # 收敛容差（残差）
  
  # CFL 数控制
  cfl:
    initial: 0.1              # 初始 CFL 数
    maximum: 5.0              # 最大 CFL 数
    adaptive: true            # 启用自适应 CFL
  
  # 检查点
  checkpoint:
    enabled: true
    interval: 100             # 每 100 步保存一次
    directory: "./checkpoints"
```

#### 瞬态求解器

```yaml
solver:
  type: "transient"
  
  # FR 离散格式
  fr_order: 2
  
  # 湍流模型
  turbulence_model: "ddes"
  
  # 时间离散
  time_integration:
    scheme: "BE"              # "BE" (Backward Euler), "RK2", "AB3"
    dt: 1.0e-4                # 时间步长 (秒)
    total_time: 0.1           # 总仿真时间 (秒)
    num_steps: 1000           # 时间步总数（自动计算，可与 dt/total_time 互斥）
  
  # 瞬态统计
  statistics:
    enabled: true
    start_time: 0.05          # 开始统计的时间（前 0.05s 为过渡期）
    interval: 10              # 每 10 步采样一次
  
  # 检查点
  checkpoint:
    enabled: true
    interval: 50
    directory: "./checkpoints"
```

---

### 边界条件配置

#### 标准边界条件

```yaml
boundary_conditions:
  # 速度入口
  inlet:
    type: "INLET"
    velocity:
      x: 30.0                 # m/s
      y: 0.0
      z: 0.0
    pressure: 101325.0        # Pa
    turbulence:
      ke: 1.0                 # 湍流动能 (m²/s²)
      omega: 100.0            # 比耗散率 (1/s)
  
  # 压力出口
  outlet:
    type: "OUTLET"
    pressure: 101325.0        # Pa
    backflow_turbulence:
      ke: 0.1
      omega: 10.0
  
  # 固定壁面（车身）
  car_body:
    type: "WALL"
    wall_function: "enhanced" # "standard", "enhanced", "low_re"
    roughness: 0.0            # 表面粗糙度 (mm)
  
  # 移动地面（模拟车轮旋转）
  ground:
    type: "WALL"
    moving: true
    velocity:
      x: 30.0                 # m/s
      y: 0.0
      z: 0.0
    wall_function: "enhanced"
  
  # 对称面
  symmetry_plane:
    type: "SYMMETRY"
  
  # 远场边界
  farfield:
    type: "FARFIELD"
    velocity:
      x: 30.0
      y: 0.0
      z: 0.0
    pressure: 101325.0
```

#### 边界组映射

```yaml
# 将 NAS 文件中的边界组名映射到边界条件
boundary_mapping:
  "INLET_GROUP": "inlet"
  "OUTLET_GROUP": "outlet"
  "CAR_BODY": "car_body"
  "GROUND": "ground"
  "SYMMETRY": "symmetry_plane"
  "FARFIELD": "farfield"
```

---

### 数值方法配置

```yaml
numerical_methods:
  # 梯度计算
  gradient:
    method: "least_squares"   # "green_gauss", "least_squares"
    limiter: "venkatakrishnan" # "none", "venkatakrishnan", "barth_jespersen"
  
  # 通量计算
  flux:
    riemann_solver: "roe"     # "roe", "hllc", "ausm"
    entropy_fix: true         # 熵修正（防止膨胀波异常）
  
  # 粘性通量
  viscous:
    enabled: true
    method: "central"         # 中心差分
  
  # 低马赫预处理（可选，用于低速流动）
  low_mach_preconditioning:
    enabled: false
    reference_velocity: 30.0  # m/s
```

---

### 输出配置

```yaml
output:
  # 输出目录
  directory: "./results"
  
  # 流场输出
  fields:
    enabled: true
    format: "vtk"             # "vtk", "hdf5"
    interval: 100             # 每 100 步输出一次
    variables:                # 输出变量
      - "pressure"
      - "velocity"
      - "vorticity"
      - "turbulence_ke"
      - "wall_shear_stress"
  
  # 气动系数历史
  coefficients:
    enabled: true
    format: "csv"
    filename: "coefficients.csv"
  
  # 收敛历史
  convergence:
    enabled: true
    format: "csv"
    filename: "convergence.csv"
    variables:
      - "iteration"
      - "residual"
      - "Cd"
      - "Cl"
      - "Cs"
  
  # 最终报告
  report:
    enabled: true
    format: "json"
    filename: "summary.json"
  
  # 日志
  logging:
    level: "INFO"             # "DEBUG", "INFO", "WARNING", "ERROR"
    file: "autoflowcfd.log"
    console: true
```

---

## 命令行参数

### 基本用法

```bash
# 稳态仿真
poetry run autoflowcfd solve steady <volume_mesh.pkl> [OPTIONS]

# 瞬态仿真
poetry run autoflowcfd solve transient <volume_mesh.pkl> [OPTIONS]
```

### 常用参数

#### 求解器参数

```bash
# 稳态仿真
poetry run autoflowcfd solve steady volume_mesh.pkl \
    --backend cpu \
    --turbulence-model sst \
    --order 2 \
    --max-iter 5000 \
    -o ./results

# 瞬态仿真
poetry run autoflowcfd solve transient volume_mesh.pkl \
    --backend cpu \
    --turbulence-model ddes \
    --order 2 \
    --time-method dual-time \
    --dt 1e-4 \
    --physical-time 0.1 \
    -o ./results
```

#### 边界条件参数

```bash
# 指定入口速度（通过配置文件）
# 在 config.yaml 中设置边界条件

# 指定输出目录
poetry run autoflowcfd solve steady volume_mesh.pkl \
    -o ./my_results

# 指定检查点间隔
poetry run autoflowcfd solve steady volume_mesh.pkl \
    --checkpoint-interval 200
```

#### 其他参数

```bash
# 详细日志
poetry run autoflowcfd solve steady volume_mesh.pkl -v

# 使用配置文件
poetry run autoflowcfd solve steady volume_mesh.pkl -c config.yaml

# 从检查点恢复
poetry run autoflowcfd solve resume ./checkpoints/latest_checkpoint.h5 \
    --max-iter 10000
```

### 完整参数列表

运行以下命令查看所有可用参数：

```bash
poetry run autoflowcfd solve steady --help
poetry run autoflowcfd solve transient --help
```

---

## 配置优先级

配置的优先级从高到低为：

1. **命令行参数**（最高优先级）
2. **YAML 配置文件**
3. **API 调用参数**
4. **默认值**（最低优先级）

**示例**：

```yaml
# config.yaml
solver:
  max_iterations: 5000
  backend: "cpu"
```

```bash
# 命令行覆盖 YAML 配置
poetry run autoflowcfd solve steady volume_mesh.pkl \
    -c config.yaml \
    --max-iter 10000 \      # 覆盖 YAML 中的 5000
    --backend gpu           # 覆盖 YAML 中的 cpu
```

最终生效配置：`max_iterations=10000`, `backend="gpu"`

---

## 配置示例

### 示例 1: 标准稳态 RANS 仿真

```yaml
# configs/steady_rans.yaml

simulation:
  name: "Standard Steady RANS"

grid:
  file: "car_model.nas"
  validate: true

compute:
  backend: "gpu"
  device_id: 0

solver:
  type: "steady"
  fr_order: 2
  turbulence_model: "sst_kw"
  max_iterations: 5000
  convergence_tolerance: 1.0e-6
  cfl:
    initial: 0.1
    maximum: 5.0
    adaptive: true

boundary_conditions:
  inlet:
    type: "INLET"
    velocity: {x: 30.0, y: 0.0, z: 0.0}
    pressure: 101325.0
  
  outlet:
    type: "OUTLET"
    pressure: 101325.0
  
  car_body:
    type: "WALL"
    wall_function: "enhanced"
  
  ground:
    type: "WALL"
    moving: true
    velocity: {x: 30.0, y: 0.0, z: 0.0}

output:
  directory: "./results/steady_rans"
  fields:
    enabled: true
    interval: 100
  coefficients:
    enabled: true
  convergence:
    enabled: true
```

**运行命令**：
```bash
poetry run autoflowcfd solve steady volume_mesh.pkl -c configs/steady_rans.yaml
```

---

### 示例 2: 瞬态 DDES 仿真

```yaml
# configs/transient_ddes.yaml

simulation:
  name: "Transient DDES Simulation"

grid:
  file: "car_model.nas"
  validate: true

compute:
  backend: "gpu"
  device_id: 0

solver:
  type: "transient"
  fr_order: 2
  turbulence_model: "ddes"
  time_integration:
    scheme: "BE"
    dt: 1.0e-4
    total_time: 0.1
  statistics:
    enabled: true
    start_time: 0.05
    interval: 10

boundary_conditions:
  inlet:
    type: "INLET"
    velocity: {x: 30.0, y: 0.0, z: 0.0}
    pressure: 101325.0
    turbulence:
      ke: 1.0
      omega: 100.0
  
  outlet:
    type: "OUTLET"
    pressure: 101325.0
  
  car_body:
    type: "WALL"
    wall_function: "enhanced"
  
  ground:
    type: "WALL"
    moving: true
    velocity: {x: 30.0, y: 0.0, z: 0.0}

output:
  directory: "./results/transient_ddes"
  fields:
    enabled: true
    interval: 10
    variables:
      - "pressure"
      - "velocity"
      - "vorticity"
  coefficients:
    enabled: true
  convergence:
    enabled: false  # 瞬态仿真不监控残差收敛
```

**运行命令**：
```bash
poetry run autoflowcfd solve transient volume_mesh.pkl -c configs/transient_ddes.yaml
```

---

### 示例 3: 参数扫描配置

```yaml
# configs/param_sweep.yaml
# 用于批量仿真，通过脚本修改关键参数

base_config:
  grid:
    file: "car_model.nas"
  
  compute:
    backend: "gpu"
  
  solver:
    type: "steady"
    fr_order: 2
    turbulence_model: "sst_kw"
    max_iterations: 3000
  
  boundary_conditions:
    inlet:
      type: "INLET"
      velocity: {x: 30.0, y: 0.0, z: 0.0}  # 此参数将在脚本中修改
      pressure: 101325.0
    
    outlet:
      type: "OUTLET"
      pressure: 101325.0
    
    car_body:
      type: "WALL"
      wall_function: "enhanced"

# Python 脚本示例
"""
import yaml
from autoflowcfd import AutoFlowCFDAPI

api = AutoFlowCFDAPI()

# 加载基础配置
with open("configs/param_sweep.yaml") as f:
    base_config = yaml.safe_load(f)["base_config"]

# 参数扫描
velocities = [20.0, 25.0, 30.0, 35.0, 40.0]

for vel in velocities:
    # 修改配置
    base_config["boundary_conditions"]["inlet"]["velocity"]["x"] = vel
    
    # 保存临时配置
    temp_config = f"config_vel_{vel}.yaml"
    with open(temp_config, "w") as f:
        yaml.dump({"boundary_conditions": base_config["boundary_conditions"]}, f)
    
    # 运行仿真
    print(f"\n运行速度: {vel} m/s")
    grid = api.load_grid(base_config["grid"]["file"])
    result = api.run_steady(grid, backend="gpu", max_iter=3000)
    coeffs = api.calculate_coefficients(result)
    
    print(f"Cd: {coeffs['Cd']:.4f}")
"""
```

---

## 最佳实践

### 1. 配置文件组织

```
project/
├── configs/
│   ├── base.yaml              # 基础配置模板
│   ├── steady_rans.yaml       # 稳态 RANS 配置
│   ├── transient_des.yaml     # 瞬态 DES 配置
│   └── param_sweep/           # 参数扫描配置目录
│       ├── case_01.yaml
│       ├── case_02.yaml
│       └── ...
├── grids/
│   └── car_model.nas
└── results/
    ├── steady_rans/
    └── transient_des/
```

---

### 2. 使用配置模板

创建基础配置模板，通过继承或覆盖减少重复：

```yaml
# configs/base_template.yaml
compute:
  backend: "gpu"

solver:
  fr_order: 2
  max_iterations: 5000
  convergence_tolerance: 1.0e-6

boundary_conditions:
  inlet:
    type: "INLET"
    velocity: {x: 30.0, y: 0.0, z: 0.0}
    pressure: 101325.0
  
  outlet:
    type: "OUTLET"
    pressure: 101325.0
```

在具体配置中引用：

```yaml
# configs/my_case.yaml
include: "base_template.yaml"  # 未来版本支持

# 仅覆盖需要修改的部分
solver:
  max_iterations: 10000

boundary_conditions:
  inlet:
    velocity: {x: 40.0, y: 0.0, z: 0.0}  # 更高速度
```

---

### 3. 验证配置文件

运行前验证配置文件语法：

```bash
# 验证配置（未来版本功能）
poetry run autoflowcfd config validate configs/steady_rans.yaml
```

或在 Python 中验证：

```python
import yaml
from autoflowcfd.config import SteadyConfig

with open("configs/steady_rans.yaml") as f:
    config_dict = yaml.safe_load(f)

try:
    config = SteadyConfig.from_dict(config_dict["solver"])
    print("✅ 配置验证通过")
except Exception as e:
    print(f"❌ 配置错误: {e}")
```

---

### 4. 注释与文档

在配置文件中添加注释说明关键参数：

```yaml
solver:
  max_iterations: 5000        # 对于 100 万单元网格，通常需要 3000-5000 步收敛
  convergence_tolerance: 1.0e-6  # 工程精度要求，学术研究可设为 1.0e-8
  
  cfl:
    initial: 0.1              # 从小 CFL 开始确保稳定性
    maximum: 5.0              # 自适应 CFL 上限，加速收敛
```

---

### 5. 版本控制

将配置文件纳入 Git 版本控制：

```bash
git add configs/*.yaml
git commit -m "Add steady RANS configuration for Ahmed body"
```

忽略结果目录：

```gitignore
# .gitignore
results/
checkpoints/
*.log
```

---

### 6. 性能调优配置

#### CPU 优化

```yaml
compute:
  backend: "cpu"
  threads: 16                 # 设置为物理核心数
  
numerical_methods:
  gradient:
    method: "green_gauss"     # CPU 上更快
```

#### GPU 优化

```yaml
compute:
  backend: "gpu"
  device_id: 0                # 指定 GPU 设备
  
solver:
  fr_order: 3                 # GPU 可高效处理高阶格式
  
numerical_methods:
  gradient:
    method: "least_squares"   # GPU 上精度更高
```

---

### 7. 调试配置

遇到问题时启用详细日志：

```yaml
output:
  logging:
    level: "DEBUG"            # 最详细日志
    file: "debug.log"
    console: true

solver:
  checkpoint:
    interval: 10              # 频繁保存检查点，便于恢复
```

运行后查看日志：

```bash
tail -f debug.log
```

---

## 常见问题

### Q1: 如何选择合适的 CFL 数？

**A**: 
- **初始 CFL**：从 0.05-0.1 开始，确保稳定性
- **最大 CFL**：稳态仿真可设为 5-10，瞬态仿真通常固定 CFL
- **自适应**：启用自适应 CFL 可自动调整，加速收敛

### Q2: 为什么仿真不收敛？

**A**: 检查以下配置：
- 降低初始 CFL 数（`cfl.initial: 0.05`）
- 增加最大迭代次数（`max_iterations: 10000`）
- 检查网格质量（`grid.validate: true`）
- 确认边界条件合理（特别是入口/出口）

### Q3: CPU 和 GPU 应该如何选择？

**A**:
- **CPU**：小网格（<100 万单元）、资源受限、无 GPU 硬件
- **GPU**：大网格（>100 万单元）、高精度需求、有 NVIDIA GPU

### Q4: 如何平衡精度和速度？

**A**:
- **快速预览**：`fr_order: 1`, `max_iterations: 1000`
- **工程开发**：`fr_order: 2`, `max_iterations: 5000`
- **高精度研究**：`fr_order: 3`, `max_iterations: 10000`

---

## 参考资源

- [API 参考](API.md) - Python API 详细说明
- [快速开始](../QUICKSTART.md) - 安装与首次运行
- [架构设计](../ARCHITECTURE.md) - 系统架构说明
- [算例教程](../examples/) - 完整配置示例

---

**最后更新**: 2026-08-17  
**版本**: AutoFlowCFD v0.2.0 (V2.0 系统改造版)
