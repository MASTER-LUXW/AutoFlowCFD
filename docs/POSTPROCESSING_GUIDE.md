# AutoFlowCFD 后处理模块使用指南

## 概述

AutoFlowCFD后处理模块提供完整的CFD仿真结果分析工具，包括气动系数计算、可视化数据导出、收敛分析和瞬态统计等功能。

## 快速开始

### 1. 气动系数计算

```python
from autoflowcfd.postprocess import CoefficientCalculator

# 创建计算器
calc = CoefficientCalculator(
    grid_data,
    solution,
    reference_area=2.2,      # 参考面积 (m²)
    reference_length=4.5,    # 参考长度 (m)
    density=1.225,           # 空气密度 (kg/m³)
    velocity=30.0            # 来流速度 (m/s)
)

# 计算气动系数
coeffs = calc.calculate()
print(f"Cd = {coeffs.Cd:.4f}")
print(f"Cl = {coeffs.Cl:.4f}")
print(f"Cm = {coeffs.Cm:.4f}")

# 计算绝对力值
forces = calc.calculate_forces()
print(f"Drag force: {forces.drag_force:.1f} N")
```

### 2. VTK场数据导出

```python
from autoflowcfd.postprocess import VTKExporter

# 创建导出器
exporter = VTKExporter(grid_data, solution)

# 导出VTK文件（ParaView可视化）
vtk_path = exporter.export(
    "result.vtk",
    fields=['velocity', 'pressure', 'k', 'omega']
)
```

### 3. 收敛分析

```python
from autoflowcfd.postprocess import ConvergenceAnalyzer, SimulationReport

# 创建分析器
analyzer = ConvergenceAnalyzer()

# 在迭代循环中添加数据
for i in range(max_iterations):
    # ... solver step ...
    analyzer.add_iteration(
        iteration=i+1,
        residuals={'continuity': res_cont, 'momentum': res_mom},
        cfl=cfl_value
    )

# 导出CSV收敛曲线
analyzer.export_csv("convergence.csv")

# 生成JSON报告
report = SimulationReport(config, analyzer)
report.generate("report.json", computation_time=3600.0)
```

### 4. 瞬态统计后处理

```python
from autoflowcfd.postprocess import TransientStatistics, PressurePSD

# 创建统计计算器
stats = TransientStatistics(grid_data, window_size=100)

# 在时间推进循环中累积样本
for step in range(num_steps):
    current_time = step * dt
    # ... solver step ...
    
    # 跳过初始过渡期后开始采样
    if current_time > 0.1:
        stats.accumulate(solution, time=current_time)

# 计算时均和RMS
result = stats.compute_statistics()
print(f"Mean fields: {list(result.mean_fields.keys())}")
print(f"RMS fields: {list(result.rms_fields.keys())}")

# PSD频谱分析
monitor_points = [(0.5, 0.0, 0.1)]
psd = PressurePSD(monitor_points, dt)

for step in range(num_steps):
    pressure = get_pressure_at_point(...)
    psd.add_sample(time=step*dt, pressures=[pressure])

freqs, psd_vals = psd.compute_psd(0)
dom_freq, peak_psd = psd.find_dominant_frequency(0, min_freq=50, max_freq=150)
print(f"Dominant frequency: {dom_freq:.2f} Hz")
```

## Demo算例运行

### Ahmed Body稳态RANS

```bash
cd examples/ahmed_body/steady
python run_steady.py
```

输出：
- `output/result.vtk` - VTK场数据
- `output/convergence.csv` - 收敛曲线
- `output/report.json` - 仿真报告

### Ahmed Body瞬态DES

```bash
cd examples/ahmed_body/transient
python run_transient.py
```

输出：
- `output/result.vtk` - VTK场数据
- `output/convergence.csv` - 收敛曲线
- `output/report.json` - 仿真报告
- 瞬态统计数据（时均、RMS、PSD）

## 性能基准测试

### CPU基准测试

```bash
python benchmarks/benchmark_cpu.py
```

目标性能（8核i7，百万级网格）：
- 稳态RANS: ≥50 iterations/minute
- 瞬态DES: ≥10 iterations/minute

### GPU基准测试

```bash
python benchmarks/benchmark_gpu.py
```

目标性能（RTX 3090，百万级网格）：
- 稳态RANS: ≥200 iterations/minute
- 瞬态DES: ≥50 iterations/minute
- 瞬态LES: ≥20 iterations/minute

## API参考

### CoefficientCalculator

**主要方法**:
- `calculate()` → AerodynamicCoefficients: 计算无量纲系数
- `calculate_forces()` → AerodynamicForces: 计算绝对力和力矩
- `calculate_by_boundary(boundary_name)` → AerodynamicCoefficients: 分边界计算

### VTKExporter

**主要方法**:
- `export(output_path, fields, format)` → Path: 导出VTK文件

**支持字段**: velocity, pressure, k, omega, nut

**支持格式**: legacy (ASCII), xml (VTU, 待完善)

### ConvergenceAnalyzer

**主要方法**:
- `add_iteration(iteration, residuals, cfl, coefficients)`: 添加迭代数据
- `export_csv(output_path)` → Path: 导出CSV
- `get_summary(computation_time)` → SimulationSummary: 获取摘要

### TransientStatistics

**主要方法**:
- `accumulate(solution, time)`: 累积样本
- `compute_statistics()` → TransientResult: 计算统计量
- `get_sampling_info()` → Dict: 获取采样信息

### PressurePSD

**主要方法**:
- `add_sample(time, pressures)`: 添加压力样本
- `compute_psd(point_index)` → Tuple[freqs, psd]: 计算功率谱
- `find_dominant_frequency(point_index, min_freq, max_freq)` → Tuple[freq, psd]: 找主导频率

## 已知限制

1. **压力积分方法**: 当前为占位实现，返回固定值。需实现真正的表面积分逻辑。
2. **VTK场数据提取**: 速度和压力为均匀场占位值。需从SolutionVector提取真实数据。
3. **XML VTK格式**: `_export_xml()`方法未完全实现，目前回退到Legacy格式。
4. **分边界计算**: `calculate_by_boundary()`返回零值，需实现边界单元过滤。

## 下一步计划

- [ ] 实现真正的压力积分方法
- [ ] 完善VTK场数据提取逻辑
- [ ] 完成XML VTU格式支持
- [ ] 优化大规模数据导出性能
- [ ] 添加更多可视化后处理功能（流线、等值面、Q准则）

## 相关文档

- [接口文档-Part2](ProjectFiles/2-4_接口文档-Part2.md): 后处理API详细规范
- [数据结构设计文档-Part2](ProjectFiles/2-3_数据结构设计文档-Part2.md): SolutionVector结构
- [迭代5开发报告](ProjectFiles/5-1_迭代5开发报告.md): 完整开发总结
