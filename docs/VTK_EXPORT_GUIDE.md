# VTK文件导出完整指南

## 📋 目录
- [快速开始](#快速开始)
- [方法一：Python API导出](#方法一python-api导出)
- [方法二：CLI命令行工具](#方法二cli命令行工具)
- [支持的字段类型](#支持的字段类型)
- [ParaView可视化步骤](#paraview可视化步骤)
- [常见问题解答](#常见问题解答)

---

## 🚀 快速开始

### 最简单的导出方式

```python
from autoflowcfd.postprocess import VTKExporter

# 假设已有grid_data和solution
exporter = VTKExporter(grid_data=grid, solution=solution)
vtk_path = exporter.export("result.vtk", fields=['velocity', 'pressure'])

print(f"VTK文件已导出: {vtk_path}")
print("请使用ParaView打开查看")
```

### 运行示例脚本

```bash
# 运行完整的VTK导出示例
python examples/export_vtk_example.py

# 输出文件位于 ./vtk_output/ 目录
# - velocity_field.vtk
# - pressure_field.vtk
# - full_flow_field.vtk
```

---

## 方法一：Python API导出

### 1. 基础用法

```python
from autoflowcfd import AutoFlowCFDAPI
from autoflowcfd.postprocess import VTKExporter

# 步骤1: 加载网格
api = AutoFlowCFDAPI()
grid = api.load_grid("sedan.nas")

# 步骤2: 运行仿真（或使用已有结果）
result = api.run_steady(
    grid_data=grid,
    backend="gpu",
    order=3,
    turbulence="sst_kw",
    max_iter=1000
)

# 步骤3: 创建导出器并导出
exporter = VTKExporter(
    grid_data=grid,
    solution=result.solution
)

# 导出所有可用字段
vtk_path = exporter.export(
    output_path="results/flow_field.vtk",
    fields=['velocity', 'pressure'],
    format='legacy'  # 或 'xml'
)
```

### 2. 导出指定字段

```python
# 仅导出速度场
exporter.export("velocity_only.vtk", fields=['velocity'])

# 导出速度和压力
exporter.export("vel_pres.vtk", fields=['velocity', 'pressure'])

# 导出完整湍流场
exporter.export("full_turb.vtk", 
                fields=['velocity', 'pressure', 'k', 'omega'])
```

### 3. 使用OutputPathManager管理路径

```python
from autoflowcfd.config import OutputPathManager

# 创建输出管理器
manager = OutputPathManager(
    base_dir="./results",
    grid_file="sedan.nas",
    mode="steady",
    turbulence="sst_kw",
    order=3,
    backend="gpu"
)
manager.create_directories()

# 获取自动生成的VTK路径
vtk_path = manager.get_field_path(iteration=500, format="vtk")

# 导出到指定路径
exporter.export(str(vtk_path), fields=['velocity', 'pressure'])
```

### 4. 瞬态仿真批量导出

```python
# 在瞬态求解循环中定期导出
output_interval = 50  # 每50步导出一次

for step in range(total_steps):
    # ... 求解步骤 ...
    
    if step % output_interval == 0:
        vtk_path = f"transient_results/step_{step:04d}.vtk"
        exporter = VTKExporter(grid_data=grid, solution=current_solution)
        exporter.export(vtk_path, fields=['velocity', 'pressure'])
        print(f"Exported: {vtk_path}")
```

---

## 方法二：CLI命令行工具

### 基本用法

```bash
autoflowcfd post export-vtk \
  --case results/steady_simulation/ \
  --output flow_field.vtk \
  --variables velocity pressure
```

### 导出指定变量

```bash
# 仅导出速度
autoflowcfd post export-vtk \
  --case results/ \
  --output velocity.vtk \
  --variables velocity

# 导出多个变量
autoflowcfd post export-vtk \
  --case results/ \
  --output full.vtk \
  --variables velocity pressure k omega
```

### 指定时间步（瞬态）

```bash
autoflowcfd post export-vtk \
  --case transient_results/ \
  --output step_100.vtk \
  --time-step 100
```

---

## 支持的字段类型

| 字段名 | 类型 | 描述 | VTK数据类型 |
|--------|------|------|-------------|
| `velocity` | 矢量 | 速度场 (u, v, w) | VECTORS |
| `pressure` | 标量 | 静压场 | SCALARS |
| `k` | 标量 | 湍流动能 | SCALARS |
| `omega` | 标量 | 比耗散率 | SCALARS |
| `nut` | 标量 | 湍流粘度 | SCALARS |

### 字段组合建议

**汽车外流场分析推荐：**
```python
# 基础分析
fields=['velocity', 'pressure']

# 湍流分析
fields=['velocity', 'pressure', 'k', 'omega']

# 完整分析
fields=['velocity', 'pressure', 'k', 'omega', 'nut']
```

---

## ParaView可视化步骤

### 1. 安装ParaView

- **下载地址**: https://www.paraview.org/download/
- **支持平台**: Windows, macOS, Linux
- **推荐版本**: 5.11 或更高

### 2. 打开VTK文件

```
1. 启动ParaView
2. File → Open
3. 选择导出的 .vtk 文件
4. 点击 "Apply" 按钮加载数据
```

### 3. 查看速度云图

#### 方法A：对称面速度云图

```
1. 在Pipeline Browser中选择数据集
2. 工具栏点击 "Slice" 创建切片
3. Properties面板设置：
   - Slice Type: Plane
   - Origin: (2.5, 0, 0.75)  # 根据模型调整
   - Normal: (0, 1, 0)       # YZ平面（对称面）
4. 点击 Apply
5. Coloring下拉框选择: Velocity → Magnitude
6. 调整颜色映射：
   - Edit → Color Map Editor
   - 选择配色方案（Jet/Rainbow）
   - 调整数值范围
```

#### 方法B：多截面速度分布

```
1. 应用多个Slice过滤器
2. 分别设置不同位置的截面：
   - X=1.0 (前部)
   - X=2.5 (中部)
   - X=4.0 (尾部)
3. 每个截面单独设置Coloring为Velocity
4. 使用Render View同时显示所有截面
```

#### 方法C：速度矢量箭头

```
1. 应用 Glyph 过滤器
2. 设置：
   - Glyph Type: Arrow
   - Scale Factor: 0.1 (调整箭头大小)
   - Mask Points: 10 (降低密度)
3. Coloring: Velocity (Magnitude)
4. Apply
```

### 4. 查看表面压力分布

```
1. 在Pipeline Browser中选择原始数据集
2. 确保显示模式为 "Surface"
3. Coloring下拉框选择: Pressure
4. 启用Scalar Bar：
   - View → Scalar Bar Visibility ✓
5. 调整压力范围：
   - Rescale to Data Range (自动)
   - 或手动设置Min/Max
6. 选择合适的颜色映射（如Blue to Red）
```

**提取车身表面压力：**

```
1. 应用 Threshold 过滤器
2. Scalars: 选择边界标识字段（如果有）
3. 设置阈值筛选车身边界单元
4. 或使用 Extract Surface 提取外表面
5. Coloring: Pressure
```

### 5. 高级可视化技巧

#### 流线图（Streamlines）

```
1. Filters → Alphabetical → Stream Tracer
2. Seed Type: Point Source 或 Line Source
3. 设置种子点位置和数量
4. Integrator Type: Runge-Kutta 4/5
5. Maximum Streamline Length: 5.0
6. Coloring: Velocity Magnitude
7. Apply
```

#### 涡量等值面

```
1. 应用 Calculator 过滤器
2. Result Array Name: Vorticity
3. Expression: curl(Velocity)
4. 应用 Contour 过滤器
5. Scalars: Vorticity
6. 设置等值面值（如50, 100, 200）
7. Coloring: Vorticity Magnitude
```

#### Q准则识别涡结构

```
1. Calculator计算Q准则
   Result Array Name: Q_criterion
   Expression: 需要自定义公式
2. Contour过滤器提取Q>0区域
3. 半透明显示观察涡结构
   - Opacity: 0.3-0.5
```

### 6. 导出图片和动画

**保存截图：**
```
File → Save Screenshot
- 选择格式：PNG/JPG/TIFF
- 设置分辨率：1920x1080 或更高
- 保存
```

**导出动画：**
```
File → Save Animation
- 选择格式：MP4/AVI/Ogg Theora
- 设置帧率和分辨率
- 保存
```

---

## 常见问题解答

### Q1: VTK文件太大怎么办？

**解决方案：**
```python
# 1. 使用XML格式（支持压缩）
exporter.export("result.vtu", format='xml')

# 2. 只导出需要的字段
exporter.export("result.vtk", fields=['velocity'])

# 3. 降低导出频率（瞬态）
if step % 100 == 0:  # 每100步导出，而非每步
    exporter.export(...)
```

### Q2: ParaView打开VTK文件后看不到数据？

**检查清单：**
- [ ] 确认VTK文件格式正确（ASCII/Binary）
- [ ] 检查节点和单元数量是否匹配
- [ ] 确认字段名称拼写正确
- [ ] 尝试在ParaView中点击 "Reset Camera"

### Q3: 如何查看特定区域的流场？

**方法：**
```python
# 1. 在ParaView中使用Clip过滤器裁剪区域
# 2. 或使用Threshold按坐标范围筛选
# 3. 或在导出前预处理网格，只保留感兴趣区域
```

### Q4: 速度/压力值为0或不合理？

**原因：**
- 当前VTK导出器中的字段提取是占位符实现
- 需要从实际的SolutionVector中提取真实数据

**解决方案：**
```python
# 等待VTKExporter完善，或手动提取数据
# 参考 examples/export_vtk_example.py 中的示例
```

### Q5: 如何批量处理多个VTK文件？

**Python脚本示例：**
```python
import glob
from pathlib import Path

vtk_files = sorted(glob.glob("results/*.vtk"))

for vtk_file in vtk_files:
    # 可以在这里添加自动化处理逻辑
    print(f"Processing: {vtk_file}")
    # 例如：提取统计数据、生成缩略图等
```

### Q6: VTK和VTU格式有什么区别？

| 特性 | Legacy (.vtk) | XML (.vtu) |
|------|---------------|------------|
| 格式 | ASCII文本 | XML + Binary |
| 文件大小 | 较大 | 较小（可压缩） |
| 兼容性 | 更好 | 现代软件支持 |
| 读写速度 | 较慢 | 较快 |
| 推荐场景 | 小规模数据 | 大规模数据 |

---

## 📚 相关资源

- **ParaView官方文档**: https://docs.paraview.org/
- **VTK文件格式规范**: https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf
- **AutoFlowCFD后处理模块**: `src/autoflowcfd/postprocess/`
- **示例脚本**: `examples/export_vtk_example.py`

---

## 💡 最佳实践

1. **导出前验证数据**
   ```python
   # 检查solution是否有效
   assert solution.rho is not None
   assert len(solution.rho) == grid.metadata.cell_count
   ```

2. **使用有意义的文件名**
   ```python
   vtk_path = f"results/ahmed_body_Re1e6_step{iteration:04d}.vtk"
   ```

3. **记录导出元数据**
   ```python
   # 保存导出配置到JSON
   metadata = {
       'grid_file': 'sedan.nas',
       'iteration': 500,
       'fields': ['velocity', 'pressure'],
       'timestamp': datetime.now().isoformat()
   }
   ```

4. **定期清理旧文件**
   ```python
   # 只保留最近N个VTK文件
   vtk_files = sorted(Path("results").glob("*.vtk"))
   for old_file in vtk_files[:-10]:  # 保留最后10个
       old_file.unlink()
   ```

---

## ⚠️ 注意事项

1. **当前实现状态**：VTKExporter中的字段提取部分尚未完全实现，当前为占位符数据。完整功能将在后续迭代中完善。

2. **内存考虑**：千万级网格的VTK文件可能达到GB级别，建议：
   - 使用XML格式
   - 只导出必要字段
   - 分批导出

3. **性能优化**：对于大规模数据，考虑：
   - 并行导出（多进程）
   - 增量导出（仅变化字段）
   - 使用HDF5中间格式

---

**最后更新**: 2026-07-31  
**维护者**: AutoFlowCFD Team
