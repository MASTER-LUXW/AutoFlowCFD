# VTK导出快速参考卡

## 🚀 30秒快速开始

```python
from autoflowcfd.postprocess import VTKExporter

exporter = VTKExporter(grid_data=grid, solution=solution)
vtk_path = exporter.export("result.vtk", fields=['velocity', 'pressure'])
```

## 📁 常用命令

### Python API
```python
# 基础导出
exporter.export("output.vtk")

# 指定字段
exporter.export("output.vtk", fields=['velocity', 'pressure', 'k'])

# XML格式
exporter.export("output.vtu", format='xml')
```

### CLI命令行
```bash
autoflowcfd post export-vtk --case results/ --output result.vtk
```

## 🎨 ParaView查看步骤

1. **打开文件**: File → Open → 选择.vtk文件 → Apply
2. **速度云图**: 
   - 添加Slice过滤器
   - Coloring → Velocity → Magnitude
3. **压力分布**: 
   - 选择原始数据
   - Coloring → Pressure
   - View → Scalar Bar Visibility ✓

## 📊 支持字段

| 字段 | 类型 | 说明 |
|------|------|------|
| velocity | VECTORS | 速度矢量场 |
| pressure | SCALARS | 压力标量场 |
| k | SCALARS | 湍流动能 |
| omega | SCALARS | 比耗散率 |
| nut | SCALARS | 湍流粘度 |

## 💡 实用技巧

### 批量导出（瞬态）
```python
for i in range(0, total_steps, 50):  # 每50步
    vtk_path = f"results/step_{i:04d}.vtk"
    exporter.export(vtk_path, fields=['velocity', 'pressure'])
```

### 使用OutputPathManager
```python
manager = OutputPathManager(base_dir="./results", ...)
vtk_path = manager.get_field_path(iteration=100, format="vtk")
exporter.export(str(vtk_path), fields=['velocity', 'pressure'])
```

### 减小文件大小
```python
# 使用XML格式（压缩）
exporter.export("output.vtu", format='xml')

# 只导出必要字段
exporter.export("output.vtk", fields=['velocity'])
```

## ⚠️ 常见问题

**Q: VTK文件打不开？**  
A: 检查文件格式，确保使用ASCII或Binary格式正确

**Q: ParaView看不到数据？**  
A: 点击"Reset Camera"按钮，检查字段名称

**Q: 文件太大？**  
A: 使用.vtu格式，减少导出字段，降低导出频率

## 📚 完整文档

- 详细指南: `docs/VTK_EXPORT_GUIDE.md`
- 示例脚本: `examples/export_vtk_example.py`
- 测试脚本: `examples/simple_vtk_test.py`

---
**提示**: 运行 `python examples/simple_vtk_test.py` 生成测试VTK文件验证ParaView安装
