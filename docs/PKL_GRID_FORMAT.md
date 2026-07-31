# PKL格式体网格文件说明

## 📦 概述

AutoFlowCFD在仿真计算过程中会自动保存**体网格对象**为 **PKL格式**（Python Pickle序列化）。

**文件位置：**
```
results/case_name/volume_mesh.pkl
```

---

## 🔍 为什么使用PKL格式？

### **传统流程的问题**

```
面网格(.nas) → 体网格生成 → CFD计算 → VTK导出
                    ↑
              需要重新生成！
```

如果使用原始 [.nas](file://d:\myWorkspace\AutoFlowCFD\examples\ahmed_body_demo.nas) 文件导出VTK：
1. ❌ 需要重新从面网格生成体网格
2. ❌ 随机种子不同可能导致网格差异
3. ❌ 加载速度慢（解析+生成需要数秒）
4. ❌ 可能与原始仿真的体网格不完全一致

### **PKL格式的优势**

```
体网格(PKL) → CFD计算 → VTK导出
    ↑              ↑
  直接使用    完全一致！
```

使用 `volume_mesh.pkl` 导出VTK：
1. ✅ 直接加载已生成的体网格
2. ✅ 保证与仿真使用的网格100%一致
3. ✅ 加载速度快（毫秒级）
4. ✅ 避免重新生成的不确定性

---

## 💻 技术实现

### **保存过程**

在 [`solve_commands.py`](d:\myWorkspace\AutoFlowCFD\src\autoflowcfd\cli\solve_commands.py) 中：

```python
import pickle
from pathlib import Path

# 仿真开始前保存体网格
output_dir = Path(steady_config.output_dir)
volume_mesh_path = output_dir / "volume_mesh.pkl"

try:
    with open(volume_mesh_path, 'wb') as f:
        pickle.dump(grid_data, f)  # 序列化GridData对象
    logger.success(f"Volume mesh saved to: {volume_mesh_path}")
except Exception as e:
    logger.warning(f"Failed to save volume mesh: {e}")
```

### **加载过程**

在 [`post_commands.py`](d:\myWorkspace\AutoFlowCFD\src\autoflowcfd\cli\post_commands.py) 中：

```python
# 检测文件格式
if grid_file.suffix.lower() == '.pkl':
    # 加载PKL格式的体网格
    with open(grid_file, 'rb') as f:
        grid_data = pickle.load(f)  # 反序列化
    logger.success(f"✓ Volume mesh loaded: {grid_data.node_count} nodes")
else:
    # 从NAS文件重新生成（不推荐）
    parser = NASParser(str(grid_file))
    grid_data = parser.parse(generate_volume_mesh=True)
    logger.warning("⚠ Re-generating volume mesh from surface mesh")
```

---

## 📊 文件格式对比

| 特性 | PKL格式 | NAS格式 |
|------|---------|---------|
| **内容** | 体网格对象 | 面网格文本 |
| **格式** | 二进制序列化 | 文本格式 |
| **文件大小** | 50-200MB | 10-50MB |
| **加载速度** | ⚡ 毫秒级 | 🐌 秒级 |
| **一致性** | ✅ 100% | ⚠️ 可能不同 |
| **可读性** | ❌ 不可读 | ✅ 可编辑 |
| **跨平台** | ❌ Python专用 | ✅ 通用 |
| **适用场景** | VTK导出/Resume | 初始仿真 |

---

## 🎯 使用指南

### **方法1: 自动检测（推荐）**

```bash
autoflowcfd post export-vtk \
  --case results/steady_simulation/ \
  --output flow_field.vtk
```

命令会自动查找 `volume_mesh.pkl`（最高优先级）。

### **方法2: 显式指定PKL文件**

```bash
autoflowcfd post export-vtk \
  --case results/steady_simulation/ \
  --grid results/volume_mesh.pkl \
  --checkpoint results/checkpoints/latest.h5 \
  --output flow_field.vtk
```

### **方法3: 使用NAS文件（不推荐）**

```bash
autoflowcfd post export-vtk \
  --case results/steady_simulation/ \
  --grid sedan.nas \
  --checkpoint results/checkpoints/latest.h5 \
  --output flow_field.vtk
```

⚠️ **警告：** 会重新生成体网格，可能导致不一致！

---

## ⚠️ 注意事项

### **1. 版本兼容性**

```python
# ✅ 相同版本之间兼容
AutoFlowCFD v0.1.0 → v0.1.0  # OK

# ⚠️ 大版本可能不兼容
AutoFlowCFD v0.1.0 → v0.2.0  # 可能失败
```

如果遇到加载错误：
```
_pickle.UnpicklingError: invalid load key
```

**解决：** 使用原始NAS文件重新生成体网格。

---

### **2. 不要手动编辑PKL文件**

```bash
# ❌ 禁止
vim volume_mesh.pkl
nano volume_mesh.pkl

# ✅ 正确做法：修改原始NAS文件并重新仿真
```

---

### **3. 备份重要PKL文件**

```bash
# 创建备份
cp results/case_name/volume_mesh.pkl \
   backups/volume_mesh_backup.pkl

# 压缩存储（可选）
gzip volume_mesh.pkl  # 生成 .pkl.gz
```

---

### **4. 验证PKL文件完整性**

```python
import pickle

try:
    with open('volume_mesh.pkl', 'rb') as f:
        grid_data = pickle.load(f)
    
    print(f"✓ PKL文件有效")
    print(f"  节点数: {grid_data.node_count}")
    print(f"  单元数: {grid_data.cell_count}")
except Exception as e:
    print(f"✗ PKL文件损坏: {e}")
```

---

## 🔧 常见问题

### **Q: 为什么没有生成volume_mesh.pkl？**

**可能原因：**
1. 仿真运行失败（在保存前出错）
2. 磁盘空间不足
3. 权限问题
4. 旧版本代码未实现

**检查日志：**
```bash
grep -i "volume mesh" results/solver.log
```

---

### **Q: PKL文件可以跨平台使用吗？**

**答案：** 
- 同操作系统：✅ 通常可以
- 不同操作系统：⚠️ 可能不兼容
- 建议：在相同环境下使用

---

### **Q: 如何将PKL转换为NAS？**

**当前状态：** 暂不支持

**未来计划：** 在v0.2版本中添加PKL→NAS转换功能。

**临时方案：**
```python
import pickle

with open('volume_mesh.pkl', 'rb') as f:
    grid_data = pickle.load(f)

# 访问网格数据
nodes_x = grid_data.nodes.x
cells_conn = grid_data.cells.connectivity

# TODO: 手动实现导出逻辑
```

---

### **Q: PKL文件太大怎么办？**

**解决方案：**
1. 压缩存储：`gzip volume_mesh.pkl`
2. 清理不需要的case目录
3. 只保留重要的PKL文件

---

## 📚 相关文档

- **VTK CLI使用指南**: [`docs/VTK_CLI_GUIDE.md`](d:\myWorkspace\AutoFlowCFD\docs\VTK_CLI_GUIDE.md)
- **VTK数据基础**: [`docs/VTK_DATA_BASIS.md`](d:\myWorkspace\AutoFlowCFD\docs\VTK_DATA_BASIS.md)
- **完整导出指南**: [`docs/VTK_EXPORT_GUIDE.md`](d:\myWorkspace\AutoFlowCFD\docs\VTK_EXPORT_GUIDE.md)

---

## 🎯 总结

**核心要点：**

1. ✅ **优先使用 `volume_mesh.pkl`** 进行VTK导出
2. ✅ PKL保证与仿真使用的体网格100%一致
3. ✅ 加载速度快，无需重新生成
4. ⚠️ 注意版本兼容性
5. ⚠️ 不要手动编辑PKL文件

**最佳实践：**
```bash
# 始终让CLI自动检测（会优先使用PKL）
autoflowcfd post export-vtk \
  --case results/case_name/ \
  --output result.vtk
```

---

**最后更新**: 2026-07-31  
**维护者**: AutoFlowCFD Team
