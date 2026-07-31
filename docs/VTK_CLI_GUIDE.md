# VTK导出CLI命令使用指南

## 📋 命令概述

```bash
autoflowcfd post export-vtk --case <case_directory> [options]
```

该命令从仿真结果目录中导出VTK文件，用于ParaView可视化。

---

## 🔑 **必需的数据**

### ✅ **1. 体网格文件（Volume Mesh）** ⭐必需

**作用：** 提供网格几何信息（节点坐标、单元连通性）

**支持两种格式：**

#### **格式A: PKL格式（推荐）** 🎯
```bash
volume_mesh.pkl
```

**特点：**
- Python Pickle序列化格式
- 直接保存计算过程中生成的体网格对象
- **无需重新生成体网格**，保证完全一致
- 加载速度快，无解析开销
- 文件位置：`results/case_name/volume_mesh.pkl`

**来源：**
- 求解器运行时自动保存
- 在稳态/瞬态仿真开始时生成并保存

**优势：**
✅ 保证与仿真使用的体网格完全一致  
✅ 避免重新生成导致的差异  
✅ 加载速度快  

---

#### **格式B: NAS格式（备选）**
```bash
sedan_volume.nas
```

**特点：**
- Nastran文本格式
- 需要从面网格重新生成体网格
- 可能因随机种子不同导致微小差异
- 文件位置：原始输入文件或 `results/grid/*.nas`

**来源：**
- 原始面网格文件 [.nas](file://d:\myWorkspace\AutoFlowCFD\examples\ahmed_body_demo.nas)
- 通过 `NASParser.parse(generate_volume_mesh=True)` 重新生成体网格

**注意：**
⚠️ 重新生成的体网格可能与原始仿真不完全一致  
⚠️ 建议优先使用 `volume_mesh.pkl`  

---

### ✅ **2. Checkpoint文件（Solution Data）** ⭐必需

**作用：** 提供流场解数据（速度、压力、湍流量等）

**格式：** `.h5` 文件（HDF5格式）

**来源：**
- 求解器运行过程中定期保存的检查点
- 或最终收敛结果

**示例位置：**
```
results/steady_simulation/checkpoints/checkpoint_0500.h5
results/steady_simulation/checkpoints/latest  (符号链接)
```

---

## 💻 **使用方法**

### **方法1: 自动检测（推荐）**

如果case目录结构规范，可以省略 `--grid` 和 `--checkpoint` 参数：

```bash
autoflowcfd post export-vtk \
  --case results/steady_simulation/ \
  --output flow_field.vtk
```

**自动检测优先级：**
1. `case/volume_mesh.pkl` （最高优先级 ✅）
2. `case/grid/*.nas`
3. `case/*.nas`

### **方法2: 显式指定PKL文件（最佳实践）**

```bash
autoflowcfd post export-vtk \
  --case results/steady_simulation/ \
  --grid results/volume_mesh.pkl \
  --checkpoint results/checkpoints/checkpoint_0500.h5 \
  --output flow_field.vtk
```

### **方法3: 使用NAS文件（不推荐用于resume）**

```bash
autoflowcfd post export-vtk \
  --case results/steady_simulation/ \
  --grid sedan_surface.nas \
  --checkpoint results/checkpoints/checkpoint_0500.h5 \
  --output flow_field.vtk
```

**警告：** 使用NAS文件会重新生成体网格，可能导致与原始仿真不完全一致！

### **方法4: 导出指定变量**

```
# 仅导出速度场
autoflowcfd post export-vtk \
  --case results/ \
  --variables velocity \
  --output velocity_only.vtk

# 导出速度和压力
autoflowcfd post export-vtk \
  --case results/ \
  --variables velocity pressure \
  --output vel_pres.vtk

# 导出完整湍流场
autoflowcfd post export-vtk \
  --case results/ \
  --variables velocity pressure k omega \
  --output full_turb.vtk
```

### **方法5: 瞬态仿真指定时间步**

```
autoflowcfd post export-vtk \
  --case results/transient/ \
  --time-step 100 \
  --output step_100.vtk
```

---

## 📂 **推荐的目录结构**

### **稳态仿真**
```
results/
└── sedan_steady_sst_kw_p2_gpu_20260731_143022/
    ├── grid/
    │   └── sedan_volume.nas          # ← 体网格文件
    ├── checkpoints/
    │   ├── checkpoint_0100.h5
    │   ├── checkpoint_0200.h5
    │   ├── checkpoint_0500.h5        # ← 最新检查点
    │   └── latest -> checkpoint_0500.h5
    ├── config.yaml
    └── solver.log
```

**导出命令：**
```bash
autoflowcfd post export-vtk \
  --case results/sedan_steady_sst_kw_p2_gpu_20260731_143022/ \
  --output final_result.vtk
```

### **瞬态仿真**
```
results/
└── sedan_transient_ddes_p3_gpu_20260731_150000/
    ├── grid/
    │   └── sedan_volume.nas
    ├── checkpoints/
    │   ├── checkpoint_0050.h5
    │   ├── checkpoint_0100.h5
    │   ├── ...
    │   ├── checkpoint_1000.h5
    │   └── latest -> checkpoint_1000.h5
    └── config.yaml
```

**批量导出多个时间步：**
```bash
# 导出第100步
autoflowcfd post export-vtk \
  --case results/sedan_transient_ddes_p3_gpu_20260731_150000/ \
  --time-step 100 \
  --output step_0100.vtk

# 导出第500步
autoflowcfd post export-vtk \
  --case results/sedan_transient_ddes_p3_gpu_20260731_150000/ \
  --time-step 500 \
  --output step_0500.vtk
```

---

## ⚠️ **常见错误与解决方案**

### **错误1: Grid file not found**

```
FileNotFoundError: Grid file not found in case directory: results/
Please specify grid file with --grid option.
Expected locations:
  - results/volume_mesh.pkl (saved volume mesh)
  - results/grid/*.nas (surface mesh)
  - results/*.nas (surface mesh)
```

**原因：** 无法自动检测到网格文件

**解决：**
```bash
# 方案1: 使用PKL文件（推荐）
autoflowcfd post export-vtk \
  --case results/ \
  --grid results/volume_mesh.pkl \
  --output result.vtk

# 方案2: 使用NAS文件
autoflowcfd post export-vtk \
  --case results/ \
  --grid /path/to/sedan.nas \
  --output result.vtk
```

### **错误2: No checkpoint files found**

```
FileNotFoundError: No checkpoint files found in: results/checkpoints
Please specify checkpoint with --checkpoint option.
```

**原因：** 检查点目录为空或不存在

**解决：**
```bash
# 显式指定检查点文件
autoflowcfd post export-vtk \
  --case results/ \
  --checkpoint /path/to/checkpoint_0500.h5 \
  --output result.vtk
```

### **错误3: Grid-solution mismatch**

```
ValueError: Grid-solution mismatch!
  Grid has 1000000 cells
  Solution expects 950000 cells
  Please use the SAME grid file that was used in the original simulation.
```

**原因：** 网格文件和检查点不匹配（使用了不同的网格）

**解决：**
- ✅ **优先使用 `volume_mesh.pkl`**，保证完全一致
- ❌ 避免使用NAS文件重新生成体网格
- 检查是否误用了面网格而非体网格

```bash
# 正确做法：使用PKL文件
autoflowcfd post export-vtk \
  --case results/case_name/ \
  --grid results/case_name/volume_mesh.pkl \
  --checkpoint results/case_name/checkpoints/latest \
  --output result.vtk
```

---

## 🔍 **如何找到正确的文件**

### **查找体网格文件**

```bash
# 方法1: 优先查找PKL文件
ls -lh results/*/volume_mesh.pkl

# 方法2: 查看仿真配置
cat results/config.yaml | grep grid_file

# 方法3: 搜索所有网格文件
find results/ \( -name "*.pkl" -o -name "*.nas" \) -type f

# 方法4: 检查grid子目录
ls -lh results/grid/
```

### **查找Checkpoint文件**

```bash
# 方法1: 查看最新检查点
ls -lh results/checkpoints/latest

# 方法2: 列出所有检查点
ls -lh results/checkpoints/checkpoint_*.h5

# 方法3: 按修改时间排序
ls -lt results/checkpoints/*.h5 | head -5
```

---

## 📊 **支持的变量类型**

| 变量名 | 描述 | VTK数据类型 | 适用场景 |
|--------|------|------------|---------|
| `velocity` | 速度矢量场 (u, v, w) | VECTORS | 速度云图、流线图 |
| `pressure` | 静压场 | SCALARS | 压力分布、气动力 |
| `k` | 湍流动能 | SCALARS | 湍流分析 |
| `omega` | 比耗散率 | SCALARS | SST模型分析 |
| `nut` | 湍流粘度 | SCALARS | 湍流粘性分析 |

**默认变量：** 如果未指定 `--variables`，默认导出 `velocity` 和 `pressure`

---

## 🎯 **完整工作流程示例**

### **步骤1: 运行稳态仿真**

```bash
autoflowcfd solve steady \
  --grid sedan.nas \
  --backend gpu \
  --order 3 \
  --turbulence sst_kw \
  --max-iter 1000 \
  --output results/
```

**输出目录：**
```
results/sedan_steady_sst_kw_p3_gpu_20260731_143022/
├── volume_mesh.pkl              # ← 自动保存的体网格
├── checkpoints/
│   ├── checkpoint_0500.h5
│   ├── checkpoint_1000.h5
│   └── latest -> checkpoint_1000.h5
└── config.yaml
```

### **步骤2: 导出VTK文件（使用PKL）**

```bash
autoflowcfd post export-vtk \
  --case results/sedan_steady_sst_kw_p3_gpu_20260731_143022/ \
  --grid results/sedan_steady_sst_kw_p3_gpu_20260731_143022/volume_mesh.pkl \
  --checkpoint results/sedan_steady_sst_kw_p3_gpu_20260731_143022/checkpoints/latest \
  --output final_result.vtk
```

**或使用自动检测（更简洁）：**
```bash
autoflowcfd post export-vtk \
  --case results/sedan_steady_sst_kw_p3_gpu_20260731_143022/ \
  --output final_result.vtk
```

**输出：**
```
======================================================================
✅ VTK Export Successful
======================================================================
Output file:     final_result.vtk
Format:          LEGACY
Variables:       velocity, pressure
Iteration:       1000
Grid cells:      1,234,567
Grid source:     volume_mesh.pkl (saved volume mesh)
======================================================================
```

**输出：**
```
======================================================================
✅ VTK Export Successful
======================================================================
Output file:     final_result.vtk
Format:          LEGACY
Variables:       velocity, pressure
Iteration:       1000
Grid cells:      1,234,567
======================================================================

💡 Next steps:
  1. Open ParaView
  2. File → Open → final_result.vtk
  3. Click Apply to load data
  4. Select coloring variable (Velocity/Pressure)
======================================================================
```

### **步骤3: ParaView可视化**

```bash
# 启动ParaView
paraview final_result.vtk

# 或在GUI中：
# File → Open → 选择final_result.vtk → Apply
```

---

## 📦 **PKL格式详解**

### **什么是volume_mesh.pkl？**

`volume_mesh.pkl` 是AutoFlowCFD在仿真开始时自动保存的**体网格对象序列化文件**。

**生成时机：**
```python
# 在 solve_commands.py 中
output_dir = Path(steady_config.output_dir)
volume_mesh_path = output_dir / "volume_mesh.pkl"

with open(volume_mesh_path, 'wb') as f:
    pickle.dump(grid_data, f)  # 序列化GridData对象
```

**文件内容：**
- 完整的 `GridData` 对象（Python对象）
- 包含节点坐标、单元连通性、边界信息等
- 已经是体网格（非面网格）

---

### **为什么优先使用PKL格式？**

| 对比项 | PKL格式 | NAS格式 |
|--------|---------|---------|
| **一致性** | ✅ 100%一致 | ⚠️ 可能因随机种子不同 |
| **加载速度** | ✅ 毫秒级 | ❌ 需要解析+生成（秒级） |
| **文件大小** | ~50-200MB | ~10-50MB |
| **可读性** | ❌ 二进制 | ✅ 文本格式 |
| **跨语言** | ❌ Python专用 | ✅ 通用格式 |
| **推荐场景** | VTK导出/Resume | 初始仿真/交换 |

---

### **如何找到volume_mesh.pkl？**

```bash
# 方法1: 查看case目录
ls -lh results/case_name/volume_mesh.pkl

# 方法2: 搜索所有PKL文件
find results/ -name "*.pkl" -type f

# 方法3: 检查最新修改时间
ls -lt results/*/volume_mesh.pkl | head -5
```

**典型位置：**
```
results/sedan_steady_sst_kw_p3_gpu_20260731_143022/
├── volume_mesh.pkl          # ← 体网格（PKL格式）
├── checkpoints/
│   └── checkpoint_0500.h5   # ← 解数据
└── config.yaml
```

---

### **PKL文件的安全性**

⚠️ **注意事项：**

1. **不要手动编辑PKL文件**
   - Pickle是二进制序列化格式
   - 直接编辑会导致损坏

2. **版本兼容性**
   ```python
   # 确保使用相同版本的AutoFlowCFD
   # GridData类结构变化可能导致加载失败
   ```

3. **跨平台兼容**
   - PKL文件在不同操作系统间可能不兼容
   - 建议在相同环境下使用

4. **备份重要PKL文件**
   ```bash
   cp results/case_name/volume_mesh.pkl \
      backups/volume_mesh_backup.pkl
   ```

---

### **从PKL转换为NAS（可选）**

如果需要将PKL格式的体网格转换为NAS格式：

```python
import pickle
from autoflowcfd.grid import GridData

# 加载PKL文件
with open('volume_mesh.pkl', 'rb') as f:
    grid_data = pickle.load(f)

# TODO: 实现GridData到NAS的导出
# grid_data.export_to_nas('volume_mesh.nas')
```

**注意：** 当前版本暂不支持PKL→NAS转换，计划在未来的版本中添加。

## 💡 **最佳实践**

### **1. 始终使用volume_mesh.pkl** 🎯

```bash
# ✅ 推荐：使用PKL文件
autoflowcfd post export-vtk \
  --case results/case_name/ \
  --grid results/case_name/volume_mesh.pkl \
  --output result.vtk

# ❌ 避免：使用NAS文件重新生成
autoflowcfd post export-vtk \
  --case results/case_name/ \
  --grid sedan.nas \
  --output result.vtk
```

**原因：**
- PKL保证与仿真使用的体网格100%一致
- 避免因随机种子不同导致的网格差异
- 加载速度更快

---

### **2. 验证PKL文件存在**

在运行仿真后，检查是否生成了PKL文件：

```bash
ls -lh results/*/volume_mesh.pkl

# 预期输出：
# -rw-r--r-- 1 user user 85M Jul 31 14:30 results/sedan_steady_.../volume_mesh.pkl
```

如果不存在，检查日志中是否有警告：
```
Failed to save volume mesh: [error message]
```

---

### **3. 备份重要的PKL文件**

```bash
# 创建备份目录
mkdir -p backups/grid_files

# 备份PKL文件
cp results/case_name/volume_mesh.pkl \
   backups/grid_files/volume_mesh_backup.pkl

# 记录元数据
echo "Case: sedan_steady_sst_kw_p3_gpu" > backups/grid_files/metadata.txt
echo "Date: $(date)" >> backups/grid_files/metadata.txt
echo "Cells: 1234567" >> backups/grid_files/metadata.txt
```

---

### **4. 使用有意义的文件名**

```bash
# ❌ 不好
--output result.vtk

# ✅ 好
--output ahmed_body_Re1e6_Cd0.285.vtk

# ✅ 更好（包含时间步信息）
--output sedan_transient_step0500_velocity.vtk
```

---

### **5. 瞬态仿真定期导出**

```bash
# 每100步导出一次
for step in 100 200 300 400 500; do
  autoflowcfd post export-vtk \
    --case results/transient/ \
    --time-step $step \
    --output "step_${step}.vtk"
done
```

---

### **6. 分别导出不同变量**

```bash
# 速度场单独导出（文件较小）
autoflowcfd post export-vtk \
  --case results/ \
  --variables velocity \
  --output velocity_only.vtk

# 完整场单独导出
autoflowcfd post export-vtk \
  --case results/ \
  --variables velocity pressure k omega \
  --output full_field.vtu  # 使用XML格式压缩
```

---

### **7. 验证导出文件**

```bash
# 检查文件大小
ls -lh *.vtk

# 快速查看文件头（确认格式正确）
head -20 result.vtk

# 预期输出：
# vtk DataFile Version 3.0
# AutoFlowCFD Export
# ASCII
# ...
```

---

### **8. 批量处理多个case**

```bash
for case_dir in results/*/; do
  echo "Processing: $case_dir"
  
  # 检查是否存在PKL文件
  if [ -f "${case_dir}volume_mesh.pkl" ]; then
    echo "  ✓ Found volume_mesh.pkl"
    
    autoflowcfd post export-vtk \
      --case "$case_dir" \
      --output "${case_dir%/}_result.vtk"
  else
    echo "  ⚠ No volume_mesh.pkl found, skipping"
  fi
done
```

---

## 📚 **相关文档**

- **VTK数据基础**: [`docs/VTK_DATA_BASIS.md`](d:\myWorkspace\AutoFlowCFD\docs\VTK_DATA_BASIS.md)
- **完整导出指南**: [`docs/VTK_EXPORT_GUIDE.md`](d:\myWorkspace\AutoFlowCFD\docs\VTK_EXPORT_GUIDE.md)
- **快速参考卡**: [`docs/VTK_QUICK_REFERENCE.md`](d:\myWorkspace\AutoFlowCFD\docs\VTK_QUICK_REFERENCE.md)
- **Python API示例**: [`examples/export_vtk_example.py`](d:\myWorkspace\AutoFlowCFD\examples\export_vtk_example.py)

---

## ❓ **FAQ**

### **Q: volume_mesh.pkl和原始的.nas文件有什么区别？**

A: 
- **`.nas` 文件**：面网格（Surface Mesh），只包含表面三角形单元
- **`volume_mesh.pkl`**：体网格（Volume Mesh），包含内部四面体单元
- 仿真时需要从面网格生成体网格，PKL文件保存了生成的结果

---

### **Q: 为什么我的case目录中没有volume_mesh.pkl？**

A: 可能的原因：
1. **仿真运行失败**：在网格生成后、保存前出错
2. **磁盘空间不足**：无法写入PKL文件
3. **权限问题**：没有写入权限
4. **旧版本代码**：早期版本可能未实现自动保存

**解决方法：**
```bash
# 检查日志
grep -i "volume mesh" results/solver.log

# 如果确实缺失，重新运行仿真
autoflowcfd solve steady --grid sedan.nas --output results/
```

---

### **Q: PKL文件可以在不同版本的AutoFlowCFD之间使用吗？**

A: 
- **小版本兼容**：v0.1.x 之间通常兼容
- **大版本可能不兼容**：v0.1 → v0.2 可能因GridData类结构变化而失败
- **建议**：使用相同版本加载PKL文件

如果遇到加载错误：
```python
# 错误示例
_pickle.UnpicklingError: invalid load key

# 解决：使用原始NAS文件重新生成
autoflowcfd post export-vtk \
  --case results/ \
  --grid sedan.nas \
  --checkpoint checkpoints/latest.h5 \
  --output result.vtk
```

---

### **Q: 我可以手动编辑volume_mesh.pkl吗？**

A: **不可以！**
- PKL是二进制序列化格式
- 直接编辑会导致文件损坏
- 如需修改网格，应修改原始NAS文件并重新生成

---

### **Q: PKL文件太大怎么办？**

A: 
- PKL文件大小取决于网格规模
- 千万级网格约50-200MB
- 如果磁盘空间紧张：
  ```bash
  # 压缩备份
  gzip volume_mesh.pkl  # 生成 volume_mesh.pkl.gz
  
  # 使用时解压
  gunzip volume_mesh.pkl.gz
  ```

---

### **Q: 我可以只导出部分区域吗？**

A: 当前实现导出整个流场。如需局部区域，可以在ParaView中使用Clip或Threshold过滤器裁剪。

---

### **Q: VTK文件太大怎么办？**

A: 
1. 使用 `.vtu` 格式（XML二进制压缩）
2. 只导出需要的变量
3. 降低导出频率（瞬态）

```bash
autoflowcfd post export-vtk \
  --case results/ \
  --output result.vtu \  # XML格式
  --variables velocity    # 仅速度
```

---

### **Q: 如何验证PKL文件的完整性？**

A: 
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

或使用CLI命令测试：
```bash
autoflowcfd post export-vtk \
  --case results/ \
  --grid volume_mesh.pkl \
  --checkpoint checkpoints/latest.h5 \
  --output test.vtk

# 如果成功导出，说明PKL文件完整
```

---

### **Q: 如何将PKL转换为其他格式？**

A: 当前版本暂不支持PKL→NAS转换。计划在未来的版本中添加此功能。

临时方案：
```python
import pickle
from autoflowcfd.grid import GridData

# 加载PKL
with open('volume_mesh.pkl', 'rb') as f:
    grid_data = pickle.load(f)

# 访问网格数据
print(f"Nodes: {grid_data.node_count}")
print(f"Cells: {grid_data.cell_count}")
print(f"Node coordinates shape: {grid_data.nodes.x.shape}")

# TODO: 手动导出为其他格式
```

---

**最后更新**: 2026-07-31  
**维护者**: AutoFlowCFD Team
