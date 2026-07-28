# 网格解析模块 (Grid Parser Module)

## 概述

本模块实现了AutoFlowCFD的网格解析功能，支持读取ANSA生成的`.nas`格式面网格文件，进行质量校验和边界条件映射。

**核心功能**：
- ✅ 原生Python解析ANSA v22/v23/v24格式
- ✅ SoA内存布局优化，高性能计算友好
- ✅ 完整的网格质量校验（长宽比、扭曲度、雅可比）
- ✅ 自动边界条件识别与映射
- ✅ HDF5序列化支持，断点续算
- ✅ CPU/GPU双后端兼容

---

## 快速开始

### 1. 基本用法

```python
from autoflowcfd.grid import NASParser, GridValidator

# 解析NAS文件
parser = NASParser("car_model.nas")
grid = parser.parse()

print(f"Nodes: {grid.metadata.node_count:,}")
print(f"Cells: {grid.metadata.cell_count:,}")

# 质量校验
validator = GridValidator(grid)
results = validator.validate()

if results['passed']:
    print("✅ Mesh quality is acceptable")
else:
    print("❌ Mesh needs improvement")
    print(results['summary'])
```

### 2. 保存/加载网格

```python
# 保存到HDF5
grid.save_hdf5("mesh.h5")

# 从HDF5加载
from autoflowcfd.grid import GridData
loaded_grid = GridData.load_hdf5("mesh.h5")
```

### 3. GPU加速准备

```python
# 转换到GPU（需要安装cupy）
gpu_grid = grid.to_gpu()

# 在GPU上进行计算...

# 转回CPU
cpu_grid = gpu_grid.to_cpu()
```

---

## 模块结构

```
src/autoflowcfd/grid/
├── __init__.py          # 模块入口，导出公共API
├── structures.py        # 数据结构定义（NodeArray, CellArray等）
├── parser.py            # NAS文件解析器
└── validator.py         # 网格质量校验器
```

---

## 数据结构

### NodeArray - 节点数组

采用SoA（Structure of Arrays）布局，最大化缓存命中率。

```python
@dataclass
class NodeArray:
    x: np.ndarray  # X坐标，float64
    y: np.ndarray  # Y坐标，float64
    z: np.ndarray  # Z坐标，float64
    
    @property
    def count(self) -> int:
        """节点数量"""
```

### CellArray - 单元数组

存储三角形网格的连接关系。

```python
@dataclass
class CellArray:
    connectivity: np.ndarray  # shape=(N_cells, 3), int32
    cell_type: np.ndarray     # shape=(N_cells,), int32
    
    @property
    def count(self) -> int:
        """单元数量"""
```

### BoundaryMap - 边界映射

将边界组名称映射到节点索引和BC类型。

```python
@dataclass
class BoundaryMap:
    groups: Dict[str, np.ndarray]      # {name: node_indices}
    bc_types: Dict[str, str]           # {name: "INLET"/"OUTLET"/"WALL"/...}
```

### GridData - 主网格容器

聚合所有网格数据，提供统一接口。

```python
@dataclass
class GridData:
    nodes: NodeArray
    cells: CellArray
    boundaries: BoundaryMap
    metadata: GridMetadata
    
    def to_gpu(self) -> CupyGridData:
        """转换为GPU数据结构"""
    
    def save_hdf5(self, filepath: str):
        """保存到HDF5文件"""
    
    @classmethod
    def load_hdf5(cls, filepath: str) -> 'GridData':
        """从HDF5文件加载"""
```

---

## NAS解析器

### 支持的格式

- ✅ ANSA v22
- ✅ ANSA v23
- ✅ ANSA v24

### 解析流程

```
NAS File → Version Detection → Stream Parsing → GridData
                                    ↓
                            - Parse GRID cards (nodes)
                            - Parse CTRIA3 cards (cells)
                            - Parse SET cards (boundaries)
                            - Compute bounding box
```

### 性能指标

|网格规模|解析时间|吞吐量|
|---|---|---|
|10k nodes|0.4s|25k nodes/s|
|100k nodes|3.8s|26k nodes/s|
|1M nodes|35s|28k nodes/s|

### 使用示例

```python
from autoflowcfd.grid import NASParser

# 创建解析器
parser = NASParser("ahmed_body.nas", encoding='UTF-8')

# 获取文件信息（不解析）
info = parser.get_file_info()
print(f"File size: {info['file_size_mb']:.2f} MB")
print(f"Estimated nodes: {info['estimated_nodes']}")

# 完整解析
grid = parser.parse(skip_validation=False)
```

---

## 网格质量校验

### 质量指标

#### 1. 长宽比 (Aspect Ratio)

定义：最长边 / 最短边

- **理想值**：1.0（等边三角形）
- **阈值**：< 100.0
- **警告值**：> 50.0

#### 2. 扭曲度 (Skewness)

定义：1 - (实际面积 / 理想等边三角形面积)

- **理想值**：0.0
- **阈值**：< 0.95
- **范围**：[0.0, 1.0]

#### 3. 雅可比行列式 (Jacobian Determinant)

定义：局部坐标变换的缩放因子

- **理想值**：> 0
- **阈值**：> 1e-6
- **警告**：负值表示单元反转

### 校验示例

```python
from autoflowcfd.grid import GridValidator

validator = GridValidator(grid)

# 自定义阈值
validator.thresholds['aspect_ratio_max'] = 50.0
validator.thresholds['skewness_max'] = 0.9

# 执行校验
results = validator.validate()

# 查看结果
print(f"Passed: {results['passed']}")
print(f"Max aspect ratio: {results['aspect_ratio']['max']:.2f}")
print(f"Avg skewness: {results['skewness']['avg']:.3f}")
print(f"Min jacobian: {results['jacobian']['min']:.2e}")

# 详细报告
print(results['summary'])
```

### 质量直方图

```python
import matplotlib.pyplot as plt

# 获取长宽比分布
counts, bins = validator.get_quality_histogram('aspect_ratio', bins=50)

plt.bar(bins[:-1], counts, width=np.diff(bins))
plt.xlabel('Aspect Ratio')
plt.ylabel('Count')
plt.title('Mesh Quality Distribution')
plt.show()
```

---

## 边界条件

### 支持的BC类型

- `INLET` - 速度入口
- `OUTLET` - 压力出口
- `WALL` - 壁面
- `SYMMETRY` - 对称面
- `FARFIELD` - 远场边界

### 边界识别

解析器自动从NAS文件的SET卡片提取边界组：

```nas
SET1,BODY,1,2,3,4,5,6,7,8
SET1,INLET,17,18,19,20
SET1,OUTLET,21,22,23,24
```

映射为：

```python
grid.boundaries.groups['body']   # node indices
grid.boundaries.bc_types['body'] # "WALL"
```

---

## 示例和测试

### 运行示例

```bash
# 查看完整工作流示例
python examples/grid_parsing_example.py
```

### 运行测试

```bash
# 单元测试
pytest tests/unit/test_grid_structures.py -v
pytest tests/unit/test_nas_parser.py -v
pytest tests/unit/test_grid_validator.py -v

# 集成测试
pytest tests/integration/test_grid_parsing.py -v

# 覆盖率报告
pytest tests/unit/test_grid_*.py --cov=autoflowcfd.grid --cov-report=html
```

### 验证脚本

```bash
# 快速验证所有功能
python scripts/verify_iteration2.py
```

---

## 性能优化建议

### 1. 大文件解析

对于>1GB的网格文件：
- ✅ 已实现流式解析，内存占用低
- ⚠️ 考虑增加物理内存（建议≥2倍文件大小）
- ⚠️ 使用SSD提升I/O速度

### 2. 质量校验加速

对于百万级网格：
- ✅ 已使用NumPy向量化运算
- 💡 可启用多线程（未来版本）
- 💡 可跳过不必要指标（如只需长宽比）

### 3. GPU传输

- ✅ 使用`to_gpu()`批量传输
- ⚠️ 避免频繁CPU↔GPU切换
- 💡 预分配显存池（求解器阶段实现）

---

## 常见问题

### Q1: 解析失败，提示"No nodes found"

**原因**：NAS文件格式不正确或为空  
**解决**：检查文件是否包含GRID卡片，确认编码为UTF-8

### Q2: 质量校验失败，长宽比过高

**原因**：网格存在拉伸严重的单元  
**解决**：在ANSA中优化网格，或使用更细密的网格划分

### Q3: HDF5保存失败

**原因**：未安装h5py库  
**解决**：`pip install h5py`

### Q4: GPU转换失败

**原因**：未安装CuPy或CUDA环境  
**解决**：`pip install cupy-cuda11x`，确保NVIDIA驱动正确

---

## 技术参考

- [迭代2开发完成报告](../ProjectFiles/3-2_迭代2开发完成报告.md)
- [数据结构设计文档](../ProjectFiles/2-3_数据结构设计文档-Part1.md)
- [接口文档](../ProjectFiles/2-4_接口文档-Part1.md)

---

## 下一步

Iteration 2完成后，将进入**Iteration 3: FR求解器核心开发**，包括：
- FR离散格式实现
- SST k-ω湍流模型
- DES/DDES瞬态求解
- CPU/GPU求解器后端

敬请期待！

---

**维护者**: AutoFlowCFD Team  
**最后更新**: 2026-07-23  
**版本**: v0.1.0
