# AutoFlowCFD 开发者指南

本文档为 AutoFlowCFD 的二次开发和贡献者提供详细指导，包括项目架构、代码规范、测试流程和扩展开发。

---

## 📋 目录

- [项目概述](#项目概述)
- [开发环境搭建](#开发环境搭建)
- [代码架构](#代码架构)
- [代码规范](#代码规范)
- [测试指南](#测试指南)
- [扩展开发](#扩展开发)
  - [新增湍流模型](#新增湍流模型)
  - [新增边界条件](#新增边界条件)
  - [新增后处理功能](#新增后处理功能)
- [性能优化](#性能优化)
- [调试技巧](#调试技巧)
- [提交流程](#提交流程)

---

## 项目概述

### 技术栈

| 层级 | 技术选型 | 说明 |
|------|---------|------|
| **语言** | Python 3.10+ | 顶层业务逻辑 |
| **数值计算** | NumPy/CuPy | CPU/GPU 数组计算 |
| **并行加速** | Numba/CUDA | CPU 多线程/GPU kernel |
| **CLI 框架** | Click | 命令行接口 |
| **配置管理** | PyYAML | YAML 配置文件 |
| **数据序列化** | HDF5/h5py | 检查点存储 |
| **可视化** | VTK/pyvista | 场数据导出 |
| **日志** | loguru | 结构化日志 |
| **测试** | pytest | 单元测试框架 |

### 核心模块

```
src/autoflowcfd/
├── cli/                  # CLI 命令行接口
├── api.py                # Python API 统一入口
├── core/                 # 求解器引擎
│   ├── backend/          # CPU/GPU 后端
│   ├── solver_*.py       # 稳态/瞬态求解器
│   ├── aero_coeffs.py    # 气动系数计算
│   └── ...
├── grid/                 # 网格解析与处理
├── boundary/             # 边界条件管理
├── config/               # 配置管理
├── postprocess/          # 后处理工具
└── utils/                # 工具函数
```

---

## 开发环境搭建

### 1. 克隆仓库

```bash
git clone https://github.com/AutoFlowCFD/AutoFlowCFD.git
cd AutoFlowCFD
```

### 2. 安装 Poetry

```bash
pip install poetry
```

### 3. 安装依赖

```bash
# 安装核心依赖和开发工具
poetry install

# （可选）启用 GPU 支持
poetry install -E gpu
```

### 4. 激活虚拟环境

```bash
# 方式一：使用 poetry run（推荐）
poetry run python --version

# 方式二：激活虚拟环境
poetry shell
```

### 5. 安装 pre-commit 钩子

```bash
pre-commit install
```

这将在每次提交前自动运行代码格式化和检查。

### 6. 验证安装

```bash
# 运行测试
poetry run pytest tests/ -v

# 检查代码质量
poetry run black --check src/
poetry run isort --check src/
poetry run mypy src/
```

---

## 代码架构

### 模块职责划分

#### 1. Grid 模块（网格处理）

**位置**: `src/autoflowcfd/grid/`

**职责**:
- 解析 NAS 文件
- 构建内存数据结构（SoA 布局）
- 网格质量校验
- 边界条件映射

**核心类**:
```python
class NASParser:
    """NAS 文件解析器"""
    def parse(self, filepath: str) -> GridData:
        ...

class GridData:
    """网格数据结构"""
    node_count: int
    cell_count: int
    nodes: NodeArray
    cells: CellArray
    boundary_map: BoundaryMap

class GridValidator:
    """网格质量校验器"""
    def validate(self, grid: GridData) -> ValidationResult:
        ...
```

**扩展示例**: 新增网格格式支持

```python
# src/autoflowcfd/grid/cgm_parser.py
from .parser import BaseGridParser
from .structures import GridData

class CGMParser(BaseGridParser):
    """CGM 格式网格解析器"""
    
    def parse(self, filepath: str) -> GridData:
        # 实现 CGM 文件解析逻辑
        ...
        return grid_data
```

---

#### 2. Core 模块（求解器引擎）

**位置**: `src/autoflowcfd/core/`

**职责**:
- FR 离散格式实现
- 时间积分方案
- 湍流模型
- 气动系数计算

**核心类**:
```python
class FRSolver:
    """FR 求解器基类"""
    def __init__(self, grid: GridData, config: SolverConfig):
        ...
    
    def compute_residual(self) -> np.ndarray:
        """计算残差"""
        ...
    
    def update_solution(self, dt: float):
        """更新解"""
        ...

class SteadySolver(FRSolver):
    """稳态求解器"""
    def solve(self) -> SteadyResult:
        ...

class TransientSolver(FRSolver):
    """瞬态求解器"""
    def solve(self) -> TransientResult:
        ...
```

**扩展示例**: 新增时间积分方案

```python
# src/autoflowcfd/core/time_integration.py
from .solver_base import BaseTimeIntegrator

class RK4Integrator(BaseTimeIntegrator):
    """四阶 Runge-Kutta 时间积分"""
    
    def integrate(self, solver, dt: float):
        k1 = solver.compute_residual()
        k2 = solver.compute_residual(solver.state + 0.5*dt*k1)
        k3 = solver.compute_residual(solver.state + 0.5*dt*k2)
        k4 = solver.compute_residual(solver.state + dt*k3)
        
        return (k1 + 2*k2 + 2*k3 + k4) / 6.0
```

---

#### 3. Boundary 模块（边界条件）

**位置**: `src/autoflowcfd/boundary/`

**职责**:
- 边界条件定义
- 边界通量计算
- 壁面函数实现

**核心类**:
```python
class BoundaryCondition:
    """边界条件基类"""
    type: str
    
    def apply(self, solver, boundary_faces):
        """应用边界条件"""
        ...

class InletBC(BoundaryCondition):
    """速度入口边界"""
    velocity: Tuple[float, float, float]
    pressure: float

class WallBC(BoundaryCondition):
    """壁面边界"""
    wall_function: str
```

**扩展示例**: 新增边界条件类型

见 [新增边界条件](#新增边界条件) 章节。

---

#### 4. Config 模块（配置管理）

**位置**: `src/autoflowcfd/config/`

**职责**:
- 配置数据结构定义
- YAML 解析与验证
- 默认值管理

**核心类**:
```python
@dataclass
class SteadyConfig:
    """稳态求解器配置"""
    backend: BackendType = BackendType.CPU
    order: int = 2
    turbulence: TurbulenceModel = TurbulenceModel.SST_KW
    max_iter: int = 5000
    convergence_tol: float = 1.0e-6
    ...
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SteadyConfig':
        """从字典创建配置"""
        ...
```

---

#### 5. Postprocess 模块（后处理）

**位置**: `src/autoflowcfd/postprocess/`

**职责**:
- 气动系数计算
- VTK 数据导出
- 收敛历史分析
- 统计量计算

**核心类**:
```python
class Postprocessor:
    """后处理器"""
    
    def calculate_coefficients(self, result) -> Dict[str, float]:
        """计算气动系数"""
        ...
    
    def export_vtk(self, result, output_file: str):
        """导出 VTK 文件"""
        ...
```

---

## 代码规范

### 1. 命名规范

```python
# 类名: PascalCase
class SteadySolver:
    ...

# 函数/方法: snake_case
def calculate_residual():
    ...

# 常量: UPPER_SNAKE_CASE
MAX_ITERATIONS = 5000

# 私有变量: 前导下划线
self._internal_state = None

# 类型注解: 必须添加
def solve(self, max_iter: int = 5000) -> SteadyResult:
    ...
```

### 2. 文档字符串

所有公共 API 必须包含 docstring：

```python
def calculate_drag_coefficient(
    self,
    pressure: np.ndarray,
    area: float,
    velocity: float
) -> float:
    """计算风阻系数.
    
    Args:
        pressure: 压力场数组 (N,)
        area: 参考面积 (m²)
        velocity: 参考速度 (m/s)
    
    Returns:
        Cd: 风阻系数
        
    Raises:
        ValueError: 如果面积为零或负数
        
    Example:
        >>> cd = calc.calculate_drag_coefficient(p, 2.5, 30.0)
        >>> print(f"Cd: {cd:.4f}")
    """
    if area <= 0:
        raise ValueError("Area must be positive")
    
    # 计算逻辑
    ...
```

### 3. 类型注解

所有函数必须添加类型注解：

```python
from typing import List, Dict, Optional, Tuple
import numpy as np

def process_grid(
    nodes: np.ndarray,
    cells: List[Tuple[int, ...]],
    tolerance: Optional[float] = None
) -> Dict[str, np.ndarray]:
    """处理网格数据."""
    ...
```

### 4. 代码格式化

使用 Black 和 isort 自动格式化：

```bash
# 格式化代码
poetry run black src/
poetry run isort src/

# 检查格式（CI 中使用）
poetry run black --check src/
poetry run isort --check src/
```

### 5. 错误处理

使用自定义异常：

```python
# src/autoflowcfd/exceptions.py
class AutoFlowCFDError(Exception):
    """AutoFlowCFD 基础异常"""
    pass

class GridParseError(AutoFlowCFDError):
    """网格解析错误"""
    pass

class SolverConvergenceError(AutoFlowCFDError):
    """求解器收敛错误"""
    pass

# 使用
if not result.converged:
    raise SolverConvergenceError(
        f"Simulation did not converge after {result.iterations} iterations. "
        f"Final residual: {result.final_residual:.6e}"
    )
```

---

## 测试指南

### 1. 测试结构

```
tests/
├── unit/                    # 单元测试
│   ├── test_grid_parser.py
│   ├── test_solver.py
│   └── test_boundary.py
├── integration/             # 集成测试
│   ├── test_steady_simulation.py
│   └── test_transient_simulation.py
└── fixtures/                # 测试数据
    ├── ahmed_body.nas
    └── reference_results.csv
```

### 2. 编写单元测试

```python
# tests/unit/test_grid_parser.py
import pytest
from autoflowcfd.grid import NASParser
from autoflowcfd.exceptions import GridParseError

class TestNASParser:
    """NAS 解析器测试"""
    
    @pytest.fixture
    def parser(self):
        """创建解析器实例"""
        return NASParser()
    
    def test_parse_valid_file(self, parser):
        """测试解析有效文件"""
        grid = parser.parse("tests/fixtures/ahmed_body.nas")
        
        assert grid.node_count > 0
        assert grid.cell_count > 0
        assert len(grid.boundary_map) > 0
    
    def test_parse_invalid_file(self, parser):
        """测试解析无效文件"""
        with pytest.raises(GridParseError):
            parser.parse("nonexistent.nas")
    
    def test_node_coordinates(self, parser):
        """测试节点坐标解析"""
        grid = parser.parse("tests/fixtures/ahmed_body.nas")
        
        # 检查节点坐标范围合理
        assert grid.nodes.x.min() < grid.nodes.x.max()
        assert grid.nodes.y.min() < grid.nodes.y.max()
        assert grid.nodes.z.min() < grid.nodes.z.max()
```

### 3. 编写集成测试

```python
# tests/integration/test_steady_simulation.py
import pytest
from autoflowcfd import AutoFlowCFDAPI

class TestSteadySimulation:
    """稳态仿真集成测试"""
    
    @pytest.fixture
    def api(self):
        """创建 API 实例"""
        return AutoFlowCFDAPI()
    
    @pytest.fixture
    def grid(self, api):
        """加载测试网格"""
        return api.load_grid("tests/fixtures/ahmed_body.nas")
    
    def test_steady_rans_cpu(self, api, grid):
        """测试 CPU 稳态 RANS 仿真"""
        result = api.run_steady(
            grid,
            backend="cpu",
            turbulence="sst_kw",
            order=1,  # 低阶快速测试
            max_iter=100
        )
        
        assert result.iterations <= 100
        assert result.final_residual < 1.0  # 宽松容差
    
    @pytest.mark.gpu
    def test_steady_rans_gpu(self, api, grid):
        """测试 GPU 稳态 RANS 仿真"""
        result = api.run_steady(
            grid,
            backend="gpu",
            turbulence="sst_kw",
            order=1,
            max_iter=100
        )
        
        assert result.iterations <= 100
```

### 4. 运行测试

```bash
# 运行所有测试
poetry run pytest tests/ -v

# 运行单元测试
poetry run pytest tests/unit/ -v

# 运行集成测试
poetry run pytest tests/integration/ -v

# 仅运行 GPU 测试
poetry run pytest -m gpu -v

# 生成覆盖率报告
poetry run pytest --cov=autoflowcfd --cov-report=html
```

### 5. 测试覆盖率要求

- **总体覆盖率**: ≥80%
- **核心模块**: ≥90%
- **公共 API**: 100%

查看覆盖率：

```bash
poetry run pytest --cov=autoflowcfd
```

---

## 扩展开发

### 新增湍流模型

AutoFlowCFD 采用插件化架构，新增湍流模型无需修改核心代码。

#### 步骤 1: 创建湍流模型类

```python
# src/autoflowcfd/core/turbulence_models/spalart_allmaras.py
import numpy as np
from numba import njit
from ..turbulence_base import BaseTurbulenceModel

class SpalartAllmarasModel(BaseTurbulenceModel):
    """Spalart-Allmaras 一方程湍流模型"""
    
    name = "spalart_allmaras"
    
    def __init__(self, config):
        super().__init__(config)
        self.sigma = 2/3
        self.kappa = 0.41
        self.Cb1 = 0.1355
        self.Cb2 = 0.622
        self.Cw1 = 3.5
        self.Cw2 = 0.3
        self.Cv1 = 7.1
    
    @njit(parallel=True)
    def compute_source_terms(
        self,
        nu_tilde: np.ndarray,
        velocity_gradient: np.ndarray,
        distance_to_wall: np.ndarray,
        ...
    ) -> tuple:
        """计算 S-A 模型源项.
        
        Args:
            nu_tilde: 修正的涡粘性
            velocity_gradient: 速度梯度
            distance_to_wall: 到壁面的距离
            ...
        
        Returns:
            production: 产生项
            destruction: 破坏项
            diffusion: 扩散项
        """
        n_cells = len(nu_tilde)
        production = np.zeros(n_cells)
        destruction = np.zeros(n_cells)
        diffusion = np.zeros(n_cells)
        
        for i in range(n_cells):
            # S-A 模型方程
            S = self._compute_strain_rate(velocity_gradient[i])
            d = distance_to_wall[i]
            
            # 产生项
            production[i] = self.Cb1 * S * nu_tilde[i]
            
            # 破坏项
            r = nu_tilde[i] / (S * d**2 + 1e-10)
            fw = self._wall_function(r)
            destruction[i] = self.Cw1 * fw * (nu_tilde[i]/d)**2
            
            # 扩散项（简化）
            diffusion[i] = self.sigma * self._laplacian(nu_tilde, i)
        
        return production, destruction, diffusion
    
    def _compute_strain_rate(self, grad_u: np.ndarray) -> float:
        """计算应变率"""
        ...
    
    def _wall_function(self, r: float) -> float:
        """壁面函数"""
        g = r + self.Cw2 * (r**6 - r)
        return ((1 + self.Cw1**6) / (g**6 + self.Cw1**6))**(1/6)
```

#### 步骤 2: 注册湍流模型

```python
# src/autoflowcfd/core/turbulence_models/__init__.py
from .sst_kw import SSTKwModel
from .des import DESModel
from .ddes import DDESModel
from .spalart_allmaras import SpalartAllmarasModel

TURBULENCE_MODELS = {
    "sst_kw": SSTKwModel,
    "des": DESModel,
    "ddes": DDESModel,
    "spalart_allmaras": SpalartAllmarasModel,  # 新增
}

def get_turbulence_model(name: str, config):
    """工厂函数：获取湍流模型实例"""
    if name not in TURBULENCE_MODELS:
        raise ValueError(f"Unknown turbulence model: {name}")
    return TURBULENCE_MODELS[name](config)
```

#### 步骤 3: 更新配置枚举

```python
# src/autoflowcfd/config/enums.py
from enum import Enum

class TurbulenceModel(str, Enum):
    """湍流模型枚举"""
    SST_KW = "sst_kw"
    DES = "des"
    DDES = "ddes"
    SPALART_ALLMARAS = "spalart_allmaras"  # 新增
```

#### 步骤 4: 编写测试

```python
# tests/unit/test_spalart_allmaras.py
import pytest
import numpy as np
from autoflowcfd.core.turbulence_models import SpalartAllmarasModel

class TestSpalartAllmarasModel:
    @pytest.fixture
    def model(self):
        config = {"sigma": 2/3, "kappa": 0.41, ...}
        return SpalartAllmarasModel(config)
    
    def test_compute_source_terms(self, model):
        nu_tilde = np.ones(100) * 1e-5
        grad_u = np.random.randn(100, 3, 3)
        d_wall = np.linspace(1e-5, 0.1, 100)
        
        prod, dest, diff = model.compute_source_terms(
            nu_tilde, grad_u, d_wall
        )
        
        assert prod.shape == (100,)
        assert dest.shape == (100,)
        assert diff.shape == (100,)
        assert np.all(prod >= 0)  # 产生项应为非负
```

#### 步骤 5: 更新文档

在 README 和配置指南中添加新模型说明。

---

### 新增边界条件

#### 步骤 1: 创建边界条件类

```python
# src/autoflowcfd/boundary/custom_bc.py
import numpy as np
from .boundary_base import BaseBoundaryCondition
from numba import njit

class PressureInletBC(BaseBoundaryCondition):
    """压力入口边界条件"""
    
    type = "PRESSURE_INLET"
    
    def __init__(self, total_pressure: float, temperature: float = 288.15):
        """
        Args:
            total_pressure: 总压 (Pa)
            temperature: 温度 (K)
        """
        self.total_pressure = total_pressure
        self.temperature = temperature
    
    @njit
    def compute_boundary_flux(
        self,
        interior_state: np.ndarray,
        face_normal: np.ndarray,
        ...
    ) -> np.ndarray:
        """计算边界通量.
        
        基于特征线理论实现压力入口边界.
        """
        # 提取内部状态
        rho_i, u_i, v_i, w_i, E_i = interior_state
        
        # 等熵关系计算入口状态
        p_total = self.total_pressure
        T_total = self.temperature
        
        # ... 实现边界条件逻辑
        
        return boundary_flux
```

#### 步骤 2: 注册边界条件

```python
# src/autoflowcfd/boundary/__init__.py
from .inlet import VelocityInletBC
from .outlet import PressureOutletBC
from .wall import WallBC
from .custom_bc import PressureInletBC  # 新增

BOUNDARY_TYPES = {
    "INLET": VelocityInletBC,
    "OUTLET": PressureOutletBC,
    "WALL": WallBC,
    "PRESSURE_INLET": PressureInletBC,  # 新增
}
```

---

### 新增后处理功能

#### 示例: 添加气动力矩计算

```python
# src/autoflowcfd/postprocess/moment_calculator.py
import numpy as np
from typing import Dict

class MomentCalculator:
    """气动力矩计算器"""
    
    def __init__(self, reference_point: tuple = (0.0, 0.0, 0.0)):
        """
        Args:
            reference_point: 力矩参考点 (x, y, z)
        """
        self.ref_point = np.array(reference_point)
    
    def calculate_moments(
        self,
        pressure: np.ndarray,
        face_areas: np.ndarray,
        face_centers: np.ndarray,
        face_normals: np.ndarray
    ) -> Dict[str, float]:
        """计算气动力矩.
        
        Args:
            pressure: 表面压力分布
            face_areas: 面元面积
            face_centers: 面元中心坐标
            face_normals: 面元法向量
        
        Returns:
            moments: 包含 Mx, My, Mz 的字典
        """
        # 计算每个面元的力
        forces = -pressure[:, np.newaxis] * face_areas[:, np.newaxis] * face_normals
        
        # 计算力臂
        moment_arms = face_centers - self.ref_point
        
        # 计算力矩: M = r × F
        moments = np.cross(moment_arms, forces, axis=1)
        
        # 求和
        total_moment = moments.sum(axis=0)
        
        return {
            'Mx': total_moment[0],  # 滚转力矩
            'My': total_moment[1],  # 俯仰力矩
            'Mz': total_moment[2]   # 偏航力矩
        }
    
    def calculate_coefficients(
        self,
        moments: Dict[str, float],
        reference_area: float,
        reference_length: float,
        dynamic_pressure: float
    ) -> Dict[str, float]:
        """计算力矩系数.
        
        Args:
            moments: 力矩值
            reference_area: 参考面积
            reference_length: 参考长度
            dynamic_pressure: 动压
        
        Returns:
            coeffs: 包含 Cl, Cm, Cn 的字典
        """
        q_inf = dynamic_pressure
        S = reference_area
        L = reference_length
        
        return {
            'Cl': moments['Mx'] / (q_inf * S * L),  # 滚转力矩系数
            'Cm': moments['My'] / (q_inf * S * L),  # 俯仰力矩系数
            'Cn': moments['Mz'] / (q_inf * S * L)   # 偏航力矩系数
        }
```

#### 集成到 API

```python
# src/autoflowcfd/api.py
from .postprocess.moment_calculator import MomentCalculator

class AutoFlowCFDAPI:
    ...
    
    def calculate_moments(
        self,
        result,
        reference_point=(0.0, 0.0, 0.0)
    ) -> Dict[str, float]:
        """计算气动力矩.
        
        Args:
            result: 仿真结果对象
            reference_point: 力矩参考点
        
        Returns:
            moments: 力矩值和力矩系数
        """
        calculator = MomentCalculator(reference_point)
        
        # 提取表面数据
        surface_data = self._extract_surface_data(result)
        
        # 计算力矩
        moments = calculator.calculate_moments(**surface_data)
        
        # 计算力矩系数
        coeffs = calculator.calculate_coefficients(
            moments,
            reference_area=self.config.reference_area,
            reference_length=self.config.reference_length,
            dynamic_pressure=self.config.dynamic_pressure
        )
        
        return {'moments': moments, 'coefficients': coeffs}
```

---

## 性能优化

### 1. Numba 优化

```python
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def compute_flux_parallel(
    left_state: np.ndarray,
    right_state: np.ndarray,
    normal: np.ndarray
) -> np.ndarray:
    """并行计算通量.
    
    关键优化:
    - parallel=True: 启用并行
    - fastmath=True: 启用快速数学运算
    - 避免 Python 对象创建
    """
    n_faces = len(left_state)
    flux = np.zeros((n_faces, 5))
    
    for i in prange(n_faces):  # 使用 prange 而非 range
        # 计算逻辑
        ...
        flux[i] = computed_flux
    
    return flux
```

### 2. 内存优化

```python
# 使用 SoA (Structure of Arrays) 而非 AoS (Array of Structures)

# ❌ 慢: AoS
class Node:
    x: float
    y: float
    z: float

nodes: List[Node]

# ✅ 快: SoA
class NodeArray:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
```

### 3. GPU 优化

```python
import cupy as cp

# 将数据转移到 GPU
d_nodes_x = cp.asarray(nodes_x)
d_nodes_y = cp.asarray(nodes_y)

# GPU 计算
d_result = custom_kernel(d_nodes_x, d_nodes_y)

# 传回 CPU（仅在必要时）
result = cp.asnumpy(d_result)
```

### 4. 性能分析

```bash
# 使用 cProfile 分析 Python 代码
poetry run python -m cProfile -o profile.stats script.py

# 使用 SnakeViz 可视化
poetry run pip install snakeviz
poetry run snakeviz profile.stats
```

```python
# 使用 Numba 性能提示
from numba import njit

@njit
def slow_function():
    ...

# 检查编译信息
print(slow_function.signatures)
print(slow_function.inspect_types())
```

---

## 调试技巧

### 1. 日志调试

```python
from loguru import logger

logger.debug("Variable value: {}", variable)
logger.info("Processing grid with {} cells", grid.cell_count)
logger.warning("CFL number is high: {}", cfl)
logger.error("Convergence failed: {}", residual)
```

### 2. 断点调试

```python
# 插入断点
import pdb; pdb.set_trace()

# Python 3.7+
breakpoint()
```

### 3. 检查点调试

```python
# 保存中间状态
from autoflowcfd.core.checkpoint import save_checkpoint

save_checkpoint(
    solver.state,
    iteration=500,
    filepath="./debug_checkpoint.h5"
)

# 加载并检查
from autoflowcfd.core.checkpoint import load_checkpoint

state = load_checkpoint("./debug_checkpoint.h5")
print(state.keys())
```

### 4. 可视化调试

```python
import pyvista as pv

# 导出当前状态用于可视化
grid = pv.UnstructuredGrid(...)
grid.point_data['pressure'] = pressure
grid.plot()
```

---

## 提交流程

### 1. 创建分支

```bash
# 从 main 分支创建功能分支
git checkout main
git pull
git checkout -b feature/add-spallart-allmaras-model
```

### 2. 提交规范

遵循 Conventional Commits 规范：

```bash
# 功能新增
git commit -m "feat: add Spalart-Allmaras turbulence model"

# Bug 修复
git commit -m "fix: correct boundary flux calculation for moving walls"

# 文档更新
git commit -m "docs: update API reference for moment calculation"

# 代码重构
git commit -m "refactor: simplify turbulence model interface"

# 测试添加
git commit -m "test: add integration tests for transient solver"

# 性能优化
git commit -m "perf: optimize flux computation with Numba parallelization"
```

### 3. 推送并创建 PR

```bash
# 推送到远程
git push origin feature/add-spallart-allmaras-model

# 然后在 GitHub 上创建 Pull Request
```

### 4. PR 检查清单

- [ ] 代码通过所有测试（`poetry run pytest`）
- [ ] 代码格式化（`poetry run black src/ && poetry run isort src/`）
- [ ] 类型检查通过（`poetry run mypy src/`）
- [ ] 测试覆盖率 ≥80%
- [ ] 文档已更新（API 文档、配置指南等）
- [ ] CHANGELOG.md 已更新
- [ ] PR 描述清晰，说明改动内容和动机

### 5. Code Review

等待维护者审核，根据反馈进行修改。审核通过后，PR 将被合并到 main 分支。

---

## 常见问题

### Q1: 如何选择合适的 FR 阶数？

**A**: 
- 1 阶：快速预览，稳定性最好
- 2 阶：工程推荐，精度与速度平衡
- 3 阶：高精度研究，计算成本高

### Q2: Numba 编译很慢怎么办？

**A**: 首次编译会较慢，后续调用会使用缓存。可以预编译：

```python
# 在模块加载时预编译
@njit
def my_function():
    ...

# 触发编译
my_function.compile()
```

### Q3: GPU 显存不足怎么办？

**A**: 
- 减少输出变量数量
- 降低检查点频率
- 使用混合精度计算（未来版本）

### Q4: 如何贡献文档？

**A**: 文档同样重要！欢迎改进现有文档或新增教程。

---

## 参考资源

- [NumPy 文档](https://numpy.org/doc/)
- [Numba 文档](https://numba.readthedocs.io/)
- [CuPy 文档](https://docs.cupy.dev/)
- [Click 文档](https://click.palletsprojects.com/)
- [pytest 文档](https://docs.pytest.org/)

---

**最后更新**: 2026-08-17  
**版本**: AutoFlowCFD v0.2.0 (V2.0 系统改造版)
