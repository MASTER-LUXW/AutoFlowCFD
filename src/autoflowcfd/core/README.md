# core 求解器模块

## 概述

本模块实现 AutoFlowCFD 的核心计算组件：基于有限体积法（FVM）的稳态 RANS
求解器与瞬态 DES/LES 求解器。

> 本文档此前描述的是一套独立的、按类拆分的实现（FRScheme/SSTKOmegaModel/
> WallFunctionModel/ConvergenceMonitor 等），但那套实现从未真正接入求解器
> 主流程，已作为无用代码整体删除。下面描述的是实际在跑的live实现。

## 主要模块

### 1. 稳态求解器（`solver_steady.py`：`FRSolver`）
- SSP-RK3 时间推进
- AUSM+up（低马赫数预条件）与 HLLC 两种无粘通量格式，二选一
- 内嵌 SST k-ω 湍流模型、标准/增强壁面函数
- 自适应 CFL、残差收敛监控

### 2. 瞬态求解器（`transient_solver_loop.py`：`TransientSolver`）
- 支持 DES/DDES/LES 时间推进
- 时均场、RMS 脉动统计
- checkpoint 保存/续算

### 3. 数值核心
- `fvm_viscous_residual.py`（`ViscousRANSResidual`）：无粘通量（AUSM+up/HLLC）
  + 粘性通量 + SST 湍流源项 + 壁面函数，稳态/瞬态共用
- `fvm_gradients.py`：Green-Gauss 梯度重构、Barth-Jespersen 限制器
- `fvm_faces.py`（`FVMFaceExtractor`）：面几何数据的共享持有者，实际面提取
  统一走 `grid.mesh_gen.face_extractor.FaceExtractor`
- `bc_handler.py`（`BoundaryConditionHandler`）：向量化边界条件应用
- `aero_coeffs.py`（`AeroCoefficientCalculator`）：Cd/Cl 等气动系数积分
- `time_integration.py`（`TimeIntegrator`）：Backward Euler / RK2 / AB3 时间格式

### 4. 计算后端（`backend/`）
- `backend/cpu_backend.py`（Numba）、`backend/gpu_backend.py`（CUDA/CuPy）
- **尚未接入求解器主流程**：目前 `FRSolver`/`TransientSolver` 里的
  `self.backend` 建好之后没有被调用——真正的数值计算在
  `fvm_viscous_residual.py` 里用 numpy 直接实现。把完整 RANS-SST 物理移植
  成 Numba/CUDA kernel 并接入主流程是一项独立的、工作量较大的后续任务。

## 快速上手

```python
from autoflowcfd.core import FRSolver
from autoflowcfd.config import SteadyConfig

config = SteadyConfig(order=2, max_iter=3000)
solver = FRSolver(grid_data, config)
result = solver.solve()

print(f"Converged: {result.converged}, iterations: {result.iterations}")
```

瞬态：

```python
from autoflowcfd.core import TransientSolver
from autoflowcfd.config import TransientConfig

config = TransientConfig(dt=1e-4, total_time=0.2)
solver = TransientSolver(grid_data, config)
result = solver.solve()
```

## 测试

```bash
python -m pytest tests/unit/test_fvm_core_v2.py -v
python -m pytest tests/unit/test_backends.py -v
python -m pytest tests/integration/test_end_to_end_steady.py -v
```

## 参考文献

- Menter, F. R. (1994). "Two-Equation Eddy-Viscosity Turbulence Models"
- Liou, M.-S. (2006). "A sequel to AUSM, Part II: AUSM+-up"
- Toro, E. F. (2009). "Riemann Solvers and Numerical Methods for Fluid Dynamics"
