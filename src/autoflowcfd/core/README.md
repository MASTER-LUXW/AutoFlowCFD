# core 求解器模块

通量重构（FR）高阶求解器：原生棱柱/四面体基，稳态（显式 SSP-RK、伪瞬态 Newton-Krylov）与瞬态（双时间步、
IMEX），RANS（SA-neg、SST 族）与 DES/LES/WMLES。四个后端：CPU 单机、单 GPU、CPU MPI 分布式、多 GPU 分布式。

> 本文档此前描述的是一套基于有限体积法的旧实现（`fvm_*.py`、`backend/cpu_backend.py`、
> `TransientSolver(grid_data, config)` 等），这些文件早已删除；2026-10-04 按当前代码重写。

## 子包

| 子包 | 内容 |
|---|---|
| `fr_operators/` | FR 算子、AUSM+up 等通量核、人工粘性与问题单元判据 |
| `fr_residual/` | CPU 无粘/粘性残差与解析单元块 Jacobian |
| `fr_solver/` | CPU 单机求解器 `FRSolver`：装配、单步推进、边界幽灵态、湍流接入；`solver/solve_loop.py` 的 `SolveLoopMixin` 是 CPU 与单 GPU 共用的求解循环 |
| `gpu/` | 单 GPU 求解器 `GPUFRSolver`（`gpu/solver/`，主机视图见 `host_view.py`）与多 GPU 分布式（`gpu/distributed/`） |
| `mpi/` | CPU MPI 分布式求解器、分区、halo、完全分布式网格加载、分布式 checkpoint 与 Order Continuation |
| `time_integration/` | 显式/双时间步/IMEX 积分器、自适应 CFL、隐式 Newton-Krylov（块预处理、多层校正）、正性限制器 |
| `turbulence/` | 湍流模型（`sa/`、`sst/`、`des/`、`sgs.py`、`wmles.py`）与统一的输运模型接口 `transported.py` |
| `utils/` | Order Continuation、壁面距离、checkpoint、来流方向等共用工具 |
| `backend/` | 后端可用性检测（`get_available_backends`，与 `--backend gpu` 同一判据）与 `SolutionVector` |

## 构造求解器

单机求解器（CPU / 单 GPU）由 `autoflowcfd.cli.solve.solver_factory.build_single_node_solver` 按后端构造，
两个求解器构造参数同名；CLI 的 `solve steady/transient/resume`、checkpoint 重建与 Python API 都经过它：

```python
from autoflowcfd.cli.solve.mesh_loader import load_mesh_for_solver
from autoflowcfd.cli.solve.solver_factory import build_single_node_solver

mesh, volume_data = load_mesh_for_solver("model_volume.pkl", order=2)
solver = build_single_node_solver("cpu", mesh, volume_data, order=2, turb_model_name="sa")  # 或 "gpu"
result = solver.solve(max_iter=1000, dt=1e-3, tol=1e-6)
print(result.converged, result.iterations, result.final_residual)
```

分布式求解器由 `solve steady/transient --n-ranks N`（多 GPU 加 `--backend gpu --multi-gpu`）构造，见
`cli/solve/steady/` 与 `cli/solve/transient_distributed.py`。

## 测试

```bash
python -m pytest tests/unit -q
```

## 参考文献

- Huynh, H. T. (2007). "A flux reconstruction approach to high-order schemes including discontinuous Galerkin methods"
- Allmaras, S. R., Johnson, F. T., Spalart, P. R. (2012). "Modifications and clarifications for the implementation of the Spalart-Allmaras turbulence model"（SA-neg）
- Menter, F. R. (1994). "Two-Equation Eddy-Viscosity Turbulence Models"
- Liou, M.-S. (2006). "A sequel to AUSM, Part II: AUSM+-up"
