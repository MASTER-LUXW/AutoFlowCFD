"""分布式求解器的全局视图：把各 rank 的 local 段汇总到 root，组装成单机后处理函数能直接使用的对象。

用途是分布式 `solve steady/transient/resume` 收尾时的气动系数（`postprocess/fr_coefficients.py::
compute_aerodynamic_coefficients_fr` 需要全局网格、算子、原始变量与动力涡粘）。2026-10-09 以前分布式路径不报告
气动系数（单机路径报告），理由写的是"分布式求解器的 local+halo 布局与单机不兼容"——汇总到 root 之后布局就与单机
相同，两种分布式模式下 root 都持有完整全局网格（传统模式每个 rank 都有；完全分布式加载由 `_root_context` 持有）。
"""

from types import SimpleNamespace
from typing import Optional

import numpy as np

from autoflowcfd.core.mpi import get_rank

from .distributed_checkpoint.state import gather_global_state


class GlobalSolverView:
    """root 上的全局视图：属性与单机求解器同名（`mesh`/`ops`/`state`/`mu_molecular`/`freestream`/
    `boundary_ghost_provider`/`_get_turbulent_viscosity_field()`），只读用途。

    边界幽灵态提供者按全局网格构建（分布式求解器自己的那份已重映射到本 rank 的压缩面编号），
    边界配置与求解器相同（`bc_overrides`、WMLES 时壁面为滑移镜像）；不构造 SEM 入口。
    """

    def __init__(self, mesh, ops, U, mu_t, mu_molecular: float, freestream: dict,
                 bc_overrides: Optional[dict] = None, wmles_active: bool = False):
        from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive

        self.mesh = mesh
        self.ops = ops
        self.state = SimpleNamespace(U=U, Q=conserved_to_primitive(U[..., :5]), n_cells=U.shape[0], n_sps=U.shape[1])
        self.mu_molecular = mu_molecular
        self.freestream = freestream
        self._mu_t = mu_t
        self._reference_area = None
        self.turb_model_name = "NONE"                 # 不构造 SEM 入口（力积分不读入口幽灵态）
        self.wmles_model = True if wmles_active else None
        from autoflowcfd.core.fr_solver.boundary import build_boundary_ghost_provider
        self.boundary_ghost_provider = build_boundary_ghost_provider(self, dict(bc_overrides or {}))

    def _get_turbulent_viscosity_field(self) -> Optional[np.ndarray]:
        """动力涡粘 rho * nu_t（湍流模型 + 亚格子），层流时 None（与单机同名方法同一约定）。"""
        return self._mu_t


def gather_global_view(solver) -> Optional[GlobalSolverView]:
    """集体调用（全部 rank 都必须调用）：root 返回全局视图，其余 rank 返回 None。

    守恒变量取 local 段（多 GPU 先从设备拷回）；动力涡粘取各后端 `_local_dynamic_eddy_viscosity()`（最近一步残差
    实际使用的那一份，含亚格子涡粘；层流 None，全部 rank 一致）。
    """
    n_local = int(solver.partition.n_local_cells)
    local_cells = solver.partition.local_cells
    n_global = int(solver.partition.n_global_cells)
    U_dev = getattr(solver, "U_gpu", None)
    U_local = np.asarray(U_dev.get() if hasattr(U_dev, "get") else solver.state.U)[:n_local]
    U_global = gather_global_state(np.ascontiguousarray(U_local[..., :5]), local_cells, n_global)
    mu_t_local = solver._local_dynamic_eddy_viscosity()
    mu_t_global = None
    if mu_t_local is not None:
        stacked = gather_global_state(np.ascontiguousarray(mu_t_local[:n_local, :, None]), local_cells, n_global)
        mu_t_global = None if stacked is None else stacked[..., 0]
    if get_rank() != 0:
        return None
    if getattr(solver, "_is_fully_distributed", False):
        mesh, ops = solver._root_context["mesh"], solver._root_context["ops"]
    else:
        mesh, ops = solver.mesh, solver.ops
    return GlobalSolverView(mesh, ops, U_global, mu_t_global, float(solver.mu_molecular), solver.freestream,
                            bc_overrides=getattr(solver, "bc_overrides", None),
                            wmles_active=getattr(solver, "wmles_model", None) is not None)
