"""CPU"完全分布式加载"（`DistributedFRSolver.from_fully_distributed_
package`）Order Continuation 决定性验证（2026-09-02）。

背景：见 `core/mpi/distributed_order_continuation.py`/
`core/mpi/distributed_mesh_loader.py::redistribute_fully_distributed_
for_new_order` 模块文档——"完全分布式加载"模式下本 rank 从未持有完整
全局网格，阶数切换需要 root（通过 `root_context`，`distributed_mesh_
load_v2` 新增的第二个返回值）重新计算+重新分发紧凑包，这是三条分布式
路径里 Order Continuation 实现难度最高的一条。

单 rank（`n_ranks=1`）场景下 `redistribute_fully_distributed_for_new_
order` 的 Send/Recv 分支天然不会被触发（`range(1, 1)` 为空，root 自己
就是唯一的 rank，直接用 `packages[0]`）——与本项目其余"单 rank 模拟
多 rank 场景"的既有验证方式一致（本机无 mpi4py）。

本文件不通过 `distributed_mesh_load_v2`（需要真实网格文件 I/O）构造
初始 package/root_context，而是直接调用它内部同样使用的
`build_fully_distributed_rank_package`，手动组装一份等价的
`root_context`——这样可以用已有的合成网格 fixture
（`_build_synthetic_mixed_mesh`），不依赖磁盘文件。
"""

import numpy as np
import pytest

from autoflowcfd.fr.operators import generate_fr_operators
from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh


def _build_initial_package_and_root_context(order, turb_model_name="NONE"):
    from autoflowcfd.core.mpi.partition import partition_mesh
    from autoflowcfd.core.mpi.distributed_mesh_loader import build_fully_distributed_rank_package
    from autoflowcfd.core.fr_solver.boundary import build_boundary_ghost_provider
    import types

    mesh = _build_synthetic_mixed_mesh(order)
    ops = generate_fr_operators(order)
    fc = mesh.face_connectivity
    n_ranks = 1
    cell_partition = partition_mesh(fc, n_ranks, n_cells=mesh.n_cells)

    freestream = {"rho_inf": 1.225, "vel_inf": 33.33, "p_inf": 101325.0}
    mu_molecular = 1.8e-5
    mach_ref = 0.1
    enable_viscous = True

    root_solver_stub = types.SimpleNamespace(
        mesh=mesh, freestream=freestream, turb_model_name=turb_model_name,
        wmles_model=None,
    )
    boundary_ghost_provider_global = build_boundary_ghost_provider(root_solver_stub, bc_overrides={})

    wall_node_indices = None
    h_max_global = h_wn_global = None

    package = build_fully_distributed_rank_package(
        mesh, ops, fc, cell_partition, 0, n_ranks,
        boundary_ghost_provider_global, freestream, mu_molecular, mach_ref,
        order, enable_viscous, turb_model_name=turb_model_name,
        wall_node_indices=wall_node_indices, h_max_global=h_max_global, h_wn_global=h_wn_global,
    )
    root_context = {
        'mesh': mesh, 'ops': ops, 'fc': fc, 'cell_partition': cell_partition,
        'boundary_ghost_provider_global': boundary_ghost_provider_global,
        'freestream': freestream, 'mu_molecular': mu_molecular, 'mach_ref': mach_ref,
        'enable_viscous': enable_viscous, 'turb_model_name': turb_model_name,
        'wall_node_indices': wall_node_indices,
        'h_max_global': h_max_global, 'h_wn_global': h_wn_global,
        'turbulence_intensity': 0.01, 'viscosity_ratio': 5.0,
        'bc_overrides': {}, 'n_ranks': n_ranks,
    }
    return package, root_context


class TestFullyDistributedOrderContinuation:
    def test_solve_at_order_2_ramps_through_p0_p1_p2(self):
        from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver

        package, root_context = _build_initial_package_and_root_context(2, turb_model_name="NONE")
        solver = DistributedFRSolver.from_fully_distributed_package(
            package, n_ranks=1, root_context=root_context,
        )

        assert solver.order == 2
        assert solver.current_order == 2  # 构造时直接在目标阶数
        assert solver._is_fully_distributed is True

        # dt 选择理由见 test_distributed_order_continuation.py 同名注释
        # （分布式路径固定步长，P2 阶数 CFL 稳定域更窄）。
        result = solver.solve(n_steps=60, dt=1e-9, output_interval=1000)

        assert solver.current_order == 2
        assert np.isfinite(result.final_residual)
        assert result.iterations > 0
        n_local = solver.partition.n_local_cells
        assert solver.state.U.shape[1] == 27  # P2: (2+1)^3
        assert np.all(np.isfinite(solver.state.U[:n_local]))

    def test_no_root_context_fails_fast_when_order_continuation_needed(self):
        """`root_context=None` 构造（未来某个调用方遗漏传入）时，阶数
        切换必须 fail-fast，而不是静默产生错误结果。"""
        from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver

        package, _root_context = _build_initial_package_and_root_context(2, turb_model_name="NONE")
        solver = DistributedFRSolver.from_fully_distributed_package(
            package, n_ranks=1, root_context=None,
        )
        with pytest.raises(NotImplementedError):
            solver.solve(n_steps=5, dt=1e-9, output_interval=1000)

    def test_sst_order_continuation_keeps_turb_fields_consistent(self):
        from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver

        package, root_context = _build_initial_package_and_root_context(2, turb_model_name="SST")
        solver = DistributedFRSolver.from_fully_distributed_package(
            package, n_ranks=1, root_context=root_context,
        )
        assert solver.turb_model is not None

        result = solver.solve(n_steps=60, dt=1e-9, output_interval=1000)

        assert solver.current_order == 2
        n_local = solver.partition.n_local_cells
        assert solver.turb_model.k_field.shape == (n_local, 27)
        assert solver.turb_model.omega_field.shape == (n_local, 27)
        assert np.isfinite(result.final_residual)
        assert np.all(np.isfinite(solver.turb_model.k_field))
        assert np.all(np.isfinite(solver.turb_model.omega_field))
