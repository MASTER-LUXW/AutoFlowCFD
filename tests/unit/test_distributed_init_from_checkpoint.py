"""`solve transient --init-from`（稳态 checkpoint 恢复为瞬态初场）的
分布式版本决定性验证（2026-09-02）。

背景：此前 CLI 显式拒绝 `--init-from` + `--n-ranks`/`--multi-gpu`/
`--fully-distributed` 的组合（"需要把全局解按分区切给各 rank，这条
路径尚未实现"）——排查后发现 `gather_global_state`/`scatter_local_
state` 这套基础设施本身早就存在（`distributed_save_results`/
`distributed_load_checkpoint` 已经在用），缺的只是"读取稳态 checkpoint
→ 按 n_vars/湍流模型调整 → 广播 → 各 rank 切片"这一段胶水代码
（`restore_distributed_state_from_checkpoint`，见 core/mpi/
distributed_checkpoint.py 模块文档），不是设计上的限制。

关键架构差异（本文件测试专门覆盖）：单机 `FRState.U` 在湍流模型激活
时是 7 变量（k/omega 打包进 U[...,5:7]），但 `DistributedFRState.
n_vars` 恒为 5——分布式路径的 k/omega 独立存储在 `solver.turb_model.
k_field`/`.omega_field`，不在 `state.U` 里。`restore_distributed_
state_from_checkpoint` 必须按这个真实数据模型处理，而不是照搬单机的
"n_vars 5<->7 打包"逻辑。

单 rank（本机无真实 mpi4py）场景下 `n_ranks>1` 分支不会被触发，与本
项目其余"单 rank 模拟"的既有验证方式一致——决定性判据是 n_vars/湍流
换算逻辑本身是否正确，这与真正的多进程通信是否发生是两个独立的问题
（后者本机无法验证，前者可以）。
"""

import numpy as np
from tests.unit._wall_source import synthetic_wall_source
import pytest

from autoflowcfd.fr.operators import generate_fr_operators
from autoflowcfd.core.time_integration.base import TimeIntegrationScheme
from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh


def _make_solver(mesh, ops, turb_model_name):
    from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver
    return DistributedFRSolver(
        mesh=mesh, ops=ops, face_connectivity=mesh.face_connectivity,
        n_ranks=1, backend="cpu", order=mesh.order, turb_model_name=turb_model_name,
        time_scheme=TimeIntegrationScheme.SSP_RK3,
        mu_molecular=1.8e-5, rho_inf=1.225, vel_inf=33.33, p_inf=101325.0,
        wall_distance_source=synthetic_wall_source(mesh),
    )


@pytest.fixture(scope="module")
def mesh_and_ops():
    order = 1
    mesh = _build_synthetic_mixed_mesh(order)
    ops = generate_fr_operators(order)
    return mesh, ops


def _nonuniform_euler_U(n_cells, n_sps, rng):
    from autoflowcfd.core.fr_residual.inviscid import primitive_to_conserved

    rho_inf, u_inf, v_inf, w_inf, p_inf = 1.225, 30.0, 5.0, -3.0, 101325.0
    Q = np.zeros((n_cells, n_sps, 5))
    Q[..., 0] = rho_inf * (1.0 + rng.uniform(-0.02, 0.02, size=(n_cells, n_sps)))
    Q[..., 1] = u_inf + rng.uniform(-3.0, 3.0, size=(n_cells, n_sps))
    Q[..., 2] = v_inf + rng.uniform(-2.0, 2.0, size=(n_cells, n_sps))
    Q[..., 3] = w_inf + rng.uniform(-2.0, 2.0, size=(n_cells, n_sps))
    Q[..., 4] = p_inf * (1.0 + rng.uniform(-0.01, 0.01, size=(n_cells, n_sps)))
    return np.stack(
        [primitive_to_conserved(Q[c, s]) for c in range(n_cells) for s in range(n_sps)]
    ).reshape(n_cells, n_sps, 5)


class TestRestoreDistributedStateFromCheckpoint:
    def test_sst_to_sst_restores_euler_state_and_turbulence(self, mesh_and_ops, tmp_path):
        """稳态 SST → 瞬态 SST：Euler 部分 + k/omega 都必须精确恢复
        （checkpoint 里 U_sps 的 7 个"变量"实际是单机语义——这里用一个
        手工构造的 7 通道数组模拟"若单机曾写过这个 checkpoint"的
        格式，验证 k=rho_k/rho 的换算）。"""
        from autoflowcfd.core.mpi.distributed_checkpoint import (
            distributed_save_checkpoint, restore_distributed_state_from_checkpoint,
        )
        from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive

        mesh, ops = mesh_and_ops
        n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
        rng = np.random.default_rng(11)

        U5 = _nonuniform_euler_U(n_cells, n_sps, rng)
        rho = U5[..., 0]
        k_true = 0.05 + rng.uniform(0, 0.01, size=(n_cells, n_sps))
        omega_true = 100.0 + rng.uniform(0, 10.0, size=(n_cells, n_sps))

        solver_a = _make_solver(mesh, ops, "sst")
        solver_a.state.U[:n_cells] = U5

        # 手工写一个 7 通道 checkpoint（模拟单机 SST checkpoint 的
        # U_sps 格式：最后两列是 rho*k/rho*omega）——分布式 solver_a 的
        # state.U 本身恒为 5 通道，这里直接构造 fields 绕开
        # distributed_save_checkpoint（它只会存 5 通道），专门测试
        # restore 端对 7 通道输入的正确换算。
        import types
        from autoflowcfd.core.utils.checkpoint import CheckpointManager
        U7 = np.concatenate([U5, (rho * k_true)[..., None], (rho * omega_true)[..., None]], axis=-1)
        config = types.SimpleNamespace(mode="steady", backend="cpu", order=mesh.order, turbulence="sst_kw")
        manager = CheckpointManager(config, output_dir=str(tmp_path))
        saved_path = manager.save(
            U7.mean(axis=1), {"iterations": [42]}, 42,
            metadata={"input_file": "dummy.nas", "order": mesh.order},
            extra_fields={"U_sps": U7},
        )

        solver_b = _make_solver(mesh, ops, "sst")
        ckpt_iter = restore_distributed_state_from_checkpoint(saved_path, solver_b)

        assert ckpt_iter == 42
        assert solver_b.state.U.shape[-1] == 5
        np.testing.assert_allclose(solver_b.state.U[:n_cells], U5, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(
            solver_b.state.Q[:n_cells, :, :5], conserved_to_primitive(U5), rtol=1e-9, atol=1e-9,
        )
        np.testing.assert_allclose(solver_b.turb_model.k_field, k_true, rtol=1e-8)
        np.testing.assert_allclose(solver_b.turb_model.omega_field, omega_true, rtol=1e-8)

    def test_none_to_sst_fills_turbulence_with_freestream_defaults(self, mesh_and_ops, tmp_path):
        """稳态 'none'（5 vars）→ 瞬态 'sst'：Euler 部分精确恢复，
        k/omega 用自由来流默认值初始化（`_set_freestream_turbulence`
        推导值，不是单机固定常数 1e-6/1e-2——分布式复用自己已有的、
        更物理自洽的 Tu/VR 公式，见函数文档）。"""
        from autoflowcfd.core.mpi.distributed_checkpoint import (
            distributed_save_checkpoint, restore_distributed_state_from_checkpoint,
        )
        from autoflowcfd.core.fr_solver.turbulence import _set_freestream_turbulence

        mesh, ops = mesh_and_ops
        n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
        rho_inf, vel_inf, p_inf = 1.225, 33.33, 101325.0
        gamma = 1.4
        e = p_inf / ((gamma - 1.0) * rho_inf) + 0.5 * vel_inf ** 2

        solver_a = _make_solver(mesh, ops, "none")
        solver_a.state.U[:n_cells, :, 0] = rho_inf
        solver_a.state.U[:n_cells, :, 1] = rho_inf * vel_inf
        solver_a.state.U[:n_cells, :, 4] = rho_inf * e

        saved_path = distributed_save_checkpoint(
            solver_a, str(tmp_path), 10, "dummy_input.nas", mesh.order, "none", "cpu",
        )

        solver_b = _make_solver(mesh, ops, "sst")
        k_inf, omega_inf = _set_freestream_turbulence(solver_b)
        restore_distributed_state_from_checkpoint(saved_path, solver_b)

        assert solver_b.state.U.shape[-1] == 5
        np.testing.assert_allclose(solver_b.state.U[:n_cells, :, 0], rho_inf)
        np.testing.assert_allclose(solver_b.state.U[:n_cells, :, 4], rho_inf * e)
        np.testing.assert_allclose(solver_b.turb_model.k_field, k_inf)
        np.testing.assert_allclose(solver_b.turb_model.omega_field, omega_inf)

    def test_shape_mismatch_raises(self, mesh_and_ops, tmp_path):
        from autoflowcfd.core.mpi.distributed_checkpoint import (
            distributed_save_checkpoint, restore_distributed_state_from_checkpoint,
        )

        mesh, ops = mesh_and_ops
        solver_a = _make_solver(mesh, ops, "none")
        saved_path = distributed_save_checkpoint(
            solver_a, str(tmp_path), 5, "dummy_input.nas", mesh.order, "none", "cpu",
        )

        order2 = 2
        mesh2 = _build_synthetic_mixed_mesh(order2)
        ops2 = generate_fr_operators(order2)
        solver_b = _make_solver(mesh2, ops2, "none")

        with pytest.raises(ValueError):
            restore_distributed_state_from_checkpoint(saved_path, solver_b)
