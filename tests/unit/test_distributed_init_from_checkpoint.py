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
    @pytest.mark.parametrize("model_name", ["sst", "sa"])
    def test_roundtrip_restores_euler_state_and_turbulence(self, mesh_and_ops, tmp_path, model_name):
        """分布式 checkpoint 写出 -> `--init-from` 读回：平均流与湍流输运场、涡粘都精确恢复
        （2026-10-04 起分布式 checkpoint 写湍流场；此前只写 U_sps）。"""
        from autoflowcfd.core.mpi.distributed_checkpoint import (
            distributed_save_checkpoint, restore_distributed_state_from_checkpoint,
        )
        from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive

        mesh, ops = mesh_and_ops
        n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
        rng = np.random.default_rng(11)
        U5 = _nonuniform_euler_U(n_cells, n_sps, rng)

        solver_a = _make_solver(mesh, ops, model_name)
        solver_a.state.U[:n_cells] = U5
        m = solver_a.turb_model
        truth = [f * (1.0 + 0.3 * rng.random(f.shape)) for f in m.transported_fields()]
        m.set_transported_fields([t.copy() for t in truth])
        m.nu_t = rng.uniform(1e-5, 1e-4, size=(n_cells, n_sps))
        saved_path = distributed_save_checkpoint(
            solver_a, str(tmp_path), 42, "dummy_input.nas", mesh.order, model_name, "cpu")

        solver_b = _make_solver(mesh, ops, model_name)
        assert restore_distributed_state_from_checkpoint(saved_path, solver_b) == 42
        np.testing.assert_allclose(solver_b.state.U[:n_cells], U5, rtol=1e-12)
        np.testing.assert_allclose(solver_b.state.Q[:n_cells, :, :5], conserved_to_primitive(U5), rtol=1e-12)
        for got, want in zip(solver_b.turb_model.transported_fields(), truth):
            np.testing.assert_array_equal(got, want)
        np.testing.assert_array_equal(solver_b.turb_model.nu_t, m.nu_t)
        assert solver_b.turb_model.production_factor == 1.0, "湍流已发展：跳过产生项斜坡"

    def test_turbulence_comes_from_model_fields_not_state_slots(self, mesh_and_ops, tmp_path):
        """fail 半边：单机 SST checkpoint 的 U_sps[...,5:7] 是从未更新的历史槽位（初值），真正的湍流场
        在 k_field/omega_field 字段里。此前分布式 --init-from 从槽位换算 k/omega。"""
        import types

        from autoflowcfd.core.mpi.distributed_checkpoint import restore_distributed_state_from_checkpoint
        from autoflowcfd.core.utils.checkpoint import CheckpointManager

        mesh, ops = mesh_and_ops
        n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
        rng = np.random.default_rng(5)
        U5 = _nonuniform_euler_U(n_cells, n_sps, rng)
        k_true = 0.05 + rng.uniform(0, 0.01, size=(n_cells, n_sps))
        omega_true = 100.0 + rng.uniform(0, 10.0, size=(n_cells, n_sps))
        stale = np.full((n_cells, n_sps, 2), 1e-3)          # 槽位里留着的初值
        U7 = np.concatenate([U5, stale], axis=-1)
        config = types.SimpleNamespace(mode="steady", backend="cpu", order=mesh.order, turbulence="sst_kw")
        saved_path = CheckpointManager(config, output_dir=str(tmp_path)).save(
            U7.mean(axis=1), {"iterations": [7]}, 7, metadata={"input_file": "dummy.nas", "order": mesh.order},
            extra_fields={"U_sps": U7, "k_field": k_true, "omega_field": omega_true})

        solver_b = _make_solver(mesh, ops, "sst")
        restore_distributed_state_from_checkpoint(saved_path, solver_b)
        np.testing.assert_array_equal(solver_b.turb_model.k_field, k_true)
        np.testing.assert_array_equal(solver_b.turb_model.omega_field, omega_true)

    def test_other_model_checkpoint_keeps_freestream(self, mesh_and_ops, tmp_path):
        """SST checkpoint -> SA 瞬态：没有 nu_tilde_field，SA 场保留来流初值（壁面解点仍为 0）。"""
        from autoflowcfd.core.mpi.distributed_checkpoint import (
            distributed_save_checkpoint, restore_distributed_state_from_checkpoint,
        )

        mesh, ops = mesh_and_ops
        solver_a = _make_solver(mesh, ops, "sst")
        saved_path = distributed_save_checkpoint(
            solver_a, str(tmp_path), 3, "dummy_input.nas", mesh.order, "sst", "cpu")
        solver_b = _make_solver(mesh, ops, "sa")
        before = solver_b.turb_model.nu_tilde_field.copy()
        restore_distributed_state_from_checkpoint(saved_path, solver_b)
        np.testing.assert_array_equal(solver_b.turb_model.nu_tilde_field, before)

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


@pytest.mark.parametrize("model_name", ["sst", "sa"])
def test_distributed_resume_restores_turbulence(mesh_and_ops, tmp_path, model_name):
    """`solve resume` 的分布式读回（`distributed_load_checkpoint`）恢复湍流输运场与涡粘。2026-10-04
    以前分布式 checkpoint 不写湍流场，resume 后湍流从来流初值重新开始。"""
    from autoflowcfd.core.mpi.distributed_checkpoint import distributed_load_checkpoint, distributed_save_checkpoint

    mesh, ops = mesh_and_ops
    rng = np.random.default_rng(2)
    solver_a = _make_solver(mesh, ops, model_name)
    m = solver_a.turb_model
    truth = [f * (1.0 + 0.3 * rng.random(f.shape)) for f in m.transported_fields()]
    m.set_transported_fields([t.copy() for t in truth])
    m.nu_t = rng.uniform(1e-5, 1e-4, size=m.nu_t.shape)
    path = distributed_save_checkpoint(solver_a, str(tmp_path), 9, "dummy_input.nas", mesh.order, model_name, "cpu")

    solver_b = _make_solver(mesh, ops, model_name)
    _, metadata, iteration = distributed_load_checkpoint(path, solver_b)
    assert iteration == 9
    for got, want in zip(solver_b.turb_model.transported_fields(), truth):
        np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(solver_b.turb_model.nu_t, m.nu_t)
    # 后处理用的单元平均同样写出（VTK 导出读它，`core/turbulence/output.py`）
    from autoflowcfd.core.turbulence.output import turbulence_cell_means, turbulence_fields_from_checkpoint
    cell = turbulence_fields_from_checkpoint(metadata["fields"])
    ref = turbulence_cell_means(m, mesh.n_prism_cells, mesh.order)
    assert set(cell) == set(ref)
    for key in ref:
        np.testing.assert_allclose(cell[key], ref[key], rtol=1e-13)
