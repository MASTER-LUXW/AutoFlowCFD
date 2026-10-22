"""真实缺陷回归测试（2026-08-22，79.1 万单元真实网格续算时发现）：`order` 与 `target_order` 被混在 checkpoint
元数据的同一个字段里。

`FRSolver.__init__(order=...)` 同时设置 `self.current_order`（状态形状）与 `self.order`（Order Continuation
爬坡的目标阶数，`solver.solve()` 里 `self.order_continuation_enabled and self.order >= 2` 判据读取它，见
`core/fr_solver/solver/`）。爬坡中途保存的 checkpoint（例如 P0 阶段、尚未发生 P0 -> P1 切换）`current_order=0`
而目标是 2——两者确实不同。只存一个值、续算时两种用途都用它，自相矛盾：

- 按 target_order（2）重建：网格 / FRSolver 的初始状态形状 (n_cells, 27, n_vars) 与保存的 (n_cells, 1, n_vars)
  U_sps 不符 -> `rebuild_solver_from_checkpoint` 报 "状态形状不匹配"。
- 按 current_order（0）重建、且 FRSolver 的 `self.order` 也留在 0（实际观测到的缺陷，来自当天较早一次只修了
  形状不符那一半的修复）：形状一致、续算静默进行，但 `self.order >= 2` 为假，`solver.solve()` 整个跳过
  `_solve_with_order_continuation`。生产算例上的实际症状：CLI 按阶数打印的 "P{n} Iter ...: Drop: ...x" 变成了
  通用的 "Iteration N: Residual = ..." 格式，残差停在原地，永远到不了 P1 / P2。

修复：`write_checkpoint` 把 `order`（current_order，用于形状匹配）与 `target_order`（solver.order，用于继续
爬坡）存成两个元数据字段；`rebuild_solver_from_checkpoint` 按 `order` 构造网格与初始状态，之后显式恢复
`solver.order = target_order`。
"""

from types import SimpleNamespace

from tests.unit._host_state import with_host_state
from autoflowcfd.core.time_integration.base import TimeIntegrationScheme
from unittest.mock import patch, MagicMock

import numpy as np

from autoflowcfd.cli.solve.checkpoint_io import write_checkpoint, rebuild_solver_from_checkpoint


def _fake_solver(n_cells=2, n_sps=1, n_vars=7, order=0):
    U = np.ones((n_cells, n_sps, n_vars))
    state = SimpleNamespace(U=U, Q=U.copy(), n_sps=n_sps, n_vars=n_vars)
    state._update_primitives = lambda: None
    return with_host_state(SimpleNamespace(
        state=state,
        order=order,
        current_order=order,
        # 2026-09-15：`write_checkpoint` 写出的单元平均现在只统计**真实
        # 自由度**（native 四面体的零填充槽位冻结在初值、会变馊，见
        # fr/native_padding.py::reduce_per_cell_over_real_sps），因此
        # P>=1 的替身必须给出单元类型划分。这里全当棱柱（n_prism ==
        # n_cells）——本文件测的是 target_order 的读写，单元类型不参与。
        mesh=SimpleNamespace(n_prism_cells=n_cells, jacobians={"det_jacs": np.ones(n_cells * n_sps)}),
        freestream={"rho_inf": 1.225, "vel_inf": 33.33, "p_inf": 101325.0},
        bc_overrides=None,
        wall_treatment="resolved", low_mach_precond=True,
        mu_molecular=1.8e-5, _turbulence_intensity=0.01, _viscosity_ratio=5.0, _sem_num_eddies=None,
        artificial_viscosity_enabled=False, artificial_viscosity_alpha=1.0,
        # write_checkpoint 记录时间格式（2026-09-25）：真实求解器恒有 time_integrator
        time_integrator=SimpleNamespace(scheme=TimeIntegrationScheme.SSP_RK3),
    ))


class TestWriteCheckpointStoresTargetOrderSeparately:
    def test_target_order_stored_distinctly_from_current_order(self, tmp_path):
        # 模拟爬坡中途保存的 checkpoint：current_order=0（P0），
        # 但这次运行真正的目标是 P2。
        write_checkpoint(
            _fake_solver(order=0), str(tmp_path), 100, "volume.nas",
            order=0, turbulence_model="sst", backend="cpu",
            target_order=2, quiet=True,
        )
        import h5py
        ckpt = tmp_path / "checkpoints" / "checkpoint_iter_000100.h5"
        with h5py.File(ckpt, "r") as f:
            assert f["metadata"].attrs["order"] == 0
            assert f["metadata"].attrs["target_order"] == 2

    def test_target_order_defaults_to_order_when_not_given(self, tmp_path):
        """旧的调用点 / 不爬坡的运行（order==target_order）不显式传 target_order
        也应继续可用。
        """
        write_checkpoint(
            _fake_solver(order=2), str(tmp_path), 100, "volume.nas",
            order=2, turbulence_model="sst", backend="cpu", quiet=True,
        )
        import h5py
        ckpt = tmp_path / "checkpoints" / "checkpoint_iter_000100.h5"
        with h5py.File(ckpt, "r") as f:
            assert f["metadata"].attrs["target_order"] == 2


class TestRebuildRestoresSolverOrderToTarget:
    def test_mid_ramp_checkpoint_reconstructs_shape_at_current_but_target_at_solver_order(self, tmp_path):
        write_checkpoint(
            _fake_solver(n_cells=2, n_sps=1, order=0), str(tmp_path), 100, "volume.nas",
            order=0, turbulence_model="sst", backend="cpu", target_order=2, quiet=True,
        )
        ckpt = tmp_path / "checkpoints" / "checkpoint_iter_000100.h5"

        # FRSolver 替身：模仿真实 __init__ 的行为，由构造时的单个 `order` 参数
        # 同时设置 current_order 与 order（n_sps=1，与 checkpoint 的 P0 形状一致）。
        def _fake_frsolver(**kwargs):
            s = _fake_solver(n_cells=2, n_sps=1, order=kwargs["order"])
            return s

        with patch(
            "autoflowcfd.cli.solve.mesh_loader.load_mesh_for_solver",
            return_value=(MagicMock(), MagicMock()),
        ), patch(
            "autoflowcfd.core.fr_solver.FRSolver", side_effect=_fake_frsolver
        ), patch(
            "autoflowcfd.cli.solve.wall_distance.compute_wall_distance_for_solver"
        ), patch(
            "autoflowcfd.cli.solve.aero_coefficients.resolve_reference_area"
        ):
            solver, iteration, metadata = rebuild_solver_from_checkpoint(str(ckpt))

        assert solver.current_order == 0  # 与保存的 (n_cells,1,n_vars) 状态形状一致
        assert solver.order == 2  # 不是 0——恢复成真实的爬坡目标
        assert metadata["target_order"] == 2

    def test_old_checkpoint_without_target_order_field_falls_back_safely(self, tmp_path):
        """修复之前写出的 checkpoint 根本没有 target_order 键——
        metadata.get("target_order", order) 必须退回 order，重现旧的
        （恒有 order==target_order）行为，而不是因为缺键崩溃。
        """
        write_checkpoint(
            _fake_solver(n_cells=2, n_sps=18, order=2), str(tmp_path), 100, "volume.nas",
            order=2, turbulence_model="sst", backend="cpu", quiet=True,
        )
        ckpt = tmp_path / "checkpoints" / "checkpoint_iter_000100.h5"
        # 去掉 target_order 键，模拟修复之前的 checkpoint。
        import h5py
        with h5py.File(ckpt, "r+") as f:
            del f["metadata"].attrs["target_order"]

        def _fake_frsolver(**kwargs):
            return _fake_solver(n_cells=2, n_sps=18, order=kwargs["order"])

        with patch(
            "autoflowcfd.cli.solve.mesh_loader.load_mesh_for_solver",
            return_value=(MagicMock(), MagicMock()),
        ), patch(
            "autoflowcfd.core.fr_solver.FRSolver", side_effect=_fake_frsolver
        ), patch(
            "autoflowcfd.cli.solve.wall_distance.compute_wall_distance_for_solver"
        ), patch(
            "autoflowcfd.cli.solve.aero_coefficients.resolve_reference_area"
        ):
            solver, iteration, metadata = rebuild_solver_from_checkpoint(str(ckpt))

        assert solver.order == 2
        assert solver.current_order == 2
