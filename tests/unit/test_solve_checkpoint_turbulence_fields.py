"""2026-08-23 在 79.1 万单元 cube_demo 的真实续算排查中发现的真实缺陷的回归
测试：`write_checkpoint` 只持久化平均流状态（`U_sps`/`Q_sps`），从不存
`turb_model.k_field`/`omega_field`。

`rebuild_solver_from_checkpoint` 经一次全新的 `FRSolver(...)` 调用重建
求解器，其内部的 `SSTModelFR.__init__` 无条件地用全新运行同样的均匀
"刚开始求解"猜测值（k=1e-6，omega=1.0）给 `k_field`/`omega_field` 做种子。
由于 checkpoint 从不带真实的、已收敛的湍流场，续算静默地得到一个平均流
恰好是收敛状态、湍流场却被重置成初始猜测的求解器——续算边界上一个真实的
物理间断，在真实重建之后直接检查 `solver.turb_model.omega_field` 得到确认
（均匀的 1.0，与全新构造的默认值一致，而不是任何随空间变化的收敛 SST 场）。

修复：`write_checkpoint` 现在也持久化 `k_field`/`omega_field`（经 `hasattr`，
对 `turb_model=None` 或 LES 这类不暴露这些属性的纯亚格子模型不起作用）；
`rebuild_solver_from_checkpoint` 在它们存在时恢复，形状检查与已有的
`U_sps` 检查一致，对修复之前写出的 checkpoint 有安全的向后兼容处理
（保留全新的均匀猜测值并打印警告）。
"""

from types import SimpleNamespace

from tests.unit._host_state import with_host_state
from autoflowcfd.core.time_integration.base import TimeIntegrationScheme
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from autoflowcfd.cli.solve.checkpoint_io import write_checkpoint, rebuild_solver_from_checkpoint


def _fake_solver(n_cells=2, n_sps=1, n_vars=7, order=0, k=None, omega=None, with_turb_model=True):
    U = np.ones((n_cells, n_sps, n_vars))
    state = SimpleNamespace(U=U, Q=U.copy(), n_sps=n_sps, n_vars=n_vars)
    state._update_primitives = lambda: None
    turb_model = None
    if with_turb_model:
        # 真实 SST 模型（checkpoint 读写按 `TransportedTurbulence.TRANSPORTED_FIELDS`）；omega_inf=1.0
        # 是恢复时可容许性投影读的来流值（sst/log_omega.py::admissible_omega）
        from autoflowcfd.core.turbulence.sst import SSTModelFR

        turb_model = SSTModelFR(n_cells, n_sps, k_inf=1e-6, omega_inf=1.0)
        if k is not None:
            turb_model.k_field = k
        if omega is not None:
            turb_model.omega_field = omega
    return with_host_state(SimpleNamespace(
        state=state,
        order=order,
        current_order=order,
        turb_model=turb_model,
        mesh=SimpleNamespace(n_prism_cells=n_cells),
        # 真实求解器在湍流初始化时设置（恢复湍流场后跳过产生项斜坡用到它）
        _turb_production_ramp_steps=50,
        freestream={"rho_inf": 1.225, "vel_inf": 33.33, "p_inf": 101325.0},
        # write_checkpoint 记录时间格式（2026-09-25）：真实求解器恒有 time_integrator
        time_integrator=SimpleNamespace(scheme=TimeIntegrationScheme.SSP_RK3),
    ))


class TestWriteCheckpointStoresTurbulenceFields:
    def test_k_and_omega_written_when_turb_model_present(self, tmp_path):
        k = np.array([[0.42], [0.73]])
        omega = np.array([[123.4], [567.8]])
        write_checkpoint(
            _fake_solver(n_cells=2, n_sps=1, k=k, omega=omega), str(tmp_path), 100, "volume.nas",
            order=0, turbulence_model="sst", backend="cpu", quiet=True,
        )
        import h5py
        ckpt = tmp_path / "checkpoints" / "checkpoint_iter_000100.h5"
        with h5py.File(ckpt, "r") as f:
            sol = f["solution"]
            assert "k_field" in sol
            assert "omega_field" in sol
            np.testing.assert_array_equal(sol["k_field"][:], k)
            np.testing.assert_array_equal(sol["omega_field"][:], omega)

    def test_no_turbulence_fields_written_when_turb_model_is_none(self, tmp_path):
        """--turbulence none：不能凭空造出 k_field/omega_field 键。"""
        write_checkpoint(
            _fake_solver(with_turb_model=False), str(tmp_path), 100, "volume.nas",
            order=0, turbulence_model="none", backend="cpu", quiet=True,
        )
        import h5py
        ckpt = tmp_path / "checkpoints" / "checkpoint_iter_000100.h5"
        with h5py.File(ckpt, "r") as f:
            assert "k_field" not in f["solution"]
            assert "omega_field" not in f["solution"]


    def test_only_the_conserved_state_is_stored(self, tmp_path):
        """原始变量由守恒变量导出、没有读取方（2026-10-08 以前另存 `Q_sps`，占单机 checkpoint 将近一半体积）。"""
        write_checkpoint(
            _fake_solver(with_turb_model=False), str(tmp_path), 100, "volume.nas",
            order=0, turbulence_model="none", backend="cpu", quiet=True,
        )
        import h5py
        with h5py.File(tmp_path / "checkpoints" / "checkpoint_iter_000100.h5", "r") as f:
            assert "U_sps" in f["solution"] and "Q_sps" not in f["solution"]


class TestRebuildRestoresTurbulenceFields:
    def _rebuild(self, ckpt_path, n_cells=2, n_sps=1, k_fresh=None, omega_fresh=None):
        """模仿 rebuild_solver_from_checkpoint 的 FRSolver(...) 调用返回一个
        *全新构造*的求解器（均匀猜测值的湍流场，与 checkpoint 里的不同），这样
        恢复才真的被测到，而不只是一次什么都不做的原样拷贝。
        """
        def _fake_frsolver(**kwargs):
            return _fake_solver(
                n_cells=n_cells, n_sps=n_sps, order=kwargs["order"],
                k=k_fresh, omega=omega_fresh,
            )

        with patch(
            "autoflowcfd.cli.solve.mesh_loader.load_mesh_for_solver",
            return_value=(MagicMock(), MagicMock()),
        ), patch(
            "autoflowcfd.core.FRSolver", side_effect=_fake_frsolver
        ), patch(
            "autoflowcfd.cli.solve.wall_distance.compute_wall_distance_for_solver"
        ):
            return rebuild_solver_from_checkpoint(str(ckpt_path))

    def test_resume_restores_converged_turbulence_field_not_fresh_guess(self, tmp_path):
        k_converged = np.array([[0.017], [0.055]])
        omega_converged = np.array([[842.0], [1953.0]])
        write_checkpoint(
            _fake_solver(n_cells=2, n_sps=1, k=k_converged, omega=omega_converged),
            str(tmp_path), 3000, "volume.nas",
            order=0, turbulence_model="sst", backend="cpu", quiet=True,
        )
        ckpt = tmp_path / "checkpoints" / "checkpoint_iter_003000.h5"

        # 全新构造的 FRSolver() 通常以均匀的"刚开始"猜测值做种子——故意与上面
        # checkpoint 里的值不同，这样静默的"没有恢复"会被抓到。
        k_fresh = np.full((2, 1), 1e-6)
        omega_fresh = np.full((2, 1), 1.0)
        solver, iteration, metadata = self._rebuild(
            ckpt, k_fresh=k_fresh, omega_fresh=omega_fresh,
        )

        np.testing.assert_array_equal(solver.turb_model.k_field, k_converged)
        np.testing.assert_array_equal(solver.turb_model.omega_field, omega_converged)

    def test_old_checkpoint_without_turbulence_fields_falls_back_to_fresh_guess(self, tmp_path, capsys):
        """修复之前的 checkpoint 根本没有 k_field/omega_field——续算不能崩溃，
        必须保留 FRSolver.__init__ 全新构造给出的默认值（有文档记录的、向后兼容的
        降级），同时警告用户。
        """
        write_checkpoint(
            _fake_solver(n_cells=2, n_sps=1), str(tmp_path), 3000, "volume.nas",
            order=0, turbulence_model="sst", backend="cpu", quiet=True,
        )
        ckpt = tmp_path / "checkpoints" / "checkpoint_iter_003000.h5"
        import h5py
        with h5py.File(ckpt, "r+") as f:
            del f["solution"]["k_field"]
            del f["solution"]["omega_field"]

        k_fresh = np.full((2, 1), 1e-6)
        omega_fresh = np.full((2, 1), 1.0)
        solver, iteration, metadata = self._rebuild(
            ckpt, k_fresh=k_fresh, omega_fresh=omega_fresh,
        )

        np.testing.assert_array_equal(solver.turb_model.k_field, k_fresh)
        np.testing.assert_array_equal(solver.turb_model.omega_field, omega_fresh)
        assert "旧版本" in capsys.readouterr().out

    def test_turbulence_field_shape_mismatch_rejected(self, tmp_path):
        """U_sps 形状相符（所以原有的平均流检查会放行），但 checkpoint 里的
        k_field/omega_field 形状与重新构造的 turb_model 不符——必须被新增的专门
        检查抓到，而不是静默地广播/截断。
        """
        write_checkpoint(
            _fake_solver(n_cells=2, n_sps=1, k=np.zeros((2, 1)), omega=np.ones((2, 1))),
            str(tmp_path), 3000, "volume.nas",
            order=0, turbulence_model="sst", backend="cpu", quiet=True,
        )
        ckpt = tmp_path / "checkpoints" / "checkpoint_iter_003000.h5"

        import click
        # 重新构造出的求解器 U 形状相符（n_cells=2）但 turb_model 形状不符
        # （n_cells=5）——模拟湍流模型的场形状与平均流状态各自漂移的情形。
        def _fake_frsolver(**kwargs):
            from autoflowcfd.core.turbulence.sst import SSTModelFR

            s = _fake_solver(n_cells=2, n_sps=1, order=kwargs["order"])
            s.turb_model = SSTModelFR(5, 1, k_inf=0.0, omega_inf=1.0)
            return s

        with patch(
            "autoflowcfd.cli.solve.mesh_loader.load_mesh_for_solver",
            return_value=(MagicMock(), MagicMock()),
        ), patch(
            "autoflowcfd.core.FRSolver", side_effect=_fake_frsolver
        ), patch(
            "autoflowcfd.cli.solve.wall_distance.compute_wall_distance_for_solver"
        ), pytest.raises(click.ClickException):
            rebuild_solver_from_checkpoint(str(ckpt))


class TestPhaseInitialResidualPersistence:
    """2026-08-23 发现的真实缺陷的回归测试（同一次真实续算排查）：Order
    Continuation 的残差下降升阶判据（order_continuation.py 里的
    `initial_residual_this_order`）是纯局部变量，跨 `solve resume` 的进程边界
    不记得阶段真实的起始残差——完整机制见
    order_continuation.py::run_order_continuation。这里测的是修复里 checkpoint
    往返的那一半（write_checkpoint/rebuild_solver_from_checkpoint 持久化
    `solver._phase_initial_residual`）；升阶判据本身的行为由
    tests/unit/test_order_continuation_resume.py 覆盖。
    """

    def _rebuild(self, ckpt_path, n_cells=2, n_sps=1):
        def _fake_frsolver(**kwargs):
            return _fake_solver(n_cells=n_cells, n_sps=n_sps, order=kwargs["order"])

        with patch(
            "autoflowcfd.cli.solve.mesh_loader.load_mesh_for_solver",
            return_value=(MagicMock(), MagicMock()),
        ), patch(
            "autoflowcfd.core.FRSolver", side_effect=_fake_frsolver
        ), patch(
            "autoflowcfd.cli.solve.wall_distance.compute_wall_distance_for_solver"
        ):
            return rebuild_solver_from_checkpoint(str(ckpt_path))

    def test_written_and_restored_exactly(self, tmp_path):
        solver = _fake_solver(n_cells=2, n_sps=1)
        solver._phase_initial_residual = 12345.6789
        write_checkpoint(
            solver, str(tmp_path), 3000, "volume.nas",
            order=0, turbulence_model="sst", backend="cpu", quiet=True,
        )
        ckpt = tmp_path / "checkpoints" / "checkpoint_iter_003000.h5"

        import h5py
        with h5py.File(ckpt, "r") as f:
            assert f["metadata"].attrs["phase_initial_residual"] == pytest.approx(12345.6789)

        restored, _, _ = self._rebuild(ckpt)
        assert restored._phase_initial_residual == pytest.approx(12345.6789)

    def test_not_yet_set_is_skipped_not_written_as_none(self, tmp_path):
        """h5py 的 attrs 不接受 None——一步都没走过的求解器（该属性从未赋值）
        不能让 write_checkpoint 出错，这个键必须直接缺席，而不是写成某个哨兵值。
        """
        solver = _fake_solver(n_cells=2, n_sps=1)
        assert not hasattr(solver, "_phase_initial_residual")
        write_checkpoint(
            solver, str(tmp_path), 3000, "volume.nas",
            order=0, turbulence_model="sst", backend="cpu", quiet=True,
        )
        ckpt = tmp_path / "checkpoints" / "checkpoint_iter_003000.h5"

        import h5py
        with h5py.File(ckpt, "r") as f:
            assert "phase_initial_residual" not in f["metadata"].attrs

    def test_old_checkpoint_without_field_leaves_attribute_unset(self, tmp_path):
        """修复之前的 checkpoint 根本没有 phase_initial_residual——续算不能崩溃，
        也不能凭空造一个值；order_continuation.py 经 getattr(...,None) 检出缺失并
        安全地退回（那一半见 test_order_continuation_resume.py）。
        """
        solver = _fake_solver(n_cells=2, n_sps=1)
        write_checkpoint(
            solver, str(tmp_path), 3000, "volume.nas",
            order=0, turbulence_model="sst", backend="cpu", quiet=True,
        )
        ckpt = tmp_path / "checkpoints" / "checkpoint_iter_003000.h5"
        # 不需要去掉任何东西——这个求解器从未设置过该属性，write_checkpoint
        # 已经跳过它（上一个测试）。

        restored, _, _ = self._rebuild(ckpt)
        assert getattr(restored, "_phase_initial_residual", None) is None
