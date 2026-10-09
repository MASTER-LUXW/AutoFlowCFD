"""solve_checkpoint_io.py 持久化 surface_mesh 的单元测试。

这里修掉的真实缺口（2026-08-22）：`write_checkpoint` 把 `input_file`/
`order`/`turbulence_model`/`backend` 存进 checkpoint 元数据，让
`solve resume` 能自动重新加载体网格，却从不存 `surface_mesh`，尽管每个
调用点（solve_steady_command.py、solve_transient_command.py、
solve_commands.py 自己的 resume()）都拿得到它——输入是原始 .nas 体网格的
算例每次续算都得手动重新传 `-s <原始面网格>`，CLI 也无法提醒那个路径是
什么。`rebuild_solver_from_checkpoint` 现在在调用方没有显式给出时退回
存下来的值，与 `backend`/`order`/`turbulence_model` 已有的行为一致。
"""

from types import SimpleNamespace

from tests.unit._host_state import with_host_state
from autoflowcfd.core.time_integration.base import TimeIntegrationScheme
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from autoflowcfd.cli.solve.checkpoint_io import write_checkpoint, rebuild_solver_from_checkpoint


def _fake_solver(n_cells=2, n_sps=1, n_vars=7):
    U = np.ones((n_cells, n_sps, n_vars))
    state = SimpleNamespace(U=U, Q=U.copy(), n_sps=n_sps, n_vars=n_vars)
    state._update_primitives = lambda: None
    return with_host_state(SimpleNamespace(
        state=state,
        freestream={"rho_inf": 1.225, "vel_inf": 33.33, "p_inf": 101325.0},
        # write_checkpoint 记录时间格式（2026-09-25）：真实求解器恒有 time_integrator
        time_integrator=SimpleNamespace(scheme=TimeIntegrationScheme.SSP_RK3),
    ))


class TestWriteCheckpointSurfaceMeshPersistence:
    def test_surface_mesh_stored_when_given(self, tmp_path):
        write_checkpoint(
            _fake_solver(), str(tmp_path), 100, "volume.nas", 0, "sst", "cpu",
            surface_mesh="surface.nas", quiet=True,
        )
        import h5py
        ckpt = tmp_path / "checkpoints" / "checkpoint_iter_000100.h5"
        with h5py.File(ckpt, "r") as f:
            assert f["metadata"].attrs["surface_mesh"].decode("utf-8") == "surface.nas"

    def test_surface_mesh_key_omitted_when_none(self, tmp_path):
        """h5py 的 attrs 存不了 None——必须整个跳过这个键，而不是写一个会破坏
        往返的空值/哨兵值。
        """
        write_checkpoint(
            _fake_solver(), str(tmp_path), 100, "volume.pkl", 0, "sst", "cpu",
            surface_mesh=None, quiet=True,
        )
        import h5py
        ckpt = tmp_path / "checkpoints" / "checkpoint_iter_000100.h5"
        with h5py.File(ckpt, "r") as f:
            assert "surface_mesh" not in f["metadata"].attrs


class TestRebuildSolverSurfaceMeshFallback:
    """`load_mesh_for_solver`/FRSolver 构造/壁面距离对单元测试太重（真实网格
    几何）——这里用替身，只隔离被测的 surface_mesh 解析逻辑。
    """

    def _write_and_rebuild(self, tmp_path, stored_surface_mesh, call_surface_mesh):
        write_checkpoint(
            _fake_solver(), str(tmp_path), 100, "volume.nas", 0, "sst", "cpu",
            surface_mesh=stored_surface_mesh, quiet=True,
        )
        ckpt = tmp_path / "checkpoints" / "checkpoint_iter_000100.h5"

        fake_mesh = MagicMock()
        fake_volume_data = MagicMock()
        fake_solver = _fake_solver()

        with patch(
            "autoflowcfd.cli.solve.mesh_loader.load_mesh_for_solver",
            return_value=(fake_mesh, fake_volume_data),
        ) as mock_load, patch(
            "autoflowcfd.core.FRSolver", return_value=fake_solver
        ), patch(
            "autoflowcfd.cli.solve.wall_distance.compute_wall_distance_for_solver"
        ):
            rebuild_solver_from_checkpoint(str(ckpt), surface_mesh=call_surface_mesh)

        return mock_load

    def test_falls_back_to_stored_surface_mesh_when_not_passed(self, tmp_path):
        mock_load = self._write_and_rebuild(
            tmp_path, stored_surface_mesh="surface.nas", call_surface_mesh=None,
        )
        assert mock_load.call_args.kwargs["surface_mesh"] == "surface.nas"

    def test_explicit_call_argument_overrides_stored_value(self, tmp_path):
        mock_load = self._write_and_rebuild(
            tmp_path, stored_surface_mesh="old_surface.nas", call_surface_mesh="new_surface.nas",
        )
        assert mock_load.call_args.kwargs["surface_mesh"] == "new_surface.nas"

    def test_none_when_neither_stored_nor_passed(self, tmp_path):
        mock_load = self._write_and_rebuild(
            tmp_path, stored_surface_mesh=None, call_surface_mesh=None,
        )
        assert mock_load.call_args.kwargs["surface_mesh"] is None
