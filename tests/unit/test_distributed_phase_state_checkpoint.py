"""Order Continuation 阶段起始残差随分布式 checkpoint 往返（2026-10-05）。

写入与恢复和单机共用 `core/utils/order_continuation/checkpoint_state.py`；此前分布式写入端
（`core/mpi/distributed_checkpoint/save.py`，CPU MPI 与多 GPU 共用）不写、四条分布式 resume 构造路径不读，
续算后升阶判据从 resume 的第一步重新起算。
"""

import pytest

from autoflowcfd.core.time_integration.base import TimeIntegrationScheme
from autoflowcfd.fr.operators import generate_fr_operators
from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh


@pytest.fixture(scope="module")
def mesh_and_ops():
    mesh = _build_synthetic_mixed_mesh(1)
    return mesh, generate_fr_operators(1)


def test_checkpoint_carries_the_phase_initial_residual(mesh_and_ops, tmp_path):
    """Order Continuation 阶段起始残差随分布式 checkpoint 往返（2026-10-05 以前分布式不写也不读：续算后
    升阶判据从 resume 的第一步重新起算）。写入/恢复与单机共用 order_continuation/checkpoint_state.py。"""
    import inspect

    from autoflowcfd.cli.solve import distributed_checkpoint_io as mod
    from autoflowcfd.core.mpi.distributed_checkpoint import (
        distributed_load_checkpoint, distributed_save_checkpoint,
    )
    from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver
    from autoflowcfd.core.utils.order_continuation.checkpoint_state import restore_phase_state

    mesh, ops = mesh_and_ops

    def _solver():
        return DistributedFRSolver(
            mesh=mesh, ops=ops, face_connectivity=mesh.face_connectivity,
            n_ranks=1, order=mesh.order, turb_model_name="none",
            time_scheme=TimeIntegrationScheme.SSP_RK3)
    solver = _solver()
    solver._phase_initial_residual = 4321.5
    path = distributed_save_checkpoint(solver, str(tmp_path), 3, "dummy_input.nas", mesh.order, "none", "cpu")
    restored = _solver()
    _U, metadata, _it = distributed_load_checkpoint(path, restored)
    assert metadata["phase_initial_residual"] == pytest.approx(4321.5)
    restore_phase_state(restored, metadata)
    assert restored._phase_initial_residual == pytest.approx(4321.5)
    # 第一次 step() 之前写出的 checkpoint 不带这个键（h5py attrs 不接受 None）
    fresh = _solver()
    path0 = distributed_save_checkpoint(fresh, str(tmp_path / "p0"), 0, "dummy_input.nas", mesh.order, "none",
                                        "cpu")
    _U, metadata0, _it = distributed_load_checkpoint(path0, fresh)
    assert "phase_initial_residual" not in metadata0
    # 四条分布式 resume 构造路径共用的收尾段恢复它
    assert "restore_phase_state(solver, metadata)" in inspect.getsource(
        mod.rebuild_distributed_solver_from_checkpoint)
