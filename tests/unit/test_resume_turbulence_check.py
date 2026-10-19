"""续算的湍流场被上界大面积截断时拒绝续算，`solve resume --reset-turbulence` 才重置（I-91）。

此前满足判据时自动把湍流场重置为来流值并继续算（只打一行 WARN），续算已不是原算例；检查只在 Order Continuation
路径上做；比例的分母含四面体零填充槽位。
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from click.testing import CliRunner

from autoflowcfd.cli.main import cli
from autoflowcfd.core.freestream_defaults import MU_MOLECULAR, P_INF, RHO_INF
from autoflowcfd.core.time_integration import TimeIntegrationScheme
from autoflowcfd.fr.native_padding import real_row_mask
from tests.validation._channel_mesh import build_channel_mesh, build_channel_mesh_prism, channel_wall_source

_BC = {"x_min": "VELOCITY_INLET", "x_max": "PRESSURE_OUTLET", "wall_bottom": "WALL", "wall_top": "WALL",
       "z_min": "SYMMETRY", "z_max": "SYMMETRY"}
_META = {"input_file": "dummy.nas", "order": 1, "turbulence_model": "sst", "backend": "cpu", "surface_mesh": None}


def _fake_solver():
    solver = MagicMock()
    solver.solve.return_value = SimpleNamespace(converged=False, iterations=10, final_residual=1.0)
    solver.order = solver.current_order = 1
    return solver


@pytest.mark.parametrize("flag", [[], ["--reset-turbulence"]])
def test_single_node_resume_forwards_the_option(tmp_path, flag):
    ck = tmp_path / "checkpoint_iter_000150.h5"
    ck.write_bytes(b"")
    solver = _fake_solver()
    with patch("autoflowcfd.cli.solve.commands.rebuild_solver_from_checkpoint") as reb, \
            patch("autoflowcfd.cli.solve.checkpoint_io.write.save_results"), \
            patch("autoflowcfd.cli.solve.checkpoint_io.write.write_checkpoint"), \
            patch("autoflowcfd.cli.solve.commands._report_aerodynamic_coefficients"):
        reb.return_value = (solver, 150, dict(_META))
        res = CliRunner().invoke(cli, ["solve", "resume", str(ck), "-n", "10"] + flag)
    assert res.exit_code == 0, res.output
    assert solver._reset_turbulence_on_resume is bool(flag)


@pytest.mark.parametrize("flag", [[], ["--reset-turbulence"]])
def test_distributed_resume_forwards_the_option(tmp_path, flag):
    ck = tmp_path / "checkpoint_iter_000150.h5"
    ck.write_bytes(b"")
    solver = _fake_solver()
    with patch("autoflowcfd.cli.solve.distributed_checkpoint_io.rebuild_distributed_solver_from_checkpoint") as reb, \
            patch("autoflowcfd.core.mpi.distributed_checkpoint.distributed_save_checkpoint"):
        reb.return_value = (solver, 150, dict(_META))
        res = CliRunner().invoke(cli, ["solve", "resume", str(ck), "-n", "10", "--n-ranks", "2"] + flag)
    assert reb.called and "no such option" not in res.output.lower(), res.output
    assert solver._reset_turbulence_on_resume is bool(flag)


def _resumed_sst_solver(build, order):
    from autoflowcfd.core.fr_solver import FRSolver
    from autoflowcfd.core.fr_solver.turbulence.wall_distance import apply_wall_distance_source

    mesh = build(order, 3, 3, 2, 0.4, 0.1, 0.08)
    mesh.boundary_bc_types = dict(_BC)
    s = FRSolver(mesh=mesh, order=order, turb_model_name="SST", time_scheme=TimeIntegrationScheme.SSP_RK3,
                 vel_inf=10.0, rho_inf=RHO_INF, p_inf=P_INF, mu_molecular=MU_MOLECULAR)
    apply_wall_distance_source(s, channel_wall_source(0.4, 0.1, 0.08))
    s.order_continuation_enabled = False
    s._resumed_from_checkpoint = True
    s._reset_turbulence_on_resume = False
    return s


def test_fixed_order_loop_refuses_a_clamped_field():
    """定阶循环同样检查（此前只在 Order Continuation 路径上）。"""
    s = _resumed_sst_solver(build_channel_mesh_prism, 1)
    s.turb_model.k_field[:] = s.turb_model.k_max
    with pytest.raises(RuntimeError, match="--reset-turbulence"):
        s.solve(max_iter=1, dt=1e-5, tol=0.0)


def test_reset_option_restores_freestream_and_continues():
    s = _resumed_sst_solver(build_channel_mesh_prism, 1)
    s.turb_model.k_field[:] = s.turb_model.k_max
    s._reset_turbulence_on_resume = True
    s.solve(max_iter=1, dt=1e-5, tol=0.0)
    assert float(s.turb_model.k_field.max()) < 0.5 * s.turb_model.k_max


def test_share_counts_real_degrees_of_freedom_only():
    """P2 四面体：填充槽位上的值不参与判据（此前分子分母都含填充槽位）。"""
    s = _resumed_sst_solver(build_channel_mesh, 2)
    k = s.turb_model.k_field
    is_prism = np.arange(k.shape[0]) < int(s.mesh.n_prism_cells)
    padded = ~real_row_mask(is_prism, k.shape[1], 2).reshape(k.shape)
    assert padded.mean() > 0.4                    # 填充槽位占存储的 8/18
    k[padded] = s.turb_model.k_max
    from autoflowcfd.core.utils.order_continuation.turbulence_reset import check_resumed_turbulence

    check_resumed_turbulence(s)                   # 真实自由度都健康：不报错、不重置
    assert np.all(k[padded] == s.turb_model.k_max)
