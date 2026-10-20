"""`solve transient --init-from` 的阶数处理与起步方式（2026-10-08，单机 CPU / 单 GPU；分布式见
`test_distributed_init_from_checkpoint.py`）。

两处缺陷（`core/utils/order_continuation/initial_field.py` 模块文档）：
* checkpoint 阶数低于目标阶数（稳态 P1 阶段存的 checkpoint -> P2 瞬态）时恢复被拒绝；
* 阶数一致时 `solve()` 照样走 Order Continuation，重置回 P0 均匀来流——init-from 的初场被静默丢掉。
"""

import click
import numpy as np
import pytest

from autoflowcfd.cli.solve.checkpoint_io.restore import restore_state_from_checkpoint
from autoflowcfd.core.time_integration import TimeIntegrationScheme
from autoflowcfd.core.utils.order_continuation.initial_field import start_from_checkpoint_field
from autoflowcfd.fr.order_interp import apply_order_interp
from tests.unit.test_gpu_solver_order_continuation import _patch_gpu_modules  # noqa: F401（自动夹具）
from tests.unit.test_sa_neg_solver import _sa_channel
from autoflowcfd.core.freestream_defaults import MU_MOLECULAR, P_INF, RHO_INF

DUAL = TimeIntegrationScheme.DUAL_TIME


def _checkpoint_fields(solver, rng):
    """可辨认的"稳态解"：在来流上叠加逐解点扰动（平均流与 SA 场）。"""
    U = solver.state.U.copy()
    U[..., 1] *= 1.0 + 0.2 * rng.random(U.shape[:2])
    nu = solver.turb_model.nu_tilde_field * (1.0 + 0.5 * rng.random(U.shape[:2]))
    return {"U_sps": U, "nu_tilde_field": nu}


def test_lower_order_checkpoint_matches_the_order_continuation_prolongation():
    """P1 checkpoint -> P2 求解器：结果与 Order Continuation 自己把同一个 P1 状态升到 P2 逐位一致。"""
    from autoflowcfd.core.utils import order_continuation

    rng = np.random.default_rng(1)
    p1 = _sa_channel("prism", 1, DUAL)
    fields = _checkpoint_fields(p1, rng)
    p2 = _sa_channel("prism", 2, DUAL)
    _, ckpt_order = restore_state_from_checkpoint("p1.h5", p2, {"fields": fields})
    assert ckpt_order == 1

    p1.state.U = fields["U_sps"].copy()
    p1.state._update_primitives()
    p1.turb_model.set_transported_fields([fields["nu_tilde_field"].copy()])
    order_continuation.interpolate_to_new_order(p1, 2)
    np.testing.assert_array_equal(p2.state.U, p1.state.U)
    np.testing.assert_array_equal(p2.turb_model.nu_tilde_field, p1.turb_model.nu_tilde_field)


def test_init_from_field_survives_the_first_step():
    """阶数一致时初场此前在第一次 `solve()` 里被重置成来流（ρu 均值回到来流值）。"""
    rng = np.random.default_rng(2)
    s = _sa_channel("prism", 2, DUAL, order_continuation=True)
    fields = _checkpoint_fields(s, rng)
    _, ckpt_order = restore_state_from_checkpoint("p2.h5", s, {"fields": fields})
    start_from_checkpoint_field(s, ckpt_order, report=False)
    assert s.order_continuation_enabled is False
    rho_u_inf = float(s.freestream["rho_inf"] * s.freestream["vel_inf"])
    before = float(s.state.U[..., 1].mean())
    s.solve(max_iter=1, dt=1e-5, tol=0.0)
    assert s.current_order == 2
    after = float(s.state.U[..., 1].mean())
    assert abs(after - before) < 0.1 * abs(before - rho_u_inf), (before, after, rho_u_inf)


def test_prolongated_start_applies_the_positivity_limiter(monkeypatch):
    s = _sa_channel("prism", 2, DUAL)
    calls = []
    monkeypatch.setattr(s, "_limit_prolongated_state", lambda: calls.append(1))
    start_from_checkpoint_field(s, 1, report=False)
    start_from_checkpoint_field(s, 2, report=False)
    assert calls == [1], "只有延拓过才需要在新阶数点集上做正性限制"


def test_higher_order_checkpoint_is_rejected():
    rng = np.random.default_rng(3)
    fields = _checkpoint_fields(_sa_channel("prism", 2, DUAL), rng)
    with pytest.raises(click.ClickException, match="高于目标阶数"):
        restore_state_from_checkpoint("p2.h5", _sa_channel("prism", 1, DUAL), {"fields": fields})


def test_single_gpu_init_from_lower_order_through_the_host_view():
    """单 GPU 经主机视图恢复（numpy 替身），延拓与 CPU 同一个函数，结果写回设备数组。"""
    from autoflowcfd.core.gpu.solver.gpu_solver import GPUFRSolver
    from tests.unit.test_single_gpu_unified import _BC
    from tests.validation._channel_mesh import build_channel_mesh_prism, channel_wall_source

    rng = np.random.default_rng(4)
    p1 = _sa_channel("prism", 1, DUAL)
    fields = _checkpoint_fields(p1, rng)
    mesh = build_channel_mesh_prism(2, 3, 4, 2, 0.4, 0.1, 0.08)
    mesh.boundary_bc_types = dict(_BC)
    g = GPUFRSolver(mesh=mesh, order=2, turb_model_name="SA", time_scheme=DUAL, vel_inf=30.0,
                    wall_distance_source=channel_wall_source(0.4, 0.1, 0.08), rho_inf=RHO_INF, p_inf=P_INF, mu_molecular=MU_MOLECULAR)
    with g.edit_host_state() as host:
        _, ckpt_order = restore_state_from_checkpoint("p1.h5", host, {"fields": fields})
    start_from_checkpoint_field(g, ckpt_order, report=False)
    assert ckpt_order == 1 and g.order_continuation_enabled is False

    n_prism = int(mesh.n_prism_cells)
    want_U = apply_order_interp(fields["U_sps"], n_prism, 1, 2)
    got_U = np.asarray(g.host_view().state.U)
    # 延拓后在 P2 点集上做了正性限制（向单元均值收缩，均值不变）；本场处处可容许，限制器不动它
    np.testing.assert_allclose(got_U, want_U, rtol=1e-12)
    want_nu = g.turb_model_gpu.mapped_fields(lambda f: apply_order_interp(f, n_prism, 1, 2), np,
                                             fields=[fields["nu_tilde_field"]])[0]
    np.testing.assert_allclose(np.asarray(g.turb_model_gpu.nu_tilde_field), want_nu, rtol=1e-12)


def _write_p1_steady_checkpoint(tmp_path, rng):
    """稳态 P1（Newton-Krylov）的求解器写出的真实 checkpoint 文件；返回 `(路径, 写入的场)`。"""
    from autoflowcfd.cli.solve.checkpoint_io import write_checkpoint

    p1 = _sa_channel("prism", 1)
    fields = _checkpoint_fields(p1, rng)
    p1.state.U = fields["U_sps"].copy()
    p1.state._update_primitives()
    p1.turb_model.set_transported_fields([fields["nu_tilde_field"].copy()])
    path = write_checkpoint(p1, str(tmp_path), 7, "channel.nas", 1, "sa", "cpu", quiet=True)
    assert path is not None
    return path, fields


def test_steady_init_from_a_lower_order_checkpoint_file(tmp_path):
    """`solve steady --init-from`：稳态 P1 的 checkpoint 文件 -> 稳态 P2，经稳态 / 瞬态共用的入口。

    2026-10-14 以前 `solve steady` 没有这个选项，`solve resume` 又只能沿用 checkpoint 记录的目标阶数：
    低阶解收敛后要升阶只能从 P0 重新爬坡。"""
    from autoflowcfd.cli.solve.checkpoint_io import start_single_node_from_checkpoint
    from autoflowcfd.core.time_integration.base import STEADY_DT

    rng = np.random.default_rng(5)
    path, fields = _write_p1_steady_checkpoint(tmp_path, rng)
    p2 = _sa_channel("prism", 2, order_continuation=True)
    start_single_node_from_checkpoint(p2, path)
    assert p2.order_continuation_enabled is False

    n_prism = int(p2.mesh.n_prism_cells)
    np.testing.assert_allclose(p2.state.U, apply_order_interp(fields["U_sps"], n_prism, 1, 2), rtol=1e-12)

    rho_u_inf = float(p2.freestream["rho_inf"] * p2.freestream["vel_inf"])
    before = float(p2.state.U[..., 1].mean())
    result = p2.solve(max_iter=2, dt=STEADY_DT, tol=0.0)
    assert p2.current_order == 2 and result.iterations == 2
    after = float(p2.state.U[..., 1].mean())
    assert np.all(np.isfinite(p2.state.U))
    assert abs(after - before) < 0.5 * abs(before - rho_u_inf), "初场被重置回来流"


@pytest.mark.parametrize("extra", [[], ["--n-ranks", "2"], ["--backend", "gpu", "--multi-gpu", "--n-ranks", "2"]])
def test_steady_init_from_reaches_every_backend(tmp_path, monkeypatch, extra):
    """`--init-from` 进入运行参数（单机 / CPU 分布式 / 多 GPU 同一个 `run_solve`）。"""
    from unittest.mock import patch

    from click.testing import CliRunner

    from autoflowcfd.cli.solve.steady import command

    grid = tmp_path / "volume.pkl"
    grid.write_bytes(b"")
    ckpt = tmp_path / "checkpoint_iter_000007.h5"
    ckpt.write_bytes(b"")
    with patch("autoflowcfd.cli.solve.run.run_solve") as run:
        out = CliRunner().invoke(command.solve_steady, [str(grid), "--order", "2", "--init-from", str(ckpt)] + extra)
    assert out.exit_code == 0, out.output
    assert run.call_args.args[0].init_checkpoint == str(ckpt)


def test_every_backend_starts_from_the_checkpoint_before_solving():
    """单机与分布式都在构造求解器之后、`solve()` 之前从 checkpoint 起步（`run.py::_build`）。"""
    import inspect

    from autoflowcfd.cli.solve import run

    build = inspect.getsource(run._build)
    assert build.count("start_single_node_from_checkpoint(solver, run.init_checkpoint)") == 1
    assert build.count("start_distributed_from_checkpoint(solver, run.init_checkpoint)") == 1
    assert inspect.getsource(run.run_solve).index("_build(run)") < inspect.getsource(run.run_solve).index(
        "solver.solve(")


