"""湍流输运方程的双时间步（2026-10-09，`core/turbulence/dual_time.py`）。

此前 DUAL_TIME 下湍流场每个物理步只按物理时间步长做一次显式更新：物理 dt 不受近壁湍流方程显式稳定极限约束，
通道算例 dt=1e-3 十步内 k 到 -1e9（cube_demo DDES P2 三步后 16 万个解点 k 为负）。现在湍流与平均流用同一套
双时间步（BDF 物理时间项 + 局部伪时间步内迭代）。本文件验证：

* 内迭代的不动点就是 BDF1/BDF2 方程的解（最小模型，可手算）；
* 真实求解器在修复前发散的步长下保持正性与有界；
* 产生项渐变按物理步计（每个物理步只推进一次）；
* 续算与连续计算逐位相同（湍流上一时间层随 checkpoint 持久化），缺了它就接不上；
* 单 GPU、CPU MPI、多 GPU 与单机 CPU 一致。
"""

import types
from types import SimpleNamespace

import numpy as np
import pytest

from autoflowcfd.core.time_integration import TimeIntegrationScheme
from autoflowcfd.core.turbulence.dual_time import (
    PhysicalTimeTerm, advance_turbulence_dual_time, reset_dual_time_history,
)
from autoflowcfd.core.utils.checkpoint_time import TURBULENCE_PREVIOUS_FIELD
from tests.unit.test_gpu_solver_order_continuation import _patch_gpu_modules  # noqa: F401（自动夹具）
from tests.validation._channel_mesh import build_channel_mesh_prism, channel_wall_source

DUAL = TimeIntegrationScheme.DUAL_TIME
LX, H, LZ = 0.4, 0.1, 0.08
_BC = {"x_min": "VELOCITY_INLET", "x_max": "PRESSURE_OUTLET", "wall_bottom": "WALL", "wall_top": "WALL",
       "z_min": "SYMMETRY", "z_max": "SYMMETRY"}


# ---------------------------------------------------------------------------
# 1. 不动点 = BDF 方程的解
# ---------------------------------------------------------------------------

class _LogDecay:
    """最小输运模型：df/dt = -lam f，未知量取 ln f（与 SST 的 ln omega 同一类变换），显式更新
    u* = u - dtau lam。"""

    TRANSPORTED_FIELDS = ("f",)

    def __init__(self, f, lam):
        self.f, self.lam, self.limited = np.array(f, dtype=float), lam, 0

    def transported_fields(self):
        return (self.f,)

    def set_transported_fields(self, fields):
        self.f = fields[0]

    def _unknown_from_field(self, j, field, xp):
        return xp.log(field)

    def _field_from_unknown(self, j, column, xp):
        return xp.exp(column)

    def apply_positivity_limiter(self):
        self.limited += 1


def test_inner_iteration_fixed_point_is_the_bdf_solution():
    lam, dt, dtau = 3.0, 0.1, 1.0                  # dtau = 10 dt：物理时间项点隐式，伪时间步大于物理步也稳定
    model = _LogDecay([2.0, 0.5, 7.0], lam)
    owner = SimpleNamespace()
    reset_dual_time_history(owner)

    def update(dtau_, term, first):
        model.f = np.exp(np.log(model.f) - dtau_ * lam)
        term.apply(model, np, dtau_)

    u0 = np.log(model.f)
    advance_turbulence_dual_time(owner, model, np, update, dtau, dt, 60)
    u1 = np.log(model.f)
    np.testing.assert_allclose(u1, u0 - lam * dt, rtol=0, atol=1e-13)           # BDF1：(u1 - u0)/dt = -lam
    np.testing.assert_array_equal(owner._dual_time_turb_prev[0], u0)
    assert model.limited == 60

    advance_turbulence_dual_time(owner, model, np, update, dtau, dt, 60)
    u2 = np.log(model.f)
    np.testing.assert_allclose(1.5 * u2 - 2.0 * u1 + 0.5 * u0, -lam * dt, rtol=0, atol=1e-13)   # BDF2
    np.testing.assert_array_equal(owner._dual_time_turb_prev[0], u1)


def test_physical_time_term_maps_b_to_the_evaluation_view_once():
    model = _LogDecay([2.0, 0.5], 1.0)
    calls = []

    def to_view(x):
        calls.append(1)
        return np.concatenate([x, x[:1]])          # 视图 = 本地 + 1 个 halo

    term = PhysicalTimeTerm(model, np, None, 0.1, to_view)
    view = _LogDecay([2.0, 0.5, 2.0], 1.0)
    term.apply(view, np, 0.05)
    term.apply(view, np, 0.05)
    assert len(calls) == 1 and view.f.shape == (3,)
    assert view.f[2] == view.f[0]


def test_zero_inner_iterations_is_rejected():
    with pytest.raises(ValueError, match="内迭代次数"):
        advance_turbulence_dual_time(SimpleNamespace(), _LogDecay([1.0], 1.0), np, lambda *a: None, 1.0, 0.1, 0)


# ---------------------------------------------------------------------------
# 2. 单机 CPU：有界、渐变按物理步计、续算逐位相同
# ---------------------------------------------------------------------------

def _cpu(model, ny=6, vel=30.0, inner=5):
    from autoflowcfd.core.fr_solver import FRSolver
    from autoflowcfd.core.fr_solver.turbulence.wall_distance import apply_wall_distance_source

    mesh = build_channel_mesh_prism(1, ny, ny, 2, LX, H, LZ)
    mesh.boundary_bc_types = dict(_BC)
    s = FRSolver(mesh=mesh, order=1, turb_model_name=model, time_scheme=DUAL, vel_inf=vel,
                 dual_time_inner_iter=inner)
    s.order_continuation_enabled = False
    apply_wall_distance_source(s, channel_wall_source(LX, H, LZ))
    return s


@pytest.mark.parametrize("model", ["SST", "DDES", "SA"])
def test_turbulence_stays_bounded_at_a_time_step_far_above_the_explicit_limit(model):
    """dt=1e-3 远高于近壁单元的湍流显式稳定极限。修复前（一次按物理 dt 的显式更新）SST/DDES 的 k 十步内到
    -1e9。"""
    s = _cpu(model)
    initial = [np.array(f, copy=True) for f in s.turb_model.transported_fields()]
    for _ in range(10):
        s.step(1e-3)
    for name, f0, f in zip(s.turb_model.TRANSPORTED_FIELDS, initial, s.turb_model.transported_fields()):
        assert np.isfinite(f).all(), name
        assert np.abs(f).max() <= 1e3 * max(np.abs(f0).max(), 1e-12), (name, float(np.abs(f).max()))
        if model != "SA":                          # SA-neg 允许 nu_tilde 为负
            assert f.min() > 0.0, (name, float(f.min()))
    assert np.isfinite(s.state.U).all()


def test_production_ramp_advances_once_per_physical_step():
    s = _cpu("SST", ny=3, vel=10.0, inner=4)
    for _ in range(3):
        s.step(2e-4)
    assert s._turb_ramp_step == 3


def _perturbed(model="SST"):
    s = _cpu(model, ny=3, vel=10.0, inner=3)
    rng = np.random.default_rng(7)
    s.state.U[..., 1:4] *= 1.0 + 0.05 * rng.standard_normal(s.state.U[..., 1:4].shape)
    s.state._update_primitives()
    return s


@pytest.mark.parametrize("model", ["SST", "SA"])
def test_resumed_run_continues_bitwise_including_turbulence(tmp_path, model):
    from autoflowcfd.cli.solve.checkpoint_io.restore import restore_solver_state_from_fields
    from autoflowcfd.cli.solve.checkpoint_io.write import write_checkpoint
    from autoflowcfd.core.utils.checkpoint import CheckpointManager

    dt = 2e-4
    reference = _perturbed(model)
    reference.solve(max_iter=3, dt=dt, tol=0.0)

    first = _perturbed(model)
    first.solve(max_iter=2, dt=dt, tol=0.0)
    path = write_checkpoint(first, str(tmp_path), 2, "mesh.nas", order=1, turbulence_model=model.lower(),
                            backend="cpu", quiet=True)
    meta = CheckpointManager(SimpleNamespace(), output_dir=str(tmp_path)).load(path)[3]
    n_fields = len(first.turb_model.TRANSPORTED_FIELDS)
    assert meta["fields"][TURBULENCE_PREVIOUS_FIELD].shape == first.state.U.shape[:2] + (n_fields,)

    resumed = _perturbed(model)
    restore_solver_state_from_fields(resumed, meta["fields"], meta)
    resumed.solve(max_iter=1, dt=dt, tol=0.0)
    np.testing.assert_array_equal(resumed.state.U, reference.state.U)
    for a, b in zip(resumed.turb_model.transported_fields(), reference.turb_model.transported_fields()):
        np.testing.assert_array_equal(a, b)

    # 对照：不恢复湍流的上一时间层（续算第一步湍流退化成 BDF1）就接不上
    bdf1 = _perturbed(model)
    fields = {k: v for k, v in meta["fields"].items() if k != TURBULENCE_PREVIOUS_FIELD}
    restore_solver_state_from_fields(bdf1, fields, meta)
    bdf1.solve(max_iter=1, dt=dt, tol=0.0)
    assert any(not np.array_equal(a, b) for a, b in zip(bdf1.turb_model.transported_fields(),
                                                        reference.turb_model.transported_fields()))


def test_previous_level_of_another_model_is_not_restored():
    """换了湍流模型（未知量个数不同）时上一时间层没有意义：不恢复，第一个物理步用 BDF1。"""
    from autoflowcfd.core.utils.checkpoint_time import restore_turbulence_previous

    s = _cpu("SA", ny=3, vel=10.0)
    n_cells, n_sps = s.turb_model.transported_fields()[0].shape
    restore_turbulence_previous(s, np.zeros((n_cells, n_sps, 2)))
    assert s._dual_time_turb_prev is None
    restore_turbulence_previous(s, np.ones((n_cells, n_sps, 1)))
    assert s._dual_time_turb_prev[0].shape == (n_cells, n_sps)


# ---------------------------------------------------------------------------
# 3. 后端一致
# ---------------------------------------------------------------------------

def _assert_turbulence_matches(got_model, ref_model, tol, what):
    for name in tuple(ref_model.TRANSPORTED_FIELDS) + ("nu_t",):
        a, b = np.asarray(getattr(got_model, name)), np.asarray(getattr(ref_model, name))
        r = np.abs(a[:b.shape[0]] - b).max() / max(np.abs(b).max(), 1e-300)
        assert r <= tol, f"{what}：{name} 相对差 {r:.2e}"


@pytest.mark.parametrize("model", ["SST", "SA"])
def test_single_gpu_matches_cpu(model):
    from autoflowcfd.core.gpu.solver.gpu_solver import GPUFRSolver

    cpu = _cpu(model, ny=3, vel=10.0, inner=3)
    mesh = build_channel_mesh_prism(1, 3, 3, 2, LX, H, LZ)
    mesh.boundary_bc_types = dict(_BC)
    gpu = GPUFRSolver(mesh=mesh, order=1, turb_model_name=model, vel_inf=10.0, time_scheme=DUAL,
                      dual_time_inner_iter=3, wall_distance_source=channel_wall_source(LX, H, LZ))
    gpu.order_continuation_enabled = False
    for _ in range(3):
        cpu.step(2e-4)
        gpu.step(2e-4)
    assert gpu._turb_ramp_step == cpu._turb_ramp_step == 3
    _assert_turbulence_matches(gpu.turb_model_gpu, cpu.turb_model, 1e-8, "单 GPU")
    for a, b in zip(gpu._dual_time_turb_prev, cpu._dual_time_turb_prev):
        np.testing.assert_allclose(np.asarray(a), b, rtol=1e-8, atol=1e-12)


@pytest.mark.parametrize("model", ["SST", "SA"])
def test_cpu_distributed_matches_single_machine(model):
    from tests.unit.test_distributed_turbulence_bc_parity import _assert_same, _pair

    single, dist = _pair(DUAL, 1, model)
    for _ in range(3):
        single.step(2e-4)
        dist.step(2e-4)
    assert dist._turb_ramp_step == single._turb_ramp_step == 3
    _assert_same(single, dist, "DUAL_TIME 三个物理步", tol=1e-8)
    for a, b in zip(dist._dual_time_turb_prev, single._dual_time_turb_prev):
        np.testing.assert_allclose(a[:b.shape[0]], b, rtol=1e-8, atol=1e-12)


@pytest.mark.parametrize("model", ["SST", "SA"])
def test_cpu_distributed_checkpoint_carries_turbulence_previous_level(tmp_path, model):
    from autoflowcfd.core.mpi.distributed_checkpoint import distributed_load_checkpoint, distributed_save_checkpoint
    from tests.unit.test_distributed_turbulence_bc_parity import _pair

    _single, a = _pair(DUAL, 1, model)
    a.order_continuation_enabled = False
    a.solve(max_iter=2, dt=2e-4, tol=0.0)
    path = distributed_save_checkpoint(a, str(tmp_path), 2, "mesh.nas", 1, model.lower(), "cpu")
    _single_b, b = _pair(DUAL, 1, model)
    distributed_load_checkpoint(path, b)
    assert len(b._dual_time_turb_prev) == len(a.turb_model.TRANSPORTED_FIELDS)
    n = a.partition.n_local_cells
    for x, y in zip(b._dual_time_turb_prev, a._dual_time_turb_prev):
        np.testing.assert_array_equal(x[:n], y[:n])


@pytest.mark.parametrize("model", ["SST", "SA"])
def test_multi_gpu_turbulence_matches_single_machine(model, monkeypatch):
    """多 GPU 的湍流双时间步（真实 `_compute_turbulence_source_distributed` + halo 交换/重排的 `to_view`）
    与单机 CPU 同一物理步的湍流推进一致。"""
    import autoflowcfd.core.gpu.distributed.gpu_distributed_init.turb_source as ts_mod
    import autoflowcfd.core.gpu.residual.gpu_flux as gpu_flux_mod
    import autoflowcfd.core.gpu.residual.gpu_gradients as gpu_gradients_mod
    import autoflowcfd.core.gpu.residual.gpu_volume_contract as gpu_volume_contract_mod
    import autoflowcfd.core.gpu.turbulence.gpu_scalar_transport as gst_mod
    import autoflowcfd.core.gpu.turbulence.gpu_turbulence_sst as gpu_turbulence_sst_mod
    from autoflowcfd.core.gpu.distributed.gpu_distributed_init.turb_source import _GPUDistributedTurbSourceMixin
    from tests.unit._gpu_cupy_shim import patch_module_get_cupy
    from tests.unit._numpy_as_cupy import NumpyAsCupy
    from tests.unit.test_distributed_turbulence_bc_parity import _pair
    from tests.unit.test_multi_gpu_coupled_nk import _multi_gpu_standin

    patch_module_get_cupy(monkeypatch, [ts_mod, gpu_flux_mod, gpu_gradients_mod, gpu_volume_contract_mod, gst_mod,
                                        gpu_turbulence_sst_mod], NumpyAsCupy())
    monkeypatch.setattr(gpu_turbulence_sst_mod, "gpu_available", True)

    single, dist = _pair(DUAL, 1, model)
    rng = np.random.default_rng(5)
    m = single.turb_model
    m.restore_transported([f * (1 + 0.2 * rng.standard_normal(f.shape)) for f in m.transported_fields()])
    g = _multi_gpu_standin(single, dist)
    g._compute_turbulence_source_distributed = types.MethodType(
        _GPUDistributedTurbSourceMixin._compute_turbulence_source_distributed, g)
    reset_dual_time_history(g)

    dtau = np.asarray(single._compute_local_time_step())
    dt, n_inner = 2e-4, 3
    for _ in range(2):                                   # 第二步走 BDF2
        advance_turbulence_dual_time(
            single, single.turb_model, np,
            lambda d, term, first: single.compute_turbulence_source(d, term, first), dtau, dt, n_inner)
        advance_turbulence_dual_time(
            g, g.turb_model_gpu, np,
            lambda d, term, first: g._compute_turbulence_source_distributed(d, term, first),
            dtau.reshape(dtau.shape[0], -1)[:, :1], dt, n_inner,
            to_view=lambda x: g._permute_to_compact(g.gpu_halo.exchange(x)))
    assert g._turb_ramp_step == single._turb_ramp_step
    _assert_turbulence_matches(g.turb_model_gpu, single.turb_model, 1e-8, "多 GPU")
