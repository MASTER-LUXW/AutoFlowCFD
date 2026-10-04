# -*- coding: utf-8 -*-
"""湍流场的单元平均输出（`core/turbulence/output.py`）：checkpoint 的 `turb_cell_*` 字段与 VTK 导出。

2026-10-04 以前 VTK 导出的 `k`/`omega` 取自守恒解第 6、7 列——SST 状态数组里从未更新的历史
槽位，导出的永远是初值；`nut` 期望 checkpoint 里有 `mu_t`，写 checkpoint 的一方从未写过，于是
退化成用那两列算的 `k/omega`。下面的用例把状态数组的那两列写成与模型场完全不同的值：输出必须
跟随模型场。
"""

from types import SimpleNamespace

import numpy as np
import pytest

from autoflowcfd.core.time_integration.base import TimeIntegrationScheme
from autoflowcfd.core.turbulence.output import (
    CHECKPOINT_PREFIX, turbulence_cell_means, turbulence_fields_from_checkpoint,
)
from autoflowcfd.fr.native_padding import real_sps_per_cell


def _sst(n_prism, n_tet, order, rng):
    from autoflowcfd.core.turbulence.sst import SSTModelFR

    n_sps = (order + 1) ** 3 if n_prism else real_sps_per_cell(order)[1]
    m = SSTModelFR(n_prism + n_tet, n_sps, k_inf=1e-3, omega_inf=10.0)
    m.k_field = rng.uniform(0.1, 1.0, m.k_field.shape)
    m.omega_field = rng.uniform(100.0, 200.0, m.omega_field.shape)
    m.nu_t = rng.uniform(1e-5, 1e-4, m.k_field.shape)
    return m


def _real_mean(field, n_prism, order):
    npr, nte = real_sps_per_cell(order)
    return np.concatenate([field[:n_prism, :npr].mean(axis=1), field[n_prism:, :nte].mean(axis=1)])


def test_sst_outputs_follow_the_model_fields():
    rng = np.random.default_rng(0)
    m = _sst(3, 2, 1, rng)
    out = turbulence_cell_means(m, 3, 1)
    assert set(out) == {"k", "omega", "nut"}
    np.testing.assert_allclose(out["k"], _real_mean(m.k_field, 3, 1), rtol=1e-14)
    np.testing.assert_allclose(out["omega"], _real_mean(m.omega_field, 3, 1), rtol=1e-14)
    np.testing.assert_allclose(out["nut"], _real_mean(m.nu_t, 3, 1), rtol=1e-14)


def test_sa_outputs_nu_tilde_and_nut():
    from autoflowcfd.core.turbulence.sa import SAModel

    m = SAModel(4, 8, nu_ref=1.5e-5, viscosity_ratio=3.0)
    m.nu_tilde_field = np.arange(32, dtype=float).reshape(4, 8) * 1e-6
    out = turbulence_cell_means(m, 4, 1)
    assert set(out) == {"nu_tilde", "nut"}
    np.testing.assert_allclose(out["nu_tilde"], _real_mean(m.nu_tilde_field, 4, 1), rtol=1e-14)


def test_laminar_has_no_turbulence_outputs():
    assert turbulence_cell_means(None, 3, 1) == {}


def test_checkpoint_carries_model_means_not_state_slots(tmp_path):
    """写 checkpoint：`turb_cell_*` 等于模型场的单元平均；状态数组的 5/6 列（历史死槽位）被写成
    完全不同的值也不影响它。"""
    import h5py

    from autoflowcfd.cli.solve.checkpoint_io import write_checkpoint

    rng = np.random.default_rng(1)
    m = _sst(2, 0, 1, rng)
    U = np.ones((2, 8, 7))
    U[..., 5] = -123.0      # 死槽位：此前 VTK 的 k 读的就是它
    U[..., 6] = -456.0
    state = SimpleNamespace(U=U, Q=U.copy(), n_sps=8, n_vars=7)
    solver = SimpleNamespace(state=state, order=1, current_order=1, turb_model=m,
                             mesh=SimpleNamespace(n_prism_cells=2),
                             freestream={"rho_inf": 1.225, "vel_inf": 30.0, "p_inf": 101325.0},
                             time_integrator=SimpleNamespace(scheme=TimeIntegrationScheme.SSP_RK3))
    write_checkpoint(solver, str(tmp_path), 7, "volume.nas", order=1, turbulence_model="sst",
                     backend="cpu", quiet=True)
    with h5py.File(tmp_path / "checkpoints" / "checkpoint_iter_000007.h5", "r") as f:
        fields = {k: f["solution"][k][:] for k in f["solution"]}
    turb = turbulence_fields_from_checkpoint(fields)
    assert set(turb) == {"k", "omega", "nut"}
    assert all(f"{CHECKPOINT_PREFIX}{k}" in fields for k in turb)
    np.testing.assert_allclose(turb["k"], _real_mean(m.k_field, 2, 1), rtol=1e-14)
    np.testing.assert_allclose(turb["omega"], _real_mean(m.omega_field, 2, 1), rtol=1e-14)


@pytest.mark.parametrize("name,label", [("k", "TurbulentKineticEnergy"), ("nu_tilde", "SANuTilde")])
def test_vtk_writes_the_given_turbulence_fields(tmp_path, name, label):
    from autoflowcfd.core.backend import SolutionVector
    from autoflowcfd.grid.structures import BoundaryMap, CellArray, GridData, GridMetadata, NodeArray
    from autoflowcfd.postprocess.vtk_export import VTKExporter

    grid = GridData(
        nodes=NodeArray(x=np.array([0.0, 1.0, 0.0]), y=np.array([0.0, 0.0, 1.0]), z=np.zeros(3)),
        cells=CellArray(connectivity=np.array([[0, 1, 2]]), cell_type=np.array([0])),
        boundaries=BoundaryMap(groups={}, bc_types={}),
        metadata=GridMetadata(node_count=3, cell_count=1, boundary_groups=[], file_format="v24"))
    value = 0.123456789
    exporter = VTKExporter(grid, SolutionVector(), turbulence={name: np.array([value])})
    path = exporter.export(str(tmp_path / "t.vtk"), fields=["turbulence"])
    text = path.read_text()
    # 单元数据（1 个单元）与点数据（3 个点，由单元值平均到点）各写一次
    sections = text.split(f"SCALARS {label} double 1")[1:]
    assert len(sections) == 2
    for sec, n in zip(sections, (1, 3)):
        vals = sec.split("LOOKUP_TABLE default")[1].split()[:n]
        np.testing.assert_allclose([float(v) for v in vals], value, rtol=1e-6)
