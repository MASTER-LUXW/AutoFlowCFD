"""续算重建：决定算例的元数据缺失一律报错；checkpoint 缺湍流场时须显式 --reset-turbulence（I-173）。

此前来流三要素之外的 mu_molecular（雷诺数）、Tu、VR、人工粘性开关 / 强度、近壁处理、湍流模型、阶数、时间格式、
低马赫预处理缺失时静默取 1.8e-5 / 0.01 / 5.0 / 关闭 / 1.0 / resolved / sst / 2 / ssp_rk3 / 开启；湍流场或 nu_t 缺失时
只打印警告、从来流初值继续算。
"""

import click
import numpy as np
import pytest

from autoflowcfd.cli.solve.checkpoint_io.rebuild import (
    PHYSICS_KEYS, physics_from_metadata, required_metadata, resolve_resume_low_mach_precond,
)
from autoflowcfd.cli.solve.checkpoint_io.restore import restore_turbulence_from_fields
from autoflowcfd.core.time_integration import TimeIntegrationScheme
from autoflowcfd.core.utils.order_continuation.turbulence_reset import check_resumed_turbulence

_FULL = {"rho_inf": 1.2, "vel_inf": 10.0, "p_inf": 1e5, "aoa_deg": 0.0, "aos_deg": 0.0, "mu_molecular": 2.5e-4,
         "turbulence_intensity": 0.02, "viscosity_ratio": 3.0, "artificial_viscosity_enabled": False,
         "artificial_viscosity_alpha": 1.0, "wall_treatment": "resolved"}


@pytest.mark.parametrize("key", PHYSICS_KEYS)
def test_every_physics_key_is_required(key):
    meta = dict(_FULL)
    del meta[key]
    with pytest.raises(click.ClickException, match=key):
        physics_from_metadata(meta)


def test_custom_viscosity_is_restored_exactly():
    assert physics_from_metadata(_FULL)["mu_molecular"] == 2.5e-4


def test_low_mach_precond_and_rebuild_keys_are_required():
    with pytest.raises(click.ClickException, match="low_mach_precond"):
        resolve_resume_low_mach_precond(None, {})
    assert resolve_resume_low_mach_precond(False, {}) is False
    with pytest.raises(click.ClickException, match="turbulence_model"):
        required_metadata({"input_file": "x.pkl", "order": 1}, ("input_file", "order", "turbulence_model"))


def _sst_solver():
    from autoflowcfd.core.fr_solver import FRSolver
    from autoflowcfd.core.freestream_defaults import MU_MOLECULAR, P_INF, RHO_INF
    from tests.unit.test_convergence_history import _BC
    from tests.validation._channel_mesh import build_channel_mesh_prism

    mesh = build_channel_mesh_prism(1, 3, 3, 2, 0.4, 0.1, 0.08)
    mesh.boundary_bc_types = dict(_BC)
    s = FRSolver(mesh=mesh, order=1, turb_model_name="SST", time_scheme=TimeIntegrationScheme.SSP_RK3, vel_inf=10.0,
                 rho_inf=RHO_INF, p_inf=P_INF, mu_molecular=MU_MOLECULAR)
    s._resumed_from_checkpoint = True
    return s


def test_missing_turbulence_fields_block_resume_unless_reset():
    s = _sst_solver()
    assert restore_turbulence_from_fields(s, {}, {}) is False
    s._reset_turbulence_on_resume = False
    with pytest.raises(RuntimeError, match="缺少湍流场.*--reset-turbulence"):
        check_resumed_turbulence(s)
    s.turb_model.transported_fields()[0][:] = 123.0             # 弄脏，确认重置的是来流值
    s._reset_turbulence_on_resume = True
    check_resumed_turbulence(s)
    for field, value in zip(s.turb_model.transported_fields(), s.turb_model.freestream_values()):
        np.testing.assert_allclose(field, value)


def test_missing_nu_t_alone_restores_the_transported_fields_but_still_needs_reset_on_resume():
    s = _sst_solver()
    fields = {name: np.full_like(f, 0.5) for name, f in zip(s.turb_model.TRANSPORTED_FIELDS,
                                                            s.turb_model.transported_fields())}
    assert restore_turbulence_from_fields(s, fields, {}) is True
    np.testing.assert_allclose(s.turb_model.transported_fields()[0], 0.5)
    assert s._missing_turbulence_fields == ("nu_t",)
    s._reset_turbulence_on_resume = False
    with pytest.raises(RuntimeError, match="nu_t"):
        check_resumed_turbulence(s)
