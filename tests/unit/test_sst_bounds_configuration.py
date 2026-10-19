"""SST k 上界的设定（I-43，2026-10-19，`turbulence/sst/bounds.py::configure_sst_model`）。

此前来流速度为 0 时 k_max = 0.5*0^2 = 0，把 k 场整体截成 0（湍流被静默关掉）；模型构造时 k_max 默认 1e6 占位，
漏调设定函数时静默按占位值运行。
"""

import numpy as np
import pytest

from autoflowcfd.core.turbulence.sst import SSTModelFR
from autoflowcfd.core.turbulence.sst.bounds import clip_to_bounds, configure_sst_model


def _model():
    return SSTModelFR(2, 3, k_inf=1e-3, omega_inf=10.0)


@pytest.mark.parametrize("vel_inf", [0.0, -5.0, float("nan"), float("inf")])
def test_non_positive_freestream_velocity_is_refused(vel_inf):
    with pytest.raises(ValueError, match="正的来流速度"):
        configure_sst_model(_model(), vel_inf, "resolved")


def test_clipping_an_unconfigured_model_is_an_error():
    m = _model()
    assert m.k_max is None
    with pytest.raises(RuntimeError, match="configure_sst_model"):
        clip_to_bounds(m, np)


def test_configured_bound_is_the_freestream_kinetic_energy():
    m = _model()
    configure_sst_model(m, 30.0, "resolved")
    assert m.k_max == 0.5 * 30.0 ** 2
    m.k_field = np.full((2, 3), 1e4)
    clip_to_bounds(m, np)
    assert np.all(m.k_field == m.k_max)
