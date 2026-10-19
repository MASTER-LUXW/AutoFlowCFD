"""SST 场更新的调用顺序与非有限值的如实报告（I-45 / I-48，2026-10-19，`turbulence/sst/update.py`、`turbulence/nonfinite.py`）。

此前源项未求值（缺 `_last_beta_blend`）时"防御性回退"到 beta2，掩盖调用顺序错误；非有限增量静默归零。
"""

import numpy as np
import pytest
from loguru import logger

from autoflowcfd.core.turbulence.sst import SSTModelFR
from autoflowcfd.core.turbulence.sst.bounds import configure_sst_model
from autoflowcfd.core.turbulence.sst.update import advance_k_log_omega


def _model():
    m = SSTModelFR(2, 3, k_inf=1e-3, omega_inf=10.0)
    configure_sst_model(m, 30.0, "resolved")
    return m


def test_update_before_source_evaluation_is_an_error():
    m = _model()
    zeros = np.zeros((2, 3))
    with pytest.raises(RuntimeError, match="compute_source_terms"):
        advance_k_log_omega(m, 1e-4, zeros, zeros, None, None, np)


def test_nonfinite_increments_are_counted_and_reported():
    m = _model()
    m._last_beta_blend = np.full((2, 3), m.beta1)
    k0, w0 = m.k_field.copy(), m.omega_field.copy()
    Sk = np.full((2, 3), 1.0)
    Sk[0, 1] = np.inf
    Sw = np.zeros((2, 3))
    Sw[1, 2] = np.nan
    messages = []
    sink = logger.add(lambda msg: messages.append(str(msg)), level="WARNING")
    try:
        advance_k_log_omega(m, 1e-4, Sk, Sw, None, None, np)
    finally:
        logger.remove(sink)
    assert m.nonfinite_count == 2
    assert any("非有限" in msg for msg in messages)
    assert m.k_field[0, 1] == k0[0, 1] and m.omega_field[1, 2] == pytest.approx(w0[1, 2])
    assert m.k_field[0, 0] > k0[0, 0]                        # 其余解点照常更新


def test_nonfinite_field_recovery_and_sa_source_are_reported():
    """场恢复（SST k/omega 场、SA nu_tilde 场）与 SA 源项的非有限值同样计数（I-48）。"""
    from autoflowcfd.core.turbulence.sa import SAModel
    from autoflowcfd.core.turbulence.sst.bounds import clip_to_bounds

    m = _model()
    m.k_field[0, 0] = np.nan
    m.omega_field[1, 1] = np.inf
    clip_to_bounds(m, np)
    assert m.nonfinite_count == 2 and m.k_field[0, 0] == 0.0 and m.omega_field[1, 1] == m.omega_inf

    sa = SAModel(1, 2, nu_ref=1.5e-5, viscosity_ratio=3.0)
    sa.nu_tilde_field[0, 1] = np.nan
    sa.apply_positivity_limiter()
    assert sa.nonfinite_count == 1 and sa.nu_tilde_field[0, 1] == sa.nu_tilde_inf
