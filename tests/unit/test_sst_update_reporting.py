"""SST 场更新的调用顺序与非有限增量的如实报告（I-45，2026-10-19，`turbulence/sst/update.py`）。

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
    assert m.nonfinite_increment_count == 2
    assert any("非有限" in msg for msg in messages)
    assert m.k_field[0, 1] == k0[0, 1] and m.omega_field[1, 2] == pytest.approx(w0[1, 2])
    assert m.k_field[0, 0] > k0[0, 0]                        # 其余解点照常更新
