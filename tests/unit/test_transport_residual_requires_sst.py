"""k/ln(omega) 输运残差只对 SST 族模型定义（I-62，2026-10-19，`turbulence/transport/residual.py`）。

此前非 SST 模型（或没有湍流模型）调用时静默返回全零输运——湍流场只剩逐点源项弛豫，不报错。SST 的 k/ln(omega)
编排同时改为经 `CpuScalarTransport` 原语调用离散（与 SA-neg 同一套参数拼装），比特级对照与改动前逐位相同。
"""

from types import SimpleNamespace

import numpy as np
import pytest

from autoflowcfd.core.turbulence.sa import SAModel
from autoflowcfd.core.turbulence.transport import compute_turbulence_transport_residual


@pytest.mark.parametrize("model", [None, SAModel(1, 1, nu_ref=1.5e-5, viscosity_ratio=3.0)])
def test_non_sst_model_is_refused(model):
    solver = SimpleNamespace(turb_model=model, state=SimpleNamespace(U=np.zeros((1, 1, 5))))
    with pytest.raises(TypeError, match="SST"):
        compute_turbulence_transport_residual(solver)
