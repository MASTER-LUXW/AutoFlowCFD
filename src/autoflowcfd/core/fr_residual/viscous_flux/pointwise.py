"""AutoFlowCFD V2.0 - 逐点量：温度（粘性物理通量的逐点核在 `core/fr_operators/flux_kernels/viscous.py`）"""


import numpy as np

from autoflowcfd.core.thermo import R_AIR


def compute_temperature(Q: np.ndarray) -> np.ndarray:
    """T = p/(rho*R)。"""
    rho = np.maximum(Q[..., 0], 1e-10)
    return Q[..., 4] / (rho * R_AIR)
