"""
AutoFlowCFD Core - Turbulence Module

湍流模型模块，包含 SST k-ω、DDES、WMLES、SGS 等模型及输运方程。
"""

from autoflowcfd.core.turbulence.sst import SSTModelFR
from autoflowcfd.core.turbulence.des import DDESModel, IDDESModel
from autoflowcfd.core.turbulence.wmles import WMLESModel
from autoflowcfd.core.turbulence.sgs import WALEModel, SmagorinskyModel
from autoflowcfd.core.turbulence.transport import compute_turbulence_transport_residual

__all__ = [
    'SSTModelFR',
    'DDESModel',
    'IDDESModel',
    'WMLESModel',
    'WALEModel',
    'SmagorinskyModel',
    'compute_turbulence_transport_residual',
]
