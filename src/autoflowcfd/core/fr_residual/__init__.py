"""
AutoFlowCFD Core - FR Residual Module

FR 残差计算模块，包含无粘残差、粘性残差和粘性通量计算。
"""

from autoflowcfd.core.fr_residual.inviscid import (
    compute_inviscid_residual_fr,
    conserved_to_primitive,
    DefaultGhostProvider,
)
from autoflowcfd.core.fr_residual.viscous import compute_viscous_residual_fr
from autoflowcfd.core.fr_residual.viscous_flux import viscous_physical_flux

__all__ = [
    'compute_inviscid_residual_fr',
    'conserved_to_primitive',
    'DefaultGhostProvider',
    'compute_viscous_residual_fr',
    'viscous_physical_flux',
]
