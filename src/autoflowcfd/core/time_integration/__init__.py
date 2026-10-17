"""
AutoFlowCFD Core - Time Integration Module

时间积分模块，包含 SSP-RK、DUAL_TIME、NEWTON_KRYLOV 时间推进方案与时间精确的显式推进（time_accurate.py）。
"""

from autoflowcfd.core.time_integration.base import TimeIntegrator, TimeIntegrationScheme

__all__ = ['TimeIntegrator', 'TimeIntegrationScheme']
