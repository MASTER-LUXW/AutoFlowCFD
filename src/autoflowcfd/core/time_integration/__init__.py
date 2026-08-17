"""
AutoFlowCFD Core - Time Integration Module

时间积分模块，包含 SSP-RK、DUAL_TIME、IMEX 等时间推进方案。
"""

from autoflowcfd.core.time_integration.base import TimeIntegrator, TimeIntegrationScheme

__all__ = ['TimeIntegrator', 'TimeIntegrationScheme']
