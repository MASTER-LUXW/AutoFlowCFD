"""Command-line interface module for AutoFlowCFD.

This module provides Click-based CLI commands for running simulations,
post-processing results, and utility functions.
"""

from .main import cli

__all__ = ["cli"]
