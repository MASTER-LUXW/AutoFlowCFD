"""NAS file parser for ANSA mesh files.

This module provides backward compatibility by re-exporting from submodules.
For new code, import directly from:
    - autoflowcfd.grid.parser_core
    - autoflowcfd.grid.nas_parser_exceptions
    - autoflowcfd.grid.nas_parser_utils
    - autoflowcfd.grid.nas_parser_boundary
"""

# Re-export from submodules for backward compatibility
from .nas_parser_exceptions import NASParserError, NASFormatError, NASParseError
from .parser_core import NASParser

__all__ = [
    'NASParser',
    'NASParserError',
    'NASFormatError',
    'NASParseError',
]
