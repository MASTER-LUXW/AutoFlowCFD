"""ANSA .nas 网格文件解析入口。

本模块只做向后兼容的转出，新代码请直接从以下子模块导入：
    - autoflowcfd.grid.nas_io.parser_core
    - autoflowcfd.grid.nas_io.nas_parser_exceptions
    - autoflowcfd.grid.nas_io.nas_parser_utils
    - autoflowcfd.grid.nas_io.nas_parser_boundary
"""

# 向后兼容转出
from .nas_parser_exceptions import NASParserError, NASFormatError, NASParseError
from .parser_core import NASParser

__all__ = [
    'NASParser',
    'NASParserError',
    'NASFormatError',
    'NASParseError',
]
