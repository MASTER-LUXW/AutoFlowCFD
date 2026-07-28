"""NAS parser exception classes.

Defines custom exceptions for NAS file parsing errors.
"""


class NASParserError(Exception):
    """NAS解析器基础异常类"""
    pass


class NASFormatError(NASParserError):
    """NAS文件格式错误"""
    pass


class NASParseError(NASParserError):
    """NAS解析过程错误"""
    pass
