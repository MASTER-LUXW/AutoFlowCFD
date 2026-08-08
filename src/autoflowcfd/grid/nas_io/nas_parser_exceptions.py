"""NAS 解析器异常类。

定义 NAS 文件解析过程中用到的自定义异常。
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
