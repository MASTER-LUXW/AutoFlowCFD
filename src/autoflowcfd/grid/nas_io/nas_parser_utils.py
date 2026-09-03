"""NAS 解析器工具函数。

提供解析 Nastran 格式文件用的辅助函数，包括浮点数解析与格式探测。
"""

import re


def parse_nastran_float(value_str: str) -> float:
    """解析 Nastran 格式的浮点数。

    处理 Nastran 的紧凑科学计数法，指数直接附着，
    没有 'e' 或 'E'。
    
    Examples:
        "5.635257-127" -> 5.635257e-127
        "-7.5-14" -> -7.5e-14
        "1.23+4" -> 1.23e+4
        "100.5" -> 100.5
    
    Args:
        value_str: 数字的字符串表示

    Returns:
        float: 解析后的浮点数值

    Raises:
        ValueError: 字符串无法解析时
    """
    if not value_str:
        raise ValueError("Empty string")

    value_str = value_str.strip()

    # 先尝试标准浮点数解析
    try:
        return float(value_str)
    except ValueError:
        pass

    # 处理 Nastran 紧凑科学计数法。
    # 模式：[符号]尾数[指数符号]指数数字，其中尾数可以是
    # "数字[.数字]"形式，也可以是以小数点开头的".数字"形式（两者都是
    # 合法的 Nastran 实数——Nastran 常用 "-.5-3" 表示 -0.0005；此前
    # 强制要求小数点前有 "\d+" 会拒绝这种写法）。
    # 示例：5.635257-127, -7.5-14, 1.23+4, -.5-3
    pattern = re.compile(r'^([+-]?(?:\d+\.?\d*|\.\d+))([+-]\d+)$')
    match = pattern.match(value_str)

    if match:
        mantissa = float(match.group(1))
        exponent = int(match.group(2))
        return mantissa * (10 ** exponent)

    # 仍然无法解析则报错
    raise ValueError(f"Cannot parse Nastran float: '{value_str}'")
