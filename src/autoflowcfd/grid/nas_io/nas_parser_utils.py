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
        value_str: String representation of the number
        
    Returns:
        float: Parsed floating point value
        
    Raises:
        ValueError: If the string cannot be parsed
    """
    if not value_str:
        raise ValueError("Empty string")
    
    value_str = value_str.strip()
    
    # 先尝试标准浮点数解析
    try:
        return float(value_str)
    except ValueError:
        pass
    
    # 处理 Nastran compact scientific notation
    # Pattern: [sign]mantissa[exponent_sign]exponent_digits, where mantissa
    # is either "digits[.digits]" or a leading-dot form ".digits" (both are
    # legal Nastran reals, e.g. Nastran commonly emits "-.5-3" for -0.0005;
    # the mandatory "\d+" before the dot previously rejected this form).
    # Example: 5.635257-127, -7.5-14, 1.23+4, -.5-3
    pattern = re.compile(r'^([+-]?(?:\d+\.?\d*|\.\d+))([+-]\d+)$')
    match = pattern.match(value_str)
    
    if match:
        mantissa = float(match.group(1))
        exponent = int(match.group(2))
        return mantissa * (10 ** exponent)
    
    # If still can't parse, raise error
    raise ValueError(f"Cannot parse Nastran float: '{value_str}'")
