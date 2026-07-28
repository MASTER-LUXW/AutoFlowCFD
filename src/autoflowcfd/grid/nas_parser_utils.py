"""NAS parser utility functions.

Provides helper functions for parsing Nastran format files, including
floating point number parsing and format detection utilities.
"""

import re


def parse_nastran_float(value_str: str) -> float:
    """解析Nastran格式的浮点数
    
    Handles Nastran's compact scientific notation where the exponent
    is directly attached without 'e' or 'E'.
    
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
    
    # Try standard float parsing first
    try:
        return float(value_str)
    except ValueError:
        pass
    
    # Handle Nastran compact scientific notation
    # Pattern: [sign]digits.digits[exponent_sign]exponent_digits
    # Example: 5.635257-127, -7.5-14, 1.23+4
    pattern = re.compile(r'^([+-]?\d+\.?\d*)([+-]\d+)$')
    match = pattern.match(value_str)
    
    if match:
        mantissa = float(match.group(1))
        exponent = int(match.group(2))
        return mantissa * (10 ** exponent)
    
    # If still can't parse, raise error
    raise ValueError(f"Cannot parse Nastran float: '{value_str}'")
