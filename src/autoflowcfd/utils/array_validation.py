"""数组形状验证工具，保障 NumPy 运算安全。

本模块提供防御性编程工具，用于防止广播错误并确保关键数值计算中
数组形状的一致性。

核心函数:
    - validate_broadcast_shapes: 检查数组是否可安全广播
    - safe_elementwise_multiply: 带自动形状验证的逐元素乘法
    - assert_matching_lengths: 断言多个数组第一维长度一致
    - validate_face_indices: 校验并清理面索引数组

示例:
    >>> from autoflowcfd.utils.array_validation import safe_elementwise_multiply
    >>> result = safe_elementwise_multiply(a, b, context="Cd 计算")
"""

import numpy as np
from typing import Tuple, Optional, Union
from loguru import logger


def validate_broadcast_shapes(
    arr1: np.ndarray,
    arr2: np.ndarray,
    operation: str = "multiplication",
    context: str = ""
) -> bool:
    """验证两个数组是否可安全广播。
    
    Args:
        arr1: 第一个数组
        arr2: 第二个数组
        operation: 运算描述（用于日志记录）
        context: 附加上下文信息
        
    Returns:
        形状兼容返回 True，否则返回 False
        
    示例:
        >>> a = np.random.rand(100, 3)
        >>> b = np.random.rand(100)
        >>> validate_broadcast_shapes(a[:, 0], b, "multiply", "力计算")
        True
    """
    try:
        # 尝试广播以检查兼容性
        np.broadcast_arrays(arr1, arr2)
        return True
    except ValueError as e:
        logger.warning(
            f"[形状验证] {operation} 在 {context} 中失败: "
            f"arr1.shape={arr1.shape}, arr2.shape={arr2.shape}. "
            f"错误: {e}"
        )
        return False


def safe_elementwise_multiply(
    arr1: np.ndarray,
    arr2: np.ndarray,
    fallback_value: float = 0.0,
    context: str = ""
) -> np.ndarray:
    """带形状验证的安全逐元素乘法。
    
    Args:
        arr1: 第一个数组
        arr2: 第二个数组
        fallback_value: 形状不匹配时的回退值
        context: 日志记录的上下文描述
        
    Returns:
        逐元素乘积或回退值
        
    示例:
        >>> stress = np.random.rand(50, 3)
        >>> areas = np.random.rand(50)
        >>> force = safe_elementwise_multiply(stress[:, 0], areas, 
        ...                                   context="摩擦阻力")
    """
    if not validate_broadcast_shapes(arr1, arr2, "multiply", context):
        logger.error(
            f"[安全乘法] 在 {context} 中检测到形状不匹配。"
            f"返回回退值。"
        )
        return np.array(fallback_value)
    
    return arr1 * arr2


def assert_matching_lengths(
    *arrays: np.ndarray,
    context: str = "",
    tolerance: int = 0
) -> None:
    """断言多个数组的第一维长度一致。
    
    Args:
        *arrays: 待检查的数组（可变参数）
        context: 检查发生位置的描述
        tolerance: 允许的长度差异（默认: 0）
        
    Raises:
        AssertionError: 长度在容差范围内不一致时抛出
        
    示例:
        >>> a = np.random.rand(100, 3)
        >>> b = np.random.rand(100)
        >>> assert_matching_lengths(a, b, context="气动系数")
        # 不抛异常——长度一致
    """
    if not arrays:
        return
    
    lengths = [arr.shape[0] for arr in arrays]
    max_len = max(lengths)
    min_len = min(lengths)
    
    if max_len - min_len > tolerance:
        raise AssertionError(
            f"[长度不匹配] 在 {context} 中: "
            f"期望长度一致但实际为 {lengths}。"
            f"差异: {max_len - min_len}（容差: {tolerance}）"
        )


def validate_face_indices(
    face_indices: np.ndarray,
    total_faces: int,
    context: str = ""
) -> np.ndarray:
    """验证并清理面索引。
    
    移除负索引和越界索引，防止下游计算中的索引错误。
    
    Args:
        face_indices: 待验证的面索引数组
        total_faces: 网格中的总面数
        context: 错误消息的描述信息
        
    Returns:
        已验证和清理的索引数组
        
    示例:
        >>> raw_indices = np.array([0, 5, -1, 1000, 10])
        >>> valid = validate_face_indices(raw_indices, 100, "body faces")
        >>> print(valid)  # [0, 5, 10] - 移除了 -1 和 1000
    """
    if len(face_indices) == 0:
        return face_indices.astype(np.int64)
    
    # 移除负索引
    valid_mask = face_indices >= 0
    
    # 移除越界索引
    valid_mask &= face_indices < total_faces
    
    if not np.all(valid_mask):
        invalid_count = np.sum(~valid_mask)
        logger.warning(
            f"[索引验证] 在 {context} 中: "
            f"发现 {invalid_count} 个无效面索引。"
            f"总数: {len(face_indices)}, 有效: {np.sum(valid_mask)}"
        )
        face_indices = face_indices[valid_mask]
    
    return face_indices.astype(np.int64)


def get_shape_summary(*arrays: np.ndarray, names: Optional[list] = None) -> str:
    """生成数组形状的可读摘要，用于调试。
    
    Args:
        *arrays: 待摘要的数组
        names: 每个数组的可选名称列表
        
    Returns:
        包含形状信息的格式化字符串
        
    示例:
        >>> a = np.random.rand(100, 3)
        >>> b = np.random.rand(100)
        >>> print(get_shape_summary(a, b, names=["应力", "面积"]))
        应力: shape=(100, 3), dtype=float64
        面积: shape=(100,), dtype=float64
    """
    if names is None:
        names = [f"array_{i}" for i in range(len(arrays))]
    
    lines = []
    for name, arr in zip(names, arrays):
        lines.append(f"{name}: shape={arr.shape}, dtype={arr.dtype}")
    
    return "\n".join(lines)


# ============================================================================
# 调试模式断言（仅在未使用 Python -O 标志时生效）
# ============================================================================

if __debug__:
    # 开发/调试模式：启用严格检查
    DEBUG_ARRAY_CHECKS = True
    logger.debug("[数组验证] 调试模式已启用 - 严格检查激活")
else:
    # 生产模式：禁用检查以提升性能
    DEBUG_ARRAY_CHECKS = False
    logger.info("[数组验证] 生产模式 - 形状检查已禁用")
