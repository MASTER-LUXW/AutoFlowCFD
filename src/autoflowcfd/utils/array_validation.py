"""Array shape validation utilities for safe NumPy operations.

This module provides defensive programming tools to prevent broadcasting errors
and ensure array shape consistency in critical numerical computations.

Key Functions:
    - validate_broadcast_shapes: Check if arrays can be safely broadcast
    - safe_elementwise_multiply: Multiply with automatic shape validation
    - assert_matching_lengths: Assert multiple arrays have same first dimension
    - validate_face_indices: Sanitize and validate face index arrays

Example:
    >>> from autoflowcfd.utils.array_validation import safe_elementwise_multiply
    >>> result = safe_elementwise_multiply(a, b, context="Cd calculation")
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
    """Validate that two arrays can be safely broadcast together.
    
    Args:
        arr1: First array
        arr2: Second array
        operation: Description of the operation (for logging)
        context: Additional context information
        
    Returns:
        True if shapes are compatible, False otherwise
        
    Example:
        >>> a = np.random.rand(100, 3)
        >>> b = np.random.rand(100)
        >>> validate_broadcast_shapes(a[:, 0], b, "multiply", "force calc")
        True
    """
    try:
        # Attempt broadcast to check compatibility
        np.broadcast_arrays(arr1, arr2)
        return True
    except ValueError as e:
        logger.warning(
            f"[Shape Validation] {operation} failed in {context}: "
            f"arr1.shape={arr1.shape}, arr2.shape={arr2.shape}. "
            f"Error: {e}"
        )
        return False


def safe_elementwise_multiply(
    arr1: np.ndarray,
    arr2: np.ndarray,
    fallback_value: float = 0.0,
    context: str = ""
) -> np.ndarray:
    """Safely perform element-wise multiplication with shape validation.
    
    Args:
        arr1: First array
        arr2: Second array
        fallback_value: Value to return if shapes don't match
        context: Context description for logging
        
    Returns:
        Element-wise product or fallback value
        
    Example:
        >>> stress = np.random.rand(50, 3)
        >>> areas = np.random.rand(50)
        >>> force = safe_elementwise_multiply(stress[:, 0], areas, 
        ...                                   context="friction drag")
    """
    if not validate_broadcast_shapes(arr1, arr2, "multiply", context):
        logger.error(
            f"[Safe Multiply] Shape mismatch detected in {context}. "
            f"Returning fallback value."
        )
        return np.array(fallback_value)
    
    return arr1 * arr2


def assert_matching_lengths(
    *arrays: np.ndarray,
    context: str = "",
    tolerance: int = 0
) -> None:
    """Assert that multiple arrays have matching first dimension lengths.
    
    Args:
        *arrays: Variable number of arrays to check
        context: Description of where this check occurs
        tolerance: Allowed difference in lengths (default: 0)
        
    Raises:
        AssertionError: If lengths don't match within tolerance
        
    Example:
        >>> a = np.random.rand(100, 3)
        >>> b = np.random.rand(100)
        >>> assert_matching_lengths(a, b, context="aero coeffs")
        # No exception raised - lengths match
    """
    if not arrays:
        return
    
    lengths = [arr.shape[0] for arr in arrays]
    max_len = max(lengths)
    min_len = min(lengths)
    
    if max_len - min_len > tolerance:
        raise AssertionError(
            f"[Length Mismatch] In {context}: "
            f"Expected matching lengths but got {lengths}. "
            f"Difference: {max_len - min_len} (tolerance: {tolerance})"
        )


def validate_face_indices(
    face_indices: np.ndarray,
    total_faces: int,
    context: str = ""
) -> np.ndarray:
    """Validate and sanitize face indices.
    
    Removes negative indices and out-of-bounds indices to prevent
    indexing errors in downstream calculations.
    
    Args:
        face_indices: Array of face indices to validate
        total_faces: Total number of faces in mesh
        context: Description for error messages
        
    Returns:
        Validated and sanitized indices
        
    Example:
        >>> raw_indices = np.array([0, 5, -1, 1000, 10])
        >>> valid = validate_face_indices(raw_indices, 100, "body faces")
        >>> print(valid)  # [0, 5, 10] - removed -1 and 1000
    """
    if len(face_indices) == 0:
        return face_indices.astype(np.int64)
    
    # Remove negative indices
    valid_mask = face_indices >= 0
    
    # Remove out-of-bounds indices
    valid_mask &= face_indices < total_faces
    
    if not np.all(valid_mask):
        invalid_count = np.sum(~valid_mask)
        logger.warning(
            f"[Index Validation] In {context}: "
            f"Found {invalid_count} invalid face indices. "
            f"Total: {len(face_indices)}, Valid: {np.sum(valid_mask)}"
        )
        face_indices = face_indices[valid_mask]
    
    return face_indices.astype(np.int64)


def get_shape_summary(*arrays: np.ndarray, names: Optional[list] = None) -> str:
    """Generate a human-readable summary of array shapes for debugging.
    
    Args:
        *arrays: Arrays to summarize
        names: Optional list of names for each array
        
    Returns:
        Formatted string with shape information
        
    Example:
        >>> a = np.random.rand(100, 3)
        >>> b = np.random.rand(100)
        >>> print(get_shape_summary(a, b, names=["stress", "areas"]))
        stress: shape=(100, 3), dtype=float64
        areas: shape=(100,), dtype=float64
    """
    if names is None:
        names = [f"array_{i}" for i in range(len(arrays))]
    
    lines = []
    for name, arr in zip(names, arrays):
        lines.append(f"{name}: shape={arr.shape}, dtype={arr.dtype}")
    
    return "\n".join(lines)


# ============================================================================
# Debug mode assertions (only active when Python -O flag is NOT used)
# ============================================================================

if __debug__:
    # Development/debug mode: enable strict checking
    DEBUG_ARRAY_CHECKS = True
    logger.debug("[Array Validation] Debug mode enabled - strict checking active")
else:
    # Production mode: disable checks for performance
    DEBUG_ARRAY_CHECKS = False
    logger.info("[Array Validation] Production mode - shape checks disabled")
