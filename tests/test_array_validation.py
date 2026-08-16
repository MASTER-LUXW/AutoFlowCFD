"""Test array validation utilities."""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from autoflowcfd.utils.array_validation import (
    validate_broadcast_shapes,
    safe_elementwise_multiply,
    assert_matching_lengths,
    validate_face_indices,
    get_shape_summary,
)


def test_validate_broadcast_shapes():
    """Test broadcast shape validation."""
    
    print("="*70)
    print("Test 1: validate_broadcast_shapes")
    print("="*70)
    
    # Compatible shapes
    a = np.random.rand(100, 3)
    b = np.random.rand(100)
    
    result = validate_broadcast_shapes(a[:, 0], b, "test", "compatible")
    print(f"✓ Compatible shapes (100,) and (100,): {result}")
    assert result == True
    
    # Incompatible shapes
    c = np.random.rand(50)
    result = validate_broadcast_shapes(a[:, 0], c, "test", "incompatible")
    print(f"✓ Incompatible shapes (100,) and (50,): {result}")
    assert result == False
    
    print()
    return True


def test_safe_elementwise_multiply():
    """Test safe element-wise multiplication."""
    
    print("="*70)
    print("Test 2: safe_elementwise_multiply")
    print("="*70)
    
    # Matching shapes
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    
    result = safe_elementwise_multiply(a, b, context="matching test")
    expected = np.array([4.0, 10.0, 18.0])
    print(f"✓ Matching shapes: {result}")
    assert np.allclose(result, expected)
    
    # Mismatched shapes
    c = np.array([1.0, 2.0])
    result = safe_elementwise_multiply(a, c, fallback_value=0.0, context="mismatch test")
    print(f"✓ Mismatched shapes returned fallback: {result}")
    assert result == 0.0
    
    print()
    return True


def test_assert_matching_lengths():
    """Test length assertion."""
    
    print("="*70)
    print("Test 3: assert_matching_lengths")
    print("="*70)
    
    # Matching lengths
    a = np.random.rand(100, 3)
    b = np.random.rand(100)
    c = np.random.rand(100, 5)
    
    try:
        assert_matching_lengths(a, b, c, context="matching test")
        print("✓ Matching lengths passed assertion")
    except AssertionError as e:
        print(f"✗ Unexpected failure: {e}")
        return False
    
    # Mismatched lengths
    d = np.random.rand(50)
    
    try:
        assert_matching_lengths(a, b, d, context="mismatch test")
        print("✗ Should have raised AssertionError")
        return False
    except AssertionError as e:
        print(f"✓ Mismatched lengths correctly raised AssertionError")
        print(f"  Message: {str(e)[:80]}...")
    
    print()
    return True


def test_validate_face_indices():
    """Test face index validation."""
    
    print("="*70)
    print("Test 4: validate_face_indices")
    print("="*70)
    
    # Valid indices
    indices = np.array([0, 5, 10, 15, 20])
    validated = validate_face_indices(indices, 100, "valid test")
    print(f"✓ Valid indices preserved: {validated}")
    assert len(validated) == 5
    
    # Mixed valid/invalid
    indices_mixed = np.array([0, 5, -1, 1000, 10])
    validated = validate_face_indices(indices_mixed, 100, "mixed test")
    print(f"✓ Invalid indices removed: {validated}")
    assert len(validated) == 3  # Only 0, 5, 10 remain
    assert -1 not in validated
    assert 1000 not in validated
    
    # Empty array
    empty = np.array([], dtype=np.int64)
    validated = validate_face_indices(empty, 100, "empty test")
    print(f"✓ Empty array handled: {validated}")
    assert len(validated) == 0
    
    print()
    return True


def test_get_shape_summary():
    """Test shape summary generation."""
    
    print("="*70)
    print("Test 5: get_shape_summary")
    print("="*70)
    
    a = np.random.rand(100, 3)
    b = np.random.rand(100)
    c = np.random.rand(50, 5, 2)
    
    summary = get_shape_summary(a, b, c, names=["stress", "areas", "tensor"])
    print(summary)
    
    assert "stress: shape=(100, 3)" in summary
    assert "areas: shape=(100,)" in summary
    assert "tensor: shape=(50, 5, 2)" in summary
    
    print("✓ Shape summary generated correctly")
    print()
    return True


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Array Validation Utilities Test Suite")
    print("="*70 + "\n")
    
    tests = [
        test_validate_broadcast_shapes,
        test_safe_elementwise_multiply,
        test_assert_matching_lengths,
        test_validate_face_indices,
        test_get_shape_summary,
    ]
    
    all_passed = True
    for test_func in tests:
        try:
            if not test_func():
                all_passed = False
        except Exception as e:
            print(f"✗ Test {test_func.__name__} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    print("="*70)
    if all_passed:
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
    else:
        print("✗✗✗ SOME TESTS FAILED ✗✗✗")
    print("="*70)
    
    sys.exit(0 if all_passed else 1)
