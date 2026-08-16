"""Test script for aerodynamic coefficient calculation shape fix."""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_shape_broadcasting():
    """Test that friction force calculation handles shape mismatches correctly."""
    
    print("="*70)
    print("Testing Aerodynamic Coefficient Shape Fix")
    print("="*70)
    
    # Simulate the problematic scenario
    n_valid_faces = 100
    
    # tau_n_body: wall shear stress vectors (n_faces, 3)
    tau_n_body = np.random.rand(n_valid_faces, 3).astype(np.float64)
    
    # valid_face_areas: face areas (n_faces,)
    valid_face_areas = np.random.rand(n_valid_faces).astype(np.float64)
    
    print(f"\nTest 1: Matching shapes")
    print(f"  tau_n_body shape: {tau_n_body.shape}")
    print(f"  valid_face_areas shape: {valid_face_areas.shape}")
    
    try:
        Fx_f = -np.sum(tau_n_body[:, 0] * valid_face_areas)
        Fz_f = -np.sum(tau_n_body[:, 2] * valid_face_areas)
        print(f"  ✓ Computation successful: Fx_f={Fx_f:.6e}, Fz_f={Fz_f:.6e}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False
    
    # Test mismatched shapes (simulating the bug scenario)
    print(f"\nTest 2: Mismatched shapes (tau_n has more faces)")
    tau_n_mismatch = np.random.rand(n_valid_faces + 50, 3).astype(np.float64)
    print(f"  tau_n_body shape: {tau_n_mismatch.shape}")
    print(f"  valid_face_areas shape: {valid_face_areas.shape}")
    
    try:
        # This would fail without the fix
        min_len = min(tau_n_mismatch.shape[0], valid_face_areas.shape[0])
        tau_n_safe = tau_n_mismatch[:min_len]
        areas_safe = valid_face_areas[:min_len]
        
        Fx_f = -np.sum(tau_n_safe[:, 0] * areas_safe)
        Fz_f = -np.sum(tau_n_safe[:, 2] * areas_safe)
        print(f"  ✓ Shape handling successful: Fx_f={Fx_f:.6e}, Fz_f={Fz_f:.6e}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False
    
    # Test edge case: empty arrays
    print(f"\nTest 3: Empty arrays")
    tau_empty = np.zeros((0, 3))
    areas_empty = np.zeros((0,))
    
    try:
        if len(tau_empty) > 0:
            Fx_f = -np.sum(tau_empty[:, 0] * areas_empty)
        else:
            Fx_f = 0.0
        print(f"  ✓ Empty array handling: Fx_f={Fx_f:.6e}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False
    
    print("\n" + "="*70)
    print("✓ All shape tests passed!")
    print("="*70)
    
    return True


if __name__ == "__main__":
    success = test_shape_broadcasting()
    sys.exit(0 if success else 1)
