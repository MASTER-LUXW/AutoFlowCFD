"""Standalone checkpoint test without full autoflowcfd dependencies."""

import sys
import numpy as np
from pathlib import Path

# Add src to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import h5py
    print("✓ h5py available")
except ImportError:
    print("✗ h5py not installed. Install with: pip install h5py")
    sys.exit(1)

# Import only the checkpoint module
from autoflowcfd.core.checkpoint import CheckpointManager


class MinimalConfig:
    """Minimal config for testing."""
    def __init__(self):
        self.mode = "steady"
        self.backend = "cpu"
        self.order = 2
        self.turbulence = "sst_kw"
        self.cfl_initial = 0.1
        self.cfl_max = 5.0
        self.output_dir = "test_checkpoints/"
        self.checkpoint_interval = 100


def test_basic_save_load():
    """Test basic save and load functionality."""
    
    print("\n" + "="*70)
    print("Test 1: Basic Save/Load")
    print("="*70)
    
    # Create config and manager
    config = MinimalConfig()
    manager = CheckpointManager(config, output_dir=config.output_dir)
    
    # Create test data
    n_cells = 100
    solution = np.random.rand(n_cells, 7).astype(np.float64)
    
    history = {
        'iterations': list(range(1, 51)),
        'residuals': {
            'continuity': [1e-2 * np.exp(-i/10) for i in range(50)],
        },
        'coefficients': {
            'Cd': [0.3 + 0.01*np.sin(i/5) for i in range(50)],
        },
        'cfl_history': [1.0] * 50,
    }
    
    # Save
    print("\nSaving checkpoint at iteration 50...")
    ckpt_path = manager.save(solution, history, iteration=50)
    
    if not ckpt_path:
        print("✗ FAILED to save checkpoint")
        return False
    
    print(f"✓ Saved: {ckpt_path}")
    
    # Load
    print("\nLoading checkpoint...")
    loaded_sol, loaded_hist, loaded_iter, metadata = manager.load(ckpt_path)
    
    # Verify
    print("\nVerifying data integrity...")
    assert np.allclose(solution, loaded_sol), "Solution mismatch!"
    assert loaded_iter == 50, f"Iteration mismatch: expected 50, got {loaded_iter}"
    assert len(loaded_hist['iterations']) == 50, "History length mismatch!"
    
    print("✓ All checks passed!")
    print(f"  - Solution shape: {loaded_sol.shape}")
    print(f"  - Iteration: {loaded_iter}")
    print(f"  - History entries: {len(loaded_hist['iterations'])}")
    print(f"  - Backend: {metadata['original_backend']}")
    
    return True


def test_multiple_checkpoints():
    """Test multiple checkpoints and cleanup."""
    
    print("\n" + "="*70)
    print("Test 2: Multiple Checkpoints & Cleanup")
    print("="*70)
    
    config = MinimalConfig()
    manager = CheckpointManager(config, output_dir=config.output_dir)
    
    # Create multiple checkpoints
    print("\nCreating 5 checkpoints...")
    for iteration in [100, 200, 300, 400, 500]:
        solution = np.random.rand(50, 7).astype(np.float64)
        history = {
            'iterations': list(range(1, iteration+1)),
            'residuals': {'continuity': [1e-2]*iteration},
            'coefficients': {'Cd': [0.3]*iteration},
            'cfl_history': [1.0]*iteration,
        }
        
        ckpt_path = manager.save(solution, history, iteration)
        if ckpt_path:
            print(f"  ✓ Iteration {iteration}: {Path(ckpt_path).name}")
    
    # List checkpoints
    print("\nListing all checkpoints...")
    checkpoints = manager.list_checkpoints()
    print(f"Found {len(checkpoints)} checkpoint(s)")
    
    # Get latest
    latest = manager.get_latest_checkpoint()
    if latest:
        print(f"Latest: {latest.name}")
    
    # Cleanup
    print("\nCleaning up (keeping last 2)...")
    deleted = manager.cleanup_old_checkpoints(keep_last=2)
    print(f"Deleted {deleted} checkpoint(s)")
    
    remaining = manager.list_checkpoints()
    print(f"Remaining: {len(remaining)} checkpoint(s)")
    for ckpt in remaining:
        print(f"  - {ckpt.name}")
    
    assert len(remaining) == 2, f"Expected 2 checkpoints, got {len(remaining)}"
    print("\n✓ Cleanup test passed!")
    
    return True


def test_config_hash():
    """Test configuration hash validation."""
    
    print("\n" + "="*70)
    print("Test 3: Configuration Hash Validation")
    print("="*70)
    
    config = MinimalConfig()
    manager = CheckpointManager(config, output_dir=config.output_dir)
    
    # Save with current config
    solution = np.random.rand(50, 7).astype(np.float64)
    history = {
        'iterations': [1],
        'residuals': {'continuity': [1e-2]},
        'coefficients': {'Cd': [0.3]},
        'cfl_history': [1.0],
    }
    
    ckpt_path = manager.save(solution, history, iteration=1)
    
    # Load and check metadata
    _, _, _, metadata = manager.load(ckpt_path)
    
    print(f"\nConfiguration hash: {metadata['config_hash'][:16]}...")
    print(f"Original backend: {metadata['original_backend']}")
    
    # Verify hash is present
    assert 'config_hash' in metadata, "Config hash missing!"
    assert len(metadata['config_hash']) == 64, "Invalid hash length!"
    
    print("✓ Config hash validation passed!")
    
    return True


def cleanup_test_files():
    """Clean up test files."""
    import shutil
    
    test_dir = Path("test_checkpoints/")
    if test_dir.exists():
        shutil.rmtree(test_dir)
        print(f"\n✓ Cleaned up test directory: {test_dir}")


if __name__ == "__main__":
    print("="*70)
    print("AutoFlowCFD Checkpoint Module Test Suite")
    print("="*70)
    
    try:
        # Run tests
        success = True
        success = success and test_basic_save_load()
        success = success and test_multiple_checkpoints()
        success = success and test_config_hash()
        
        # Summary
        print("\n" + "="*70)
        if success:
            print("✓✓✓ ALL TESTS PASSED ✓✓✓")
        else:
            print("✗✗✗ SOME TESTS FAILED ✗✗✗")
        print("="*70)
        
        # Cleanup
        cleanup_test_files()
        
    except Exception as e:
        print(f"\n✗ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to cleanup even on failure
        try:
            cleanup_test_files()
        except:
            pass
        
        sys.exit(1)
