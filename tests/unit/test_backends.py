"""Unit tests for solver backends."""

import unittest
import numpy as np
from autoflowcfd.core import create_backend, get_available_backends


class TestBackendFactory(unittest.TestCase):
    """Test cases for backend factory function."""
    
    def test_get_available_backends(self):
        """Test backend availability detection."""
        backends = get_available_backends()
        
        # CPU should always be available
        self.assertIn('cpu', backends)
        self.assertTrue(backends['cpu'])
        
        # GPU may or may not be available
        self.assertIn('gpu', backends)
    
    def test_create_cpu_backend(self):
        """Test CPU backend creation."""
        backend = create_backend("cpu", n_threads=2)
        
        self.assertEqual(backend.backend_type, "cpu")
        self.assertTrue(backend.available)
    
    def test_create_auto_backend(self):
        """Test auto backend selection."""
        backend = create_backend("auto")
        
        # Should return either CPU or GPU backend
        self.assertIn(backend.backend_type, ["cpu", "gpu"])
        self.assertTrue(backend.available)
    
    def test_invalid_backend_type(self):
        """Test that invalid backend type raises ValueError."""
        with self.assertRaises(ValueError):
            create_backend("invalid_type")  # type: ignore
    
    def test_gpu_backend_not_available(self):
        """Test GPU backend when CUDA is not available."""
        backends = get_available_backends()
        
        if not backends['gpu']:
            with self.assertRaises(RuntimeError):
                create_backend("gpu")


class TestNumbaBackend(unittest.TestCase):
    """Test cases for Numba CPU backend."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.backend = create_backend("cpu", n_threads=2)
        self.n_cells = 100
        self.n_nodes = 50
    
    def test_initialize(self):
        """Test backend initialization."""
        self.backend.initialize(self.n_cells, self.n_nodes)
        
        self.assertEqual(self.backend.n_cells, self.n_cells)
        self.assertEqual(self.backend.n_nodes, self.n_nodes)
    
    def test_compute_flux(self):
        """Test flux computation."""
        self.backend.initialize(self.n_cells, self.n_nodes)
        
        solution = np.random.rand(self.n_cells, 5)
        connectivity = np.random.randint(0, self.n_cells, (self.n_cells, 2))
        normals = np.random.rand(self.n_cells, 3)
        
        flux = self.backend.compute_flux(solution, connectivity, normals)
        
        # Check output shape
        self.assertEqual(flux.shape[0], self.n_cells)
        self.assertEqual(flux.shape[1], 5)
        
        # Check values are finite
        self.assertTrue(np.all(np.isfinite(flux)))
    
    def test_update_solution(self):
        """Test solution update."""
        self.backend.initialize(self.n_cells, self.n_nodes)
        
        solution = np.random.rand(self.n_cells, 5)
        residuals = np.random.rand(self.n_cells, 5) * 0.01
        
        updated = self.backend.update_solution(solution, residuals, dt=1e-5, cfl=1.0)
        
        # Check shape preserved
        self.assertEqual(updated.shape, solution.shape)
        
        # Check values changed
        self.assertFalse(np.array_equal(updated, solution))
    
    def test_get_device_info(self):
        """Test device information query."""
        info = self.backend.get_device_info()
        
        self.assertIn('backend', info)
        self.assertEqual(info['backend'], 'Numba CPU')


if __name__ == '__main__':
    unittest.main()
