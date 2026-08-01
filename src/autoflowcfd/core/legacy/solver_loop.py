"""Steady-state solver main loop.

This module implements the main iteration loop for steady-state simulations,
including solution update, convergence monitoring, and result collection.

Key Components:
    - SteadySolverLoop: Main solver iteration controller

NOT CURRENTLY USED: FRSolver.solve() (solver_steady.py) has its own inline
pseudo-time iteration loop and never constructs or calls SteadySolverLoop.
"""

import numpy as np
from typing import Optional, List
import time
from loguru import logger

from ..fvm_core import FVMResidualComputer
from ...config.solver_config import SteadyConfig
from .convergence import ConvergenceMonitor
from ..time_integration import TimeIntegrator, TimeIntegrationScheme


class SteadySolverLoop:
    """Main solver iteration loop for steady-state simulations."""
    
    def __init__(self, config: SteadyConfig, 
                 residual_computer: FVMResidualComputer,
                 convergence_monitor: ConvergenceMonitor,
                 time_integrator: TimeIntegrator):
        self.config = config
        self.residual_computer = residual_computer
        self.convergence_monitor = convergence_monitor
        self.time_integrator = time_integrator
        
        # History tracking
        self.cd_history: List[float] = []
        self.cl_history: List[float] = []
        self.residuals_history: List[float] = []
    
    def run(self, solution: np.ndarray, grid_data, 
            get_cell_volumes_func, apply_bc_func,
            compute_coeffs_func, identify_body_faces_func,
            compute_ref_area_func, apply_constraints_func,
            bc_handler=None,  # Add bc_handler parameter for ramp mechanism
            max_iter: Optional[int] = None):
        """Execute solver iteration loop.
        
        Args:
            solution: Initial solution array
            grid_data: Grid data structure
            get_cell_volumes_func: Function to get cell volumes
            apply_bc_func: Function to apply boundary conditions
            compute_coeffs_func: Function to compute aerodynamic coefficients
            identify_body_faces_func: Function to identify body faces
            compute_ref_area_func: Function to compute reference area
            apply_constraints_func: Function to apply solution constraints
            max_iter: Maximum iterations (overrides config if provided)
            
        Returns:
            Dictionary with results and history
        """
        actual_max_iter = max_iter if max_iter is not None else self.config.max_iter
        
        logger.info(f"Starting steady-state simulation (max_iter={actual_max_iter})...")
        
        start_time = time.time()
        
        # Identify body surfaces and compute reference area
        body_face_indices = identify_body_faces_func()
        ref_area = compute_ref_area_func(body_face_indices)
        
        converged = False
        iteration = 0
        
        for iteration in range(1, actual_max_iter + 1):
            iter_start = time.time()
            
            try:
                # Get cell volumes
                cell_volumes = get_cell_volumes_func()
                
                # Compute residuals
                try:
                    residuals = self.residual_computer.compute_residuals(
                        solution=solution,
                        face_connectivity=self.residual_computer.flux_calculator.face_connectivity 
                            if hasattr(self.residual_computer.flux_calculator, 'face_connectivity') 
                            else None,
                        face_normals=self.residual_computer.flux_calculator.face_normals
                            if hasattr(self.residual_computer.flux_calculator, 'face_normals')
                            else None,
                        face_areas=self.residual_computer.flux_calculator.face_areas
                            if hasattr(self.residual_computer.flux_calculator, 'face_areas')
                            else None,
                        boundary_flags=self.residual_computer.flux_calculator.boundary_flags
                            if hasattr(self.residual_computer.flux_calculator, 'boundary_flags')
                            else None,
                        cell_volumes=cell_volumes,
                        apply_bc_func=apply_bc_func
                    )
                except Exception as e:
                    logger.error(f"Iteration {iteration} failed: {str(e)}")
                    raise RuntimeError(f"Solver failed at iteration {iteration}: {str(e)}")
                
                # Check for NaN/Inf in residuals
                if np.any(np.isnan(residuals)) or np.any(np.isinf(residuals)):
                    logger.error(f"Iteration {iteration}: NaN/Inf detected in residuals")
                    raise RuntimeError(f"Solver diverged at iteration {iteration}: NaN/Inf in residuals")
                
                # Residual norm
                residual_norm = float(np.sqrt(np.sum(residuals[:, 0]**2)))
                
                # Check for residual explosion
                if residual_norm > 1e15:
                    logger.error(f"Iteration {iteration}: Residual explosion detected (Res={residual_norm:.6e})")
                    raise RuntimeError(f"Solver diverged at iteration {iteration}: Residual explosion")
                
                self.residuals_history.append(residual_norm)
                
                # Update convergence monitor and get CFL
                should_continue = self.convergence_monitor.update(
                    residuals=residuals[:, 0],  # Use continuity residual
                    cd=self.cd_history[-1] if self.cd_history else None,
                    cl=self.cl_history[-1] if self.cl_history else None
                )
                
                # Get cell velocities and characteristic lengths for CFL calculation
                rho = solution[:, 0]
                rhou = solution[:, 1]
                rhov = solution[:, 2]
                rhow = solution[:, 3]
                
                u = rhou / np.maximum(rho, 1e-10)
                v = rhov / np.maximum(rho, 1e-10)
                w = rhow / np.maximum(rho, 1e-10)
                
                velocities = np.column_stack([u, v, w])
                
                # Estimate characteristic lengths from cell volumes (cube root)
                cell_volumes_safe = np.maximum(cell_volumes, 1e-15)
                char_lengths = cell_volumes_safe ** (1.0/3.0)
                
                # Let time integrator compute adaptive dt based on CFL
                # Pass residual_norm for adaptive adjustment
                residual_norm_for_dt = residual_norm if iteration > 1 else None
                
                # Use time integrator to update solution with proper dt calculation
                solution = self.time_integrator.step(
                    solution=solution,
                    residuals=residuals,
                    cell_volumes=cell_volumes,
                    velocities=velocities,
                    characteristic_lengths=char_lengths,
                    residual_norm=residual_norm_for_dt
                )
                
                # Apply physical constraints to prevent numerical blow-up
                apply_constraints_func()

                # Compute coefficients
                if iteration % 5 == 0 or iteration == 1:
                    Cd, Cl = compute_coeffs_func(iteration)
                    self.cd_history.append(Cd)
                    self.cl_history.append(Cl)
                else:
                    Cd = self.cd_history[-1] if self.cd_history else 0.0
                    Cl = self.cl_history[-1] if self.cl_history else 0.0
                
                # Update ramp factor for boundary conditions (if available)
                if bc_handler is not None and hasattr(bc_handler, 'update_ramp_factor'):
                    ramp_factor = bc_handler.update_ramp_factor(iteration, actual_max_iter)
                    if iteration <= 10 or iteration % 5 == 0:
                        logger.info(f"[Iter {iteration}] Ramp factor: {ramp_factor:.3f} (velocity: {bc_handler.get_current_inlet_velocity():.2f} m/s)")

                # Log progress
                iter_time = time.time() - iter_start
                cfl_current = self.convergence_monitor.cfl_current
                
                logger.info(
                    f"Iter {iteration:4d}/{actual_max_iter} | "
                    f"Res: {residual_norm:.6e} | "
                    f"Cd: {Cd:.4f} | Cl: {Cl:.4f} | "
                    f"CFL: {cfl_current:.2f} | "
                    f"Time: {iter_time:.2f}s"
                )
                
                # Check convergence/divergence
                if not should_continue:
                    if self.convergence_monitor.converged:
                        logger.info(f"Converged at iteration {iteration}!")
                        converged = True
                    elif self.convergence_monitor.diverged:
                        logger.error("Simulation diverged!")
                        raise RuntimeError("Simulation diverged")
                    else:
                        logger.info("Reached maximum iterations")
                    break

            except Exception as e:
                logger.error(f"Iteration {iteration} failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
                raise RuntimeError(f"Solver failed at iteration {iteration}: {e}")
        
        # Final statistics
        elapsed_time = time.time() - start_time
        avg_iter_time = elapsed_time / iteration if iteration > 0 else 0
        
        logger.info(f"\nSimulation completed:")
        logger.info(f"  Iterations: {iteration}")
        logger.info(f"  Elapsed time: {elapsed_time:.2f}s")
        logger.info(f"  Avg iter time: {avg_iter_time:.3f}s/iter")
        logger.info(f"  Converged: {converged}")
        
        if self.cd_history:
            logger.info(f"  Final Cd: {self.cd_history[-1]:.4f}")
            logger.info(f"  Final Cl: {self.cl_history[-1]:.4f}")
        
        if self.residuals_history:
            initial_res = self.residuals_history[0]
            final_res = self.residuals_history[-1]
            reduction = np.log10(initial_res / max(final_res, 1e-16))
            logger.info(f"  Residual reduction: {reduction:.2f} orders")
        
        return {
            'converged': converged,
            'iterations': iteration,
            'final_residual': self.residuals_history[-1] if self.residuals_history else 0.0,
            'cd_history': self.cd_history.copy(),
            'cl_history': self.cl_history.copy(),
            'residuals_history': self.residuals_history.copy(),
            'solution': solution.copy(),
            'ref_area': ref_area,
            'body_face_indices': body_face_indices,
        }
