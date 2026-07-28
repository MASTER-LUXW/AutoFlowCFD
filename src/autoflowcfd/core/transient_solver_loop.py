"""Transient solver main loop implementation.

Implements the time-marching loop for transient DES/LES simulations
with statistics collection and checkpoint management.
"""

import numpy as np
from typing import Dict
import time

from .transient_result import TransientResult
from .transient_helpers import TransientSolverHelpers
from .transient_solver_base import TransientSolverBase


class TransientSolver(TransientSolverBase):
    """Transient solver for DES/LES simulations.
    
    Implements time-accurate simulation loop with statistics collection,
    checkpoint management, and adaptive time stepping.
    """
    
    def solve(
        self,
        solution: np.ndarray,
        grid_data,
        boundary_map: Dict,
        bc_params: Dict
    ) -> TransientResult:
        """Run transient simulation.
        
        Args:
            solution: Initial solution field
            grid_data: Grid data structure
            boundary_map: Boundary condition mapping
            bc_params: Boundary condition parameters
            
        Returns:
            Transient simulation results
        """
        print(f"[Transient] Starting simulation")
        print(f"[Transient] Total time: {self.total_time:.4f} s")
        print(f"[Transient] Time step: {self.dt:.6f} s")
        print(f"[Transient] Expected steps: {int(self.total_time / self.dt)}")
        
        start_time = time.time()
        
        # Main time marching loop
        while self.current_time < self.total_time:
            step_start = time.time()
            
            # Execute one time step
            solution = self._execute_timestep(solution, grid_data, boundary_map, bc_params)
            
            # Progress reporting
            step_time = time.time() - step_start
            if self.n_steps % 100 == 0:
                self._report_progress(start_time, step_time)
            
            # Check termination
            residuals = self._last_residuals if hasattr(self, '_last_residuals') else None
            if residuals is not None:
                cd, cl = self.cd_history[-1] if self.cd_history else (0.0, 0.0)
                should_continue = self.convergence_monitor.update(residuals, cd=cd, cl=cl)
                if not should_continue:
                    print("[Transient] Early termination requested")
                    break
        
        # Final report
        total_elapsed = time.time() - start_time
        print(f"\n[Transient] Simulation completed")
        print(f"[Transient] Total steps: {self.n_steps}")
        print(f"[Transient] Physical time: {self.current_time:.6f} s")
        print(f"[Transient] Wall clock time: {total_elapsed:.2f} s")
        print(f"[Transient] Average step time: {total_elapsed/max(1,self.n_steps)*1000:.2f} ms")
        
        # Create result object
        result = TransientResult(
            solution_final=solution,
            total_time=self.current_time,
            n_steps=self.n_steps,
            cd_history=self.cd_history,
            cl_history=self.cl_history,
            time_stamps=self.time_stamps,
            checkpoint_path=self.checkpoint_path
        )
        
        return result
    
    def _execute_timestep(
        self,
        solution: np.ndarray,
        grid_data,
        boundary_map: Dict,
        bc_params: Dict
    ) -> np.ndarray:
        """Execute one time step."""
        # 1. Apply boundary conditions
        solution = self.backend.apply_boundary_conditions(
            solution, boundary_map, bc_params
        )
        
        # 2. Compute fluxes
        flux = self.backend.compute_flux(
            solution,
            grid_data.cells.connectivity,
            grid_data.face_normals
        )
        
        # 3. Compute residuals
        residuals = self.backend.compute_residuals(
            solution,
            flux,
            grid_data.cell_volumes,
            grid_data.boundary_mask
        )
        
        self._last_residuals = residuals
        
        # 4. Update solution
        velocities = TransientSolverHelpers.extract_velocities(solution)
        char_lengths = TransientSolverHelpers.compute_characteristic_lengths(grid_data)
        
        solution = self.time_integrator.step(
            solution,
            residuals,
            grid_data.cell_volumes,
            velocities,
            char_lengths
        )
        
        # 5. Synchronize
        self.backend.synchronize()
        
        # 6. Compute coefficients
        cd, cl = TransientSolverHelpers.compute_aero_coefficients(solution, grid_data)
        
        # 7. Record history
        self.current_time += self.dt
        self.n_steps += 1
        
        self.cd_history.append(cd)
        self.cl_history.append(cl)
        self.time_stamps.append(self.current_time)
        
        # 8. Sample/checkpoint
        if self._should_sample():
            self._collect_statistics(solution)
        
        if self._should_checkpoint():
            self.checkpoint_path, self.last_checkpoint_time = TransientSolverHelpers.save_checkpoint(
                solution, self.current_time, self.n_steps, self.last_checkpoint_time
            )
        
        return solution
    
    def _report_progress(self, start_time: float, step_time: float) -> None:
        """Report simulation progress."""
        elapsed = time.time() - start_time
        remaining = elapsed / self.n_steps * (self.total_time / self.dt - self.n_steps)
        
        cd = self.cd_history[-1] if self.cd_history else 0.0
        cl = self.cl_history[-1] if self.cl_history else 0.0
        
        print(f"[Transient] Step {self.n_steps}: "
              f"t={self.current_time:.6f}s, "
              f"Cd={cd:.4f}, Cl={cl:.4f}, "
              f"CFL={self.convergence_monitor.cfl_current:.4f}, "
              f"dt_step={step_time:.3f}s, "
              f"ETA={remaining:.1f}s")
    
    def _collect_statistics(self, solution: np.ndarray) -> None:
        """Collect statistical data for post-processing."""
        pass
