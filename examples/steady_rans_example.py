"""Example: Using the FR solver for steady-state RANS simulation.

This example demonstrates how to use the AutoFlowCFD solver components
to perform a steady-state RANS simulation with SST k-omega turbulence model.
"""

import numpy as np
from autoflowcfd.core import (
    FRScheme, FROrder,
    create_backend,
    SSTKOmegaModel,
    ConvergenceMonitor
)


def run_steady_rans_example():
    """Run a simple steady RANS simulation example."""
    
    print("=" * 70)
    print("AutoFlowCFD - Steady RANS Simulation Example")
    print("=" * 70)
    
    # Configuration
    n_cells = 1000
    max_iterations = 500
    cfl_initial = 0.1
    
    print(f"\n[Config] Grid size: {n_cells} cells")
    print(f"[Config] Max iterations: {max_iterations}")
    print(f"[Config] Initial CFL: {cfl_initial}")
    
    # Step 1: Create FR scheme
    print("\n[Step 1] Creating FR discretization scheme...")
    fr_scheme = FRScheme(order=FROrder.SECOND)
    print(f"         Order: {fr_scheme.order.name}")
    print(f"         Correction points: {fr_scheme.num_correction_points}")
    
    # Step 2: Initialize backend
    print("\n[Step 2] Initializing computational backend...")
    backend = create_backend("cpu", n_threads=4)
    backend.initialize(n_cells, n_nodes=500)
    print(f"         Backend: {backend.backend_type}")
    device_info = backend.get_device_info()
    print(f"         Device: {device_info['backend']}")
    
    # Step 3: Initialize turbulence model
    print("\n[Step 3] Initializing SST k-ω turbulence model...")
    turb_model = SSTKOmegaModel()
    k_field, omega_field = turb_model.initialize_turbulence_fields(
        n_cells, u_infinity=30.0, length_scale=0.1
    )
    print(f"         Initial k: {k_field.mean():.6f}")
    print(f"         Initial ω: {omega_field.mean():.6f}")
    
    # Step 4: Initialize solution field
    print("\n[Step 4] Initializing solution field...")
    # Solution vector: [rho, rho*u, rho*v, rho*w, E]
    rho_inf = 1.225
    u_inf = 30.0
    E_inf = 101325.0 / (1.4 - 1.0) + 0.5 * rho_inf * u_inf**2
    
    solution = np.zeros((n_cells, 5))
    solution[:, 0] = rho_inf  # density
    solution[:, 1] = rho_inf * u_inf  # x-momentum
    solution[:, 2] = 0.0  # y-momentum
    solution[:, 3] = 0.0  # z-momentum
    solution[:, 4] = E_inf  # total energy
    
    print(f"         Initial density: {solution[:, 0].mean():.4f} kg/m³")
    print(f"         Initial velocity: {u_inf:.1f} m/s")
    
    # Step 5: Setup convergence monitor
    print("\n[Step 5] Setting up convergence monitor...")
    monitor = ConvergenceMonitor(
        convergence_threshold=1e-3,
        max_iterations=max_iterations,
        cfl_initial=cfl_initial,
        check_interval=50
    )
    
    # Step 6: Main iteration loop (simplified)
    print("\n[Step 6] Starting iteration loop...")
    print("-" * 70)
    
    for iteration in range(1, max_iterations + 1):
        # Compute residuals (simplified - placeholder)
        residuals = np.random.rand(n_cells, 5) * 0.01 * np.exp(-iteration / 100.0)
        
        # Update convergence monitor
        cd = 0.3 + 0.01 * np.sin(iteration * 0.05)  # Placeholder Cd
        should_continue = monitor.update(residuals, cd=cd)
        
        # Progress reporting
        if iteration % 50 == 0 or iteration == 1:
            residual_norm = np.linalg.norm(residuals)
            print(f"  Iter {iteration:4d}: "
                  f"Residual = {residual_norm:.6e}, "
                  f"Cd = {cd:.4f}, "
                  f"CFL = {monitor.cfl_current:.4f}")
        
        # Check termination
        if not should_continue:
            print(f"\n  Converged at iteration {iteration}")
            break
    
    print("-" * 70)
    
    # Step 7: Report results
    print("\n[Results]")
    print(f"  Total iterations: {monitor.current_iteration}")
    
    if len(monitor.history.residuals) > 0:
        initial_res = monitor.history.residuals[0]
        final_res = monitor.history.residuals[-1]
        reduction = final_res / initial_res
        
        print(f"  Initial residual: {initial_res:.6e}")
        print(f"  Final residual:   {final_res:.6e}")
        print(f"  Reduction factor: {reduction:.6e}")
    
    if len(monitor.cd_history) > 0:
        cd_mean = np.mean(monitor.cd_history[-50:])
        print(f"  Mean Cd (last 50): {cd_mean:.4f}")
    
    print(f"  Final CFL: {monitor.cfl_current:.4f}")
    print(f"  Convergence rate: {monitor.get_convergence_rate():.6e}")
    
    print("\n" + "=" * 70)
    print("✅ Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    run_steady_rans_example()
