"""Legacy / experimental core modules - NOT on the live solve path.

The actual production solve path (FRSolver.solve() / TransientSolver.solve())
is built from BoundaryConditionHandler, ViscousRANSResidual
(fvm_viscous_residual.py), TimeIntegrator, FVMFaceExtractor, and
AeroCoefficientCalculator - all of which live one level up, directly under
autoflowcfd.core. SST k-omega, MUSCL reconstruction/limiting, and residual
assembly are re-implemented inline in fvm_viscous_residual.py/bc_handler.py
rather than through the classes in this subpackage.

Everything here is a parallel, unused-in-production implementation stack
that predates the current inline path. It's kept (not deleted) because it
still has its own standalone unit tests (tests/unit/test_fr_scheme.py,
test_muscl_reconstruction.py, test_time_and_turbulence.py,
tests/integration/test_iteration3_solver.py, test_convergence_integration.py,
test_end_to_end_steady.py) - a fix made only here will NOT affect an actual
solve; fix the live modules in the parent package instead.

Still re-exported at the autoflowcfd.core package level for backward
compatibility (see core/__init__.py) - existing `from autoflowcfd.core
import FRScheme` style imports are unaffected by this module living here.
"""
