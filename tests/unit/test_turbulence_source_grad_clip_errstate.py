"""Regression test for a real bug found 2026-08-22 (user directly saw the
raw RuntimeWarning in a real P0->P1 order-continuation transition log):

`fr_solver/turbulence.py::compute_turbulence_source`'s grad_k/grad_omega
magnitude clipping (`np.linalg.norm(grad_k, axis=-1)` followed by a
1e6-cap) had no `np.errstate` protection. On degenerate cells (metric
ratio adj(J)/det(J) blown up by collapsed-coordinate geometry - see
troubled_cell.py's degenerate-cell diagnostics), floating-point noise in
a theoretically-constant gradient gets amplified past 1e150; squaring
that inside `np.linalg.norm`'s internals overflows float64 and raises a
RuntimeWarning that reaches the user's console, even though the
subsequent clip already handles an `inf` input correctly (inf > 1e6 is
always True, so the value gets scaled down, not turned into NaN).

`turbulence/transport.py` has the *exact* same clipping pattern and was
already wrapped in `np.errstate(over='ignore', invalid='ignore')` on
2026-08-21 - this file (the older, original location the transport.py
comment itself references) was missed at the time. This test pins the
suppression behaviour on the isolated numeric pattern (not the full
`compute_turbulence_source` call, which needs a large solver mock with
real mesh/ops geometry for `compute_scalar_gradient` - not worth
mocking out just to re-prove numpy's own documented errstate semantics);
what matters is that this file's code is now wrapped identically to the
already-covered transport.py location.
"""

import warnings

import numpy as np

import autoflowcfd.core.fr_solver.turbulence as turbulence_module


class TestGradClipErrstateWrapping:
    def test_source_file_wraps_grad_clip_in_errstate(self):
        """Guards against the wrapping being silently removed by a future
        edit: the exact code block that computes grad_k_mag/grad_omega_mag
        via np.linalg.norm must sit inside a `with np.errstate(...)` in
        this module's source."""
        import inspect

        # 2026-09-25：梯度裁剪随源项求值一起拆进了 `evaluate_turbulence_rates`
        # （显式与隐式 k-omega 更新共用的求值件），`compute_turbulence_source`
        # 现在只是编排。
        from autoflowcfd.core.fr_solver.turbulence.source import evaluate_turbulence_rates

        src = inspect.getsource(evaluate_turbulence_rates)
        errstate_idx = src.index("with np.errstate(over='ignore', invalid='ignore'):")
        norm_idx = src.index("grad_k_mag = np.linalg.norm(grad_k, axis=-1)")
        assert errstate_idx < norm_idx, (
            "np.errstate wrapping must appear before the grad_k_mag norm "
            "computation it's meant to protect"
        )

    def test_overflow_prone_norm_and_clip_is_warning_free_under_errstate(self):
        """Reproduces the actual numeric failure mode in isolation: a
        gradient component large enough that squaring it overflows
        float64 (>~1.34e154), run through the identical
        norm-then-clip-to-1e6 logic this file uses, inside the same
        errstate context - must produce zero warnings and a correctly
        clipped (not NaN) result."""
        grad_k = np.zeros((2, 1, 3))
        grad_k[0, 0, 0] = 1e200  # squaring this overflows float64

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with np.errstate(over='ignore', invalid='ignore'):
                max_grad_mag = 1e6
                grad_k_mag = np.linalg.norm(grad_k, axis=-1)
                if np.any(grad_k_mag > max_grad_mag):
                    scale_k = max_grad_mag / np.maximum(grad_k_mag, 1e-10)
                    grad_k = grad_k * np.clip(scale_k[..., np.newaxis], 0, 1)

        assert len(caught) == 0, f"expected no warnings, got: {[str(w.message) for w in caught]}"
        assert np.isfinite(grad_k).all()
        assert grad_k[0, 0, 0] == 0.0  # inf-magnitude component clipped to exactly 0, not NaN
