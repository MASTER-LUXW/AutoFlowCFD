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



class TestGradClipErrstateWrapping:
    def test_source_file_wraps_grad_clip_in_errstate(self):
        """Guards against the wrapping being silently removed by a future
        edit. 2026-09-26: the clipping lives in one shared helper
        (`sst/bounds.py::clip_gradient_magnitude`, used by the CPU source,
        CPU transport, single-GPU and multi-GPU paths); the norm must sit
        inside `np.errstate` there, and the source evaluation must call it."""
        import inspect

        from autoflowcfd.core.fr_solver.turbulence.source import evaluate_turbulence_rates
        from autoflowcfd.core.turbulence.sst.bounds import clip_gradient_magnitude

        src = inspect.getsource(clip_gradient_magnitude)
        errstate_idx = src.index('with np.errstate(over="ignore", invalid="ignore"):')
        norm_idx = src.index("mag = np.linalg.norm(grad, axis=-1)")
        assert errstate_idx < norm_idx, (
            "np.errstate wrapping must appear before the norm computation it's meant to protect")
        assert "clip_gradient_magnitude(grad_k, np)" in inspect.getsource(evaluate_turbulence_rates)

    def test_overflow_prone_norm_and_clip_is_warning_free_under_errstate(self):
        """Reproduces the actual numeric failure mode in isolation: a
        gradient component large enough that squaring it overflows
        float64 (>~1.34e154), run through the shared clipping helper -
        must produce zero warnings and a correctly clipped (not NaN) result."""
        from autoflowcfd.core.turbulence.sst.bounds import clip_gradient_magnitude

        grad_k = np.zeros((2, 1, 3))
        grad_k[0, 0, 0] = 1e200  # squaring this overflows float64
        grad_k[1, 0, 1] = 3.0

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = clip_gradient_magnitude(grad_k, np)

        assert len(caught) == 0, f"expected no warnings, got: {[str(w.message) for w in caught]}"
        assert np.isfinite(out).all()
        assert out[0, 0, 0] == 0.0     # 模长溢出成 inf -> 缩放 0（不是 NaN），与此前行为一致
        np.testing.assert_array_equal(out[1], grad_k[1])          # 未超限的点逐位不变
