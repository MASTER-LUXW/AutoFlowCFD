"""逐点函数按单元分块的多线程求值（`core/utils/pointwise_threads.py`）：与整场调用逐位相同。"""

import warnings

import numba
import numpy as np
import pytest

from autoflowcfd.core.utils.pointwise_threads import MIN_CELLS_PER_CHUNK, map_cell_chunks


@pytest.fixture
def four_threads():
    before = numba.get_num_threads()
    numba.set_num_threads(min(4, numba.config.NUMBA_NUM_THREADS))
    yield
    numba.set_num_threads(before)


def test_chunked_evaluation_is_bitwise_identical_for_arrays_and_tuples(four_threads):
    rng = np.random.default_rng(0)
    n = 5 * MIN_CELLS_PER_CHUNK + 37
    a, b = rng.standard_normal((n, 6)), rng.standard_normal((n, 6, 3))
    seen = []

    def f(x, y, scale):
        seen.append(x.shape[0])
        return np.exp(x) * scale, np.linalg.norm(y, axis=-1) ** 6

    whole = f(a, b, 2.5)
    seen.clear()
    chunked = map_cell_chunks(f, (a, b), 2.5)
    assert len(seen) > 1 and sum(seen) == n                      # 真的分了块
    for w, c in zip(whole, chunked):
        np.testing.assert_array_equal(w, c)
    np.testing.assert_array_equal(map_cell_chunks(lambda x: x * x, (a,)), a * a)


def test_small_inputs_and_other_array_modules_are_called_directly(four_threads):
    calls = []

    def f(x):
        calls.append(x.shape[0])
        return x + 1.0

    small = np.zeros((2 * MIN_CELLS_PER_CHUNK - 1, 2))
    map_cell_chunks(f, (small,))
    big = np.zeros((4 * MIN_CELLS_PER_CHUNK, 2))
    map_cell_chunks(f, (big,), xp=object())                      # 不是 numpy：原样调用
    assert calls == [small.shape[0], big.shape[0]]


def test_worker_threads_use_the_callers_errstate(four_threads):
    """`np.errstate` 是线程局部的：调用方屏蔽的溢出警告不能在工作线程里重新冒出来，调用方没屏蔽的照常报。"""
    x = np.full((4 * MIN_CELLS_PER_CHUNK, 2), 1e200)

    def overflow(v):
        return v * v

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with np.errstate(over="ignore"):
            assert np.isinf(map_cell_chunks(overflow, (x,))).all()
        with np.errstate(over="raise"), pytest.raises(FloatingPointError):
            map_cell_chunks(overflow, (x,))


def test_sa_pointwise_terms_are_unchanged_by_chunking(four_threads):
    """SA 的源项 / 阻尼 / 涡粘经分块求值与直接调用逐位相同（含负支、壁面点 d = 0）。"""
    from autoflowcfd.core.turbulence.sa.model import _pointwise_source
    from autoflowcfd.core.turbulence.sst.kernels import compute_strain_and_vorticity_magnitude

    rng = np.random.default_rng(1)
    n = 3 * MIN_CELLS_PER_CHUNK
    nt = 1e-4 * rng.standard_normal((n, 6))
    nu = np.full((n, 6), 1.5e-5)
    d = np.abs(rng.standard_normal((n, 6))) * 1e-2
    d[::7, 0] = 0.0
    # 涡量模在分块之外整场算一次（与 `SAModel.compute_source_terms` 相同）
    omega = compute_strain_and_vorticity_magnitude(rng.standard_normal((n, 6, 3, 3)) * 50.0)[1]
    with np.errstate(all="ignore"):
        whole = _pointwise_source(nt, nu, d, omega, 0.7, np)
        chunked = map_cell_chunks(_pointwise_source, (nt, nu, d, omega), 0.7, np)
    for w, c in zip(whole, chunked):
        np.testing.assert_array_equal(w, c)


def test_gauss_legendre_is_cached_and_read_only():
    from autoflowcfd.fr.quadrature_points import gauss_legendre

    p1, w1 = gauss_legendre(4)
    p2, w2 = gauss_legendre(4)
    assert p1 is p2 and w1 is w2
    np.testing.assert_array_equal(p1, np.polynomial.legendre.leggauss(4)[0])
    with pytest.raises(ValueError):
        p1[0] = 0.0
