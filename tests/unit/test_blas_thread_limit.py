"""`core/fr_solver/solver/threads.py::blas_threads_limited` 的自检。

为什么需要一个专门的测试：限线程是纯性能调优，**失效时完全没有任何可见
症状**——求解结果一字不差，只是白白损失每步耗时。真实踩过两次：

* 2026-09-13：只在 numpy 包目录内找 OpenBLAS，而 numpy 2.x Windows wheel
  把它放在 site-packages/numpy.libs/（同级目录），ctypes 调用从未执行；
* 2026-09-25：找到第一个 OpenBLAS 就返回，SciPy 自带的那份
  （`libscipy_openblas`，符号前缀 `scipy_openblas_`）从未被限制。

因此这里既检查"每个实例都被找到"，也**实测**限制确实作用在各自正在使用
的那个实例上（同一个 gemm 在限制内必须明显变慢），并钉住退出时逐个恢复到
进入前的值。
"""
import glob
import importlib.util
import os
import time

import numpy as np
import pytest

from autoflowcfd.core.fr_solver.solver import blas_threads_limited
from autoflowcfd.core.fr_solver.solver.threads import _blas_thread_controls, _package_lib_dirs


def _bundled_openblas(package: str) -> list:
    """`package` 的 wheel 自带的 OpenBLAS 动态库（本项目标准环境两者都有）。"""
    if importlib.util.find_spec(package) is None:
        return []
    return [p for d in _package_lib_dirs(package)
            for p in glob.glob(os.path.join(d, "*openblas*"))]


def _control_for(package: str):
    """`package` 那份 OpenBLAS 的 `(get, set)`。"""
    libs = {os.path.normcase(os.path.realpath(p)) for p in _bundled_openblas(package)}
    for path, get, set_ in _blas_thread_controls():
        if os.path.normcase(os.path.realpath(path)) in libs:
            return get, set_
    return None


def _best_time(fn, rep: int = 3) -> float:
    fn()  # 预热
    best = float("inf")
    for _ in range(rep):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


_A = np.random.default_rng(0).standard_normal((1400, 1400))


def _numpy_gemm():
    _A @ _A


def _scipy_gemm():
    from scipy.linalg import blas
    blas.dgemm(1.0, _A, _A)


_GEMM = {"numpy": _numpy_gemm, "scipy": _scipy_gemm}


class TestBlasThreadsLimited:
    @pytest.mark.parametrize("package", ["numpy", "scipy"])
    def test_every_bundled_openblas_is_controlled(self, package):
        """钉住：numpy 与 SciPy 各自那份 OpenBLAS 都必须被找到。"""
        if not _bundled_openblas(package):
            pytest.skip(f"{package} 未自带 OpenBLAS")
        assert _control_for(package) is not None

    def test_only_package_lib_dirs_are_loaded(self):
        """只加载 numpy/SciPy 自己的库目录，不碰 site-packages 里 pip 中断升级
        留下的 `~umpy.libs` 之类的残留副本。"""
        for path, _get, _set in _blas_thread_controls():
            assert "~" not in os.path.basename(os.path.dirname(path)), path

    def test_limits_all_and_restores_previous_values(self):
        controls = _blas_thread_controls()
        if not controls:
            pytest.skip("当前环境没有可控的 BLAS")
        before = [get() for _p, get, _s in controls]
        for _p, _g, set_ in controls:
            set_(3)  # 进入前的值取一个不是 cpu_count 的数，区分"恢复原值"与"猜默认值"
        try:
            with blas_threads_limited(1) as lim:
                assert lim.n_applied == len(controls)
                assert [get() for _p, get, _s in controls] == [1] * len(controls)
            assert [get() for _p, get, _s in controls] == [3] * len(controls)
        finally:
            for (_p, _g, set_), n in zip(controls, before):
                set_(n)

    def test_honours_opt_out_env_var(self, monkeypatch):
        """`AFCFD_NO_BLAS_THREAD_LIMIT=1` 时完全不动 BLAS 配置。"""
        monkeypatch.setenv("AFCFD_NO_BLAS_THREAD_LIMIT", "1")
        before = [get() for _p, get, _s in _blas_thread_controls()]
        with blas_threads_limited(1) as lim:
            assert lim.n_applied == 0
            assert [get() for _p, get, _s in _blas_thread_controls()] == before

    @pytest.mark.skipif((os.cpu_count() or 1) < 4,
                        reason="核数太少，单/多线程 gemm 耗时差异不足以稳定判定")
    @pytest.mark.parametrize("package", ["numpy", "scipy"])
    def test_limit_takes_effect_on_live_blas(self, package):
        """**实测**：限制必须作用在正在使用的那个实例上。

        判据用"同一个 gemm 在限制内变慢"而不是任何自报状态——后者正是两次
        失效时看起来完全正常的东西。阈值 1.3x（本机实测 numpy 5.8x、SciPy
        5.3x），只为区分"真的生效"与"完全没生效"。
        """
        control = _control_for(package)
        if control is None:
            pytest.skip(f"{package} 未自带 OpenBLAS")
        get, set_ = control
        before = get()
        set_(os.cpu_count())
        try:
            t_multi = _best_time(_GEMM[package])
            with blas_threads_limited(1):
                t_single = _best_time(_GEMM[package])
        finally:
            set_(before)
        assert t_single > 1.3 * t_multi, (
            f"{package}: 限制 BLAS 线程后 gemm 未见明显变慢（单线程 {t_single*1e3:.0f}ms "
            f"vs 多线程 {t_multi*1e3:.0f}ms）——set_num_threads 没有作用到正在使用的实例上")
