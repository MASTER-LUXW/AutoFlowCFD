"""`core/fr_solver/solver.py::_limit_blas_threads` 的自检。

为什么需要一个专门的测试：这个函数是纯性能调优、内部全程 try/except
静默失败，**失效时完全没有任何可见症状**——求解结果一字不差，只是白白
损失 9~11% 的每步耗时。2026-09-13 真实踩过这个坑：第一版实现只在
numpy 包目录内的 `.libs`/`libs` 下找 OpenBLAS 动态库，而本机 numpy 2.x
Windows wheel 把它放在 **site-packages/numpy.libs/**（numpy 包的*同级*
目录），于是 ctypes 调用从未执行、函数照常"成功"返回。是靠"限制前后测
同一个大 gemm 的耗时"这个探针才发现的。

因此这里既检查返回值（找到并调用了某个后端的 set_num_threads），也
**实测**限制确实作用在正在使用的那个 BLAS 实例上（同一个 gemm 在限制
后必须明显变慢、恢复线程数后必须变快回来）。
"""
import os
import time

import numpy as np
import pytest

from autoflowcfd.core.fr_solver.solver import _limit_blas_threads


def _numpy_uses_openblas() -> bool:
    """判断当前 numpy 是否是 OpenBLAS 后端（本项目 requirements 的标准情形）。"""
    np_dir = os.path.dirname(np.__file__)
    site_dir = os.path.dirname(np_dir)
    import glob
    pats = [
        os.path.join(site_dir, "*.libs", "*openblas*"),
        os.path.join(np_dir, ".libs", "*openblas*"),
        os.path.join(np_dir, "libs", "*openblas*"),
    ]
    return any(glob.glob(p) for p in pats)


def _time_gemm(n: int = 1400, rep: int = 3) -> float:
    a = np.random.default_rng(0).standard_normal((n, n))
    a @ a  # 预热
    best = float("inf")
    for _ in range(rep):
        t0 = time.perf_counter()
        a @ a
        best = min(best, time.perf_counter() - t0)
    return best


class TestLimitBlasThreads:
    def test_returns_bool_and_never_raises(self):
        """契约：纯性能调优，任何环境下都不允许抛异常。"""
        result = _limit_blas_threads(1)
        assert isinstance(result, bool)
        # 恢复，避免影响同进程后续测试的耗时特征
        _limit_blas_threads(os.cpu_count() or 4)

    def test_honours_opt_out_env_var(self, monkeypatch):
        """`AFCFD_NO_BLAS_THREAD_LIMIT=1` 时必须完全不动 BLAS 配置
        （给需要自己管线程的用户/CI 留的逃生门）。"""
        monkeypatch.setenv("AFCFD_NO_BLAS_THREAD_LIMIT", "1")
        assert _limit_blas_threads(1) is False

    @pytest.mark.skipif(not _numpy_uses_openblas(),
                        reason="当前 numpy 不是 OpenBLAS 后端（本测试钉住的是"
                               "OpenBLAS 动态库路径搜索逻辑）")
    def test_finds_openblas_in_this_environment(self):
        """钉住路径搜索逻辑：本项目标准环境（OpenBLAS 后端）下必须找得到。

        这条断言就是 2026-09-13 那次静默失效的回归——当时函数返回"成功"
        但实际什么都没做。
        """
        try:
            assert _limit_blas_threads(1) is True
        finally:
            _limit_blas_threads(os.cpu_count() or 4)

    @pytest.mark.skipif((os.cpu_count() or 1) < 4,
                        reason="核数太少，单/多线程 gemm 耗时差异不足以稳定判定")
    @pytest.mark.skipif(not _numpy_uses_openblas(), reason="非 OpenBLAS 后端")
    def test_limit_actually_takes_effect_on_live_blas(self):
        """**实测**：限制必须作用在正在使用的那个 BLAS 实例上。

        判据用"同一个 gemm 限制后变慢"而不是任何自报状态——后者正是上次
        失效时看起来完全正常的东西。阈值取 1.3x（远低于本机实测的 5.9x，
        留足不同机器的余量），只为区分"真的生效"与"完全没生效"。
        """
        try:
            assert _limit_blas_threads(1) is True
            t_single = _time_gemm()
            assert _limit_blas_threads(os.cpu_count()) is True
            t_multi = _time_gemm()
            assert t_single > 1.3 * t_multi, (
                f"限制 BLAS 线程后 gemm 未见明显变慢（单线程 {t_single*1e3:.0f}ms "
                f"vs 多线程 {t_multi*1e3:.0f}ms）——说明 set_num_threads 没有"
                f"作用到正在使用的 BLAS 实例上"
            )
        finally:
            _limit_blas_threads(os.cpu_count() or 4)
