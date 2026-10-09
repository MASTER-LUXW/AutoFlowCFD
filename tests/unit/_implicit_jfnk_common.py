"""JFNK 隐式求解器测试（test_implicit_jfnk*.py）共用的线性/非线性模型问题。"""

import numpy as np


#: 与本项目真实来流条件同量级的参考量级
_SCALES = np.array([1.225, 40.0, 40.0, 40.0, 2.5e5])


_N = 200


_NV = 5


#: 伪瞬态项趋零（纯 Newton）用的 dtau。取 1e12 而不是 inf：`I/dtau` 在
#: float64 下是 1e-12，相对 `J` 的量级（O(1)~O(1e4)）可忽略，同时避免
#: 除零。
_DTAU_PURE_NEWTON = 1e12


def _linear_system(seed=7):
    """`R(U) = A (U - U_star)`，`A` 良态且变量间耦合、行列按量级缩放。"""
    rng = np.random.default_rng(seed)
    a_var = rng.normal(size=(_NV, _NV)) + _NV * np.eye(_NV)
    a = a_var * _SCALES[:, None] * (1.0 / _SCALES)[None, :]
    u_star = _SCALES[None, :] * (1.0 + 0.1 * rng.normal(size=(_N, _NV)))

    def residual(u_flat):
        return (u_flat - u_star) @ a.T

    return residual, u_star, a


def _rms(x):
    return float(np.linalg.norm(x) / np.sqrt(x.size))


def _step_limited_system(delta, seed=7):
    """`R` 在"离基态超过 `delta`（无量纲 RMS 位移）"之外整体放大 1e6 倍。

    用途：造出一个**方向不可信、但缩小 `dtau` 就可信**的情形，这正是
    PTC 缩 `dtau` 存在的理由（`implicit/dtau_control.py`）。

    为什么用一个不连续的"远支"而不是某个光滑强非线性：要钉住的是
    "一步被残差判据拒绝之后会不会缩 `dtau` 重试"这条**控制逻辑**，
    它需要的是"大步一定被拒、小步一定被接受"这个确定性，而光滑非线性
    要靠调参数去凑这个分界（调出来的分界还会随 seed 漂）。这里不连续
    是刻意的：它模拟"越过这一步残差求值就是垃圾"，而 Fréchet 差分用的
    `eps ~ 1e-8` 始终落在近支，所以 `J` 仍是近支的真实 Jacobian。
    """
    rng = np.random.default_rng(seed)
    a_var = rng.normal(size=(_NV, _NV)) + _NV * np.eye(_NV)
    a = a_var * _SCALES[:, None] * (1.0 / _SCALES)[None, :]
    u_star = _SCALES[None, :] * (1.0 + 0.1 * rng.normal(size=(_N, _NV)))
    u0 = _SCALES[None, :] * np.ones((_N, _NV))

    def residual(u_flat):
        d = (u_flat - u0) / _SCALES[None, :]
        base = (u_flat - u_star) @ a.T
        if float(np.sqrt(np.mean(d ** 2))) > delta:
            return base * 1.0e6
        return base

    return residual, u0
