"""低马赫数伪时间预处理：CPU 与 GPU 同一个函数（`core/utils/preconditioning.py`，I-115）的两条执行分支一致。

numpy 数组走逐点 numba 核，其余数组模块（CuPy）走同一算式的向量化表达式 `_apply_low_mach_precond_vectorized`。
2026-10-22 以前 GPU 有一份从守恒变量自己推导原始变量的拷贝（`gpu_preconditioning.py`，beta^2 与有效声速各写
一份）。这里把向量化表达式用 numpy 执行、与 numba 核逐点比对（公式与运算顺序这一层），并钉住两条分支共同的
语义：n_vars>5 时湍流分量原样保留；非物理状态（rho<=0 或 p<=0）原样透传而不是产生 NaN；`out=` 就地写与拷贝
语义逐位相同。真实设备层（有 CuPy 时）把同一组输入上 GPU 跑一遍。
"""
import numpy as np
import pytest

from autoflowcfd.core.utils.preconditioning import (
    _apply_low_mach_precond_vectorized,
    apply_low_mach_preconditioner,
)

GAMMA = 1.4


def _primitive(rho, u, v, w, p):
    return np.stack([rho, u, v, w, p], axis=-1)


def _vectorized(R, Q, mach_ref, out=None):
    return _apply_low_mach_precond_vectorized(np, R, Q, float(mach_ref), 1.1, out)


#: 两条执行分支（同一个公开函数在 numpy 与其余数组模块上的实现）
BRANCHES = {"numba": apply_low_mach_preconditioner, "vectorized": _vectorized}


class TestVectorizedMatchesKernel:
    @pytest.mark.parametrize("mach", [0.005, 0.05, 0.1, 0.3, 1.2])
    def test_apply_matches(self, mach):
        rng = np.random.default_rng(2026)
        rho = np.full((3, 4), 1.225)
        p = np.full((3, 4), 101325.0)
        a = np.sqrt(GAMMA * p / rho)
        Q = _primitive(rho, mach * a, 0.11 * mach * a, -0.07 * mach * a, p)
        R = rng.standard_normal((3, 4, 5)) * np.array([1.0, 30.0, 30.0, 30.0, 1e4])
        got = _vectorized(R.copy(), Q, 0.1)
        ref = apply_low_mach_preconditioner(R.copy(), Q, 0.1)
        assert np.abs(got - ref).max() / max(float(np.abs(ref).max()), 1e-300) <= 1e-12

    @pytest.mark.parametrize("branch", list(BRANCHES))
    def test_degenerates_to_identity_at_high_mach(self, branch):
        rho = np.full((2, 2), 1.0)
        p = np.full((2, 2), 1.0e5)
        a = np.sqrt(GAMMA * p / rho)
        Q = _primitive(rho, 1.5 * a, 0.0 * a, 0.0 * a, p)
        R = np.ones((2, 2, 5)) * np.array([1.0, 20.0, 20.0, 20.0, 5e3])
        np.testing.assert_allclose(BRANCHES[branch](R.copy(), Q, 0.1), R, rtol=0.0, atol=1e-12)


def _state(n_cells, n_sps, u=30.0):
    rho = np.full((n_cells, n_sps), 1.2)
    return _primitive(rho, np.full((n_cells, n_sps), u), np.zeros((n_cells, n_sps)), np.zeros((n_cells, n_sps)),
                      np.full((n_cells, n_sps), 1.0e5))


@pytest.mark.parametrize("branch", list(BRANCHES))
class TestSharedSemantics:
    def test_turbulence_components_untouched(self, branch):
        """n_vars=7（SST 把 k/omega 挂在 5:）时湍流分量必须原样保留。"""
        R = np.random.default_rng(11).standard_normal((2, 3, 7))
        out = BRANCHES[branch](R.copy(), _state(2, 3), 0.1)
        assert out.shape == R.shape
        np.testing.assert_array_equal(out[..., 5:], R[..., 5:])
        assert not np.allclose(out[..., :5], R[..., :5])

    def test_nonphysical_state_passes_through(self, branch):
        """rho<=0 或 p<=0（正性限制器尚未介入的瞬态）原样透传，不产生 NaN。"""
        Q = _state(1, 3)
        Q[0, 1, 0] = 0.0          # rho = 0
        Q[0, 2, 4] = -1.0         # p < 0
        out = BRANCHES[branch](np.ones((1, 3, 5)), Q, 0.1)
        assert np.all(np.isfinite(out))
        np.testing.assert_array_equal(out[0, 1], np.ones(5))
        np.testing.assert_array_equal(out[0, 2], np.ones(5))
        assert not np.allclose(out[0, 0], np.ones(5)), "物理状态那一行没有被预处理，本测试失去判别力"

    @pytest.mark.parametrize("n_vars", [5, 7])
    def test_inplace_out_matches_copy_semantics(self, branch, n_vars):
        """`out=residual` 就地写与默认拷贝语义逐位相同（第 i 个分量只依赖 r_i，公式一旦跨分量耦合即失败）。"""
        rng = np.random.default_rng(99)
        scale = np.array([1.0, 30.0, 30.0, 30.0, 1e4] + [1.0] * (n_vars - 5))
        R = rng.standard_normal((4, 3, n_vars)) * scale
        Q = _state(4, 3, u=34.0)
        ref = BRANCHES[branch](R.copy(), Q, 0.1)
        buf = R.copy()
        got = BRANCHES[branch](buf, Q, 0.1, out=buf)
        assert got is buf
        np.testing.assert_array_equal(got, ref)

    def test_out_into_a_third_array_and_input_untouched(self, branch):
        R = np.arange(2 * 2 * 5, dtype=float).reshape(2, 2, 5)
        R_before = R.copy()
        dst = np.empty_like(R)
        out = BRANCHES[branch](R, _state(2, 2, u=34.0), 0.1, out=dst)
        assert out is dst
        np.testing.assert_array_equal(R, R_before)
        np.testing.assert_array_equal(dst, BRANCHES[branch](R.copy(), _state(2, 2, u=34.0), 0.1))


# 只跳过"真实设备层"那一类，**不能**放在模块级：模块级 `pytest.importorskip` 会把本机可以真跑的部分一起跳过。
try:
    import cupy as _cupy
    _HAS_CUPY = True
except Exception:      # noqa: BLE001 - CuPy 缺失/CUDA 不可用都算不可用
    _cupy = None
    _HAS_CUPY = False


@pytest.mark.skipif(not _HAS_CUPY, reason="真实设备层验证需要 CuPy + CUDA")
class TestRealDeviceMatchesCPU:
    """真实设备层：本机无 CuPy 时整类跳过，必须在有真实 GPU 的环境跑过。"""

    @pytest.mark.parametrize("mach", [0.01, 0.1, 0.5])
    def test_real_gpu_matches_cpu(self, mach):
        rng = np.random.default_rng(7)
        rho = np.full((5, 8), 1.225)
        p = np.full((5, 8), 101325.0)
        a = np.sqrt(GAMMA * p / rho)
        Q = _primitive(rho, mach * a, 0.2 * mach * a, -0.1 * mach * a, p)
        R = rng.standard_normal((5, 8, 5)) * np.array([1.0, 30.0, 30.0, 30.0, 1e4])
        got = _cupy.asnumpy(apply_low_mach_preconditioner(_cupy.asarray(R), _cupy.asarray(Q), 0.1))
        ref = apply_low_mach_preconditioner(R.copy(), Q, 0.1)
        assert np.abs(got - ref).max() / max(float(np.abs(ref).max()), 1e-300) <= 1e-12


class TestGpuSolverWiring:
    """GPU 求解器上这两个机制的**接线**（不需要 CuPy 就能验的部分）。

    这些是移植时最容易漏、而且漏了之后不会报错只会静默退化的点：
      * `low_mach_precond_enabled` 的开关语义（构造参数 + 时间方案门控）必须与 CPU 版一致；
      * GPU 此前**完全没有**自适应 CFL 控制器（恒用构造时的固定 cfl），
        `--cfl-start/--cfl-max` 只对 CPU 生效——本轮补齐，这里钉住；
      * 阶数切换复位控制器已移到四个后端共用的 Order Continuation 循环
        （2026-10-05，见 test_order_continuation_resume.py::TestPerPhaseCflReset）。
    """

    @staticmethod
    def _fake_solver(time_scheme="ssp_rk3", requested=True, cfl_start=None, cfl_max=None):
        """不构造真实 GPUFRSolver（需要 CuPy + 网格），只跑构造函数调用的那两个
        与 GPU 无关的**共享**函数（2026-09-25 起全部后端都经它们构造，此前本替身
        自己抄了一份开关/控制器逻辑，连带抄了已删除的 `cfl` 硬编码兜底）。"""
        from autoflowcfd.core.time_integration.adaptive_cfl.policy import build_cfl_policy
        from autoflowcfd.core.utils.preconditioning import resolve_low_mach_precond

        class _S:
            pass

        s = _S()
        s.low_mach_precond_enabled = resolve_low_mach_precond(requested, time_scheme)
        s._cfl_controller, s.fixed_cfl_number = build_cfl_policy(
            time_scheme, cfl_start=cfl_start, cfl_max=cfl_max)
        return s

    def test_switch_semantics_match_cpu(self):
        """全部后端必须经同一个函数决定开关（`configure_time_stepping` -> `resolve_low_mach_precond`；此前五份
        写法，多 GPU 完全分布式加载那条路径根本没设这个属性）。"""
        from tests.unit._module_source import module_source

        import autoflowcfd.core.fr_solver.solver as cpu_single
        import autoflowcfd.core.gpu.distributed.gpu_distributed as gpu_multi
        import autoflowcfd.core.gpu.solver.gpu_solver as gpu_single
        import autoflowcfd.core.mpi.distributed_solver as cpu_mpi
        # 分布式两种加载方式各自共用一个装配（CPU：distributed_solver/assembly.py；多 GPU：gpu_distributed/setup.py，
        # 完全分布式构造经它，I-110）
        for mod in (cpu_single, gpu_single, gpu_multi, cpu_mpi):
            src = module_source(mod)
            assert "configure_time_stepping(" in src, mod.__name__
            assert "AFCFD_LOW_MACH_PRECOND" not in src, (
                f"{mod.__name__} 又读起了环境变量（开关只来自构造参数，见 resolve_low_mach_precond）")

    @pytest.mark.parametrize("scheme,expected", [
        ("ssp_rk3", True), ("ssp_rk2", True), ("newton_krylov", True),
        ("dual_time", False), ("forward_euler", False),
    ])
    def test_precond_gated_by_time_scheme(self, scheme, expected):
        s = self._fake_solver(time_scheme=scheme)
        assert s.low_mach_precond_enabled is expected

    def test_environment_variable_has_no_effect(self, monkeypatch):
        """2026-10-18 以前 `AFCFD_LOW_MACH_PRECOND` 优先于构造参数，checkpoint 记不下它。"""
        monkeypatch.setenv("AFCFD_LOW_MACH_PRECOND", "0")
        assert self._fake_solver(requested=True).low_mach_precond_enabled is True
        monkeypatch.setenv("AFCFD_LOW_MACH_PRECOND", "1")
        assert self._fake_solver(requested=False).low_mach_precond_enabled is False

    def test_controller_exists_and_honors_cli_values(self):
        s = self._fake_solver(cfl_start=0.2, cfl_max=0.9)
        assert s._cfl_controller is not None
        assert s._cfl_controller.cfl_start == 0.2
        assert s._cfl_controller.cfl_max == 0.9

    def test_controller_defaults_come_from_controller_signature(self):
        """不传 cfl_start/cfl_max：取控制器签名的默认值（唯一事实来源），不再
        有"缺省退回构造参数 cfl"的兜底（那个 cfl 默认 1.0，远超 P>=1 稳定极限）。"""
        from autoflowcfd.core.time_integration.adaptive_cfl import AdaptiveCFLController

        s = self._fake_solver()
        ref = AdaptiveCFLController()
        assert (s._cfl_controller.cfl_start, s._cfl_controller.cfl_max) == (
            ref.cfl_start, ref.cfl_max)

    def test_turbulence_uses_physical_dt_in_source(self):
        """GPU 湍流场更新必须取物理 dt（不能跟着预处理放大 7 倍）。

        2026-09-25 起步长由 `step()` 按与 CPU 同一规则选定后传入
        `compute_turbulence_source_gpu`（逐点数值由
        `test_gpu_solver_turbulence_source.py` 的非均匀 dt 对照覆盖）。"""
        from autoflowcfd.core.gpu.solver.gpu_solver import step as gstep
        src = __import__("inspect").getsource(gstep)
        assert "return_physical_too=True" in src and "dt_physical[:, None]" in src, (
            "GPU 湍流更新没有用物理波速那一份逐点 dt——k/omega 的显式更新没有"
            "point-implicit 阻尼，不能跟着预处理放大")
