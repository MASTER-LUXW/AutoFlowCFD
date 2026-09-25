"""GPU 低马赫数预处理与 CPU 实现的一致性验证（2026-09-14）。

分两层，原因是本机没有 CUDA/CuPy：

1. **公式层（本机就能真跑）**：`gpu_preconditioning.py` 里的函数全部只用
   `cp.maximum/clip/sqrt/where/sum` 这类 numpy 同名 API，把 `get_cupy`
   替身成 `numpy` 就能在 CPU 上逐点执行同一份代码，与已验证正确的 CPU
   实现（`core/utils/preconditioning.py`，见
   `test_low_mach_preconditioner.py` 的谱等价/可逆性验证）交叉比对。
   这不是"用 numpy 假装测了 GPU"——被执行的确实是 GPU 模块里那份源码，
   验证的是**公式与运算顺序**这一层，也正是移植最容易出错的那一层
   （符号写反、beta^2 用错速度、H 少一项等）。
2. **真实设备层（本机自动跳过）**：有 CuPy 时把同一组输入上真实 GPU
   跑一遍，与 CPU 结果比对。这一层必须在有真实 GPU 的环境上跑过才算
   最终验证通过——与项目既有 GPU 交叉验证测试同一约定。

另外钉住两条移植时最容易丢的语义：n_vars>5 时湍流分量必须原样保留；
非物理状态（rho<=0 或 p<=0）必须原样透传而不是产生 NaN。
"""
import numpy as np
import pytest

from autoflowcfd.core.utils.preconditioning import (
    apply_low_mach_preconditioner,
    preconditioned_sound_speed,
    _precond_beta2,
)
from tests.unit._gpu_cupy_shim import patch_module_get_cupy

GAMMA = 1.4


def _u_from_primitive(rho, u, v, w, p):
    q2 = u * u + v * v + w * w
    return np.stack([rho, rho * u, rho * v, rho * w,
                     p / (GAMMA - 1.0) + 0.5 * rho * q2], axis=-1)


def _gpu_module_with_numpy(monkeypatch):
    """把 GPU 模块的 `get_cupy` 替身成 numpy，返回该模块。"""
    import autoflowcfd.core.gpu.gpu_preconditioning as gm
    patch_module_get_cupy(monkeypatch, gm, np)
    return gm


class TestFormulaMatchesCPU:
    """公式层：GPU 模块的源码用 numpy 执行，结果必须与 CPU 实现一致。"""

    @pytest.mark.parametrize("mach", [0.005, 0.05, 0.1, 0.3, 1.2])
    def test_apply_matches_cpu(self, monkeypatch, mach):
        gm = _gpu_module_with_numpy(monkeypatch)
        rng = np.random.default_rng(2026)
        rho = np.full((3, 4), 1.225)
        p = np.full((3, 4), 101325.0)
        a = np.sqrt(GAMMA * p / rho)
        u = mach * a
        v = 0.11 * mach * a
        w = -0.07 * mach * a
        U = _u_from_primitive(rho, u, v, w, p)
        Q = np.stack([rho, u, v, w, p], axis=-1)

        R = rng.standard_normal((3, 4, 5)) * np.array([1.0, 30.0, 30.0, 30.0, 1e4])
        got = gm.apply_low_mach_preconditioner_gpu(R.copy(), U, 0.1)
        ref = apply_low_mach_preconditioner(R.copy(), Q, 0.1)

        denom = max(float(np.abs(ref).max()), 1e-300)
        assert np.abs(got - ref).max() / denom <= 1e-12, (
            f"GPU 与 CPU 的 Gamma R 不一致 (mach={mach})")

    @pytest.mark.parametrize("mach_ref", [0.02, 0.1, 0.3])
    def test_beta2_matches_cpu(self, monkeypatch, mach_ref):
        gm = _gpu_module_with_numpy(monkeypatch)
        q2 = np.array([0.0, 1.0, 1e2, 1e4, 1e5])
        a2 = np.full_like(q2, GAMMA * 101325.0 / 1.225)
        np.testing.assert_allclose(
            gm.precond_beta2_gpu(q2, a2, mach_ref, 1.1),
            _precond_beta2(q2, a2, mach_ref, 1.1), rtol=1e-14)

    def test_sound_speed_matches_cpu(self, monkeypatch):
        """伪时间步长用的有效声速：两侧必须共用同一个 beta^2 定义（按速度
        模取）。这条一旦分叉，GPU 的 dt 与 Gamma 就不再成对，重现 CPU 侧
        2026-08-24 那次失稳。"""
        gm = _gpu_module_with_numpy(monkeypatch)
        a = np.full(6, 340.29)
        vel_mag = np.array([0.0, 3.0, 34.0, 100.0, 340.0, 500.0])
        np.testing.assert_allclose(
            gm.preconditioned_sound_speed_gpu(vel_mag, a, 0.1),
            preconditioned_sound_speed(vel_mag, a, 0.1), rtol=1e-14)

    def test_degenerates_to_identity_at_high_mach(self, monkeypatch):
        gm = _gpu_module_with_numpy(monkeypatch)
        rho = np.full((2, 2), 1.0)
        p = np.full((2, 2), 1.0e5)
        a = np.sqrt(GAMMA * p / rho)
        U = _u_from_primitive(rho, 1.5 * a, 0.0 * a, 0.0 * a, p)
        R = np.ones((2, 2, 5)) * np.array([1.0, 20.0, 20.0, 20.0, 5e3])
        out = gm.apply_low_mach_preconditioner_gpu(R.copy(), U, 0.1)
        np.testing.assert_allclose(out, R, rtol=0.0, atol=1e-12)


class TestPortingSemantics:
    def test_turbulence_components_untouched(self, monkeypatch):
        """n_vars=7（SST 把 k/omega 挂在 5:）时湍流分量必须原样保留。"""
        gm = _gpu_module_with_numpy(monkeypatch)
        rng = np.random.default_rng(11)
        rho = np.full((2, 3), 1.2)
        p = np.full((2, 3), 1.0e5)
        U5 = _u_from_primitive(rho, np.full((2, 3), 30.0),
                               np.zeros((2, 3)), np.zeros((2, 3)), p)
        U = np.concatenate([U5, rng.standard_normal((2, 3, 2))], axis=-1)
        R = rng.standard_normal((2, 3, 7))
        out = gm.apply_low_mach_preconditioner_gpu(R.copy(), U, 0.1)
        assert out.shape == R.shape
        np.testing.assert_array_equal(out[..., 5:], R[..., 5:])
        assert not np.allclose(out[..., :5], R[..., :5])

    def test_nonphysical_state_passes_through(self, monkeypatch):
        """rho<=0 或 p<=0（正性限制器尚未介入的瞬态）原样透传，不产生 NaN。"""
        gm = _gpu_module_with_numpy(monkeypatch)
        U = np.zeros((1, 3, 5))
        U[0, 0] = [1.225, 1.225 * 30.0, 0.0, 0.0, 101325.0 / 0.4 + 0.5 * 1.225 * 900]
        U[0, 1] = [0.0, 0.0, 0.0, 0.0, 0.0]          # rho = 0
        U[0, 2] = [1.0, 0.0, 0.0, 0.0, -1.0]          # p < 0
        R = np.ones((1, 3, 5))
        out = gm.apply_low_mach_preconditioner_gpu(R.copy(), U, 0.1)
        assert np.all(np.isfinite(out)), "非物理状态下产生了 NaN/Inf"
        np.testing.assert_array_equal(out[0, 1], np.ones(5))
        np.testing.assert_array_equal(out[0, 2], np.ones(5))
        assert not np.allclose(out[0, 0], np.ones(5)), (
            "物理状态那一行没有被预处理，本测试失去判别力")

    @pytest.mark.parametrize("n_vars", [5, 7])
    def test_inplace_out_matches_copy_semantics(self, monkeypatch, n_vars):
        """`out=residual` 就地写的结果必须与默认拷贝语义**逐位相同**。

        这是整份移植里最容易出别名错误的一处：`r0..r4` 是 residual 的
        视图，就地写时若某个分量的新值依赖别的分量的**旧**值，逐个赋值
        就会读到已经被改写的数据。这里用同一组输入跑两条路径对比，把
        "第 i 个分量只依赖 r_i"这条前提钉死——公式一旦改成跨分量耦合，
        本测试立刻失败。
        """
        gm = _gpu_module_with_numpy(monkeypatch)
        rng = np.random.default_rng(99)
        rho = np.full((4, 3), 1.225)
        p = np.full((4, 3), 101325.0)
        a_snd = np.sqrt(GAMMA * p / rho)
        U5 = _u_from_primitive(rho, 0.1 * a_snd, 0.03 * a_snd, -0.02 * a_snd, p)
        U = (U5 if n_vars == 5 else
             np.concatenate([U5, rng.standard_normal((4, 3, n_vars - 5))], axis=-1))
        scale = np.array([1.0, 30.0, 30.0, 30.0, 1e4] + [1.0] * (n_vars - 5))
        R = rng.standard_normal((4, 3, n_vars)) * scale

        ref = gm.apply_low_mach_preconditioner_gpu(R.copy(), U, 0.1)
        buf = R.copy()
        got = gm.apply_low_mach_preconditioner_gpu(buf, U, 0.1, out=buf)
        assert got is buf, "out= 时必须返回 out 本身"
        np.testing.assert_array_equal(got, ref)

    def test_out_into_a_third_array(self, monkeypatch):
        """out 既不是 residual 也不是新数组时：结果正确且 residual 不被改。"""
        gm = _gpu_module_with_numpy(monkeypatch)
        rho = np.full((2, 2), 1.225)
        p = np.full((2, 2), 101325.0)
        U = _u_from_primitive(rho, np.full((2, 2), 34.0),
                              np.zeros((2, 2)), np.zeros((2, 2)), p)
        R = np.arange(2 * 2 * 5, dtype=float).reshape(2, 2, 5)
        R_before = R.copy()
        dst = np.empty_like(R)
        out = gm.apply_low_mach_preconditioner_gpu(R, U, 0.1, out=dst)
        assert out is dst
        np.testing.assert_array_equal(R, R_before)
        np.testing.assert_array_equal(
            dst, gm.apply_low_mach_preconditioner_gpu(R.copy(), U, 0.1))

    def test_does_not_mutate_input(self, monkeypatch):
        """GPU 版返回新数组、不就地改 residual（CPU 版支持 out=residual
        就地写，两者语义刻意不同，调用方依赖这一点保留 raw 残差做监控）。"""
        gm = _gpu_module_with_numpy(monkeypatch)
        rho = np.full((1, 1), 1.225)
        p = np.full((1, 1), 101325.0)
        U = _u_from_primitive(rho, np.full((1, 1), 34.0),
                              np.zeros((1, 1)), np.zeros((1, 1)), p)
        R = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0]]])
        R_before = R.copy()
        gm.apply_low_mach_preconditioner_gpu(R, U, 0.1)
        np.testing.assert_array_equal(R, R_before)


# 只跳过"真实设备层"那一类，**不能**放在模块级：模块级
# `pytest.importorskip` 会把整个文件（含本机完全可以真跑的公式层）一起
# 跳过——第一版就是这么写的，实测本机只报一条 SKIPPED、公式层一条都
# 没执行，等于白写。
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
        u, v, w = mach * a, 0.2 * mach * a, -0.1 * mach * a
        U = _u_from_primitive(rho, u, v, w, p)
        Q = np.stack([rho, u, v, w, p], axis=-1)
        R = rng.standard_normal((5, 8, 5)) * np.array([1.0, 30.0, 30.0, 30.0, 1e4])

        from autoflowcfd.core.gpu.gpu_preconditioning import (
            apply_low_mach_preconditioner_gpu,
        )
        got = _cupy.asnumpy(apply_low_mach_preconditioner_gpu(
            _cupy.asarray(R), _cupy.asarray(U), 0.1))
        ref = apply_low_mach_preconditioner(R.copy(), Q, 0.1)
        denom = max(float(np.abs(ref).max()), 1e-300)
        assert np.abs(got - ref).max() / denom <= 1e-12


class TestGpuSolverWiring:
    """GPU 求解器上这两个机制的**接线**（不需要 CuPy 就能验的部分）。

    这些是移植时最容易漏、而且漏了之后不会报错只会静默退化的点：
      * `low_mach_precond_enabled` 的开关语义（含环境变量优先、
        时间方案门控）必须与 CPU 版一致；
      * GPU 此前**完全没有**自适应 CFL 控制器（恒用构造时的固定 cfl），
        `--cfl-start/--cfl-max` 只对 CPU 生效——本轮补齐，这里钉住；
      * 阶数切换必须复位控制器：CPU MPI 分布式靠"重建 _local_solver"
        免费拿到复位，单机 GPU 是**原地**改 mesh/ops，控制器会跨阶数
        存活下来，必须显式 reset（否则阶数跳变的残差突变会被当成恶化，
        把 CFL 一路打到下限）。
    """

    @staticmethod
    def _fake_solver(time_scheme="ssp_rk3", env=None, monkeypatch=None,
                     cfl_start=None, cfl_max=None):
        """不构造真实 GPUFRSolver（需要 CuPy + 网格），只跑构造函数调用的那两个
        与 GPU 无关的**共享**函数（2026-09-25 起全部后端都经它们构造，此前本替身
        自己抄了一份开关/控制器逻辑，连带抄了已删除的 `cfl` 硬编码兜底）。"""
        import os as _os

        from autoflowcfd.core.time_integration.adaptive_cfl.policy import build_cfl_policy
        from autoflowcfd.core.utils.preconditioning import resolve_low_mach_precond

        class _S:
            pass

        s = _S()
        old = _os.environ.get("AFCFD_LOW_MACH_PRECOND")
        try:
            if env is not None:
                _os.environ["AFCFD_LOW_MACH_PRECOND"] = env
            s.low_mach_precond_enabled = resolve_low_mach_precond(True, time_scheme)
        finally:
            if env is not None:
                if old is None:
                    _os.environ.pop("AFCFD_LOW_MACH_PRECOND", None)
                else:
                    _os.environ["AFCFD_LOW_MACH_PRECOND"] = old
        s._cfl_controller, s.fixed_cfl_number = build_cfl_policy(
            time_scheme, cfl_start=cfl_start, cfl_max=cfl_max)
        return s

    def test_switch_semantics_match_cpu(self):
        """全部后端必须经同一个 `resolve_low_mach_precond` 决定开关（此前五份
        写法，多 GPU 完全分布式加载那条路径根本没设这个属性）。"""
        from tests.unit._module_source import module_source

        import autoflowcfd.core.fr_solver.solver as cpu_single
        import autoflowcfd.core.gpu.distributed.gpu_distributed as gpu_multi
        import autoflowcfd.core.gpu.distributed.gpu_distributed_fully_distributed as gpu_multi_fd
        import autoflowcfd.core.gpu.solver.gpu_solver as gpu_single
        import autoflowcfd.core.mpi.distributed_solver as cpu_mpi
        for mod in (cpu_single, gpu_single, gpu_multi, gpu_multi_fd, cpu_mpi):
            src = module_source(mod)
            assert "resolve_low_mach_precond(" in src, mod.__name__
            assert 'os.environ.get("AFCFD_LOW_MACH_PRECOND")' not in src, (
                f"{mod.__name__} 又自己解析了一遍环境变量")

    @pytest.mark.parametrize("scheme,expected", [
        ("ssp_rk3", True), ("ssp_rk2", True), ("newton_krylov", True),
        ("dual_time", False), ("imex_euler", False), ("forward_euler", False),
    ])
    def test_precond_gated_by_time_scheme(self, scheme, expected):
        s = self._fake_solver(time_scheme=scheme, env=None)
        assert s.low_mach_precond_enabled is expected

    @pytest.mark.parametrize("env,expected", [("0", False), ("1", True)])
    def test_env_override(self, env, expected):
        s = self._fake_solver(time_scheme="ssp_rk3", env=env)
        assert s.low_mach_precond_enabled is expected

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

    def test_order_change_resets_controller(self, monkeypatch):
        """阶数切换必须复位控制器——**行为**测试，不是源码字符串匹配。

        `_interpolate_to_new_order` 在函数体内部 import 真正的插值实现，
        所以可以在调用前把那个模块属性替身掉，于阶数切换路径上只保留
        控制器复位这一件事来验证，不需要 CuPy 或真实网格。
        """
        from autoflowcfd.core.gpu.solver import gpu_solver_order_continuation as oc
        monkeypatch.setattr(oc, "gpu_solver_interpolate_to_new_order",
                            lambda solver, target_p: None)
        from autoflowcfd.core.gpu.solver.gpu_solver import GPUFRSolver
        from autoflowcfd.core.time_integration.adaptive_cfl import AdaptiveCFLController

        ctrl = AdaptiveCFLController(cfl_start=0.1, cfl_max=0.5, ramp_steps=0)
        # 先把 CFL 推离起始值（模拟"前一阶数已经爬起来了"）
        r = 1.0
        for _ in range(400):
            r *= 0.99916
            ctrl.update(r)
        assert ctrl.cfl_number > 0.1, "本测试需要控制器先离开起始值"

        class _S:
            pass
        s = _S()
        s._cfl_controller = ctrl
        GPUFRSolver._interpolate_to_new_order(s, 2)

        assert ctrl.cfl_number == pytest.approx(0.1), (
            "GPU 阶数切换没有复位自适应 CFL 控制器——阶数跳变的残差突变会被"
            "当成解在恶化，把 CFL 一路打到下限")
        assert ctrl._prev_residual == 0.0, "残差基线没有随复位清掉"

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
