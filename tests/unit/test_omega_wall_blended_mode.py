"""Menter 混合式 omega 壁面目标值（B-6，2026-09-15 联网核实文献后实现）。

## 两档公式

    amplified（默认，既有行为逐位不变）
        omega_wall = 10 * 6*nu/(beta1*d1^2)

    blended（Menter 二项混合式）
        omega_vis  = 6*nu/(beta1*d1^2)
        omega_log  = sqrt(k)/(C_mu^0.25 * kappa * d1)
        omega_wall = sqrt(omega_vis^2 + omega_log^2)

## 文献依据

与 OpenFOAM `omegaWallFunction` 的实现逐项一致：
`omegaVis = 6*nuw/(beta1_*sqr(y))`、`omegaLog = sqrt(k)/(Cmu25*kappa_*y)`、
`omega = sqrt(sqr(omegaVis) + sqr(omegaLog))`，`Cmu25 = pow025(Cmu)`。
常数取该实现的默认值：`beta1_ = 0.075`（omegaWallFunction 构造函数）、
`Cmu = 0.09`、`kappa = 0.41`（nutWallFunction 默认值，omegaWallFunction
经 nutw 取用）。较新版本改用分数权重线性混合，本项目实现的是经典二项式
——Menter 的原始形式，且两个极限下都渐近精确。

## 为什么值得有这一档

`amplified` 里那个 10 倍是 Menter 为**有限体积、近壁不解析**的情形设计的
数值手段（把 omega 抬得足够高以在粗近壁网格上强制出正确渐近行为），不是
物理值。混合式没有任何这类自由因子。注意**壁面解析**（低 Re）网格上粘性
支占绝对主导，于是两档的实际差别基本就是那个 10 倍。

## 默认值没有改

这是湍流模型的物理改动。本项目在 omega 壁面处理上已有两次"数学上更对
但被真实数据证伪"的先例（显式 SIPG 罚项、点隐式动态松弛系数），所以只
提供开关与判据，改默认值必须有真实长程数据。
"""

import os

import numpy as np
import pytest

from autoflowcfd.core.turbulence.transport import (
    _OMEGA_WALL_CMU,
    _OMEGA_WALL_KAPPA,
    _OMEGA_WALL_MODES,
    resolve_omega_wall_mode,
)


class _Env:
    """环境变量的 with-块辅助（测试之间必须互不影响）。"""

    def __init__(self, **kw):
        self.kw = kw
        self.old = {}

    def __enter__(self):
        for k, v in self.kw.items():
            self.old[k] = os.environ.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return self

    def __exit__(self, *a):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class TestSwitchSemantics:
    def test_default_is_amplified(self):
        """默认必须是既有行为。"""
        with _Env(AFCFD_OMEGA_WALL_MODE=None):
            assert resolve_omega_wall_mode() == "amplified"

    @pytest.mark.parametrize("mode", list(_OMEGA_WALL_MODES))
    def test_valid_modes_accepted(self, mode):
        with _Env(AFCFD_OMEGA_WALL_MODE=mode):
            assert resolve_omega_wall_mode() == mode

    def test_case_insensitive(self):
        with _Env(AFCFD_OMEGA_WALL_MODE="BLENDED"):
            assert resolve_omega_wall_mode() == "blended"

    @pytest.mark.parametrize("bad", ["menter", "", "amplify", "blend"])
    def test_invalid_rejected_not_silently_defaulted(self, bad):
        """非法取值必须显式报错——与本项目其余开关同一条约定。"""
        with _Env(AFCFD_OMEGA_WALL_MODE=bad):
            with pytest.raises(ValueError, match="AFCFD_OMEGA_WALL_MODE"):
                resolve_omega_wall_mode()


class TestConstantsMatchTheLiterature:
    def test_cmu_and_kappa(self):
        """OpenFOAM nutWallFunction 的默认常数。"""
        assert _OMEGA_WALL_CMU == pytest.approx(0.09)
        assert _OMEGA_WALL_KAPPA == pytest.approx(0.41)

    def test_cmu25_value(self):
        """`Cmu25 = pow025(Cmu)` = 0.09^0.25，混合式对数支的分母因子。"""
        assert _OMEGA_WALL_CMU ** 0.25 == pytest.approx(0.5477225575, rel=1e-9)


class TestFormulaBranches:
    """直接验证两档公式的数值，参照值手算。"""

    BETA1 = 0.075

    def _formula(self, mode, nu, d1, k):
        with _Env(AFCFD_OMEGA_WALL_MODE=mode):
            from autoflowcfd.core.turbulence.transport import _omega_wall_formula
            import types
            n = 1
            solver = types.SimpleNamespace(
                turb_model=types.SimpleNamespace(
                    k_field=np.full((n, 8), k, dtype=float)),
                mesh=types.SimpleNamespace(n_prism_cells=n),
            )
            return _omega_wall_formula(
                solver, np.zeros(n, dtype=np.int64),
                np.full(n, nu), np.full(n, d1), self.BETA1)[0]

    def test_amplified_matches_hand_computed(self):
        nu, d1 = 1.5e-5, 2.0e-4
        got = self._formula("amplified", nu, d1, 0.1)
        expect = 10.0 * 6.0 * nu / (self.BETA1 * d1 ** 2)
        assert got == pytest.approx(expect, rel=1e-12)

    def test_blended_matches_hand_computed(self):
        nu, d1, k = 1.5e-5, 2.0e-4, 0.1
        got = self._formula("blended", nu, d1, k)
        o_vis = 6.0 * nu / (self.BETA1 * d1 ** 2)
        o_log = np.sqrt(k) / (0.09 ** 0.25 * 0.41 * d1)
        assert got == pytest.approx(np.hypot(o_vis, o_log), rel=1e-12)

    def test_amplified_is_independent_of_k(self):
        """放大式不含 k——换 k 结果必须逐位不变。"""
        a = self._formula("amplified", 1.5e-5, 2.0e-4, 1e-6)
        b = self._formula("amplified", 1.5e-5, 2.0e-4, 1e3)
        assert a == b

    def test_blended_depends_on_k(self):
        a = self._formula("blended", 1.5e-5, 2.0e-4, 1e-8)
        b = self._formula("blended", 1.5e-5, 2.0e-4, 1e2)
        assert b > a * 1.5

    def test_viscous_limit_blended_approaches_vis_branch(self):
        """k -> 0（纯粘性子层）时混合式必须退化到粘性支本身，
        即恰好是 amplified 档的 1/10。"""
        nu, d1 = 1.5e-5, 2.0e-4
        blended = self._formula("blended", nu, d1, 0.0)
        amplified = self._formula("amplified", nu, d1, 0.0)
        assert blended == pytest.approx(amplified / 10.0, rel=1e-12)

    def test_log_limit_blended_approaches_log_branch(self):
        """d1 较大、k 较大时对数支主导，混合值应逼近 omega_log。"""
        nu, d1, k = 1.5e-5, 5.0e-2, 1.0
        got = self._formula("blended", nu, d1, k)
        o_log = np.sqrt(k) / (0.09 ** 0.25 * 0.41 * d1)
        assert got == pytest.approx(o_log, rel=1e-3)

    def test_blended_is_never_below_either_branch(self):
        """二项混合是两支的欧氏和，必然 >= 各支。"""
        for nu, d1, k in ((1.5e-5, 1e-4, 1e-3), (2e-5, 1e-2, 0.5),
                          (1e-5, 5e-3, 5.0)):
            got = self._formula("blended", nu, d1, k)
            o_vis = 6.0 * nu / (self.BETA1 * d1 ** 2)
            o_log = np.sqrt(k) / (0.09 ** 0.25 * 0.41 * d1)
            assert got >= o_vis - 1e-30
            assert got >= o_log - 1e-30

    def test_blended_without_k_field_raises(self):
        """没有 k 场时必须报错，不能静默退回 amplified。"""
        import types
        with _Env(AFCFD_OMEGA_WALL_MODE="blended"):
            from autoflowcfd.core.turbulence.transport import _omega_wall_formula
            solver = types.SimpleNamespace(
                turb_model=types.SimpleNamespace(k_field=None),
                mesh=types.SimpleNamespace(n_prism_cells=1))
            with pytest.raises(RuntimeError, match="k_field"):
                _omega_wall_formula(solver, np.zeros(1, dtype=np.int64),
                                    np.full(1, 1.5e-5), np.full(1, 1e-4),
                                    self.BETA1)


class TestTwoDimensionsAreIndependent:
    """本档与 `AFCFD_OMEGA_WALL_D1`（长度尺度口径）是两个独立维度，
    而且两者的偏差会**相乘**。"""

    def test_min_convention_overestimate_compounds_with_amplification(self):
        """order=1 的 `min` 口径已高估 5.60 倍，叠加 10 倍放大约 56 倍。"""
        from autoflowcfd.fr.quadrature_points import gauss_legendre
        pts, _ = gauss_legendre(2)          # order=1
        d = (np.asarray(pts) + 1.0) / 2.0
        conv_factor = (0.5 / d.min()) ** 2
        assert conv_factor == pytest.approx(5.60, rel=2e-3)
        assert conv_factor * 10.0 == pytest.approx(56.0, rel=2e-3)

    def test_both_switches_exist_and_are_separate(self):
        import inspect

        # 必须 inspect **子模块** omega_wall，不是包 `__init__`：
        # `transport` 2026-09-24 拆成了子包，`inspect.getsource(transport)`
        # 只会返回 `__init__.py`，那两个字符串都不在里面 —— 一条会直接
        # 失败、另一条（下面那个 `not in`）会**静默通过**，比没有测试更糟。
        from autoflowcfd.core.turbulence.transport import omega_wall
        src = inspect.getsource(omega_wall)
        assert "AFCFD_OMEGA_WALL_D1" in src
        assert "AFCFD_OMEGA_WALL_MODE" in src


class TestGpuMirrorIsWired:
    """GPU 侧必须读同一个开关、用同一套常数——否则同一个环境变量在两个
    后端意味着不同的东西（本项目 2026-09-15 审计的第 7 类缺陷）。"""

    def test_gpu_reads_the_same_switch_and_constants(self):
        from tests.unit._module_source import module_source

        from autoflowcfd.core.gpu.turbulence import gpu_scalar_transport as gst
        src = module_source(gst)
        assert "resolve_omega_wall_mode()" in src
        assert "_OMEGA_WALL_CMU" in src
        assert "_OMEGA_WALL_KAPPA" in src
        # 旧的硬编码 60.0 常数必须已经消失
        assert "60.0 * nu_owner" not in src

    def test_gpu_takes_k_field_parameter(self):
        import inspect

        from autoflowcfd.core.gpu.turbulence.gpu_scalar_transport import (
            compute_omega_wall_target_gpu,
        )
        assert "turb_k_field" in inspect.signature(
            compute_omega_wall_target_gpu).parameters

    def test_cpu_hardcoded_60_is_gone(self):
        import inspect

        # 同上：必须 inspect 子模块，否则这条 `not in` 会静默通过
        # （字符串只是搬到了 omega_wall.py，并不是真的不存在）。
        from autoflowcfd.core.turbulence.transport import omega_wall
        src = inspect.getsource(omega_wall)
        assert "60.0 * nu_owner" not in src


class TestTransientTimeAccuracyWarning:
    """`solve transient` 的默认组合是 `--time-method rk3` +
    `--turbulence-model ddes`，而 rk3/imex 下 `step()` **忽略** --dt、按
    逐单元局部 CFL 步长推进（见 core/fr_solver/step.py::step 的 dt 语义
    一节），各单元推进的物理时间并不相同。

    DES/LES/WMLES 的全部意义就在于时间上解析湍流结构，所以这个默认组合
    跑出来的场不能当作非稳态数据解读——而此前完全没有提示。不直接拒绝是
    因为 rk3 + DES 作为"先把流场吹起来"的快速冒烟有实际用途（本项目既有
    的 DDES/LES CLI 端到端验证就是这么跑的）。
    """

    @pytest.mark.parametrize("model", ["ddes", "iddes", "les", "wmles"])
    @pytest.mark.parametrize("method", ["rk3", "imex"])
    def test_warning_source_covers_every_time_resolved_model(self, model, method):
        import inspect

        from autoflowcfd.cli.solve import transient as stc
        src = inspect.getsource(stc)
        assert '_TIME_RESOLVED_MODELS' in src
        assert model in src
        assert '时间精度警告' in src
        assert 'dual-time' in src

    def test_sst_is_not_warned(self):
        """SST 是 RANS 模型，不在时间解析模型清单里。"""
        import inspect

        from autoflowcfd.cli.solve import transient as stc
        src = inspect.getsource(stc)
        i = src.index('_TIME_RESOLVED_MODELS = (')
        decl = src[i:src.index(')', i)]
        assert 'sst' not in decl

    def test_help_text_states_only_dual_time_is_time_accurate(self):
        from click.testing import CliRunner

        from autoflowcfd.cli.solve.transient import transient
        out = CliRunner().invoke(transient, ['--help']).output
        flat = ' '.join(out.split())
        assert '只有 dual-time 是时间精确的' in flat


class TestArtificialViscosityIsInertAtP1:
    """`--artificial-viscosity` 在 order=1（生产阶数）上是精确的无操作。

    Persson-Peraire 的 ramp 判据是 `s0 = -4*log10(order)`，order=1 时 s0=0，
    触发门限成了"顶模态能量占全胞 >= 10%"（`S_e >= 10^(s0-kappa) = 0.1`）。
    而 P1 的"顶模态"**就是全部非常数模态**，已解析的物理与混叠无法区分。

    实测（plate_demo 363,392 单元 ANSA 网格，order=1，固定 CFL 0.03，其余
    参数逐项相同的 A/B）：开与不开该开关，**前 51 步的残差与 Cd 逐字符
    完全相同**。

    这一点要紧，因为项目对退化单元残差放大的既定修复路线是"网格质量门
    + 耗散"，而耗散那一半在生产阶数上不可用。
    """

    @pytest.mark.parametrize("order,expect_threshold", [
        (1, 1.0e-1),      # s0=0      -> 10^(0-1)
        (2, 6.25e-3),     # s0=-1.204 -> 10^(-2.204)
        (3, 1.235e-3),    # s0=-1.909 -> 10^(-2.909)
    ])
    def test_sensor_threshold_by_order(self, order, expect_threshold):
        from autoflowcfd.core.fr_operators.artificial_viscosity import SENSOR_KAPPA
        s0 = -4.0 * np.log10(max(order, 1))
        assert 10 ** (s0 - SENSOR_KAPPA) == pytest.approx(expect_threshold, rel=2e-3)

    def test_p1_threshold_demands_ten_percent_of_total_energy(self):
        """P1 的门限要求顶模态占全胞能量 10%——而顶模态就是全部非常数
        模态，所以这等于要求"胞内变化占 10% 以上"，正常解析的流场达不到。"""
        from autoflowcfd.core.fr_operators.artificial_viscosity import SENSOR_KAPPA
        s0_p1 = -4.0 * np.log10(1)
        assert s0_p1 == 0.0
        assert 10 ** (s0_p1 - SENSOR_KAPPA) == pytest.approx(0.1)

    def test_enabling_at_p1_warns_explicitly(self):
        """不接受静默无操作：用户以为打开了一层保护，实际什么都没发生。"""
        import inspect

        from autoflowcfd.core.fr_solver import solver as solver_mod
        from tests.unit._module_source import module_source
        src = module_source(solver_mod)
        assert "artificial_viscosity_enabled and order <= 1" in src
        assert "warnings.warn" in src
        i = src.index("artificial_viscosity_enabled and order <= 1")
        ctx = src[i:i + 1200]
        assert "无操作" in ctx
        assert "order>=2" in ctx

    def test_startup_log_shows_both_switches(self):
        """启动日志必须显示人工粘性与滤波档——否则"这份日志是哪个配置跑
        出来的"只能事后考古。"""
        import inspect

        from autoflowcfd.core.fr_solver import solver as solver_mod
        from tests.unit._module_source import module_source
        src = module_source(solver_mod)
        assert "Artificial viscosity:" in src
        assert "Modal filter mode:" in src
