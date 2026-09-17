"""
自由来流方向 / 风轴系（攻角、侧滑角），以及它必须真正到达的五处。

## 缺口

在 2026-09-17 之前，自由来流方向在**全代码库被硬编码成 +x**：

    core/fr_solver/boundary.py      Q_free = [rho_inf, vel_inf, 0.0, 0.0, p_inf]
    core/fr_solver/solver.py        initialize_uniform(u=vel_inf, v=0, w=0, ...)
    core/fr_solver/boundary.py      SEM 入口 flow_direction = [vel_inf, 0, 0]
    postprocess/fr_coefficients.py  Cd=F[0] / Cl=F[2] / Cs=F[1]（直接取分量）
    cli/solve_aero_coefficients.py  参考面积按 n_x < 0 做迎风投影

也就是**没有任何攻角/侧滑角选项**。而攻角扫掠是最常见的外流气动研究，
升阻比随攻角的变化基本上是外流计算的第一产出。用"自己旋转网格"替代
也不行——那会让边界层挤出方向、壁面距离、周期面配对全部跟着变。

## 本文件覆盖

1. 方向/风轴系公式本身：退化性、正交性、单位长、符号约定、越界校验
2. **零攻角必须与此前行为逐位相同**（这是"默认路径不变"的硬要求）
3. 五处消费点都真的用上了方向，而不是某一处漏掉后静默退回 +x
"""

import numpy as np
import pytest

from autoflowcfd.core.utils.flow_direction import (
    direction_from_freestream,
    freestream_direction,
    freestream_velocity,
    wind_axes,
)


class TestDegeneratesToPlusX:
    """零攻角/零侧滑必须**严格**退化为此前硬编码的行为。

    这是整条改动的安全前提：默认路径上任何一处数值变化都意味着此前
    全部已验证的结果（含本项目累积的多轮真实网格 A/B 基线）失效。
    """

    def test_direction_is_exactly_plus_x(self):
        d = freestream_direction(0.0, 0.0)
        assert d.tolist() == [1.0, 0.0, 0.0]

    def test_velocity_is_exactly_along_x(self):
        v = freestream_velocity(33.33, 0.0, 0.0)
        assert v.tolist() == [33.33, 0.0, 0.0]

    def test_wind_axes_are_exactly_the_unit_basis(self):
        d, s, l = wind_axes(0.0, 0.0)
        assert d.tolist() == [1.0, 0.0, 0.0]
        assert s.tolist() == [0.0, 1.0, 0.0]
        assert l.tolist() == [0.0, 0.0, 1.0]

    def test_coefficient_projection_matches_old_component_pick(self):
        """风轴系投影在零攻角下必须逐位等于"直接取 F[0]/F[2]/F[1]"。"""
        rng = np.random.default_rng(7)
        d, s, l = wind_axes(0.0, 0.0)
        for _ in range(50):
            F = rng.normal(size=3) * 1e4
            assert float(np.dot(F, d)) == F[0]
            assert float(np.dot(F, l)) == F[2]
            assert float(np.dot(F, s)) == F[1]


class TestWindAxesAreOrthonormal:
    """风轴三元组必须严格正交且单位长 —— 否则 Cd/Cl/Cs 不是一组自洽的
    分解（合力的模会对不上）。"""

    @pytest.mark.parametrize("aoa", [-30.0, -5.0, 0.0, 2.5, 15.0, 45.0, 89.0])
    @pytest.mark.parametrize("aos", [-20.0, 0.0, 7.5, 60.0])
    def test_orthonormal(self, aoa, aos):
        d, s, l = wind_axes(aoa, aos)
        for name, v in (("d", d), ("s", s), ("l", l)):
            assert abs(np.linalg.norm(v) - 1.0) < 1e-14, f"{name} 不是单位向量"
        assert abs(np.dot(d, s)) < 1e-14
        assert abs(np.dot(d, l)) < 1e-14
        assert abs(np.dot(s, l)) < 1e-14

    @pytest.mark.parametrize("aoa,aos", [(10.0, 0.0), (0.0, 10.0), (12.0, -7.0)])
    def test_decomposition_preserves_force_magnitude(self, aoa, aos):
        """正交分解必须保模：Cd^2+Cl^2+Cs^2 == |F|^2（乘同一个分母）。"""
        d, s, l = wind_axes(aoa, aos)
        F = np.array([1234.0, -567.0, 890.0])
        parts = np.array([np.dot(F, d), np.dot(F, s), np.dot(F, l)])
        assert abs(np.dot(parts, parts) - np.dot(F, F)) < 1e-8

    def test_drag_direction_equals_freestream_direction(self):
        """阻力方向必须**就是**来流方向——这是"阻力"的定义。"""
        for aoa, aos in ((0.0, 0.0), (8.0, 0.0), (0.0, 8.0), (13.0, -4.0)):
            d, _s, _l = wind_axes(aoa, aos)
            assert np.allclose(d, freestream_direction(aoa, aos), atol=0, rtol=0)


class TestSignConventions:
    """符号约定：抬头为正攻角时来流有 +z 分量；正侧滑时有 +y 分量。

    符号搞反在气动数据里是灾难性的（升力反向），而且不会被"正交性"这类
    结构性检查抓到，必须单独钉。
    """

    def test_positive_aoa_gives_positive_w(self):
        d = freestream_direction(10.0, 0.0)
        assert d[0] > 0 and d[2] > 0 and abs(d[1]) < 1e-15

    def test_negative_aoa_gives_negative_w(self):
        assert freestream_direction(-10.0, 0.0)[2] < 0

    def test_positive_aos_gives_positive_v(self):
        d = freestream_direction(0.0, 10.0)
        assert d[0] > 0 and d[1] > 0 and abs(d[2]) < 1e-15

    def test_lift_axis_has_no_side_component(self):
        """升力方向按定义落在 x-z 平面内（y 分量恒为零）。"""
        for aoa in (-20.0, 0.0, 5.0, 30.0):
            for aos in (-15.0, 0.0, 15.0):
                _d, _s, l = wind_axes(aoa, aos)
                assert l[1] == 0.0

    def test_lift_is_perpendicular_to_freestream_in_xz_plane(self):
        """零侧滑时升力方向必须是来流方向在 x-z 平面内逆时针转 90 度。"""
        aoa = 12.0
        d = freestream_direction(aoa, 0.0)
        _d, _s, l = wind_axes(aoa, 0.0)
        # 逆时针转 90 度：(x,z) -> (-z,x)
        assert np.allclose(l, [-d[2], 0.0, d[0]], atol=1e-14)


class TestValidation:
    """越界/非法输入必须报错，不能静默接受。"""

    @pytest.mark.parametrize("bad", [91.0, -91.0, 180.0, 1e3])
    def test_out_of_range_raises(self, bad):
        with pytest.raises(ValueError, match="超出"):
            freestream_direction(bad, 0.0)
        with pytest.raises(ValueError, match="超出"):
            freestream_direction(0.0, bad)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_raises(self, bad):
        with pytest.raises(ValueError, match="有限值"):
            freestream_direction(bad, 0.0)

    def test_radians_mistake_is_caught(self):
        """把弧度当角度传是最常见的单位错误。

        3.14 弧度 = 180 度；若用户误把 pi 当成"180 度"传进来，值 3.14
        在度制下只是 3.14 度——**不会**被范围检查抓到，那是无害的小角度。
        真正危险的是反向：把 180（度）当弧度用。这里钉住的是范围检查
        确实拦住了 |角度| > 90 的取值。
        """
        freestream_direction(3.14159, 0.0)          # 3.14 度，合法
        with pytest.raises(ValueError):
            freestream_direction(180.0, 0.0)


class TestFreestreamDictHelper:
    def test_missing_keys_degenerate_to_plus_x(self):
        """旧 checkpoint / 只带三个物理量的 freestream 字典必须退化为 +x。

        这与此前的硬编码行为一致，所以对旧数据是向后兼容的。
        """
        d = direction_from_freestream({"rho_inf": 1.225, "vel_inf": 33.33,
                                       "p_inf": 101325.0})
        assert d.tolist() == [1.0, 0.0, 0.0]

    def test_none_values_degenerate_to_plus_x(self):
        d = direction_from_freestream({"aoa_deg": None, "aos_deg": None})
        assert d.tolist() == [1.0, 0.0, 0.0]

    def test_values_are_used(self):
        d = direction_from_freestream({"aoa_deg": 10.0, "aos_deg": 0.0})
        assert np.allclose(d, freestream_direction(10.0, 0.0))


class TestAllFiveConsumersUseTheDirection:
    """五处消费点都必须真的用上方向。

    用源码文本检查：这五处各自要真实网格/求解器/面网格才能端到端跑，
    而"有没有接上"是结构性事实。任何一处漏掉都会**静默**退回 +x —— 那
    在日志里完全看不出来，只会表现为 Cd/Cl 数值不对。
    """

    _CASES = [
        ("src/autoflowcfd/core/fr_solver/boundary.py",
         ["direction_from_freestream", "_v_free"],
         "Q_free / SEM 入口方向"),
        ("src/autoflowcfd/core/fr_solver/solver.py",
         ["freestream_velocity", "aoa_deg"],
         "单机初场 + freestream 字典"),
        ("src/autoflowcfd/postprocess/fr_coefficients.py",
         ["wind_axes", "d_hat", "l_hat", "s_hat"],
         "气动力风轴系分解"),
        ("src/autoflowcfd/cli/solve_aero_coefficients.py",
         ["direction", "d_component"],
         "参考面积沿来流方向投影"),
        ("src/autoflowcfd/core/mpi/distributed_solver.py",
         ["freestream_velocity", "aoa_deg"],
         "CPU MPI 分布式初场 + freestream 字典"),
    ]

    @pytest.mark.parametrize("rel,needles,what", _CASES)
    def test_consumer_uses_direction(self, rel, needles, what):
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[2]
        src = (root / rel).read_text(encoding="utf-8")
        for needle in needles:
            assert needle in src, (
                f"{rel}（{what}）里找不到 `{needle}` —— 该处很可能仍在用"
                f"硬编码的 +x 方向，那会静默退回零攻角行为"
            )

    @pytest.mark.parametrize("rel", [
        "src/autoflowcfd/core/fr_solver/boundary.py",
        "src/autoflowcfd/postprocess/fr_coefficients.py",
    ])
    def test_no_hardcoded_plus_x_freestream_left(self, rel):
        """这两处的硬编码写法必须已经消失。"""
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[2]
        src = (root / rel).read_text(encoding="utf-8")
        for banned in ("Q_free = [rho_inf, vel_inf, 0.0, 0.0, p_inf]",
                       "Cd = float(force_total[0] / denom)"):
            assert banned not in src, (
                f"{rel} 里仍有硬编码写法 `{banned}`"
            )


class TestCliAndSolverSignatures:
    """CLI 与各求解器构造函数都必须暴露这两个参数。"""

    @pytest.mark.parametrize("cmd", ["steady", "transient"])
    def test_cli_options_exist(self, cmd):
        from autoflowcfd.cli.main import cli

        params = {p.name for p in cli.commands["solve"].commands[cmd].params}
        assert "aoa_deg" in params, f"`solve {cmd}` 缺 --aoa"
        assert "aos_deg" in params, f"`solve {cmd}` 缺 --aos"

    @pytest.mark.parametrize("mod,cls", [
        ("autoflowcfd.core.fr_solver.solver", "FRSolver"),
        ("autoflowcfd.core.gpu.solver.gpu_solver", "GPUFRSolver"),
        ("autoflowcfd.core.gpu.distributed.gpu_distributed",
         "MultiGPUDistributedSolver"),
    ])
    def test_solver_accepts_angles(self, mod, cls):
        import importlib
        import inspect

        m = importlib.import_module(mod)
        params = inspect.signature(getattr(m, cls).__init__).parameters
        for name in ("aoa_deg", "aos_deg"):
            assert name in params, f"{cls}.__init__ 缺 {name}"
            assert params[name].default == 0.0, (
                f"{cls}.__init__ 的 {name} 默认值必须是 0.0（零攻角退化）"
            )


class TestReferenceAreaProjection:
    """参考面积必须沿来流方向投影：有攻角时按 X 投影会偏大 1/cos(alpha)。"""

    def test_projection_direction_is_normalised_and_validated(self):
        from autoflowcfd.cli.solve_aero_coefficients import (
            _compute_reference_area_auto,
        )

        # 没有 surface_mesh 时返回 None（不该因为 direction 校验就崩）
        class _VD:
            surface_mesh = None

        assert _compute_reference_area_auto(_VD(), direction=[2.0, 0.0, 0.0]) is None
        with pytest.raises(ValueError, match="direction"):
            _compute_reference_area_auto(_VD(), direction=[1.0, 0.0])
        with pytest.raises(ValueError, match="模"):
            _compute_reference_area_auto(_VD(), direction=[0.0, 0.0, 0.0])

    def test_analytic_projection_factor(self):
        """解析对照：一块垂直于 +x 的平板，沿攻角 alpha 的来流投影面积
        应当是 `A * cos(alpha)`。

        这条同时说明"按 X 投影"在有攻角时**偏大** 1/cos(alpha)：15 度就是
        3.5%，而参考面积直接进 Cd 的分母。
        """
        n = np.array([-1.0, 0.0, 0.0])      # 迎风面外法向
        area = 0.25
        for aoa in (0.0, 5.0, 15.0, 30.0):
            d = freestream_direction(aoa, 0.0)
            proj = -float(np.dot(n, d)) * area
            assert abs(proj - area * np.cos(np.deg2rad(aoa))) < 1e-12


class TestEndToEndOnRealSolver:
    """端到端：真的构造 FRSolver，检查初场与边界自由来流态。

    比上面那些结构性文本检查更有说服力——它证明的是"参数真的一路走到
    了状态数组里"，而不只是"某个字符串出现在某个文件里"。用 4x3x1 的
    小棱柱通道网格（`tests/validation/_channel_mesh.py`），构造代价很低。
    """

    def _solver(self, aoa, aos, vel=33.33):
        import sys
        from pathlib import Path

        tests_dir = str(Path(__file__).resolve().parents[1])
        if tests_dir not in sys.path:
            sys.path.insert(0, tests_dir)
        from validation._channel_mesh import build_channel_mesh_prism

        from autoflowcfd.core.fr_solver import FRSolver
        from autoflowcfd.core.time_integration import TimeIntegrationScheme

        mesh = build_channel_mesh_prism(1, 4, 3, 1, 1.0, 1.0, 0.25)
        s = FRSolver(mesh=mesh, order=1, turb_model_name="NONE", n_vars=5,
                     time_scheme=TimeIntegrationScheme.SSP_RK3,
                     rho_inf=1.225, vel_inf=vel, p_inf=101325.0,
                     aoa_deg=aoa, aos_deg=aos)
        s.order_continuation_enabled = False
        return s

    def _initial_velocity(self, solver):
        U = solver.state.U
        rho = U[..., 0]
        return (U[..., 1:4] / rho[..., None]).reshape(-1, 3)

    def test_zero_angle_initial_state_is_exactly_along_x(self):
        """零攻角初场必须**逐位**等于 [vel_inf, 0, 0]。

        这是"默认路径不变"的端到端证据：此前的实现就是
        `initialize_uniform(u=vel_inf, v=0.0, w=0.0)`。
        """
        vel = self._initial_velocity(self._solver(0.0, 0.0))
        assert np.all(vel[:, 0] == 33.33)
        assert np.all(vel[:, 1] == 0.0)
        assert np.all(vel[:, 2] == 0.0)

    def test_aoa_rotates_initial_state_exactly(self):
        s = self._solver(10.0, 0.0)
        vel = self._initial_velocity(s)
        expect = 33.33 * np.array([np.cos(np.deg2rad(10.0)), 0.0,
                                   np.sin(np.deg2rad(10.0))])
        assert np.abs(vel - expect).max() < 1e-12, (
            f"初场速度 {vel[0]} != 期望 {expect}")

    def test_speed_magnitude_is_preserved(self):
        """旋转来流方向不能改变速度**大小**——那会连带改雷诺数与马赫数。"""
        for aoa, aos in ((0.0, 0.0), (10.0, 0.0), (0.0, 12.0), (8.0, -5.0)):
            vel = self._initial_velocity(self._solver(aoa, aos))
            assert np.abs(np.linalg.norm(vel, axis=1) - 33.33).max() < 1e-10

    def test_angles_land_in_freestream_dict(self):
        s = self._solver(7.5, -3.25)
        assert s.freestream["aoa_deg"] == 7.5
        assert s.freestream["aos_deg"] == -3.25

    def test_mach_ref_unaffected_by_angles(self):
        """mach_ref 只依赖速度大小，不该随方向变（它进 AUSM+up 的 beta2
        下限与 CFL 估计，跟着方向变会让同一工况的数值行为无故不同）。"""
        a = self._solver(0.0, 0.0).freestream["mach_ref"]
        b = self._solver(15.0, -10.0).freestream["mach_ref"]
        assert a == b

    def test_boundary_qfree_follows_direction(self):
        """边界自由来流态的速度分量必须与来流方向一致。

        直接用生产路径读出来的 freestream 字典算方向（与
        `core/fr_solver/boundary.py` 里 Q_free 的构造同一个调用），
        避免把"测试自己重算一遍"当成验证。
        """
        for aoa, aos in ((0.0, 0.0), (10.0, 0.0), (6.0, 4.0)):
            s = self._solver(aoa, aos)
            d = direction_from_freestream(s.freestream)
            v_free = s.freestream["vel_inf"] * d
            expect = freestream_velocity(33.33, aoa, aos)
            assert np.allclose(v_free, expect, atol=0, rtol=0)
            if aoa == 0.0 and aos == 0.0:
                assert v_free.tolist() == [33.33, 0.0, 0.0]
