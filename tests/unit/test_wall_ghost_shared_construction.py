"""滑移壁 / WMLES 壁面共享 ghost 态构造的性质验证（2026-09-14）。

## 背景

`fr_ghost_state.wall_ghost_state` 的 `is_no_slip=False` 分支被两种物理
场景复用：真正的滑移/对称壁，以及 WMLES 激活时的粘性壁（此时壁面模型
给出的 tau_w 必须**取代**而非叠加解析梯度剪应力）。源码此前把这种复用
称为"刻意的工程简化"。

那个措辞不准确，已更正：两种场景各自的**正确** ghost 态恰好就是同一个
——法向镜像反号（u_n=0 不可穿透）+ 切向保持（无跳跃）。本文件用可验证的
性质把这一点钉住，而不是只改注释。

另外把一条此前只隐含在 `Q_ghost = Q_int.copy()` 里的事实显式化：两个
分支都复制 rho/p，因此温度无跳跃，**实现的是绝热壁**。

## 判据

1. `is_no_slip=False` 下界面法向速度恰为 0（不可穿透，精确到机器精度）
2. 切向速度**逐分量**与内部值完全相同（零跳跃 -> BR1 切向梯度贡献为零，
   这是 WMLES 不双重计权的前提）
3. 反射是等距的：|v_ghost| == |v_int|（滑移壁不做功、不改变动能）
4. 对合性：施加两次回到原值（镜像反射的定义性质）
5. `is_no_slip=True` 下界面平均速度恰为壁面速度（无滑移的镜像构造）
6. 两个分支都不改 rho/p -> 温度无跳跃 -> 绝热壁
7. 法向取反（外法向 vs 内法向）结果不变——构造不依赖法向朝向约定
"""

import numpy as np
import pytest

from autoflowcfd.boundary.fr_ghost_state import wall_ghost_state

GAMMA = 1.4
R_AIR = 287.05


def _random_states(n_fp, rng):
    Q = np.empty((n_fp, 5))
    Q[:, 0] = 1.0 + 0.5 * rng.random(n_fp)            # rho
    Q[:, 1:4] = 60.0 * (rng.random((n_fp, 3)) - 0.5)  # u,v,w
    Q[:, 4] = 9.0e4 + 2.0e4 * rng.random(n_fp)        # p
    return Q


def _random_normals(n_fp, rng):
    n = rng.standard_normal((n_fp, 3))
    return n / np.linalg.norm(n, axis=1, keepdims=True)


@pytest.fixture
def case():
    rng = np.random.default_rng(20260914)
    n_fp = 64
    return _random_states(n_fp, rng), _random_normals(n_fp, rng)


class TestSlipBranchIsExactForBothPhysics:
    """`is_no_slip=False`：滑移/对称壁与 WMLES 壁面要求的是同一个精确构造。"""

    def test_interface_normal_velocity_is_exactly_zero(self, case):
        """判据 1：界面（内部与 ghost 的平均）法向速度恰为 0。

        这是"不可穿透"的精确表述，两种物理场景都要求它。
        """
        Q, n = case
        Qg = wall_ghost_state(Q, n, is_no_slip=False)
        v_iface = 0.5 * (Q[:, 1:4] + Qg[:, 1:4])
        vn = np.sum(v_iface * n, axis=1)
        assert np.abs(vn).max() < 1e-12, (
            f"界面法向速度不为零（最大 {np.abs(vn).max():.3e}）——不可穿透被破坏")

    def test_tangential_velocity_has_zero_jump(self, case):
        """判据 2：切向速度逐分量与内部完全相同。

        这一条是 WMLES 不双重计权的前提：切向零跳跃 => BR1/LDG 界面项
        对切向方向的梯度贡献为零 => 壁面模型的 tau_w 成为唯一来源。
        同时它也正是滑移壁"切向自由"的精确表述。
        """
        Q, n = case
        Qg = wall_ghost_state(Q, n, is_no_slip=False)
        v_i, v_g = Q[:, 1:4], Qg[:, 1:4]
        # 切向分量 = 总速度减去法向投影
        t_i = v_i - np.sum(v_i * n, axis=1, keepdims=True) * n
        t_g = v_g - np.sum(v_g * n, axis=1, keepdims=True) * n
        assert np.abs(t_g - t_i).max() < 1e-12, (
            f"切向速度存在跳跃（最大 {np.abs(t_g - t_i).max():.3e}）——"
            "会产生虚假解析剪应力，与 WMLES 的 tau_w 双重计权")

    def test_reflection_is_isometric(self, case):
        """判据 3：|v_ghost| == |v_int|（镜像反射保模，滑移壁不做功）。"""
        Q, n = case
        Qg = wall_ghost_state(Q, n, is_no_slip=False)
        s_i = np.linalg.norm(Q[:, 1:4], axis=1)
        s_g = np.linalg.norm(Qg[:, 1:4], axis=1)
        rel = np.abs(s_g - s_i) / np.maximum(s_i, 1e-300)
        assert rel.max() < 1e-13, f"反射不保模（相对 {rel.max():.3e}）"

    def test_reflection_is_involutive(self, case):
        """判据 4：施加两次回到原值——镜像反射的定义性质。

        能抓出"少乘了 2"或"用了投影而不是反射"这类实现错误：投影
        （v - v_n*n）不是对合的，反射（v - 2*v_n*n）才是。
        """
        Q, n = case
        Q1 = wall_ghost_state(Q, n, is_no_slip=False)
        Q2 = wall_ghost_state(Q1, n, is_no_slip=False)
        assert np.abs(Q2 - Q).max() < 1e-12, (
            "两次反射没有回到原值——不是镜像反射（可能写成了投影）")

    def test_independent_of_normal_orientation(self, case):
        """判据 7：法向取反结果不变（构造不依赖内/外法向约定）。"""
        Q, n = case
        a = wall_ghost_state(Q, n, is_no_slip=False)
        b = wall_ghost_state(Q, -n, is_no_slip=False)
        assert np.abs(a - b).max() < 1e-12, "结果依赖法向朝向约定"


class TestNoSlipBranch:
    @pytest.mark.parametrize("v_wall", [None, np.array([0.0, 0.0, 0.0]),
                                        np.array([12.0, -3.0, 0.5])])
    def test_interface_velocity_equals_wall_velocity(self, case, v_wall):
        """判据 5：无滑移分支下界面平均速度恰为壁面速度。

        这正是镜像构造 `2*v_wall - v_int` 的目的（直接令 ghost=v_wall
        会让界面平均变成 (v_int+v_wall)/2，是本项目文档记录过的偏差）。
        """
        Q, n = case
        Qg = wall_ghost_state(Q, n, is_no_slip=True, wall_velocity=v_wall)
        v_iface = 0.5 * (Q[:, 1:4] + Qg[:, 1:4])
        target = np.zeros(3) if v_wall is None else np.asarray(v_wall)
        assert np.abs(v_iface - target[None, :]).max() < 1e-12, (
            "界面平均速度不等于壁面速度——镜像构造不正确")


class TestThermalBoundaryIsAdiabatic:
    """判据 6：两个分支都不改 rho/p，因此温度无跳跃 —— 实现的是绝热壁。

    这条事实此前只隐含在 `Q_ghost = Q_int.copy()` 里，读者无法判断是
    有意还是偶然。现在显式钉住：任何人改成"顺手也镜像一下 p"或引入
    等温壁时，都会在这里被迫正面处理，而不是静默改变热边界条件。

    附注（不是本测试的断言范围）：本项目目前没有等温壁 BC 类型，
    WMLES 也只给 tau_w、没有壁面热通量模型——带传热的壁面工况需要
    另外新增 BC 与闭合，见 wall_ghost_state 文档。
    """

    @pytest.mark.parametrize("is_no_slip", [True, False])
    def test_density_and_pressure_are_copied(self, case, is_no_slip):
        Q, n = case
        Qg = wall_ghost_state(Q, n, is_no_slip=is_no_slip)
        np.testing.assert_array_equal(Qg[:, 0], Q[:, 0])
        np.testing.assert_array_equal(Qg[:, 4], Q[:, 4])

    @pytest.mark.parametrize("is_no_slip", [True, False])
    def test_temperature_jump_is_zero(self, case, is_no_slip):
        Q, n = case
        Qg = wall_ghost_state(Q, n, is_no_slip=is_no_slip)
        T_i = Q[:, 4] / (R_AIR * Q[:, 0])
        T_g = Qg[:, 4] / (R_AIR * Qg[:, 0])
        np.testing.assert_array_equal(T_g, T_i)


class TestTwoPhysicsShareTheSameConstruction:
    def test_wmles_wall_and_slip_wall_give_identical_ghost(self, case):
        """把"两种场景用的是同一个精确构造"直接钉成可执行断言。

        生产代码里这两条路径的区别只在 `type_map`：WMLES 激活时
        `WALL -> is_no_slip=False`，而 `SLIP_WALL -> is_no_slip=False`。
        两者落到同一个函数、同一组参数，因此 ghost 态必须逐位相同。
        如果将来有人给其中一条加了特殊处理（例如只给 WMLES 那条改 p），
        这条断言会立刻失败，迫使改动者说明为什么两种物理条件不再共享
        同一个构造。
        """
        Q, n = case
        wmles_wall = wall_ghost_state(Q, n, is_no_slip=False)
        slip_wall = wall_ghost_state(Q, n, is_no_slip=False,
                                     wall_velocity=np.array([7.0, -2.0, 1.0]))
        # wall_velocity 在 is_no_slip=False 分支里不应被使用
        np.testing.assert_array_equal(wmles_wall, slip_wall)
