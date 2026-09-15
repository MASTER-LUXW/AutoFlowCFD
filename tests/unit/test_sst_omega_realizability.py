"""真实 bug 回归测试（2026-09-07）：SST 模型 omega realizability 下限
（`_omega_realizability_min`）此前只依赖 `0.1*max(S_mag)`，P0 阶段
`grad_vel`/`S_mag` 恒为零（P0 是分片常数场，多项式导数恒为零，见
`fr_solver/turbulence.py` 模块文档"为什么P0阶段摩擦阻力算不出来"一节
同一个事实），这条本该防止 omega 衰减过度的 realizability 下限在整个
P0 阶段因此完全失效（恒为 0），一旦某个 SP 的显式积分把 omega 打到
`apply_positivity_limiter` 的裸正性下限 1e-12（只防负值，不是物理意义
上的下限），P0 阶段没有任何机制能让它恢复——是 cube_demo 791,492 单元
真实网格 Order Continuation P0->P1 升阶后 k_mean 持续增长这次排查的
真正根因（用历史 checkpoint 数据 iter_001900~002300 直接定位到具体
单元 omega 塌陷到 1e-12 而 k 未同步跌落，升阶后巨大的 k/omega 比值被
nu_t 湍流粘性比上限钳到 ~1.47——不是零——形成异常畅通的扩散通道持续
从周围抽取 k，最终把这些单元顶到 k_max 安全上限）。

修复：`_omega_realizability_min = max(0.1*max(S_mag), 0.1*omega_inf)`，
S_mag 恒零时（P0）由来流 omega_inf 的保守比例兜底，不再塌陷到 0。
"""

import numpy as np
import pytest

from autoflowcfd.core.turbulence.sst import SSTModelFR


def _build_source_term_inputs(n_cells, n_sps, grad_U_value=0.0):
    """构造 compute_source_terms 所需的最小合法输入集。"""
    Q = np.zeros((n_cells, n_sps, 5))
    Q[..., 0] = 1.225  # rho
    Q[..., 1] = 30.0   # u
    Q[..., 4] = 101325.0  # p
    grad_U = np.full((n_cells, n_sps, 3, 3), grad_U_value)
    d_wall = np.full((n_cells, n_sps), 0.01)
    grad_k = np.zeros((n_cells, n_sps, 3))
    grad_omega = np.zeros((n_cells, n_sps, 3))
    return Q, grad_U, d_wall, grad_k, grad_omega


class TestOmegaRealizabilityMinAtP0:
    """P0 阶段（grad_U 恒零，S_mag 恒零）：realizability 下限不应该
    塌陷到 0，必须由 omega_inf 的保守比例兜底。"""

    def test_realizability_min_nonzero_when_strain_rate_is_zero(self):
        n_cells, n_sps = 2, 1
        omega_inf = 2268.1  # 与 cube_demo 真实来流条件同量级
        model = SSTModelFR(n_cells, n_sps, k_inf=0.1666, omega_inf=omega_inf)
        Q, grad_U, d_wall, grad_k, grad_omega = _build_source_term_inputs(
            n_cells, n_sps, grad_U_value=0.0,
        )

        model.compute_source_terms(Q, grad_U, d_wall, mu=1.8e-5,
                                    grad_k=grad_k, grad_omega=grad_omega)

        # 修复前这里恒为 0（0.1*max(S_mag)=0.1*0=0）——真实 bug 的直接
        # 数值证据。
        # 2026-09-15 起下限是**逐点**数组（见 sst.py 该处第二次 bug 修复），
        # 但 P0 下 S_mag 恒为钳位值 1e-10，0.1*S_mag=1e-11 远小于
        # 0.1*omega_inf，所以每一点都恰好等于 0.1*omega_inf——与修复前的
        # 标量取值逐位相同，这正是那次改动的安全保证。
        rmin = np.asarray(model._omega_realizability_min)
        assert rmin.shape == (n_cells, n_sps)
        np.testing.assert_allclose(rmin, 0.1 * omega_inf, rtol=1e-12)
        assert np.all(rmin > 0)

    def test_positivity_limiter_recovers_collapsed_omega_at_p0(self):
        """真实复现场景：某个 SP 的 omega 被(模拟的)显式积分打到裸正性
        下限附近（1e-12 量级），P0 阶段（S_mag=0）调用
        apply_positivity_limiter 后必须被 omega_inf 兜底的 realizability
        下限拉回物理合理范围，而不是继续停留在 1e-12。"""
        n_cells, n_sps = 2, 1
        omega_inf = 2268.1
        model = SSTModelFR(n_cells, n_sps, k_inf=0.1666, omega_inf=omega_inf)
        Q, grad_U, d_wall, grad_k, grad_omega = _build_source_term_inputs(
            n_cells, n_sps, grad_U_value=0.0,
        )
        model.compute_source_terms(Q, grad_U, d_wall, mu=1.8e-5,
                                    grad_k=grad_k, grad_omega=grad_omega)

        # 模拟真实复现：某个单元的 k 保持正常、omega 塌陷到裸正性下限附近
        # （真实历史数据：cell=81600 在 iter~2000 时 k=0.0514, omega=1e-12）。
        model.k_field[0, 0] = 0.0514
        model.omega_field[0, 0] = 1e-12
        model.k_field[1, 0] = 0.1666  # 另一个单元保持正常（对照组）
        model.omega_field[1, 0] = omega_inf

        model.apply_positivity_limiter()

        # 塌陷单元必须被拉回到 realizability 下限（0.1*omega_inf），
        # 不能继续停留在裸正性下限 1e-12——否则 k/omega 比值依然是
        # 天文数字，nu_t 湍流粘性比钳制器依然会被触发。
        assert model.omega_field[0, 0] == pytest.approx(0.1 * omega_inf)
        assert model.omega_field[0, 0] > 1e-6  # 明确排除"还停留在裸正性下限附近"
        # 正常单元不受影响。
        assert model.omega_field[1, 0] == pytest.approx(omega_inf)

    def test_p1_with_real_strain_rate_still_uses_larger_bound(self):
        """S_mag 非零且其对应的下限比 0.1*omega_inf 更大时（典型 P1+
        近壁高剪切场景），不应该被 omega_inf 这个新增下限"拉低"——两者
        取更大值，不能是新增下限意外覆盖掉本该更严格的 S_mag 下限。

        2026-09-15 起同时验证**局部性**：只有高应变的那一点下限被抬高，
        其余点保持 0.1*omega_inf。修复前 `0.1*max(S_mag)` 是全域标量，
        一个尖峰会把整场 omega 一起抬起来——真实网格上那正是发散的起点。
        所以这里用 4 个单元而不是 1 个，否则局部性无从检验。
        """
        n_cells, n_sps = 4, 1
        omega_inf = 100.0  # 故意设小，让 S_mag 下限更大
        model = SSTModelFR(n_cells, n_sps, k_inf=0.1, omega_inf=omega_inf)
        # grad_U 对角项非零 -> S_mag 非零且较大
        Q, grad_U, d_wall, grad_k, grad_omega = _build_source_term_inputs(
            n_cells, n_sps, grad_U_value=0.0,
        )
        grad_U[0, 0, 0, 0] = 1e5  # du/dx 很大 -> S_mag 很大

        model.compute_source_terms(Q, grad_U, d_wall, mu=1.8e-5,
                                    grad_k=grad_k, grad_omega=grad_omega)

        rmin = np.asarray(model._omega_realizability_min)
        # 高应变的那一点下限被抬高
        assert rmin[0, 0] > 0.1 * omega_inf
        # **其余点不受影响**——这是 2026-09-15 修复的核心：下限是逐点的
        # realizability 约束，不是"全场耦合到单个最差点"。修复前
        # `0.1*max(S_mag)` 是标量，这里每一个点都会被抬到同一个值。
        assert n_cells * n_sps > 1, "本判据需要至少两个点才有意义"
        others = np.ones(rmin.shape, dtype=bool)
        others[0, 0] = False
        np.testing.assert_allclose(rmin[others], 0.1 * omega_inf, rtol=1e-12)


class TestKFloorAtP0:
    """真实 bug 回归测试（2026-09-11）：cube_demo 791,492 单元真实网格
    P0 阶段长程续算（iter 1600→2500）里，撞到裸正性下限 1e-12 的单元数
    2→9→39→62 持续扩散——P0 架构上生成项 P_k 恒为零（同一个 grad_vel
    恒零的事实，见 omega realizability 下限文档），k 只能靠 point-implicit
    阻尼过的耗散项+输运衰减，足够长的 P0 停留时间下任何单元都可能被
    压到裸下限，且不限于对流补给弱的回流区（真实撞底单元里有局部速度
    接近自由来流 27~30 m/s 的、非回流区单元）。

    修复：apply_positivity_limiter 新增 k 的来流下限 `max(1e-12,
    1e-3*k_inf)`，与 omega realizability 下限同源同构但取更保守的比例
    （k=0 本身合法，不像 omega=0 是数学奇点，故不能照抄 0.1 这个量级）。
    """

    def test_k_floor_recovers_collapsed_cell(self):
        n_cells, n_sps = 2, 1
        k_inf = 0.1666
        model = SSTModelFR(n_cells, n_sps, k_inf=k_inf, omega_inf=2268.1)

        # 模拟真实复现：某单元 k 被打到裸正性下限附近。
        model.k_field[0, 0] = 1e-12
        model.k_field[1, 0] = 0.15  # 对照单元，正常值

        model.apply_positivity_limiter()

        assert model.k_field[0, 0] == pytest.approx(1e-3 * k_inf)
        assert model.k_field[0, 0] > 1e-6  # 明确排除"还停留在裸下限附近"
        # 正常单元不受影响（远高于新下限，不应被下限"拉低"或改变）。
        assert model.k_field[1, 0] == pytest.approx(0.15)

    def test_k_floor_does_not_override_higher_values(self):
        """新下限只在 k 已经跌破时兜底，不能把高于下限的正常值意外
        拉到下限本身。"""
        n_cells, n_sps = 1, 1
        k_inf = 0.1666
        model = SSTModelFR(n_cells, n_sps, k_inf=k_inf, omega_inf=2268.1)
        model.k_field[0, 0] = 0.05  # 远高于 1e-3*k_inf

        model.apply_positivity_limiter()

        assert model.k_field[0, 0] == pytest.approx(0.05)

    def test_k_floor_absent_when_k_inf_not_set(self):
        """防御性：k_inf 属性缺失（旧版本/异常构造路径）时不应该报错，
        应静默跳过新下限，只保留原有裸正性下限行为。"""
        n_cells, n_sps = 1, 1
        model = SSTModelFR(n_cells, n_sps, k_inf=0.1666, omega_inf=2268.1)
        del model.k_inf
        model.k_field[0, 0] = 1e-13

        model.apply_positivity_limiter()

        assert model.k_field[0, 0] == pytest.approx(1e-12)
