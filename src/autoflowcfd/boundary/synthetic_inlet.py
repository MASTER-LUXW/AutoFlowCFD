"""
AutoFlowCFD V2.0 - 合成湍流入口 (SEM) 实现 (Tier-1 重建版, 对应 BD-02)

真正的合成涡方法 (Synthetic Eddy Method, Jarrin 2006)，取代旧版本本质上
是与入口几何无关的静态随机噪声：

1. **涡核分布范围**：旧版本把涡核撒在硬编码的 [0,10]^3 立方体里，与传入
   的实际入口 SPs 坐标 `positions` 毫无关系；现在从入口面的真实包围盒
   构造"影响区"（沿流向额外扩展一个积分尺度，保证边缘处的涡不会突然
   被截断）。
2. **涡强度**：旧版本用原始 `randn(N,3)`，与目标雷诺应力张量无关；现在
   用目标雷诺应力张量的 Cholesky 分解 a_ij（R_ij = a_ik a_jk）对独立的
   ±1 随机符号做线性变换，保证生成脉动的协方差精确逼近目标雷诺应力
   （标准 SEM 构造，见 Jarrin 2006 PhD thesis §4）。
3. **归一化**：旧版本没有按涡数量 N 归一化，脉动方差会随 N 变化；现在
   用 1/sqrt(N_eddies) 归一化，配合形函数的二阶矩归一化
   （∫f(r)^2 dr=1），保证 N_eddies 变化时目标雷诺应力保持不变。
4. **时间演化**：旧版本的涡核构造一次后永不更新，对同一 positions 重复
   调用返回完全相同的"脉动"，不是真正的瞬态湍流；现在涡核随平均流对流
   （advance 方法按时间步平移涡核），离开下游边界的涡在上游边界重新
   以随机位置/符号生成，是 SEM 定义性的关键机制。

参考文献：Jarrin, N. (2008). "Synthetic Inflow Boundary Conditions for
the Numerical Simulation of Turbulence." PhD thesis, University of
Manchester.
"""

from typing import Optional, Tuple

import numpy as np


def _cholesky_from_reynolds_stress(reynolds_stress: np.ndarray) -> np.ndarray:
    """从雷诺应力张量 R_ij 求 Cholesky 分解 a_ij（下三角，R=a@a.T）。

    若 R 不是（数值上）正定（例如用户直接给了对角近似），退回到
    对角元素开方的近似分解，而不是让 np.linalg.cholesky 抛异常中止。
    """
    try:
        return np.linalg.cholesky(reynolds_stress)
    except np.linalg.LinAlgError:
        a = np.zeros((3, 3))
        for i in range(3):
            a[i, i] = np.sqrt(max(reynolds_stress[i, i], 0.0))
        return a


class SyntheticEddyMethod:
    """合成涡方法 (SEM) 处理器：为 LES/DES 入口提供满足指定雷诺应力
    张量、随时间演化的速度脉动。

    使用流程：
        sem = SyntheticEddyMethod(num_eddies=200, length_scale=0.05)
        sem.configure_inlet_box(inlet_positions, mean_velocity_normal=[1,0,0])
        # 每个时间步：
        sem.advance(dt, mean_velocity)
        u_fluct = sem.generate_fluctuations(inlet_positions, reynolds_stress)
    """

    def __init__(self, num_eddies: int = 200, length_scale: float = 0.1, seed: Optional[int] = None):
        self.num_eddies = num_eddies
        self.length_scale = length_scale
        self._rng = np.random.default_rng(seed)

        self.box_min: Optional[np.ndarray] = None
        self.box_max: Optional[np.ndarray] = None
        self.flow_axis: int = 0  # 主流向对应的坐标轴索引（用于上游再生面判断）

        self.eddy_centers: Optional[np.ndarray] = None  # (N,3)
        self.eddy_signs: Optional[np.ndarray] = None  # (N,3), 各分量独立 ±1

    def configure_inlet_box(self, positions: np.ndarray, flow_direction: np.ndarray) -> None:
        """根据真实入口 SPs 坐标构造涡核的"影响区"包围盒。

        Args:
            positions: 入口 SPs 物理坐标，形状 (N, 3)
            flow_direction: 主流方向（不必归一化），用于确定流向轴以及
                涡核对流/再生所沿的方向
        """
        flow_direction = np.asarray(flow_direction, dtype=float)
        self.flow_axis = int(np.argmax(np.abs(flow_direction)))
        self._flow_sign = 1.0 if flow_direction[self.flow_axis] >= 0 else -1.0

        pos_min = positions.min(axis=0)
        pos_max = positions.max(axis=0)
        # 影响区沿流向上游/下游各扩展一个积分尺度，保证入口平面边缘处的
        # SPs 仍能被"即将进入"或"刚离开"的涡核覆盖到（否则边缘涡强度
        # 系统性偏弱）；横向（非流向）不扩展，直接用入口面的真实包围盒。
        margin = np.zeros(3)
        margin[self.flow_axis] = self.length_scale
        self.box_min = pos_min - margin
        self.box_max = pos_max + margin

        self._reseed_all_eddies()

    def _reseed_all_eddies(self) -> None:
        self.eddy_centers = self._rng.uniform(self.box_min, self.box_max, size=(self.num_eddies, 3))
        self.eddy_signs = self._rng.choice([-1.0, 1.0], size=(self.num_eddies, 3))

    def advance(self, dt: float, mean_velocity: np.ndarray) -> None:
        """把所有涡核沿平均流对流一个时间步，离开影响区下游边界的涡
        在上游边界随机位置以新的随机符号重新生成（SEM 的核心机制，
        保证脉动真正随时间演化，而不是同一个静态场反复复用）。

        Args:
            dt: 时间步长
            mean_velocity: 平均对流速度矢量 (3,)，通常取入口来流速度
        """
        if self.eddy_centers is None:
            raise RuntimeError("configure_inlet_box() must be called before advance()")

        self.eddy_centers += np.asarray(mean_velocity)[np.newaxis, :] * dt

        axis = self.flow_axis
        if self._flow_sign >= 0:
            left_box = self.eddy_centers[:, axis] < self.box_min[axis]
            downstream = self.eddy_centers[:, axis] > self.box_max[axis]
        else:
            downstream = self.eddy_centers[:, axis] < self.box_min[axis]
            left_box = self.eddy_centers[:, axis] > self.box_max[axis]

        n_regen = int(np.sum(downstream))
        if n_regen > 0:
            # 在上游边界面（流向坐标固定为上游端）随机重新生成，
            # 其余两个方向仍在包围盒内随机取值，符号重新随机抽取。
            new_positions = self._rng.uniform(self.box_min, self.box_max, size=(n_regen, 3))
            upstream_coord = self.box_min[axis] if self._flow_sign >= 0 else self.box_max[axis]
            new_positions[:, axis] = upstream_coord
            self.eddy_centers[downstream] = new_positions
            self.eddy_signs[downstream] = self._rng.choice([-1.0, 1.0], size=(n_regen, 3))

        # 理论上不会发生（对流不会让涡瞬间越过整个盒子），但如果时间步
        # 异常大导致涡直接越过上游边界，同样重新生成，避免涡核漂出
        # 有效区间后长期不产生任何贡献。
        if np.any(left_box):
            n2 = int(np.sum(left_box))
            new_positions = self._rng.uniform(self.box_min, self.box_max, size=(n2, 3))
            self.eddy_centers[left_box] = new_positions
            self.eddy_signs[left_box] = self._rng.choice([-1.0, 1.0], size=(n2, 3))

    def generate_fluctuations(
        self, positions: np.ndarray, mean_u: np.ndarray, reynolds_stress: np.ndarray
    ) -> np.ndarray:
        """在给定位置生成满足目标雷诺应力的速度脉动，叠加到平均速度上。

        Args:
            positions: 入口 SPs 物理坐标 (N, 3)
            mean_u: 平均速度剖面，形状 (3,) 或可广播到 (N,3)
            reynolds_stress: 目标雷诺应力张量 R_ij，形状 (3,3)（对角占优
                即可，比如各向同性湍流 R=diag(u'^2,u'^2,u'^2)）

        Returns:
            u_total: 瞬时速度 (N, 3)
        """
        if self.eddy_centers is None:
            raise RuntimeError("configure_inlet_box() must be called before generate_fluctuations()")

        a_chol = _cholesky_from_reynolds_stress(np.asarray(reynolds_stress))

        n_points = positions.shape[0]
        raw_fluct = np.zeros((n_points, 3))
        sigma = self.length_scale
        vol_box = np.prod(self.box_max - self.box_min)
        # 单个涡核形函数的归一化系数：f_sigma = sqrt(V_box/sigma^3) * f(r1)f(r2)f(r3)，
        # 使得对随机分布在 V_box 内的单个涡核，E[f_sigma(x-x_k)^2] = 1
        # （标准 SEM 归一化，Jarrin 2008 式4.5；最终整体场再除以 sqrt(N) 使
        # Var[u'_i] 精确逼近 1，从而线性变换到目标雷诺应力后方差精确匹配——
        # 已用统计重构目标张量数值验证，见
        # tests/unit/test_synthetic_inlet.py）
        per_eddy_scale = np.sqrt(vol_box / sigma**3)

        for k in range(self.num_eddies):
            center = self.eddy_centers[k]
            r = (positions - center[np.newaxis, :]) / sigma  # (N,3)
            inside = np.all(np.abs(r) < 1.0, axis=1)
            if not np.any(inside):
                continue
            # 各向同性张量形函数：f(r)=sqrt(3/2)*(1-|r|)，满足 ∫_{-1}^1 f(r)^2 dr = 1
            shape_1d = np.where(np.abs(r) < 1.0, np.sqrt(1.5) * (1.0 - np.abs(r)), 0.0)
            shape_val = shape_1d[:, 0] * shape_1d[:, 1] * shape_1d[:, 2] * per_eddy_scale  # (N,)
            raw_fluct += self.eddy_signs[k][np.newaxis, :] * shape_val[:, np.newaxis]

        raw_fluct /= np.sqrt(self.num_eddies)

        # 线性变换到目标雷诺应力：u'_i = a_ij * raw_fluct_j
        fluctuations = raw_fluct @ a_chol.T

        return np.asarray(mean_u)[np.newaxis, :] + fluctuations


if __name__ == "__main__":
    sem = SyntheticEddyMethod(num_eddies=200, length_scale=0.05, seed=0)
    inlet_pos = np.column_stack(
        [np.zeros(50), np.random.default_rng(1).uniform(-0.5, 0.5, 50), np.random.default_rng(2).uniform(-0.3, 0.3, 50)]
    )
    mean_u = np.array([30.0, 0.0, 0.0])
    sem.configure_inlet_box(inlet_pos, flow_direction=mean_u)
    R = np.diag([2.0, 2.0, 2.0])  # 各向同性湍流，u'^2=2 m^2/s^2

    u1 = sem.generate_fluctuations(inlet_pos, mean_u, R)
    sem.advance(dt=1e-4, mean_velocity=mean_u)
    u2 = sem.generate_fluctuations(inlet_pos, mean_u, R)
    print("time-varying (u1 != u2):", not np.allclose(u1, u2))
    print("u1 shape:", u1.shape, "mean fluct magnitude:", np.std(u1 - mean_u))
