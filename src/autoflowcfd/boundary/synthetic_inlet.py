"""
AutoFlowCFD V2.0 - 合成湍流入口（合成涡方法 SEM，Jarrin 2008）

    u'_i(x) = a_ij * (1/sqrt(N)) * sum_k  eps_k,j * f_sigma(x)(x - x_k)
    f_sigma(x)(r) = sqrt(V_B / sigma(x)^3) * prod_m f(r_m / sigma(x)),   f(s) = sqrt(3/2) (1 - |s|)，|s| < 1

- 涡核 x_k 均匀分布在影响盒 B 内（体积 V_B），各分量随机符号 eps_k,j = +/-1；a 是目标雷诺应力的 Cholesky 因子
  （R = a a^T）。对盒内均匀分布的单个涡核 E[f_sigma^2] = 1（只要 x 周围边长 2 sigma(x) 的立方体完全在盒内），
  所以 E[u'_i u'_j] = R_ij 与 N 无关。
- **影响盒**取在入口的局部正交坐标系里（e1 = 来流方向），入口全部点的局部坐标包围盒在**三个方向**各外扩
  sigma_max：横向不外扩时入口横向边缘处的雷诺应力只复现约 1/2、角点约 1/4（2026-10-17 以前只沿与来流夹角
  最小的坐标轴外扩，且盒子与坐标轴对齐，入口倾斜或有攻角时不在入口局部坐标系里，I-71）。
- **涡核尺度** sigma(x) = max(k^(3/2)/epsilon, Delta(x))（Jarrin 2008 §4.4，入口无边界层故无 kappa delta 项）：
  k、epsilon = beta* k omega 由来流湍流强度与涡粘比给定（与 RANS 来流同一组量，单一来源
  `core/fr_solver/turbulence/init.py::_set_freestream_turbulence`）；Delta(x) 为该点所在单元的 LES 滤波宽度
  （`core/turbulence/sgs.py::les_filter_width`，计入阶数），保证涡核可被网格分辨。2026-10-17 以前取"入口包围盒
  最大跨度 / 10"，与来流湍流无关（风洞 10 m 入口给出 1 m 的涡，I-74）。
- **涡核数** 默认按 Jarrin 的密度判据 N = V_B / sigma_min^3（最小涡核处每点平均被 8 个涡核覆盖）；用户给出更少
  的涡核时脉动统计仍无偏，但概率分布高度间歇（多数时刻为零、偶发尖峰），给出警告。
- **时间演化**：涡核随来流对流；越过盒下游面的涡核按越过的距离回绕到上游面（保持盒内均匀分布，不在上游面
  堆积），横向坐标与符号重新随机抽取。
- **求值**：每个物理步对入口全部点一次性求值（涡核 KD 树，切比雪夫距离查 |x - x_k|_inf < sigma(x)），
  按面切片取用；2026-10-17 以前每个面每次残差求值都在 Python 里遍历全部涡核。

参考文献：Jarrin, N. (2008). "Synthetic Inflow Boundary Conditions for the Numerical Simulation of Turbulence."
PhD thesis, University of Manchester.
"""

from typing import Optional

import numpy as np
from loguru import logger
from scipy.spatial import cKDTree

#: 形函数 f(s) = sqrt(3/2) (1 - |s|) 满足 int_{-1}^{1} f^2 ds = 1
_F_COEFF = np.sqrt(1.5)


def inlet_frame(flow_direction) -> np.ndarray:
    """入口局部正交坐标系 `(3, 3)`，各行依次为 e1（来流方向）、e2、e3（右手系）。"""
    e1 = np.asarray(flow_direction, dtype=np.float64)
    norm = np.linalg.norm(e1)
    if not norm > 0.0:
        raise ValueError("SEM 来流方向为零向量")
    e1 = e1 / norm
    helper = np.eye(3)[int(np.argmin(np.abs(e1)))]           # 与 e1 夹角最大的坐标轴，叉积不退化
    e2 = np.cross(e1, helper)
    e2 /= np.linalg.norm(e2)
    return np.stack([e1, e2, np.cross(e1, e2)])


def sem_length_scale(k_inf: float, omega_inf: float, beta_star: float, delta: np.ndarray) -> np.ndarray:
    """逐点涡核尺度 sigma = max(k^(3/2)/epsilon, Delta)，epsilon = beta* k omega（即 sqrt(k) / (beta* omega)）。"""
    ell = np.sqrt(k_inf) / (beta_star * omega_inf)
    return np.maximum(ell, np.asarray(delta, dtype=np.float64))


def reynolds_stress_factor(reynolds_stress: np.ndarray) -> np.ndarray:
    """R = a a^T 的下三角因子 a。R 必须对称半正定：全零（来流无湍流）给零因子，其余必须正定。

    2026-10-17 以前分解失败时静默退回"对角元开方"，把非法的目标张量伪装成另一个张量。
    """
    R = np.asarray(reynolds_stress, dtype=np.float64)
    if not np.allclose(R, R.T, rtol=0.0, atol=1e-12 * max(np.abs(R).max(), 1e-300)):
        raise ValueError(f"SEM 目标雷诺应力不对称：{R.tolist()}")
    if not np.any(R):
        return np.zeros((3, 3))
    return np.linalg.cholesky(R)


class SyntheticEddyMethod:
    """一个入口组的涡核集合（入口局部坐标），各 rank / 各面切片共享同一个实例。

    使用：
        sem = SyntheticEddyMethod(points, sigma, flow_direction, seed=...)
        sem.advance(dt, velocity)                      # 每个物理步一次
        raw = sem.unit_fluctuations(points, sigma)     # (n, 3)，各分量方差为 1，再乘 a^T 得 u'
    """

    def __init__(self, points: np.ndarray, sigma: np.ndarray, flow_direction, num_eddies: Optional[int] = None,
                 seed: Optional[int] = None):
        points = np.asarray(points, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or sigma.shape != points.shape[:1]:
            raise ValueError(f"SEM 入口点 {points.shape} 与涡核尺度 {sigma.shape} 形状不符")
        if not np.all(sigma > 0.0):
            raise ValueError("SEM 涡核尺度必须为正")
        self.frame = inlet_frame(flow_direction)
        self.origin = points.mean(axis=0)
        xi = self.to_local(points)
        s_max = float(sigma.max())
        self.box_min = xi.min(axis=0) - s_max
        self.box_max = xi.max(axis=0) + s_max
        self.volume = float(np.prod(self.box_max - self.box_min))
        self.recommended_num_eddies = int(np.ceil(self.volume / float(sigma.min()) ** 3))
        self.num_eddies = self.recommended_num_eddies if num_eddies is None else int(num_eddies)
        if self.num_eddies < 1:
            raise ValueError(f"SEM 涡核数必须为正（{self.num_eddies}）")
        if self.num_eddies < self.recommended_num_eddies:
            logger.warning(
                f"SEM 涡核数 {self.num_eddies} 少于 Jarrin 密度判据 V_B/sigma_min^3 = {self.recommended_num_eddies}："
                f"入口脉动统计无偏，但分布高度间歇（多数时刻为零）。不给 --sem-num-eddies 时按判据自动取值")
        self._rng = np.random.default_rng(seed)
        self.eddy_centers = self._rng.uniform(self.box_min, self.box_max, size=(self.num_eddies, 3))
        self.eddy_signs = self._random_signs(self.num_eddies)
        self.version = 0                      # 每次 advance 加一；求值缓存按它失效
        self._tree: Optional[cKDTree] = None  # 涡核中心的 k-d 树，涡核移动后重建

    def to_local(self, points: np.ndarray) -> np.ndarray:
        return (np.asarray(points, dtype=np.float64) - self.origin) @ self.frame.T

    def _random_signs(self, n: int) -> np.ndarray:
        return self._rng.choice([-1.0, 1.0], size=(n, 3))

    def advance(self, dt: float, velocity) -> None:
        """涡核沿对流速度 `velocity`（全局坐标，取来流速度）平移 dt；越过下游面的回绕到上游面并重抽横向坐标与符号。"""
        c = self.eddy_centers
        c += (self.frame @ np.asarray(velocity, dtype=np.float64)) * float(dt)
        length = self.box_max - self.box_min
        crossed = (c[:, 0] < self.box_min[0]) | (c[:, 0] > self.box_max[0])
        # 流向回绕保留越过的距离（盒内仍均匀）；横向理论上不动（e1 即来流方向），同样回绕保证留在盒内
        c[:] = self.box_min + np.mod(c - self.box_min, length)
        n_new = int(crossed.sum())
        if n_new:
            c[crossed, 1:] = self._rng.uniform(self.box_min[1:], self.box_max[1:], size=(n_new, 2))
            self.eddy_signs[crossed] = self._random_signs(n_new)
        self.version += 1
        self._tree = None

    def unit_fluctuations(self, points: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """`(n, 3)`：sum_k eps_k f_sigma(x)(x - x_k) / sqrt(N)，各分量期望方差为 1（全局坐标分量）。"""
        xi = self.to_local(points)
        sigma = np.asarray(sigma, dtype=np.float64)
        n = xi.shape[0]
        tree = self._tree
        if tree is None:
            tree = self._tree = cKDTree(self.eddy_centers)
        hits = tree.query_ball_point(xi, r=sigma, p=np.inf)
        counts = np.fromiter((len(h) for h in hits), dtype=np.int64, count=n)
        out = np.zeros((n, 3))
        if counts.sum() == 0:
            return out
        pt = np.repeat(np.arange(n), counts)
        ed = np.fromiter((e for h in hits for e in h), dtype=np.int64, count=int(counts.sum()))
        s = (xi[pt] - self.eddy_centers[ed]) / sigma[pt, None]
        f = np.prod(_F_COEFF * np.clip(1.0 - np.abs(s), 0.0, None), axis=1)
        f *= np.sqrt(self.volume / sigma[pt] ** 3)
        for j in range(3):
            out[:, j] = np.bincount(pt, weights=self.eddy_signs[ed, j] * f, minlength=n)
        return out / np.sqrt(self.num_eddies)


def synthetic_inlet_seed(group_name: str) -> int:
    """入口组的确定性随机种子（组名的 CRC32）。

    2026-10-04 以前种子为 None：同一算例两次运行的入口脉动不同（结果不可复现）；CPU 分布式传统模式下
    各 rank 各自构造 SEM、涡核互不相同，入口合成湍流在分区边界上不连续。
    """
    import zlib

    return zlib.crc32(str(group_name).encode("utf-8"))


def synthetic_inlets(provider):
    """边界幽灵态提供者里的全部 SEM 实例（`InletSEMGhostState.sem`）。"""
    return [cfg["sem"].sem for cfg in provider.code_to_config.values() if cfg.get("sem") is not None]


def advance_synthetic_inlets(provider, freestream: dict, dt: float) -> None:
    """每个物理步把全部 SEM 入口的涡核沿来流对流一步（全部后端共用）。

    平动速度取真实来流向量（`flow_direction.direction_from_freestream`，与入口坐标系的 e1 一致）。
    此前 CPU 单机写死 `(vel_inf, 0, 0)`（有攻角/侧滑角时与入口盒方向不一致），单 GPU、多 GPU、CPU 分布式
    则从不推进（GPU 上入口涡结构在时间上冻结）。
    """
    from autoflowcfd.core.utils.flow_direction import direction_from_freestream

    velocity = float(freestream["vel_inf"]) * direction_from_freestream(freestream)
    for sem in synthetic_inlets(provider):
        sem.advance(dt, velocity)
