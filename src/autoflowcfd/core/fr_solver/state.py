"""
AutoFlowCFD V2.0 - FR 求解器状态数据结构 (S-01)

本模块定义 FRState 类，用于管理存储在 Solution Points (SPs) 上的
守恒变量 (Conservative Variables) 及其时间导数。
"""

import numpy as np
from numba import njit, prange
from dataclasses import dataclass

#: 求解器状态数组的变量数：平均流守恒变量 (rho, rho*u, rho*v, rho*w, rho*E)。湍流输运场存在模型对象上
#: （`turbulence/transported.py`）。2026-10-05 以前 SST 族的 CPU 单机状态另带两个 k/omega 槽位
#: `U[..., 5:7]`：全仓库无人读取、残差恒为零，却让 CPU SST 的残差 RMS 分母多算两列（比 GPU/分布式小
#: sqrt(7/5) 倍）、写出的 checkpoint 是 7 列（恢复到 5 列的单 GPU 时形状校验拒绝）、隐式步每次残差求值
#: 多拷一份整状态。旧 checkpoint 的 7 列状态恢复时只取前 5 列（`cli/solve/checkpoint_io/restore.py`）。
N_MEAN_FLOW_VARS = 5


def uniform_conservative(rho: float, u: float, v: float, w: float,
                         p: float, gamma: float = 1.4) -> np.ndarray:
    """均匀流场的守恒量 (5,) = `[rho, rho*u, rho*v, rho*w, rho*E]`。

    **全仓库"均匀原始量 -> 守恒量"的唯一公式**（量热完全气体）。

    为什么要单独立出来（2026-09-24）：此前这个公式在 10 处各写一遍，
    而且写法不止一种 —— `rho * (p/((g-1)*rho) + 0.5*|v|^2)` 与
    `p/(g-1) + 0.5*rho*|v|^2` 并存（数学相等、浮点舍入不同）。更要紧的是
    其中 8 处还把速度写死成 `(vel_inf, 0, 0)`，**攻角/侧滑角被静默丢掉**，
    见 `core/utils/flow_direction.py::freestream_conservative_state` 文档。

    `rho*E` 的写法与原 `FRState.initialize_uniform` 逐字一致，所以单机
    CPU 路径（两条黄金轨迹）逐位不变。
    """
    e = p / ((gamma - 1.0) * rho) + 0.5 * (u ** 2 + v ** 2 + w ** 2)
    return np.array([rho, rho * u, rho * v, rho * w, rho * e],
                    dtype=np.float64)


@dataclass
class SolverResult:
    """求解结果数据类。"""
    converged: bool
    iterations: int
    final_residual: float


class FRState:
    """
    FR 求解器状态容器。

    Attributes:
        U: 守恒变量数组，形状为 (n_cells, n_sps_per_cell, N_MEAN_FLOW_VARS)
        dU_dt: 残差/时间导数数组，形状同 U。
        Q: 原始变量数组 (rho, u, v, w, p)，用于通量计算。
    """

    def __init__(self, n_cells: int, n_sps_per_cell: int):
        """
        初始化 FRState。

        Args:
            n_cells: 单元数量
            n_sps_per_cell: 每个单元的解点数量
        """
        self.n_cells = n_cells
        self.n_sps = n_sps_per_cell
        self.n_vars = N_MEAN_FLOW_VARS

        # 形状: (n_cells, n_sps_per_cell, n_vars)
        self.U = np.zeros((n_cells, n_sps_per_cell, self.n_vars), dtype=np.float64)
        self.dU_dt = np.zeros_like(self.U)
        self.Q = np.zeros_like(self.U)  # 原始变量用于通量计算

    def initialize_uniform(self, rho=1.0, u=0.0, v=0.0, w=0.0, p=1.0):
        """用均匀流场初始化状态（公式的唯一事实来源见模块级 `uniform_conservative`）。"""
        self.U[:, :, :] = uniform_conservative(rho, u, v, w, p)
        self._update_primitives()

    def _update_primitives(self):
        """从守恒变量 U 更新原始变量 Q。

        numba 并行实现（2026-09-25，算式与此前的 numpy 版本逐项相同：密度下限
        1e-10、压力下限 1 Pa，下限都用比较实现以保证 NaN 照样传播）。此前每次约
        85 ms、单线程，每次残差求值前都要调一次。
        """
        _update_primitives_kernel(self.U, self.Q)

    def get_residual_norm(self) -> float:
        """计算残差的 RMS 范数（按单元数归一化），用于收敛性判断。

        使用 RMS (Root Mean Square) 而非原始 L2 范数，使残差量级与网格尺寸无关，
        便于不同网格间的收敛行为对比。RMS = L2 / sqrt(N)，其中 N 是总自由度数。
        """
        n_total = self.dU_dt.size  # n_cells * n_sps * n_vars
        if n_total == 0:
            return 0.0
        return np.linalg.norm(self.dU_dt) / np.sqrt(n_total)


@njit(cache=True, parallel=True)
def _update_primitives_kernel(U, Q):
    for c in prange(U.shape[0]):
        for s in range(U.shape[1]):
            rho = U[c, s, 0]
            if rho < 1e-10:
                rho = 1e-10
            u = U[c, s, 1] / rho
            v = U[c, s, 2] / rho
            w = U[c, s, 3] / rho
            ke = 0.5 * (u * u + v * v + w * w)
            p = (U[c, s, 4] - rho * ke) * 0.4  # gamma = 1.4
            if p < 1.0:
                p = 1.0
            Q[c, s, 0] = rho
            Q[c, s, 1] = u
            Q[c, s, 2] = v
            Q[c, s, 3] = w
            Q[c, s, 4] = p
