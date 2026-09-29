"""AutoFlowCFD V2.0 - 隐式 Newton 步的物理性松弛与逐行局部伪时间步长。

`jfnk.py::step_newton_krylov` 求出方向之后，逐行给出允许的步长比例 `alpha`
（守恒变量看 rho 与 p，湍流标量看 k/omega），同一单元的行取单元内最小
（`rows_per_cell`）；被松弛的行下一步降局部 `dtau`（`update_local_dtau_scale`）。
从 `jfnk.py` 拆出（2026-09-26，单文件 500 行规范），逻辑未改。
"""

import numpy as np

from .reductions import LocalReductions

#: 物理性限幅允许的单步最大相对**下降**（对 `rho` 与 `p`，湍流的 `k`/`omega`）；
#: 增长按对数对称给出：单步 `u_new/u in [1-c, 1/(1-c)]`，c=0.5 即"最多减半、
#: 最多加倍"。这些量都是跨多个量级的正值量，对数对称才是同一个"变化幅度"。
#: 这是 SU2/FUN3D 那类"非物理点上收紧步长"的标准做法 —— Newton 方向在
#: 远离解时可以指向 `rho<0`，直接走过去会让下一次残差求值算在非物理态
#: 上、整个迭代失去意义。
#:
#: **逐单元松弛，不是全局取最小**（2026-09-25）：每个单元按自己全部解点
#: 的限值取一个因子 `alpha_c in [0,1]`，只缩放该单元的更新（单元内各解点
#: 共用一个因子，保持单元内更新的多项式形状）。此前是全场取最小的一个
#: 标量 `theta`：plate_demo P0+SST（17.9 万单元，NK）第 14 步湍流 Newton
#: 的 `theta_phys = 7.9e-5`——一个单元想把 omega 降一半以上，整个湍流场
#: 就只能走万分之一步；湍流冻住后平均流 `R_new/R = 0.9995`，残差此后
#: 逐位不动。SU2 的 `ComputeUnderRelaxationFactor` 同样是逐点的。
PHYSICALITY_MAX_RELATIVE_CHANGE = 0.5

#: 逐行局部伪时间步长缩放（跨步持久化，`update_local_dtau_scale`）：本步被物理性
#: 松弛（`alpha < 1`）的行，下一步的 `dtau` 乘以 `max(alpha, LOCAL_DTAU_CUT_MIN)`；
#: 未被松弛的行乘以 `LOCAL_DTAU_GROW`，恢复到 1 为止；下限 `LOCAL_DTAU_FLOOR`。
#:
#: **为什么松弛之外还要降局部步长**（2026-09-26）：松弛只把离谱的 Newton 方向按比例
#: 缩小，方向本身不变。大 `dtau` 下 `I/dtau` 压不住输运算子的近奇异模态时，那些点
#: 的 Newton 修正会大到荒谬——plate_demo P1+SST 锐边贴壁棱柱上 k 约 1e-4、Newton
#: 要改 dk = -29、+1180，dw 甚至 1.9e7（该点源项的局部对角估计只给出约 -0.3），
#: 逐点松弛因子 1e-6，约 0.8% 的解点每步都被冻住、湍流残差每步只降到 0.9。只降那些
#: 点的 `dtau` 让局部系统重新对角占优，修正回到有界的伪瞬态；与 SU2 按欠松弛因子
#: 调局部 CFL（`CFL_ADAPT`）同一思路。只作用在行本地，没有集体通信。
LOCAL_DTAU_CUT_MIN = 0.1
LOCAL_DTAU_GROW = 2.0
LOCAL_DTAU_FLOOR = 1e-8


def update_local_dtau_scale(scale, alpha_rows, xp):
    """逐行局部 `dtau` 缩放的一步更新（规则见 `LOCAL_DTAU_CUT_MIN` 上方的说明）。"""
    cut = scale * xp.maximum(alpha_rows, LOCAL_DTAU_CUT_MIN)
    grow = xp.minimum(scale * LOCAL_DTAU_GROW, 1.0)
    return xp.maximum(xp.where(alpha_rows < 1.0, cut, grow), LOCAL_DTAU_FLOOR)


def _pressure(u_flat: np.ndarray) -> np.ndarray:
    """`p = (gamma-1)(rho_E - |m|^2 / (2 rho))`，形状 `(N,)`。

    这里刻意**不**做 `state._update_primitives()` 里那套 `rho>=1e-10` /
    `p>=1.0` 的钳制：本函数的用途正是**检测**非物理，钳制会把要检测的
    东西抹掉。
    """
    rho = u_flat[:, 0]
    m2 = u_flat[:, 1] ** 2 + u_flat[:, 2] ** 2 + u_flat[:, 3] ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        return 0.4 * (u_flat[:, 4] - 0.5 * m2 / rho)


def density_pressure_row_limits(u0_flat, du_flat, red: LocalReductions):
    """逐行（逐解点）`alpha_i in [0, 1]`，使 `U0_i + alpha_i*dU_i` 保持物理。

    只约束两个真正会破坏残差求值的量：

    * **密度**：解析给出 `rho_new/rho in [1-c, 1/(1-c)]`；
    * **压力**：`p` 是 `U` 的非线性函数
      （`p = (gamma-1)(rho_E - |m|^2/(2 rho))`），所以不解那个二次
      不等式，而是先按密度定 `alpha`、再对 `p` 做一次**保守回缩**：
      若 `U0 + alpha*dU` 处的 `p` 越出同一个比例区间，按实际超出比例把
      `alpha` 再缩一次。只往"缩小"方向走。

    `c = PHYSICALITY_MAX_RELATIVE_CHANGE`。逐单元取最小由调用方
    （`step_newton_krylov`）统一做。

    **为什么不是"只要不变负就行"**：`rho` 掉到原值的 1e-6 虽然还是正数，
    但那一点的温度/声速会离谱到让下一次残差求值毫无意义、并污染整个
    Krylov 基。限制**相对变化**才是有效的护栏。
    """
    c = PHYSICALITY_MAX_RELATIVE_CHANGE
    xp = red.xp
    alpha = xp.clip(_relative_change_limits(xp, u0_flat[:, 0], du_flat[:, 0], c), 0.0, 1.0)

    p0 = _pressure(u0_flat)
    p1 = _pressure(u0_flat + alpha[:, None] * du_flat)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = (p1 - p0) / xp.maximum(xp.abs(p0), 1e-300)
        # 超出比例区间时按线性化回缩（NaN 即非物理试探点上 rho->0：不参与，
        # 那一点的 alpha 已由密度约束）
        shrink = xp.where(rel < -c, c / -rel, xp.where(rel > _up(c), _up(c) / rel, 1.0))
    return alpha * xp.where(xp.isnan(shrink), 1.0, shrink)


def _up(c):
    """与最大相对下降 `c` 对数对称的最大相对增长 `1/(1-c) - 1`。"""
    return c / (1.0 - c)


def _relative_change_limits(xp, u0, du, c):
    """逐点允许的最大步长比例，使 `(u0 + alpha*du)/u0 in [1-c, 1/(1-c)]`
    （正值量；`du == 0` 处 `+inf`）。

    **增长也要约束**（2026-09-25）：此前只约束下降。全局取最小 theta 的年代
    这一点被掩盖了——一个单元的下降限值把全场都冻住，增长也跟着被冻住；
    改成逐单元松弛后，plate_demo P0+SST 第 15 步 k 的最大值一步从 0.13 跳到
    114（未被约束的单元里 Newton 方向把 k 放大近千倍）。取对数对称而不是
    `|du|/u <= c`：后者把增长也压在 +50%，棱柱通道 NK+SST 算例上湍流发展期
    50~75% 的单元被松弛、120 步内收敛不了（对数对称下正常收敛）。"""
    up, down = du > 0.0, du < 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        lim_up = _up(c) * u0 / xp.where(up, du, 1.0)
        lim_down = c * u0 / xp.where(down, -du, 1.0)
    return xp.where(up, lim_up, xp.where(down, lim_down, xp.inf))


class ScaledFieldRowLimits:
    """标量场（湍流 `k`/`omega`）的逐行限值：每列单步变化以
    `s = max(|u|, scale_j)` 为基准，下降不超过 `c*s`、增长不超过 `(1/(1-c)-1)*s`，
    `c = PHYSICALITY_MAX_RELATIVE_CHANGE`。

    `u` 远大于尺度下限时就是对数对称的相对变化限幅（与守恒变量那一版同一理由：
    omega 掉到原值的 1e-6 仍是正数，但涡粘会被放大六个数量级、下一次残差求值
    毫无意义；反方向同理，见 `_relative_change_limits`）。在尺度下限附近变成绝对
    限幅：被输运的 k/omega 按 `turbulence/sst/bounds.py` 的约定可以越过下限甚至
    为负（realizability 只作用于模型项求值），纯相对限幅在那里退化成冻结——
    2026-09-26 之前正是贴下限的解点把整个单元的松弛因子压到 1e-4。

    `log_columns` 里的列是对数量（`ln omega`，见 `turbulence/sst/log_omega.py`）：单步
    `|du| <= ln(1/(1-c))`，即原量最多减半/加倍——与上面的对数对称规则是同一个约束，
    只是换到对数变量上表达（对数量本身没有"相对变化"可言）。

    **单元量级**（`rows_per_cell > 1`，2026-09-29）：基准再与该行所在单元真实解点的
    `|u|` 均值取大。解点值是同一个单元多项式的分量，"变化是否过大"要相对多项式的
    量级衡量；只看点值时，欠分辨前沿上恰好落在振荡低谷的解点（值近零）被冻结。
    plate_demo P1 湍流发展暂态实测：单元内 k 跨 6 个量级（如 [~0, 494]），约 0.5% 的
    k 行（~7500 行）每步被压到 alpha~1e-4、局部 dtau 钉在 ~5e-6，而多项式整体需要
    O(1~100) 的变化；P0 下的同一个限幅正是按单元量级算的。

    做成类而不是闭包：在整个 Newton 步存活（项目规范）。
    """

    __slots__ = ("_scales", "_log_columns", "_rows_per_cell", "_real_weight")

    def __init__(self, scales, log_columns=(), rows_per_cell: int = 1, real_rows=None):
        """`rows_per_cell`/`real_rows`：状态按单元连续排列时每单元的行数，与标记真实解点
        （非零填充槽位）的逐行布尔掩码（在状态所在的数组模块上）；默认不取单元量级。"""
        self._scales = np.asarray(scales, dtype=np.float64)
        self._log_columns = tuple(int(c) for c in log_columns)
        self._rows_per_cell = int(rows_per_cell)
        self._real_weight = None if real_rows is None else real_rows.astype(np.float64)

    def __call__(self, u0_flat, du_flat, red: LocalReductions):
        xp = red.xp
        mag = xp.abs(u0_flat)
        base = xp.maximum(mag, xp.asarray(self._scales)[None, :])
        if self._rows_per_cell > 1:
            n_col = u0_flat.shape[1]
            w = (xp.ones(u0_flat.shape[0]) if self._real_weight is None else self._real_weight)
            w = w.reshape(-1, self._rows_per_cell, 1)
            cell_mag = (mag.reshape(-1, self._rows_per_cell, n_col) * w).sum(axis=1) / w.sum(axis=1)
            base = xp.maximum(base, xp.repeat(cell_mag, self._rows_per_cell, axis=0))
        lim = _relative_change_limits(xp, base, du_flat, PHYSICALITY_MAX_RELATIVE_CHANGE)
        if self._log_columns:
            step = -np.log(1.0 - PHYSICALITY_MAX_RELATIVE_CHANGE)
            for c in self._log_columns:
                with np.errstate(divide="ignore"):
                    lim[:, c] = step / xp.abs(du_flat[:, c])
        return xp.clip(lim.min(axis=1), 0.0, 1.0)


def _cellwise_relaxation(alpha_rows, rows_per_cell: int, red: LocalReductions):
    """逐行限值 -> 逐单元因子（单元内取最小）按行展开，返回
    `(alpha_rows_cellwise, alpha_min, limited_cell_fraction)`（后两个是全局量）。"""
    xp = red.xp
    alpha_cell = alpha_rows.reshape(-1, rows_per_cell).min(axis=1)
    alpha_min = red.min(alpha_cell)
    limited = red.sum(alpha_cell < 1.0) / max(red.count(alpha_cell), 1.0)
    return xp.repeat(alpha_cell, rows_per_cell), float(alpha_min), float(limited)
