"""AutoFlowCFD V2.0 - inexact-Newton 的 forcing term（Eisenstat-Walker）。

## 为什么不能把线性系统解准

Newton 外迭代每一步要解 `(I/dtau + J) dU = -R`。把它解到机器精度是**浪费**
的：远离解时 Newton 方向本身只是个粗略的下降方向，把它算准 12 位有效
数字不会让外迭代少走几步，但会让 GMRES 多跑几十次残差求值。

inexact Newton 的判据是：只要求

    || R + (I/dtau + J) dU ||  <=  eta * || R ||

其中 `eta` 是 forcing term。取定值（例如 0.1）可用但不最优；
Eisenstat & Walker (1996) 的 "choice 2" 用**上一步实际取得的残差下降**
来自适应决定下一步该解多准：

    eta_k = gamma * ( ||R_k|| / ||R_{k-1}|| )^alpha

含义是"上一步残差掉得多（Newton 方向好），就把下一步解得更准；掉得少
（还在非线性主导区），就别浪费迭代"。标准参数 `gamma=0.9, alpha=2`。

两个必须有的护栏（原文即建议）：

* **安全下限**：`eta` 不能掉到远小于最终要达到的非线性容差，否则最后
  几步在做无意义的高精度线性求解。取
  `eta = max(eta, 0.5 * tol_nonlinear / ||R||)`。
* **过度收缩保护**：`eta_k` 相比 `eta_{k-1}` 掉得太快时（`eta_{k-1}^alpha
  > 0.1`）回退到 `max(eta_k, gamma * eta_{k-1}^alpha)`，避免一次偶然的
  大幅下降把后面所有步都锁在过严的容差上。

## 与 PTC 的关系

`dtau` 小的时候系统由对角项主导、GMRES 几步就能满足任何 `eta`，forcing
term 基本不起作用；它真正省时间是在 `dtau` 放大、系统接近纯 Newton 的
后期 —— 那时一次线性求解可能要几十次残差求值。
"""

import numpy as np

#: Eisenstat-Walker "choice 2" 的标准参数。
_EW_GAMMA = 0.9
_EW_ALPHA = 2.0

#: `eta` 的上界。
#:
#: **0.9 -> 0.1（2026-09-19，被真实运行证伪后改）**。原值 0.9 取自
#: Eisenstat-Walker 原文，但那是给**纯 Newton** 用的：原文假设残差单调
#: 下降，`eta` 松只是少解几位、方向还是 Newton 方向。
#:
#: 在**伪瞬态延拓**里这个假设不成立，而且后果是定性的：预处理是
#: `M^{-1} = dtau`，所以 GMRES 的第一个迭代给出的就是
#: `dU ≈ dtau*(-R)` —— **前向 Euler 步**。`eta=0.9` 时 GMRES 一步就满足
#: 容差、直接返回，于是"隐式步"退化成一个 CFL 等于 dtau 的显式步。
#:
#: 实测（Blasius P1 原生棱柱，1728 单元）：
#:
#:     CFL    eta 上界 0.9        eta 上界 0.1
#:     0.03   残差 950 -> 9.7e4   （与显式同，本来就在显式极限内）
#:     10     残差 950 -> 6.5e19  <- 发散：等于跑 CFL 10 的显式步
#:     100    每步撞满迭代上限
#:
#: 而 PTC 的全部价值就在于"用显式不可能的大 dtau"。所以上界必须压到
#: "解出来的方向真的还是隐式方向"那一档。0.1 的含义是"至少解一位有效
#: 数字"，配合下方 `forcing.py` 之外的**残差接受判据**（见
#: `jfnk.py::_accept_step`）构成安全的 PTC。
_ETA_MAX = 0.1

#: `eta` 的硬下限。低于它的线性求解精度对外迭代没有可观测收益，只是
#: 把残差求值花在舍入噪声上（一阶 Fréchet 差分本身的相对误差就是
#: `~sqrt(eps_mach) ~ 1e-8`，见 `jacobian_vector.py`）。
_ETA_MIN = 1.0e-4


class EisenstatWalkerForcing:
    """按 Eisenstat-Walker choice 2 给出每个 Newton 步的线性求解容差。"""

    __slots__ = ("_eta", "_res_prev")

    def __init__(self):
        self._eta = _ETA_MAX
        self._res_prev = None

    def next_eta(self, res_norm: float, tol_nonlinear: float) -> float:
        """给出当前 Newton 步该用的相对线性容差 `eta`。

        Args:
            res_norm: 当前非线性残差范数 `||R_k||`。
            tol_nonlinear: 外迭代的目标绝对容差，用来定安全下限。

        Returns:
            `eta`，落在 `[_ETA_MIN, _ETA_MAX]` 内。
        """
        res_norm = float(res_norm)
        if self._res_prev is None or self._res_prev <= 0.0 or res_norm <= 0.0:
            eta = _ETA_MAX
        else:
            ratio = res_norm / self._res_prev
            eta = _EW_GAMMA * ratio ** _EW_ALPHA
            # 过度收缩保护（原文 (2.6)）
            eta_prev_pow = _EW_GAMMA * self._eta ** _EW_ALPHA
            if eta_prev_pow > 0.1:
                eta = max(eta, eta_prev_pow)
        # 安全下限：别比"最终非线性容差的一半"还严
        if res_norm > 0.0:
            eta = max(eta, 0.5 * tol_nonlinear / res_norm)
        eta = float(min(_ETA_MAX, max(_ETA_MIN, eta)))
        self._eta = eta
        self._res_prev = res_norm
        return eta
