"""AutoFlowCFD V2.0 - inexact-Newton 的 forcing term（Eisenstat-Walker）。

## 为什么不能把线性系统解准

Newton 外迭代每一步要解 `(I/dtau + J) dU = -R`。把它解到机器精度是**浪费**
的：远离解时 Newton 方向本身只是个粗略的下降方向，把它算准 12 位有效
数字不会让外迭代少走几步，但会让 GMRES 多跑几十次残差求值。

inexact Newton 的判据是：只要求

    || R + (I/dtau + J) dU ||  <=  eta * || R ||

其中 `eta` 是 forcing term。取定值（例如 0.1）可用但不最优；Eisenstat & Walker (1996) 的
"choice 1" 按**线性模型与真实残差的吻合程度**自适应决定下一步该解多准：

    eta_k = | ||R_k|| - ||R_{k-1} + J_{k-1} s_{k-1}|| | / ||R_{k-1}||

`||R_{k-1} + J s||` 取上一步 GMRES 实际达到的线性残差（与 PETSc SNES 的 EW 实现同一做法，不额外求值）。
含义：线性模型预测得准（真实残差 ~ 线性残差），说明 Newton 模型可信、值得解得更准；预测得不准
（真实下降远小于线性求解给出的下降），再解准也是白算。

**为什么不用 choice 2**（2026-10-05 改，此前是 `eta_k = 0.9 (||R_k||/||R_{k-1}||)^2`）：choice 2 只看
"上一步残差掉了多少"，看不到下降其实受线性模型以外的因素限制。湍流平板 SA P3（3072 单元）收敛末段
的实测：eta 按 choice 2 收紧到 0.088 -> 0.031 -> 0.0079 -> 0.0026 -> 0.001，GMRES 76 -> 253 次/步，
而非线性残差每步只降 3.2 / 5.4 / 10.7 / 18.6 / 29 倍（远小于 1/eta）——5 步共 804 次 GMRES、约 660 s，
占 P3 阶段到收敛耗时的一半以上。A/B 数据见 `ProjectFiles/V2.0/38_*.md` 第 25.13 节。

护栏（原文即建议）：

* **安全下限**：`eta` 不能掉到远小于最终要达到的非线性容差，否则最后
  几步在做无意义的高精度线性求解。取
  `eta = max(eta, 0.5 * tol_nonlinear / ||R||)`。
* **过度收缩保护**：`eta_{k-1}^phi > 0.1`（`phi = (1+sqrt5)/2`）时取 `max(eta_k, eta_{k-1}^phi)`，
  避免一次偶然的大幅下降把后面的步锁在过严的容差上。
* **线性模型不对应实际步时**（步被回溯 `theta < 1`、被物理性限幅逐单元松弛，或线性残差非有限）：
  下一步取上界 `_ETA_MAX`（宽松方向）。

## 与 PTC 的关系

`dtau` 小的时候系统由对角项主导、GMRES 几步就能满足任何 `eta`，forcing
term 基本不起作用；它真正省时间是在 `dtau` 放大、系统接近纯 Newton 的
后期 —— 那时一次线性求解可能要几十次残差求值。
"""

#: Eisenstat-Walker choice 1 过度收缩保护的指数（黄金分割比，原文 (2.2)）。
_EW_PHI = 0.5 * (1.0 + 5.0 ** 0.5)

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
#: `globalization.py::accept_step`）构成安全的 PTC。
_ETA_MAX = 0.1

#: `eta` 的硬下限。低于它的线性求解精度对外迭代没有可观测收益，只是
#: 把残差求值花在舍入噪声上（一阶 Fréchet 差分本身的相对误差就是
#: `~sqrt(eps_mach) ~ 1e-8`，见 `jacobian_vector.py`）。
_ETA_MIN = 1.0e-4


class EisenstatWalkerForcing:
    """按 Eisenstat-Walker choice 1 给出每个 Newton 步的线性求解容差（见模块文档）。"""

    __slots__ = ("_eta", "_res_prev", "_lin_prev")

    def __init__(self):
        self._eta = _ETA_MAX
        self._res_prev = None
        self._lin_prev = None    # 上一步线性模型预测的残差 ||R + J s||（绝对值）；不可用时 None

    def next_eta(self, res_norm: float, tol_nonlinear: float) -> float:
        """给出当前 Newton 步该用的相对线性容差 `eta`。

        Args:
            res_norm: 当前非线性残差范数 `||R_k||`。
            tol_nonlinear: 外迭代的目标绝对容差，用来定安全下限。

        Returns:
            `eta`，落在 `[_ETA_MIN, _ETA_MAX]` 内。
        """
        res_norm = float(res_norm)
        if self._lin_prev is None or self._res_prev is None or self._res_prev <= 0.0 or res_norm <= 0.0:
            eta = _ETA_MAX
        else:
            eta = abs(res_norm - self._lin_prev) / self._res_prev
            guard = self._eta ** _EW_PHI           # 过度收缩保护（原文 (2.2)）
            if guard > 0.1:
                eta = max(eta, guard)
        # 安全下限：别比"最终非线性容差的一半"还严
        if res_norm > 0.0:
            eta = max(eta, 0.5 * tol_nonlinear / res_norm)
        eta = float(min(_ETA_MAX, max(_ETA_MIN, eta)))
        self._eta = eta
        self._res_prev = res_norm
        self._lin_prev = None                      # 本步结束由 record_step 给出
        return eta

    def record_step(self, linear_rel_residual: float, full_step: bool) -> None:
        """一个 Newton 步结束后调用：记下线性模型对本步预测的残差 `||R + J s||`。

        Args:
            linear_rel_residual: GMRES 实际达到的相对线性残差（相对本步 `||R||`）。
            full_step: 实际走的是否就是 GMRES 给出的完整方向（未回溯、未被物理性限幅松弛）。
                否则线性模型不对应实际步，下一步取上界。
        """
        lin = float(linear_rel_residual)
        if full_step and self._res_prev is not None and lin == lin and lin < float("inf"):
            self._lin_prev = lin * self._res_prev
        else:
            self._lin_prev = None
