"""AutoFlowCFD V2.0 - PTC 的 `dtau` 缩放：由 Newton 步自身的成败驱动。

## 它修的是一个被真实运行抓到的**停滞**

`jfnk.py` 的残差接受判据在方向不可信时返回 `theta = 0`（这一步原地不
动），并把处置写成"交给外层自适应 CFL 缩小 dtau"。真实运行证明**那条
交接从来没有发生过**：

    Blasius（无前缘奇点档，1728 单元 P1 原生棱柱基）CFL=10，200 步
      step  16  R_eval  479  res 2.106184e+06  theta=1.000   <- 最好点
      step  32  R_eval  901  res 4.036959e+06  theta=0.220
      step  48  R_eval 1502  res 4.096611e+06  theta=0.000
      step  64  R_eval 2110  res 4.096611e+06  theta=0.000
      ...（一直到 step 200，残差**逐位不变**，每步烧掉 38 次残差求值）

原因有两条，都是结构性的：

1. 外层自适应 CFL 控制器是按**残差历史**工作的，而 `theta = 0` 的一步
   让残差逐位不变 —— 控制器看到的是"完全平坦"，既没有下降也没有恶化，
   于是它没有任何理由缩小 CFL。停滞信号原理上传不到它那里。
2. 那次对照是固定 CFL 跑的（`--cfl-start/--cfl-max` 同值），控制器本身
   就不动 —— 而固定 CFL 是隐式路径的常用用法（隐式的意义正是敢用大
   CFL），所以不能把"必须开自适应"当成前提。

## 修法

PTC 的 `dtau` 由**两层**给出，各管各的事：

* 外层自适应 CFL 控制器给出**天花板**（按残差历史工作，跨格式共用，
  一个事实来源，见 `core/utils/cfl.py`）；
* 本模块给出天花板之下的**缩放因子** `scale in (0, 1]`，由 Newton 步
  自身的成败驱动 —— 它看的是 `theta`，也就是控制器看不到的那件事。

`scale` 只会降到天花板以下、永远不越过它，所以"CFL 上限由谁决定"这件事
仍然只有一个答案，本模块不参与。

## 为什么这一定能走出停滞（不是"再试试看"）

`dtau -> 0` 时 PTC 系统退化成显式前向 Euler（`dU = -dtau R`，见
`preconditioner.py`），而显式小步长在残差接受判据下必然被接受（步长
趋零时 `||R(U0 + theta dU)|| -> ||R(U0)||`，判据是 `<= 1.5 ||R(U0)||`）。
所以逐档缩小 `dtau` 必然在有限档内拿到一个被接受的步 —— 最坏情况退化
成显式推进的速度，而不是原地不动。

## 为什么要涨回去

一次失败就把 `dtau` 永久压低会让隐式路径退化成显式：本项目已经吃过
"越界之后收缩救不回来"的亏（项目记忆
`adaptive_cfl_four_defects_and_soft_ceiling` 第 12 条），但那条教训说的
是**越过稳定边界之后**收缩救不回来，与这里不同：这里 `theta = 0` 的一步
根本没有被走出去（状态逐位不变），没有留下任何需要"救回来"的污染。所以
恢复是安全的，条件也定得严：只有一步被**完整**接受（既没有被物理性限幅
削过、也没有回溯过，`theta == 1`）才涨一档。
"""

import numpy as np

#: 一步不被接受时 `scale` 乘的因子。0.25 是每档 4 倍，比 0.5 更快地穿过
#: "方向不可信"的区间（实测上面那个停滞在 CFL=10 出现、CFL=1 不出现，
#: 也就是需要跨越约一个数量级，0.5 要 4 档、0.25 只要 2 档）。
FAIL_SHRINK = 0.25

#: 一步被完整接受（`theta == 1`）时 `scale` 乘的因子，上限 1.0。
#: 取 2.0 而不是 1/FAIL_SHRINK=4.0：恢复比收缩慢是控制器的常规设计
#: （避免在边界附近来回振荡）。
OK_GROW = 2.0

#: `scale` 下限。低于这个值意味着即使退化成显式前向 Euler、步长还要再
#: 小 8 个数量级才能被接受 —— 那不是步长问题，是残差求值本身出了问题
#: （非有限值、或状态已经非物理），必须报出来而不是继续缩。
MIN_SCALE = 1e-8


class PtcDtauScale:
    """PTC `dtau` 的缩放因子状态机，`scale in [MIN_SCALE, 1]`。

    **做成类而不是模块级函数 + 调用方自己存一个浮点数**：调用方只需要
    持有这一个对象并在 Newton 步前后调用它，不需要知道两个因子、上下限
    与"什么才算完整接受"这些策略细节（它们全在这里，只有一个事实来源）。
    """

    __slots__ = ("scale",)

    def __init__(self, scale: float = 1.0):
        s = float(scale)
        if not np.isfinite(s) or s <= 0.0:
            raise ValueError(f"dtau 缩放因子必须是正有限值，收到 {scale!r}")
        self.scale = min(1.0, max(MIN_SCALE, s))

    def cut(self) -> bool:
        """缩小一档，返回是否真的缩了（已在下限时返回 `False`）。"""
        if self.scale <= MIN_SCALE:
            return False
        self.scale = max(MIN_SCALE, self.scale * FAIL_SHRINK)
        return True

    def reward(self, theta: float) -> None:
        """一步被完整接受（残差接受判据没有回溯，`theta >= 1`）时放大一档；
        被回溯说明当前 `dtau` 已经在边界上，保持不动。

        物理性限幅是逐单元的局部松弛（`jfnk.py::PHYSICALITY_MAX_RELATIVE_CHANGE`），
        不进入这里：个别单元被松弛不代表全局 `dtau` 过大，让它拖住全场的
        `dtau` 正是逐单元松弛要消除的"一个单元冻结全场"。
        """
        if theta >= 1.0:
            self.scale = min(1.0, self.scale * OK_GROW)

    def __repr__(self) -> str:  # pragma: no cover - 仅用于日志/调试
        return f"PtcDtauScale(scale={self.scale:.3e})"
