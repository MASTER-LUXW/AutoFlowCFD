"""AutoFlowCFD V2.0 - 隐式（Newton-Krylov）稳态的 CFL 律：SER。

## 为什么隐式不能用显式那套控制器

`controller.py` 的 `AdaptiveCFLController` 是围绕**显式稳定极限**设计的：
放大要连续 5 步确认、每次只放大 10%、带冷却期与软上限，因为显式格式
越过稳定边界一次之后收缩救不回来（模块文档第 12 条）。这些前提在
Newton-Krylov + 伪瞬态延拓（PTC）上都不成立：

* PTC 的 `dtau` 没有线性稳定极限，它只决定"离纯 Newton 有多近"——大
  `dtau` 收敛快、小 `dtau` 鲁棒；
* 一步走坏了有**当场**的保护：`jfnk.py` 的残差接受判据回溯/拒绝，
  `dtau_control.py` 缩小 `dtau` 重试。越界不会留下污染。

真实对照（plate_demo P1 层流，36 万单元，预处理 NK，同一个 P0 收敛解
起步，CFL 起点 5）：显式控制器第 8 步把 CFL 放到 5.5 之后就停在 5.78，
残差下降越来越慢（第 22→24 步 5.04e6→4.93e6，每步 <1%）；本模块的
SER 律同样起点，CFL 随残差下降自然上升（第 8 步 11.5），同步数残差
更低。

## 律本身

Mulder & van Leer (1985) 的 Switched Evolution Relaxation：

    CFL_{n+1} = CFL_n * (R_{n-1} / R_n) ^ exponent

限幅到 `[shrink_limit, growth_limit]` 每步，再钳到 `[cfl_min, cfl_max]`。
含义：残差降多少，`dtau` 就放大多少——离不动点越近，线性化越准，越可以
向纯 Newton（`dtau -> inf`）靠拢。

* `growth_limit = 2`：每步最多翻倍。更大的倍率让一次偶然的大幅下降
  （例如 Order Continuation 插值后第一步）直接把 CFL 抬好几个量级，
  下一步 GMRES 在远未收敛的线性化上发散。
* `shrink_limit = 0.1`：与 PETSc `TSPSEUDO`/SU2 同一量级；非有限残差
  （发散本身）直接按它收缩。

### R 用哪一个范数：Newton 实际在解的那个方程的

Newton 解的是 `F = Gamma R = 0`（低马赫伪时间预处理开启时），残差接受
判据也是按 `||F||` 定的；SER 必须看同一个 `||F||`（PTC 理论里 SER 的
量就是被求解系统的残差，Kelley & Keyes 1998）。**不能**看物理残差
`||R||`：`Gamma` 按 `M^2` 量级缩放连续/能量分量，两者零点相同、量级与
走向都可以不同。真实数据（棱柱通道 + SST，冲击启动第 1 步）：物理
`||R||` 38.9 -> 2.1e4（涨 543 倍），而 `||Gamma R||` 1160 -> 1596（涨
1.4 倍，正在接受判据的 1.5 倍之内）——按物理残差，SER 当场把 CFL 砍到
十分之一并一直钉在下限。调用方（`fr_solver/step.py`）传的是 Newton 步
报告的 `||F||`。

### 残差上升时：只在 Newton 步没被完整接受时才收缩

冲击启动（均匀初场 + 壁面）里残差**本来就会**上升很多步——边界层在
物理地发展。同一个通道算例上 `||Gamma R||` 连续 60 步单调上升，而每一步
Newton 都被完整接受（`theta = 1`、没有缩 dtau）。纯 SER 在这段里每步
都收缩，CFL 塌到下限、伪时间几乎不推进，暂态就永远走不完。

所以把两件事分开：

* 残差下降 -> 放大（SER 本义）；
* 残差上升、但 Newton 步被**完整**接受 -> **保持**：Newton 在这个
  `dtau` 下正确地跟踪着一个物理暂态，收缩只会让它走得更慢；
* 残差上升、且 Newton 步**没有**被完整接受（回溯或缩了 dtau）-> 按
  比值收缩：当前天花板已经在方向不可信的区间里，下一步不应再从这里起跳。

步内的紧急处置（当场缩 dtau 重试）仍然由 `dtau_control.PtcDtauScale`
负责，本控制器只调天花板，两层不重叠。

## 与 `dtau_control.PtcDtauScale` 的分工

本控制器给出 `dtau` 的**天花板**（按残差历史）；`PtcDtauScale` 在天花板
之下按 Newton 步自身的成败缩放（它看的是 `theta`，本控制器看不到）。
两层各管各的事，见 `dtau_control.py` 模块文档。
"""

from __future__ import annotations

import math

from loguru import logger


class SERCFLController:
    """隐式稳态（Newton-Krylov）的 SER CFL 律。

    接口与 `AdaptiveCFLController` 相同的那一部分（`cfl_start/cfl_max/
    cfl_min/cfl_number`、`update(residual)`、`reset()`），全部调用方
    （`step.py`、Order Continuation、日志）不需要区分二者。
    """

    def __init__(
        self,
        cfl_start: float = 5.0,
        cfl_max: float = 1.0e4,
        cfl_min: float = 0.5,
        exponent: float = 1.0,
        growth_limit: float = 2.0,
        shrink_limit: float = 0.1,
    ):
        for name, v in (("cfl_start", cfl_start), ("cfl_max", cfl_max),
                        ("cfl_min", cfl_min), ("exponent", exponent),
                        ("growth_limit", growth_limit), ("shrink_limit", shrink_limit)):
            if not (math.isfinite(v) and v > 0.0):
                raise ValueError(f"SER CFL 参数 {name} 必须是正有限值，收到 {v!r}")
        if not cfl_min <= cfl_max:
            raise ValueError(f"cfl_min={cfl_min} 大于 cfl_max={cfl_max}")
        if not (shrink_limit < 1.0 < growth_limit):
            raise ValueError(
                f"需要 shrink_limit < 1 < growth_limit，收到 {shrink_limit}/{growth_limit}")
        self.cfl_start = float(cfl_start)
        self.cfl_max = float(cfl_max)
        self.cfl_min = float(cfl_min)
        self.exponent = float(exponent)
        self.growth_limit = float(growth_limit)
        self.shrink_limit = float(shrink_limit)
        self.cfl_number = self._clamp(self.cfl_start)
        self._prev_residual = None

    def _clamp(self, cfl: float) -> float:
        return min(self.cfl_max, max(self.cfl_min, cfl))

    def update(self, current_residual: float, step_ok: bool = True) -> float:
        """按本步残差给出下一步的 CFL。首步只记录基准。

        Args:
            current_residual: Newton 所解系统的残差范数 `||F||`（见模块文档）。
            step_ok: 本步 Newton 是否被**完整**接受（`theta == 1` 且没有缩
                dtau）。残差上升时只有它为 False 才收缩。
        """
        r = float(current_residual)
        if not (math.isfinite(r) and r > 0.0):
            # 发散本身：按最大收缩倍率收缩；不更新基准（非有限值不可比）
            self.cfl_number = self._clamp(self.cfl_number * self.shrink_limit)
            logger.warning(
                f"[SER-CFL] 残差非有限/非正（{current_residual!r}），CFL 收缩到 "
                f"{self.cfl_number:.3g}")
            return self.cfl_number
        if self._prev_residual is not None:
            factor = (self._prev_residual / r) ** self.exponent
            factor = min(self.growth_limit, max(self.shrink_limit, factor))
            if factor >= 1.0 or not step_ok:
                self.cfl_number = self._clamp(self.cfl_number * factor)
        self._prev_residual = r
        return self.cfl_number

    def reset(self) -> None:
        """Order Continuation 换阶：新离散问题的残差量级与旧的不可比，
        回到起点重新按 SER 爬升。"""
        logger.info(f"[SER-CFL] reset（CFL was {self.cfl_number:.3g}）")
        self.cfl_number = self._clamp(self.cfl_start)
        self._prev_residual = None
