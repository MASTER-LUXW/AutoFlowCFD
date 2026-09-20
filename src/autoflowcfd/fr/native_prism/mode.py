"""AutoFlowCFD V2.0 - 棱柱基的选择（坍缩坐标 / 原生 PKD⊗Legendre）。

## 这是一个**迁移期**开关，判定已经做出

`AFCFD_PRISM_BASIS = collapsed | native`，**2026-09-20 起默认 `native`**。

### 默认值改成 native 的三份证据

1. **真实网格 A/B（决定性的那一份）**：`plate_demo_volume_les`，179,237
   单元，P1 + LES，同一套默认档、同一个 CFL 控制器：

       坍缩基：第 148 步残差 nan（发散中止）
       原生基：跑过 246 步，残差**单调下降**（Drop 2.5x），
               CFL 稳在上限 0.060，Cd 平滑收敛式下滑

2. **等熵涡 P2（`tests/validation/test_isentropic_vortex.py`）**：光滑、
   周期、无粘的平动算例，800 棱柱单元：

       坍缩基：第 23 步非有限（发散）
       原生基：走完 150 步，残差降到初值的 0.557

   这一条是在删除"机制3 残差量级离群清零"之后才**暴露**出来的 —— 此前
   机制3 从第 22 步起每步清零 3~18 个槽位，把发散压成"通过"。也就是说
   坍缩棱柱基在这个算例上一直是发散的，只是被遮住了（完整记录见
   `core/fr_residual/inviscid.py` 里那段机制3 的删除依据）。

3. **Couette 精确保持性（A1）**：P2 从 1.78e-2 修到 8.40e-9（约 210 万
   倍）；P3 从"第 5 步发散"变成 1600 步 4.38e-8。

### 还剩的一步

`collapsed` 这半边现在只剩"对照用"这一个理由。终态仍然是**把坍缩棱柱基
整套删除、把这个开关一并删掉**，与四面体当年走过的完全同一条路
（`--tet-basis-mode` 先加开关、真实网格验证 79 万单元 P2 灾难性发散被
native 解决、然后删除坍缩实现并去掉参数，见项目记忆
`collapsed-tet-basis-deleted-2026-09-03`）。本项目的既定立场是同一语义
只留一个实现。

## 为什么是模块级解析而不是参数透传

归约/填充相关的消费点里有一些**只拿得到数组**，拿不到 `ops` 或 `solver`
（GPU 侧只有数组与 flat face、分布式 checkpoint 只有 gather 出来的全局
数组）。本项目对这类跨切面的数值选择已有既成模式（`AFCFD_FILTER_MODE`、
`AFCFD_TROUBLED_SENSOR`、`AFCFD_VISC_OVERINT`），沿用它而不是再造一种。

## 两条基的自由度数（决定填充布局）

    order   坍缩 (p+1)^3   原生 (p+1)^2(p+2)/2
      1          8              6
      2         27             18
      3         64             40
      4        125             75

全局统一 SPs 宽度仍然是 `(p+1)^3`（棱柱坍缩基的值），原生棱柱与原生四面体
一样**零填充**进去 —— 这样网格几何、checkpoint、GPU 数组形状全都不用改。
自由度本身的内存收益要等第二步（收窄全局宽度）才拿到，那一步不在本次范围内。
"""

import os
from typing import Tuple

__all__ = [
    "PRISM_BASIS_MODES",
    "resolve_prism_basis_mode",
    "prism_basis_is_native",
]

#: 合法取值。
PRISM_BASIS_MODES: Tuple[str, ...] = ("collapsed", "native")

#: 环境变量名。
_ENV = "AFCFD_PRISM_BASIS"

#: 默认值（2026-09-20 起为 `native`，依据见模块文档"三份证据"）。
_DEFAULT = "native"


def resolve_prism_basis_mode() -> str:
    """读 `AFCFD_PRISM_BASIS`，返回 `"collapsed"` 或 `"native"`。

    Raises:
        ValueError: 取值不在 `PRISM_BASIS_MODES` 里。**不静默退回默认值**：
            拼错环境变量而静默跑了另一条基，会让整个 A/B 对照失去意义
            （本项目已经吃过一次"固定 CFL 请求被静默丢弃、两条不同配置
            给出逐位相同轨迹"的亏）。
    """
    raw = os.environ.get(_ENV)
    if raw is None or raw == "":
        return _DEFAULT
    mode = raw.strip().lower()
    if mode not in PRISM_BASIS_MODES:
        raise ValueError(
            f"{_ENV}={raw!r} 不合法，只能是 "
            f"{' | '.join(PRISM_BASIS_MODES)}。不静默退回默认值：拼错了"
            f"而静默跑另一条基会让 A/B 对照失去意义。")
    return mode


def prism_basis_is_native() -> bool:
    """`resolve_prism_basis_mode() == "native"` 的简写。"""
    return resolve_prism_basis_mode() == "native"
