"""AutoFlowCFD V2.0 - 棱柱基的选择（坍缩坐标 / 原生 PKD⊗Legendre）。

## 这是一个**迁移期**开关，有明确的终态

`AFCFD_PRISM_BASIS = collapsed | native`，**迁移期默认 `collapsed`**（已长期
验证的既有行为）。

终态是：真实网格 A/B 判定原生基修掉「自由流保持性只有 ~1e-9」与「伪横流 /
P2 发散」之后，**把坍缩棱柱基整套删除、把这个开关一并删掉**，原生成为棱柱
的唯一实现。这与四面体当年走过的完全同一条路 —— `--tet-basis-mode` 也是
先加开关、真实网格验证（79 万单元 P2 灾难性发散被 native 解决）、然后
删除坍缩实现并去掉参数（见项目记忆
`collapsed-tet-basis-deleted-2026-09-03`）。

所以这里**不是**"加一个开关留着以后慢慢选"：本项目的既定立场是同一语义
只留一个实现，这个开关只在"两条实现并存以便做决定性对照"这段时间内存在。

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

#: 迁移期默认值 —— 已长期验证的既有行为。
_DEFAULT = "collapsed"


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
