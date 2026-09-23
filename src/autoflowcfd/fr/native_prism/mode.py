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

#: 合法取值。坍缩档已于 2026-09-23 删除，所以只剩一个。
PRISM_BASIS_MODES: Tuple[str, ...] = ("native",)

#: 环境变量名。
_ENV = "AFCFD_PRISM_BASIS"

#: 坍缩棱柱基**已于 2026-09-23 删除**（B3）。保留这个常量只为把旧配置
#: 明确顶回去 —— 静默忽略会让人以为还能切回坍缩档。
_REMOVED_MODE = "collapsed"


def resolve_prism_basis_mode() -> str:
    """棱柱基档位。**恒为 `"native"`** —— 坍缩棱柱基已删除。

    ## 为什么删除（2026-09-23，B3）

    坍缩棱柱基自 2026-09-20 起已不是默认档，只剩"对照用"一个理由；而它
    的存在同时是一处**正确性障碍**：内部面 IP 罚项（2026-09-23 补上，见
    `core/fr_operators/flux_kernels.viscous_ip_penalty_tilde`）的数学前提是
    "罚项经 `M^-1 ∮` 提升"，原生路径用 DG 提升算子 `lift_native` 满足它，
    而坍缩路径的界面项分配走 **1D Radau/VCJH 修正函数**
    （`g_left`/`g_right` + `_distribute_point`），把罚项塞进那套机制不构成
    IP 方法。实测（均匀基态、16 单元、纯粘性算子数值雅可比逐块谱）：

        基          动量块 max Re（无罚项 -> 有罚项）   正实部
        原生        +4.09e-10 -> +0.00e+00              0/288 -> 0/288
        坍缩        +7.61e-07 -> +5.80e+00              0/384 -> 10/384

    也就是同一个罚项在原生档上无害、在坍缩档上**把动量块从严格耗散变成
    有 10 个增长模态**。与其为一个即将删除的对照档单独构造一套罚项提升，
    直接删掉它 —— 这也是本项目 2026-09-03 删除坍缩四面体基时的同一判断。

    ## 保留下来的历史实测数据（删除前的负控制证据）

    这些数字是"为什么原生基是对的"的依据，代码删除后仍然有效，记录在此
    避免随实现一起蒸发（与 `collapsed-tet-basis-deleted-2026-09-03` 同一
    做法）：

        判据                              坍缩        原生
        真实网格 179,237 单元 P1+LES      第148步 nan  246步单调降2.5x
        等熵涡 P2（800 棱柱、CFL 0.1）    第23步非有限 150步降到初值0.557
        Couette 精确保持性 P2             1.78e-2     8.40e-9（210万倍）
        伪横流最小复现 P2 CFL 0.1         第82步 nan   400步有限 5.25e-2
        GCL 残差 P2 / P3                  1.5e-15 /    7.1e-17 /
                                          4.5e-14      1.4e-16
        过积分后体积项误差 P2             9.5e-3      2.9e-12（机器零）
        cond(V) oo=4 / max|D| oo=4        4.72e5 /    6.33e1 /
                                          33928.8      7.7
        粘性算子能量块正实部（16单元P1）  60/128      33/96
        粘性算子动量块 max Re             +5.80e+00   +0.00e+00

    Raises:
        ValueError: 显式设成已删除的 `"collapsed"` 时。**不静默接受** ——
            本项目多次因"配置被静默丢弃"出真实缺陷（固定 CFL 请求被丢弃、
            滤波档双解析器），所以这里宁可硬失败。
    """
    raw = os.environ.get(_ENV)
    if raw is None or raw == "":
        return "native"
    mode = raw.strip().lower()
    if mode == _REMOVED_MODE:
        raise ValueError(
            f"{_ENV}={raw!r}：坍缩棱柱基已于 2026-09-23 删除（原生基是"
            f"唯一实现）。删除理由与删除前的全部负控制实测数据见本函数"
            f"文档；不要为了跑对照而恢复它 —— 那条路径与内部面 IP 罚项"
            f"数学上不兼容。")
    if mode != "native":
        raise ValueError(
            f"{_ENV}={raw!r} 不是合法取值（现在只有 native）")
    return "native"


def prism_basis_is_native() -> bool:
    """恒为 `True`（坍缩棱柱基已删除，见 `resolve_prism_basis_mode`）。

    过渡期保留：调用点在 B3-b 里逐个消掉，消完后本函数一并删除。
    """
    return resolve_prism_basis_mode() == "native"
