"""
AutoFlowCFD V2.0 - 伪时间推进的 CFL 律：显式格式的自适应控制器（`AdaptiveCFLController`）与隐式 Newton-Krylov 的 SER
律（`ser.py::SERCFLController`）；用哪一个、固定 CFL 取多少由 `policy.py` 决定（全部后端唯一来源）。

显式控制器按残差历史调节 CFL，前提是**不越过显式稳定边界**（越界之后收缩救不回来，第 8 条）。各条机制都只在有长程
真实网格数据支撑时落地；推导、反例与实测数据见项目文档 `ProjectFiles/V2.0/39_自适应CFL控制器-设计记录与实测依据.md`（原文条目号与下列
编号的对应在那份文档开头）。

1. **五级调节**（`update.py`）：前 `ramp_steps` 步固定 `cfl_start`；之后按单步残差比 `ratio = R_n / R_{n-1}`：
   `< growth_threshold`（0.9）连续确认后 ×`growth_factor`，`< crawl_threshold`（0.95）连续确认后 ×`crawl_factor`，
   0.95~1.0 为死区（交给第 2 条），`> 1.0` 为轻度恶化（第 3 条），`> shrink_threshold`（1.1，NaN/inf 也归此）立即
   ×`shrink_factor`。调节后有 `cooldown_steps` 冷却。
2. **窗口趋势放大**（`gates.py`）：渐近段每步只降千分之几是健康的，单步比值落在死区；最近 `trend_window` 步累计比值
   `< trend_threshold`（0.995）时 ×`trend_factor`。残差范数不显含 dt，在稳定极限内放大只会降得更快，反馈符号正确。
3. **轻度收缩要连续确认、并由同一窗口仲裁**：显式稳态的残差范数本不单调，单步上升立刻收缩会让 CFL 单向棘轮下滑到
   下限。轻度收缩要连续 `mild_shrink_confirm_steps` 步，且窗口内没有累计下降；重度收缩不受约束（安全底线）。
4. **迟滞带**：收缩用独立阈值 `trend_shrink_threshold`（1.0，窗口确实变差），与放大阈值 0.995 之间留迟滞带，消除在
   稳定边界上的极限环。
5. **软上限**：每次收缩把上限设为 `ceiling_backoff`（0.95）× 收缩前 CFL，此后放大不越过它，从上方收敛到稳定边界、
   不再周期性穿越；只有收缩发生在 `cfl_start` 之上时才以它为下界（`cfl_start` 只是调用方的断言）。窗口清空后的盲区
   内只对持续 `trend_window // 2` 步的恶化反应。`ceiling_release_steps` 步后缓慢释放；`reset()`（换阶）作废。
6. **绝对基准**（`_res_best`，自构造 / `reset()` 以来的最小有限残差），只阻止动作、不主动收缩：(a)
   `R / R_best > grow_block_ratio`（2.0）时禁止任何放大——已离开收敛盆的轨迹上局部下降不携带稳定性信息；(b) 软上限
   释放额外要求当前残差不差于上限设定时。健康轨迹上 `R / R_best` 恒为 1，两条永不触发。
7. **边界参数协调**：`cfl_min > cfl_max` 时把 `cfl_min` 钳到 `cfl_max` 并告警，`cfl_start` 钳进 `[cfl_min, cfl_max]`；
   收缩与软上限都双向钳制——名为收缩的分支不得把 CFL 调高或越过 `cfl_max`。
8. **越界之后收缩救不回来，所以 `cfl_max` 默认压在验证过的稳定边界以下**：失稳特征滞后 CFL 改动 8~17 步，判据看见
   恶化时越界早已发生。默认 `cfl_start` 0.03 / `cfl_max` 0.06 / `cfl_min` 0.01（plate P1 层流 0.10 稳定、0.30
   发散；直接谱测量通量线性极限约 0.39）。比较不同 CFL 的运行要按伪时间而不是迭代步对齐。
9. **启动暂态里的轻度收缩闸门**：暂态中残差可以回升而积分量单调收敛，单步比值把区域轮换误判成失稳。轻度收缩还要
   `R / R_best > mild_shrink_block_ratio`（1.5）；重度收缩不受约束。若出现"闸门挡住收缩之后发散"，应压低 `cfl_max`
   （第 8 条），不要调小这个阈值。这是缓解：跨步比值把"这一步做坏了"与"解在暂态"混在一起，根治要试算 / 接受
   （隐式步就是这样做的），或换一个暂态里单调的监控量。
"""

# 包根 re-export 各子模块的公开名与测试在用的私有名。

from .controller import (  # noqa: F401
    AdaptiveCFLController,
)
from .ser import (  # noqa: F401
    SERCFLController,
)

__all__ = [
    "AdaptiveCFLController",
    "SERCFLController",
]
