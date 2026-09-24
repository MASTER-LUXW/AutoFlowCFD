"""AutoFlowCFD V2.0 - 邻居极值越界（BJ 型）troubled-cell 判据。

## 为什么需要第二个判据

项目对"退化单元把残差放大若干个量级"的既定路线是"网格质量门 + 耗散"
（见 `ProjectFiles` 与项目记忆 `industry_practice_degenerate_cell_gcl`）。
但耗散这一半在**生产阶数 P1 上目前是空的**：

- **人工粘性 / Persson-Peraire 门控**：实测 A/B 前 51 步残差与 Cd 逐字符
  相同（精确无操作），代码里已 warn。**根本原因（2026-09-16 在真实
  checkpoint 上直接量出来，比此前的阈值论证更确切）**：生产门控探的是
  **守恒密度**（`DEFAULT_SENSOR_VAR_INDEX = 0`），而在
  plate_demo_volume_les iter 100 的真实解上，order=1 的 Persson 掩码
  按变量分别是

      rho    0.000%      <- 生产门控实际用的这个，掩码**完全为空**
      rho_u  3.546%
      rho_v  98.827%
      rho_w  98.756%
      rho_E  0.000%

  也就是说 `AFCFD_FILTER_MODE=sensor` + `persson` 在这个算例上与 `off`
  等价（一个单元都不标），而真正携带非光滑内容的是**横向动量**——
  与"P2 的失效模态在能量上而传感器探密度"是同一类缺陷，现在在 P1 上
  也量到了。

  （一处自我更正：此前把这件事记成"Persson 在 order=1 时门限退化到
  不可能满足、掩码恒空"。用合成随机场实测，order=1 下它其实会触发
  190/200；所以"恒空"的说法不对，真实原因是**被探的那个变量恰好光滑**。）
- **模态滤波**：`legacy` 档把唯一的非常数阶清掉（P1 恒等于 P0），
  `mild` 档 `sigma_top=0.99` 又因为每个 RK stage 都施加而随步数复合
  累积（0.99^300 ~ 0.05），不是"轻微"。

于是 P1 在质量门未通过的网格上没有任何抑制机制。

## 判据本身

Barth-Jespersen 型：一个单元"可疑"当且仅当它的解点值**超出了由它自己
与全部面邻居的单元均值张成的区间**：

    nb_max = max(mean_self, mean_of_each_face_neighbour)
    nb_min = min(mean_self, mean_of_each_face_neighbour)
    越界    <=>  max_sp > nb_max + tol   或   min_sp < nb_min - tol

**为什么线性场不会被误判**：一维均匀网格上取线性场，单元 i 的均值等于
其形心值，解点极值为 `形心 ± h/2 * grad`，而邻居均值为
`形心 ± h * grad` —— 解点极值严格落在邻居均值区间**内部**。这是 BJ
判据的经典性质（限制器在线性场上恒不激活），与阶数无关，所以它在 P1
上**不退化**，正是 Persson-Peraire 缺的那一半。

非均匀/退化网格上 BJ 会有已知的虚假激活（相邻单元尺度差异大时邻居均值
区间收窄）。标准补丁是 Venkatakrishnan 的 epsilon：容差取

    tol = rel_tol * (nb_max - nb_min) + abs_tol

第一项让"邻域本身变化就很小"的光滑区自动失活（相对判据），第二项是
绝对地板，防止 `nb_max == nb_min` 的均匀区因浮点噪声触发。

## 标定与"为什么不外科"（数据表见 ProjectFiles）

在 plate_demo_volume_les iter 100 的真实守恒场上做过完整标定，两条结论
决定了本判据现在的形态，完整数据表见
`ProjectFiles/V2.0/20_判据标定-BJ型越界判据在真实网格上的定量标定.md`：

1. **绝对地板必须用来流参考量级**，不能用"全场 cell_mean 的 RMS"。后者
   对 `rho_v`/`rho_w` 这类"大部分域内均值约 0、RMS 被尾迹局部主导"的
   变量既不是局部量级也不是物理量级，标记比例 11.08%；改用
   `ref_scales`（与 `fr_solver/residual_diagnostics.py::_reference_scales`
   同一套构造）后降到 8.43%。
2. **越界并不集中在退化单元**：要把标记比例压到 3.68%，体积最小 1% 单元
   的召回就掉到 42.7%；要保住 98.9% 的召回就得标记 8.43% 的计算域，两头
   不可兼得。这说明"网格质量门 + 耗散"这条既定路线的**耗散那一半在这类
   网格上做不成外科式的**，瓶颈仍然是网格（与项目记忆
   `industry_practice_degenerate_cell_gcl` 一致，这是第一份定量数据）。

本判据**不是**稳定性问题的解决方案，不要这样引用。它的定位是：P1 上
唯一可用的 troubled-cell 判据（Persson-Peraire 在 P1 原理上不适用），
以及一个可量化的"解的单调性越界"指标。默认值见
`troubled_sensor_mode.py::resolve_troubled_sensor`（2026-09-17 起是
`bounds`，真实网格上让 `legacy` 的"iter 112 发散"变成"216 步残差单调
下降 3.5 倍"）。

## 施加方式与守恒性

本模块只产出布尔掩码，不施加任何操作。消费方是
`core/fr_solver/filter.py::build_sensor_gated_filter_func_arrays`——它对
被标记单元施加完整的模态滤波矩阵、其余单元完全不动。在 P1 上这等价于
**把被标记单元局部降到 P0**（troubled-cell 降阶，标准做法）：常数不可能
过冲，所以单调；模态滤波对常数模态的 `sigma(0) = 1`，所以单元的模态
常数分量不变。

**关于守恒性的诚实说明**：模态常数分量等于"守恒均值 ∫u dV / V"的前提是
`det(J)` 在单元内为常数。直边四面体与直棱柱挤出满足这一点（本项目
`compute_native_tet_jacobians` 的实现就基于"直边单元 Jacobian 逐单元
为常数"这一事实），曲边单元不满足。所以在曲边网格上这个门控是"近似
守恒"，不是精确守恒——这一点必须写清楚，不能因为"legacy 档一直这么
干"就默认它精确。

同理，本模块算判据用的 `cell_mean` 取解点的**算术**平均而非求积加权
平均：它只是一个判据（决定"要不要动这个单元"），不参与任何守恒投影，
算术平均足够且更便宜。


## 文件分工(2026-09-24 拆包, 原 506 行)

    scatter.py           按面把邻居极值散射累加到单元（BJ 包络的底层归约）
    mask.py              BJ 型越界判据：解点值超出顶点邻域极值区间即标记

本 `__init__.py` re-export 全部既有名, 所以全仓库导入一字不改。
"""

from .constants import (  # noqa: F401
    DEFAULT_BOUNDS_ABS_FRAC,
    DEFAULT_BOUNDS_REL_TOL,
)
from .scatter import (  # noqa: F401
    _scatter_minmax,
)
from .mask import (  # noqa: F401
    compute_bounds_violation_mask,
)


# 档位解析（`AFCFD_TROUBLED_SENSOR`）在更上一层的 `troubled_sensor_mode.py`
# （2026-09-19 拆出）。注意是 `..`：本模块 2026-09-24 变成子包之后，`.`
# 的含义下沉了一层。这条 re-export 让全仓库 8 处
# `from ...bounds_sensor import resolve_troubled_sensor` 一个字都不用改。
from ..troubled_sensor_mode import resolve_troubled_sensor  # noqa: F401
__all__ = [
    "resolve_troubled_sensor",
    "DEFAULT_BOUNDS_ABS_FRAC",
    "DEFAULT_BOUNDS_REL_TOL",
    "compute_bounds_violation_mask",
]
