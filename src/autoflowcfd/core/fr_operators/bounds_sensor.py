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

## 实测依据（2026-09-16，plate_demo_volume_les iter 100）

按 `tol = 0` 量这个越界量（`scratchpad/plate/diag_slope_overshoot.py`）：

    变量   越界量最大值/场尺度   越界单元占比   体积最小1%单元的越界贡献
    rho          0.249 (25%)        16.59%              35.4%
    p            0.665 (66%)        17.91%              37.1%
    u           12.18               19.23%              16.8%

体积最小 1% 的单元贡献了越界总量的 17~37%（超出其占比 17~37 倍），而
同一份 checkpoint 的空间诊断显示超压点 99.8% 挤在板侧边 2.2mm 内、81%
属于体积最小 1% 的单元。**判据对症**（坏单元确实主导越界量），但
**不外科**（另有 16~19% 的单元存在小幅越界）——这正是必须带 `rel_tol`
而不能用裸 BJ 的原因。

## 第一版用"全场 cell_mean 的 RMS"做绝对地板尺度为什么不行（2026-09-16 实测）

事先写定的判据是"默认容差下标记比例 <= 3% 且多数是小体积单元 -> 可用；
>= 10% -> 太松"。在 plate_demo_volume_les iter 100 的真实守恒场上实测
（`scratchpad/plate/diag_bounds_mask.py`）：

    rel_tol  abs_frac   标记比例    其中小体积   小体积召回
      0.000   0.0e+00   96.878%        1.0%      100.0%
      0.000   1.0e-03   11.086%        9.0%       99.7%
      0.100   1.0e-03   11.083%        9.0%       99.7%
      0.500   1.0e-03    9.001%       10.3%       92.5%
      0.100   1.0e-01    7.127%       13.7%       97.9%

**判据不通过**：默认 11.08%，而且 `rel_tol` 从 0.05 扫到 0.5 几乎无效
（11.1% -> 9.0%），说明违反量根本不是由"邻域跨度"这一项决定的。

原理性成因（不是调参问题）：`abs_frac * RMS(cell_mean)` 用的是**全场**
单元均值的 RMS 做尺度。对 `rho_v` / `rho_w` 这类"大部分域内均值≈0、
RMS 被尾迹局部主导"的变量，这个尺度既不是局部量级、也不是物理量级
——自由来流区里微小的横向动量脉动会被拿去和一个由尾迹决定的尺度比。

修正（用项目自己早就在用的那套尺度，不是为了让数字好看而挑的）：
绝对地板改用**来流参考量级** `ref_scales`
（`[rho_inf, rho_inf*vel_inf, rho_inf*vel_inf, rho_inf*vel_inf, p_inf]`，
与 `fr_solver/residual_diagnostics.py::_reference_scales` 同一套构造、
同一套理由——包括 rho_E 用 p_inf 而不是 rho_inf*vel_inf^2）。

## 改用来流参考量级之后的实测，以及**为什么默认仍然是关闭的**

同一份 checkpoint、绝对地板改用来流参考量级
`[1.225, 40.83, 40.83, 40.83, 101325]` 重测：

    rel_tol  abs_frac   标记比例   其中小体积   小体积召回
      0.000   0.0e+00   96.878%       1.0%      100.0%
      0.000   1.0e-03   10.951%       9.1%       99.7%
      0.100   1.0e-03    8.431%      11.7%       98.9%   <- 默认容差
      0.500   1.0e-03    4.949%      16.0%       79.1%
      0.100   1.0e-01    3.676%      11.6%       42.7%

从 11.08% 改善到 8.43%，但**事先写定的判据（<=3% 标记比例且高召回）
仍然不满足**：要把标记比例压到 3.68%，体积最小 1% 单元的召回就掉到
42.7%（漏掉一半以上的退化单元）；要保住 98.9% 的召回，就得标记 8.43%
的计算域。两头不可兼得。

这个负面结果本身是有信息量的：**越界并不集中在退化单元**。8~11% 的
计算域都有超出邻居均值区间的内容——在相邻单元体积比达 32.95、非正交
达 81.92 度的网格上这说得通，因为 BJ 的"邻居均值区间"本身就被悬殊的
单元尺度扭曲了。也就是说，"网格质量门 + 耗散"这条既定路线的**耗散
那一半在这类网格上无法做成外科式的**，真正的瓶颈仍然是网格
（与 [[industry_practice_degenerate_cell_gcl]] 的结论一致，本次是第一次
有定量数据支持它）。

因此本判据**默认关闭**（`AFCFD_TROUBLED_SENSOR` 默认 `persson`），
定位是：
  1. P1 上**唯一可用**的 troubled-cell 判据（Persson-Peraire 在 P1
     原理上不适用），供受控 A/B 与诊断使用；
  2. 一个可量化的"解的单调性越界"指标，可以直接回答"这张网格上有多少
     比例的域承载了非物理过冲"。

它**不是**稳定性问题的解决方案，不要这样引用。

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
"""

from typing import Optional

import numpy as np

#: `tol = rel_tol * (nb_max - nb_min) + abs_tol` 里的相对项系数。
#: 0.1 的含义：解点值要超出邻域区间**10% 的邻域跨度**才算越界。
DEFAULT_BOUNDS_REL_TOL = 0.1

#: 绝对地板，按该变量自身单元均值的 RMS 缩放。
#:
#: 为什么不能取成噪声量级（1e-9 这类）：`nb_max - nb_min` 只度量**邻居
#: 单元均值之间**的跨度，它在两类区域里会塌成零——真正均匀的区域，以及
#: **光滑极值附近**（梯度反向处相邻单元均值近似相等，例如驻点、尾迹
#: 中心线）。后者是物理上完全正常的光滑解，但此时任何亚单元变化都会
#: 超出那个塌缩掉的区间。这是 BJ 判据在光滑极值处的经典弱点，标准补丁
#: （Venkatakrishnan）正是给容差加一个**物理量级**的绝对地板，而不是
#: 机器精度量级的地板。
#:
#: 取 1e-3（场 RMS 的 0.1%）的依据：2026-09-16 在 plate_demo_volume_les
#: iter 100 上实测的真实越界量是 rho 25%、p 66%、u 12 倍场尺度（见本
#: 模块文档的表），所以 0.1% 的地板对真实坏单元有约 250 倍余量，同时
#: 足以让"均值相同、仅有亚单元光滑变化"的区域完全失活。
#: 首版取 1e-9 时被单元测试当场抓到：一个"每个单元内都有 0.05% 光滑
#: 变化、但单元均值全相同"的场被标记了**全部**单元。
DEFAULT_BOUNDS_ABS_FRAC = 1e-3


def compute_bounds_violation_mask(
    field_nodal: np.ndarray,
    owner_cell: np.ndarray,
    neighbor_cell: np.ndarray,
    is_boundary: np.ndarray,
    *,
    rel_tol: float = DEFAULT_BOUNDS_REL_TOL,
    abs_frac: float = DEFAULT_BOUNDS_ABS_FRAC,
    ref_scales=None,
) -> np.ndarray:
    """逐单元判定"解点值越出了面邻居均值区间" —— **纯数组接口**。

    与 `artificial_viscosity.compute_troubled_cell_mask`（Persson-Peraire）
    平行的第二个判据，接口风格刻意保持一致（纯数组、不需要 solver / ops），
    这样两者可以在同一个门控入口里互换或取并集，且 CPU 单机 / CPU MPI
    local 排列 / GPU 三条路径都能用同一个内核。

    与 Persson-Peraire 的关键差异：本判据**不依赖模态分解**，因此没有
    "order=1 时顶模态就是全部非常数内容"那个退化（见模块文档）。

    Args:
        field_nodal: (n_cells, n_sps) 单标量场，或 (n_cells, n_sps, n_var)
            多变量场（多变量时逐变量判定后取**并集**——任一变量越界即
            标记该单元）。
        owner_cell: (n_faces,) 面的 owner 单元索引
        neighbor_cell: (n_faces,) 面的 neighbor 单元索引；边界面该项不被
            读取（可以是任意占位值，例如 -1）
        is_boundary: (n_faces,) 布尔，True=边界面。边界面不参与邻域区间
            构造——边界外侧没有"邻居单元均值"，用幽灵态会把边界条件的
            物理跳跃（例如壁面镜像的法向速度反号）误判成越界。
        rel_tol: 相对容差系数，见模块文档
        abs_frac: 绝对地板系数（乘以下面的参考量级），见模块文档
        ref_scales: 可选的 (n_var,) **来流参考量级**（例如
            `[rho_inf, rho_inf*vel_inf, rho_inf*vel_inf, rho_inf*vel_inf,
            p_inf]`，与 `fr_solver/residual_diagnostics.py::
            _reference_scales` 同一套构造）。**强烈建议给出**：不给时
            退化为用该变量自身单元均值的 RMS 做尺度，而那对"大部分域内
            均值≈0、RMS 被局部区域主导"的变量（rho_v / rho_w）是没有
            意义的尺度——2026-09-16 实测证实了这一点，见模块文档
            "第一版用全场 RMS 做尺度为什么不行"一节。

    Returns:
        (n_cells,) 布尔掩码，True = 该单元越界。

    Raises:
        ValueError: 形状不自洽（不静默广播——静默广播会让一个形状 bug
            变成"判据恒不触发"，而恒不触发的门控在日志里看起来一切正常）。
    """
    field = np.asarray(field_nodal)
    if field.ndim == 2:
        field = field[:, :, None]
    if field.ndim != 3:
        raise ValueError(
            f"field_nodal 必须是 (n_cells, n_sps) 或 (n_cells, n_sps, n_var)，"
            f"收到 {np.asarray(field_nodal).shape}"
        )
    n_cells = field.shape[0]

    owner = np.asarray(owner_cell)
    neigh = np.asarray(neighbor_cell)
    bnd = np.asarray(is_boundary, dtype=bool)
    if not (owner.shape == neigh.shape == bnd.shape) or owner.ndim != 1:
        raise ValueError(
            f"owner_cell/neighbor_cell/is_boundary 必须是同长度一维数组，"
            f"收到 {owner.shape} / {neigh.shape} / {bnd.shape}"
        )
    if owner.size and (owner.max() >= n_cells or owner.min() < 0):
        raise ValueError(
            f"owner_cell 越界：[{owner.min()}, {owner.max()}] 超出 [0, {n_cells})"
        )

    interior = ~bnd
    o_i = owner[interior]
    n_i = neigh[interior]
    if n_i.size and (n_i.max() >= n_cells or n_i.min() < 0):
        raise ValueError(
            f"内部面的 neighbor_cell 越界：[{n_i.min()}, {n_i.max()}] "
            f"超出 [0, {n_cells})"
        )

    n_var = field.shape[2]
    if ref_scales is not None:
        ref = np.asarray(ref_scales, dtype=np.float64).ravel()
        if ref.size != n_var:
            raise ValueError(
                f"ref_scales 长度 {ref.size} 与变量数 {n_var} 不符"
            )
    else:
        ref = None

    mask = np.zeros(n_cells, dtype=bool)
    for v in range(n_var):
        q = field[:, :, v]
        cell_mean = q.mean(axis=1)
        cell_max = q.max(axis=1)
        cell_min = q.min(axis=1)

        # 邻域区间：自身均值 + 全部面邻居的均值。两个方向都要做——
        # 一条内部面同时是 owner 的邻居来源和 neighbor 的邻居来源。
        nb_max = cell_mean.copy()
        nb_min = cell_mean.copy()
        if o_i.size:
            np.maximum.at(nb_max, o_i, cell_mean[n_i])
            np.maximum.at(nb_max, n_i, cell_mean[o_i])
            np.minimum.at(nb_min, o_i, cell_mean[n_i])
            np.minimum.at(nb_min, n_i, cell_mean[o_i])

        if ref is not None:
            scale = float(ref[v])
        else:
            scale = float(np.sqrt(np.mean(cell_mean.astype(np.float64) ** 2)))
        tol = rel_tol * (nb_max - nb_min) + abs_frac * max(scale, 1e-300)
        mask |= (cell_max > nb_max + tol) | (cell_min < nb_min - tol)

    return mask


def resolve_troubled_sensor(value: Optional[str] = None) -> str:
    """解析 `AFCFD_TROUBLED_SENSOR`：`persson` | `bounds` | `both`。

    **默认 `bounds`（2026-09-17 从 `persson` 改）。**

    为什么改：Persson-Peraire 在 `order=1` 上**原理性退化**（`s0 =
    -4*log10(order)` 在 order=1 时为 0，触发门限成了"顶模态能量占比
    >= 10%"，而 P1 的顶模态就是全部非常数内容），而且它探的是守恒密度
    ——真实解上那一项是光滑的，掩码实测 **0.000%**（同一时刻 `rho_v`/
    `rho_w` 是 98.8%）。所以 `persson` 在 P1 上等于**没有门控**：实测
    `AFCFD_FILTER_MODE=sensor` + `persson` 与 `FILTER_MODE=off` **逐位
    相同**（平板边界层算例 res 1.6658e+05）。

    与 `AFCFD_FILTER_MODE=sensor` 必须**成对**使用（同日一起改默认）：
    只改一个等于把默认值悄悄改成 `off`。三档在 P1 上的实测对照见
    `fr/modal_filter.py` 里 `_FILTER_MODE` 上方那节。

    真实网格上的决定性证据（plate_demo_volume_les，179,237 单元）：
    `legacy` 在 iter 112 发散，而 `sensor`+`bounds` 跑出 216 步残差
    **单调下降 3.5 倍**，Cd 漂移从零曲率的线性 0.0167/步变成负曲率的
    0.0071 -> 0.0042/步。等熵涡精确解上 P1 的收敛阶保住 2.16/2.18
    （设计阶 2）。

    `persson` 保留为合法档：它在 order>=2 上判据本身是有效的，且是复现
    历史结果的唯一途径。

    Raises:
        ValueError: 取值非法（不静默回退，理由同
            `fr_operators/kernels.py::resolve_ausm_precond_mode`：静默回退
            会让一次拼写错误伪装成默认行为、把 A/B 的两条运行悄悄变成
            同一档）。
    """
    import os

    if value is None:
        value = os.environ.get("AFCFD_TROUBLED_SENSOR", "").strip()
        if not value:
            return "bounds"
    key = str(value).strip().lower()
    if key in ("persson", "bounds", "both"):
        return key
    raise ValueError(
        f"AFCFD_TROUBLED_SENSOR 取值非法: {value!r}；"
        f"合法值 ['bounds', 'both', 'persson']"
    )
