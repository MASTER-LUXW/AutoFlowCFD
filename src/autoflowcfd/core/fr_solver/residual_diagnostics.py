"""AutoFlowCFD V2.0 - 残差诊断（参照 Fluent/STAR-CCM+ 的按方程归一化 +
最大残差定位方案，2026-09-12）。

背景（真实排查发现，见本次会话 cube_demo 791,492 单元真实网格 P0 阶段
"残差持续增大"排查记录）：此前 `FRState.get_residual_norm()` 把全部
5 个守恒变量（rho, rho_u, rho_v, rho_w, rho_E）合并进同一个 RMS/L2
范数——这 5 个变量的物理量纲/量级天差地别（本项目真实来流条件下
rho~1.2, rho_u~40, rho_E~2.6e5，energy 比 density 大 5 个数量级），
平方求和后 energy 分量的绝对贡献几乎总是压倒性的，其余方程的收敛/
发散行为完全被淹没。同一次排查还发现：79万单元里少数几个方块前驻点
单元（AUSM+up 在低马赫数驻点区域的已知病理，与网格/湍流模型无关）
贡献的残差幅值比全局RMS还大，但只用一个合并 RMS 数字完全看不出来，
排查者只能手写诊断脚本反复重建 solver 才定位到具体单元——这在
Fluent/STAR-CCM+ 里是 CLI 本该直接打印出来的信息：
  - Fluent："Scaled Residuals"——每个方程（continuity/x-mom/y-mom/
    z-mom/energy/k/epsilon...）分别算一个残差，每个都除以该方程自己
    的参考通量量级做归一化，同时显示当前值。
  - STAR-CCM+：同样按方程分别报告 RMS 残差，且提供"Max"（最大残差）
    监视器，可以直接看到最大残差出现的位置。

本模块提供的诊断（不改变、不替代现有 `get_residual_norm()` 驱动的
`residual_drop_threshold`/Order Continuation 升阶判据——那条判据继续
用现有的合并 RMS，保持已经验证过的行为不变；本模块只是在打印层面
新增诊断信息，帮助用户/排查者一眼区分"大部分单元已收敛、被少数异常
单元卡住统计量"与"真的在整体发散"这两种此前混淆过的情形）：

1. `compute_scaled_residuals`：每个守恒变量分别算 RMS，除以基于自由
   来流条件构造的参考量级（rho_inf/rho_inf*vel_inf/p_inf），得到
   量纲一致、可以互相比较的"scaled residual"，与 Fluent 的呈现方式
   对齐。
2. `find_max_residual_location`：全场（含全部变量）绝对值最大的
   dU_dt 分量所在的 (cell, sp, var)，供打印"Max: X at cell Y (var Z)"
   这类诊断行——与 STAR-CCM+ 的 Max 监视器对齐。
"""

from typing import NamedTuple, Optional

import numpy as np


class ResidualDiagnostics(NamedTuple):
    """一次残差诊断的完整结果。"""
    rms_per_var: np.ndarray          # (n_vars,) 未归一化 RMS，逐变量
    scaled_rms_per_var: np.ndarray   # (n_vars,) 除以参考量级后的 RMS
    max_abs: float                   # 全场 |dU_dt| 最大值（原始量纲）
    max_abs_cell: int                # 最大值所在单元索引
    max_abs_sp: int                  # 最大值所在解点索引
    max_abs_var: int                 # 最大值所在变量索引（0=rho,1..3=动量,4=能量）


_VAR_NAMES = ("rho", "rho_u", "rho_v", "rho_w", "rho_E")


def _reference_scales(freestream: dict, n_vars: int) -> np.ndarray:
    """按自由来流条件构造每个守恒变量的参考量级（Fluent"scaled
    residual"同一个思路：用该方程里量本身天然的量级做归一化，不是
    用初始残差——两者不冲突，`residual_drop_threshold` 仍然用初始
    残差归一化，本函数是另一个互补的诊断视角）。

    rho_E 用 p_inf 做参考（rho*E 与压力同量纲，J/m^3=Pa；本项目真实
    来流条件下 rho*E 的绝对量级由静温对应的内能 p_inf/(gamma-1) 主导，
    不是动能项，用 p_inf 而不是 rho_inf*vel_inf^2 更贴近其真实量级，
    避免低马赫数工况下参考值过小、把本来正常的能量残差错误放大）。
    """
    # 直接取键，不给兜底（2026-09-24）：`freestream` 对全部求解器类无条件
    # 设置，兜底永远不会生效；一旦某条路径真的丢了字段，`.get(k, 33.33)`
    # 会把缺陷伪装成"用了一个合理的来流"。真实例子：
    # `test_bounds_sensor_mirror` 一直传着键名全错的
    # `{"rho": 1.0, "u": 1.0, ...}`，作者想要单位量级，实际用的是兜底的
    # p_inf=101325 —— 没人发现，因为什么都没报错。
    rho_inf = max(freestream["rho_inf"], 1e-10)
    vel_inf = max(freestream["vel_inf"], 1e-10)
    p_inf = max(freestream["p_inf"], 1e-10)

    scales = np.array([
        rho_inf,                 # rho
        rho_inf * vel_inf,       # rho_u
        rho_inf * vel_inf,       # rho_v
        rho_inf * vel_inf,       # rho_w
        p_inf,                   # rho_E
    ])
    if n_vars > 5:
        # k/omega（若 state.U 里真的携带，见调用方文档——本项目实际把
        # k/omega 维护在 turb_model.k_field/omega_field，state.U 对应
        # 槽位是死代码，这里只是防御性地不让形状不匹配崩溃，不代表
        # 这两个槽位的诊断有实际意义）。
        scales = np.concatenate([scales, np.ones(n_vars - 5)])
    return scales


def compute_scaled_residuals(dU_dt: np.ndarray, freestream: dict) -> "ResidualDiagnostics":
    """计算按方程分别归一化的残差 + 最大残差定位。

    Args:
        dU_dt: (n_cells, n_sps, n_vars) 残差数组（`state.dU_dt`，
            `step()` 调用后即为最新值，符号约定见 fr_solver/step.py）。
        freestream: `solver.freestream` 字典（rho_inf/vel_inf/p_inf）。

    Returns:
        ResidualDiagnostics
    """
    n_vars = dU_dt.shape[2]

    rms_per_var = np.sqrt(np.mean(dU_dt.reshape(-1, n_vars).astype(np.float64) ** 2, axis=0))
    scales = _reference_scales(freestream, n_vars)
    scaled_rms_per_var = rms_per_var / scales

    flat_idx = np.argmax(np.abs(dU_dt))
    cell_idx, sp_idx, var_idx = np.unravel_index(flat_idx, dU_dt.shape)
    max_abs = float(np.abs(dU_dt[cell_idx, sp_idx, var_idx]))

    return ResidualDiagnostics(
        rms_per_var=rms_per_var,
        scaled_rms_per_var=scaled_rms_per_var,
        max_abs=max_abs,
        max_abs_cell=int(cell_idx),
        max_abs_sp=int(sp_idx),
        max_abs_var=int(var_idx),
    )


#: `cell_volume_percentile` 的缓存：键是 `id(cell_volumes)` 与数组长度，
#: 值是 0~100 的体积分位数组。分位数要对全部单元做一次 argsort（79 万
#: 单元约 0.1s），逐步重算会白白拖慢每一步诊断。
_VOL_PCT_CACHE: dict = {}


def cell_volume_percentile(cell_volumes) -> Optional[np.ndarray]:
    """逐单元的体积分位（0 = 全场最小，100 = 最大）。

    为什么把它接到残差诊断里（2026-09-16，真实排查驱动）：
    plate_demo_volume_les 上 P1 固定 CFL 0.03 的三条对照运行，100 步里
    最大残差**恒定落在同一个单元**（cell18708，rho_E，8.7e10 → 3.65e11），
    随后全局发散。要判断"这是退化单元机制还是壁面处理机制"，当时只能
    另写脚本重新加载整张体网格（约 10 分钟）算出该单元体积分位 0.523%、
    并统计出超压点 81% 落在体积最小的 1% 单元里、99.8% 距板侧边 2cm 内。

    那次结论是决定性的（LES 网格上主导机制是边缘退化薄单元，不是壁面
    处理），但**这条信息本该在日志里一眼看到**：残差被 `det(J)` 除，
    体积极小的单元天然把任何通量不平衡放大若干个量级，所以"最大残差
    单元的体积分位"是区分这两类机制最直接的单个数字。

    Args:
        cell_volumes: (n_cells,) 单元体积；None 时返回 None（调用方
            按"没有几何信息"处理，不报错——合成网格测试替身可能不带它）。

    Returns:
        (n_cells,) 的 0~100 分位数组，或 None。
    """
    if cell_volumes is None:
        return None
    vols = np.asarray(cell_volumes)
    if vols.ndim != 1 or vols.size == 0:
        return None
    key = (id(cell_volumes), vols.size)
    cached = _VOL_PCT_CACHE.get(key)
    if cached is not None and cached.size == vols.size:
        return cached
    pct = np.argsort(np.argsort(vols)) * (100.0 / max(vols.size - 1, 1))
    _VOL_PCT_CACHE[key] = pct
    return pct


def format_scaled_residual_line(diag: "ResidualDiagnostics",
                                cell_volumes=None) -> str:
    """格式化成一行可读文本，供 CLI 打印（与 Fluent scaled residuals
    表格、STAR-CCM+ Max 监视器同一个信息量级，压缩成单行）。

    Args:
        diag: `compute_scaled_residuals` 的结果
        cell_volumes: 可选的 (n_cells,) 单元体积。给出时在最大残差定位
            后面附上该单元的体积分位（`vol 0.5%` 表示它属于全场体积最小
            的 0.5%）——见 `cell_volume_percentile` 文档里记录的那次
            排查：这个数字直接区分"退化单元机制"与"壁面处理机制"。
    """
    n_vars = diag.scaled_rms_per_var.shape[0]
    parts = []
    for i in range(min(n_vars, len(_VAR_NAMES))):
        parts.append(f"{_VAR_NAMES[i]}={diag.scaled_rms_per_var[i]:.3e}")
    scaled_str = " ".join(parts)
    var_name = _VAR_NAMES[diag.max_abs_var] if diag.max_abs_var < len(_VAR_NAMES) else f"var{diag.max_abs_var}"
    loc = f"{var_name}@cell{diag.max_abs_cell}"
    pct = cell_volume_percentile(cell_volumes)
    if pct is not None and diag.max_abs_cell < pct.size:
        loc += f", vol {pct[diag.max_abs_cell]:.2f}%"
    return (f"Scaled[{scaled_str}] | Max={diag.max_abs:.3e} "
            f"({loc})")


class SolverDivergedError(RuntimeError):
    """残差变成 inf/nan —— 求解已经不可恢复，必须立刻中止。

    为什么需要一个专门的异常而不是"打印警告后继续"（2026-09-16，真实
    事故驱动）：plate_demo_volume_les 上一条 AUSM+up 预处理档对照运行在
    iter 103 катastrophic 发散（Cd 冲到 3.1e48）、iter 104 残差 inf、
    iter 105 起全部 nan，而求解循环**毫不在意地继续迭代**——直到人工发现
    为止已经白烧了若干步机时，而且每一步都照常调用 `checkpoint_callback`，
    会把 NaN 状态写进 checkpoint、并在收尾时用 NaN 覆盖 `final_state.pkl`。

    工业级求解器在这里的标准行为是立刻中止并给出非零退出码，而不是产出
    一份看起来完整、内容全是 NaN 的结果目录。
    """


def check_residual_finite(res, iteration: int, order=None,
                          last_finite=None, extra_hint: str = "") -> None:
    """残差非有限即抛 `SolverDivergedError`。

    必须在**调用 checkpoint 回调之前**检查，否则 NaN 状态会被落盘。

    Args:
        res: 本步残差范数
        iteration: 1 起算的迭代号（打印用）
        order: 当前多项式阶数，None 表示调用方不区分阶数
        last_finite: 上一次有限的残差值，用于告诉用户"从哪里开始坏的"
        extra_hint: 调用方补充的定位提示（例如分布式路径的 rank 信息）

    Raises:
        SolverDivergedError: `res` 不是有限值。
    """
    if np.isfinite(res):
        return
    tag = "" if order is None else f"P{order} "
    lf = "（无）" if last_finite is None else f"{last_finite:.6e}"
    raise SolverDivergedError(
        f"{tag}Iter {iteration}: 残差为 {res} —— 求解已发散，立刻中止。\n"
        f"  上一个有限残差 = {lf}\n"
        f"  本步之后不再迭代、不再写 checkpoint（避免用 NaN 覆盖已有的\n"
        f"  正常 checkpoint 与 final_state）。\n"
        f"  常见成因（按本项目实测频率排序）：\n"
        f"    1. CFL 超过该网格/阶数的真实稳定边界 —— 用 --cfl-start/\n"
        f"       --cfl-max 下调；注意越界一次之后收缩救不回来（见\n"
        f"       adaptive_cfl.py 模块文档第 12 条）\n"
        f"    2. 网格质量门未通过而用 --skip-quality-check 强行求解 ——\n"
        f"       退化单元会把残差放大若干个量级\n"
        f"    3. AUSM+up 预处理档设成了 legacy（AFCFD_AUSM_PRECOND_MODE）\n"
        f"       —— 该档在固壁上有 7.3 倍虚假超压，实测会在固定 CFL 0.03\n"
        f"       下于百步量级发散，见 fr_operators/kernels.py 的 PRECOND_*\n"
        f"       常量注释\n"
        f"  先看日志里最后几行 `Max=...(var@cellN, vol X%)` 的 `vol X%`：\n"
        f"  那是最大残差单元的体积分位。若它持续很小（<1%，即该单元属于\n"
        f"  全场体积最小的 1%），成因是上面第 2 条（退化薄单元被 1/det(J)\n"
        f"  放大），换 CFL 或换通量格式都只能推迟发散、不能避免；若它是\n"
        f"  正常量级，才去查第 1、3 条。\n"
        + (f"  {extra_hint}\n" if extra_hint else "")
    )
