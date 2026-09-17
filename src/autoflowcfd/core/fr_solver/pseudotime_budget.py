# -*- coding: utf-8 -*-
"""累计伪时间预算 —— 区分"收敛慢"与"根本还没走到"。

## 为什么需要它（一次真实的、代价很大的误判）

2026-09-15~17 有一条结论被当成最高优先级缺陷追了好几天：plate_demo
（0.5m 方板正对来流）上 Cd 从 0 线性爬到 5.77（文献 1.10~1.18）、滞止面
压力系数中位 +3.41（定常流的上界是 +1）、350 步零曲率。看起来像壁面处理
有系统性缺陷，甚至被记成"C 类默认值决策的前置条件"。

用求解器自己的 dt 算一下累计伪时间就推翻了它：

    局部对流时标 h/U          tau/T = 0.877     （刚走完约一个）
    绕板对流 L_body/U         tau/T = 0.018     需要约  19,200 步
    全域对流 L_domain/U       tau/T = 0.0017    需要约 211,000 步

**350 步只走完了一个绕板特征时间的 1.8%。** 滞止区还没有时间把多余压力从
板边缘泄掉，所以"单调线性上升、零曲率"恰恰是启动暂态最开头的必然形态；
而"定常流总压不可能超过来流滞止压力"这条判据**对暂态不适用**（非定常
Bernoulli 有 ∂φ/∂t 项）。

这个数一直算得出来，只是**从来没被打印出来过**，所以日志里看不出"残差在
降但物理场只走了 1.8%"。本模块把它变成求解器自己报告的一等诊断量。

## 局部时间步进下"累计伪时间"是逐单元的

显式伪时间推进用逐单元 dt（`cfl.py::compute_local_time_step`），所以不存在
单一的"当前时间"。本模块按单元累加 `tau_cell += dt_cell`，再报分位数。

与之比较的时标有三层，含义各不相同，都报出来才不误导：

  1. `h_cell / |u|`（局部）：单元内信息更新一次。tau/T ~ 1 只说明局部量
     （壁面法向速度、边界层剖面）开始成形。
  2. `L_body / U`（物体尺度）：流绕过物体一次 —— **气动力系数有意义的
     最低门槛**。
  3. `L_domain / U`（全域）：整个计算域的流场建立。

`L_body` 取 `sqrt(reference_area)`（求解器已为气动系数算过它）；拿不到时
退回 `L_domain`，并在输出里标明用的是哪个，不悄悄替换。

## 一条重要的限制

低马赫数伪时间预处理开启时（`solver.low_mach_precond_enabled`），dt 是按
**预处理后**的波速定的，压力扰动在伪时间里也按 `c_precond ~ mach_ref*a`
传播而不是物理声速 `a`。所以"压力平衡"这一项的时标是 `L/c_precond` 而不是
`L/a`。本模块只按**对流**时标 `L/U` 报（`c_precond` 与 `U` 同量级，两者
不必分开），并在文档里记下这一点，避免有人拿 `L/a` 去算出一个乐观 11 倍的
数字。
"""

from typing import Dict, Optional

import numpy as np

__all__ = ["pseudo_time_budget", "format_pseudo_time_budget"]


def _extent(mesh) -> Optional[float]:
    """计算域最大边长（取节点坐标包围盒）。"""
    nodes = getattr(mesh, "_node_coords", None)
    if nodes is None:
        nodes = getattr(mesh, "node_coords", None)
    if nodes is None:
        return None
    nodes = np.asarray(nodes)
    if nodes.ndim != 2 or nodes.shape[0] == 0:
        return None
    return float((nodes.max(axis=0) - nodes.min(axis=0)).max())


def pseudo_time_budget(
    dt_cells: np.ndarray,
    *,
    vel_inf: float,
    cell_volumes: Optional[np.ndarray] = None,
    body_length: Optional[float] = None,
    domain_length: Optional[float] = None,
    tau_accum: Optional[np.ndarray] = None,
    n_steps: int = 1,
) -> Dict[str, float]:
    """算累计伪时间与它对各层物理时标的比。

    Args:
        dt_cells: 逐单元伪时间步长 (n_cells,) 或 (n_cells, n_sps)（取第 0
            个解点——同一单元各解点共用一个 dt）。
        vel_inf: 自由来流速度大小（m/s）。
        cell_volumes: 逐单元体积，用来给出 `h = V^(1/3)`（局部时标）。
        body_length: 物体特征长度（建议 `sqrt(reference_area)`）。
        domain_length: 计算域最大边长。
        tau_accum: 已累加的逐单元伪时间；None 时按 `n_steps * dt_cells`
            估算（**只在 dt 全程不变时才准确**，所以自适应 CFL 下必须传真
            实累加量）。
        n_steps: 与 `tau_accum=None` 配套的步数。

    Returns:
        dict，键见 `format_pseudo_time_budget`。`tau_over_*` 取不到对应时标
        时为 `nan`（不编造替代值）。
    """
    dt = np.asarray(dt_cells, dtype=float)
    if dt.ndim > 1:
        dt = dt[:, 0]
    tau = np.asarray(tau_accum, dtype=float) if tau_accum is not None \
        else float(n_steps) * dt
    if tau.ndim > 1:
        tau = tau[:, 0]

    u = max(float(vel_inf), 1e-30)
    out: Dict[str, float] = {
        "n_steps": float(n_steps),
        "dt_min": float(dt.min()),
        "dt_median": float(np.median(dt)),
        "dt_max": float(dt.max()),
        "tau_median": float(np.median(tau)),
        "tau_min": float(tau.min()),
        "tau_max": float(tau.max()),
        "estimated": float(tau_accum is None),
    }

    if cell_volumes is not None:
        h = np.asarray(cell_volumes, dtype=float) ** (1.0 / 3.0)
        t_local = h / u
        out["tau_over_local"] = float(np.median(tau / np.maximum(t_local, 1e-300)))
    else:
        out["tau_over_local"] = float("nan")

    for key, L in (("body", body_length), ("domain", domain_length)):
        if L is None or not np.isfinite(L) or L <= 0.0:
            out[f"tau_over_{key}"] = float("nan")
            out[f"steps_for_{key}"] = float("nan")
            out[f"L_{key}"] = float("nan")
            continue
        T = float(L) / u
        out[f"L_{key}"] = float(L)
        out[f"tau_over_{key}"] = out["tau_median"] / T
        # 按"目前的中位 dt"外推到 tau/T = 1 所需的总步数
        out[f"steps_for_{key}"] = T / max(out["dt_median"], 1e-300)
    return out


def format_pseudo_time_budget(b: Dict[str, float], *, compact: bool = False) -> str:
    """把 `pseudo_time_budget` 的结果格式化成日志行。

    `compact=True` 给出可以挂在每步残差行后面的一个字段
    （`tau/T_body=0.018`）；否则给出启动/收尾用的多行摘要。
    """
    if compact:
        v = b.get("tau_over_body")
        if v is None or not np.isfinite(v):
            v = b.get("tau_over_domain", float("nan"))
            tag = "T_dom"
        else:
            tag = "T_body"
        if not np.isfinite(v):
            return "tau/T=n/a"
        return f"tau/{tag}={v:.4f}"

    est = "（按当前 dt 外推）" if b.get("estimated") else ""
    lines = [
        f"   伪时间预算{est}：累计 tau 中位 {b['tau_median']:.3e} s"
        f"（{int(b['n_steps'])} 步，dt 中位 {b['dt_median']:.3e} s，"
        f"min {b['dt_min']:.3e} / max {b['dt_max']:.3e}）",
    ]
    if np.isfinite(b.get("tau_over_local", float("nan"))):
        lines.append(
            f"     局部对流 h/U      ：tau/T = {b['tau_over_local']:.3f}"
            f"   （~1 表示局部量开始成形）")
    for key, label, note in (
        ("body", "物体尺度 L_body/U", "气动力系数有意义的最低门槛"),
        ("domain", "全域 L_domain/U ", "整个计算域流场建立"),
    ):
        r = b.get(f"tau_over_{key}", float("nan"))
        if not np.isfinite(r):
            continue
        lines.append(
            f"     {label}：tau/T = {r:.4f}"
            f"   需要约 {b[f'steps_for_{key}']:.0f} 步   （{note}，"
            f"L = {b[f'L_{key}']:.4g} m）")
    if np.isfinite(b.get("tau_over_body", float("nan"))) and \
            b["tau_over_body"] < 0.5:
        lines.append(
            "     注意：tau/T_body < 0.5 —— 气动力系数与压力分布此刻**还在"
            "启动暂态里**，不能拿去与文献值比、也不能据此判定壁面处理有"
            "缺陷：定常判据（例如「总压不超过来流滞止压力」）对暂态不适用")
    return "\n".join(lines)
