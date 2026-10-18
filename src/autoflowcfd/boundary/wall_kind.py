"""物理无滑移固壁的唯一判据。

壁面距离（湍流模型）、壁面模型（WMLES）、BJ 越界判据的 Dirichlet 表、气动力积分都只认"静止的不可穿透无滑移
固壁"：网格类型 WALL（幽灵态配置里 `physical_no_slip=True`）；滑移壁（SLIP_WALL、隧道壁）不算。用户的
`bc_overrides` 覆盖某个组时以覆盖后的配置为准。

2026-10-17 以前（I-92）四处各写一份：Dirichlet 表回退到 `is_no_slip`（WMLES 下被关掉的那个开关）再缺省为真，
WMLES 与气动力缺省为真；壁面距离只读网格类型，不认 `bc_overrides`。
"""

from typing import Dict, Optional, Set


def is_physical_no_slip_wall(cfg: dict) -> bool:
    """幽灵态配置是否为物理无滑移固壁：`type == "WALL"` 且 `physical_no_slip`（直接构造、没有该键的配置取
    `is_no_slip`，再缺省为真——与 `build_boundary_ghost_provider` 的类型表同一语义）。"""
    return cfg.get("type") == "WALL" and bool(cfg.get("physical_no_slip", cfg.get("is_no_slip", True)))


def physical_no_slip_wall_groups(bc_types: Dict[str, str], bc_overrides: Optional[Dict[str, dict]] = None) -> Set[str]:
    """物理无滑移固壁的组名：网格类型为 WALL 的组，被 `bc_overrides` 覆盖的组按覆盖后的配置判断。"""
    overrides = bc_overrides or {}
    out = set()
    for name, bc in bc_types.items():
        if name in overrides:
            if is_physical_no_slip_wall(overrides[name]):
                out.add(name)
        elif bc == "WALL":
            out.add(name)
    return out
