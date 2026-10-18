"""边界条件类型：按属性名识别、用户显式指定与完整性检查（唯一实现）。

边界条件类型是物理输入。面网格的属性名只用来**识别明确无歧义**的类型：名字按非字母数字字符切成词，英文关键字
按整词匹配（`sym` 不会命中 `asymmetric`），中文关键字按子串匹配（中文没有分词符）；没有命中任何类型、或同时命中
两种类型（例如 `TUNNEL_WALL`）的组记为 `UNSPECIFIED`，必须由用户用 `--bc-type 组名=类型` 显式给出。体网格生成与
导入要求每个组都有确定的类型（`require_specified_bc_types`），否则报错并列出缺类型的组。

2026-10-17 以前（I-132）：按子串匹配（`sym` 命中 `asymmetric`，`exit`/`surface` 同类误判）；`farfield`/`freestream`
被归为**滑移壁**（不可穿透），而求解器有真正的远场（来流特征边界）类型；没命中的名字与取不到属性名的面一律默认
WALL，只给 warning；用户没有任何途径指定类型。
"""

import re
from typing import Dict, Iterable, Optional

from loguru import logger

#: 求解器支持的边界条件类型（`core/fr_solver/boundary/ghost.py` 的类型表与周期边界）
SOLVER_BC_TYPES = ("WALL", "SLIP_WALL", "VELOCITY_INLET", "PRESSURE_OUTLET", "SYMMETRY", "FARFIELD", "PERIODIC")

#: 名字无法确定类型的组
UNSPECIFIED = "UNSPECIFIED"


class BoundaryTypeError(ValueError):
    """边界条件类型的指定有误或不完整（缺类型、组名不存在、类型不受支持）。"""

#: 英文关键字（整词匹配，小写）
_WORD_KEYWORDS = {
    "VELOCITY_INLET": ("inlet", "inflow", "entrance"),
    "PRESSURE_OUTLET": ("outlet", "outflow", "exit"),
    "SYMMETRY": ("symmetry", "sym", "symm"),
    # 周期边界的配对组与平移/旋转参数无法从名字推出，见 `face_connectivity_periodic.py`
    "PERIODIC": ("periodic",),
    "FARFIELD": ("farfield", "freestream", "far"),
    # 风洞/隧道壁按无摩擦壁处理，不挤出边界层（横跨计算域的隧道壁上挤出边界层会立即塌缩）
    "SLIP_WALL": ("slip", "tunnel"),
    "WALL": ("wall", "body", "ground", "floor"),
}

#: 中文关键字（子串匹配）
_CJK_KEYWORDS = {
    "VELOCITY_INLET": ("入口",),
    "PRESSURE_OUTLET": ("出口",),
    "SYMMETRY": ("对称",),
    "PERIODIC": ("周期",),
    "FARFIELD": ("远场",),
    "SLIP_WALL": ("风洞", "洞壁"),
    "WALL": ("车体", "车身", "壁面", "地面"),
}


def detect_boundary_type(name: str) -> str:
    """按属性名识别边界条件类型；没有命中或命中多种类型时返回 `UNSPECIFIED`。"""
    words = set(re.split(r"[^0-9a-z]+", name.lower())) - {""}
    hits = {t for t, kws in _WORD_KEYWORDS.items() if words.intersection(kws)}
    hits |= {t for t, kws in _CJK_KEYWORDS.items() if any(k in name for k in kws)}
    if len(hits) == 1:
        return hits.pop()
    if len(hits) > 1:
        logger.warning(f"边界组 '{name}' 的名字同时对应 {sorted(hits)}，类型有歧义，需用 --bc-type 显式指定")
    return UNSPECIFIED


def parse_bc_type_options(items: Optional[Iterable[str]]) -> Dict[str, str]:
    """`--bc-type 组名=类型`（可重复）-> {组名: 类型}。类型不区分大小写，必须是 `SOLVER_BC_TYPES` 之一。"""
    out: Dict[str, str] = {}
    for item in items or ():
        name, sep, bc = str(item).partition("=")
        name, bc = name.strip(), bc.strip().upper()
        if not sep or not name or not bc:
            raise BoundaryTypeError(f"--bc-type 需写成 组名=类型，收到 {item!r}")
        if bc not in SOLVER_BC_TYPES:
            raise BoundaryTypeError(f"--bc-type {item!r}：类型 {bc} 不受支持（支持 {', '.join(SOLVER_BC_TYPES)}）")
        if name in out and out[name] != bc:
            raise BoundaryTypeError(f"--bc-type 对组 '{name}' 给了两个不同的类型：{out[name]} 与 {bc}")
        out[name] = bc
    return out


def apply_bc_types(boundaries, bc_types: Optional[Dict[str, str]]) -> None:
    """用户显式给出的类型写进 `BoundaryMap.bc_types`（原地）；组名必须存在。"""
    if not bc_types:
        return
    unknown = sorted(set(bc_types) - set(boundaries.groups))
    if unknown:
        raise BoundaryTypeError(f"--bc-type 指定的组 {unknown} 不在网格的边界组 {sorted(boundaries.groups)} 中")
    for name, bc in bc_types.items():
        if bc not in SOLVER_BC_TYPES:
            raise BoundaryTypeError(f"组 '{name}' 的类型 {bc} 不受支持（支持 {', '.join(SOLVER_BC_TYPES)}）")
        boundaries.bc_types[name] = bc


def require_specified_bc_types(boundaries) -> None:
    """每个边界组都必须有确定的类型；否则报错并列出缺类型的组与指定方法。"""
    missing = sorted(name for name, bc in boundaries.bc_types.items() if bc == UNSPECIFIED)
    if missing:
        raise BoundaryTypeError(
            f"边界组 {missing} 的类型无法从名字确定（没有命中关键字或命中多种类型）。边界条件类型是物理输入，"
            f"请用 --bc-type 组名=类型 逐个给出（可重复；类型取 {', '.join(SOLVER_BC_TYPES)}），"
            f"例如 --bc-type {missing[0]}=WALL")
