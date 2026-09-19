"""AutoFlowCFD V2.0 - cube face 编码的 numba 查找表。

从原 `kernel_helpers.py` 拆出（2026-09-19）。这些表**必须覆盖全部 15 个
编码**，理由与派生来源见下方各条注释。纯搬家，未改任何逻辑。
"""

import numpy as np

from .data import _PRISM_QUAD_CODES
from .geometry import CUBE_FACE_AXIS_SIDE
from autoflowcfd.grid.connectivity.face_connectivity import CUBE_FACE_CODES


# ============================================================================
# 查找表
# ============================================================================

# cube face code -> (axis, side) 查表。**必须覆盖全部 15 个编码**：numba
# nopython 不做边界检查，对 code>=len 索引会读到未定义内存（原生四面体
# 编码当年就踩过这条，见 `face_flux_points/exact_normal_kernel.py` 里那段
# "真实 bug 修复背景"）。
#
#   [0, 6)   坍缩立方体面 a=-1/a=+1/b=-1/b=+1/c=-1/c=+1
#   [6, 10)  原生四面体面 —— axis 槽位复用成 excluded_vertex、side 是哑值
#            （那条分支不走 (axis,side) 语义，见主 kernel 里的分派）
#   [10, 15) 原生棱柱面 —— 填的是**真实**的 (axis, side)：原生棱柱面的
#            通量点与坍缩立方体面的通量点已验证是同一批物理点、同一顺序，
#            所以定位仍走坍缩的 `_newton_locate_nb`，它需要真实的 axis/side。
#
# **由 `face_flux_points/geometry.py::CUBE_FACE_AXIS_SIDE` 逐项派生**（2026-09-19）。
# 此前这里是一份手抄的字面量数组，靠注释+一条测试"钉住两者不漂移" ——
# 而原生棱柱接入时正是从这类重复里漏出真实缺陷（`_PQ_CODES` 漏了原生那
# 三个侧面编码，`_PRISM_QUAD_CODES` 同样漏了，导致原生档第一次端到端
# 构造直接 `KeyError: (0, 14)`）。派生之后"漂移"在结构上不可能发生。
#
# 导入方向安全：`face_flux_points/geometry.py` 与 `face_flux_points/data.py` 都不
# import 本模块（只有 `face_flux_points/kernel.py`/`_ms_numba.py` 会），
# 所以这里反向 import 不构成环。
_FACE_AXIS = np.zeros(len(CUBE_FACE_CODES), dtype=np.int32)
_FACE_SIDE = np.zeros(len(CUBE_FACE_CODES), dtype=np.float64)
for _name, _code in CUBE_FACE_CODES.items():
    _ax, _sd = CUBE_FACE_AXIS_SIDE[_name]
    _FACE_AXIS[_code] = _ax
    _FACE_SIDE[_code] = _sd
del _name, _code, _ax, _sd

#: **多源面**（棱柱的三个四边形侧面）的 cube face code —— 这类面在对侧
#: 被三角化成两个面，插值矩阵要走 multi-source kernel。坍缩编码是
#: a=-1/a=+1/b=-1 = 0/1/2，原生棱柱的对应面是 f3/f2/f4 = 13/12/14。
#: **漏掉原生那三个会让棱柱侧面走单源路径、静默拿到错的邻居插值**。
#:
#: 从 `face_flux_points/data.py::_PRISM_QUAD_CODES` 派生（那个集合是唯一
#: 定义；这里只是把它变成 numba 能索引的有序数组）。
_PQ_CODES = np.array(sorted(_PRISM_QUAD_CODES), dtype=np.int32)

#: 原生面编码区间（与 `grid/connectivity/face_connectivity.py` 的
#: `NATIVE_TET_FACE_CODE_RANGE`/`NATIVE_PRISM_FACE_CODE_RANGE` 同源）。
#: numba 分支判据统一用这两个常量，不要再写 `code >= 6` 这种字面量 ——
#: 加了棱柱编码之后那个字面量的含义从"是原生四面体面"变成了"是任意
#: 原生面"，而两者要分派到**不同**的算子组。
_NATIVE_TET_LO = 6
_NATIVE_TET_HI = 10
_NATIVE_PRISM_LO = 10
_NATIVE_PRISM_HI = 15
