"""AutoFlowCFD V2.0 - 单元块预处理的内存预算与分档（平均流与湍流两个缓存共享）。

档位偏好：块 ILU（对角块 + 逆 + 面邻居耦合块）> 块 Jacobi 两份（`J_cc` 与逆）> 单份
（冻结 dtau，`block_jacobi.py` 模块文档"内存"）> 不装配（对角预处理）。多层预处理的粗层
（`coarse/`）只有单元均值量级的未知量，不进预算。
"""

from typing import Optional

#: 全部单元块预处理（平均流 + 湍流两个缓存**合计**，float32）允许的字节数。
#:
#: **为什么是一个合计预算**（2026-09-29）：此前平均流与湍流的缓存各自按自己的上限
#: （ILU 12 GiB、两份 8 GiB）选档，互相看不见。plate_demo P3（17.9 万单元）湍流只有
#: 2 个变量，耦合块却随自由度平方增长，块 ILU 要 8.38 GiB；加上平均流单份 8.9 GiB、
#: 求解器常驻 7.7 GiB、残差求值峰值约 4.5 GiB、Krylov 基 4.6 GiB，合计约 34 GiB，
#: GMRES 的矩阵向量乘在 31.5 GB 开发机上 OOM。11 GiB：放得下 P3 平均流单份（8.9 GiB）+
#: 湍流单份（1.4 GiB），与上述其余部分合计约 27 GiB。分档见 `plan_block_mode`（平均流优先）。
PRECOND_TOTAL_BYTES = 11 * 2 ** 30

#: 档位偏好顺序：块 ILU > 块 Jacobi 两份 > 单份（冻结 dtau）> 不装配（对角预处理）。
BLOCK_MODES = ("ilu", "two", "single")

_MEAN_FLOW_VARS = 5
_TURBULENCE_VARS = 2

#: 估计耦合块数用的每单元平均面邻居数（四面体 4、棱柱 5，内部单元为主）。
_MEAN_FACE_NEIGHBORS = 4.2

def estimate_block_bytes(n_prism: int, n_tet: int, n_real_prism: int,
                         n_real_tet: int, n_var: int) -> int:
    """`J_cc` + 逆两份（float32）的总字节数。"""
    bp, bt = n_real_prism * n_var, n_real_tet * n_var
    return 2 * 4 * (n_prism * bp * bp + n_tet * bt * bt)


def block_mode_bytes(mode: str, n_prism: int, n_tet: int, n_real_prism: int, n_real_tet: int,
                     n_var: int) -> int:
    """某一档（`BLOCK_MODES`）的字节数。"""
    if mode == "ilu":
        return estimate_ilu_bytes(n_prism, n_tet, n_real_prism, n_real_tet, n_var)
    two = estimate_block_bytes(n_prism, n_tet, n_real_prism, n_real_tet, n_var)
    return two if mode == "two" else two // 2


def _best_mode(budget: int, sizes, n_var: int) -> Optional[str]:
    for mode in BLOCK_MODES:
        if block_mode_bytes(mode, *sizes, n_var) <= budget:
            return mode
    return None


def plan_block_mode(n_var: int, n_prism: int, n_tet: int, n_real_prism: int, n_real_tet: int,
                    with_turbulence: bool) -> Optional[str]:
    """在合计预算 `PRECOND_TOTAL_BYTES` 内为平均流（`n_var=5`）或湍流（`n_var=2`）选档；
    None 表示一档也放不下（对角预处理）。

    平均流优先：它在"给湍流留出单份"之后的预算里取最好的一档；湍流在平均流选定之后
    的剩余里取最好的一档。两个缓存各自调用、结果一致（同一个函数、同一组尺寸）。
    `with_turbulence=False`（层流）时平均流独占预算。
    """
    sizes = (n_prism, n_tet, n_real_prism, n_real_tet)
    if n_var not in (_MEAN_FLOW_VARS, _TURBULENCE_VARS):
        return _best_mode(PRECOND_TOTAL_BYTES, sizes, n_var)
    turbulence = with_turbulence or n_var == _TURBULENCE_VARS
    reserve = block_mode_bytes("single", *sizes, _TURBULENCE_VARS) if turbulence else 0
    # 平均流优先：留出湍流单份后放不下时不再预留（湍流在剩余里取档，最坏退回对角预处理）
    mean_mode = (_best_mode(PRECOND_TOTAL_BYTES - reserve, sizes, _MEAN_FLOW_VARS)
                 or _best_mode(PRECOND_TOTAL_BYTES, sizes, _MEAN_FLOW_VARS))
    if n_var == _MEAN_FLOW_VARS:
        return mean_mode
    used = 0 if mean_mode is None else block_mode_bytes(mean_mode, *sizes, _MEAN_FLOW_VARS)
    return _best_mode(PRECOND_TOTAL_BYTES - used, sizes, _TURBULENCE_VARS)


def estimate_ilu_bytes(n_prism: int, n_tet: int, n_real_prism: int,
                       n_real_tet: int, n_var: int) -> int:
    """块 ILU 的总字节数（对角块、对角逆、面邻居耦合块，float32）。"""
    n = n_prism + n_tet
    if n == 0:
        return 0
    mean_dof = (n_prism * n_real_prism + n_tet * n_real_tet) * n_var / n
    coupling = int(_MEAN_FACE_NEIGHBORS * n * mean_dof * mean_dof * 4)
    return estimate_block_bytes(n_prism, n_tet, n_real_prism, n_real_tet, n_var) + coupling
