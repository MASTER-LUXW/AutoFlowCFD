"""AutoFlowCFD V2.0 - 单元块预处理的内存预算与分档（平均流与湍流两个缓存共享）。

档位偏好：块 ILU（对角块 + 逆 + 面邻居耦合块）> 块 Jacobi 两份（`J_cc` 与逆）> 单份
（冻结 dtau，`block_jacobi.py` 模块文档"内存"）> 不装配（对角预处理）。多层预处理的粗层
（`coarse/`）只有单元均值量级的未知量，不进预算。

## 预算按实际可用内存定（2026-10-15）

合计预算 = 规划时本 rank 可用内存（`core/utils/system_memory.py`）− 之后还要分配的其余部分：

* Krylov 基：按 `linear_solve.py::krylov_basis_bytes` 给出（重启长度、灵活模式、单精度存储都在那里定；需要存
  预处理向量的列按所选档位定，见 `plan_block_memory`）；
* 残差求值与 Newton 步工作数组：`RESIDUAL_WORK_FACTOR` 个按统一槽位存的 float64 状态数组；
* `MEMORY_HEADROOM`：操作系统与其余进程的波动。

此前是写死的 11 GiB（按 plate_demo P3 在 31.5 GB 机器上的一次实测定的），不看求解器其余部分占了多少：
cube_demo 粗网格（28.6 万单元：9.06 万棱柱 + 19.6 万四面体）SA P2 平均流选了"两份块 Jacobi"（9.12 GiB），加上湍流块 ILU 1.07 GiB、
常驻约 8.5 GiB、Krylov 基约 7.7 GiB、残差求值瞬时约 3.8 GiB，第 2 步 GMRES 时进程 29.4 GB、内存不足；
"单份"档（4.56 GiB）放得下。在大内存机器上，写死的上限又把本可以用块 ILU 的算例压在更弱的档上。

档位只影响线性求解的迭代数，不影响收敛到的解；可用内存随机器负载变化时档位可能不同，所选档与预算
在日志里写明。

## 两个缓存共享一个预算

平均流优先：它在"给湍流留出单份"之后的预算里取最好的一档；湍流在平均流选定之后的剩余里取最好的
一档（`plan_block_modes`，两个缓存由调用方一次规划）。湍流未知量个数由模型给出（SA 1 个、SST 2 个）。
2026-10-15 以前这里按"湍流就是 2 个变量"写死，SA（1 个变量）的湍流缓存走"不在合计预算内"的分支、
独占全部预算，平均流给它预留的又是 2 个变量的单份。
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

#: 档位偏好顺序：块 ILU > 块 Jacobi 两份 > 单份（冻结 dtau）> 不装配（对角预处理）。
BLOCK_MODES = ("ilu", "two", "single")

#: 估计耦合块数用的每单元平均面邻居数（四面体 4、棱柱 5，内部单元为主）。
_MEAN_FACE_NEIGHBORS = 4.2

#: 残差求值与 Newton 步工作数组占用的内存，以"一个按统一槽位存的 float64 状态数组"（`n_cells × n_sps ×
#: n_vars × 8` 字节）为单位。标定（cube_demo 粗网格 28.6 万单元 SA P2，状态数组 0.346 GiB，分段内存探针）：
#: 峰值在 GMRES 阶段——除 Krylov 基之外的瞬时量 5.8 GiB（矩阵向量乘里的残差求值 3.77 GiB、预处理作用与
#: 真实行展开的全尺寸临时数组），比值 16.8，取 17。只按残差接受判据处的 3.77 GiB 标定（比值 10.9、取 12）
#: 低估了 2 GiB：同一算例平均流单份 + 湍流 ILU 时进程 27.6 GB，约 4 GB 被换出到磁盘，一步 GMRES 超过一小时。
RESIDUAL_WORK_FACTOR = 17

#: 为让每个块缓存都至少放得下单份，GMRES 重启长度最多缩到这里（`plan_block_memory`）。对角预处理在大 CFL 下
#: 几乎不起作用（迭代数随 CFL 增长），缩短重启长度只多几次迭代——但缩得太多同样会停滞（cube_demo 同一线性
#: 系统 GMRES(30) 198 次、GMRES(60) 46 次，`gmres.py` 模块文档），所以不低于 30、并且只缩到刚好放得下为止。
RESTART_FLOOR_FOR_BLOCKS = 30

#: 缩短重启长度的步长。
RESTART_STEP = 5

#: 预留给操作系统、其余进程与分配碎片的余量（字节）。
MEMORY_HEADROOM = 2 * 2 ** 30


def estimate_block_bytes(n_prism: int, n_tet: int, n_real_prism: int,
                         n_real_tet: int, n_var: int) -> int:
    """`J_cc` + 逆两份（float32）的总字节数。"""
    bp, bt = n_real_prism * n_var, n_real_tet * n_var
    return 2 * 4 * (n_prism * bp * bp + n_tet * bt * bt)


def estimate_ilu_bytes(n_prism: int, n_tet: int, n_real_prism: int,
                       n_real_tet: int, n_var: int) -> int:
    """块 ILU 的总字节数（对角块、对角逆、面邻居耦合块，float32）。"""
    n = n_prism + n_tet
    if n == 0:
        return 0
    mean_dof = (n_prism * n_real_prism + n_tet * n_real_tet) * n_var / n
    coupling = int(_MEAN_FACE_NEIGHBORS * n * mean_dof * mean_dof * 4)
    return estimate_block_bytes(n_prism, n_tet, n_real_prism, n_real_tet, n_var) + coupling


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


def plan_block_modes(sizes, n_mean: int, n_turb: int, budget: int) -> Tuple[Optional[str], Optional[str]]:
    """在合计预算 `budget` 内为平均流（`n_mean` 个未知量）与湍流（`n_turb` 个，0 为没有湍流缓存）选档，
    返回 `(平均流档, 湍流档)`；None 表示一档也放不下（对角预处理）。`sizes = (n_prism, n_tet, n_real_prism,
    n_real_tet)`。"""
    reserve = block_mode_bytes("single", *sizes, n_turb) if n_turb else 0
    # 平均流优先：留出湍流单份后放不下时不再预留（湍流在剩余里取档，最坏退回对角预处理）
    mean_mode = _best_mode(budget - reserve, sizes, n_mean) or _best_mode(budget, sizes, n_mean)
    if not n_turb:
        return mean_mode, None
    used = 0 if mean_mode is None else block_mode_bytes(mean_mode, *sizes, n_mean)
    return mean_mode, _best_mode(budget - used, sizes, n_turb)


@dataclass(frozen=True)
class BlockPlan:
    """一次规划的结果：两个缓存的档位，与定出它的内存账（日志用）。"""

    mean: Optional[str]
    turbulence: Optional[str]
    budget: int
    available: int
    reserve: int
    restart_cap: Optional[int] = None

    def describe(self) -> str:
        gib = 2 ** 30
        cap = "" if self.restart_cap is None else f"，GMRES 重启长度缩到 {self.restart_cap}"
        return (f"合计预算 {self.budget / gib:.2f} GiB = 可用 {self.available / gib:.2f} GiB − 预留 "
                f"{self.reserve / gib:.2f} GiB（Krylov 基、残差求值与 Newton 工作数组、余量）{cap}")


def plan_for_budget(cell_is_prism, n_real_prism: int, n_real_tet: int, n_mean: int, n_turb: int, budget: int,
                    available: Optional[int] = None, reserve: int = 0, restart_cap: Optional[int] = None) -> BlockPlan:
    """在给定合计预算下为两个缓存定档（`plan_block_memory` 先由可用内存算出预算再调它；测试直接给预算）。"""
    cell_is_prism = np.asarray(cell_is_prism, dtype=bool)
    n_prism = int(cell_is_prism.sum())
    sizes = (n_prism, int(cell_is_prism.size - n_prism), n_real_prism, n_real_tet)
    mean, turb = plan_block_modes(sizes, n_mean, n_turb, int(budget))
    return BlockPlan(mean, turb, int(budget), int(budget if available is None else available), int(reserve),
                     restart_cap)


def plan_block_memory(red, cell_is_prism, n_sps: int, n_real_prism: int, n_real_tet: int,
                      n_mean: int, n_turb: int) -> BlockPlan:
    """按本 rank 当前可用内存为两个块缓存定档（模块文档）。分布式下各 rank 的预算取全局最小，档位全局一致。

    Args:
        red: 全局归约（`red.xp` 决定查主机内存还是设备内存）。
        cell_is_prism: `(n_cells,)` 布尔掩码（本 rank 的单元）。
        n_sps: 每单元槽位数（原生基的统一槽位，含零填充）。
        n_real_prism, n_real_tet: 两类单元的真实解点数。
        n_mean, n_turb: 平均流与湍流未知量个数（没有湍流缓存时 `n_turb = 0`）。
    """
    from autoflowcfd.core.utils.system_memory import available_bytes

    from .forcing import _ETA_MIN
    from .gmres import basis_itemsize
    from .linear_solve import krylov_basis_bytes, krylov_restart

    cell_is_prism = np.asarray(cell_is_prism, dtype=bool)
    n_prism = int(cell_is_prism.sum())
    n_tet = int(cell_is_prism.size - n_prism)
    n_vars = n_mean + n_turb
    real_entries = (n_prism * n_real_prism + n_tet * n_real_tet) * n_vars
    fixed = RESIDUAL_WORK_FACTOR * cell_is_prism.size * n_sps * n_vars * 8 + MEMORY_HEADROOM
    available = available_bytes(red.xp)

    def plan(flexible_fraction: float, restart_cap: Optional[int]) -> BlockPlan:
        reserve = krylov_basis_bytes(real_entries, red, flexible=flexible_fraction, restart=restart_cap) + fixed
        budget = int(red.min(red.xp.asarray([float(max(available - reserve, 0))])))
        return plan_for_budget(cell_is_prism, n_real_prism, n_real_tet, n_mean, n_turb, budget, available, reserve,
                               restart_cap)

    def plan_at(restart_cap: Optional[int]) -> BlockPlan:
        # 只有块 ILU 档的那一块会用多层（非线性）预处理、要存预处理向量（`gmres.py` 模块文档"部分灵活"）。先按
        # "平均流线性、湍流非线性"规划；平均流选到块 ILU 时它也变成非线性，按整份灵活重新规划（结果再降档也
        # 安全：预留只多不少）。
        first = plan(n_turb / n_vars, restart_cap)
        return first if first.mean != "ilu" else plan(1.0, restart_cap)

    def complete(p: BlockPlan) -> bool:
        return p.mean is not None and (n_turb == 0 or p.turbulence is not None)

    full = plan_at(None)
    if complete(full):
        return full
    # 有缓存连单份都放不下：缩短 GMRES 重启长度换内存，取第一个让每个缓存都有块预处理的长度
    default_restart = krylov_restart(real_entries, red, flexible=1.0, itemsize=basis_itemsize(_ETA_MIN))
    for m in range(default_restart - RESTART_STEP, RESTART_FLOOR_FOR_BLOCKS - 1, -RESTART_STEP):
        candidate = plan_at(m)
        if complete(candidate):
            return candidate
    return full
