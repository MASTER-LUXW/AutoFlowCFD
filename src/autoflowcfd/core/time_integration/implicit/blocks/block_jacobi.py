"""AutoFlowCFD V2.0 - PTC-Newton-Krylov 的单元块 Jacobi 预处理（着色有限差分装配）。

## 为什么需要（真实测量）

`preconditioner.py` 的逐 SP 对角预处理 `P = I/dtau` 只在 `dtau` 小的时候
好用：`dtau` 大时 `I/dtau + J` 由 `J` 主导，而 FR/DG 的 `J` 在单元内部是
稠密的（高阶模态之间的刚性耦合），对角预处理完全看不到它。plate_demo
P1 层流（17.9 万单元，预处理 NK，CFL 34，同一状态、同一 `rtol=0.1`）：

    对角预处理     GMRES 300 次仍未达到容差（真实相对残差 0.126），925 s
    单元块 Jacobi   GMRES  20 次达到（0.067），65 s（含每次迭代的块作用）

同一次运行里 SER 律把 CFL 推到 30~50 时，对角预处理下每个 Newton 步
要 60~150 次 GMRES（每步 3~7 分钟），成了整个隐式路径的瓶颈。

## 装配

着色有限差分装配（`cell_blocks.py::CellBlockJacobian`，成本与单元数无关、
P0 顺带给出面邻居耦合块）或解析装配（下一节）。

## 解析装配（P>=1 的默认）

差分装配要 `色数 x 真实解点数 x 5` 次整场残差求值，P2 起不可承受（plate_demo
P2 450 次、单次 5.5 s）。P>=1 由各后端传入解析装配器
（`fr_residual/jacobian`，逐点导数与参考算子的收缩，同一个矩阵、代价约十几次
残差求值），它同时给出面邻居耦合块，合计预算放得下（`plan_block_modes`）时预处理
换成块 ILU（`block_ilu.py`，大 CFL 下 GMRES 迭代数约为块 Jacobi 的 0.4 倍）。
P0 与解析装配不覆盖的离散（见 `fr_residual/jacobian/backend.py::unsupported_reason`）
仍用差分装配。

## 复用与刷新

`J_cc` 只依赖状态，不依赖 `dtau`：每个 Newton 步只按新的 `dtau` 重新
求逆（逐块 `J_cc + diag(1/dtau)`，廉价），`J_cc` 本身跨步复用，直到

* 上一步的 GMRES 迭代数超过"刚装配完那一步"的基线加上**一次装配折合的迭代数**
  （`refresh_cost.py`：解析装配按块类型与阶数查实测标定表，差分装配按精确的残差求值
  次数折算，下限 `REFRESH_SLACK_MIN`）：多出来的迭代比重装配一次还贵时才重装配。
  更早是固定的 `2 倍 + 10`（差分）/ `1.5 倍 + 3`（解析）：plate_demo P1 基线 3 次时阈值
  只有 7 次，GMRES 8 次就触发一次 9~11 s 的装配，而多一次迭代只要约 1 s；反过来 P0 差分
  装配对廉价的迭代又刷新得太勤。2026-09-26 起改用实测墙钟耗时之比，2026-10-04 因跨进程
  轨迹不可复现改为确定性查表（各 rank 自然一致，不需要归约）；或
* 上一步没有被接受（`theta == 0`）；或
* **本步**的 GMRES 在上一条的迭代预算内没有达到容差（`stale_budget` /
  `recover`，由 `jfnk.py` 当场调用；同时启用预处理的对角位移，`precond_shift.py`）。此前只按上一步的迭代数决定下一步是否
  重装配：plate_demo P1+SST 上湍流块在一步里从 10 次跳到 200 次（用满、只到
  0.29），平均流同一步也用满 200 次，那一步耗时 1173 s——过时的预处理把整步
  的迭代预算烧完，重装配要等下一步才发生。

冻结 Jacobian 的预处理是隐式 CFD 的标准做法；过时的预处理子仍然是合法
的预处理子，只是迭代数上升，由上面的刷新判据兜住。此前另有"复用满 20 步无条件重装配"的硬上限
（2026-10-05 删除）：湍流平板 SA P3（3072 单元）上它在每步只要 1~3 次 GMRES 的阶段强制重装配
（每次约 15 s，折合约 30 次迭代），去掉后到收敛的耗时 1216 s -> 1133 s、GMRES 1290 -> 1206 次
（`ProjectFiles/V2.0/38_*.md` 第 25.13 节）。

## 内存

逐单元只存**真实**自由度（原生基的零填充槽位不进块，见
`fr/native_padding.py::real_sps_per_cell`）。`J_cc` 与逆各一份，float32
存储（逐块求逆在 float64 下做）：预处理子的精度只影响迭代数，不影响
GMRES 解的精度。

合计预算（`block_budget.py`：按可用内存定，平均流与湍流共享）放不下两份、放得下一份时用**单份**（冻结 dtau）：
装配后立即按当时的 `dtau` 原地求逆、不再保留 `J_cc`；之后每步直接用这份逆，
当前 `dtau` 与求逆时的 `dtau`（几何平均比值）相差超过 `DTAU_REBUILD_RATIO` 倍才
重装配。预处理子里的 `dtau` 略微过时只影响迭代数（它仍是合法的预处理子），迭代
数上升由上面的刷新判据兜住。plate_demo P3（1.99 万棱柱块 200x200 + 15.9 万四面体块
100x100）两份 19.1 GiB、一份 9.5 GiB：与求解器本身（残差峰值约 11.8 GiB）和紧凑
Krylov 基（约 4.8 GiB）共存于 31.5 GB 开发机只能是一份。

一份也放不下时不装配、退回对角预处理并打 WARNING（例如 P3 棱柱一块 200x200，
79 万单元就要上百 GB）。

## 单元级粗校正（`two+cc` / `single+cc` 档）

P>=1 平均流放不下块 ILU 时，块 Jacobi 之外叠加单元级粗校正（`coarse/cell_coarse.py`）：每次装配同时请求
单元对 5x5 粗块（解析装配器在界面核里流式约化，`fr_residual/jacobian/coarse.py`），粗矩阵的稀疏结构与聚合
层次只依赖网格、跨装配复用；粗层近似逆按伪时间项与块 ILU 档的分解同一判据复用（`DTAU_REBUILD_RATIO` /
`PTC_NEGLIGIBLE`）。
"""

from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from loguru import logger

from .block_budget import COARSE_SUFFIX, BlockPlan, block_mode_bytes
from .cell_block_inverse import CellBlockJacobiPreconditioner, invert_blocks_in_place
from .cell_blocks import CellBlockJacobian
from .precond_shift import block_diagonal_scale, shifted_dtau
from .preconditioner import PseudoTransientDiagonal
from ..reductions import LocalReductions
from .refresh_cost import assembly_cost_iters, block_kind

if TYPE_CHECKING:
    from .block_ilu import BlockCouplingStructure


#: 单份模式：当前 dtau 与求逆时 dtau 的几何平均比值超过这个倍数（任一方向）就重装配；
#: 块 ILU 档：超过它就重新分解（`J_cc` 不变时分解跨步复用，见 `preconditioner`）。
DTAU_REBUILD_RATIO = 2.0

#: 块 ILU 档：伪时间项 `1/dtau` 相对 `J_cc` 对角的几何平均比值，分解时与当前都低于它时，
#: 分解与 dtau 无关、不论 dtau 怎么变都复用（收敛末段 SER 让 CFL 每步翻倍，此时 `I/dtau`
#: 已可忽略，按比值判据会每步白白重分解）。
PTC_NEGLIGIBLE = 1e-2


def _order_of(n_real_prism: int, n_real_tet: int) -> int:
    """由每单元真实自由度反解阶数（刷新判据查标定表用）。"""
    from autoflowcfd.fr.native_padding import real_sps_per_cell

    for p in range(8):
        if real_sps_per_cell(p) == (n_real_prism, n_real_tet):
            return p
    raise ValueError(f"每单元真实自由度 ({n_real_prism}, {n_real_tet}) 不对应任何阶数")


class BlockJacobiCache:
    """跨 Newton 步持有 `J_cc` 并决定何时重装配（刷新判据见模块文档）。

    换阶（Order Continuation）后 `n_sps` 与块尺寸都变了，调用方必须丢弃
    整个缓存对象重建（`order_continuation/interp.py` 与 forcing 状态一起
    失效）。
    """

    __slots__ = ("cell_is_prism", "colors", "n_sps", "n_real_prism", "n_real_tet",
                 "jac", "age", "baseline_iters", "last_iters", "last_accepted",
                 "disabled_reason", "n_builds", "red", "n_var", "assembler", "use_ilu", "coupling",
                 "_coupling_graph_fn", "_coupling_graph", "kind", "refresh_slack",
                 "single_copy", "_inverted_dtau", "_coarse", "cross",
                 "_factored", "_factored_dtau", "_ptc_diag", "shifted", "plan",
                 "cell_coarse", "_coarse_weights", "_coarse_weights_host", "_real_slots", "_coarse_topology",
                 "_coarse_matrix", "_coarse_solve", "_coarse_dtau")

    def __init__(self, *, cell_is_prism, colors: np.ndarray, n_sps: int,
                 n_real_prism: int, n_real_tet: int, n_var: int, plan: BlockPlan,
                 red: Optional[LocalReductions] = None, coupling_graph=None, with_turbulence: bool = False,
                 global_coarse=None, coarse_weights=None):
        """`colors`：块装配用的单元着色（单机 `coloring.greedy_cell_coloring`；分布式
        是全局一致着色里本 rank 那一段，保证同色单元跨 rank 也不相邻）。

        `assembler`（每个 Newton 步由调用方设置，见 `begin_step`）：解析装配器
        `(u0_flat, r0_flat) -> (blocks_prism, blocks_tet)`；为 None 时用着色差分装配。

        `coupling_graph`：可选回调 `() -> coloring.CouplingGraph`，只在首次需要时调用。
        每单元一个真实解点（P0）且没有解析装配器时，差分装配按它的距离 2 着色同时给出
        面邻居耦合块，预处理用块 ILU（见 `cell_blocks.py` 模块文档）。

        `plan`：调用方按可用内存一次规划好的两个缓存的档位（`block_budget.py::plan_block_memory`）；本缓存
        取其中对应自己的那一档（`with_turbulence` 的平均流缓存与湍流缓存共享这一个预算）。

        `with_turbulence`：平均流缓存是否与隐式湍流缓存耦合（刷新判据的代价标定按它分类，`refresh_cost.py`）。

        `global_coarse`（`coarse.CoarseCommContext`，分布式后端给出）：块 ILU 档在本 rank 的
        预处理之外再叠全局粗校正（`coarse/global_coarse.py`），装配器须同时给出跨 rank 耦合块。

        `coarse_weights`：`(n_cells, n_sps)` 单元级粗校正的限制权重（`coarse/cell_coarse.py::restriction_weights`，
        与状态同一数组模块）；档位带 `+cc` 时必须给出。
        """
        self.red = red if red is not None else LocalReductions()
        self.n_var = int(n_var)
        self.assembler = None
        self.coupling: Optional["BlockCouplingStructure"] = None   # 块 ILU 档的面邻居耦合结构
        self._coupling_graph_fn = coupling_graph
        self._coupling_graph = None
        self.kind = block_kind(self.n_var, with_turbulence)
        self.refresh_slack: Optional[int] = None   # 一次装配折合的 GMRES 迭代数，每次装配后按 refresh_cost.py 给出
        self.single_copy = False
        self._inverted_dtau = None
        self._factored = None            # 块 ILU 档：当前 J_cc 下按 `_factored_dtau` 构造的预处理子
        self._factored_dtau = None
        self._ptc_diag = None            # 逐（单元, 解点）的 max_v |J_cc 对角|（PTC 项可忽略判据、对角位移用）
        self.shifted = False             # 预处理的伪时间项是否带下限（`precond_shift.py`，`recover` 启用）
        self.cross: Any = None           # 跨 rank 耦合（`coarse.CrossRankCoupling`，分布式块 ILU 档）
        self.cell_is_prism = np.asarray(cell_is_prism, dtype=bool)
        self.n_sps, self.n_real_prism, self.n_real_tet = n_sps, n_real_prism, n_real_tet
        self.jac: Optional[CellBlockJacobian] = None
        self.age = 0
        self.baseline_iters: Optional[int] = None
        self.last_iters: Optional[int] = None
        self.last_accepted = True
        self.n_builds = 0
        sizes = (int(self.cell_is_prism.sum()), int((~self.cell_is_prism).sum()), n_real_prism, n_real_tet)
        self.plan = plan
        mode = plan.turbulence if self.kind.startswith("turbulence") else plan.mean
        # 块 ILU 要面邻居耦合块（解析装配或 P0 差分装配给出）
        self.use_ilu = mode == "ilu"
        self.single_copy = mode is not None and mode.startswith("single")
        self.cell_coarse = mode is not None and mode.endswith(COARSE_SUFFIX)
        self._coarse_weights = coarse_weights if self.cell_coarse else None
        self._coarse_weights_host = None
        self._real_slots = None
        self._coarse_topology: Any = None       # 单元级粗校正：`CellCoarseTopology`（跨装配复用）
        self._coarse_matrix: Any = None         # 本次装配的 `CellCoarseMatrix`
        self._coarse_solve: Any = None          # 按 `_coarse_dtau` 构造的 `CellCoarseSolve`
        self._coarse_dtau = None
        if self.cell_coarse:
            if coarse_weights is None:
                raise ValueError("单元级粗校正档需要限制权重（coarse_weights）")
            from autoflowcfd.core.utils.array_module import to_host

            self._coarse_weights_host = np.ascontiguousarray(to_host(coarse_weights), dtype=np.float64)
            n_real = np.where(self.cell_is_prism, n_real_prism, n_real_tet)
            self._real_slots = self.red.xp.asarray(np.arange(n_sps)[None, :] < n_real[:, None])
        self.disabled_reason = None
        from ..coarse import CoarsePreconditionerFactory
        self._coarse = CoarsePreconditionerFactory(max(n_real_prism, n_real_tet) > 1, global_coarse)
        self.colors = np.asarray(colors, dtype=np.int64)
        if mode is None:
            self.disabled_reason = (
                f"单元块 Jacobi（n_var={self.n_var}）一份也需要 "
                f"{block_mode_bytes('single', *sizes, self.n_var) / 2 ** 30:.1f} GiB，超出{plan.describe()}，"
                f"改用逐 SP 对角预处理——大 CFL 下 GMRES 迭代数会显著上升")
            logger.warning("[NK] " + self.disabled_reason)
        else:
            logger.info(f"[NK] 单元块预处理（n_var={self.n_var}）：{mode}，"
                        f"{block_mode_bytes(mode, *sizes, self.n_var) / 2 ** 30:.2f} GiB"
                        f"（{plan.describe()}）")

    def _refresh_threshold(self, baseline: int) -> int:
        if self.refresh_slack is None:
            raise RuntimeError("refresh_slack 在第一次装配之后才有值")
        return int(baseline + self.refresh_slack)

    def _needs_rebuild(self, dtau_flat) -> bool:
        if self.jac is None or not self.last_accepted:
            return True
        if self.single_copy and self._dtau_drifted(dtau_flat):
            return True
        if self.baseline_iters is not None and self.last_iters is not None:
            return self.last_iters > self._refresh_threshold(self.baseline_iters)
        return False

    def _dtau_drifted(self, dtau_flat, ref=None) -> bool:
        """当前 dtau 相对 `ref`（默认：单份模式求逆时的 dtau）的几何平均比值是否超过
        `DTAU_REBUILD_RATIO`（几何平均不让少数被局部缩小 dtau 的行触发整场重来）。分布式下取全局平均。"""
        xp = self.red.xp
        ref = self._inverted_dtau if ref is None else ref
        log_ratio = xp.log(xp.asarray(dtau_flat, dtype=xp.float64).ravel() / ref)
        mean = self.red.sum(log_ratio) / max(self.red.count(log_ratio), 1.0)
        return abs(mean) > np.log(DTAU_REBUILD_RATIO)

    def _ptc_relative(self, dtau_flat) -> float:
        """`1/dtau` 相对 `J_cc` 对角的几何平均比值（只取真实行；分布式下取全局平均）。"""
        xp = self.red.xp
        diag = self._diag_scale()
        real = diag > 0.0
        log_rel = -xp.log(xp.asarray(dtau_flat, dtype=xp.float64).ravel()[real] * diag[real])
        return float(np.exp(self.red.sum(log_rel) / max(self.red.count(log_rel), 1.0)))

    def _diag_scale(self):
        """逐（单元, 解点）的 `J_cc` 对角量级（每次装配后首次用到时计算；单份模式在原地求逆之前算）。"""
        if self._ptc_diag is None:
            self._ptc_diag = block_diagonal_scale(self.jac, self.cell_is_prism, self.n_sps, self.n_real_prism,
                                                  self.n_real_tet, self.red.xp)
        return self._ptc_diag

    def _preconditioner_dtau(self, dtau_flat):
        """预处理里用的伪时间步：启用对角位移后带下限（`precond_shift.py`），否则就是系统的 dtau。"""
        if not self.shifted or self.jac is None:
            return dtau_flat
        return shifted_dtau(dtau_flat, self._diag_scale(), self.red.xp)

    @property
    def recoverable(self) -> bool:
        """线性求解失败后 `recover` 是否还有动作可做。"""
        return self.disabled_reason is None and self.jac is not None and (self.age > 0 or not self.shifted)

    def recover(self, residual, u0_flat, r0_flat, scales, dtau_flat) -> bool:
        """本步线性求解没到容差时调用：复用中的块按本步基态重装配，并启用预处理的对角位移
        （`precond_shift.py`）。有动作返回 True（调用方重新求解）；刚装配的块且位移已启用时无可补救。"""
        if not self.recoverable:
            return False
        if not self.shifted:
            self.shifted = True
            logger.info(f"[NK] 单元块预处理（n_var={self.n_var}）：线性求解未达容差，启用对角位移")
        if self.age > 0 or self.single_copy:
            self._build(residual, u0_flat, r0_flat, scales, dtau_flat, reason="本步线性求解未达容差")
        return True

    def _factorization_stale(self, dtau_flat) -> bool:
        """块 ILU 档：按 `_factored_dtau` 做的分解对当前 dtau 是否已过时（判据见 `preconditioner`）。"""
        if self._factored is None:
            return True
        return self._stale_against(dtau_flat, self._factored_dtau)

    def _stale_against(self, dtau_flat, ref) -> bool:
        """依赖 dtau 的构造（块 ILU 分解、单元级粗层近似逆）按 `ref` 做的是否对当前 dtau 已过时：几何平均比值
        超过 `DTAU_REBUILD_RATIO`，除非伪时间项在两个时刻都已可忽略（`PTC_NEGLIGIBLE`）。"""
        if not self._dtau_drifted(dtau_flat, ref):
            return False
        return not (self._ptc_relative(dtau_flat) < PTC_NEGLIGIBLE
                    and self._ptc_relative(ref) < PTC_NEGLIGIBLE)

    def stale_budget(self) -> Optional[int]:
        """复用中的 `J_cc`（`age > 0`）在本步的 GMRES 迭代预算：超过它就说明
        线性化已明显过时（与跨步刷新判据同一个阈值）。刚装配的块、或没有
        基线时返回 None（用满全部预算）。"""
        if self.disabled_reason is not None or self.jac is None or self.age == 0:
            return None
        if self.baseline_iters is None:
            return None
        return self._refresh_threshold(self.baseline_iters)

    def begin_step(self, residual, u0_flat, r0_flat, scales, dtau_flat) -> None:
        """每个 Newton 步开始时调用一次：按刷新判据决定是否重装配 `J_cc`。

        必须与 `preconditioner()` 分开：一个 Newton 步内 `dtau` 逐档缩小
        重试时每档都要重新求逆，但 `J_cc` 只依赖基态，不能每档重装配。

        `dtau_flat`：本步的伪时间步长（缩档之前）。单份模式据它判断存下的逆是否已经
        过时（`_dtau_drifted`），重装配时也按它求逆。
        """
        if self.disabled_reason is not None or not self._needs_rebuild(self._preconditioner_dtau(dtau_flat)):
            return
        self._build(residual, u0_flat, r0_flat, scales, dtau_flat, reason="刷新判据")

    def _build(self, residual, u0_flat, r0_flat, scales, dtau_flat, *, reason: str) -> None:
        import time

        t0 = time.time()
        # 先释放旧块再装配：新旧两份同时存在时 P3 单份模式要 2 x 8.9 GiB（plate_demo 实测
        # 步内重装配时 OOM）；调用方同样先丢掉持有旧块的预处理对象（jfnk.py）
        self.jac = None
        self.coupling = None
        self.cross = None
        self._factored = None
        self._factored_dtau = None
        self._ptc_diag = None
        self._coarse_matrix = None
        self._coarse_solve = None
        self._coarse_dtau = None
        if self.cell_coarse and self.assembler is None:
            raise RuntimeError("单元级粗校正档要解析装配器给出粗块（P>=1 才有这一档）")
        if self.assembler is not None:
            self.assembler.want_coupling = self.use_ilu
            if self.cell_coarse:
                self.assembler.coarse_weights = self._coarse_weights_host
            out = self.assembler(u0_flat, r0_flat)
            blocks_prism, blocks_tet = out[0], out[1]
            if self.cell_coarse:
                self._set_coarse_matrix(out[2], out[3] if len(out) > 3 else None)
            if self.use_ilu:
                from .block_ilu import BlockCouplingStructure
                n_real = np.where(self.cell_is_prism, self.n_real_prism, self.n_real_tet)
                self.coupling = BlockCouplingStructure(out[2], self.cell_is_prism.size, n_real, self.n_var)
                self.cross = self._coarse.cross_coupling(out[3] if len(out) > 3 else None, self.coupling)
            self.jac = CellBlockJacobian.from_blocks(
                blocks_prism, blocks_tet, n_sps=self.n_sps, n_var=self.n_var,
                cell_is_prism=self.cell_is_prism, n_real_prism=self.n_real_prism,
                n_real_tet=self.n_real_tet, xp=self.red.xp)
        else:
            graph = self._fd_coupling_graph()
            self.jac = CellBlockJacobian(
                residual, u0_flat, r0_flat, scales, n_sps=self.n_sps,
                cell_is_prism=self.cell_is_prism, n_real_prism=self.n_real_prism,
                n_real_tet=self.n_real_tet, colors=self.colors, red=self.red, coupling_graph=graph)
            if graph is not None:
                from .block_ilu import BlockCouplingStructure
                self.coupling = BlockCouplingStructure(
                    self.jac.coupling, self.cell_is_prism.size, np.ones(self.cell_is_prism.size, np.int64),
                    self.n_var)
                self.cross = self._coarse.cross_coupling(self.jac.cross_coupling, self.coupling)
        if self.single_copy:
            self._diag_scale()                                  # 块原地求逆之后对角量级就取不到了
            dtau_flat = self._preconditioner_dtau(dtau_flat)
            invert_blocks_in_place(self.jac, dtau_flat)
            self._inverted_dtau = self.red.xp.asarray(dtau_flat, dtype=self.red.xp.float64).ravel().copy()
        self.age = 0
        self.baseline_iters = None
        self.last_iters = None
        self.last_accepted = True
        self.n_builds += 1
        analytic = self.assembler is not None
        self.refresh_slack = assembly_cost_iters(
            self.kind, _order_of(self.n_real_prism, self.n_real_tet) if analytic else 0, analytic,
            self.jac.n_residual_evals)
        logger.info(f"[NK] 单元块 Jacobian 重装配（第 {self.n_builds} 次，{reason}，"
                    f"{'块 ILU' if self.coupling is not None else '块 Jacobi'}"
                    f"{'（单份，冻结 dtau）' if self.single_copy else ''}，"
                    + ("解析装配" if self.assembler is not None
                       else f"差分装配 {self.jac.n_residual_evals} 次残差求值")
                    + f"，{time.time() - t0:.1f}s）")

    def _set_coarse_matrix(self, coarse, cross=None) -> None:
        """装配给出的单元级粗块（分布式另有跨 rank 粗块）-> 粗矩阵（稀疏结构与聚合层次首次构造后复用，单元对
        必须与首次一致）。"""
        from ..coarse.cell_coarse import CellCoarseMatrix, CellCoarseTopology

        xr, xc = (None, None) if cross is None else (cross.rows, cross.cols)
        if self._coarse_topology is None:
            self._coarse_topology = CellCoarseTopology(self.cell_is_prism.size, coarse.rows, coarse.cols, self.n_var,
                                                       xr, xc)
        elif not self._coarse_topology.matches(coarse.rows, coarse.cols, xr, xc):
            raise RuntimeError("单元级粗块的单元对与首次装配不一致（网格拓扑不应在缓存生命周期内变化）")
        self._coarse_matrix = CellCoarseMatrix(self._coarse_topology, coarse.diag, coarse.blocks,
                                               None if cross is None else cross.blocks)

    def _coarse_corrected(self, base, dtau_flat):
        """块 Jacobi 叠加单元级粗校正；粗层近似逆按 `_factorization_stale` 的同一判据复用。"""
        from ..coarse.cell_coarse import CellCoarseSolve, CoarseCorrectedPreconditioner

        if self._coarse_solve is None or self._stale_against(dtau_flat, self._coarse_dtau):
            self._coarse_solve = None    # 先释放旧的，不同时持有两份
            mat = self._coarse_matrix
            self._coarse_solve = CellCoarseSolve(mat, mat.ptc_diagonal(self._coarse_weights_host, dtau_flat,
                                                                       self.n_sps), self._coarse.ctx)
            self._coarse_dtau = self.red.xp.asarray(dtau_flat, dtype=self.red.xp.float64).ravel().copy()
        return CoarseCorrectedPreconditioner(base, self._coarse_solve, self._coarse_weights, self._real_slots,
                                             self.n_sps)

    def _fd_coupling_graph(self):
        """差分装配要不要同时截取耦合块：P0（每单元一个真实解点）、内存允许块 ILU、
        后端给了耦合图时返回 `CouplingGraph`，否则 None（只装配对角块）。"""
        if self._coupling_graph_fn is None or not self.use_ilu or max(self.n_real_prism, self.n_real_tet) != 1:
            return None
        if self._coupling_graph is None:
            self._coupling_graph = self._coupling_graph_fn()
        return self._coupling_graph

    @property
    def restart_cap(self):
        """内存规划给出的 GMRES 重启长度上限（None 为不限，`block_budget.py::plan_block_memory`）。"""
        return self.plan.restart_cap

    @property
    def flexible_fraction(self) -> float:
        """需要存预处理向量的分量占比（`linear_solve.py::krylov_restart`）：多层 K 循环整体灵活为 1；单元级粗校正
        只存粗层向量（每单元 `n_var` 个数，相对真实分量的占比）；线性为 0。"""
        if self.disabled_reason is None and self._coarse_matrix is not None:
            n_real = np.where(self.cell_is_prism, self.n_real_prism, self.n_real_tet)
            return float(self.cell_is_prism.size / n_real.sum())
        return 1.0 if self.flexible else 0.0

    @property
    def flexible(self) -> bool:
        """`preconditioner()` 给出的是否是非线性预处理（多层 K 循环、单元级粗校正的粗层 K 循环），GMRES 据此走
        灵活模式。`begin_step` 之后才有定论（耦合结构由首次装配给出）。"""
        if self.disabled_reason is not None:
            return False
        if self._coarse_matrix is not None:
            return True
        return self.coupling is not None and self._coarse.flexible(self.coupling)

    def preconditioner(self, dtau_flat: np.ndarray, n_var: int):
        """给定 `dtau` 下的预处理子（`begin_step` 之后调用，可多次）。块 ILU 档交给
        `coarse/selection.py`（本地多层或块 ILU，分布式再叠全局粗校正）。

        块 ILU 档的分解在 `J_cc` 不变时跨步复用，直到 dtau 相对分解时的几何平均比值超过
        `DTAU_REBUILD_RATIO`——除非伪时间项在两个时刻都已可忽略（`PTC_NEGLIGIBLE`）。复用的分解
        仍是合法的预处理子（右预处理 GMRES 解的是精确系统），dtau 略微过时只影响迭代数，由
        刷新判据兜住。此前每次调用都重新分解：湍流平板 P3（3072 单元）早期每步 GMRES 只 1 次，
        分解（当时 11 s）却占步耗时 75%。判据全部经全局归约，各 rank 一致（粗校正的构造是集体操作）。
        """
        if self.disabled_reason is not None:
            return PseudoTransientDiagonal(dtau_flat, n_var)
        if self.jac is None:
            raise RuntimeError("取预处理子之前须先装配单元块 Jacobian")
        dtau_flat = self._preconditioner_dtau(dtau_flat)
        if self.coupling is None:
            base = CellBlockJacobiPreconditioner(self.jac, dtau_flat, inverted=self.single_copy)
            return base if self._coarse_matrix is None else self._coarse_corrected(base, dtau_flat)
        if self._factorization_stale(dtau_flat):
            self._factored = None        # 先释放旧分解，不同时持有两份
            self._factored = self._coarse.make(self.jac, self.coupling, self.cross, dtau_flat)
            self._factored_dtau = self.red.xp.asarray(dtau_flat, dtype=self.red.xp.float64).ravel().copy()
        return self._factored

    def record(self, gmres_iters: int, accepted: bool) -> None:
        """一个 Newton 步结束后调用：更新刷新判据的依据。"""
        if self.disabled_reason is not None:
            return
        if self.baseline_iters is None:
            self.baseline_iters = gmres_iters
        self.last_iters = gmres_iters
        self.last_accepted = accepted
        self.age += 1
