"""AutoFlowCFD V2.0 - Newton 步里的线性求解：`(I/dtau + J) dU = -R` 的右预处理 GMRES 包装。

从 `jfnk.py` 拆出（2026-10-15，单文件不超 500 行）：GMRES 的迭代数上限与重启长度（按 Krylov 基内存预算
自适应）、预处理块定档时预留的 Krylov 基字节数、只存真实解点行的压缩（`_RealRows`）、在全局化同一个加权
范数里求解的 `_solve_direction`。Newton 外步（物理性限幅、残差接受判据、dtau 缩档）留在 `jfnk.py`。
"""

from typing import Optional

from .forcing import _ETA_MIN
from .globalization import ResidualNorm
from .gmres import FlexibleSplit, basis_itemsize, gmres_right
from .jacobian_vector import MatrixFreeJacobian
from .reductions import LocalReductions

#: GMRES 重启长度的**下限**。Krylov 基向量按 `(N, n_var)` 存，重启长度直接决定
#: 峰值内存：P2 下 79 万单元一个基向量 1.2 GB，大问题只能取小值。
GMRES_RESTART = 30

#: 重启长度的上限（正交化代价随 m 线性增长，m=200 时每次迭代约 200 次向量更新）。
GMRES_RESTART_MAX = 200

#: Krylov 基允许的内存（每个 rank / 每块设备）。重启长度按它自适应
#: （`krylov_restart`），夹在 `[GMRES_RESTART, GMRES_RESTART_MAX]` 之间。
#:
#: **为什么不是固定 30**（2026-09-26）：大 CFL 下迭代数上百时 GMRES(30) 反复
#: 重启、丢掉 Krylov 子空间，会停滞。plate_demo P0（17.9 万单元，一个基向量
#: 7 MB）CFL 1000 的同一线性系统、同一块 ILU、`rtol=0.1`：GMRES(30) 600 次仍停在
#: 0.136，GMRES(100) 181 次、GMRES(200) 121 次达到。固定 30 是按 P2 大网格的内存
#: 定的，小问题完全没必要受它限制。
KRYLOV_BASIS_BYTES = 2 * 2 ** 30

#: 单个 Newton 步允许的最大 Krylov 迭代数（= 最大残差求值次数）。
#:
#: **60 -> 200（2026-09-19，被真实运行修正）**：`dtau` 大的时候预处理后
#: 的算子是 `I + dtau*J`，特征值随 `dtau` 线性增长，GMRES 需要的迭代数
#: 跟着涨。实测 Blasius P1 在 CFL 100 下每步用满 60 次仍未达到 `eta`，
#: 于是拿到一个**半收敛**的方向 —— 而半收敛的方向既不是 Newton 方向也
#: 不是显式方向，是最坏的情形。
#:
#: 放开到 200 的同时加了两道闸，所以"成本失控"这件事由它们承担而不是
#: 由这个上限承担：
#:   * `gmres_info > 0`（未达容差）会被记录并交给残差接受判据；
#:   * 残差接受判据（`globalization.py::accept_step`）不接受让残差恶化的步，回溯到接受
#:     或者放弃这一步（`theta=0`，外层自适应 CFL 随之缩小 dtau）。
GMRES_MAX_ITER = 200


def krylov_restart(n_local_entries: int, red: LocalReductions, flexible: float = 0.0, itemsize: int = 8,
                   cap: Optional[int] = None) -> int:
    """按 Krylov 基内存预算给出的重启长度（见 `KRYLOV_BASIS_BYTES`）。

    分布式下各 rank 的 GMRES 必须走同样多步（内积是集体操作），所以取全局最小。
    灵活模式每步多存一个 `M^{-1} v_j`（`gmres.py`）：`flexible` 是需要存它的分量占比（整体灵活为 1、部分灵活为
    非线性列的占比、不灵活为 0；布尔值按 0/1）。
    `itemsize`：基向量每个元素的字节数（`gmres.py::basis_itemsize`）；下限 `GMRES_RESTART` 是按双精度
    定的内存量，单精度存储时同样的内存是两倍的向量数。
    `cap`：内存规划给出的上限（`block_budget.py::plan_block_memory` 为让每个块缓存都放得下而缩短重启长度时）。
    """
    per_step = itemsize * max(int(n_local_entries), 1) * (1.0 + float(flexible))
    m_local = int(KRYLOV_BASIS_BYTES // per_step) - 1
    m = int(red.min(red.xp.asarray([float(m_local)])))
    m = int(min(GMRES_RESTART_MAX, max(GMRES_RESTART * 8 // itemsize, m)))
    return m if cap is None else min(m, int(cap))


def krylov_basis_bytes(n_local_entries: int, red: LocalReductions, flexible: float = 1.0,
                       restart: Optional[int] = None) -> int:
    """一次 Newton 步的 Krylov 基最多占多少字节（预处理块定档时预留，`block_budget.py`）：重启长度与存储精度
    与 `step_newton_krylov` 同一规则（forcing 容差下限 `_ETA_MIN` 时的精度），`m + 1` 个基向量，灵活模式另加
    `m` 个预处理向量（按 `flexible` 占比，部分灵活见 `gmres.py`）。`restart` 给出时按它（规划缩短重启长度时）。"""
    itemsize = basis_itemsize(_ETA_MIN)
    m = krylov_restart(n_local_entries, red, flexible=flexible, itemsize=itemsize) if restart is None else int(restart)
    return int(((m + 1) + m * float(flexible)) * int(n_local_entries) * itemsize)


class _RealRows:
    """Krylov 向量只存真实解点行（原生基的零填充槽位不进基向量）。

    零填充行的残差恒为零（调用方保证、`step_newton_krylov` 入口核查）、Jacobian
    行列也为零，`A` 在那里就是 `I/dtau`、右端项为零，解在那里恒为零——在真实行
    上求解与全尺寸求解数学上等价。P3 原生基真实行只占约 42%（棱柱 40/64、四面体
    20/64），plate_demo P3 一个全尺寸基向量 0.46 GB、紧凑后 0.16 GB。
    做成类而不是闭包：在整个 Newton 步存活（项目规范）。
    """

    __slots__ = ("idx", "n_dof", "n_var")

    def __init__(self, real_rows, n_dof: int, n_var: int, xp):
        self.idx = xp.nonzero(xp.asarray(real_rows, dtype=bool))[0]
        self.n_dof, self.n_var = int(n_dof), int(n_var)

    def expand(self, x_1d, xp, out=None):
        """真实行写回全尺寸场。`out`：调用方复用的缓冲区（零填充行必须已经是零、且只被这里写真实行）。"""
        full = xp.zeros((self.n_dof, self.n_var), dtype=x_1d.dtype) if out is None else out
        full[self.idx] = x_1d.reshape(-1, self.n_var)
        return full

    def compact(self, y):
        return y[self.idx].reshape(-1)


def _solve_direction(jac: MatrixFreeJacobian, prec, dtau, r0, n_var: int, eta: float,
                     gmres_restart: int, gmres_max_iter: int, red: LocalReductions,
                     rows: Optional[_RealRows] = None, norm: Optional[ResidualNorm] = None,
                     review_at: Optional[int] = None):
    """解 `(I/dtau + J) dU = -R`，返回 `(du, gmres_iters, gmres_info, linear_rel_residual)`。

    `dtau` 是本次求解的逐行步长，Krylov 算子的对角项 `I/dtau` 在这里按它加；`prec` 只提供预处理
    作用（`apply`）。2026-10-10 以前对角项由 `prec.add_ptc_term` 提供：块 ILU 档的预处理对象跨步
    复用（`block_jacobi.py::preconditioner`，dtau 变化不到 2 倍不重分解），它带着**分解时**的 dtau，
    于是被求解的不是请求的那个系统。稳态下只是伪时间步与请求值差不到 2 倍；双时间步的物理时间项并在
    这个对角项里，BDF1 -> BDF2 时 `a0` 从 1 变 1.5 而算子仍按 1，Newton 从二次收敛退化成每次只降到
    0.45 倍（等熵涡，`tests/unit/test_dual_time_newton.py`）。
    线性求解用后端无关的 `gmres.py::gmres_right`（右预处理：它的收敛判据
    就是真实线性残差 `||R + (I/dtau+J) dU|| <= eta ||R||`，正是 inexact
    Newton 要求的量）。

    **线性残差在全局化的同一个范数里最小化**（2026-10-03）：`norm` 带逐行权重 `w`
    时（`ResidualNorm`），对 `D A M^{-1} D^{-1}`（`D = diag(sqrt(w))`）做 GMRES，解
    `x = M^{-1} D^{-1} y`（`D` 另含逐列的参考量级倒数，见 `ResidualNorm`）。它的残差就是 `D (b - A x)`，即接受判据与 SER 所用的加权范数；
    `D A M^{-1} D^{-1}` 与 `A M^{-1}` 相似，预处理后算子的谱不变。此前 GMRES 在未加权
    2 范数里最小化：残差被体积小 1e6 倍的单元主导，GMRES 在那里把线性残差压到
    `eta` 以下，在占全域体积的大单元上却让它变大——湍流平板层流 P2（`tests/validation/
    _flat_plate_case.py`）被回溯的步上未加权线性残差 0.03~0.10、体积加权 1.1~2.5 倍
    （非线性误差只有 0.001~0.003），线性模型本身就让全局度量恶化，接受判据回溯、SER
    收缩，CFL 卡在 ~60 近 200 步。

    `du` 为 `None` 表示线性求解产出了非有限方向（调用方据此缩小 `dtau`
    重试或放弃这一步，**不**静默用一个截断后的方向凑一步）。

    `jac` 由调用方构造并跨重试复用：它只依赖基态 `(U0, R0)` 与参考量级，
    与 `dtau` 无关，所以缩小 `dtau` 重试时不需要重算基残差、也不需要
    重建 Krylov 算子的基态部分（`n_matvec` 在它内部累计）。
    """
    n_dof = r0.shape[0]
    xp = red.xp
    dtau_col = xp.asarray(dtau, dtype=xp.float64).reshape(n_dof, 1)

    # D = sqrt(w) / scale：行权重与逐列的参考量级（`ResidualNorm` 的同一个范数）；加权时 GMRES 解的是 D A M^{-1} D^{-1}
    d = None
    if norm is not None and (norm.weights is not None or norm.inv_scales is not None):
        d_full = xp.ones((n_dof, n_var))
        if norm.weights is not None:
            d_full = d_full * xp.sqrt(norm.weights)[:, None]
        if norm.inv_scales is not None:
            d_full = d_full * norm.inv_scales[None, :]
        d = xp.ascontiguousarray(d_full).reshape(-1) if rows is None else rows.compact(d_full)
        if not bool(red.all_finite(d)) or red.min(d) <= 0.0:
            raise ValueError("残差范数权重在真实行上必须为正有限值（加权线性求解需要 D 可逆）")

    def to_vector(field):
        """`(n_dof, n_var)` 场 -> 求解向量（只取真实行）。"""
        return field.reshape(-1) if rows is None else rows.compact(field)

    # 真实行展开到全尺寸场的缓冲区，整个求解期间复用：零填充行恒为零，只有真实行被改写。每次新分配时
    # cube_demo 粗网格（28.6 万单元）P2 一个全尺寸场 0.37 GB，每次 GMRES 迭代分配 2~3 次、每次约 0.07 s
    # 花在缺页上。Krylov 算子与预处理作用共用一份：两个消费方（`jac.matvec`、预处理作用）都只读输入、不跨
    # 调用持有它，并且每次展开的结果在下一次展开之前就已被消费完（GMRES 先取预处理向量、再作用 Krylov
    # 算子，两者不嵌套；预处理作用的结果当场压回求解向量）。不用两份是为了不在残差求值的峰值上多占一个
    # 全尺寸场（`block_budget.py::RESIDUAL_WORK_FACTOR`）。
    if rows is None:
        def to_field(x_1d):
            return x_1d.reshape(n_dof, n_var)
    else:
        field_buf = xp.zeros((n_dof, n_var))

        def to_field(x_1d):
            return rows.expand(x_1d, xp, out=field_buf)

    # 对角项 `v / dtau` 在求解向量上算：与全尺寸 `to_vector(J v + v / dtau)` 的真实行逐位相同
    dtau_vec = to_vector(xp.ascontiguousarray(xp.broadcast_to(dtau_col, (n_dof, n_var))))

    def _apply_A(x_1d):
        out = to_vector(jac.matvec(to_field(x_1d)))
        if rows is None:
            out = out.copy()    # `matvec` 的返回值是它的工作数组（`jacobian_vector.py`），GMRES 会原地改写结果
        out += x_1d / dtau_vec
        if d is not None:
            out *= d
        return out

    def preconditioner_action(op):
        """`(n_dof, n_var)` 上的预处理作用 -> 求解向量（真实行、加权）上的右预处理作用。"""
        def action(y_1d):
            return to_vector(op(to_field(y_1d if d is None else y_1d / d)))
        return action

    b = to_vector(-r0)
    if d is not None:
        b = d * b
    split = None
    if prec.flexible and prec.nonlinear_columns is not None:
        mask = to_vector(xp.broadcast_to(xp.asarray(prec.nonlinear_columns), (n_dof, n_var)).copy())
        split = FlexibleSplit(mask, preconditioner_action(prec.apply_linear))

    du_1d, iters, info, rel = gmres_right(
        _apply_A, b, preconditioner_action(prec.apply), rtol=eta,
        restart=gmres_restart, max_iter=gmres_max_iter, red=red, flexible=prec.flexible, review_at=review_at,
        split=split)
    if info < 0 or not red.all_finite(du_1d):
        return None, iters, -1, float("nan")
    # 方向由调用方持有，不能放进复用的缓冲区
    du = du_1d.reshape(n_dof, n_var) if rows is None else rows.expand(du_1d, xp)
    return du, iters, int(info), float(rel)
