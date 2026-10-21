"""AutoFlowCFD V2.0 - 平均流的一个 PTC-Newton-Krylov 步（全部后端共用）。

单机 CPU（`fr_solver/step.py`）与单机 GPU（`gpu/solver/gpu_solver/step.py`）
的 NEWTON_KRYLOV 分支都调用本文件的 `step_mean_flow_newton`；它们只提供
残差函数、状态、伪时间步长、归约对象与面相邻关系，隐式步本身的一切
（跨步状态、只解平均流 5 个变量、块 Jacobi 缓存、诊断日志）只有这一份。

## 一次 step() 做一个 Newton 步

外层 `solve()` 已经在做残差监控、自适应 CFL、checkpoint、Order
Continuation，Newton 外迭代放在它里面让这些机制原样生效（见 `jfnk.py`
模块文档）。

## 跨步状态（挂在求解器对象上，换阶时由 Order Continuation 清空）

    _newton_forcing         inexact-Newton 的 forcing term（Eisenstat-Walker）
    _newton_dtau_scale      PTC 的 dtau 缩放（步内缩档的剩余档数跨步延续）
    _newton_block_precond   单元块 Jacobi 缓存（冻结 J_cc，按判据重装配）
    _newton_last_info       上一步的诊断
"""

from typing import Callable, Optional

import numpy as np
from loguru import logger

from ..blocks.block_budget import plan_block_memory
from ..blocks.block_jacobi import BlockJacobiCache
from .forcing import EisenstatWalkerForcing
from .jfnk import step_newton_krylov
from ..reductions import LocalReductions

# 平均流守恒变量个数（唯一来源在状态容器；coupled_step 从这里取）
from autoflowcfd.core.fr_solver.state import N_MEAN_FLOW_VARS  # noqa: E402,F401


from .globalization import LINEAR_SOLVE_MAX_REL_RESIDUAL  # noqa: E402,F401  （唯一定义在 globalization.py）


def newton_step_ok(info: dict) -> bool:
    """本步 Newton 是否被**完整**接受（SER 律据此区分物理暂态与方向不可信）。

    三个条件：残差接受判据没有回溯（`theta >= 1`）、没有缩 dtau 档、线性求解
    至少把线性残差降到 `LINEAR_SOLVE_MAX_REL_RESIDUAL` 以下。第三条缺失时的真实
    停滞（plate_demo P0+SST，17.9 万单元）：CFL 3419 下块 Jacobi 预处理的 GMRES
    用满 200 次仍达不到容差，方向几乎无效，`R_new/R = 1.0000`；步照样"完整
    接受"、残差不动，SER 于是保持 CFL，永远停在预处理够不着的 CFL 上。
    """
    return (info["theta"] >= 1.0 and info["n_dtau_cuts"] == 0
            and info["linear_rel_residual"] <= LINEAR_SOLVE_MAX_REL_RESIDUAL)


def reset_newton_state(solver) -> None:
    """把隐式步的全部跨步状态置初值（模块文档"跨步状态"那几项 + 紧耦合步的湍流块缓存
    `_newton_turb_state`，见 `coupled_step.py`）。全部后端在构造时与换阶后调用这一个函数。

    换阶必须失效：forcing term 记的是上一步的残差量级（升阶后残差通常抬升
    一个量级以上，沿用会让第一步的线性容差要么过严要么过松）；dtau 缩放是
    按旧阶数的稳定性缩出来的；块 Jacobi 的块尺寸（真实解点数 x 变量数）随
    阶数变化，冻结的 `J_cc` 属于旧的离散空间。
    """
    solver._newton_forcing = None
    solver._newton_last_info = None
    solver._newton_unstable_steps = 0
    solver._dual_time_inner = None
    solver._newton_dtau_scale = 1.0
    solver._newton_local_dtau = None
    solver._newton_block_precond = None
    solver._newton_turb_state = None


def _format_newton_info(tag: str, info) -> str:
    """一个 Newton 步诊断的紧凑文本：Krylov 迭代数与是否达到 forcing 容差、
    残差接受判据的回溯比例 theta、逐单元物理性松弛（最小因子/被松弛单元
    占比）、缩档数，以及本步前后 `||R||` 之比——"步被完整接受但残差不动"
    这类停滞只有最后一项看得出来。"""
    if not info:
        return ""
    gm = ("ok" if info["gmres_info"] == 0 else "fail" if info["gmres_info"] < 0
          else f"maxit {info['linear_rel_residual']:.2g}")
    ratio = info["res_norm_new"] / info["res_norm"] if info["res_norm"] > 0 else 0.0
    txt = (f"{tag}: gmres={info['gmres_iters']}({gm}) theta={info['theta']:.3g} "
           f"relax={info['theta_physicality']:.3g}@{100 * info['limited_fraction']:.2g}% "
           f"R_new/R={ratio:.4f}")
    if info["n_dtau_cuts"]:
        txt += f" cuts={info['n_dtau_cuts']}"
    if info.get("local_dtau_min", 1.0) < 1.0:
        # 逐行局部伪时间步长缩放的最小值（被物理性松弛的行降过局部 dtau，见 physicality.py）
        txt += f" ldtau={info['local_dtau_min']:.2g}"
    if info.get("linear_growth", 0.0) > 1.0:
        # 线性系统自己预测残差放大（见 note_linear_growth）
        txt += f" lin_growth={info['linear_growth']:.2f}"
    return txt


#: 线性预测放大倍数超过它、且该步没有被完整接受，记为一次"线性化不稳定"的步。
UNSTABLE_LINEAR_GROWTH = 1.5

#: 累计这么多次之后给出提示（之后每隔同样的次数再提示一次）。
UNSTABLE_STEPS_BEFORE_WARNING = 5


def note_linear_growth(solver, info: dict) -> None:
    """记录一个 Newton 步的线性预测放大倍数，反复出现时提示"线性化算子有不稳定模态"。

    伪瞬态 Newton 步解的是 `(I/dtau + J) dU = -R`，于是线性化残差 `R + J dU = -dU/dtau`，
    `linear_growth = ||dU/dtau|| / ||R||`。`J` 的特征值实部全为正时 `(I + dtau J)^{-1}` 不放大，这个比值
    不超过 1 的量级；比值明显大于 1 说明 `J` 有实部为负、量级约 `1/dtau` 的特征值，`dtau` 再放大
    `I/dtau + J` 就接近奇异，Newton 方向失去意义、被残差接受判据回溯，自适应 CFL 因此停在
    `dtau ~ 1/增长率` 附近，残差不再下降。

    **这个诊断只说明离散算子有不稳定模态，不说明它是物理的还是数值的。** 2026-10-10 在 cube_demo（SA，P1）上
    测到 CFL 37/75/150 下比值 0.80/1.52/4.75，超出线性预测的残差 99% 集中在贴壁第一层的约 300 个棱柱里，当时
    据此判断为"分离剪切层的 Kelvin-Helmholtz 不稳定、该阶数下没有稳定的定常解、应改用瞬态"。那个判断是错的：
    没有看解本身。同一批单元里的速度从升到 P1 起就在增长（来流 30 m/s，升阶后 40 步 162 m/s，接着算瞬态
    10 步到 307 m/s、同一单元内解点速度反号），根因是壁面上外插出的涡粘为负、粘性通量与罚项变成反耗散
    （`fr_operators/flux_kernels/viscous.py::viscous_physical_flux_point` 文档"涡粘按非负取用"）。所以提示里
    先让使用者检查解是否物理。
    """
    if newton_step_ok(info) or not info.get("linear_growth", 0.0) > UNSTABLE_LINEAR_GROWTH:
        return
    solver._newton_unstable_steps += 1
    if solver._newton_unstable_steps % UNSTABLE_STEPS_BEFORE_WARNING == 0:
        logger.warning(
            "隐式步：已有 %d 个 Newton 步的线性系统自身预测残差放大（本步 %.2f 倍）且未被完整接受——离散算子有"
            "增长率约 1/dtau 的不稳定模态，伪时间步长无法继续放大。它可能是数值失稳，也可能是流动本身不稳定，"
            "先看解：全场最大速度、最大马赫数是否仍在来流的合理倍数内，P>=1 时同一单元内解点之间的速度差是否"
            "异常。解已经不物理就是离散的问题，换时间格式不会解决；解物理而残差不降（钝体尾迹脱落一类）才"
            "考虑瞬态计算并做时间平均。"
            % (solver._newton_unstable_steps, info["linear_growth"]))


def newton_monitor_suffix(solver) -> str:
    """迭代监控行的 Newton 诊断后缀（紧耦合步另附两个子系统的残差范数），显式格式返回空串。
    常规求解循环与 Order Continuation 的监控行共用这一份格式。"""
    info = getattr(solver, "_newton_last_info", None)
    parts = [_format_newton_info("NK", info)]
    if info and "res_norm_turbulence" in info:
        # 紧耦合步（coupled_step.py）：两个子系统各自的步前残差范数
        parts.append(f"|R_mean|={info['res_norm_mean']:.3e} |R_turb|={info['res_norm_turbulence']:.3e}")
    parts = [x for x in parts if x]
    return (" | " + " | ".join(parts)) if parts else ""


def step_mean_flow_newton(
    solver, residual: Callable, u_flat, dtau_flat, scales: np.ndarray, *,
    red: LocalReductions, cell_is_prism: np.ndarray, cell_colors: Callable[[], np.ndarray],
    order: int, positivity, block_assembler=None,
    coupling_graph: Optional[Callable] = None, global_coarse=None, physical_time=None,
):
    """平均流的一个 PTC-Newton-Krylov 步，返回 `(U_new_flat, info)`。

    Args:
        solver: 持有跨步状态的求解器对象（见模块文档那四个属性）。
        residual: `R(U_flat) -> (N, n_vars)`（启用低马赫预处理时是 `Gamma R`）。
        u_flat: `(N, n_vars)` 当前状态（`red.xp` 上）。
        dtau_flat: `(N,)` 逐 SP 伪时间步长（天花板，见 `jfnk.py`）。
        scales: `(n_vars,)` 守恒变量参考量级（`residual_diagnostics._reference_scales`）。
        red: 全局归约（单进程 numpy / cupy，或分布式子类）。
        cell_is_prism: `(n_cells,)` 主机端布尔掩码（单机"棱柱在前"，分布式
            local 排列里棱柱/四面体交错，所以不能用"前 n_prism 个"表达）。
        cell_colors: 返回 `(n_cells,)` 块 Jacobi 着色（主机端 numpy），只在
            首次构造缓存时调用。单机：按面相邻关系贪心着色；分布式：全局
            一致着色里本 rank 那一段（同色单元跨 rank 也不相邻，见
            `cell_blocks.py::CellBlockJacobian` 文档）。
        order: 当前阶数（决定每类单元的真实解点数）。
        positivity: 该后端的 `PositivityLimiter`（`time_integration/positivity`）；
            物理性限幅在它的点集上求值（解点 + 面通量点 + 过积分细点，见
            `PositivityLimiter.density_pressure_limits`）。
        block_assembler: 本步的解析单元块装配器 `(u0_flat, r0_flat) ->
            (blocks_prism, blocks_tet)`（`fr_residual/jacobian`，P>=1）；None 时块
            Jacobi 用着色差分装配（P0、熵稳定体积项等解析装配不覆盖的离散）。
            每步重新传入：它持有本步冻结的涡粘。
        coupling_graph: `() -> coloring.CouplingGraph`（距离 2 着色 + 本地模板单元对），
            只在首次构造缓存时保存、首次需要时调用。P0 没有解析装配器，差分装配按它
            同时截取面邻居耦合块、预处理用块 ILU（`cell_blocks.py` 模块文档）；
            None 时 P0 只有块 Jacobi。
        global_coarse: 分布式后端的 `coarse.CoarseCommContext`（块 ILU 档叠全局粗校正，
            `implicit/coarse/global_coarse.py`），只在首次构造缓存时使用；单机为 None。
        physical_time: 双时间步的 BDF 物理时间项（`physical_time.py`），稳态为 None。
    """
    from autoflowcfd.fr.native_padding import real_row_mask, real_sps_per_cell

    if u_flat.shape[1] != N_MEAN_FLOW_VARS:
        raise ValueError(f"平均流 Newton 步的未知量是 {N_MEAN_FLOW_VARS} 个守恒变量，收到 {u_flat.shape[1]} 列")
    if solver._newton_forcing is None:
        solver._newton_forcing = EisenstatWalkerForcing()
    if solver._newton_block_precond is None:
        cell_is_prism = np.asarray(cell_is_prism, dtype=bool)
        n_cells = cell_is_prism.size
        n_real_prism, n_real_tet = real_sps_per_cell(int(order))
        if u_flat.shape[0] % n_cells:
            raise ValueError(f"状态行数 {u_flat.shape[0]} 不是单元数 {n_cells} 的整数倍")
        n_sps = u_flat.shape[0] // n_cells
        # 平均流单独做 Newton 步只在没有输运型湍流方程时（层流、LES 一类）：预处理预算由它独占
        plan = plan_block_memory(red, cell_is_prism, n_sps, n_real_prism, n_real_tet, N_MEAN_FLOW_VARS, 0)
        solver._newton_block_precond = BlockJacobiCache(
            cell_is_prism=cell_is_prism, colors=cell_colors(), plan=plan,
            n_sps=n_sps, n_real_prism=n_real_prism,
            n_real_tet=n_real_tet, n_var=N_MEAN_FLOW_VARS, red=red, coupling_graph=coupling_graph,
            global_coarse=global_coarse)

    solver._newton_block_precond.assembler = block_assembler
    u_new_mf, info = step_newton_krylov(
        residual, u_flat, dtau_flat, np.asarray(scales),
        forcing=solver._newton_forcing, dtau_scale=solver._newton_dtau_scale,
        block_precond=solver._newton_block_precond, physicality=positivity.density_pressure_limits,
        rows_per_cell=u_flat.shape[0] // np.asarray(cell_is_prism).size, red=red,
        local_dtau_scale=getattr(solver, "_newton_local_dtau", None),
        norm_weights=positivity.norm_weights,
        real_rows=red.xp.asarray(real_row_mask(cell_is_prism, u_flat.shape[0] // np.asarray(cell_is_prism).size,
                                               order)),
        physical_time=physical_time)
    solver._newton_last_info = info
    note_linear_growth(solver, info)
    solver._newton_dtau_scale = info["dtau_scale"]
    solver._newton_local_dtau = info["local_dtau_scale"]
    if info["theta"] <= 0.0:
        logger.warning(
            "Newton 步未能前进（theta=0, gmres_info=%s, gmres_iters=%d, dtau_scale=%.3e, "
            "本步已缩 %d 档）——dtau 缩到下限仍拿不到被接受的步，那不再是步长问题"
            "（dtau->0 即显式前向 Euler、必然被接受），检查残差求值在当前状态上是否"
            "已经非物理" % (info["gmres_info"], info["gmres_iters"], info["dtau_scale"],
                       info["n_dtau_cuts"]))
    elif info["n_dtau_cuts"] > 0:
        logger.info("Newton 步缩 %d 档 dtau 后被接受（dtau_scale=%.3e, theta=%.3f）"
                    % (info["n_dtau_cuts"], info["dtau_scale"], info["theta"]))
    return u_new_mf, info
