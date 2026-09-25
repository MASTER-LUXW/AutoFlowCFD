"""AutoFlowCFD V2.0 - 湍流源项与输运的每步求值。

从 `core/fr_solver/turbulence.py` 拆出（2026-09-24）。纯搬家，逻辑未改。
"""

from typing import Optional

import numpy as np
from loguru import logger

from autoflowcfd.core.turbulence.des import IDDESModel
from autoflowcfd.core.fr_residual.viscous import compute_gradients as _compute_gradients_generic
from .init import (
    _filter_matrices_are_identity,
    _update_production_ramp,
)


def compute_turbulence_source(solver, dt) -> Optional[tuple]:
    """计算湍流模型源项（对应 FRSolver.compute_turbulence_source）。

    Args:
        dt: k/omega 场显式更新使用的时间步长，直接转发给
            `turb_model.update_fields`。调用方（fr_solver/step.py）
            按 scheme 传入不同的量：稳态加速模式（SSP-RK/IMEX）传
            逐 SP 的局部 CFL 步长数组 dt_local（形状 (n_cells, n_sps)，
            与 dk_total/domega_total 广播兼容），DUAL_TIME 模式传标量
            物理 dt。之前这里统一收到的是原始物理 dt 标量，未经
            cfl.py 的阶数/粘性/几何刚性收紧，真实复现过在合成 Couette
            +SST 算例与 cube_demo 生产网格上都会让 omega 场显式积分
            失稳（一步内放大几十倍，Order Continuation 升阶后几步内
            发散至 inf/NaN）——修复见 step.py::step 文档。
    """
    if solver.turb_model is None:
        return None

    # 更新湍流产项渐变因子（每步调用，production_factor 从 0 渐增到 1）
    _update_production_ramp(solver)

    Q, grad_vel, d_wall, mu = prepare_turbulence_inputs(solver)
    Sk, S_omega, dk_dt, domega_dt, transport_k, transport_omega = evaluate_turbulence_rates(
        solver, Q, grad_vel, d_wall, mu, apply_des=True)

    solver.turb_model.update_fields(dt, dk_dt, domega_dt,
                                     transport_k=transport_k,
                                     transport_omega=transport_omega)
    finalize_turbulence_update(solver, dt)
    return (Sk, S_omega)


def prepare_turbulence_inputs(solver):
    """一步之内只依赖平均流的输入：`(Q, grad_vel, d_wall, mu)`。

    显式路径每步调用一次；隐式路径（`implicit.py`）在整个湍流 Newton 步内
    冻结它们（平均流不动），只让 `k/omega` 变化。
    """
    Q = solver.state.Q
    # 真实 bug 修复（2026-09-03，cube_demo 791,492 单元真实网格 Order
    # Continuation P0->P1 跨阶后延迟发散排查发现）：此前这里对*守恒*变量
    # U 求梯度、直接切片 [1:4] 当速度梯度用——U[...,1:4] 是动量
    # (rho*u,rho*v,rho*w)，grad(rho*u) != rho*grad(u)，除非密度梯度处处
    # 为零。低马赫数流场里密度接近均匀，这个误差通常小到不可见，一旦
    # 出现哪怕很小的局部密度扰动（真实复现：Order Continuation 插值截断
    # 误差），这里算出的"应变率"就会混入一个虚假的 u_i*grad(rho)/rho
    # 分量，經 SST 产生项(P_k~nu_t*S^2)反馈进涡粘系数，涡粘再反馈进动量
    # 残差放大速度/密度扰动——形成真实的正反馈失稳（真实网格上表现为
    # P1 阶段前~20步几乎不动、随后 100 步内速度峰值从 50 m/s 涨到 600+
    # m/s，k/omega 双双撞上安全上限）。同一代码库里
    # `fr_residual/viscous_flux.py` 的主残差路径一直是对的（先
    # conserved_to_primitive 转 Q 再求梯度、再切片），这里改成同一模式。
    grad_vel = _compute_gradients_generic(Q[:, :, 1:4], solver.ops, solver.mesh)

    d_wall = solver.wall_distance
    if d_wall is not None:
        expected_shape = (solver.state.n_cells, solver.state.n_sps)
        if d_wall.shape != expected_shape:
            logger.warning(
                f"Wall distance shape mismatch: expected {expected_shape}, got {d_wall.shape}. "
                f"Rescaling to match current state..."
            )
            if d_wall.ndim == 2:
                mean_d = np.mean(d_wall, axis=1, keepdims=True)
                d_wall = np.tile(mean_d, (1, solver.state.n_sps))
                solver.wall_distance = d_wall
            else:
                raise RuntimeError(f"Cannot rescale wall distance from shape {d_wall.shape}")

    if d_wall is None:
        if solver.turb_model_name in ["SST", "DDES", "IDDES", "WMLES", "LES"]:
            raise RuntimeError(
                f"Wall distance field not computed for turbulence model '{solver.turb_model_name}'. "
                f"Please call compute_wall_distance_field() before solving, or ensure wall distance "
                f"is provided during solver initialization. Industrial-grade calculation requires "
                f"accurate wall distance, not simplified estimates."
            )
        else:
            n_cells, n_sps = solver.state.U.shape[:2]
            volumes = solver._get_cell_volumes()
            h_char = np.power(np.abs(volumes), 1.0 / 3.0)
            d_wall = np.tile(h_char[:, np.newaxis], (1, n_sps))
            logger.warning(f"Using characteristic length scale as wall distance estimate")

    mu = getattr(solver, 'mu_molecular', 1.8e-5)  # 分子粘度（k/omega方程自身扩散系数用分子粘度，与平均流粘性应力
    # 张量所用的有效粘度[core/fr_solver.py::_get_turbulent_viscosity_field]是两个不同量）
    return Q, grad_vel, d_wall, mu


def evaluate_turbulence_rates(solver, Q, grad_vel, d_wall, mu, *, apply_des: bool):
    """在 `turb_model` 当前的 `k_field/omega_field` 上求 `dk/dt`、`domega/dt`。

    返回 `(Sk, S_omega, dk_dt, domega_dt, transport_k, transport_omega)`：
    `dk_dt/domega_dt` 是**源项部分**（已除以 rho），输运部分单独返回——
    显式路径的 `update_fields` 只对源项做点隐式阻尼，所以两者必须分开。

    副作用：`compute_source_terms` 会刷新模型上的 `nu_t`、混合 `beta` 与
    realizability 下限（都是当前场的函数）。`apply_des=True` 时还按刚算出的
    `nu_t` 刷新 DES 长度尺度（供**下一步**用，见下方原注释）——隐式路径的
    每次残差求值必须传 `False`，否则 Newton 内部的试探场会改写它。
    """
    grad_k = None
    grad_omega = None
    if solver.turb_model_name in ["SST", "DDES", "IDDES"]:
        k_expanded = solver.turb_model.k_field[:, :, np.newaxis]
        omega_expanded = solver.turb_model.omega_field[:, :, np.newaxis]

        from autoflowcfd.core.fr_residual.viscous import compute_scalar_gradient

        grad_k = compute_scalar_gradient(k_expanded, solver.ops, solver.mesh)
        grad_omega = compute_scalar_gradient(omega_expanded, solver.ops, solver.mesh)

        # 正性保持检查：防止梯度过大导致负值（工业计算的梯度限幅处理）
        #
        # errstate 包裹（真实 bug，2026-08-22，用户直接在真实网格 P0->P1
        # 转阶后的输出里看到这条警告发现）：跟 turbulence/transport.py
        # 同一处（见该文件对应注释的完整原理）完全同一个失效模式——退化
        # 单元上理论为常数的 k/omega 场求梯度，度量比值 adj(J)/det(J) 把
        # 浮点噪声放大到 >1e150，np.linalg.norm 内部 x*x 先于下面的裁剪
        # 逻辑溢出到 inf。2026-08-21 修 transport.py 那份独立副本时，这里
        # （同一份裁剪逻辑最早的位置，transport.py 的注释原文引用的正是
        # 本函数）被遗漏了——两处形状不同（这里是 (n_cells,n_sps)，
        # transport.py 是 (n_cells,n_sps,3)->(...,)但语义一致）但都是同一
        # 组 np.linalg.norm+裁剪代码，只补了一处。inf 输入不影响下面裁剪
        # 结果的正确性（inf>max_grad_mag 恒真，scale=max_grad_mag/inf=0，
        # 裁剪趋于 0 不是 nan），errstate 只是抑制警告噪音。
        with np.errstate(over='ignore', invalid='ignore'):
            max_grad_mag = 1e6
            grad_k_mag = np.linalg.norm(grad_k, axis=-1)
            grad_omega_mag = np.linalg.norm(grad_omega, axis=-1)

            if np.any(grad_k_mag > max_grad_mag):
                scale_k = max_grad_mag / np.maximum(grad_k_mag, 1e-10)
                grad_k *= np.clip(scale_k[:, :, np.newaxis], 0, 1)

            if np.any(grad_omega_mag > max_grad_mag):
                scale_omega = max_grad_mag / np.maximum(grad_omega_mag, 1e-10)
                grad_omega *= np.clip(scale_omega[:, :, np.newaxis], 0, 1)

    # DDES 的有效长度尺度 (sst_model.des_length_scale) 依赖涡粘 nu_t，而
    # nu_t 只在 compute_source_terms 内部才会被重新计算（sst_model.nu_t 是
    # 上一次调用留下的缓存），两者互相依赖对方的输出，天然只能"慢一拍"：
    # apply_to_sst_model 必须排在 compute_source_terms 之后，用这一步刚算
    # 出的 nu_t 算出 des_length_scale，供*下一步*使用——这是有意为之的
    # 近似（同阶数内连续迭代时物理上完全合理），不是疏忽，因此调用顺序
    # 本身不能颠倒（真实网格已验证：颠倒后 nu_t 变成读取上一步的旧维度
    # 缓存，问题只是从 des_length_scale 转移到 nu_t，没有解决）。
    #
    # 真正需要处理的是"跨阶数切换"这一刻：des_length_scale 是按上一个阶数
    # 的 SPs 维度算出的，阶数切换后与已经正确插值过的 k_field 形状不匹配。
    # 修复见 order_continuation.py：阶数切换时显式清空 des_length_scale，
    # 让切换后的第一步自动退回标准 RANS 耗散项（不依赖过期维度的缓存），
    # 而不是在这里颠倒调用顺序。
    Sk, S_omega = solver.turb_model.compute_source_terms(Q, grad_vel, d_wall, mu, grad_k=grad_k, grad_omega=grad_omega)

    if apply_des and solver.ddes_model is not None:
        rho = Q[:, :, 0]
        nu_field = mu / np.maximum(rho, 1e-10)
        if isinstance(solver.ddes_model, IDDESModel):
            # IDDES 的 Δ/f_B/f_e 公式需要逐单元 h_max/h_wn（见
            # init_turbulence_models 里 IDDES 分支的一次性缓存），与
            # DDESModel 基类只需要 cell_volumes 的 cube_root(V) 网格
            # 尺度公式结构不同，走独立的 apply_to_sst_model_iddes。
            solver.ddes_model.apply_to_sst_model_iddes(
                solver.turb_model, d_wall, solver._iddes_h_max, solver._iddes_h_wn,
                nu_field, grad_vel,
            )
        else:
            cell_volumes = solver._get_cell_volumes()
            # h_max（2026-09-02）：`init_turbulence_models` 的 DDES 分支
            # 现在也会设置 `solver._iddes_h_max`（与 IDDES 同一处几何量、
            # 同一个一次性缓存，见该分支文档）——传入后 apply_to_sst_
            # model 改用各向异性感知的 max_edge 网格尺度，不再是
            # cube_root(V)。`getattr` 兜底：极少数不经过 init_
            # turbulence_models（例如脱离 solver 直接构造 DDESModel 的
            # 测试场景）没有这个属性时，退化为 cube_root，不报错。
            solver.ddes_model.apply_to_sst_model(
                solver.turb_model, d_wall, cell_volumes, nu_field, grad_vel,
                h_max=getattr(solver, '_iddes_h_max', None),
            )

    # Sk/S_omega 是 compute_source_terms 按标准 SST 公式算出的 rho*k、
    # rho*omega 方程源项（P_k/D_k/P_omega/D_omega/CD_omega 都显式带 rho
    # 因子），但 turb_model.k_field/omega_field 存的是 k、omega 本身
    # （不是 rho*k/rho*omega，初值 1e-6/1.0 也是 k/omega 量级而非 rho*k/
    # rho*omega 量级）——直接 self.k_field += dt*Sk 会缺一个 1/rho，
    # 量纲不对。这里换算成 dk/dt ≈ Sk/rho（对缓变 rho 的标准近似：
    # d(rho*k)/dt = rho*dk/dt + k*drho/dt ≈ rho*dk/dt）再传给
    # update_fields。
    rho = Q[:, :, 0]
    dk_dt = Sk / np.maximum(rho, 1e-10)
    domega_dt = S_omega / np.maximum(rho, 1e-10)

    # 完整输运项（对流+扩散）：对 SST/DDES 模型计算 k/omega 的 FR 空间输运
    # 残差，使 k/omega 不再仅是逐点 ODE 源项弛豫，而是真正随流场对流、
    # 跨单元扩散。见 core/turbulence_transport.py 模块文档。
    transport_k = None
    transport_omega = None
    if solver.turb_model_name in ["SST", "DDES", "IDDES"]:
        from autoflowcfd.core.turbulence.transport import compute_turbulence_transport_residual
        # grad_vel 复用上面已经为 compute_source_terms 算过的同一份值
        # （同一个 solver.state.U，两处之间没有任何修改），避免
        # compute_physical_gradient 这个已知热点被重复调用——见
        # compute_turbulence_transport_residual 参数文档的性能说明。
        #
        # grad_k/grad_omega 刻意不复用、仍让本函数内部重新计算：上面
        # 那两份在传给 compute_source_terms 之前经过了一次条件触发的
        # 梯度幅值裁剪（grad_k_mag/grad_omega_mag > 1e6 时 *= 缩放，
        # 见上方"正性保持检查"），而 transport 内部的 CD_kw/F1
        # 计算历来用的是未裁剪的原始梯度——两者在裁剪实际触发的
        # （罕见）情形下不是同一个数值，复用会在那个边界情形下悄悄
        # 改变 transport 的 F1/CD_kw 取值，不是纯粹的性能优化。这里
        # 没有把握判断"两处都用裁剪后的值"在物理上是否更对，宁可
        # 保持这部分原有行为不变，只拿 grad_vel 这个确定安全（两处
        # 之间毫无改动、任何情形下都是同一个数值）的部分。
        #
        # 之前这里 `try/except Exception` 把任何失败（包括真正的编程
        # 错误——形状不匹配、numba 编译失败等）都静默降级为"仅源项
        # 更新"，只打一条 warning，不中断求解——与本项目在别处反复强调
        # 的"不允许静默地什么都不做"（见 boundary/fr_ghost_state.py::
        # BoundaryGhostStateProvider 文档）、"必须先查清原因，不能静默
        # 截断/忽略"（见 face_flux_points/merge.py 文档）等原则相悖：
        # 真实 bug 会被这个 except 吞掉，求解器带着一个悄悄退化、外部
        # 毫无察觉的湍流模型继续跑完整个仿真。真实复现过的输运计算失败
        # 目前没有已知的"预期内、可安全忽略"的情形，故不再兜底捕获，
        # 让真正的错误照常抛出、中断求解。
        # grad_k/grad_omega 同样复用（#7 内存修复，2026-08-28，
        # cube_demo 79万单元 P2+DDES 首次真实 CLI 冒烟测试触发 OOM
        # 崩溃后追查发现）：本函数上面几行刚为 compute_source_terms
        # 算好、裁剪过的同一份 grad_k/grad_omega，此前这里只传了
        # grad_vel、没有一并传 grad_k/grad_omega——
        # compute_turbulence_transport_residual 本身早就支持接收这两者
        # （见该函数文档），调用方一直没有真正利用，导致内部又重新算
        # 一遍完全相同的梯度（~1GB 冗余数组，79万单元 P2 阶段）。数学上
        # 严格等价：本函数上面的裁剪是原地 `grad_k *= clip(...)`，
        # compute_turbulence_transport_residual 内部对已经满足裁剪阈值
        # 的输入重新检查同一个阈值必然是 no-op，不会改变数值结果。
        transport_k, transport_omega = compute_turbulence_transport_residual(
            solver, grad_vel=grad_vel, grad_k=grad_k, grad_omega=grad_omega,
            flat_face_override=getattr(solver, "_turbulence_flat_face_override", None),
        )

    return Sk, S_omega, dk_dt, domega_dt, transport_k, transport_omega


def finalize_turbulence_update(solver, dt, *, omega_wall_relaxation: bool = True) -> None:
    """`k/omega` 更新之后的两道后处理：模态滤波（非恒等滤波矩阵时）与
    omega 壁面松弛。显式与隐式路径共用，顺序不变。

    `omega_wall_relaxation=False`：隐式路径把同一个壁面条件作为残差内的
    强约束施加（`implicit.py`），不能再做一次步后投影——那会让 Newton 解
    的方程与实际被执行的更新不一致，残差永远降不下去。"""
    # 真实 bug 修复（2026-09-12，cube_demo 791,492 单元真实网格 P1 直连
    # 长程测试发现）：k/omega 场同样需要与平均流一致的模态滤波，见
    # fr_solver/filter.py::filter_scalar_field 完整推导——此前"湍流走
    # 单步显式更新、不经过多级 RK 因此不会积累混叠"的排除理由已被真实
    # 数据证伪（P1 独立发散，omega 8.6% 单元逼近安全上限，定位到具体
    # 单元内部相邻解点间出现数量级跳变，外插到面后被上风格式放大成
    # 巨大虚假对流残差）。用与平均流完全同一套 filter_prism/filter_tet
    # 矩阵，P0（n_sps=1）下矩阵退化为单位矩阵，天然是无操作。
    if solver.mesh.n_sps_per_cell > 1 and not _filter_matrices_are_identity(solver.ops):
        from autoflowcfd.core.fr_solver.filter import (
            compute_turb_troubled_mask, filter_scalar_field,
            filter_scalar_field_gated, resolve_turb_filter_gate,
        )
        n_prism = solver.mesh.n_prism_cells
        # 门控维度与平均流的 `AFCFD_FILTER_MODE` **独立**（2026-09-15）：
        # 真实网格 250 步对照决定性证明两维的效果可以完全分离——off 与
        # sensor 两档的平均流轨迹几乎逐位相同，om_max 却差一个量级，差异
        # 全部来自这里。理由与实测数据见 filter.py::resolve_turb_filter_gate。
        # 默认 "all" 与此前行为逐位一致。
        if resolve_turb_filter_gate() == "sensor":
            order = getattr(solver, "current_order", None)
            if order is None:
                order = solver.order
            troubled = compute_turb_troubled_mask(
                solver.turb_model.k_field, solver.turb_model.omega_field,
                int(order), n_prism=n_prism)
            solver._turb_filter_troubled_frac = float(np.mean(troubled))
            solver.turb_model.k_field = filter_scalar_field_gated(
                solver.turb_model.k_field, solver.ops.filter_prism,
                solver.ops.filter_tet, troubled, n_prism=n_prism,
            )
            solver.turb_model.omega_field = filter_scalar_field_gated(
                solver.turb_model.omega_field, solver.ops.filter_prism,
                solver.ops.filter_tet, troubled, n_prism=n_prism,
            )
        else:
            solver.turb_model.k_field = filter_scalar_field(
                solver.turb_model.k_field, n_prism, solver.ops.filter_prism, solver.ops.filter_tet,
            )
            solver.turb_model.omega_field = filter_scalar_field(
                solver.turb_model.omega_field, n_prism, solver.ops.filter_prism, solver.ops.filter_tet,
            )
        # 滤波可能把场值推到正性下限以下（滤波器系数含负权重，理论上
        # 可能），滤波后必须重新过一遍正性/上界限制器，不能假设滤波
        # 输出天然满足这些约束。
        solver.turb_model.apply_positivity_limiter()

    # 真实 bug 修复（2026-09-04）：omega 壁面 Wilcox 解析值只在对流项
    # （近壁趋于零，因为无滑移）生效，扩散项（近壁 omega 动力学的主导
    # 机制）此前完全没有把这个约束传递进去——见 transport.py::
    # enforce_omega_wall_relaxation 文档，这是 grad_vel 修复后长程复现
    # 里仍持续发散的第二个独立根因（边界层单元 omega 长期不受约束地
    # 衰减到下界，经 nu_t 近零分母奇点放大湍流粘性比，持续向平均流
    # 注入过量粘性应力）。
    #
    # 2026-09-05 曾尝试把下面这个松弛改成按 dt/d1/扩散系数物理推导的
    # "点隐式"动态系数（数学上无条件稳定），真实网格验证证伪（细网格
    # 近壁单元动态系数天然趋近 1，几步内把 k_mean 从 38 打到 0.17）
    # 已完整撤销，见 enforce_omega_wall_relaxation 文档。**当前实现是
    # 固定 relax=0.5，`dt` 只是为了不破坏调用方签名而保留的未使用参数
    # ——不要被这行调用误导，真正的行为以被调用函数的文档为准。**
    if omega_wall_relaxation and solver.turb_model_name in ["SST", "DDES", "IDDES"]:
        from autoflowcfd.core.turbulence.transport import enforce_omega_wall_relaxation
        enforce_omega_wall_relaxation(
            solver, dt, flat_face_override=getattr(solver, "_turbulence_flat_face_override", None),
        )
