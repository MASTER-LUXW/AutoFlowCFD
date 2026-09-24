"""AutoFlowCFD V2.0 - 粘性残差顶层：体积项 + 界面项 + IP 罚项

从 `src/autoflowcfd/core/fr_residual/viscous_flux.py`(原 560 行)拆出(2026-09-24, 项目"单文件不超 500 行"规范)。**纯搬家, 逻辑未改**。
"""

import os

import numpy as np

from autoflowcfd.core.fr_operators.gradients import compute_physical_gradient

from autoflowcfd.core.fr_operators.flux_kernels import (
    resolve_viscous_ip_constant,
)

from autoflowcfd.core.fr_operators.flux_kernels import viscous_physical_flux_batch

from autoflowcfd.core.fr_operators.volume_contract import (
    contract_shared_operator_2axis,
    compute_adj_j,
    contravariant_flux_from_metric,
    get_overintegration_context,
)
from .overintegration import _viscous_volume_overintegrated, resolve_viscous_overintegration
from .pointwise import compute_temperature


def compute_viscous_residual_fr(U: np.ndarray, mesh, ops, mu: float, Pr: float,
                                 mu_t_field=None, Pr_t: float = 0.9,
                                 boundary_ghost_provider=None,
                                 flat_face_override=None) -> np.ndarray:
    """计算真实面耦合的 FR 粘性残差 dU/dt（物理空间，已除以 det(J)）。

    Args:
        U: 守恒变量 (n_cells,n_sps,n_vars)，只使用前5个欧拉变量
        mesh: HighOrderMesh（需要 face_connectivity, face_flux_points, jacobians）
        ops: FROperators
        mu: 分子动力粘度（标量）
        Pr: 分子普朗特数
        mu_t_field: 湍流涡粘度场，形状 (n_cells, n_sps) 或 None（层流/未激活
            湍流模型时视为全零）。这是 T-01/T-04/T-06 湍流-平均流耦合的
            接入点——调用方（core/fr_solver.py）负责把 SST/DDES/WALE 算出
            的 nu_t 乘以密度后传入，见该模块修复说明。
        Pr_t: 湍流普朗特数
        boundary_ghost_provider: 边界面幽灵态提供者，与无粘残差
            （fr_residual_inviscid.py）共用同一个实例/同一套 WALL/INLET/
            OUTLET/FARFIELD/SYMMETRY 逻辑（见 boundary/fr_ghost_state.py），
            签名 (face_idx, Q_owner_fp, true_normal) -> Q_ghost_fp。
            None 时退化为 DefaultGhostProvider（零梯度外插，Q_ghost=Q_owner，
            BR1 跳跃恒为零）。

            此前这里完全不使用该参数、边界面统一取 Q_n=Q_o（内部值原样
            镜像），导致 BR1 平均 Q_avg 恒等于 Q_o、跳跃项 jump_owner 恒为
            零——即固壁上不存在任何由粘性方程施加的边界约束，无滑移
            剪应力不存在（真实数值验证：真实网格上 WALL 面的粘性校正项
            逐面精确为 0，与是否退化/网格质量无关，是恒等式）。现在真正
            调用 boundary_ghost_provider 取得反映边界条件的幽灵原始变量
            （例如 WALL 用速度镜像取反构造 Q_avg 速度=0，真正的无滑移）。

            **速度梯度 gv_n / mut_n 仍取内部值镜像**——标准 BR1/LDG 做法：
            它们没有独立的边界"真值"，边界约束通过状态跳跃在通量里体现
            （动量方向另有 IP 罚项，见 fr_operators/flux_kernels.py::
            viscous_boundary_penalty_tilde）。特别**不能**"顺手"把无滑移壁
            的 gv 也做法向镜像：壁面切向速度的法向导数**就是**壁面剪应力
            本身，镜像掉等于把它抹成零。

            **温度梯度 gT_n 已按热边界类型分派（2026-09-15 修复）**：
            此前 gT_n 也一律取内部值，对能量方程留下一处真实的不自洽——
            壁面 ghost 态复制 rho/p（温度无跳跃，见 fr_ghost_state.py::
            wall_ghost_state 的"热边界条件"一节，语义上是绝热壁），而
            IP 罚项只覆盖动量分量（`for v in range(1,4)`），于是面上平均
            法向温度梯度等于内部值、一般非零：实现出来的壁面热条件既不是
            绝热（q_w=0）也不是等温，而是"按内部梯度透射"。

            现在 WALL/SYMMETRY（见 fr_ghost_state.py::
            ADIABATIC_THERMAL_BC_TYPES）改为法向分量镜像
            `∇T_ghost = ∇T_int − 2(∇T_int·n)n`，于是 BR1 面平均
            `∇T_avg = ∇T_int − (∇T_int·n)n` 的法向分量**精确为零**，投影到
            该面的离散传导热通量 `a·q_avg` 恒等于零——是恒等式，不是
            "收敛到某个容差"。镜像用的 n 取自逆变行 `adj_row`（与面法向
            平行），这样"恒为零"的正是真正进入残差的那个投影量本身，而
            不是一个与之差一个截断误差的替代量。INLET/OUTLET/FARFIELD
            保持透射（流入/流出边界上法向热通量本就应该非零）。非
            `BoundaryGhostStateProvider` 的 provider（DefaultGhostProvider、
            测试 stub）没有 BC 语义，一律按透射处理，既有行为逐位不变。

            **修复前的实测量级，以及为什么那个数字不能再被引用**：
            79 万单元 cube_demo P1 iter=300 检查点上，温度场全域变化
            5.086 K 但**胞内** |∇T| 只有 mean=5.9e-12 / p99=6.4e-11 /
            max=3.7e-10 K/m，据此虚假壁面热通量密度
            |q_n| = k|∇T| ≈ 6.1e-14 W/m^2，相对来流焓通量密度
            rho*U*cp*T = 1.18e7 W/m^2 约 5e-21。**这个 5e-21 是被另一个
            缺陷压出来的**：legacy 模态滤波器每个 RK stage 清掉一整阶模态
            （见 fr/modal_filter.py 模块文档），P1 的解实际退化成分片常数，
            胞内梯度当然是机器零。滤波器一旦放开，实测胞内 |∇u|/(U/h) 从
            7.7e-16 跳到 O(1)，这一项的真实影响必须重新测量。
            （校验说明：上述 |∇T| 数字所用的 `compute_physical_gradient`
            调用方式已在已知线性场 T=x / T=3y 上验证到机器精度
            7.6e-16，不是算子用错导致的虚低。）

    Returns:
        residual: (n_cells, n_sps, 5)
    """
    from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive, DefaultGhostProvider

    ghost_provider = boundary_ghost_provider if boundary_ghost_provider is not None else DefaultGhostProvider()

    n_cells = mesh.n_cells
    n_sps = mesh.n_sps_per_cell
    n1d = mesh.n_points_1d

    mu_t_field = np.zeros((n_cells, n_sps)) if mu_t_field is None else mu_t_field

    Q = conserved_to_primitive(U[..., :5])  # (n_cells,n_sps,5)
    T = compute_temperature(Q)  # (n_cells,n_sps)

    grad_Q = compute_physical_gradient(Q, mesh, ops)  # (n_cells,n_sps,5,3)
    grad_vel = grad_Q[:, :, 1:4, :]  # (n_cells,n_sps,3,3): grad_vel[...,i,j]=d(u_i)/dx_j
    grad_T = compute_physical_gradient(T[:, :, None], mesh, ops)[:, :, 0, :]  # (n_cells,n_sps,3)

    det_jacs = mesh.jacobians["det_jacs"].reshape(n_cells, n_sps)
    inv_jacs = mesh.jacobians["inv_jacs"].reshape(n_cells, n_sps, 3, 3)
    adj_j = compute_adj_j(det_jacs, inv_jacs)

    # 体积项：整条 物理通量→逆变通量→散度 链按单元分块执行（B-12 P2 OOM
    # 修复第④级，2026-08-26）。历史背景：本函数体积项最初是纯 einsum 实现，
    # 后按与 fr_residual_inviscid.py 相同的性能优化路径改为 `viscous_physical_
    # flux_batch`（复用已逐位验证过的 numba 逐点 kernel）+ `np.matmul`/
    # `contract_shared_operator_2axis`（与原 einsum 公式严格等价，见
    # fr_volume_contract.py 模块文档）。retest5（cube 79万单元生产网格，
    # P0+P1 全部走完、P2 首次无粘残差求值也已通过——证明过积分链分块修复③
    # 生效）显示失败点前移到这里：SSP-RK3 第二阶段（L2）粘性残差求值时全场
    # 分配 G_phys（~2.6GiB = 791492 单元×27 SP×15 分量）MemoryError 崩溃。
    # 根因与内存账同 fr_residual/inviscid.py 过积分链分块注释（仪表化实测见
    # _tmp_review/diag_p2_13_prod.log：峰值 freeCommit 仅 0.52GB）：本段的两个
    # 全场数组 G_phys/G_tilde 各 ~2.6GiB，加上 grad_vel_flat 全场强制拷贝
    # ~1.4GiB，与第一阶段驻留量共存时超出剩余 commit 额度。这条链每一步都是
    # cell 局部的（`viscous_physical_flux_batch` 是 numba 逐点纯 gather、
    # `np.matmul`/`contract_shared_operator_2axis` 按 cell 批量，cell 之间零数据
    # 依赖），把 cell 轴切块、块内走完整条链，每步计算形状/求和顺序与全场版
    # 逐位一致，数值结果不变（等价性判据见
    # tests/unit/test_fr_viscous_flux_kernel_crosscheck.py）。块大小同样取 32768：
    # 块内瞬态 ≈0.3GiB（G_phys+G_tilde ~0.2GiB + grad_vel 块拷贝 ~0.07GiB），
    # 对比全场版同时驻留 ~6.6GiB。prism/tet 两段分开切块是因为两者用不同的
    # 坍缩坐标微分矩阵（见 FROperators.D_3d_tet/D_3d_prism 文档），且单元存储
    # 本来就是 prism 在前 tet 在后，块不会跨类型。附带收益：原全场 `np.
    # ascontiguousarray(grad_vel.reshape(-1,3,3))` 强制拷贝（~1.4GiB 真实新分配，
    # grad_vel 是 grad_Q 的非连续切片）降为每块一次小块拷贝；grad_T 同为梯度
    # 输出的非连续切片，按块拷贝量级相同（~21MiB/块）。Q/mu_t_field 是连续数组，
    # 块切片的 ascontiguousarray 是 no-op view。Q/grad_vel/grad_T/adj_j/det_jacs
    # 在下方界面项 kernel 里还要用，不删。
    n_prism = mesh.n_prism_cells
    _VISC_CHUNK_CELLS = 32768
    div_comp = np.zeros((n_cells, n_sps, 5))
    # native 四面体（路径C）：体积项散度同样必须改用 D_native_tet_padded，
    # 理由与 gradients.py::compute_physical_gradient 同一处文档——D_3d_tet
    # 是坍缩坐标专属微分矩阵，对 native 单纯形基节点没有意义。
    _tet_op_D = ops.D_native_tet_padded if getattr(ops, "D_native_tet_padded", None) is not None else ops.D_3d_tet
    # 去混叠（AFCFD_VISC_OVERINT=on）：粘性通量是 tau/u·tau/k_cond*grad_T
    # 这些乘积再乘 adj(J)，直接在 coarse SPs 上微分等价于"先混叠再求导"。
    # 理由、实测背景与保留的上游局限见 `resolve_viscous_overintegration`。
    # 默认 off，行为逐位不变。
    # IP 罚项常数按阶数解析一次（trace 不等式常数 ~ (p+1)(p+3)/3，
    # 见 `flux_kernels.resolve_viscous_ip_constant`）。
    _c_ip = resolve_viscous_ip_constant(int(mesh.order))
    _oi = (get_overintegration_context(mesh, ops)
           if resolve_viscous_overintegration() == "on" else None)
    if _oi is not None:
        div_comp = _viscous_volume_overintegrated(
            Q, grad_vel, grad_T, mu_t_field, mu, Pr, Pr_t, _oi, n_sps)
        residual = div_comp / det_jacs[..., None]
        del div_comp
    else:
      for seg_lo, seg_hi, op_D in (
        (0, n_prism, ops.D_3d_prism),
        (n_prism, n_cells, _tet_op_D),
      ):
        for c0 in range(seg_lo, seg_hi, _VISC_CHUNK_CELLS):
            c1 = min(c0 + _VISC_CHUNK_CELLS, seg_hi)
            G_phys = viscous_physical_flux_batch(
                np.ascontiguousarray(Q[c0:c1].reshape(-1, 5)),
                np.ascontiguousarray(grad_vel[c0:c1].reshape(-1, 3, 3)),
                np.ascontiguousarray(grad_T[c0:c1].reshape(-1, 3)),
                mu, Pr,
                np.ascontiguousarray(mu_t_field[c0:c1].reshape(-1)),
                Pr_t,
            ).reshape(c1 - c0, n_sps, 3, 5)
            # 度量×通量融合 kernel（性能优化 2026-09-13，见 volume_contract.py
            # ::contravariant_flux_from_metric 文档：原 `np.matmul(adj_j, G_phys)`
            # 是逐点 3x3@3x5 批量微型 gemm，调用开销主导且不随核数并行）。
            # 数学上逐位等价（adj_j 本身就是 det_jacs*inv_jacs，这里直接从
            # 两者融合算出同一个乘积）；`adj_j` 仍保留物化，界面项 kernel 要用。
            G_tilde = contravariant_flux_from_metric(
                det_jacs[c0:c1], inv_jacs[c0:c1], G_phys
            )
            del G_phys  # 块内用完即弃，下一轮迭代变量重新绑定
            div_comp[c0:c1] = contract_shared_operator_2axis(op_D, G_tilde)
            del G_tilde
      # 注意：粘性项是 +div(G)（见模块文档的符号约定）
      residual = div_comp / det_jacs[..., None]
      del div_comp  # ~854MiB，用完即弃

    # --- 界面项：numba 逐点标量 kernel（性能优化，替代原纯 Python
    # `for f in range(fc.n_faces)` 逐面循环，理由/验证方式与
    # fr_residual_inviscid.py 的同类改动完全一致，见
    # fr_viscous_flux_kernel.py 模块文档、
    # tests/unit/test_fr_viscous_flux_kernel_crosscheck.py 的新旧实现
    # 逐位对比验证）。
    from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
    from autoflowcfd.core.fr_residual.inviscid_kernel import compute_boundary_ghost_states

    # #2（2026-08-28）：见 core/fr_residual/inviscid.py::
    # compute_inviscid_residual_fr 同名参数文档——CPU 分布式路径必须传入
    # 预先按 local+halo 压缩索引空间构造好的 flat_face_override，不能让
    # 这里对 DistributedMeshAdapter 重新调用 get_flat_face_geometry。
    flat = flat_face_override if flat_face_override is not None else get_flat_face_geometry(mesh, ops)
    Q_ghost = compute_boundary_ghost_states(flat, Q, adj_j, ghost_provider)
    # 边界温度梯度按热边界类型分派（WALL/SYMMETRY 法向镜像 ⇒ 离散壁面热通量
    # 精确为零；INLET/OUTLET/FARFIELD 保持透射），见
    # boundary/fr_ghost_state.py::build_boundary_adiabatic_mask。
    from autoflowcfd.boundary.fr_ghost_state import build_boundary_adiabatic_mask
    bnd_adiabatic = build_boundary_adiabatic_mask(flat.n_faces, flat.is_boundary, ghost_provider)

    if n_sps == 1:
        # P0 专用路径：使用 n_sps=1 特化 kernel（消除 SP 循环，外插写成
        # 标量乘——n_sps=1 下与矩阵乘是同一个运算，不是近似，见
        # viscous_p0_kernel.py 模块文档的用词更正）
        # 性能优化：Order Continuation P0 阶段 n_sps=1，通用 kernel 的
        # for s in range(n_sps) 循环虽只有 1 次迭代但仍有分支/索引开销，
        # P0 专用 kernel 在编译期消除所有 SP 循环。
        from autoflowcfd.core.fr_residual.viscous_p0_kernel import (
            compute_viscous_interface_correction_p0_kernel,
        )
        import numba
        n_threads = numba.get_num_threads()
        correction = compute_viscous_interface_correction_p0_kernel(
            Q, grad_vel, grad_T, mu_t_field,
            det_jacs, mu, Pr, Pr_t,
            flat.owner_cell, flat.neighbor_cell, flat.is_boundary,
            flat.owner_is_primary, flat.neighbor_is_primary,
            flat.owner_adj_row_exact, flat.neighbor_adj_row_exact,
            flat.neighbor_src0_cell, flat.neighbor_src0_mat,
            flat.neighbor_src1_idx, flat.neighbor_src1_cell, flat.neighbor_src1_mat,
            flat.owner_src0_cell, flat.owner_src0_mat,
            flat.owner_src1_idx, flat.owner_src1_cell, flat.owner_src1_mat,
            flat.mixed_nb_partner, flat.mixed_nb_mask,
            flat.mixed_ow_partner, flat.mixed_ow_mask,
            Q_ghost, bnd_adiabatic,
            n_threads,
            flat.owner_cube_face, flat.neighbor_cube_face,
            flat.ref_area_weight,
            flat.boundary_extrap_native, flat.lift_native,
            flat.face_area, flat.cell_volume,
            _c_ip,
        )
    else:
        # P≥1 通用路径：图着色或 per-thread buffer
        from autoflowcfd.core.fr_residual.viscous_flux_kernel import (
            compute_viscous_interface_correction_kernel,
        )
        from autoflowcfd.core.fr_residual.viscous_flux_kernel_colored import (
            compute_viscous_interface_correction_kernel_colored,
        )
        # 图着色方案：同色面无 owner_cell 冲突，直接写入共享 buffer
        # 内存从 O(n_threads * n_cells * n_sps * 5) 降至 O(n_cells * n_sps * 5)
        # 着色结果已缓存在 flat 中（build 时一次性计算，不再重复着色）
        # 通过环境变量或配置可切换回 per-thread buffer 方案
        use_coloring = os.environ.get("AFCFD_USE_COLORING", "1") == "1"
        
        if use_coloring:
            correction = np.zeros((n_cells, n_sps, 5))
            for c in range(flat.n_colors):
                face_indices = flat.color_face_indices[c]
                if len(face_indices) == 0:
                    continue
                compute_viscous_interface_correction_kernel_colored(
                    Q, grad_vel, grad_T, mu_t_field,
                    det_jacs, mu, Pr, Pr_t,
                    flat.owner_cell, flat.neighbor_cell, flat.is_boundary,
                    flat.owner_is_primary, flat.neighbor_is_primary,
                    flat.owner_adj_row_exact, flat.neighbor_adj_row_exact,
                    flat.neighbor_src0_cell, flat.neighbor_src0_mat,
                    flat.neighbor_src1_idx, flat.neighbor_src1_cell, flat.neighbor_src1_mat,
                    flat.owner_src0_cell, flat.owner_src0_mat,
                    flat.owner_src1_idx, flat.owner_src1_cell, flat.owner_src1_mat,
                    flat.mixed_nb_partner, flat.mixed_nb_mask,
                    flat.mixed_ow_partner, flat.mixed_ow_mask,
                    Q_ghost, bnd_adiabatic,
                    face_indices, correction,
                    flat.owner_cube_face, flat.neighbor_cube_face,
                    flat.ref_area_weight,
                    flat.boundary_extrap_native, flat.lift_native,
                    flat.face_area, flat.cell_volume,
                    _c_ip,
                )
        else:
            # 回退到 per-thread buffer 方案（小网格 + 低线程数可能更快）
            import numba
            n_threads = numba.get_num_threads()
            correction = compute_viscous_interface_correction_kernel(
                Q, grad_vel, grad_T, mu_t_field,
                det_jacs, mu, Pr, Pr_t,
                flat.owner_cell, flat.neighbor_cell, flat.is_boundary,
                flat.owner_is_primary, flat.neighbor_is_primary,
                flat.owner_adj_row_exact, flat.neighbor_adj_row_exact,
                flat.neighbor_src0_cell, flat.neighbor_src0_mat,
                flat.neighbor_src1_idx, flat.neighbor_src1_cell, flat.neighbor_src1_mat,
                flat.owner_src0_cell, flat.owner_src0_mat,
                flat.owner_src1_idx, flat.owner_src1_cell, flat.owner_src1_mat,
                flat.mixed_nb_partner, flat.mixed_nb_mask,
                flat.mixed_ow_partner, flat.mixed_ow_mask,
                Q_ghost, bnd_adiabatic,
                n_threads,
                flat.owner_cube_face, flat.neighbor_cube_face,
                flat.ref_area_weight,
                flat.boundary_extrap_native, flat.lift_native,
                flat.face_area, flat.cell_volume,
                _c_ip,
            )
    residual = residual + correction

    # 机制3 已删除，依据见 `fr_residual/inviscid.py` 同一处的完整记录
    # （真实网格消融对照：触发 15 次但残差轨迹只差 ~1e-10、结局不变）。
    return residual
