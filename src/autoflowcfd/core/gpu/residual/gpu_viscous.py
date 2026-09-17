"""
AutoFlowCFD V2.0 - GPU 版粘性残差计算

与 core/fr_viscous_flux.py 对应的 CuPy 版本。
包含：
- 粘性物理通量（应力张量 + 热传导 + Boussinesq 假设）
- BR1 界面耦合（界面原始变量取平均，梯度取平均+镜像，边界面 Interior
  Penalty 罚项）——2026-08-23 之前这里完全没有实现，本模块自己的文档
  却一直写着"包含"，`compute_viscous_residual_fr_gpu` 实际上只算了
  体积散度项；`GPUFRSolver`（gpu_solver.py）不是死代码，`solve steady
  --backend gpu` 真实会走到这里，任何跟壁面/边界层相关的 GPU 粘性流动
  通量都缺一整项。修复对照 CPU 端 viscous_flux_kernel.py 的
  `compute_viscous_interface_correction_kernel` 逐字移植数学公式，
  按图着色分色 + owner_is_primary/neighbor_is_primary 分组去重（与
  gpu_inviscid.py::_compute_interface_correction_gpu 不同——那里没有
  这层过滤，本文件新增代码保留过滤是为了不重蹈这次 resume 调查里
  P0 端棱柱四边形侧面重复计数的同一类 bug，即便当前 FlatFaceGeometry
  的图着色分组是否真的需要它尚未确认，这层过滤本身零代价）。
- 体积项散度（张量收缩 + 度量项）

公式与 CPU 版完全一致，见 core/fr_viscous_flux.py 和
core/fr_residual/viscous_flux_kernel.py 模块文档。
"""

import numpy as np
from typing import Optional
from loguru import logger

from autoflowcfd.core.gpu import get_cupy
from autoflowcfd.core.gpu.residual.gpu_volume_contract import (
    gpu_contract_shared_operator_1axis,
    gpu_contract_shared_operator_2axis,
)
from autoflowcfd.core.gpu.residual.gpu_flux import viscous_physical_flux_gpu, conserved_to_primitive_gpu
from autoflowcfd.core.gpu.residual.gpu_gradients import compute_physical_gradient_gpu
from autoflowcfd.core.gpu.residual.gpu_inviscid import _native_or_collapsed_contrib

GAMMA = 1.4
R_AIR = 287.0
_VISCOUS_BOUNDARY_IP_C = 4.0


def compute_temperature_gpu(Q):
    """GPU 版温度计算。T = p/(rho*R)。"""
    cp = get_cupy()
    rho = cp.maximum(Q[..., 0], 1e-10)
    return Q[..., 4] / (rho * R_AIR)


def _viscous_volume_overintegrated_gpu(cp, Q, grad_vel, grad_T, mu_t_field,
                                       mu, Pr, Pr_t, segs,
                                       n_cells, n_sps):
    """粘性体积项 `div(adj(J)*G(Q,grad_vel,grad_T,mu_t))` 的去混叠版（GPU），
    返回 (n_cells, n_sps, 5)。

    与 CPU 端 `core/fr_residual/viscous_flux.py::
    _viscous_volume_overintegrated` 逐项对应：

      ① Q/grad_vel/grad_T/mu_t 各自**精确插值**到 FINE 点（各自次数 <= order，
         所以插值本身无误差）；
      ② 在 FINE 点**重新求值** `viscous_physical_flux_gpu`——非线性函数本身
         在细点求值，而不是把 coarse 上算好的乘积插过去。这正是去混叠的
         全部内容；
      ③ 用细点度量（`segs` 里每段自带的 `adj_seg`，= det_fine*inv_fine
         已预乘）算逆变通量；
      ④ 用 FINE 网格自己的微分矩阵求散度；
      ⑤ 精确插值限制回 coarse SPs。

    为什么粘性项需要这个（与 CPU 端 `resolve_viscous_overintegration` 同一
    条理由）：粘性通量是 `tau ~ mu*grad_u`、`u·tau`、`k_cond*grad_T` 这些
    **乘积**，再乘 adj(J)，直接在 coarse SPs 上微分等价于"先混叠再求导"。
    只有"乘积被微分"的地方过积分才有意义。

    上游局限与 CPU 端完全相同、这里不重复：`grad_vel`/`grad_T` 本身是在
    coarse SPs 上用 coarse 微分矩阵算出的（见
    `core/fr_residual/gradients.py`），把它们插到细点只是**精确重构同一个
    多项式**，不会凭空恢复梯度算子自身的截断内容。
    """
    div_comp = cp.zeros((n_cells, n_sps, 5), dtype=cp.float64)
    mut_is_array = mu_t_field is not None and hasattr(mu_t_field, 'shape')
    # 每段自带自己的 n_fine 与**已切好**的细点度量（2026-09-17，与 CPU 端
    # 同一次改动）：四面体过积分的细网格轴不再填充到棱柱宽度，两段的
    # n_fine 不同，所以不能再用一份共享的 adj_j_fine 按全局 [lo:hi] 切。
    # 本循环一次处理整段，`adj_seg` 正好就对应 [lo:hi]，直接用即可。
    for lo, hi, _n_fine_seg, adj_seg, c2f, D_fine, f2c in segs:
        if hi <= lo:
            continue
        Q_f = gpu_contract_shared_operator_1axis(c2f, Q[lo:hi])
        n_fine = Q_f.shape[1]
        nb = hi - lo
        gv_f = gpu_contract_shared_operator_1axis(
            c2f, grad_vel[lo:hi].reshape(nb, n_sps, 9)).reshape(nb, n_fine, 3, 3)
        gT_f = gpu_contract_shared_operator_1axis(c2f, grad_T[lo:hi])
        if mut_is_array:
            mut_f = gpu_contract_shared_operator_1axis(
                c2f, mu_t_field[lo:hi][..., None])[..., 0]
        else:
            mut_f = 0.0 if mu_t_field is None else mu_t_field
        G_phys_f = viscous_physical_flux_gpu(
            Q_f, gv_f, gT_f, mu, Pr, mu_t=mut_f, Pr_t=Pr_t,
        )  # (nb, n_fine, 3, 5)
        del Q_f, gv_f, gT_f
        G_tilde_f = cp.matmul(adj_seg, G_phys_f)
        del G_phys_f
        div_f = gpu_contract_shared_operator_2axis(D_fine, G_tilde_f)
        del G_tilde_f
        div_comp[lo:hi] = gpu_contract_shared_operator_1axis(f2c, div_f)
        del div_f
    return div_comp


def compute_viscous_residual_fr_gpu(
    U,
    mesh,
    ops,
    mu=1.8e-5,
    Pr=0.72,
    mu_t_field=None,
    Pr_t=0.9,
    boundary_ghost_provider=None,
    mesh_data=None,
    ops_data=None,
    flat_face_gpu=None,
    flat_face_cpu=None,
    device_id=0,
):
    """GPU 版粘性残差计算。

    与 core/fr_viscous_flux.py::compute_viscous_residual_fr 公式一致。

    Args:
        U: CuPy 数组 (n_cells, n_sps, n_vars)
        mesh: HighOrderMesh
        ops: FROperators
        mu: 分子动力粘度
        Pr: 分子普朗特数
        mu_t_field: CuPy 数组 (n_cells, n_sps)，湍流涡粘度（可选）
        Pr_t: 湍流普朗特数
        boundary_ghost_provider: 边界幽灵态提供者
        mesh_data: 预上传的网格数据
        ops_data: 预上传的算子数据
        flat_face_gpu: 预构建的 GPU 面几何（可选，None 时自动构建，见
            gpu_inviscid.py::compute_inviscid_residual_fr_gpu 同名参数）
        device_id: GPU 设备 ID

    Returns:
        viscous_residual: CuPy 数组 (n_cells, n_sps, 5)
    """
    cp = get_cupy()
    input_is_numpy = isinstance(U, np.ndarray)
    if input_is_numpy:
        with cp.cuda.Device(device_id):
            U = cp.asarray(U)
            # mu_t_field 一并转换（2026-08-25 代码审查）：湍流场景下调用方可能
            # 同时传 numpy 的 U 和 mu_t_field，只转 U 会让 numpy 数组直接进入
            # 下面的 cupy 运算（物理通量/界面校正）触发混合运算错误。
            if mu_t_field is not None and isinstance(mu_t_field, np.ndarray):
                mu_t_field = cp.asarray(mu_t_field)

    # #1（2026-08-28）：见 gpu_inviscid.py::compute_inviscid_residual_fr_gpu
    # 同名注释——分布式多 GPU 路径下 n_cells/n_prism 必须从显式传入的
    # （已按 local+halo 压缩索引空间构造好的）mesh_data 读取，不能用
    # `mesh`（完整全局网格，供下面 get_flat_face_geometry 取真实
    # face_connectivity 用）的全局尺寸。
    if mesh_data is not None and 'n_cells' in mesh_data:
        n_cells = mesh_data['n_cells']
        n_prism = mesh_data.get('n_prism', mesh.n_prism_cells)
    else:
        n_cells = mesh.n_cells
        n_prism = mesh.n_prism_cells
    n_sps = mesh.n_sps_per_cell

    # 准备网格数据
    if mesh_data is None:
        from autoflowcfd.core.gpu.residual.gpu_inviscid_volume import prepare_mesh_data, prepare_ops_data
        mesh_data = prepare_mesh_data(cp, mesh, device_id)
        ops_data = prepare_ops_data(cp, ops, device_id)

    det_jacs = mesh_data['det_jacs']

    # 1. 计算物理梯度
    #
    # 真实 bug 修复（第三个独立发现，2026-09-03，用真实非均匀密度场
    # 交叉验证时发现——常密度/小密度扰动流场差异很小，容易被当成噪声
    # 忽略）：此前这里对**守恒变量** `U`（rho, rho*u, rho*v, rho*w,
    # rho*E）求梯度，再直接把 `grad_U[...,1:4,:]` 当"速度梯度"用——
    # 但 `grad(rho*u) = rho*grad(u) + u*grad(rho)`，不是 `grad(u)`，
    # 只有密度处处均匀（`grad(rho)=0`）时两者才恰好相等，这正是本项目
    # 大量"均匀自由流场"/"小密度扰动"测试从未捕捉到这个 bug 的原因。
    # CPU 版 `viscous_flux.py::compute_viscous_residual_fr` 一直是对
    # **原始变量** `Q`（`grad_Q=compute_physical_gradient(Q,...)`，
    # `grad_vel=grad_Q[:,:,1:4,:]`，`Q[1:4]` 本来就是 u/v/w 本身）求梯度
    # ——改为与 CPU 一致：对 `Q` 求梯度，不对 `U` 求梯度。
    Q = conserved_to_primitive_gpu(U[..., :5])
    grad_Q = compute_physical_gradient_gpu(Q, mesh_data, ops_data)

    # 速度梯度和温度梯度
    grad_vel = grad_Q[..., 1:4, :]  # (n_cells, n_sps, 3, 3)
    grad_T_scalar = compute_temperature_gpu(Q)
    grad_T = compute_physical_gradient_gpu(
        grad_T_scalar[..., None], mesh_data, ops_data
    )  # (n_cells, n_sps, 1, 3) → squeeze
    grad_T = grad_T[..., 0, :]  # (n_cells, n_sps, 3)

    # 2. 体积项：粘性物理通量 + 散度
    # mu_t_field 是调用方按 CPU 版约定（core/fr_residual/viscous_flux.py）
    # 传入的动力涡粘度 mu_t = rho * nu_t，与分子粘度 mu 量纲一致。
    # 真实 bug 修复（2026-08-23）：这里此前把 mu_t_field 先并入 mu_eff=
    # mu+mu_t_field 再传给 viscous_physical_flux_gpu(mu=mu_eff, mu_t=0.0)
    # ——该函数内部热传导系数按 `k = mu*Cp/Pr + mu_t*Cp/Pr_t` 分子/湍流
    # 分别取普朗特数（见 gpu_flux.py::viscous_physical_flux_gpu 文档），
    # mu_t 被错误地传成 0.0 意味着热传导整体（含湍流部分）都用了层流
    # Pr=0.72 而不是湍流 Pr_t=0.9，应力张量本身（只依赖 mu+mu_t 之和）
    # 不受影响，但湍流热通量系统性偏大（Pr/Pr_t≈0.8 倍）。CPU 端
    # viscous_physical_flux_point（flux_kernels.py）一直是 mu、mu_t 分开
    # 传参，这里改成一致：不再构造 mu_eff，直接把 mu（标量分子粘度）和
    # mu_t_field（湍流涡粘度数组，层流为 0）分别传入。
    mu_t_arg = 0.0 if mu_t_field is None else mu_t_field

    # 去混叠（`AFCFD_VISC_OVERINT=on`）：2026-09-15 补齐——此前 CPU 端
    # `viscous_flux.py::_viscous_volume_overintegrated` 已实现这条分支，
    # GPU 端完全没有，于是同一个环境变量在两个后端意味着**不同的数值
    # 方案**，CPU-GPU 交叉校验会在开关打开后无声地对不上（与 2026-09-15
    # 同一轮审计在 k/omega 输运上发现的 GPU 替身缺口是同一类问题）。
    # 默认 off，关闭时下面的 coarse 路径逐位不变。
    from autoflowcfd.core.fr_residual.viscous_flux import (
        resolve_viscous_overintegration,
    )
    from autoflowcfd.core.gpu.gpu_overintegration import (
        get_overintegration_segs_gpu,
    )
    _oi_segs = (get_overintegration_segs_gpu(mesh_data, ops_data, n_cells, n_prism)
                if resolve_viscous_overintegration() == "on" else None)
    if _oi_segs is not None:
        div_G = _viscous_volume_overintegrated_gpu(
            cp, Q, grad_vel, grad_T, mu_t_arg, mu, Pr, Pr_t,
            _oi_segs, n_cells, n_sps)
        # `adj_j` 仍需物化：下方界面项 kernel 要用。
        adj_j = mesh_data['adj_j']
    else:
      G_phys = viscous_physical_flux_gpu(
          Q, grad_vel, grad_T, mu, Pr, mu_t=mu_t_arg, Pr_t=Pr_t,
      )  # (n_cells, n_sps, 3, 5)
      # 逆变通量
      adj_j = mesh_data['adj_j']
      G_tilde = cp.matmul(adj_j, G_phys)  # (n_cells, n_sps, 3, 5)

      # 散度
      div_G = cp.zeros((n_cells, n_sps, 5), dtype=cp.float64)
      if n_prism > 0:
        div_G[:n_prism] = gpu_contract_shared_operator_2axis(
            ops_data['D_3d_prism'], G_tilde[:n_prism]
        )
      if n_cells > n_prism:
        # 四面体坍缩坐标基已删除（2026-09-03，见 fr/operators.py 模块
        # 文档）：`ops_data['D_3d_tet']` 现在恒别名到 `D_native_tet_
        # padded`，不再需要按 tet_basis_mode 分派。
        div_G[n_prism:] = gpu_contract_shared_operator_2axis(
            ops_data['D_3d_tet'], G_tilde[n_prism:]
        )

    # 粘性残差体积项 = +div(G) / det(J)（注意：粘性项是正号，与无粘的负号相反）
    viscous_residual = div_G / det_jacs[..., None]

    # 3. 界面项（BR1 平均 + 边界 Interior Penalty 罚项）
    # #1（2026-08-28）：见 gpu_inviscid.py::compute_inviscid_residual_fr_gpu
    # 同名注释——分布式路径必须用 dist_flat_face.base_flat（已按
    # local+halo 压缩索引重映射、只含本 rank 面）而不是全局 flat_face，
    # 否则边界幽灵态计算会用全局单元编号误当压缩索引读 Q。
    if flat_face_cpu is not None:
        flat_face = flat_face_cpu
    else:
        from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
        flat_face = get_flat_face_geometry(mesh, ops)
    if flat_face_gpu is None:
        from autoflowcfd.core.gpu.gpu_face_geometry import build_gpu_flat_face
        flat_face_gpu = build_gpu_flat_face(flat_face, device_id)

    # 边界幽灵态：ghost_provider 是任意 Python 可调用对象，numba/CuPy 都
    # 调不了，只能在 CPU 上跑（复用已验证的 P1+ CPU 实现，只循环边界面，
    # 约占全部面的 3%，不是新逻辑）——与 gpu_inviscid.py::
    # _compute_boundary_ghost_states_gpu 现在用的是同一个 CPU round-trip
    # 模式（此前那边有个真实 bug：完全没调用 ghost_provider，用"owner
    # cell 零梯度外插"代替，V2.0 专家组盲审发现并已修复，两处现已一致）。
    from autoflowcfd.core.fr_residual.inviscid import DefaultGhostProvider
    from autoflowcfd.core.fr_residual.inviscid_kernel import compute_boundary_ghost_states
    ghost_provider = (
        boundary_ghost_provider if boundary_ghost_provider is not None else DefaultGhostProvider()
    )
    Q_cpu = cp.asnumpy(Q)
    Q_ghost_np = compute_boundary_ghost_states(flat_face, Q_cpu, None, ghost_provider)
    # 边界温度梯度按热边界类型分派（WALL/SYMMETRY 法向镜像 ⇒ 离散壁面热
    # 通量精确为零；INLET/OUTLET/FARFIELD 透射），与 CPU
    # viscous_flux_kernel.py 模块文档"边界温度梯度"一节逐字对应。
    from autoflowcfd.boundary.fr_ghost_state import build_boundary_adiabatic_mask
    bnd_adiabatic_np = build_boundary_adiabatic_mask(
        flat_face.n_faces, flat_face.is_boundary, ghost_provider)
    with cp.cuda.Device(device_id):
        Q_ghost_gpu = cp.asarray(Q_ghost_np)
        bnd_adiabatic_gpu = cp.asarray(bnd_adiabatic_np)

    if mu_t_field is None:
        mu_t_gpu = cp.zeros((n_cells, n_sps, 1), dtype=cp.float64)
    else:
        mu_t_gpu = mu_t_field[..., None]

    interface_correction = _compute_viscous_interface_correction_gpu(
        Q, grad_vel, grad_T, mu_t_gpu,
        det_jacs, mu, Pr, Pr_t,
        flat_face_gpu, Q_ghost_gpu, bnd_adiabatic_gpu,
        n_cells, n_sps, n_prism, device_id,
    )

    viscous_residual = viscous_residual + interface_correction

    # 真实 bug 修复（2026-09-03，排查"native tet P2 粘性残差与 CPU 不符"
    # 时决定性定位——用真实合成网格逐项对照 CPU 参考实现，发现 P1 完全
    # 一致但 P2 有真实数值分歧，volume/interface 两项单独拿出来对照都
    # 逐位相同，只有把两者相加、比较"最终返回值"才复现分歧，说明分歧
    # 发生在"相加之后"这一步——CPU 版 `viscous_flux.py::compute_
    # viscous_residual_fr` 的最后一步是 `return suppress_residual_
    # outliers(residual, U[...,:5])`（机制3：按 cell/SP/变量粒度检测
    # 残差量级异常并清零，见 troubled_cell.py 模块文档），但本函数此前
    # 直接 `return viscous_residual`——完全没有这一步。P1 阶段残差量级
    # 小，从未越过异常判据的阈值，两者恰好"看起来"一致；P2 阶段（这次
    # 合成测试用的随机扰动流场）残差量级更大，CPU 侧机制3 真实触发、
    # 清零了部分 SP，GPU 侧未清零，二者从这一步起分道扬镳——不是
    # native/collapsed 专属缺陷，是任意网格上 GPU 粘性残差路径一直
    # 缺失的一个功能点（`gpu_inviscid.py::compute_inviscid_residual_fr_
    # gpu` 早就在做同一件事，这里之前被漏掉了，两个函数本该对称）。
    from autoflowcfd.core.fr_operators.troubled_cell import suppress_residual_outliers
    residual_np = cp.asnumpy(viscous_residual)
    U_np = cp.asnumpy(U)
    result = suppress_residual_outliers(residual_np, U_np[..., :5])
    return result if input_is_numpy else cp.asarray(result)


def _extrap_to_fp(cp, mat, src_cell, field):
    """把按 src0/src1 机制索引到的 cell 场外插到面 FP 网格。

    mat: (nF, n_fp, n_sps)，src_cell: (nF,)，
    field: (n_cells, n_sps, *rest) -> 返回 (nF, n_fp, *rest)。
    与 gpu_inviscid.py::_compute_interface_correction_gpu 里
    `cp.matmul(owner_src0_mat, Q_owner)` 同一机制的通用化版本（那里
    只对 Q 这一个 5 分量场手写了一次，这里额外要对 grad_vel(3,3)/
    grad_T(3)/mu_t(1) 复用，做成通用 helper 避免抄 4 遍同样的 reshape）。
    """
    sub = field[src_cell]
    shape = sub.shape
    flat = sub.reshape(shape[0], shape[1], -1)
    out = cp.matmul(mat, flat)
    return out.reshape(mat.shape[0], mat.shape[1], *shape[2:])


def _add_src1_to_fp(cp, out, src1_idx, src1_cell, src1_mat, field):
    """叠加稀疏第二来源（分裂面场景），对应 CPU kernel 里的 `idx1 >= 0`
    分支——大多数面 idx1 全为 -1，这里用布尔掩码只处理真正需要的那一小
    撮面，不对整批面做无意义的零矩阵乘法。"""
    has1 = src1_idx >= 0
    if not bool(cp.any(has1)):
        return out
    sel = cp.where(has1)[0]
    idx1 = src1_idx[sel]
    c1 = src1_cell[idx1]
    m1 = src1_mat[idx1]
    sub = field[c1]
    shape = sub.shape
    flat = sub.reshape(shape[0], shape[1], -1)
    add = cp.matmul(m1, flat).reshape(sel.shape[0], m1.shape[1], *shape[2:])
    out[sel] = out[sel] + add
    return out


def _extrap_side(cp, idx, src0_cell_all, src0_mat_all, src1_idx_all, src1_cell_all, src1_mat_all,
                  Q_gpu, grad_vel_gpu, grad_T_gpu, mu_t_gpu):
    """按 src0(+src1) 机制把某一侧（owner 或 neighbor）的 Q/grad_vel/
    grad_T/mu_t 外插到面 idx 对应的 FP 网格。"""
    E_cell = src0_cell_all[idx]
    E_mat = src0_mat_all[idx]
    s1_idx = src1_idx_all[idx]

    Q_fp = _extrap_to_fp(cp, E_mat, E_cell, Q_gpu)
    Q_fp = _add_src1_to_fp(cp, Q_fp, s1_idx, src1_cell_all, src1_mat_all, Q_gpu)
    gv_fp = _extrap_to_fp(cp, E_mat, E_cell, grad_vel_gpu)
    gv_fp = _add_src1_to_fp(cp, gv_fp, s1_idx, src1_cell_all, src1_mat_all, grad_vel_gpu)
    gT_fp = _extrap_to_fp(cp, E_mat, E_cell, grad_T_gpu)
    gT_fp = _add_src1_to_fp(cp, gT_fp, s1_idx, src1_cell_all, src1_mat_all, grad_T_gpu)
    mut_fp = _extrap_to_fp(cp, E_mat, E_cell, mu_t_gpu)
    mut_fp = _add_src1_to_fp(cp, mut_fp, s1_idx, src1_cell_all, src1_mat_all, mu_t_gpu)
    return Q_fp, gv_fp, gT_fp, mut_fp[..., 0]


def _self_extrap_side(cp, cell_idx, cube_face_code, axis, side, n_prism,
                       boundary_extrap, boundary_extrap_native,
                       Q_gpu, grad_vel_gpu, grad_T_gpu, mu_t_gpu,
                       compact_cell_type=None):
    """自身面外插——某一侧单元用自己的场值按自身面几何外插到 FP
    （owner-primary 块的 `Q_o`/`gv_o`/`gT_o`/`mut_o`，或 neighbor-primary
    块的 `Q_n_native`/`gv_n_native`/`gT_n_native`/`mut_n_native`）。

    与 CPU 版 `viscous_flux_kernel.py` 的 `E_o = boundary_extrap[
    celltype_o, oax, oside_idx]` / `Q_o = _extrap_matmul(Q[oc], E_o)`
    逐字对应，native 分派复用与 `gpu_inviscid.py::_compute_interface_
    correction_gpu` 完全同一个 `_native_self_extrap` helper。

    真实 bug 修复（问题清单 #5 排查附带发现，2026-09-02，均匀自由流场
    残差本应精确为零，实测非零且量级与真实物理量相当才发现）：此前
    `_compute_viscous_interface_correction_gpu` 对"自身"状态错误地
    复用了 `_extrap_side`（src0/src1 跨单元交叉引用机制），owner-primary
    块传的是 `owner_src0_cell`/`owner_src0_mat`，neighbor-primary 块传的
    是 `neighbor_src0_cell`/`neighbor_src0_mat`——但这两组数组的真实
    语义是"对侧记录用来查询*另一侧*单元值的交叉引用表"（`owner_src0_*`
    是给 neighbor-primary 块查 owner 值用的，`neighbor_src0_*` 是给
    owner-primary 块查 neighbor 值用的，见 `gpu_inviscid.py::_compute_
    interface_correction_gpu` 里 `Q_n=_extrap_q_to_fp(...,ff.neighbor_
    src0_mat[idx_o],...)`/`Q_o_at_n=_extrap_q_to_fp(...,ff.owner_src0_
    mat[idx_n],...)` 的对称用法），不是"本单元查自己"——对 owner-primary
    面自身而言 `owner_src0_mat[idx_o]` 恒为零/未设置（那一行本来就不是
    为这个查询设计的），导致 `Q_o`/`gv_o`/`gT_o`/`mut_o` 恒为零而不是
    真实自身状态：BR1 平均态 `Q_avg=0.5*(Q_o+Q_n)` 与边界 IP 罚项
    `pen=-scale*(Q_o[1:4]-Q_n[1:4])` 全部从错误的"自身值"算起，均匀
    自由流场下 `Q_o=0 != Q_n=真实自由流值`，IP 罚项产出物理量级的
    虚假非零残差（实测 ~0.43，CPU 参考给出 ~2e-12）。

    Args:
        cell_idx: (n,) 本侧单元全局索引（owner 块传 `oc`，neighbor 块
            传 `nc`）
        cube_face_code: (n,) 本侧 owner_cube_face/neighbor_cube_face
        axis, side: (n,) 本侧 owner_axis/owner_side 或
            neighbor_axis/neighbor_side
        n_prism: 分布式 compact 索引空间棱柱数（`compact_cell_type`
            不存在时的单机路径回退阈值判据）
        boundary_extrap: (2,3,2,n_fp,n_sps) collapsed 自身外插表
        boundary_extrap_native: native 四面体自身外插表（可能是空数组）
        compact_cell_type: 分布式路径下逐单元类型查表（None 时用
            `cell_idx < n_prism` 阈值判据，与 gpu_inviscid.py 同一约定）

    Returns:
        (Q, gv, gT, mut)：形状分别为 (n,n_fp,5)/(n,n_fp,3,3)/(n,n_fp,3)/
        (n,n_fp)
    """
    from autoflowcfd.core.gpu.residual.gpu_inviscid import _native_self_extrap

    if compact_cell_type is not None:
        celltype = compact_cell_type[cell_idx]
    else:
        celltype = cp.where(cell_idx < n_prism, 0, 1)
    side_idx = cp.where(side < 0, 0, 1)
    is_native = cube_face_code >= 6

    # 真实 bug 修复（2026-09-03，见 gpu_inviscid.py::_compute_interface_
    # correction_gpu 同名注释——同一处、同一根因）：`axis` 对 native 面
    # 存的是复用的 excluded_vertex（0~3），`boundary_extrap` 的轴维度
    # 只有 3，不能无条件拿 native 面的 `axis` 去 gather。
    axis_safe = cp.where(is_native, 0, axis)
    E_collapsed = boundary_extrap[celltype, axis_safe, side_idx]  # (n,n_fp,n_sps)
    E = _native_self_extrap(cp, is_native, cube_face_code, boundary_extrap_native, E_collapsed)

    Q = _extrap_to_fp(cp, E, cell_idx, Q_gpu)
    gv = _extrap_to_fp(cp, E, cell_idx, grad_vel_gpu)
    gT = _extrap_to_fp(cp, E, cell_idx, grad_T_gpu)
    mut = _extrap_to_fp(cp, E, cell_idx, mu_t_gpu)[..., 0]
    return Q, gv, gT, mut


def _viscous_tilde_flux_pair(Q_common, gv_common, gT_common, mut_common,
                              Q_own, gv_own, gT_own, mut_own,
                              adjrow, mu, Pr, Pr_t):
    """算 (G_tilde_common, G_tilde_own) 一对张量，两者之差就是 jump。
    对应 CPU 端每个 FP 上 `viscous_physical_flux_point` 调用两次
    （一次给 BR1 平均态，一次给自身原始态）再各自投影到 tilde 方向的
    那一段——这里把它向量化到 (n_faces*n_fp,) 展平批量。"""
    cp = get_cupy()
    n, n_fp = Q_common.shape[0], Q_common.shape[1]

    G_common = viscous_physical_flux_gpu(
        Q_common.reshape(n * n_fp, 5), gv_common.reshape(n * n_fp, 3, 3),
        gT_common.reshape(n * n_fp, 3), mu, Pr, mu_t=mut_common.reshape(n * n_fp), Pr_t=Pr_t,
    ).reshape(n, n_fp, 3, 5)
    G_own = viscous_physical_flux_gpu(
        Q_own.reshape(n * n_fp, 5), gv_own.reshape(n * n_fp, 3, 3),
        gT_own.reshape(n * n_fp, 3), mu, Pr, mu_t=mut_own.reshape(n * n_fp), Pr_t=Pr_t,
    ).reshape(n, n_fp, 3, 5)

    a0 = adjrow[..., 0:1]
    a1 = adjrow[..., 1:2]
    a2 = adjrow[..., 2:3]
    G_tilde_common = a0 * G_common[..., 0, :] + a1 * G_common[..., 1, :] + a2 * G_common[..., 2, :]
    G_tilde_own = a0 * G_own[..., 0, :] + a1 * G_own[..., 1, :] + a2 * G_own[..., 2, :]
    return G_tilde_common, G_tilde_own


def _compute_viscous_interface_correction_gpu(
    Q_gpu, grad_vel_gpu, grad_T_gpu, mu_t_gpu,
    det_jacs, mu, Pr, Pr_t,
    flat_face_gpu, Q_ghost_gpu, bnd_adiabatic_gpu,
    n_cells, n_sps, n_prism, device_id,
):
    """GPU 粘性界面校正（BR1 平均 + 边界 IP 罚项），按图着色逐色处理。

    与 core/fr_residual/viscous_flux_kernel.py::
    compute_viscous_interface_correction_kernel 逐字对应的数学移植，
    架构（分色、src0/src1 批量外插、scatter-add）仿照
    gpu_inviscid.py::_compute_interface_correction_gpu 已验证过的模式。

    与无粘 GPU 界面校正的两处刻意不同（都是有意的正确性选择，不是
    疏漏）：
    1. 保留 owner_is_primary/neighbor_is_primary 过滤——CPU 端
       viscous_flux_kernel.py 对此有明确、不可协商的约定（模块文档：
       "owner_is_primary/neighbor_is_primary 分组去重"），本函数照做；
       无粘 GPU 版本没有这层过滤，是否因此有对应的重复计数问题不在
       本次修复范围内，未去验证/修改那个文件。
    2. tilde 投影用 owner_adj_row_exact/neighbor_adj_row_exact（未归一化
       原始 adj(J) 行），不是无粘版本用的单位 true_normal——粘性 kernel
       没有 true_normal 对齐安全阀，这是自洽方向*唯一*的输入，理由见
       viscous_flux_kernel.py 模块文档。

    owner-侧和 neighbor-侧分开两个独立代码块（不能合并成无粘版本那种
    "算一次通量分配两侧"的写法）：两侧的 G_tilde_own 分别用各自侧的
    adjrow 和各自的"自身原始态"算出来，是两个不同的物理量，不是同一个
    共享数值通量的两次分配。

    真实 bug 修复（问题清单 #5 排查附带发现，2026-09-02）："自身原始态"
    （`Q_o`/`gv_o`/`gT_o`/`mut_o`，`Q_n_native`/`gv_n_native`/
    `gT_n_native`/`mut_n_native`）现在用 `_self_extrap_side`（自身面
    boundary_extrap/native 查表外插，与 CPU 版一致）计算，不再错误地
    复用 `_extrap_side`（src0/src1 跨单元交叉引用机制，专门给"对侧"
    数据用）——完整推导见 `_self_extrap_side` 文档。新增 `n_prism`
    形参就是给这个自身外插用的（`compact_cell_type` 不存在时的分类
    阈值，与 `gpu_inviscid.py` 同一约定）。
    """
    cp = get_cupy()
    correction = cp.zeros((n_cells, n_sps, 5), dtype=cp.float64)

    from autoflowcfd.core.gpu.residual.gpu_inviscid import _scatter_add_to_correction
    from autoflowcfd.core.gpu.residual.gpu_inviscid_volume import distribute_face_correction_to_sps

    ff = flat_face_gpu

    for c in range(ff.n_colors):
        face_idx = ff.color_face_indices[c]
        if face_idx.shape[0] == 0:
            continue

        is_bnd = ff.is_boundary[face_idx]
        owner_primary = ff.owner_is_primary[face_idx]
        neighbor_primary = ff.neighbor_is_primary[face_idx]

        # ── owner-primary 贡献块 ──
        mask_o = owner_primary
        if bool(cp.any(mask_o)):
            idx_o = face_idx[mask_o]
            oc = ff.owner_cell[idx_o]
            oax = ff.owner_axis[idx_o]
            oside = ff.owner_side[idx_o]
            is_bnd_o = ff.is_boundary[idx_o]

            # 自身原始态：与 CPU 版 `E_o=boundary_extrap[...]`/native
            # 分派完全对应，不能用 `_extrap_side`（那是"对侧交叉引用"
            # 机制，见 `_self_extrap_side` 文档"真实 bug 修复"一节）。
            compact_cell_type = getattr(ff, 'compact_cell_type', None)
            oc_code_o = ff.owner_cube_face[idx_o]
            # is_native_o/oax_safe 提到这里（本块唯一算一次）：分配步骤
            # （下方 distribute_face_correction_to_sps 调用）与
            # `_native_or_collapsed_contrib` 都需要用到，见两处各自的
            # "真实 bug 修复"说明。
            is_native_o = oc_code_o >= 6
            oax_safe = cp.where(is_native_o, 0, oax)
            Q_o, gv_o, gT_o, mut_o = _self_extrap_side(
                cp, oc, oc_code_o, oax, oside, n_prism,
                ff.boundary_extrap, ff.boundary_extrap_native,
                Q_gpu, grad_vel_gpu, grad_T_gpu, mu_t_gpu,
                compact_cell_type=compact_cell_type,
            )
            Q_n, gv_n, gT_n, mut_n = _extrap_side(
                cp, idx_o, ff.neighbor_src0_cell, ff.neighbor_src0_mat,
                ff.neighbor_src1_idx, ff.neighbor_src1_cell, ff.neighbor_src1_mat,
                Q_gpu, grad_vel_gpu, grad_T_gpu, mu_t_gpu,
            )
            adjrow_o = ff.owner_adj_row_exact[idx_o]

            # 边界面：状态用幽灵态，速度梯度/涡粘镜像内部值本身（不能改成
            # 用 sources/幽灵态梯度，见 viscous_flux_kernel.py 模块文档
            # "边界面梯度处理"一节）；温度梯度按热边界类型分派（同文档
            # "边界温度梯度"一节）。
            bmask3 = is_bnd_o[:, None, None]
            Q_ghost_sub = Q_ghost_gpu[idx_o]
            Q_n = cp.where(bmask3, Q_ghost_sub, Q_n)
            gv_n = cp.where(is_bnd_o[:, None, None, None], gv_o, gv_n)
            # ∇T 法向分量镜像：gT - 2*((gT·a)/|a|^2)*a（|a|=0 的退化面原样返回，
            # 那种面的通量投影本来就是零）。用逆变行 adjrow_o 而非
            # true_normal，理由见 flux_kernels.py::mirror_normal_component。
            a2_o = cp.sum(adjrow_o * adjrow_o, axis=-1, keepdims=True)
            d_o = cp.sum(gT_o * adjrow_o, axis=-1, keepdims=True) / cp.where(a2_o > 0.0, a2_o, 1.0)
            gT_mirror_o = cp.where(a2_o > 0.0, gT_o - 2.0 * d_o * adjrow_o, gT_o)
            adiab_o = bnd_adiabatic_gpu[idx_o][:, None, None]
            gT_n = cp.where(bmask3, cp.where(adiab_o, gT_mirror_o, gT_o), gT_n)
            mut_n = cp.where(is_bnd_o[:, None], mut_o, mut_n)

            # 混合拆分面（B-8，镜像 CPU viscous_flux_kernel.py 同名分支）：边界半区用配对面幽灵态，
            # 梯度镜像内部值——与真边界面同规则，逐 FP 生效。
            mp_o = ff.mixed_nb_partner[idx_o]
            mixed_sel_o = (mp_o[:, None] >= 0) & ff.mixed_nb_mask[idx_o]  # (nO, n_fp)
            mixed3_o = mixed_sel_o[..., None]
            # Q_ghost_gpu 形状 (n_faces, n_fp, 5)，与 Q_n 形状一致，逐 FP 直接替换。
            Q_ghost_partner_o = Q_ghost_gpu[cp.maximum(mp_o, 0)]  # (nO, n_fp, 5)
            Q_n = cp.where(mixed3_o, Q_ghost_partner_o, Q_n)
            gv_n = cp.where(mixed_sel_o[:, :, None, None], gv_o, gv_n)
            adiab_mp_o = bnd_adiabatic_gpu[cp.maximum(mp_o, 0)][:, None, None]
            gT_n = cp.where(mixed3_o, cp.where(adiab_mp_o, gT_mirror_o, gT_o), gT_n)
            mut_n = cp.where(mixed_sel_o, mut_o, mut_n)
            # 逐 FP 的"边界半区"标记（真边界面全 FP 生效 + 混合面仅掩码 FP 生效），下方 IP 罚项共用。
            is_bnd_i_o = is_bnd_o[:, None] | mixed_sel_o

            Q_avg = 0.5 * (Q_o + Q_n)
            gv_avg = 0.5 * (gv_o + gv_n)
            gT_avg = 0.5 * (gT_o + gT_n)
            mut_avg = 0.5 * (mut_o + mut_n)

            G_tilde_common, G_tilde_own = _viscous_tilde_flux_pair(
                Q_avg, gv_avg, gT_avg, mut_avg, Q_o, gv_o, gT_o, mut_o,
                adjrow_o, mu, Pr, Pr_t,
            )
            jump_owner = G_tilde_common - G_tilde_own

            if bool(cp.any(is_bnd_i_o)):
                a0 = adjrow_o[..., 0]
                a1 = adjrow_o[..., 1]
                a2 = adjrow_o[..., 2]
                adj_mag_o = cp.sqrt(a0 * a0 + a1 * a1 + a2 * a2)  # (nO,n_fp)
                vol_o = cp.mean(det_jacs[oc], axis=-1)  # (nO,)
                h = cp.maximum(vol_o ** (1.0 / 3.0), 1e-300)
                eta = _VISCOUS_BOUNDARY_IP_C * (mu + mut_o) / h[:, None]
                scale = eta * adj_mag_o * oside[:, None]
                pen = -scale[..., None] * (Q_o[..., 1:4] - Q_n[..., 1:4])
                pen_full = cp.zeros_like(jump_owner)
                pen_full[..., 1:4] = pen
                jump_owner = cp.where(is_bnd_i_o[..., None], jump_owner + pen_full, jump_owner)

            # 真实 bug 修复（V2.0 专家组盲审第四轮，2026-08-28）：分配
            # 改用 dist_fp_of_sp/dist_axis_coord_of_sp gather，不再用
            # `ff.g_left[idx_o]` 按面索引去索引这个长度仅 n1d 的数组
            # （会在真实 GPU 上 IndexError），见
            # gpu_inviscid_volume.py::distribute_face_correction_to_sps
            # 文档的完整推导（gpu_inviscid.py 同一处修复）。
            # 真实 bug 修复（2026-09-03）：用 `oax_safe`（native 面 clip
            # 到 0），不能用原始 `oax`——见 gpu_inviscid.py 同名注释、
            # 同一根因（`dist_fp_of_sp`/`dist_axis_coord_of_sp` 同样
            # 只有 3 个轴）。
            contrib_o_collapsed = distribute_face_correction_to_sps(
                cp, jump_owner, oax_safe, oside, ff.dist_fp_of_sp, ff.dist_axis_coord_of_sp,
                ff.g_left, ff.g_right,
            )
            # native 四面体（路径C）GPU 移植（2026-09-02）：分配步骤改用
            # DG 提升算子 `lift_native[excluded_vertex] @ (true_area_weight
            # ⊙ jump)`，与 CPU 版 viscous_flux_kernel.py 逐字对应——粘性
            # kernel 不需要像无粘那样处理 side_factor/true_normal 安全阀
            # （viscous_flux_kernel.py 模块文档："viscous kernels 只需要
            # owner_adj_row_exact/neighbor_adj_row_exact 就足够做线性
            # 收缩，不像无粘那样需要额外的方向安全阀"），本函数前面的
            # tilde 通量计算全程未受影响，native/collapsed 唯一的分派点
            # 就是这里的面校正分配方式。
            # oc_code_o/is_native_o 已在本块开头（自身外插处）算过，直接复用。
            contrib_o = _native_or_collapsed_contrib(
                cp, is_native_o, oc_code_o, ff.lift_native, ff.true_area_weight[idx_o],
                jump_owner, contrib_o_collapsed,
            )
            contrib_o = contrib_o / det_jacs[oc][..., None]
            _scatter_add_to_correction(correction, contrib_o, oc, n_cells, n_sps)

        # ── neighbor-primary 贡献块（仅内部面）──
        mask_n = neighbor_primary & (~is_bnd)
        if bool(cp.any(mask_n)):
            idx_n = face_idx[mask_n]
            nc = ff.neighbor_cell[idx_n]
            nax = ff.neighbor_axis[idx_n]
            nside = ff.neighbor_side[idx_n]

            # 自身原始态：同上方 owner-primary 块同名注释，同一处修复
            # （`_extrap_side`+`neighbor_src0_*` 是"对侧交叉引用"机制，
            # 不是"自身外插"）。
            compact_cell_type_n = getattr(ff, 'compact_cell_type', None)
            nc_code_n = ff.neighbor_cube_face[idx_n]
            is_native_n = nc_code_n >= 6
            nax_safe = cp.where(is_native_n, 0, nax)
            Q_n_native, gv_n_native, gT_n_native, mut_n_native = _self_extrap_side(
                cp, nc, nc_code_n, nax, nside, n_prism,
                ff.boundary_extrap, ff.boundary_extrap_native,
                Q_gpu, grad_vel_gpu, grad_T_gpu, mu_t_gpu,
                compact_cell_type=compact_cell_type_n,
            )
            Q_o_at_n, gv_o_at_n, gT_o_at_n, mut_o_at_n = _extrap_side(
                cp, idx_n, ff.owner_src0_cell, ff.owner_src0_mat,
                ff.owner_src1_idx, ff.owner_src1_cell, ff.owner_src1_mat,
                Q_gpu, grad_vel_gpu, grad_T_gpu, mu_t_gpu,
            )
            adjrow_n = ff.neighbor_adj_row_exact[idx_n]

            # 混合拆分面（B-8）：neighbor 侧对称处理——边界半区对侧状态取配对面幽灵态，梯度镜像。
            mp_n = ff.mixed_ow_partner[idx_n]
            mixed_sel_n = (mp_n[:, None] >= 0) & ff.mixed_ow_mask[idx_n]  # (nN, n_fp)
            mixed3_n = mixed_sel_n[..., None]
            Q_ghost_partner_n = Q_ghost_gpu[cp.maximum(mp_n, 0)]  # (nN, n_fp, 5)
            Q_o_at_n = cp.where(mixed3_n, Q_ghost_partner_n, Q_o_at_n)
            gv_o_at_n = cp.where(mixed_sel_n[:, :, None, None], gv_n_native, gv_o_at_n)
            a2_n = cp.sum(adjrow_n * adjrow_n, axis=-1, keepdims=True)
            d_n = cp.sum(gT_n_native * adjrow_n, axis=-1, keepdims=True) / cp.where(a2_n > 0.0, a2_n, 1.0)
            gT_mirror_n = cp.where(a2_n > 0.0, gT_n_native - 2.0 * d_n * adjrow_n, gT_n_native)
            adiab_mp_n = bnd_adiabatic_gpu[cp.maximum(mp_n, 0)][:, None, None]
            gT_o_at_n = cp.where(mixed3_n, cp.where(adiab_mp_n, gT_mirror_n, gT_n_native), gT_o_at_n)
            mut_o_at_n = cp.where(mixed_sel_n, mut_n_native, mut_o_at_n)

            Q_avg_n = 0.5 * (Q_n_native + Q_o_at_n)
            gv_avg_n = 0.5 * (gv_n_native + gv_o_at_n)
            gT_avg_n = 0.5 * (gT_n_native + gT_o_at_n)
            mut_avg_n = 0.5 * (mut_n_native + mut_o_at_n)

            G_tilde_common_n, G_tilde_own_n = _viscous_tilde_flux_pair(
                Q_avg_n, gv_avg_n, gT_avg_n, mut_avg_n,
                Q_n_native, gv_n_native, gT_n_native, mut_n_native,
                adjrow_n, mu, Pr, Pr_t,
            )
            jump_neighbor = G_tilde_common_n - G_tilde_own_n

            # 混合拆分面（B-8）：neighbor 侧边界 IP 罚项，镜像 CPU kernel 同名分支（罚项的“内部态”
            # 是本单元外插值、“对侧”是幽灵态）。
            if bool(cp.any(mixed_sel_n)):
                a0n = adjrow_n[..., 0]
                a1n = adjrow_n[..., 1]
                a2n = adjrow_n[..., 2]
                adj_mag_n = cp.sqrt(a0n * a0n + a1n * a1n + a2n * a2n)
                vol_n = cp.mean(det_jacs[nc], axis=-1)
                h_n = cp.maximum(vol_n ** (1.0 / 3.0), 1e-300)
                eta_n = _VISCOUS_BOUNDARY_IP_C * (mu + mut_n_native) / h_n[:, None]
                scale_n = eta_n * adj_mag_n * nside[:, None]
                pen_n = -scale_n[..., None] * (Q_n_native[..., 1:4] - Q_o_at_n[..., 1:4])
                pen_full_n = cp.zeros_like(jump_neighbor)
                pen_full_n[..., 1:4] = pen_n
                jump_neighbor = cp.where(mixed_sel_n[..., None], jump_neighbor + pen_full_n, jump_neighbor)

            # 见上方 owner-primary 块同名注释，同一处修复——用
            # `nax_safe`，不能用原始 `nax`。
            contrib_n_collapsed = distribute_face_correction_to_sps(
                cp, jump_neighbor, nax_safe, nside, ff.dist_fp_of_sp, ff.dist_axis_coord_of_sp,
                ff.g_left, ff.g_right,
            )
            # native 四面体（路径C）GPU 移植（2026-09-02）：见上方
            # owner-primary 块同名注释，同一处修复。nc_code_n/is_native_n
            # 已在本块开头（自身外插处）算过，直接复用。
            contrib_n = _native_or_collapsed_contrib(
                cp, is_native_n, nc_code_n, ff.lift_native, ff.true_area_weight[idx_n],
                jump_neighbor, contrib_n_collapsed,
            )
            contrib_n = contrib_n / det_jacs[nc][..., None]
            _scatter_add_to_correction(correction, contrib_n, nc, n_cells, n_sps)

    return correction
