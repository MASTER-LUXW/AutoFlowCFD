"""AutoFlowCFD V2.0 - 粘性体积项与残差入口(GPU)

从 `src/autoflowcfd/core/gpu/residual/gpu_viscous.py`(原 723 行)拆出(2026-09-24, 项目"单文件不超 500 行"规范)。**纯搬家, 逻辑未改**。
"""

import numpy as np


from autoflowcfd.core.gpu import get_cupy

from autoflowcfd.core.gpu.residual.gpu_volume_contract import (
    gpu_contract_shared_operator_1axis,
    gpu_contract_shared_operator_2axis,
)

from autoflowcfd.core.gpu.residual.gpu_flux import viscous_physical_flux_gpu, conserved_to_primitive_gpu

from autoflowcfd.core.gpu.residual.gpu_gradients import compute_physical_gradient_gpu


from autoflowcfd.core.fr_operators.flux_kernels import (
    R_AIR,
    resolve_viscous_ip_constant,
)
from .interface import _compute_viscous_interface_correction_gpu


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
        resolve_viscous_ip_constant(int(mesh.order)),
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
    # 机制3 已于 2026-09-19 删除（完整依据见 `fr_residual/inviscid.py`
    # 同一处）。上面那段"CPU 侧触发、GPU 侧未触发所以两条路径分道扬镳"
    # 的记录因此成为历史：两侧现在都不做这一步，对称性由"都没有"保证。
    # 同时省掉一次 `GPU -> CPU -> GPU` 往返。
    return (cp.asnumpy(viscous_residual) if input_is_numpy
            else viscous_residual)
