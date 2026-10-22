"""AutoFlowCFD V2.0 - 粘性体积项与残差入口(GPU)"""

import numpy as np

from autoflowcfd.core.fr_residual.viscous_flux.constants import PRANDTL_TURBULENT
from autoflowcfd.core.gpu import get_cupy

from autoflowcfd.core.gpu.residual.gpu_volume_contract import gpu_contract_shared_operator_1axis

from autoflowcfd.core.gpu.residual.gpu_flux import viscous_physical_flux_gpu, conserved_to_primitive_gpu

from autoflowcfd.core.gpu.residual.gpu_gradients import compute_physical_gradient_gpu

from autoflowcfd.core.thermo import R_AIR
from autoflowcfd.core.fr_operators.flux_kernels import resolve_viscous_ip_constant
from .interface import _compute_viscous_interface_correction_gpu


def compute_temperature_gpu(Q):
    """GPU 版温度计算。T = p/(rho*R)。"""
    cp = get_cupy()
    rho = cp.maximum(Q[..., 0], 1e-10)
    return Q[..., 4] / (rho * R_AIR)


def _viscous_volume_term_gpu(cp, Q, grad_vel, grad_T, mu_t_field, mu, Pr, Pr_t, segs, lifted_div,
                             n_cells, n_sps):
    """粘性体积项 `K G~`（GPU，参考空间，调用方再除以 det J），返回 (n_cells, n_sps, 5)。

    与 CPU 端 `core/fr_residual/viscous_flux/overintegration.py::viscous_volume_term`
    逐项对应（为什么必须过积分、体积算子 K 的含义都见那里）：Q / grad_vel / grad_T / mu_t
    精确插值到细点，在细点上重新求值粘性通量，乘细点度量（`segs` 每段自带的
    `adj_seg = det*inv`），与 K 一次收缩。
    """
    div_comp = cp.zeros((n_cells, n_sps, 5), dtype=cp.float64)
    mut_is_array = mu_t_field is not None and hasattr(mu_t_field, 'shape')
    from autoflowcfd.core.gpu.gpu_overintegration import contract_lifted_divergence_gpu

    for (lo, hi, _n_fine_seg, adj_seg, c2f, _D_fine, _f2c), (K_all, combo) in zip(segs, lifted_div):
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
        div_comp[lo:hi] = contract_lifted_divergence_gpu(cp, K_all, combo, G_tilde_f)
        del G_tilde_f
    return div_comp


def compute_viscous_residual_fr_gpu(
    U,
    mesh,
    ops,
    mu,
    Pr,
    mu_t_field=None,
    Pr_t=PRANDTL_TURBULENT,
    boundary_ghost_provider=None,
    mesh_data=None,
    ops_data=None,
    flat_face_gpu=None,
    flat_face_cpu=None,
    device_id=0,
):
    """GPU 版粘性残差计算。

    与 core/fr_residual/viscous_flux/residual.py::compute_viscous_residual_fr 公式一致。

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
            gpu_inviscid/::compute_inviscid_residual_fr_gpu 同名参数）
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

    # #1（2026-08-28）：见 gpu_inviscid/::compute_inviscid_residual_fr_gpu
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
    # CPU 版 `core/fr_residual/viscous_flux/residual.py::compute_viscous_residual_fr` 一直是对
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

    if n_sps == 1:
        # P0：分片常数场的体积散度恒为零；本侧通量由界面核减去（见 interface.py）
        div_G = cp.zeros((n_cells, n_sps, 5), dtype=cp.float64)
    else:
        # 与 CPU 同一个离散（core/fr_residual/viscous_flux/overintegration.py）：细点上重新
        # 求值粘性通量，体积算子 K 同时减去修正项的本侧通量迹
        from autoflowcfd.core.gpu.gpu_overintegration import (
            get_overintegration_segs_gpu, lifted_divergence_gpu,
        )
        _oi_segs = get_overintegration_segs_gpu(mesh_data, ops_data, n_cells, n_prism)
        if _oi_segs is None:
            raise RuntimeError(
                "P>=1 GPU 粘性残差需要过积分细点度量与 overint_* 算子，这里缺失：网格或算子是按"
                "不完整的阶数几何构造/上传的")
        div_G = _viscous_volume_term_gpu(
            cp, Q, grad_vel, grad_T, mu_t_arg, mu, Pr, Pr_t,
            _oi_segs, lifted_divergence_gpu(mesh_data, ops_data), n_cells, n_sps)

    # 粘性残差体积项 = +div(G) / det(J)（注意：粘性项是正号，与无粘的负号相反）
    viscous_residual = div_G / det_jacs[..., None]

    # 3. 界面项（BR1 平均 + 边界 Interior Penalty 罚项）
    # #1（2026-08-28）：见 gpu_inviscid/::compute_inviscid_residual_fr_gpu
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

    # 边界幽灵态：设备上计算（与 CPU、GPU 无粘残差同一份 compute_boundary_ghost_states）
    from autoflowcfd.boundary.fr_ghost_state import require_ghost_provider
    from autoflowcfd.core.fr_residual.inviscid_kernel import compute_boundary_ghost_states
    ghost_provider = require_ghost_provider(boundary_ghost_provider, "compute_viscous_residual_fr_gpu")
    Q_ghost_gpu = compute_boundary_ghost_states(flat_face_gpu, Q[..., :5], ghost_provider)
    # 边界面公共粘性通量按粘性边界种类分派，与 CPU
    # core/fr_operators/flux_kernels/viscous_bc.py 同一份分类。
    from autoflowcfd.boundary.fr_ghost_state import build_viscous_boundary_kind
    vbc_kind_np = build_viscous_boundary_kind(flat_face.n_faces, flat_face.is_boundary, ghost_provider)
    with cp.cuda.Device(device_id):
        vbc_kind_gpu = cp.asarray(vbc_kind_np)

    if mu_t_field is None:
        mu_t_gpu = cp.zeros((n_cells, n_sps, 1), dtype=cp.float64)
    else:
        mu_t_gpu = mu_t_field[..., None]

    interface_correction = _compute_viscous_interface_correction_gpu(
        Q, grad_vel, grad_T, mu_t_gpu,
        det_jacs, mu, Pr, Pr_t,
        flat_face_gpu, Q_ghost_gpu, vbc_kind_gpu,
        n_cells, n_sps, n_prism, device_id,
        resolve_viscous_ip_constant(int(mesh.order)),
    )

    viscous_residual = viscous_residual + interface_correction

    # 与 CPU 版一致：不对残差做异常抑制。
    return (cp.asnumpy(viscous_residual) if input_is_numpy
            else viscous_residual)
