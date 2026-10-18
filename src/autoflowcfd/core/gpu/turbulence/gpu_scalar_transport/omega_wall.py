"""AutoFlowCFD V2.0 - omega 壁面目标值/Dirichlet 掩码(GPU)

从 `src/autoflowcfd/core/gpu/turbulence/gpu_scalar_transport.py`(原 743 行)拆出(2026-09-24, 项目"单文件不超 500 行"规范)。**纯搬家, 逻辑未改**。
"""


def compute_omega_wall_target_gpu(cp, ff, wall_mask, wall_distance_gpu, Q_gpu, mu, beta1,
                                 omega_max, turb_k_field, wall_treatment, with_k_slope: bool = False):
    """CuPy 版 `_compute_omega_wall_target`：omega_wall = 60*nu/(beta1*d1^2)
    （Wilcox 解析式），逐字对应 CPU 版同名函数——全程 GPU 原生实现（不像
    边界幽灵态那样需要 CPU round-trip：wall_distance_gpu/Q_gpu/owner_cell
    都已经常驻显存，没有必要为这个小计算专门下载/上传）。

    真实 bug 修复（2026-09-05，代码复审发现，与 CPU 版
    `transport.py::_compute_omega_wall_target` 2026-09-05 那次真实网格
    验证决定性发现的 bug 同一个根因，此前只修了 CPU 版、GPU 镜像版本
    漏了）：`d1 = cp.maximum(d1, 1e-8)` 只防止除零，不防止结果本身失控
    ——cube_demo 真实网格上确认存在 wall_distance 恰好卡在这个 1e-8
    下限的退化 WALL 面 owner 单元，代入公式算出 `omega_wall~1e14`，比
    "应急安全上限" `omega_max`（SST 模型 `k_max`/`omega_max` 属性，
    默认 1e6）大 8 个数量级。此前唯一的消费者（对流项上风 ghost）碰巧
    被壁面处趋零的对流通量掩盖，没暴露这个缺口；当时同日新增的显式步后
    松弛（直接把这个值混合进 omega_field，2026-10-01 已删除）与扩散残差的
    面 Dirichlet 都没有这层天然保护，所以在这里、唯一的真值来源处夹到
    `omega_max`。

    Args:
        wall_mask: (n_faces,) CuPy bool，WALL 边界面掩码（由
            `compute_wall_dirichlet_masks_gpu` 一次性算出并缓存）
        omega_max: omega 上界（`turb_model_gpu.omega_max`，随最近壁面解点给定）
        turb_k_field: k 场（auto 的对数支用到）
        wall_treatment: 近壁处理方式（`turb_model_gpu.wall_treatment`）
        with_k_slope: 另返回目标值对数对 owner 单元 k 均值的导数（湍流解析块 Jacobian 用）

    Returns:
        (omega_wall_value_face, has_value_face[, k_slope_face])，与 CPU 版返回语义一致
    """
    n_faces = ff.n_faces
    n_fp = ff.boundary_extrap_native.shape[-2]
    omega_wall_value_face = cp.zeros((n_faces, n_fp), dtype=cp.float64)
    wall_idx = cp.where(wall_mask)[0]
    if wall_idx.shape[0] > 0:
        owner_cells = ff.owner_cell[wall_idx]
        # d1 口径与 CPU 端共用一个函数（`sst/bounds.py::omega_wall_length_scale`）
        from autoflowcfd.core.turbulence.sst.bounds import omega_wall_length_scale

        # 只统计真实自由度（与 CPU `transport/omega_wall.py::_compute_omega_wall_target` 同一组共享辅助）：
        # 阶数从解点数反解，棱柱数取自面几何（`GPUFlatFaceGeometry.n_prism`）
        from autoflowcfd.fr.native_padding import order_from_n_sps, reduce_rows_over_real_sps

        row_is_prism = owner_cells < ff.n_prism
        d1 = omega_wall_length_scale(wall_distance_gpu[owner_cells], row_is_prism, cp, wall_treatment)
        _rows = Q_gpu[owner_cells, :, 0]
        _order = order_from_n_sps(_rows.shape[1])
        rho_owner = reduce_rows_over_real_sps(_rows, row_is_prism, _order, 'mean', xp=cp)
        nu_owner = mu / cp.maximum(rho_owner, 1e-10)
        # 公式随近壁处理方式，与 CPU 同一个函数（`turbulence/sst/bounds.py::omega_wall_value`）
        from autoflowcfd.core.turbulence.sst.bounds import omega_wall_value
        k_owner = None
        if wall_treatment == "auto":
            k_owner = reduce_rows_over_real_sps(turb_k_field[owner_cells], row_is_prism, _order, 'mean', xp=cp)
        omega_wall = omega_wall_value(cp, nu_owner, d1, k_owner, beta1, wall_treatment)
        omega_wall = cp.minimum(omega_wall, omega_max)
        omega_wall_value_face[wall_idx, :] = omega_wall[:, None]
    if not with_k_slope:
        return omega_wall_value_face, wall_mask
    # 目标值的对数对 owner 单元 k 均值的导数（湍流解析块 Jacobian 用，与 CPU 同一个函数）
    k_slope_face = cp.zeros(n_faces)
    if wall_idx.shape[0] > 0 and wall_treatment == "auto":
        from autoflowcfd.core.turbulence.sst.bounds import log_omega_wall_k_slope

        k_slope_face[wall_idx] = log_omega_wall_k_slope(cp, nu_owner, d1, k_owner, beta1, omega_max)
    return omega_wall_value_face, wall_mask, k_slope_face


def compute_turbulence_face_masks_gpu(mesh, boundary_ghost_provider):
    """k/omega 输运的两张面拓扑掩码 `(wall_mask, open_code_mask)`（numpy）。

    纯拓扑查询、与流场无关，调用方初始化时算一次并上传缓存。判据与 CPU
    **同一份实现**（`turbulence/transport/omega_wall.py` 的
    `wall_dirichlet_face_mask` / `open_boundary_code_mask`）——此前这里有
    一份逐字复制的 WALL 判据，2026-09-12 的滑移壁修复就不得不在两处各改
    一遍。`open_code_mask` 不含"真边界面"条件，由
    `compute_scalar_convection_residual_gpu` 与面邻居源推出的真边界求与。
    """
    from autoflowcfd.core.turbulence.transport import (
        open_boundary_code_mask,
        wall_dirichlet_face_mask,
    )

    n_faces = mesh.face_connectivity.n_faces
    return (wall_dirichlet_face_mask(boundary_ghost_provider, n_faces),
            open_boundary_code_mask(boundary_ghost_provider, n_faces))
