"""AutoFlowCFD V2.0 - omega 壁面目标值/松弛/Dirichlet 掩码(GPU)

从 `src/autoflowcfd/core/gpu/turbulence/gpu_scalar_transport.py`(原 743 行)拆出(2026-09-24, 项目"单文件不超 500 行"规范)。**纯搬家, 逻辑未改**。
"""


def compute_omega_wall_target_gpu(cp, ff, wall_mask, wall_distance_gpu, Q_gpu, mu, beta1,
                                 omega_max=1e6, turb_k_field=None):
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
    被壁面处趋零的对流通量掩盖，没暴露这个缺口；GPU 版
    `enforce_omega_wall_relaxation_gpu`（本文件同日新增，直接把这个值
    混合进 omega_field 本身）不再有这层天然保护，必须在这里、唯一的
    真值来源处夹到 `omega_max`。

    Args:
        wall_mask: (n_faces,) CuPy bool，WALL 边界面掩码（由
            `compute_wall_dirichlet_masks_gpu` 一次性算出并缓存）
        omega_max: 湍流模型的应急安全上限（`solver.turb_model_gpu.omega_max`，
            调用方未提供时退回 1e6，与 CPU 版同一个默认值）

    Returns:
        (omega_wall_value_face, has_value_face)，与 CPU 版返回语义一致
    """
    n_faces = ff.n_faces
    n_fp = ff.boundary_extrap_native.shape[-2]
    omega_wall_value_face = cp.zeros((n_faces, n_fp), dtype=cp.float64)
    wall_idx = cp.where(wall_mask)[0]
    if wall_idx.shape[0] > 0:
        owner_cells = ff.owner_cell[wall_idx]
        # 长度尺度口径可切换，与 CPU 端 core/turbulence/transport.py::
        # _compute_omega_wall_target 同一处 2026-09-15 发现逐字对应
        # （`min` 既不是单元中心也不是单元高度，会让这个经标定的壁面
        # 函数产生随阶数变化的系统性高估：order=1/2/3 分别 5.60x /
        # 19.68x / 51.86x）。**默认仍为 `min`**，理由见 CPU 端注释。
        import os as _os
        _d1_mode = _os.environ.get("AFCFD_OMEGA_WALL_D1", "min").lower()
        if _d1_mode not in ("min", "mean"):
            raise ValueError(
                f"AFCFD_OMEGA_WALL_D1={_d1_mode!r} 不是合法取值（min | mean）")
        _wd = wall_distance_gpu[owner_cells]
        d1 = _wd.min(axis=1) if _d1_mode == "min" else _wd.mean(axis=1)
        d1 = cp.maximum(d1, 1e-8)
        # 只统计真实自由度，与 CPU 端 transport.py::
        # _compute_omega_wall_target 同一处 2026-09-15 审计逐字对应。
        # 阶数从 SP 数反解（棱柱是张量积 n_sps=(order+1)^3），棱柱数从
        # flat face 几何取——都不需要改本函数签名。
        # 用与 CPU 端**同一个**共享辅助（2026-09-15 统一）：原先这里是
        # 一份手写的等价逻辑，等价性只能靠人工比对维护；共享版还附带
        # "order 与 n_sps 必须自洽"的显式校验。
        from autoflowcfd.fr.native_padding import (
            order_from_n_sps, reduce_rows_over_real_sps,
        )
        _np_prism = getattr(ff, 'n_prism', None)
        _rows = Q_gpu[owner_cells, :, 0]
        _order = order_from_n_sps(_rows.shape[1])
        if _np_prism is None:
            # 拿不到棱柱数就不能判断行的单元类型；此时退回全场平均并
            # 保持既有行为（flat face 几何一定带 n_prism，这条分支只在
            # 测试替身缺字段时才会走到）。
            rho_owner = cp.mean(_rows, axis=1)
        else:
            rho_owner = reduce_rows_over_real_sps(
                _rows, owner_cells < _np_prism, _order, 'mean', xp=cp)
        nu_owner = mu / cp.maximum(rho_owner, 1e-10)
        # omega 壁面公式族可切换（2026-09-15，B-6），与 CPU 端
        # transport.py::_omega_wall_formula 同一套判据与常数（文献依据见
        # 那里的 _OMEGA_WALL_MODES 一节）。默认 amplified，逐位不变。
        from autoflowcfd.core.turbulence.transport import (
            _OMEGA_WALL_CMU, _OMEGA_WALL_KAPPA, resolve_omega_wall_mode,
        )
        omega_vis = 6.0 * nu_owner / (beta1 * d1 ** 2)
        if resolve_omega_wall_mode() == "amplified":
            omega_wall = 10.0 * omega_vis
        else:
            if turb_k_field is None:
                raise RuntimeError(
                    "AFCFD_OMEGA_WALL_MODE=blended 需要 turb_k_field（对数支 "
                    "omega_log = sqrt(k)/(Cmu^0.25*kappa*d1) 用到它）。调用方"
                    "必须显式传入 k 场，不接受静默退回 amplified 档。")
            k_rows = turb_k_field[owner_cells]
            if _np_prism is None:
                k_owner = cp.mean(k_rows, axis=1)
            else:
                k_owner = reduce_rows_over_real_sps(
                    k_rows, owner_cells < _np_prism, _order, 'mean', xp=cp)
            omega_log = (cp.sqrt(cp.maximum(k_owner, 0.0))
                         / (_OMEGA_WALL_CMU ** 0.25 * _OMEGA_WALL_KAPPA * d1))
            omega_wall = cp.sqrt(omega_vis ** 2 + omega_log ** 2)
        omega_wall = cp.minimum(omega_wall, omega_max)
        omega_wall_value_face[wall_idx, :] = omega_wall[:, None]
    return omega_wall_value_face, wall_mask


def enforce_omega_wall_relaxation_gpu(cp, solver, relax=None):
    """CuPy 版 `transport.py::enforce_omega_wall_relaxation`——GPU SST/
    DDES/IDDES 输运路径此前完全没有移植这个修复（2026-09-05 代码复审
    发现，不是本次新引入的差异）：`compute_omega_wall_target_gpu` 算出
    的 Wilcox 解析值此前只喂给对流项上风 ghost（`compute_scalar_
    convection_residual_gpu` 的 `wall_dirichlet_value_face` 参数），
    扩散项同样没有把这个约束传递进去——与 CPU 版被修复前完全同一个
    架构缺口（见 CPU 版 `enforce_omega_wall_relaxation` 文档的完整
    推导：边界层单元 omega 长期不受约束衰减到下界，经 nu_t 近零分母
    奇点放大湍流粘性比，持续向平均流注入过量粘性应力）。GPU SST/DDES/
    IDDES 长期运行（真实生产场景，例如 cube_demo 这类真实网格）会
    出现与 CPU 版修复前完全相同的中长期发散机制。

    实现逐字对应 CPU 版（同样的固定 relax=0.5 事后松弛，不是 CPU 版
    2026-09-05 那次被真实数据证伪撤销的"点隐式"动态松弛——不要重复
    那次已经证伪的尝试，见 CPU 版文档完整失败记录）：`update_fields_gpu`
    之后，对 WALL 面 owner 单元的 `omega_field` 做一次向解析壁面目标值
    的固定比例松弛。`np.add.at`（CPU 版处理"同一 owner 单元是多个 WALL
    面的 owner（角部单元）"）在这里换成 GPU 原生的 `cp.scatter_add`
    （本模块模块文档已说明：正确处理重复索引累加，不需要图着色）。

    Args:
        solver: `GPUFRSolver` 实例
        relax: 松弛系数，默认 0.5（与 CPU 版同一个经验证的安全值）
    """
    if relax is None:
        relax = 0.5
    wall_mask = getattr(solver, "_wall_mask_k_gpu", None)
    if wall_mask is None or not cp.any(wall_mask):
        return

    Q = solver.Q_gpu
    turb = solver.turb_model_gpu
    ff = solver.flat_face_gpu
    omega_max = getattr(turb, "omega_max", 1e6)
    omega_wall_value_face, has_wall = compute_omega_wall_target_gpu(
        cp, ff, wall_mask, solver.wall_distance_gpu, Q, solver.mu_molecular,
        getattr(turb, "beta1", 0.075), omega_max=omega_max,
        # blended 档的对数支需要 k（见 compute_omega_wall_target_gpu）
        turb_k_field=getattr(turb, "k_field", None),
    )

    wall_face_idx = cp.where(has_wall)[0]
    if wall_face_idx.shape[0] == 0:
        return
    owner_cells = ff.owner_cell[wall_face_idx]
    target = omega_wall_value_face[wall_face_idx, 0]  # 同一面上恒为同一常数，见函数文档

    n_cells = solver.mesh.n_cells
    # 同一个 owner 单元可能是多个 WALL 面的 owner（角部单元）——用
    # scatter_add 累加再除以命中次数取平均目标值，不能直接花式索引赋值
    # 覆盖（后写的面会覆盖先写的面，不是真正的平均），与 CPU 版
    # `np.add.at` 同一个理由。
    sum_target = cp.zeros(n_cells, dtype=cp.float64)
    count = cp.zeros(n_cells, dtype=cp.float64)
    cp.scatter_add(sum_target, owner_cells, target)
    cp.scatter_add(count, owner_cells, 1.0)
    hit_cells = cp.where(count > 0)[0]
    avg_target = sum_target[hit_cells] / count[hit_cells]

    turb.omega_field[hit_cells, :] = (
        (1.0 - relax) * turb.omega_field[hit_cells, :] + relax * avg_target[:, None]
    )


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
