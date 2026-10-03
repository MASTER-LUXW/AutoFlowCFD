"""AutoFlowCFD V2.0 - 湍流模型（SST/DDES/IDDES 源项、LES 亚格子涡粘）用的速度梯度。

## P0：单元内梯度恒为零，必须带上界面跳变

`gradients.py::compute_physical_gradient` 是解多项式在单元内的导数（"破碎梯度"）。
P0 时解是分片常数，破碎梯度**恒为零**：SST 的 S、Omega 全场为零、产生项为零，整个
P0 阶段湍流模型是惰性的。plate_demo 实测：P0 收敛后剪切层 nu_t/nu 的 p99 恰为来流值
5.0，升到 P1 后又用 70 步才长到 87；改用下式后 P0 结束时 p90/p99 = 58.8/173（同时 P0
照样收敛到 CFL 上限 1e4）。P0 用提升修正梯度

    g = grad(phi) + sum_faces lift[(phi* - phi_side) n_side]    （P0 时第一项为零）

`phi*` 是两侧平均（BR1 公共值，与平均流粘性项同一取法）；物理边界取本侧与幽灵态的
平均——无滑移壁幽灵速度为 `2 u_wall - u`，平均即壁面速度；对称面法向分量抵消；出口
幽灵即本侧外插，修正为零。P0 时上式就是有限体积的 Green–Gauss 梯度（面值取两侧
平均），即一阶有限体积 RANS 产生项所用的梯度。

## P>=1：用单元内多项式导数，不加提升

破碎梯度在 P>=1 是一致的 O(h^p) 近似。提升项在欠分辨处把跳变（壁面弱边界条件、
远场与发展流不相容、粗网格截断）按 ~(p+1)^2/h 放大进梯度：槽道冲击启动实测，
修正 S 比破碎 S 在贴壁处大 10 倍、核心区 7.5 倍、出口段 9 倍（P3），伪生成项让 P3
隐式稳态 110 步残差从 1e2 涨到 2.4e6 发散；P1 收敛解在远场出口角点出现 k = -0.38
（破碎梯度下两者都正常收敛）。阶数判断只在本模块的两个包装函数里
（`source_velocity_gradient` / GPU `source_velocity_gradient_gpu`）。

## 提升的约定与一份算法

`dphi/dx_d` 的跳变量是 `(phi* - phi_side) n_side,d`，按扩散的符号（`sign=+1`）提升
（`gradient = div(phi I)`），与 `turbulence/transport/face_frames.py` 的提升约定同一份；
离散散度定理 `sum_cells int g = sum_boundary int phi* n dA` 随之成立。
`corrected_gradient` 只做数组运算（边界幽灵态替换、平均、跳变、调用提升），后端注入
"按坐标系外插两侧"与"两侧跳变提升"两个原语：CPU 用 `turbulence/transport/faces.py`
的 numba 核，GPU 用 `gpu/turbulence/gpu_scalar_transport/faces.py` 的 cupy 实现（两者
逐项对应、已有逐位对照测试）。

分布式紧凑空间（local + 一层 halo）上，halo 单元外侧的面不在本 rank，提升修正梯度在
halo 行上不完整，包装函数收到 `halo_refresh`（`mpi/compact_halo.py`）时取所属 rank 的值；
P>=1 的单元内梯度只依赖单元自身，不需要刷新。
"""

from autoflowcfd.core.fr_operators.gradients import compute_physical_gradient


def _with_boundary_values(xp, flat, other, ghost, frame):
    """把物理边界通量点上的"另一侧"换成幽灵态（`other`/`ghost` 为 `(n_faces, n_fp, V)`）。

    owner 坐标系：没有任何邻居来源的真边界面整面替换；混合拆分面（B-8）的边界
    半区按配对边界面的幽灵态逐通量点替换（两个坐标系都有，配对边界面的 owner 就是
    本侧单元，逐通量点顺序一致——与 `transport/faces.py::boundary_diffusion_targets`
    同一约定）。neighbor 坐标系下的真边界面没有本侧单元，两侧都是 0、不被提升。
    """
    if frame == "owner":
        true_bnd = (flat.neighbor_src0_cell < 0) & (flat.neighbor_src1_idx < 0)
        other = xp.where(true_bnd[:, None, None], ghost, other)
        partner, mask = flat.mixed_nb_partner, flat.mixed_nb_mask
    else:
        partner, mask = flat.mixed_ow_partner, flat.mixed_ow_mask
    in_mixed = ((partner >= 0)[:, None] & mask)[..., None]
    return xp.where(in_mixed, ghost[xp.maximum(partner, 0)], other)


def corrected_gradient(xp, phi, grad_broken, ghost, flat, extrapolate_pair, lift):
    """提升修正梯度，`(n_cells, n_sps, V, 3)`（定义见模块文档）。

    V 个分量一起做边界替换与平均，`V x 3` 个（分量, 方向）跳变量堆叠后一次提升。

    Args:
        xp: 数组模块（numpy / cupy）。
        phi: `(n_cells, n_sps, V)` 解点值。
        grad_broken: `(n_cells, n_sps, V, 3)` 单元内物理梯度（与 `phi` 同一份）。
        ghost: `(n_faces, n_fp, V)` 物理边界通量点上的幽灵态（只读边界行）。
        flat: 面几何（CPU `FlatFaceGeometry` / GPU 面几何，字段同名）。
        extrapolate_pair: `(phi_component, frame) -> (phi_self, phi_other)`，`frame` 为
            `"owner"`/`"neighbor"`，两侧都在该坐标系的通量点顺序里；物理边界上的
            `phi_other` 会被这里覆盖，后端原语的边界规则不影响结果。
        lift: `(jump_owner, jump_neighbor) -> (n_cells, n_sps, M)`，跳变量
            `(n_faces, n_fp, M)`，按扩散符号（+1）提升两侧各自坐标系下的跳变量并除以 det。
    """
    n_cells, n_sps, n_var = phi.shape
    jumps = []
    for frame, normal in (("owner", flat.owner_unit_normal), ("neighbor", flat.neighbor_unit_normal)):
        pairs = [extrapolate_pair(xp.ascontiguousarray(phi[..., v]), frame) for v in range(n_var)]
        phi_self = xp.stack([p[0] for p in pairs], axis=-1)                       # (n_faces, n_fp, V)
        phi_other = _with_boundary_values(xp, flat, xp.stack([p[1] for p in pairs], axis=-1), ghost, frame)
        half = 0.5 * (phi_other - phi_self)                                        # phi* - phi_self
        jumps.append((half[..., :, None] * normal[:, :, None, :]).reshape(half.shape[:2] + (n_var * 3,)))
    corr = lift(xp.ascontiguousarray(jumps[0]), xp.ascontiguousarray(jumps[1]))
    return grad_broken + corr.reshape(n_cells, n_sps, n_var, 3)


def needs_lifting(n_sps: int) -> bool:
    """只有 P0（每单元 1 个解点）用提升修正梯度（理由与实测见模块文档）。"""
    return int(n_sps) == 1


def source_velocity_gradient(Q, mesh, ops, flat, ghost_provider, halo_refresh=None):
    """CPU：湍流模型用的速度梯度 `(n_cells, n_sps, 3, 3)`，`[..., i, j] = du_i/dx_j`。

    P0 为提升修正梯度（Green–Gauss），P>=1 为单元内多项式导数（模块文档）。`Q` 为原始
    变量，前 5 列是 `(rho, u, v, w, p)`（湍流求解器的状态另有 k、omega 两列，这里不读）；
    边界幽灵态与平均流残差同一个提供者、同一个函数
    （`fr_residual/inviscid_kernel.py::compute_boundary_ghost_states`）。
    `halo_refresh`：分布式紧凑空间上刷新 halo 行（只在提升时需要）。
    """
    import numpy as np

    from autoflowcfd.core.fr_residual.inviscid import DefaultGhostProvider
    from autoflowcfd.core.fr_residual.inviscid_kernel import compute_boundary_ghost_states
    from autoflowcfd.core.turbulence.transport.faces import (
        _extrapolate_scalar_to_faces, _extrapolate_scalar_to_faces_neighbor_frame, _lift_side_jumps,
    )

    vel = np.ascontiguousarray(Q[:, :, 1:4])
    grad_broken = compute_physical_gradient(vel, mesh, ops)
    if not needs_lifting(vel.shape[1]):
        return grad_broken
    provider = ghost_provider if ghost_provider is not None else DefaultGhostProvider()
    Q_ghost = compute_boundary_ghost_states(flat, np.ascontiguousarray(Q[..., :5]), provider)

    def extrapolate_pair(comp, frame):
        if frame == "owner":
            return _extrapolate_scalar_to_faces(comp, flat, ops, mesh)
        return _extrapolate_scalar_to_faces_neighbor_frame(comp, flat)

    def lift(jump_owner, jump_neighbor):
        return _lift_side_jumps(jump_owner, jump_neighbor, +1.0, flat, mesh)

    grad = corrected_gradient(np, vel, grad_broken, np.ascontiguousarray(Q_ghost[..., 1:4]), flat,
                              extrapolate_pair, lift)
    return grad if halo_refresh is None else halo_refresh(grad)
