"""AutoFlowCFD V2.0 - 湍流标量的**扩散**残差（含去混叠体积项）。

从 `core/turbulence/transport.py` 拆出（2026-09-24）。纯搬家，逻辑未改。

残差约定：`+div(Gamma*grad(phi))/det(J)`（含 BR1 界面校正），与
`viscous_flux.py` 的"粘性项是 +div(G)"完全同一约定 —— 这个符号曾经写错成
反扩散、把 k/omega 场两极分化到正性限制器的上下界，见包 `__init__.py`
"符号约定"一节记录的那次修复。
"""

import numpy as np


from autoflowcfd.core.fr_operators.gradients import (
    compute_physical_scalar_gradient,
)
from autoflowcfd.core.fr_operators.volume_contract import (
    contract_shared_operator_1axis,
    contract_shared_operator_2axis,
    contravariant_flux_from_metric,
)
from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
from autoflowcfd.core.turbulence.transport_kernel import (
    scalar_volume_divergence_kernel,
)

from .faces import (
    _distribute_correction_to_cells,
    _extrapolate_scalar_to_faces,
)
from .convection import (
    _TURB_OVERINT_CHUNK_CELLS,
    _turb_overint_ops,
    resolve_turb_overintegration,
)


def _scalar_diffusion_volume_overintegrated(gamma_field, grad_phi, oi, n_sps):
    """扩散体积项 `div(adj(J)*Gamma*grad_phi)` 的去混叠版，返回
    (n_cells,n_sps)。

    与对流版同一条链路：Gamma 与 grad_phi 各自精确插值到 FINE 点后**在
    FINE 点相乘**，用 FINE 点度量算逆变通量、FINE 微分矩阵求散度，再
    精确限制回 coarse。

    **`Gamma` 自身的混叠：已量化、刻意不实施（2026-09-15 结论）**。
    `Gamma = mu + sigma*rho*nu_t` 里 `nu_t` 是**商**、根本不是多项式，
    它在 coarse 点上的节点值本身已经是一个投影结果——本函数只能把这份
    节点表示精确插值到 FINE 点，无法像平均流那样"在 FINE 点重新求值
    非线性通量函数"（`fr_residual/inviscid.py` 能那样做是因为它手里有 Q）。
    所以这里去掉的是 **Gamma×grad_phi 乘积以及与度量项乘积**的混叠。

    要不要把 Gamma 自身那层也去掉，做过受控测量（`Gamma = mu +
    s*rho*k/omega`，rho/k/omega 全取一次场，与解析散度比较，相对 L-inf）：

        order=1 prism: 插值 Gamma 1.0019e-02 -> 细点精确 3.2491e-05  (308x 更好)
        order=1 tet  : 插值 Gamma 9.3413e-03 -> 细点精确 1.4530e-04  ( 64x 更好)
        order=2 prism: 插值 Gamma 1.7408e-02 -> 细点精确 3.4978e-02  (0.50x **更差**)
        order=2 tet  : 插值 Gamma 1.8643e-03 -> 细点精确 1.6869e-03  (1.11x)

    **order=2 上"细点精确求值"反而更差**，不是测量噪声：在细点精确求值
    一个有理函数、再对它的 degree-over_order 插值求导，会把更多高频内容
    带进微分算子；而插值过的 Gamma 本身更平滑。也就是说这条改动**不是
    单调有益**的。

    代价侧同样不小：生产里 `sigma` 来自 SST 混合函数 `F1`，而 `F1` 依赖
    `wall_distance`——那是纯几何量，**必须重新 KD-Tree 查询、不能插值**
    （2026-09-05 真实 bug 修复：阶数切换时把 wall_distance 当解多项式场
    插值导致 d1 系统性偏大），细点是 791492x64 ≈ 5070 万个查询点；此外
    每步还要在 8 倍点数上重算 `F1`/`nu_t`/`CD_kw`。

    综合判断：在 order=2 净负、order=1 的收益又落在一个相对误差已经只有
    1e-2 的项上（对比本轮去掉的 200%~498%），不值这个代价。**这是一条
    有数据支撑的结论，不是待办项**——若将来工作阶数或精度诉求变化需要
    重新评估，上面的数字与代价分析可以直接复用。
    """
    n_cells = gamma_field.shape[0]
    # 每段自带自己的 n_fine 与已切好的细点度量（2026-09-17）：native
    # 四面体的过积分细网格轴不再填充到棱柱的 (oo+1)^3 宽度，两段的
    # n_fine 不同了。度量按**段内局部**索引切（`i0 = c0 - seg_lo`）——
    # 用全局 c0 去切段内数组会静默取到错误的单元。
    div_G = np.empty((n_cells, n_sps))
    for (seg_lo, seg_hi, n_fine, det_seg, inv_seg,
         op_c2f, op_D_fine, op_f2c) in oi["segs"]:
        for c0 in range(seg_lo, seg_hi, _TURB_OVERINT_CHUNK_CELLS):
            c1 = min(c0 + _TURB_OVERINT_CHUNK_CELLS, seg_hi)
            i0, i1 = c0 - seg_lo, c1 - seg_lo
            gam_f = contract_shared_operator_1axis(
                op_c2f, np.ascontiguousarray(gamma_field[c0:c1, :, None]))
            grad_f = contract_shared_operator_1axis(
                op_c2f, np.ascontiguousarray(grad_phi[c0:c1]))        # (n,n_fine,3)
            G_phys_f = (gam_f * grad_f)[..., None]                    # (n,n_fine,3,1)
            del gam_f, grad_f
            G_tilde_f = contravariant_flux_from_metric(
                np.ascontiguousarray(det_seg[i0:i1]),
                np.ascontiguousarray(inv_seg[i0:i1]), G_phys_f)
            del G_phys_f
            div_f = contract_shared_operator_2axis(op_D_fine, G_tilde_f)
            del G_tilde_f
            div_G[c0:c1] = contract_shared_operator_1axis(op_f2c, div_f)[..., 0]
            del div_f
    return div_G


def compute_scalar_diffusion_residual(
    scalar_field: np.ndarray,
    gamma_field: np.ndarray,
    mesh,
    ops,
    wall_dirichlet_zero_face: np.ndarray = None,
    wall_dirichlet_value_face: np.ndarray = None,
    has_wall_dirichlet_value: np.ndarray = None,
    flat_face_override=None,
) -> np.ndarray:
    """计算标量扩散 FR 残差（体积项 + BR1 界面校正），返回值为 dphi/dt。

    扩散方程: d(rho*phi)/dt = div(Gamma * grad(phi))

    符号约定（2026-08-25 修复）：本函数返回值被调用方直接作为 dphi/dt
    相加（见模块文档"符号约定"），与平均流粘性残差 viscous_flux.py 的
    "粘性项是 +div(G)"同一约定——物理扩散使峰值摊平、谷值抬升，
    dphi/dt = +div(Gamma*grad(phi))/det(J)。此前误写成 -div(...)（反扩散），
    指数放大棋盘模态导致 k/omega 场双峰触限、求解停滞（见模块文档）。
    界面校正对应地用 `residual - interface_correction`：分配 kernel 对 owner
    侧是 -=（见 transport_kernel.py::distribute_corrections_to_cells_kernel），
    因此这里减去它等于对 dphi/dt 施加 +lift(G_common - G_internal)——
    与对流 `residual + interface_correction` 的表面差异全部来自体积项符号，
    不是扩散物理要求相反符号（此前注释"扩散是反梯度通量，校正应减小残差"
    的物理表述有误，一并更正）。

    Args:
        scalar_field: (n_cells, n_sps) 标量场
        gamma_field: (n_cells, n_sps) 有效扩散系数 Gamma
        mesh: HighOrderMesh
        ops: FROperators
        wall_dirichlet_zero_face: (n_faces,) bool，可选。保留参数以兼容调用方：
            2026-08-25 校正改用梯度差形式后，WALL Dirichlet-zero 的奇镜像
            ghost（ghost=-owner，见 extrapolate_scalar_to_faces_kernel 文档）
            对梯度场是对称的（奇函数的导数是偶函数），边界面上梯度跳跃自然为
            零，不需要单独处理；此前用状态跳跃校正时这个掩码决定 phi 的 ghost
            取值，校正改梯度差后不再有数值作用，留作后续补壁面扩散通量的接口。
        wall_dirichlet_value_face, has_wall_dirichlet_value: 同理保留以兼容
            调用方（omega 壁面解析式），出于与上面 wall_dirichlet_zero_face
            完全相同的理由（本函数不再外插 scalar_field 自身，只外插
            gamma_field/grad_phi），当前同样对本函数的数值结果没有影响——
            壁面 Dirichlet 值目前只通过 `compute_scalar_convection_residual`
            的上风 ghost 生效，diffusion 侧的解析壁面通量是更大的独立工作
            （与 k=0 情形是同一个已有的架构限制，不是本次新引入的差异）。

            2026-09-05 曾尝试补上这里的 SIPG（对称内罚 Galerkin）风格
            Dirichlet 罚通量（`penalty = gamma_face*(C_pen/d1)*(target-
            phi_owner)`，显式加进 flux_jump_phys），真实网格验证（从
            cube_demo 791,492 单元真实 checkpoint 续算 20 步）**决定性
            证伪**：C_pen/d1 这个有效"弹簧系数"在细网格近壁单元上
            （y+~1 设计意味着 d1 可以小到 1e-5~1e-4 量级）过大，用和
            其余残差项相同的显式时间积分（没有做成 point-implicit，
            对比 sst.py::update_fields 里 destruction 项那样的处理）
            必然刚性超调——20 步内把全域 omega_mean 从 2.8e4 打到
            5.4e11（远超 1e6 的安全上限，且是全域均值，不是局部
            异常），k_mean 被连带压垮到 0.16，是真实的数值失稳而不是
            "改善"。已完整撤销（Git 历史可查这次尝试+撤销的完整过程），
            扩散侧解析壁面通量这个架构缺口依然存在，如需真正补上，
            必须先把罚项做成 point-implicit（或大幅限制 dt/加严格的
            CFL 缩放），不能像这次一样直接显式代入——留给后续需要
            专门处理数值刚性的独立工作，不要重复这次已经证伪的显式
            实现方式。当前生产代码依赖 `enforce_omega_wall_relaxation`
            （事后松弛，已用真实 900+ 步续算验证是稳定的）作为这个
            架构缺口的安全缓解措施。

        flat_face_override: 见 `compute_scalar_convection_residual` 同名
            参数文档，分布式路径复用同一个约定。

    Returns:
        residual: (n_cells, n_sps) 扩散残差
    """
    n_cells = mesh.n_cells
    n_sps = mesh.n_sps_per_cell
    n_prism = mesh.n_prism_cells

    det_jacs = mesh.jacobians["det_jacs"].reshape(n_cells, n_sps)
    inv_jacs = mesh.jacobians["inv_jacs"].reshape(n_cells, n_sps, 3, 3)

    # === 体积项 ===
    # 计算标量梯度（度量项一致）——grad_phi 下面界面项的逐分量外插还要
    # 复用，不能纳入分块（只能分块处理它之后、只在体积项内部一次性
    # 使用的 adj_j/G_phys/G_tilde）。
    grad_phi = compute_physical_scalar_gradient(scalar_field, mesh, ops)  # (n_cells, n_sps, 3)

    # 按单元分块执行 adj_j 构造→扩散通量→逆变通量→散度 全链路（真实
    # 内存修复，2026-09-01，理由与 compute_scalar_convection_residual
    # 体积项分块同一处文档：cube_demo 79万单元 P2+SST 组合内存峰值实测
    # 约需 37GB，超过常见 32GB 工作站配置）。
    _tet_op_D = ops.D_native_tet_padded if getattr(ops, "D_native_tet_padded", None) is not None else ops.D_3d_tet
    # 性能优化（2026-09-13，与对流项体积项同一次剖析、同一手法）：整条
    # "度量×扩散通量→散度"链换成 `contravariant_flux_from_metric` +
    # `scalar_volume_divergence_kernel` 两个 numba prange kernel，取代
    # 原来 Python 层分块 + 逐点 3x3@3x1 微型 gemm + 3 次 tensordot 的
    # numpy 链路（见两个 kernel 各自的文档）。
    # 去混叠（AFCFD_TURB_OVERINT=on），理由见 `resolve_turb_overintegration`
    # 与 `_scalar_diffusion_volume_overintegrated`（含那里明确写出的、
    # Gamma 自身混叠仍在的局限）。默认 off，行为逐位不变。
    _oi = _turb_overint_ops(mesh, ops) if resolve_turb_overintegration() == "on" else None
    if _oi is not None:
        div_G = _scalar_diffusion_volume_overintegrated(
            gamma_field, grad_phi, _oi, n_sps)
    else:
        G_phys = gamma_field[:, :, None] * grad_phi                      # (n_cells,n_sps,3)
        G_tilde = contravariant_flux_from_metric(det_jacs, inv_jacs, G_phys[..., None])[..., 0]
        del G_phys
        div_G = np.empty((n_cells, n_sps))
        if n_prism > 0:
            scalar_volume_divergence_kernel(
                np.ascontiguousarray(G_tilde[:n_prism]),
                np.ascontiguousarray(ops.D_3d_prism), div_G[:n_prism],
            )
        if n_cells > n_prism:
            scalar_volume_divergence_kernel(
                np.ascontiguousarray(G_tilde[n_prism:]),
                np.ascontiguousarray(_tet_op_D), div_G[n_prism:],
            )
        del G_tilde

    # 扩散对 dphi/dt 的贡献是 +div(G)/det(J)（见本函数文档符号约定，
    # 与 viscous_flux.py::"residual = div_comp / det_jacs"同一约定）。
    # 退化单元溢出保护：理由/验证方式同 compute_scalar_convection_
    # residual 里对应的 errstate（见该函数文档），同一类已知、已在
    # compute_turbulence_transport_residual 末尾被下游清零处理的溢出。
    with np.errstate(over='ignore', invalid='ignore'):
        residual = div_G / det_jacs
    del div_G

    # === 界面项（BR1 平均通量校正）===
    flat = flat_face_override if flat_face_override is not None else get_flat_face_geometry(mesh, ops)
    n_fp = flat.n_fp

    # 外插 Gamma 和标量梯度到面通量点。标量场本身不再外插：2026-08-25 校正改
    # 用梯度差形式后，phi 的界面取值（原 BR1 phi_avg）不再进入校正项；
    # gamma_field 用 Neumann 默认（扩散系数与是否 Dirichlet 无关），
    # grad_phi 逐分量双侧外插——BR1 公共通量需要平均梯度，两侧缺一不可。
    gamma_owner_fp, gamma_neighbor_fp = _extrapolate_scalar_to_faces(gamma_field, flat, ops, mesh)

    grad_owner_fp = np.zeros((flat.n_faces, n_fp, 3))
    grad_neighbor_fp = np.zeros((flat.n_faces, n_fp, 3))
    for d in range(3):
        go, gn = _extrapolate_scalar_to_faces(grad_phi[:, :, d], flat, ops, mesh)
        grad_owner_fp[:, :, d] = go
        grad_neighbor_fp[:, :, d] = gn
    del grad_phi  # 体积项+这里的逐分量外插都用完了，终于可以释放

    # BR1 平均：gamma_face = 0.5*(gamma_o + gamma_n)
    gamma_face = 0.5 * (gamma_owner_fp + gamma_neighbor_fp)
    del gamma_owner_fp, gamma_neighbor_fp

    # 通量差（真实修复，2026-08-25）：G_common - G_internal =
    # gamma_face*(grad_avg - grad_owner) = gamma_face*0.5*(grad_n - grad_o)
    # （phi_avg 的梯度即两侧外插梯度的平均）。此前实现因"grad_phi_neighbor
    # 不能直接外推到面 FPs"而放弃梯度差、改用状态跳跃 gamma_face*0.5*(phi_n
    # - phi_o) 冒充通量差——外插算子对任何标量场（包括梯度分量）本来就同样
    # 适用，这个前提不成立；且状态跳跃缺一个 1/长度因子，量纲与通量密度差
    # ~h 倍，与反扩散体积项叠加后成为 k/omega 场双峰触限失稳的放大器。
    delta_grad = 0.5 * (grad_neighbor_fp - grad_owner_fp)  # (n_faces, n_fp, 3)
    with np.errstate(over='ignore', invalid='ignore'):
        # 取法向分量：扩散通量差是矢量差的法向投影，坐标不变；简单三分量
        # 求和会随坐标系旋转变号/变幅值，不是标量不变量。
        flux_jump_phys = gamma_face * np.sum(delta_grad * flat.true_normal, axis=-1)
    del grad_owner_fp, grad_neighbor_fp, delta_grad, gamma_face

    # 面元幅值因子（真实修复，2026-08-25 代码审查）：上面用单位法向点积算出的是物理
    # 通量密度差，而平均流无粘/粘性界面项送进同一套分配链路的跳越量都是协变
    # 通量（物理通量 × |adj_row|，含面元幅值）：inviscid_kernel.py L197
    # `F_common_n * adj_mag`（归一化只用于方向对齐检查，幅值随后乘回）、
    # viscous_flux_kernel.py L182 `adjrow_o · G`。缺这个 ~O(h²) 因子会把校正放大
    # ~1/h²（细网格 10²~10³ 倍），破坏体积项与界面项的量级平衡——实测：
    # 修复前抛物线场内部残差均值被界面项主导（+143 界面 / -21 体积），
    # 补 |adj_row| 后界面项回到与体积项同量级。true_normal 是单位向量，必须补回。
    #
    # native 四面体（路径C）真实 bug 修复（2026-08-30，理由同
    # compute_scalar_convection_residual 同名注释）：不在这里统一预乘
    # |adj_row|，改为传未加权的 flux_jump_phys，加权方式按面类型分派
    # 下沉到 _distribute_correction_to_cells/kernel 内部。
    raw_jump_fp = flux_jump_phys

    # 分配回 SPs（kernel 对 owner 侧 -=、neighbor 侧 +=）
    interface_correction = _distribute_correction_to_cells(raw_jump_fp, flat, ops, mesh)
    # 扩散校正合成符号（见本函数文档符号约定）：kernel 返回的 owner 侧贡献是
    # -lift(correction_fp)，这里 residual - interface_correction =
    # +lift(G_common - G_internal)，即对 dphi/dt 施加标准 +lift 校正：
    # 邻居值/梯度更高时 owner 获得正的扩散增量，方向与物理一致。
    with np.errstate(over='ignore', invalid='ignore'):
        residual = residual - interface_correction

    return residual
