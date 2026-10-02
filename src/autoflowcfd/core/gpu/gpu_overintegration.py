"""
AutoFlowCFD V2.0 - GPU 侧过积分（去混叠）上下文

CPU 端对应 `core/fr_operators/volume_contract.py::
get_overintegration_context`，这里是同一份算子/细点度量在 GPU 上的取法。
拆成独立模块的理由：GPU 侧现在有三个消费方——无粘体积项
（`residual/gpu_inviscid_volume.py`，一直开着）、k/omega 输运
（`turbulence/gpu_scalar_transport.py`）、粘性体积项
（`residual/gpu_viscous.py`，2026-09-15 补齐），让后两者从"湍流"模块
互相 import 会造成不该有的方向依赖。
"""

from autoflowcfd.core.utils.array_module import array_module as _array_module

#: 过积分算子键（两条上传路径 `array_manager.py` / `gpu_inviscid_volume.prepare_ops_data`
#: 共用这一份名单）；缺任何一个就退回 coarse 路径。`overint_lifted_div_*` 是无粘体积
#: 算子 K（修正项本侧通量迹已并入，`fr/face_flux_trace.py`）。
OVERINT_OPS_KEYS = (
    'overint_interp_c2f_prism', 'overint_D_fine_prism', 'overint_restrict_f2c_prism',
    'overint_interp_c2f_tet', 'overint_D_fine_tet', 'overint_restrict_f2c_tet',
    'overint_lifted_div_prism', 'overint_lifted_div_tet',
)


def upload_overintegration_ops_gpu(cp, ops, out: dict) -> None:
    """把 `OVERINT_OPS_KEYS` 里非 None 的算子上传进 `out`（两条上传路径共用）。"""
    import numpy as np

    for name in OVERINT_OPS_KEYS:
        op = getattr(ops, name, None)
        if op is not None:
            out[name] = cp.asarray(np.ascontiguousarray(op, dtype=np.float64))


def lifted_divergence_gpu(mesh_data, ops_data):
    """与 `get_overintegration_segs_gpu` 两段对应的 `(K_all, combo)`：按槽位组合各一份的无粘
    体积算子与该段逐单元组合编号（CPU 端 `get_overintegration_context` 的 `lifted_div`）。"""
    return ((ops_data['overint_lifted_div_prism'], mesh_data['k_combo_prism']),
            (ops_data['overint_lifted_div_tet'], mesh_data['k_combo_tet']))


def contract_lifted_divergence_gpu(cp, K_all, combo, X):
    """GPU 版 `volume_contract.contract_lifted_divergence`：`out[c] = K_all[combo[c]] : X[c]`。

    GPU 上按组合分组、每组一次批量收缩（组合数四面体 81、棱柱 9，组内是连续 gemm）；只有
    一组时不做 gather。
    """
    from autoflowcfd.core.gpu.residual.gpu_volume_contract import gpu_contract_shared_operator_2axis

    n = int(combo.shape[0])
    if n == 0:
        return cp.zeros((0, K_all.shape[1], X.shape[-1]))
    lo, hi = int(combo.min()), int(combo.max())
    if lo == hi:
        return gpu_contract_shared_operator_2axis(K_all[lo], X)
    order = cp.argsort(combo)
    sorted_c = combo[order]
    cuts = [0] + (cp.flatnonzero(sorted_c[1:] != sorted_c[:-1]) + 1).tolist() + [n]
    out = cp.empty((n, K_all.shape[1], X.shape[-1]))
    for a, b in zip(cuts[:-1], cuts[1:]):
        idx = order[a:b]
        out[idx] = gpu_contract_shared_operator_2axis(K_all[int(sorted_c[a])], X[idx])
    return out


def upload_fine_metrics_gpu(cp, jacobians_fine):
    """把网格的分段细点度量上传为 GPU 过积分要用的两个键（单机与多 GPU 的上传路径共用）。

    上传预乘好的 `adj = det * inv` 与无粘体积算子的逐单元组合编号：`adj_j_fine_prism (n_prism, n_fine, 3, 3)`
    逐细点、`adj_j_fine_tet (n_tet, 3, 3)` 逐单元（直边四面体 Jacobian 逐单元
    常数，见 `grid/high_order/order_jacobians.build_fine_metrics`）。此前两条上传
    路径各按 `(n_cells, n_fine)` 全场上传 det/inv/adj 三份，而 GPU 过积分只读
    adj——plate_demo P3 下四面体段在显存里约 4.7 GB，现在 14 MB。
    """
    import numpy as np

    def _adj(det, inv):
        return cp.asarray(np.ascontiguousarray(det[..., None, None] * inv, dtype=np.float64))

    return {
        'adj_j_fine_prism': _adj(jacobians_fine['prism_det'], jacobians_fine['prism_inv']),
        'adj_j_fine_tet': _adj(jacobians_fine['tet_det'], jacobians_fine['tet_inv']),
        # 无粘体积算子的逐单元槽位组合编号（与细点度量同一份逐单元几何，见 build_fine_metrics）
        'k_combo_prism': cp.asarray(np.ascontiguousarray(jacobians_fine['prism_k_combo'], dtype=np.int64)),
        'k_combo_tet': cp.asarray(np.ascontiguousarray(jacobians_fine['tet_k_combo'], dtype=np.int64)),
    }


def get_overintegration_segs_gpu(mesh_data, ops_data, n_cells, n_prism):
    """GPU 侧过积分分段；任一算子/细点度量缺失则返回 None（退回 coarse）。

    与 CPU 端 `get_overintegration_context` 逐字对应。细点度量用上传好的
    `adj_j_fine_prism`（逐细点）与 `adj_j_fine_tet`（逐单元，本函数广播到该段
    自己的 `n_fine_tet`，见 `upload_fine_metrics_gpu`）。

    缺失的唯一正常情形是 `order == 0`：P0 是分片常数场、多项式导数恒为
    零，没有可去混叠的内容，细点几何与 overint 算子都不构造。

    Returns:
        `((lo, hi, n_fine, adj_seg, c2f, D_fine, f2c), ...)` 两段（棱柱在前、
        四面体在后，与"棱柱在前"的单元存储顺序一致），或 None。每段自带
        **自己的** `n_fine` 与已切好的细点度量。

    `n_fine_tet` 从 `overint_D_fine_tet` 的**形状**推导（与 CPU 端同一个
    做法）——矩阵是单一事实来源，从形状推导在结构上不可能与它不同步。
    """
    if 'adj_j_fine_prism' not in mesh_data:
        return None
    for k in OVERINT_OPS_KEYS:
        if k not in ops_data:
            return None
    adj_prism = mesh_data['adj_j_fine_prism']
    adj_tet_cell = mesh_data['adj_j_fine_tet']
    # 两段的 n_fine 都从**矩阵自身的形状**推导（矩阵是单一事实来源），棱柱段再与
    # 上传的细点度量对账，与 CPU 端 `volume_contract.get_overintegration_context`
    # 同一道闸。
    n_fine_prism = int(ops_data['overint_D_fine_prism'].shape[0])
    if tuple(adj_prism.shape[:2]) != (n_prism, n_fine_prism):
        raise ValueError(
            f"棱柱过积分细点数不一致：算子 overint_D_fine_prism 的 n_fine="
            f"{n_fine_prism}，而上传的棱柱细点度量是 {tuple(adj_prism.shape)}"
        )
    n_fine_tet = int(ops_data['overint_D_fine_tet'].shape[0])
    xp = _array_module(adj_tet_cell)
    # 显式物化而不是留一个 0 步长的广播视图：GPU 侧这一段是**整段**一次
    # `matmul`（不像 CPU 端按 32768 单元分块后逐块 ascontiguousarray），
    # 0 步长视图喂给 cuBLAS 的行为不该依赖库的内部处理。这是本次调用内的
    # 临时量，不常驻显存。
    adj_tet = xp.ascontiguousarray(
        xp.broadcast_to(adj_tet_cell[:, None], (n_cells - n_prism, n_fine_tet, 3, 3)))
    return (
        (0, n_prism, n_fine_prism, adj_prism,
         ops_data['overint_interp_c2f_prism'],
         ops_data['overint_D_fine_prism'], ops_data['overint_restrict_f2c_prism']),
        (n_prism, n_cells, n_fine_tet, adj_tet,
         ops_data['overint_interp_c2f_tet'],
         ops_data['overint_D_fine_tet'], ops_data['overint_restrict_f2c_tet']),
    )
