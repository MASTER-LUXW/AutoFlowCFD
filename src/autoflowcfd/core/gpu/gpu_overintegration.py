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

#: 六个过积分算子键；缺任何一个就退回 coarse 路径。
OVERINT_OPS_KEYS = (
    'overint_interp_c2f_prism', 'overint_D_fine_prism', 'overint_restrict_f2c_prism',
    'overint_interp_c2f_tet', 'overint_D_fine_tet', 'overint_restrict_f2c_tet',
)


def get_overintegration_segs_gpu(mesh_data, ops_data, n_cells, n_prism):
    """GPU 侧过积分分段；任一算子/细点度量缺失则返回 None（退回 coarse）。

    与 CPU 端 `get_overintegration_context` 逐字对应。GPU 侧的细点度量用
    `mesh_data['adj_j_fine']`（= det_fine*inv_fine，`array_manager.py`
    已预乘好并上传，无粘体积项也是用它），所以不需要再分别取 det/inv。
    缺失的唯一正常情形是 `order == 0`：P0 是分片常数场、多项式导数恒为
    零，没有可去混叠的内容，细点几何与 overint 算子都不构造。

    Returns:
        `((lo, hi, n_fine, adj_seg, c2f, D_fine, f2c), ...)` 两段（棱柱在前、
        四面体在后，与"棱柱在前"的单元存储顺序一致），或 None。每段自带
        **自己的** `n_fine` 与已切好的细点度量。

    ## 为什么每段各自带 n_fine（2026-09-17，与 CPU 端同一次改动）

    native 四面体的过积分**细网格轴**不再填充到棱柱的 `(oo+1)^3` 宽度：
    它只有 `(oo+1)(oo+2)(oo+3)/6` 个真实细点（P1: 10 vs 27，P2: 20 vs
    64），填充槽位恒为零、对结果零贡献，却让整条过积分链在空点上白算，
    其中 `D_fine` 的收缩是 O(n_fine^2)（P2 上 10.2 倍无效 FLOPs）。
    CPU 侧实测 P1 加速 3.04x、P2 加速 4.63x，最大相对差 1.4e-16 / 0.0。

    四面体段的细点度量取 `adj_j_fine[n_prism:, :1]` 再**广播**到该段自己的
    `n_fine_tet` 宽：直边四面体的 Jacobian 逐单元为**常数**，全部细点槽位
    存的是同一个值，所以第 0 列就是那个常数。与 CPU 端
    `volume_contract.get_overintegration_context` 完全同一条做法（第一版
    两边都是"切前 n_fine_tet 列"，那等价但额外要求
    `n_fine_tet <= n_fine_prism`，把 P3 的过积分阶数从理想的 6 夹到 5，
    实测去混叠误差因此从 4.80e-6 退到 3.37e-3）。

    `n_fine_tet` 从 `overint_D_fine_tet` 的**形状**推导（与 CPU 端同一个
    做法）——矩阵是单一事实来源，从形状推导在结构上不可能与它不同步。
    """
    if 'adj_j_fine' not in mesh_data:
        return None
    for k in OVERINT_OPS_KEYS:
        if k not in ops_data:
            return None
    adj_all = mesh_data['adj_j_fine']
    n_fine_prism = int(adj_all.shape[1])
    n_fine_tet = int(ops_data['overint_D_fine_tet'].shape[0])
    # CuPy 与 NumPy 的 broadcast_to 同名同义；`adj_all` 的后续轴（3,3 等）
    # 原样保留，只把细点轴从 1 扩到 n_fine_tet。
    xp = _array_module(adj_all)
    # 显式物化而不是留一个 0 步长的广播视图：GPU 侧这一段是**整段**一次
    # `matmul`（不像 CPU 端按 32768 单元分块后逐块 ascontiguousarray），
    # 0 步长视图喂给 cuBLAS 的行为不该依赖库的内部处理。物化的大小与
    # 改动前那个切片视图的逻辑大小相同，不是新增的内存开销。
    adj_tet = xp.ascontiguousarray(
        xp.broadcast_to(
            adj_all[n_prism:, :1],
            (n_cells - n_prism, n_fine_tet) + tuple(adj_all.shape[2:])))
    return (
        (0, n_prism, n_fine_prism, adj_all[:n_prism],
         ops_data['overint_interp_c2f_prism'],
         ops_data['overint_D_fine_prism'], ops_data['overint_restrict_f2c_prism']),
        (n_prism, n_cells, n_fine_tet, adj_tet,
         ops_data['overint_interp_c2f_tet'],
         ops_data['overint_D_fine_tet'], ops_data['overint_restrict_f2c_tet']),
    )
