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
        `((lo, hi, c2f, D_fine, f2c), ...)` 两段（棱柱在前、四面体在后，
        与"棱柱在前"的单元存储顺序一致），或 None。
    """
    if 'adj_j_fine' not in mesh_data:
        return None
    for k in OVERINT_OPS_KEYS:
        if k not in ops_data:
            return None
    return (
        (0, n_prism, ops_data['overint_interp_c2f_prism'],
         ops_data['overint_D_fine_prism'], ops_data['overint_restrict_f2c_prism']),
        (n_prism, n_cells, ops_data['overint_interp_c2f_tet'],
         ops_data['overint_D_fine_tet'], ops_data['overint_restrict_f2c_tet']),
    )
