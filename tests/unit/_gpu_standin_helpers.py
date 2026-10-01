"""GPU 测试替身（numpy 代 CuPy）的**补全**辅助（2026-09-15）。

## 为什么需要它

本目录下有四处各自独立的 `mesh_data`/`ops_data` 替身构造
（`test_gpu_scalar_transport.py`、`test_gpu_scalar_transport_native_tet.py`、
`test_gpu_distributed_turbulence.py`、`test_gpu_solver_turbulence_source.py`），
都只放"这里用得到的键"。问题是"用得到"的判断会过期：

- `D_native_tet_padded`：生产 GPU 路径按
  `'D_native_tet_padded' in ops_data` 分派，替身漏了它 -> GPU 侧静默对
  四面体用坍缩坐标算子；
- 过积分（去混叠）六个算子 + `adj_j_fine`：`AFCFD_TURB_OVERINT` 默认改为
  `on` 之后，替身漏了它们 -> GPU 侧静默退回 coarse 体积项。

两种情况下 "CPU-GPU 等价" 这类交叉验证都变成**拿两条不同的数值方案
对比**——2026-09-15 把 `AFCFD_TURB_OVERINT` 默认改成 `on` 时，四个文件
一起失败才暴露出来。把补全逻辑收到一处，避免下一次新增 GPU 侧能力时
再犯同一个遗漏。
"""

import numpy as np

#: 过积分（去混叠）算子键：直接用生产上传路径的同一份名单（不再抄第二份——
#: 2026-10-01 新增体积算子 K 的键时，抄的那份就漏了）。
from autoflowcfd.core.gpu.gpu_overintegration import OVERINT_OPS_KEYS as OVERINT_KEYS  # noqa: E402


def complete_gpu_standin(mesh, ops, ops_data, mesh_data=None, compact_ids=None):
    """把生产 GPU 路径会上传、而替身容易漏掉的键补齐（就地修改）。

    Args:
        mesh, ops: 真实的 HighOrderMesh / FROperators
        ops_data: 放算子的 dict。有些替身把算子和网格数据放在同一个
            dict 里（分布式那份就是 `ops_data=mesh_data`），这种情况下
            两个参数传同一个对象即可。
        mesh_data: 放网格数据的 dict；None 时用 `ops_data`。
        compact_ids: 分布式 compact 索引空间的全局单元 id；给出时细点度量
            按它切片（与 coarse 侧同一套切法）。

    Returns:
        (ops_data, mesh_data)，便于链式使用。
    """
    if mesh_data is None:
        mesh_data = ops_data

    if getattr(ops, 'D_native_tet_padded', None) is not None:
        ops_data['D_native_tet_padded'] = ops.D_native_tet_padded

    if all(getattr(ops, k, None) is not None for k in OVERINT_KEYS):
        for k in OVERINT_KEYS:
            ops_data[k] = getattr(ops, k)

    if getattr(mesh, 'jacobians_fine', None) is not None:
        # 与生产上传路径同一个函数（numpy 充当数组模块）
        from autoflowcfd.core.gpu.gpu_overintegration import upload_fine_metrics_gpu
        from autoflowcfd.grid.high_order.order_jacobians import select_fine_metrics

        fine = mesh.jacobians_fine
        if compact_ids is not None:
            fine = select_fine_metrics(fine, compact_ids, mesh.n_prism_cells)
        mesh_data.update(upload_fine_metrics_gpu(np, fine))

    return ops_data, mesh_data
