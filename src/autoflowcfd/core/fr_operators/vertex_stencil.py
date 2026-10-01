"""AutoFlowCFD V2.0 - 单元的**顶点邻域**（共享任一顶点的全部单元）。

## 为什么需要它：BJ 判据用面邻居在三维四面体上是错的模板

`bounds_sensor.py` 的 BJ 越界判据原先用**面邻居**的单元均值张成包络。
在三维四面体网格上这个模板**不够**：一个四面体只有 4 个面邻居，它们的
单元均值在三个方向上不能把本单元夹住，于是 O(h |grad u|) 的**合法光滑
变化**被当成越界。

实测（TGV 解析初场，三重周期、纯四面体、P2，三档网格加密）：

    模板       越界量随 h 的观测阶 p      默认容差下的标记比例（n=16）
    面邻居     0.72 / 0.96（即 O(h)）            100.00%
    顶点邻居   1.58 / 1.28                         6.18%

    n=16（24576 单元, h=0.216）  越界max/尺度   越界中位/尺度
    面邻居                        1.950e-01      8.04e-02
    顶点邻居                      8.576e-02      **0.0**

两条结论：

1. **面模板下越界量是 O(h^1)**，所以经典 TVB 修正（Cockburn-Shu 的
   `M h^2` 容差）**修不了它** —— `h^2` 在 h 小时比 O(h) 衰减更快，细网格
   上判据仍会标记全部单元。那条方案由此被测量否掉。
2. 换成顶点模板后标记比例 100% -> 6.18%、**中位越界降到恰好 0**（过半
   单元完全不越界），阶数也从 ~0.85 提到 ~1.4。

顶点（node-based）模板本来就是非结构限制器里 Barth-Jespersen 的常见
实现形式，理由正是这条"面邻居在三维不能形成包围"。

## 代价：与面模板同阶

每个四面体贡献 4 个 `(node, cell)` 对，每个棱柱 6 个；而面模板的
`(face, cell)` 对数是 `2 * n_faces`，四面体网格上 `n_faces ≈ 2 n_cells`
所以也是约 `4 n_cells`。两者同阶，没有数量级差别。

算法是**两趟散射**，不是逐顶点 Python 循环：

    ① 逐顶点归约：node_max[node] = max over cells sharing it
    ② 散射回单元：nb_max[cell] = max(nb_max[cell], node_max[node])

两趟都复用 `bounds_sensor._scatter_minmax`（那一层已经把 NumPy 的
`ufunc.at` 与 CuPy 的 `cupyx.scatter_max/min` 统一了），所以 CPU 单机 /
CPU MPI / 单 GPU / 多 GPU 四条后端共用同一个内核。

## 分布式（2026-10-01）

共享同一顶点的两个单元通过面邻接可能相隔 2 个以上面跳，而 halo 是 1 层
面邻居，所以分布式不能靠 halo 拼出完整顶点邻域。改为在**逐顶点归约之后**
对跨 rank 共享的顶点做一次 MAX/MIN 全局归约（`VertexStencil.reduce_nodes`，
构造见 `core/mpi/vertex_stencil_mpi.py`）：每个 rank 只用自己的 local 单元
建模板，共享顶点的极值合并后就是全局极值。max/min 与顺序无关，结果与单机
逐位相同、与分区数无关。此前的 `remap_to_compact`（试图把全局模板映射进
local+halo 空间，halo 不够时硬失败）从未被调用，随之删除。
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass(frozen=True)
class VertexStencil:
    """`(node, cell)` 对的紧凑表示 + 顶点数。

    Attributes:
        node_of_pair: `(n_pairs,)` int64，每一对的顶点全局编号。
        cell_of_pair: `(n_pairs,)` int64，每一对的单元索引。
        n_nodes: 顶点总数（`node_of_pair` 的取值上界 + 1），决定逐顶点
            归约数组的长度。
        reduce_nodes: 分布式下把**跨 rank 共享顶点**的逐顶点极值合并成全局
            极值的可调用 `reduce_nodes(node_max, node_min)`（原地、集合通信，
            全部 rank 必须同时调用）；单机为 None。见
            `core/mpi/vertex_stencil_mpi.py`。
    """

    node_of_pair: np.ndarray
    cell_of_pair: np.ndarray
    n_nodes: int
    reduce_nodes: Optional[Callable] = None


def build_vertex_stencil(mesh) -> Optional[VertexStencil]:
    """从网格的单元-顶点连接构造顶点模板；没有可用连接时返回 `None`。

    "棱柱在前、四面体在后"的单元排列与
    `mesh.jacobians`/`state.U` 的行顺序一致（见
    `grid/high_order/order_jacobians.py::_combine_prism_and_tet_jacobians`），
    所以这里的 `cell_of_pair` 直接就是那套索引，不需要任何重映射。

    Returns:
        `VertexStencil` 或 `None`（网格没有 `_fixed_*_conn`，例如某些
        只关心归约语义的合成单元测试网格）。**返回 None 而不是抛异常**：
        调用方（`bounds_sensor`）据此退回面模板，并且那条退回是显式的、
        有文档的，不是静默降级。
    """
    prism_conn = getattr(mesh, "_fixed_prism_conn", None)
    tet_conn = getattr(mesh, "_fixed_tet_conn", None)
    n_prism = int(getattr(mesh, "n_prism_cells", 0) or 0)

    pairs_node = []
    pairs_cell = []
    if prism_conn is not None and len(prism_conn) > 0:
        conn = np.asarray(prism_conn, dtype=np.int64)
        n, k = conn.shape
        pairs_node.append(conn.ravel())
        pairs_cell.append(np.repeat(np.arange(n, dtype=np.int64), k))
    if tet_conn is not None and len(tet_conn) > 0:
        conn = np.asarray(tet_conn, dtype=np.int64)
        n, k = conn.shape
        pairs_node.append(conn.ravel())
        pairs_cell.append(
            np.repeat(np.arange(n, dtype=np.int64) + n_prism, k))
    if not pairs_node:
        return None

    node_of_pair = np.ascontiguousarray(np.concatenate(pairs_node))
    cell_of_pair = np.ascontiguousarray(np.concatenate(pairs_cell))
    n_nodes = int(node_of_pair.max()) + 1 if node_of_pair.size else 0
    return VertexStencil(node_of_pair=node_of_pair,
                         cell_of_pair=cell_of_pair,
                         n_nodes=n_nodes)


def accumulate_vertex_envelope(xp, scatter_minmax, nb_max, nb_min,
                               cell_mean, stencil: VertexStencil) -> None:
    """把顶点邻域的单元均值极值**累加**到 `nb_max`/`nb_min`（原地）。

    Args:
        xp: `numpy` 或 `cupy`（由调用方按数组类型解析）。
        scatter_minmax: `bounds_sensor._scatter_minmax`，注入进来而不是
            反向 import —— 那会在两个模块之间造成循环依赖，而这一层本来
            只需要"一个对重复索引做 min/max 归约的散射"这个能力。
        nb_max / nb_min: `(n_cells,)`，调用方已用本单元均值初始化。
        cell_mean: `(n_cells,)` 该变量的逐单元均值。
        stencil: `build_vertex_stencil` 的产物。

    两趟散射，见模块文档"代价"一节。逐顶点数组每次调用新建：它只有
    `(n_nodes,)` 大小（四面体网格上约 `n_cells/5`），相对 `(n_cells,
    n_sps, n_var)` 的解场可忽略，换来的是不必在调用方之间传递可变缓存。
    """
    node = stencil.node_of_pair
    cell = stencil.cell_of_pair
    # 逐顶点归约的中性元：用 ±inf，这样没有任何单元的顶点（不可能，但
    # 保持形式上正确）不会污染结果。
    node_max = xp.full(stencil.n_nodes, -xp.inf, dtype=nb_max.dtype)
    node_min = xp.full(stencil.n_nodes, xp.inf, dtype=nb_min.dtype)
    scatter_minmax(xp, node_max, node_min, node, cell_mean[cell])
    if stencil.reduce_nodes is not None:
        # 分布式：共享顶点上还有别的 rank 的单元，合并成全局极值后再散射回
        # 本 rank 单元 —— max/min 与求值顺序无关，所以结果与单机逐位相同。
        stencil.reduce_nodes(node_max, node_min)
    # 第二趟：把每个顶点的极值散射回共享它的全部单元
    scatter_minmax(xp, nb_max, nb_min, cell, node_max[node])
    scatter_minmax(xp, nb_max, nb_min, cell, node_min[node])

