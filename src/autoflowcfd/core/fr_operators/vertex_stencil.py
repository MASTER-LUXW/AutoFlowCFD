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
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class VertexStencil:
    """`(node, cell)` 对的紧凑表示 + 顶点数。

    Attributes:
        node_of_pair: `(n_pairs,)` int64，每一对的顶点全局编号。
        cell_of_pair: `(n_pairs,)` int64，每一对的单元索引。
        n_nodes: 顶点总数（`node_of_pair` 的取值上界 + 1），决定逐顶点
            归约数组的长度。
    """

    node_of_pair: np.ndarray
    cell_of_pair: np.ndarray
    n_nodes: int


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
    # 第二趟：把每个顶点的极值散射回共享它的全部单元
    scatter_minmax(xp, nb_max, nb_min, cell, node_max[node])
    scatter_minmax(xp, nb_max, nb_min, cell, node_min[node])


def remap_to_compact(stencil: VertexStencil, global_to_compact: dict,
                     n_local: int, n_compact: int) -> VertexStencil:
    """把**全局**顶点模板重映射到分布式的 local+halo 紧凑索引空间。

    Args:
        stencil: 用全局单元索引建好的模板。
        global_to_compact: `全局单元 id -> 紧凑索引`。紧凑空间是
            `[0, n_local)` 为本 rank 单元、`[n_local, n_compact)` 为 halo
            （见 `core/mpi/distributed_compute.DistributedMeshAdapter`
            类文档）。
        n_local: 本 rank 的单元数。
        n_compact: 紧凑空间大小（local + halo）。

    Returns:
        重映射后的模板，只含两端都落在紧凑空间里的对。

    Raises:
        ValueError: **halo 不足以覆盖某个 local 单元的完整顶点邻域**。

            为什么必须硬失败：共享同一顶点的两个单元在共形网格里通过
            面邻接是连通的，但可能相隔 **2 个以上**面跳，而本项目的 halo
            是 1 层面邻居。缺一部分顶点邻居会让包络在分区边界上变窄，
            于是**同一个算例换 rank 数得到不同的掩码** —— 结果依赖分区，
            这是求解器不可接受的（与 `fr_solver/filter.py` 里
            `halo_extend` 那段说明同一条理由）。

            静默退回"只用面邻居"也不行：那正是本模块要修的那个缺陷
            （欠解析光滑场上标记 100%），而且会让单机与分布式两条后端
            的判据不一致 —— 本项目反复出过"两份实现只改一份"的缺陷。
    """
    n_global = int(stencil.cell_of_pair.max()) + 1 if stencil.cell_of_pair.size else 0
    lut = np.full(n_global, -1, dtype=np.int64)
    for g, c in global_to_compact.items():
        if 0 <= int(g) < n_global:
            lut[int(g)] = int(c)
    mapped = lut[stencil.cell_of_pair]
    keep = mapped >= 0

    # 完整性校验：每个 local 单元在紧凑空间里保留的对数，必须等于它在
    # 全局模板里的对数 —— 也就是"它的顶点邻域一个都没丢"。
    # 逐对计数而不是逐顶点：同一个 (node, cell) 对被保留 <=> 该 cell 在
    # 紧凑空间里；而我们要保证的是该 cell 的**每个顶点的每个共享单元**
    # 都在，所以要按顶点检查。
    node_kept = np.zeros(stencil.n_nodes, dtype=np.int64)
    node_total = np.zeros(stencil.n_nodes, dtype=np.int64)
    np.add.at(node_total, stencil.node_of_pair, 1)
    np.add.at(node_kept, stencil.node_of_pair[keep], 1)
    incomplete_node = node_kept != node_total
    if np.any(incomplete_node):
        # 只有"某个 local 单元用到了不完整的顶点"才是真问题：halo 单元
        # 的掩码本来就被丢弃。
        local_pairs = keep & (mapped < n_local)
        touched = np.unique(stencil.node_of_pair[local_pairs])
        bad = touched[incomplete_node[touched]]
        if bad.size:
            n_bad_cells = int(np.unique(
                mapped[local_pairs & np.isin(stencil.node_of_pair, bad)]).size)
            raise ValueError(
                f"顶点邻域在分区边界上不完整：{bad.size} 个顶点、涉及 "
                f"{n_bad_cells} 个 local 单元的顶点邻居落在 halo 之外。"
                f"本项目的 halo 是 1 层面邻居，而共享顶点的单元可能相隔 "
                f"2 个以上面跳。继续下去会让同一算例换 rank 数得到不同的"
                f"掩码（结果依赖分区）。当前分布式路径请用 "
                f"AFCFD_TROUBLED_SENSOR=persson 或 AFCFD_FILTER_MODE=off；"
                f"要在分布式上用 bounds 判据，需要先把 halo 扩到 2 层"
                f"（那是一项独立改动，不能靠退回面邻居模板绕过 —— 面模板"
                f"正是本模块要修的那个缺陷）。")

    return VertexStencil(
        node_of_pair=np.ascontiguousarray(stencil.node_of_pair[keep]),
        cell_of_pair=np.ascontiguousarray(mapped[keep]),
        n_nodes=stencil.n_nodes,
    )
