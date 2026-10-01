"""AutoFlowCFD V2.0 - 单元块 `J_cc = dR_c/dU_c` 的着色有限差分装配（块 Jacobi / 块 ILU 用）。

## 装配：着色有限差分，成本与单元数无关

块 `J_cc = dR_c / dU_c`（单元 `c` 的全部真实解点 x 全部变量）。按面
相邻关系给单元做**距离 1 着色**（相邻单元不同色），同色单元**同时**
扰动同一个 `(解点 s, 变量 v)`：单元 `c` 的残差只受它自己与面邻居的状态
影响，而面邻居与它不同色、没有被扰动，于是

    [R(U + h e_{色k,s,v}) - R(U)] / h   在色 k 的每个单元 c 上
                                         恰好是 J_cc 的第 (s,v) 列

总残差求值次数 = `色数 x 真实解点数上限 x 变量数`，**与单元数无关**
（plate_demo P1：5 色 x 6 x 5 = 150 次）。旧文档里"每单元 `n_sps*n_var`
次残差求值、成本不可接受"的论证漏掉了着色，已同步更正。

**模板是距离 1**（2026-09-26 更正）：旧文档写过"BR1 梯度提升让残差依赖
距离 2 的单元、同色距离 2 单元会混入 `J_cc`"。实际上梯度是纯单元内局部梯度
（`fr_operators/gradients.py`，签名里没有面数据），残差只经本单元与面邻居，
距离 1 着色下差分装配出的 `J_cc` 是精确的（稠密逐列差分 Jacobian 对照：面邻居
之外的块全部为零）。

## 面邻居耦合块（P0）

`coupling_graph`（`coloring.py::CouplingGraph`）给出时改用**距离 2 着色**：同色
单元互不相邻、也不共享面邻居，同一批差分同时给出 `J_cc` 与 `J_cy`，块 ILU 就有了
耦合块。每色要 `真实解点数 x 变量数` 次残差求值，所以只在 P0（每单元一个解点、
约 20~30 色 x 5 次）用；P>=1 的耦合块由解析装配给出（`fr_residual/jacobian`）。
"""

import numpy as np

from .reductions import LocalReductions

_SQRT_EPS = float(np.sqrt(np.finfo(np.float64).eps))


class CellBlockJacobian:
    """单元对角块 `J_cc = dR_c/dU_c`（棱柱、四面体各一组，只含真实自由度）。

    后端无关：状态/残差/块在 `red.xp`（numpy 或 cupy）上；着色与单元类型
    划分是主机端 numpy。**分布式约束**：每次残差求值都是集体操作，所有
    rank 必须调用同样多次——所以"本色本解点有没有要扰动的单元"用
    `red.max` 取全局结论，而不是本地没有就跳过（那会让各 rank 的调用
    次数不一致而死锁）。着色必须是全局一致的（同色单元跨 rank 也不相邻）。
    """

    __slots__ = ("n_sps", "n_var", "prism_cells", "tet_cells", "n_real_prism",
                 "n_real_tet", "blocks_prism", "blocks_tet", "n_residual_evals", "xp", "coupling",
                 "cross_coupling")

    def __init__(self, residual, u0_flat, r0_flat, scales: np.ndarray, *, n_sps: int,
                 cell_is_prism: np.ndarray, n_real_prism: int, n_real_tet: int,
                 colors: np.ndarray, red: LocalReductions = None, coupling_graph=None):
        """`coupling_graph`（`coloring.CouplingGraph`）给出时按它的**距离 2 着色**
        扰动，同一批残差差分同时截取面邻居耦合块 `J_cy`（`self.coupling`，块 ILU 用）：
        同色单元互不相邻、也不共享面邻居，未被扰动的单元 `c` 至多有一个被扰动的
        面邻居 `y`，`c` 行上的差分就是 `J_cy` 的一列。此时忽略 `colors`。分布式下同时截取
        图里给出的跨 rank 耦合块（`self.cross_coupling`，列为 halo 单元的紧凑空间编号）。"""
        red = red if red is not None else LocalReductions()
        xp = self.xp = red.xp
        u0 = xp.ascontiguousarray(u0_flat, dtype=xp.float64)
        r0 = xp.ascontiguousarray(r0_flat, dtype=xp.float64)
        n_var = u0.shape[1]
        cell_is_prism = np.asarray(cell_is_prism, dtype=bool)
        n_cells = cell_is_prism.size
        if u0.shape[0] != n_cells * n_sps:
            raise ValueError(f"状态行数 {u0.shape[0]} != n_cells*n_sps = {n_cells}*{n_sps}")
        scales = np.asarray(scales, dtype=np.float64).ravel()
        self.n_sps, self.n_var = n_sps, n_var
        self.prism_cells = np.nonzero(cell_is_prism)[0]
        self.tet_cells = np.nonzero(~cell_is_prism)[0]
        self.n_real_prism, self.n_real_tet = n_real_prism, n_real_tet
        bp, bt = n_real_prism * n_var, n_real_tet * n_var
        self.blocks_prism = xp.zeros((self.prism_cells.size, bp, bp), dtype=xp.float32)
        self.blocks_tet = xp.zeros((self.tet_cells.size, bt, bt), dtype=xp.float32)
        # 单元号 -> 在各自块数组里的下标
        slot = np.empty(n_cells, dtype=np.int64)
        slot[self.prism_cells] = np.arange(self.prism_cells.size)
        slot[self.tet_cells] = np.arange(self.tet_cells.size)

        # 差分步长：与 `MatrixFreeJacobian` 同一取法——在按参考量级无量纲化
        # 的空间里取 `sqrt(eps_mach) * (1 + rms(U~))`，再换回物理量纲。
        u0_rms_scaled = red.rms(u0 / xp.asarray(scales)[None, :])
        h = _SQRT_EPS * (1.0 + u0_rms_scaled) * scales

        cross = None
        if coupling_graph is not None:
            colors = coupling_graph.colors
            cross = _CouplingCapture(coupling_graph, cell_is_prism, n_sps, n_real_prism, n_real_tet, n_var, xp)
        colors = np.asarray(colors, dtype=np.int64)
        n_colors = int(red.max(xp.asarray([colors.max() if colors.size else -1]))) + 1
        n_eval = 0
        for k in range(n_colors):
            if cross is not None:
                cross.select_color(k)
            in_k = colors == k
            groups = []
            for cells, blocks, n_real in ((np.nonzero(in_k & cell_is_prism)[0], self.blocks_prism, n_real_prism),
                                          (np.nonzero(in_k & ~cell_is_prism)[0], self.blocks_tet, n_real_tet)):
                if cells.size:
                    rows = cells[:, None] * n_sps + np.arange(n_real)[None, :]
                    groups.append((xp.asarray(slot[cells]), blocks, n_real,
                                   xp.asarray(rows), cells.size))
            local_s = max((g[2] for g in groups), default=0)
            n_s = int(red.max(xp.asarray([local_s])))       # 全局一致的调用次数
            for s in range(n_s):
                for v in range(n_var):
                    up = u0.copy()
                    for _, _, n_real, rows, _ in groups:
                        if s < n_real:
                            up[rows[:, s], v] += h[v]
                    dr = (xp.asarray(residual(up), dtype=xp.float64) - r0) / h[v]
                    n_eval += 1
                    col = s * n_var + v
                    for slots, blocks, n_real, rows, nc in groups:
                        if s < n_real:
                            blocks[slots, :, col] = dr[rows.ravel()].reshape(nc, n_real * n_var)
                    if cross is not None:
                        cross.store(dr, s, col)
        self.n_residual_evals = n_eval
        self.coupling, self.cross_coupling = (None, None) if cross is None else cross.result()


    @classmethod
    def from_blocks(cls, blocks_prism, blocks_tet, *, n_sps: int, n_var: int, cell_is_prism,
                    n_real_prism: int, n_real_tet: int, xp=np):
        """由外部装配好的块构造（解析装配，见 `fr_residual/jacobian`）；布局与差分装配相同。"""
        obj = cls.__new__(cls)
        cell_is_prism = np.asarray(cell_is_prism, dtype=bool)
        obj.xp = xp
        obj.n_sps, obj.n_var = int(n_sps), int(n_var)
        obj.prism_cells = np.nonzero(cell_is_prism)[0]
        obj.tet_cells = np.nonzero(~cell_is_prism)[0]
        obj.n_real_prism, obj.n_real_tet = int(n_real_prism), int(n_real_tet)
        for name, blocks, cells, n_real in (("blocks_prism", blocks_prism, obj.prism_cells, n_real_prism),
                                            ("blocks_tet", blocks_tet, obj.tet_cells, n_real_tet)):
            expect = (cells.size, n_real * n_var, n_real * n_var)
            if tuple(blocks.shape) != expect:
                raise ValueError(f"{name} 形状 {tuple(blocks.shape)} 与期望 {expect} 不符")
            setattr(obj, name, xp.asarray(blocks, dtype=xp.float32))
        obj.n_residual_evals = 0
        obj.coupling = obj.cross_coupling = None
        return obj


class _CouplingCapture:
    """`CellBlockJacobian` 差分循环里截取面邻居耦合块（距离 2 着色，见其文档）。

    耦合块按 (行单元类型, 列单元类型) 分四组存放，布局与解析装配的
    `fr_residual/jacobian/coupling.py::CouplingBlocks` 相同（块内 `(解点, 变量)`
    行主序），块 ILU 直接消费。跨 rank 耦合块（列是 halo 单元，分布式）另存一组，
    同样分四组；每组记下列单元的颜色，本色扰动时取出列单元同色的那些块。
    """

    __slots__ = ("inner", "halo", "xp", "n_sps", "n_var", "_active")

    def __init__(self, graph, cell_is_prism, n_sps, n_real_prism, n_real_tet, n_var, xp):
        self.xp, self.n_sps, self.n_var = xp, int(n_sps), int(n_var)
        colors = np.asarray(graph.colors, dtype=np.int64)
        rows, cols = np.asarray(graph.rows, dtype=np.int64), np.asarray(graph.cols, dtype=np.int64)
        h_rows, h_cols = np.asarray(graph.halo_rows, dtype=np.int64), np.asarray(graph.halo_cols, dtype=np.int64)
        self.inner = self._groups(rows, cols, cell_is_prism[rows], cell_is_prism[cols], colors[cols],
                                  cell_is_prism, n_real_prism, n_real_tet)
        self.halo = self._groups(h_rows, h_cols, cell_is_prism[h_rows], np.asarray(graph.halo_col_is_prism, bool),
                                 np.asarray(graph.halo_colors, dtype=np.int64), cell_is_prism, n_real_prism,
                                 n_real_tet)
        self._active = []

    def _groups(self, rows, cols, row_p_of, col_p_of, col_color, cell_is_prism, n_real_prism, n_real_tet):
        groups = []
        for row_p in (True, False):
            for col_p in (True, False):
                sel = (row_p_of == row_p) & (col_p_of == col_p)
                nr = n_real_prism if row_p else n_real_tet
                ny = n_real_prism if col_p else n_real_tet
                r = rows[sel]
                groups.append(dict(row_p=row_p, col_p=col_p, rows=r, cols=cols[sel], color=col_color[sel],
                                   nr=nr, ny=ny,
                                   blocks=self.xp.zeros((r.size, nr * self.n_var, ny * self.n_var),
                                                        dtype=self.xp.float32)))
        return groups

    def select_color(self, k: int):
        """本色被扰动的列单元对应的耦合块：`(组, 块下标, 行单元的真实解点行)`。"""
        self._active = []
        for g in self.inner + self.halo:
            idx = np.nonzero(g["color"] == k)[0]
            if idx.size:
                rsp = g["rows"][idx][:, None] * self.n_sps + np.arange(g["nr"])[None, :]
                self._active.append((g, self.xp.asarray(idx), self.xp.asarray(rsp.ravel()), idx.size))

    def store(self, dr, s, col):
        for g, idx, rsp, n in self._active:
            if s < g["ny"]:
                g["blocks"][idx, :, col] = dr[rsp].reshape(n, g["nr"] * self.n_var)

    def result(self):
        """`(rank 内耦合块, 跨 rank 耦合块)`，均为 `CouplingBlocks`。"""
        from autoflowcfd.core.fr_residual.jacobian.coupling import CouplingBlocks, CouplingGroup

        host = (lambda a: a) if self.xp is np else (lambda a: a.get())
        return tuple(CouplingBlocks(groups=[
            CouplingGroup(row_is_prism=g["row_p"], col_is_prism=g["col_p"], rows=g["rows"], cols=g["cols"],
                          blocks=host(g["blocks"]))
            for g in groups if g["rows"].size]) for groups in (self.inner, self.halo))
