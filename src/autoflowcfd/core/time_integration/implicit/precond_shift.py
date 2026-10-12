"""AutoFlowCFD V2.0 - 预处理矩阵的对角位移：伪时间项相对对角块量级的下限。

## 问题

块预处理（块 ILU、多层）作用在装配出的矩阵 `I/dtau + J_a` 上；`J_a` 是真实 Jacobian 的近似（平均流块冻结
涡粘、只含面邻居耦合，湍流块同理）。`dtau` 大到 `I/dtau` 可忽略时，被分解的就是 `J_a` 本身：它在近零模态
上与真实 `J` 的差被原样放大，预处理后的算子谱散开，GMRES 不收敛。cube_demo SA P1（37.4 万单元，稳态第
206 步的同一个状态、同一个线性系统，容差 0.1，重启长度 60）只改预处理里用的伪时间步：

    系统 CFL    预处理的等效 CFL    GMRES 次数          步后残差比    整步耗时
    3000        3000（与系统相同）   200（只到 0.43）    1.00          1163 s
    3000        900                 51                  0.927         358 s
    3000        300                 32                  0.930         250 s
    3000        90                  20                  0.932         179 s
    1e4         300                 56                  0.850         377 s
    1e4         100                 49                  0.870         338 s

同一次运行里还有旁证：复用着的旧分解（CFL 1500 时做的）在 CFL 1815 的系统上第 70 次迭代到 0.22，重装配、
按当前 CFL 重新分解后同一步要 171 次。

## 做法

预处理矩阵对角上的伪时间项逐行取 `max(1/dtau, PRECONDITIONER_SHIFT * d)`，`d` 是该（单元, 解点）处 `J_cc`
对角元绝对值的最大值——带对角位移的不完全分解，位移与对角量级成比例。被求解的线性系统不变（Krylov 算子
自己加 `I/dtau`，`jfnk.py::_solve_direction`），右预处理 GMRES 的解仍是原系统的解。

**只在线性求解失败之后启用**（`block_jacobi.py::BlockJacobiCache.recover`，启用后保持到块缓存重建，即
换阶）：`J_a` 足够准的算例靠的正是大 CFL 下预处理接近精确（湍流平板 P3 每步 GMRES 1~3 次），位移会让它们
白白变慢。启用前后的预处理都是合法的右预处理，只影响迭代数。

## 标定

`PRECONDITIONER_SHIFT = 0.04`：cube_demo 上 `1/(dtau d)` 的几何平均在 CFL 3e4 时是 1.28e-4（平均流块）/
3.05e-4（湍流块），0.04 对应平均流块的等效 CFL 约 96、湍流块约 230，即上表里迭代数最少的一档。
"""

#: 启用位移后伪时间项相对 `J_cc` 对角量级的下限（模块文档"标定"）。
PRECONDITIONER_SHIFT = 0.04


def block_diagonal_scale(jac, cell_is_prism, n_sps: int, n_real_prism: int, n_real_tet: int, xp):
    """逐（单元, 解点）`J_cc` 对角元绝对值的最大值，`(n_cells * n_sps,)`；零填充槽位为 0。"""
    n_var = jac.n_var
    d = xp.zeros(cell_is_prism.size * n_sps, dtype=xp.float64)
    for blocks, cells, n_real in ((jac.blocks_prism, jac.prism_cells, n_real_prism),
                                  (jac.blocks_tet, jac.tet_cells, n_real_tet)):
        if cells.size:
            diag = xp.abs(xp.diagonal(blocks, axis1=1, axis2=2)).reshape(cells.size, n_real, n_var)
            rows = xp.asarray(cells)[:, None] * n_sps + xp.arange(n_real)[None, :]
            d[rows.ravel()] = diag.max(axis=-1).ravel()
    return d


def shifted_dtau(dtau_flat, diag, xp):
    """带位移的预处理伪时间步：`1/dtau_p = max(1/dtau, PRECONDITIONER_SHIFT * diag)`；`diag == 0` 的行
    （零填充槽位）原样。返回与 `dtau_flat` 同形状的新数组。"""
    dtau = xp.asarray(dtau_flat, dtype=xp.float64)
    flat = dtau.ravel()
    floor = PRECONDITIONER_SHIFT * diag
    capped = xp.where(floor * flat > 1.0, 1.0 / xp.where(floor > 0.0, floor, 1.0), flat)
    return capped.reshape(dtau.shape)
