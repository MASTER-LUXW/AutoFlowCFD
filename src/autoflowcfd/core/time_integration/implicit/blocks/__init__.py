"""AutoFlowCFD V2.0 - 隐式推进的单元块预处理。

    block_jacobi.py        单元块预处理缓存（装配、档位、刷新与复用）
    cell_blocks.py         单元块 Jacobian（着色差分装配）
    cell_block_inverse.py  单元块逐块求逆与块 Jacobi 预处理作用
    block_ilu.py           单元块 ILU（多色序、对角修正型 D-ILU）
    block_budget.py        块预处理的内存预算与分档（平均流与湍流共享）
    coloring.py            距离 1 / 距离 2 单元着色与残差模板单元对
    preconditioner.py      伪瞬态（PTC）对角预处理
    precond_shift.py       预处理矩阵的对角位移（伪时间项相对对角块量级的下限）
    refresh_cost.py        块刷新判据的确定性代价模型
"""
