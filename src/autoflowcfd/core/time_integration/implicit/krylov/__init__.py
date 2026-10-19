"""AutoFlowCFD V2.0 - 隐式推进的 Krylov 线性求解。

    gmres.py            右预处理 GMRES（灵活模式、单精度基存储）
    linear_solve.py     Newton 步里的线性求解（重启长度、真实行压缩、加权范数、Krylov 基内存）
    jacobian_vector.py  Fréchet 差分的矩阵自由 `J v`
"""
