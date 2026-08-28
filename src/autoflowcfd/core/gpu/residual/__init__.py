"""GPU 残差计算核（无粘/粘性通量、体积项张量收缩、物理梯度、P0 kernel）。

对应 CPU 侧 core/fr_residual/ + core/fr_operators/ 的 GPU 独立实现
（CuPy 矢量化，非 numba，两者不共用代码，靠交叉一致性测试保证一致）。
"""
