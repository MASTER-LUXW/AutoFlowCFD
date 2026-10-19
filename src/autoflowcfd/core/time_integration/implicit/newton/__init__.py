"""AutoFlowCFD V2.0 - 隐式推进的 Newton 外步与伪时间控制。

    jfnk.py            Newton 外步（物理性限幅、残差接受判据、dtau 缩档）
    mean_flow_step.py  平均流 Newton 步（稳态 NK 的单步入口、跨步状态与线性增长记录）
    coupled_step.py    平均流 + 湍流紧耦合 Newton 步
    physical_time.py   双时间步（BDF）的隐式内迭代
    globalization.py   残差范数、线性求解相对残差上限、步接受判据
    physicality.py     物理性松弛与逐行局部伪时间步长
    dtau_control.py    `dtau` 缩放（由 Newton 步成败驱动）
    forcing.py         inexact-Newton 的 Eisenstat-Walker forcing term
"""
