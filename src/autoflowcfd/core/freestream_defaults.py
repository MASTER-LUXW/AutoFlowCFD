"""AutoFlowCFD V2.0 - 默认来流条件与空气分子粘度，全仓库唯一定义。

海平面标准大气的密度与静压、120 km/h 来流速度、标准状态下空气的动力粘度。配置类（`config/solver_config.py`）、
各后端求解器构造参数、CLI 网格生成的 y+ 估计都从这里取默认值。

2026-10-19（I-46）以前这组值在配置类（两份）、四个求解器构造函数、分布式求解器的 `solver_kwargs.get(...)` 与
完全分布式包读取（`package['freestream'].get(...)`）、CLI 各写一份；分布式包读取的缺省值还会在包缺键时静默顶上，
按另一个物理算例运行。旧 checkpoint 缺键时的恢复值（`cli/solve/checkpoint_io/rebuild.py::physics_from_metadata`）
是那时的内置值，属于历史常量，不随这里的默认值改变。
"""

#: 来流密度 kg/m^3（海平面标准大气）
RHO_INF = 1.225

#: 来流速度 m/s（120 km/h）
VEL_INF = 33.33

#: 来流静压 Pa（海平面标准大气）
P_INF = 101325.0

#: 空气分子动力粘度 Pa*s（标准状态）
MU_MOLECULAR = 1.8e-5
