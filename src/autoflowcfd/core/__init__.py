"""AutoFlowCFD V2.0 核心：FR 求解器与它的数值部件。

包导入不加载任何子模块，按需从子包导入（2026-10-22 I-123 以前这里预导入 FRSolver、湍流模型等：顶层包改为延迟
导出 API 后，新进程里先导入 `autoflowcfd.boundary` 会经由这份预导入回头导入尚未初始化完的 `boundary.fr_ghost_state`，
循环导入失败）：

    fr_solver/          单机 CPU 求解器（FRSolver）
    fr_residual/        FR 残差与解析 Jacobian
    fr_operators/       通量 / 梯度 / 收缩等逐点与批量核
    turbulence/         湍流模型（SA、SST、DES、WALE、壁面模型）
    time_integration/   显式 / 时间精确 / 隐式 Newton-Krylov、自适应 CFL、正性限制器
    utils/              Order Continuation、壁面距离、checkpoint、预处理等共用工具
    gpu/                单 GPU / 多 GPU 后端
    mpi/                MPI 分布式后端
"""
