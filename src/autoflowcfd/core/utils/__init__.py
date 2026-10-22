"""AutoFlowCFD Core - 共用工具：checkpoint、Order Continuation、面图着色、壁面距离、预处理、数组模块等。

包导入不加载任何子模块（叶子工具如 `array_module` 被边界条件等底层模块使用；2026-10-22 I-123 以前这里预导入
Order Continuation，经由它拉起整个求解器，底层模块先导入时循环导入失败），按需从子模块导入。
"""
