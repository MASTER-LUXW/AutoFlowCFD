"""稳态/瞬态求解 CLI 命令 —— 从 solve_commands.py 拆出，控制单文件行数。

本文件原来直接包含 `solve steady`/`solve transient` 两个 click 子命令的
完整实现（510 行，超过 400 行硬性拆分阈值），现在按功能进一步拆开：
    - solve_aero_coefficients.py: 两个命令共用的气动系数计算/打印、
      参考面积自动估算辅助函数
    - solve_steady_command.py: `solve steady` 命令本体
    - solve_transient_command.py: `solve transient` (DES/LES) 命令本体

本文件现在只是一个薄的重新导出入口：
    - 导入两个命令模块以触发它们的 `@solve.command()` 注册（副作用导入，
      与 solve_commands.py 导入本文件的方式一致）
    - 重新导出 `_report_aerodynamic_coefficients`，因为 solve_commands.py
      的 `resume` 命令直接从 `autoflowcfd.cli.solve_steady_commands` 导入
      这个符号，外部调用方式不应因为这次拆分而改变
"""

from autoflowcfd.cli.solve_aero_coefficients import (  # noqa: F401
    _compute_reference_area_auto,
    _report_aerodynamic_coefficients,
)

# 导入子命令模块，触发 @solve.command() 注册
from autoflowcfd.cli import solve_steady_command  # noqa: F401
from autoflowcfd.cli import solve_transient_command  # noqa: F401
