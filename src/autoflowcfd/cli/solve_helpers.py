"""solve 命令组的共享辅助函数 —— 重新导出入口。

本文件原来直接包含网格加载、壁面距离场计算、结果/checkpoint 持久化五个
辅助函数（456 行，超过 400 行硬性拆分阈值），现在按功能进一步拆开：
    - solve_mesh_loader.py: load_mesh_for_solver
    - solve_wall_distance.py: compute_wall_distance_for_solver
    - solve_checkpoint_io.py: save_results / restore_state_from_checkpoint /
      write_checkpoint

本文件现在只是一个薄的重新导出入口：外部代码（solve_commands.py、
solve_steady_command.py、solve_transient_command.py、
core/mpi/distributed_mesh_loader.py 等）一律仍从
`autoflowcfd.cli.solve_helpers` 导入即可，不需要关心内部是怎么拆的。
"""

from autoflowcfd.cli.solve_mesh_loader import load_mesh_for_solver  # noqa: F401
from autoflowcfd.cli.solve_wall_distance import compute_wall_distance_for_solver  # noqa: F401
from autoflowcfd.cli.solve_checkpoint_io import (  # noqa: F401
    save_results,
    restore_state_from_checkpoint,
    rebuild_solver_from_checkpoint,
    write_checkpoint,
)
