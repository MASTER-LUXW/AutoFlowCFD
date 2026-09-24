"""AutoFlowCFD V2.0 - `AFCFD_FILTER_MODE` 档位解析。

从 `core/fr_solver/filter.py` 拆出（2026-09-24）。纯搬家，逻辑未改。

档位与后端的支持矩阵（`_SENSOR_MODE_SUPPORTED_BACKENDS`）也在这里：
`sensor` 档的平均流门控需要在 RK stage 内部逐 stage 求指标，不是所有后端
都接线过 —— 不静默降级，不支持就报错。
"""

import os


#: 目前实现了 `sensor` 档的后端。`legacy`/`off`/`mild`/`project` 四档
#: **不需要**出现在这里——它们是在算子构造期改 `ops.filter_prism`/
#: `filter_tet` 本身（见 fr/modal_filter.py 的模块级常量），因此对全部
#: 后端自动生效；只有 `sensor` 需要在推进循环里逐单元门控。
#:
#: ## 另外三个后端缺的到底是什么（2026-09-17 查清，写成可执行的规格）
#:
#: 不是"接线没写"，是一处**真实的架构约束**：`bounds`（BJ 型）判据要的是
#: **面邻居的单元均值**，而分区边界上的邻居是 halo 单元；而 RK stage 的
#: 滤波回调 `filter_func(U_flat)` 只拿到**本地** U（`n_local` 个单元），
#: 拿不到 halo。所以补齐它需要把 halo 的单元均值也传进 stage 回调——那是
#: 时间积分器回调契约的改动，不是几行接线。
#:
#: 三个后端各自缺的：
#:   * `cpu-mpi`：面连接数组（`owner_cell_local`/`neighbor_cell_local`/
#:     `is_boundary`）与 `cell_is_prism` 都已就位（见
#:     `mpi/distributed_solver.py` 里 `build_filter_func_by_cell_type` 的
#:     调用处），**只缺 halo 单元均值**。注意索引空间：那些数组是"棱柱
#:     在前"的置换排列，而 `filter_func` 收到的是原生排列，映射是
#:     `native = perm[permuted]`（见 `distributed_flat_face` 的 inv_perm
#:     文档）——接线时必须转，否则会静默用错单元。
#:   * `gpu-single` / `gpu-mpi`：除上面那条外，还缺 `compute_bounds_
#:     violation_mask` 的 CuPy 版——它用 `np.add.at`/`np.maximum.at` 做
#:     scatter 归约，CuPy 没有直接对应（要用 `cupyx.scatter_add` 与
#:     手写的 scatter-max），不是把 np 换成 cp 就行。
#:
#: 在补齐之前，`resolve_filter_mode` 对**默认值**落到 sensor 的情形退到
#: `project` 并打一条量化了数值后果的警告；对**显式**请求仍然报错。
_SENSOR_MODE_SUPPORTED_BACKENDS = ("cpu-single", "cpu-mpi",
                                  "gpu-single", "gpu-mpi")


def resolve_filter_mode(backend: str) -> str:
    """读取 `AFCFD_FILTER_MODE` 并校验当前后端是否支持它。

    存在的理由：`sensor` 档需要在推进循环里逐单元求传感器指示器，不是
    算子构造期就能定下来的，必须逐后端接线。如果未接线的后端只是"读不到
    这个分支所以按 legacy 跑"，同一个环境变量在不同后端就意味着不同的
    数值方案，而且**没有任何提示**——那正是本项目一贯不接受的静默行为
    （同一原则见 gpu_time_integration.py 把"只用第一个 SP"改成显式校验）。

    **当前接线状态（2026-09-18 起全部四条）**：

      cpu-single  `fr_solver/step.py`
      cpu-mpi     `core/mpi/distributed_solver.py::
                  _build_sensor_gated_filter_func_distributed`
      gpu-single  `core/gpu/solver/gpu_solver_init.py::
                  _build_sensor_gated_filter_gpu`
      gpu-mpi     `core/gpu/distributed/gpu_distributed_init.py::
                  _build_sensor_gated_filter_distributed_gpu`

    四条共用**同一个**门控实现 `build_sensor_gated_filter_func_arrays`
    ——它与两个判据内核（Persson-Peraire、BJ 越界）都已改成数组模块
    无关，传 CuPy 矩阵进去整条回调就走 CuPy 的同名函数（BJ 的邻域散射
    归约经 `bounds_sensor._scatter_minmax` 分派到
    `cupyx.scatter_max/scatter_min`）。各后端只提供索引换算与 halo 扩展。

    **两条 GPU 路径的验证边界（必须如实说明）**：本机没有 CUDA/CuPy，
    所以 GPU 分支只能靠"同一份数组模块无关代码用 NumPy 跑"来验证逻辑
    （逐位对照见 `tests/unit/test_sensor_gate_distributed.py` 与
    `test_sensor_gate_gpu_paths.py`）；`cupyx.scatter_max/scatter_min`
    与 `cp.einsum` 这两处 CuPy API 调用本身无法在此执行，需要在真实
    GPU 环境上跑一次交叉验证才算完整确认。

    Args:
        backend: 调用方后端标识，取 `_SENSOR_MODE_SUPPORTED_BACKENDS` 里的
            值或任意其它字符串（如 "cpu-mpi"/"gpu-single"/"gpu-mpi"）

    Returns:
        规范化（小写）后的模式名。

    Raises:
        NotImplementedError: `sensor` 档但 `backend` 不在已接线列表里
            （既包括拼错的后端名，也包括将来新增而忘了接线的后端）。
    """
    # **默认值必须与 `fr/modal_filter.py` 的同一个环境变量解析保持一致**
    # （2026-09-17 真实 bug）：那边定滤波**矩阵**的 sigma，这边定 `step.py`
    # 走不走**逐单元门控**分支。把默认值从 legacy 改成 sensor 时只改了
    # 那一处，于是默认路径变成"矩阵是精确投影、但全局逐 stage 施加"——
    # 功能上等于 legacy（实测 legacy 与 project 在 P1 上逐位相同），
    # 壁面剪应力照样被清零（实测 du/dy 恒为 0）。两处必须同源，所以这里
    # 直接复用那个模块的常量而不是再写一遍默认值。
    from autoflowcfd.fr.modal_filter import FILTER_MODE as _MATRIX_MODE

    _raw = os.environ.get("AFCFD_FILTER_MODE")
    explicit = _raw is not None and _raw.strip() != ""
    mode = (_raw.lower() if explicit else _MATRIX_MODE.lower())

    # 后端名**无条件**校验（2026-09-19）：它是代码级标识，四条真实调用点
    # 传的都是固定字符串，拼错只可能是 bug。此前这条校验寄生在下面那个
    # `mode == "sensor"` 判据里，于是默认值从 `sensor` 改成 `off` 之后
    # 拼错的后端名会**静默放行** —— 正是本项目一贯不接受的那类静默行为。
    if backend not in _SENSOR_MODE_SUPPORTED_BACKENDS:
        raise NotImplementedError(
            f"未知后端标识 {backend!r}；已知的四条是 "
            f"{list(_SENSOR_MODE_SUPPORTED_BACKENDS)}。后端名是代码级标识，"
            f"拼错只可能是 bug，不静默放行。")

    if mode == "sensor" and backend not in _SENSOR_MODE_SUPPORTED_BACKENDS:
        # 无论显式请求还是默认值，一律报错。
        #
        # 此前这里分两支：显式请求报错，默认值退到 `project` 并打警告。
        # 那条退档分支的正当性建立在"默认值必须让每个后端都能跑起来"
        # 之上——而 2026-09-18 起**四条后端全部接线**，它再也不会被
        # 任何真实后端触发，成了死代码。留着它反而有害：将来新增一条
        # 后端而忘了接线时，它会把"默认档在新后端上变成 project"这件事
        # 降级成一条容易被忽略的 warning，而 project 的数值后果是精确
        # 抹掉最高一阶多项式内容（P1 退化成 P0，实测壁面法向速度梯度
        # 从 1734 变成 0）。所以直接报错，逼调用方去接线。
        raise NotImplementedError(
            f"AFCFD_FILTER_MODE=sensor 尚未在后端 '{backend}' 上接线"
            f"（已接线：{', '.join(_SENSOR_MODE_SUPPORTED_BACKENDS)}）。"
            f"传感器门控需要在推进循环里逐单元求传感器指示器，不是算子"
            f"构造期就能定下来的，必须逐后端实现——判据内核与门控实现"
            f"本身是后端无关的（`build_sensor_gated_filter_func_arrays`），"
            f"新后端只需提供索引换算与（分布式时的）halo 扩展。"
            f"legacy/off/mild/project 四档对全部后端都有效。")
    return mode
