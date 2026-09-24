"""AutoFlowCFD V2.0 - 多 GPU 分布式 checkpoint 的保存与加载、状态回传

从 `src/autoflowcfd/core/gpu/distributed/gpu_distributed_init.py` 的 `_GPUDistributedInitMixin` 拆出（2026-09-24，项目「单文件不超
500 行」规范）。mixin 是本仓库既有惯例（`_SolverGeometryMixin`、
`_GPUSolverInitMixin` 等），沿用它而不是另发明一套。

**只含方法，没有状态**：全部属性由 `_GPUDistributedInitMixin` 的 `__init__` 建立，
这里通过 `self` 访问。
"""

from autoflowcfd.core.gpu import get_cupy


class _GPUDistributedCheckpointMixin:
    """多 GPU 分布式 checkpoint 的保存与加载、状态回传"""

    def get_state_cpu(self):
        """下载 GPU 状态到 CPU。"""
        cp = get_cupy()
        return {'U': cp.asnumpy(self.U_gpu)}

    def save_checkpoint_distributed(
        self, output_dir: str, iteration: int, input_file: str,
        order: int, turbulence_model: str, backend: str = "gpu",
        history: dict = None, target_order: int = None,
    ):
        """分布式 checkpoint 保存（#4，V2.0 专家组盲审第4轮，2026-08-28）。

        真实 bug 修复：此前这里调用 `distributed_save_checkpoint(
        U_local_cpu, local_cells, n_global_cell, path, self.rank,
        self.n_ranks)`（6 个位置参数），但该函数真实签名是 `(solver,
        output_dir, iteration, input_file, order, turbulence_model,
        backend, history=None)`（第一个是完整的 solver 对象，不是裸
        数组，且读取 `solver.state.U`/`solver.partition.n_local_cells/
        n_global_cells/local_cells`——CPU `DistributedFRSolver` 的接口
        形状）——参数个数和类型都对不上，且即便签名对齐，
        `MultiGPUDistributedSolver` 当时还有更深层的架构缺陷
        （`self.U_gpu` 按全局单元数分配、不是真正的"只存 local cells"），
        使得 checkpoint 即便"看起来跑通"也不会是真正的分布式局部状态。
        这两个前提现在都已经解决（见 __init__ 里 `self.U_gpu`/
        `self.mesh_data` 按 local+halo 压缩索引空间重构的说明）：这里
        先把 `self.U_gpu`（GPU，local cells）下载并同步进
        `self.state.U`（`DistributedFRState` 已在 __init__ 构造好但
        此前从未真正被写入过），再按真实签名调用
        `distributed_save_checkpoint`。

        Args:
            output_dir: 输出目录
            iteration: 当前迭代数
            input_file: 原始网格文件路径
            order: checkpoint 保存那一刻 `U_sps` 字段实际对应的阶数——
                Order Continuation 接入多GPU分布式路径后（2026-09-02
                续接，见 gpu_distributed.py::solve 文档），调用方必须传
                `self.current_order`（不是固定的目标阶数），否则爬坡
                阶段中途存的 checkpoint 会出现"metadata 记的阶数与
                `U_sps` 实际形状不符"的错配——与 CPU
                `distributed_save_checkpoint` 同名参数同一处修复同一个
                理由。
            turbulence_model: 湍流模型名（目前恒为 'none'，见 __init__）
            backend: 后端名，默认 'gpu'
            history: 收敛历史（可选）
            target_order: Order Continuation 的最终目标阶数
                （`self.order`）。None（默认，兼容旧调用方）时回退到
                `order` 本身。

        Returns:
            checkpoint 文件路径（仅 root rank 有值）
        """
        cp = get_cupy()
        from autoflowcfd.core.mpi.distributed_checkpoint import distributed_save_checkpoint

        n_local = self.partition.n_local_cells
        self.state.U[:n_local] = cp.asnumpy(self.U_gpu)

        return distributed_save_checkpoint(
            self, output_dir, iteration, input_file, order, turbulence_model, backend,
            history=history, target_order=target_order,
        )

    def load_checkpoint_distributed(self, path: str):
        """分布式 checkpoint 加载（#4）。见 save_checkpoint_distributed
        文档说明同一处修复。加载结果写回 `self.state.U`/`self.U_gpu`
        （local cells 部分）。

        Returns:
            (metadata, iteration)：见 distributed_load_checkpoint 返回值
        """
        cp = get_cupy()
        from autoflowcfd.core.mpi.distributed_checkpoint import distributed_load_checkpoint

        U_local, metadata, iteration = distributed_load_checkpoint(path, self)
        n_local = self.partition.n_local_cells
        self.state.U[:n_local] = U_local
        with cp.cuda.Device(self.device_id):
            self.U_gpu = cp.asarray(U_local)
        self.iteration = iteration
        return metadata, iteration
