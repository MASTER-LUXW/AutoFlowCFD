"""AutoFlowCFD V2.0 - 紧耦合 Newton 后端适配器（`coupled_step.py` 模块文档"后端适配器"）中与后端无关的部分。

* `CoupledBackend`：四个后端（单机 CPU / 单 GPU / CPU-MPI / 多 GPU）共用的 `install`（写入新状态、湍流正性限幅、
  在新状态上刷新涡粘）与 `cell_colors`（取隐式湍流适配器的着色，两套块与耦合图同一份）；
* `SingleNodeCoupledBackend`：单机两个后端另外共用的平均流解析块装配器（主机上装配，GPU 的块由缓存上传）与
  "没有全局粗校正"（本地多层预处理的最粗层就是全局的）。

2026-10-22（I-214）以前这几个方法在四个适配器里逐字各写一份。
"""

from typing import Any


class CoupledBackend:
    """紧耦合适配器的公共部分。子类在 `__init__` 里设置 `solver`、`turb`（隐式湍流适配器）与协议其余属性。"""

    __slots__ = ()

    turb: Any                   # 隐式湍流适配器（子类的槽位）

    def set_trial(self, x) -> None:
        """写入试探状态（平均流 5 列 + 湍流未知量经模型写回），各后端实现。"""
        raise NotImplementedError

    def cell_colors(self):
        return self.turb.cell_colors()

    def install(self, x) -> None:
        """写入新状态，做湍流正性限幅，并在新的平均流与湍流场上刷新涡粘（分布式后端同时留下 compact 一份
        `turb.mu_t_compact` 供下一步粘性步长），供本步之后的一切消费方使用。"""
        self.set_trial(x)
        turb = self.turb
        turb.positivity()
        turb.prepare_inputs()
        turb.rates(commit=True)


class SingleNodeCoupledBackend(CoupledBackend):
    """单机（CPU / 单 GPU）适配器的公共部分；子类另设 `order`、`n_sps`、`_nu_av`（本步冻结的人工扩散系数）。"""

    __slots__ = ()

    def mean_assembler(self, mu_t):
        """平均流解析单元块装配器（`fr_residual/jacobian`）；阶数不在解析装配覆盖范围内时 None（退回差分块）。"""
        from autoflowcfd.core.fr_residual.jacobian.backend import MeanFlowBlockAssembler, unsupported_reason

        s = self.solver
        if unsupported_reason(order=self.order):
            return None
        return MeanFlowBlockAssembler(
            mesh=s.mesh, ops=s.ops, ghost_provider=s.boundary_ghost_provider, mu=s.mu_molecular,
            mach_ref=s.freestream["mach_ref"], low_mach=s.low_mach_precond_enabled, mu_t=mu_t,
            nu_av=self._nu_av, n_sps=self.n_sps, wall_model=s.wall_model)

    def coarse_context(self):
        return None             # 单机：本地多层预处理的最粗层就是全局的
