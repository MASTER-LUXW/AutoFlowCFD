"""AutoFlowCFD V2.0 - 单 GPU 求解器的主机视图：让 CPU 侧的 checkpoint 读写、`--init-from`、结果保存与气动力系数
直接作用于 `GPUFRSolver`。

单 GPU 此前只有 `solve steady --backend gpu` 走 `GPUFRSolver`，且不写中间 checkpoint、最终状态不含湍流场；瞬态、
续算与 Python API 的 `backend='gpu'` 走 `FRSolver` 的"只把无粘残差搬到 GPU"分支（2026-10-04 统一）。CPU 侧
这些函数读的是 `solver.state.U/Q`、`solver.turb_model`（输运场、涡粘）、`solver.sgs_model`、`solver.mesh` 等主机
属性，这里给出同名属性：

* 设备上的状态、湍流场与亚格子涡粘按需拉到主机（`state`、`turb_model`、`sgs_model` 是主机副本）；其余属性
  （网格、来流、阶数、时间积分器、物理参数、累计伪时间……）原样转发到求解器；对视图设置其它属性也写到求解器上
  （恢复路径要写 `_turb_ramp_step`、`tau_accum` 等跨步状态）；
* 恢复类操作作用在视图上之后调用 `push()`：平均流写回设备，湍流场经 **GPU 模型自己的** `restore_transported`
  写回（SST 的 omega 可容许性投影、SA-neg 的壁面解点置零都在里面），涡粘与产生项斜坡因子随之写回。
"""

from types import SimpleNamespace

import numpy as np

from autoflowcfd.core.gpu import get_cupy


def _host(a):
    return np.asarray(a.get() if hasattr(a, "get") else a)


class _HostTurbulence:
    """输运湍流模型的主机副本：checkpoint 写出与单元平均读它的输运场、涡粘与输出键名；恢复路径的
    `restore_transported` 记下新场，由 `GPUHostView.push` 交给 GPU 模型。"""

    def __init__(self, model):
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_restored", None)
        for name in model.TRANSPORTED_FIELDS:
            object.__setattr__(self, name, _host(getattr(model, name)).copy())
        object.__setattr__(self, "nu_t", None if model.nu_t is None else _host(model.nu_t).copy())
        object.__setattr__(self, "production_factor", float(model.production_factor))

    def __getattr__(self, name):
        return getattr(self._model, name)        # 常数、TRANSPORTED_FIELDS、OUTPUT_FIELD_KEYS ...

    def transported_fields(self) -> tuple:
        return tuple(getattr(self, name) for name in self._model.TRANSPORTED_FIELDS)

    def restore_transported(self, fields, source: str = "checkpoint") -> None:
        fields = [np.asarray(f) for f in fields]
        object.__setattr__(self, "_restored", (fields, source))
        for name, f in zip(self._model.TRANSPORTED_FIELDS, fields):
            object.__setattr__(self, name, f)


class GPUHostView:
    """`GPUFRSolver` 的主机视图（见模块文档）。

    `state` / `turb_model` / `sgs_model` 在首次访问时才从设备拷回（求解循环每步算 Cd/Cl 只读 `state.Q`，
    不必把湍流场一起拷回）；`push()` 只写回取过的部分。
    """

    _HOST_ATTRS = ("state", "turb_model", "sgs_model")

    def __init__(self, solver):
        object.__setattr__(self, "_solver", solver)

    def __getattr__(self, name):
        if name in GPUHostView._HOST_ATTRS:
            value = getattr(self, "_pull_" + name)()
            object.__setattr__(self, name, value)
            return value
        return getattr(self._solver, name)

    def __setattr__(self, name, value):
        if name in GPUHostView._HOST_ATTRS:
            object.__setattr__(self, name, value)
        else:
            setattr(self._solver, name, value)

    def _pull_state(self):
        s = self._solver
        U = _host(s.U_gpu).copy()
        n_cells, n_sps, n_vars = U.shape
        return SimpleNamespace(U=U, Q=_host(s.Q_gpu).copy(), n_cells=n_cells, n_sps=n_sps, n_vars=n_vars,
                               _update_primitives=self._update_primitives)

    def _pull_turb_model(self):
        model = self._solver.turb_model_gpu
        return _HostTurbulence(model) if getattr(model, "TRANSPORTED_FIELDS", ()) else None

    def _pull_sgs_model(self):
        sgs = self._solver.sgs_model_gpu
        return None if sgs is None else SimpleNamespace(nu_t=None if sgs.nu_t is None else _host(sgs.nu_t))

    def _update_primitives(self) -> None:
        from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive

        st = self.state
        st.U = np.ascontiguousarray(st.U)
        st.Q = conserved_to_primitive(st.U[..., :5])

    def _get_turbulent_viscosity_field(self, mu_t_turb=None):
        """动力涡粘 `rho * nu_t`（湍流模型 + 亚格子模型；与 CPU 同名方法同一个函数）。"""
        from autoflowcfd.core.fr_solver.turbulence.corrections import get_turbulent_viscosity_field

        return get_turbulent_viscosity_field(self) if mu_t_turb is None else mu_t_turb

    def push(self) -> None:
        """把视图上（恢复后）的状态写回设备（见模块文档）。"""
        cp = get_cupy()
        s = self._solver
        pulled = self.__dict__
        with cp.cuda.Device(s.device_id):
            if "state" in pulled:
                s.U_gpu = cp.asarray(np.ascontiguousarray(self.state.U))
                s._update_primitives_gpu()
            ht = pulled.get("turb_model")
            if ht is not None:
                model = s.turb_model_gpu
                if ht._restored is not None:
                    fields, source = ht._restored
                    model.restore_transported([cp.asarray(f) for f in fields], source=source)
                if ht.nu_t is not None:
                    model.nu_t = cp.asarray(ht.nu_t)
                model.production_factor = ht.production_factor
