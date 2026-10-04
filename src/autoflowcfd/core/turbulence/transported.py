"""AutoFlowCFD V2.0 - 带输运方程的湍流模型的统一接口（SST k-ln(omega)、SA-neg）。

## 为什么要有它（2026-10-04）

紧耦合 Newton（`time_integration/implicit/coupled_step.py`）、四个后端的耦合适配器、湍流解析
块 Jacobian、分布式视图此前都把"湍流未知量 = (k, ln omega)"写死：每个后端各自把
`k_field`、`log_omega(omega_field)` 拼成第 6、7 列，各自按 `omega_from_log(.., omega_max)` 写回，
各自列一遍要在试探求值后恢复的模型缓存。引入第二个输运模型（SA-neg，一个未知量）时这些都要
再写一份。这里把"模型的输运场 <-> Newton 未知量"的映射收到模型自己身上，通用机制只认本接口。

## 接口（模型类继承 `TransportedTurbulence` 并给出三个类属性与两个钩子）

    TRANSPORTED_FIELDS   被输运场的属性名（物理量形式，例如 ("k_field", "omega_field")）
    NEWTON_LOG_COLUMNS   Newton 未知量里取对数的列（物理性限幅按对数量处理，见
                         `time_integration/implicit/physicality.py::ScaledFieldRowLimits`）
    CACHED_ATTRS         源项求值刷新的模型缓存（涡粘等）：试探求值之后必须恢复，否则
                         试探场会泄漏进平均流用的涡粘
    _unknown_from_field(j, field, xp)   第 j 个场 -> 第 j 列未知量
    _field_from_unknown(j, column, xp)  第 j 列未知量 -> 第 j 个场
    unknown_scales()     各列未知量的尺度（物理性限幅的尺度下限与差分步长）
    update_fields(dt, sources, transports)   显式路径的一步场更新（含正性/上界限幅）
    apply_positivity_limiter()               非有限值恢复与上界

各后端的湍流求值返回 `TurbulenceRates`：逐个未知量的源项部分与输运部分分开给出（显式更新
只对源项做点隐式阻尼），隐式路径取两者之和。

未知量按行排列（每行一个解点，与平均流 5 列拼成耦合未知量），数组模块由调用方给出
（numpy 或 cupy，与模型场所在的模块一致）。
"""

from typing import NamedTuple, Optional, Tuple


class TurbulenceRates(NamedTuple):
    """一次湍流求值的结果（逐个 Newton 未知量，`d(未知量)/dt` 量纲）。

    Attributes:
        raw: 模型源项本身（带 rho，诊断与显式路径的返回值用）
        source: 源项部分（已换算到未知量的时间导数）
        transport: 输运部分（对流 + 扩散 + 变换带出的逐点项），没有输运时为 None
    """
    raw: tuple
    source: tuple
    transport: Tuple[Optional[object], ...]

    def total(self) -> tuple:
        """隐式路径用的完整时间导数（源项 + 输运）。"""
        return tuple(s if t is None else s + t for s, t in zip(self.source, self.transport))


class TransportedTurbulence:
    """带输运方程的湍流模型的公共部分（只含方法，状态由具体模型的 `__init__` 建立）。"""

    TRANSPORTED_FIELDS: Tuple[str, ...] = ()
    NEWTON_LOG_COLUMNS: Tuple[int, ...] = ()
    CACHED_ATTRS: Tuple[str, ...] = ()

    @property
    def n_transported(self) -> int:
        """被输运的标量个数（耦合 Newton 里湍流子系统的列数）。"""
        return len(self.TRANSPORTED_FIELDS)

    def _unknown_from_field(self, j: int, field, xp):
        raise NotImplementedError

    def _field_from_unknown(self, j: int, column, xp):
        raise NotImplementedError

    def unknown_scales(self) -> Tuple[float, ...]:
        raise NotImplementedError

    def update_fields(self, dt, sources, transports) -> None:
        raise NotImplementedError

    def apply_positivity_limiter(self) -> None:
        raise NotImplementedError

    def transported_fields(self) -> tuple:
        """各输运场的当前值（物理量形式，按 `TRANSPORTED_FIELDS` 顺序）。"""
        return tuple(getattr(self, name) for name in self.TRANSPORTED_FIELDS)

    def set_transported_fields(self, fields) -> None:
        for name, value in zip(self.TRANSPORTED_FIELDS, fields):
            setattr(self, name, value)

    def newton_unknowns(self, xp):
        """`(n_rows, n_transported)` 的 Newton 未知量（每行一个解点）。"""
        cols = [self._unknown_from_field(j, field, xp).ravel()
                for j, field in enumerate(self.transported_fields())]
        return xp.stack(cols, axis=1)

    def set_newton_unknowns(self, x, xp) -> None:
        """由 `(n_rows, n_transported)` 的未知量写回各输运场（形状沿用当前场）。"""
        shape = getattr(self, self.TRANSPORTED_FIELDS[0]).shape
        self.set_transported_fields(
            [self._field_from_unknown(j, xp.ascontiguousarray(x[:, j]), xp).reshape(shape)
             for j in range(self.n_transported)])

    def field_snapshot(self):
        """输运场与源项缓存的快照（试探求值之后用 `field_restore` 还原）。"""
        return (self.transported_fields(),
                {a: getattr(self, a) for a in self.CACHED_ATTRS if hasattr(self, a)})

    def field_restore(self, snap) -> None:
        fields, cache = snap
        self.set_transported_fields(fields)
        for a, v in cache.items():
            setattr(self, a, v)
