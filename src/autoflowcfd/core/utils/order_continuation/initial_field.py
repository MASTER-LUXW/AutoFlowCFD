"""AutoFlowCFD V2.0 - 从 checkpoint 场起步（`solve transient --init-from`，四个后端共用的唯一实现）。

两件事，2026-10-08 以前都不对：

1. **阶数。** 求解器按目标阶数构造，checkpoint 是另一个阶数写的（典型：稳态在 P1 阶段存的 checkpoint，瞬态目标
   P2）时，恢复因解点数不符被拒绝。低阶多项式本身就是高阶多项式，可以精确延拓——Order Continuation 升阶用的
   就是同一个算子（`fr/order_interp.py`）。这里把 checkpoint 的逐解点字段（平均流、湍流输运场、涡粘）延拓到
   目标阶数；湍流场在模型的未知量空间里延拓（SST 的 `ln omega`，在物理 omega 上延拓会在单元内下冲出负值）。
   checkpoint 阶数高于目标阶数时拒绝：降阶不是精确操作，会丢掉高阶分量。
2. **起步方式。** 恢复完之后 `solve()` 照样走 Order Continuation：它不知道状态来自 checkpoint，于是重置回 P0
   均匀来流从头爬坡——init-from 的初场被静默丢掉（`solve resume` 不受影响，那条路径标记了
   `_resumed_from_checkpoint`）。从 checkpoint 场起步时直接在目标阶数上推进：爬坡是为从均匀流起步准备的，已发展
   的流场不需要；对时间精确的瞬态，低阶上推进的那段物理时间也没有意义。
"""

from typing import Tuple

import numpy as np


def prolongate_checkpoint_fields(fields: dict, target_order: int, cell_is_prism, model) -> Tuple[dict, int]:
    """把 checkpoint 的逐解点字段从写出时的阶数精确延拓到 `target_order`。

    Args:
        fields: checkpoint 字段（`U_sps` 必须有；湍流输运场按 `model.TRANSPORTED_FIELDS` 取，`nu_t` 有则一并延拓）
        target_order: 目标阶数
        cell_is_prism: `(n_cells,)` 布尔掩码（checkpoint 的单元顺序；棱柱与四面体各用自己的延拓矩阵）
        model: 求解器的输运湍流模型（没有时 None），提供未知量空间的映射 `mapped_fields`

    Returns:
        `(字段字典, checkpoint 阶数)`；阶数相同时原样返回字典。

    Raises:
        ValueError: 单元数不符，或 checkpoint 阶数高于目标阶数
    """
    from autoflowcfd.fr.native_padding import order_from_n_sps
    from autoflowcfd.fr.order_interp import apply_order_interp

    U = np.asarray(fields["U_sps"])
    is_prism = np.asarray(cell_is_prism, dtype=bool)
    if U.ndim != 3 or U.shape[0] != is_prism.shape[0]:
        raise ValueError(f"状态形状 {U.shape} 与求解器的单元数 {is_prism.shape[0]} 不符（网格已变化），拒绝恢复")
    ckpt_order = order_from_n_sps(U.shape[1])
    if ckpt_order == target_order:
        return fields, ckpt_order
    if ckpt_order > target_order:
        raise ValueError(
            f"checkpoint 是 P{ckpt_order}，高于目标阶数 P{target_order}：降阶不是精确操作（会丢掉高阶分量），"
            f"请用 --order >= {ckpt_order}")

    def lift(field):
        field = np.asarray(field)
        out = np.empty((field.shape[0], (target_order + 1) ** 3) + field.shape[2:])
        for mask, n_prism in ((is_prism, None), (~is_prism, 0)):
            if mask.any():
                part = field[mask]
                out[mask] = apply_order_interp(part, part.shape[0] if n_prism is None else 0,
                                               ckpt_order, target_order)
        return out

    out = dict(fields)
    out["U_sps"] = lift(U)
    names = tuple(getattr(model, "TRANSPORTED_FIELDS", ())) if model is not None else ()
    if names and all(name in fields for name in names):
        lifted = model.mapped_fields(lift, np, fields=[np.asarray(fields[name]) for name in names])
        out.update(zip(names, lifted))
    if "nu_t" in fields:
        out["nu_t"] = lift(fields["nu_t"])
    return out, ckpt_order


def start_from_checkpoint_field(solver, checkpoint_order: int, report: bool = True) -> None:
    """恢复完 checkpoint 场之后调用：延拓过时在目标阶数的点集上做守恒的正性限制（与 Order Continuation 升阶后
    同一个接口，低阶多项式只在低阶那组点上保证可容许），并让 `solve()` 直接在目标阶数上推进。"""
    if checkpoint_order != solver.order:
        solver._limit_prolongated_state()
        if report:
            print(f"   已将 checkpoint 场由 P{checkpoint_order} 精确延拓到 P{solver.order}")
    solver.order_continuation_enabled = False
    if report:
        print(f"   直接在 P{solver.order} 上推进（初场来自 checkpoint，不做阶数爬坡）")
