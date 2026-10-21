"""checkpoint 读取（`CheckpointManager.load` 的实现）：布局校验、旧布局迁移、收敛历史。

布局版本（写出端 `checkpoint.py::CheckpointManager.save` 记录）：

* 棱柱基 `prism_basis`：不是原生基的 checkpoint 硬失败（`_check_prism_basis`）；
* 解点存储宽度 `sp_layout`：缺失即 2026-10-19 之前的 `(p+1)^3` 宽度，精确截到当前宽度（`_migrate_legacy_sp_width`）；
* 四面体节点集 `tet_node_set`：P4+ 不一致硬失败（`_check_tet_node_set`）。

2026-10-19（I-86）删除配置哈希比对与假的跨后端转换（见 `checkpoint.py` 模块文档）；读失败不再统一改包成
RuntimeError，原始异常（布局不符的 ValueError、文件缺失的 FileNotFoundError……）直接抛给调用方。
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import h5py
import numpy as np
from loguru import logger


def decode_attr(value):
    """把 HDF5 字节串属性解码成 Python 字符串。

    2026-09-20 从 `load_checkpoint` 内部提到模块级：`_check_prism_basis`
    也要用它，两处各写一份就成了两个事实来源。
    """
    if isinstance(value, bytes):
        return value.decode('utf-8')
    return value


#: checkpoint 记录的棱柱基（唯一实现是原生基）
PRISM_BASIS = "native"


def _check_prism_basis(meta_group, checkpoint_path) -> None:
    """校验 checkpoint 是原生棱柱基下写的，否则硬失败。

    没有 `prism_basis` 属性（坍缩坐标棱柱基时期写的）或值不是 `"native"` 的 checkpoint，解点布局与原生基不同
    （同一个解点数组在两条基下含义不同），续算会把它重解释成错解，所以拒绝而不是告警。与 `config_hash` 不一致
    （"可能影响结果"，只告警）不是同一级别：这里是"数据的含义变了"。
    """
    if 'prism_basis' not in meta_group.attrs:
        raise ValueError(
            f"Checkpoint {checkpoint_path} 没有记录棱柱基：它是坍缩坐标棱柱基时期写的，解点布局与现行原生基"
            f"不同，不能续算——请在当前版本下从头开始计算。")

    saved = decode_attr(meta_group.attrs['prism_basis'])
    if saved != PRISM_BASIS:
        raise ValueError(
            f"Checkpoint {checkpoint_path} 是在棱柱基 {saved!r} 下写的，"
            f"当前只有原生基（{PRISM_BASIS!r}）。两条基的每单元解点布局不同（原生基有零填充槽位），"
            f"跨基 resume 会静默重解释同一份数组、给出看不出异常的错解。只能在原生基下从头开始。")


#: 解点存储宽度的版本：每单元 `sps_width(order)` 个槽位（`fr/native_padding.py`）。没有这个属性的 checkpoint
#: 是 2026-10-19 之前写的，宽度为 `(order+1)^3`。
SP_LAYOUT = "native_dof"


def _solution_layout(meta_group, u_width: int) -> Tuple[int, Optional[int]]:
    """由 `U_sps` 的每单元槽位数给出 `(解的阶数, 旧布局宽度)`；当前布局时后者为 None。

    没有 `sp_layout` 属性的是 2026-10-19 之前写的 checkpoint，宽度为 `(p+1)^3`（旧宽度 8/27/64 与任何
    `sps_width` 都不相等，判别无歧义）；P0 两种布局宽度相同（均为 1），不需要迁移。
    """
    from autoflowcfd.fr.native_padding import order_from_n_sps, sps_width

    if 'sp_layout' in meta_group.attrs:
        saved = decode_attr(meta_group.attrs['sp_layout'])
        if saved != SP_LAYOUT:
            raise ValueError(f"checkpoint 的解点存储布局 {saved!r} 不认识（当前为 {SP_LAYOUT!r}）")
        return order_from_n_sps(u_width), None
    order = round(u_width ** (1.0 / 3.0)) - 1
    if order < 0 or (order + 1) ** 3 != u_width:
        raise ValueError(f"旧布局 checkpoint 的 U_sps 每单元解点数 {u_width} 不是 (p+1)^3，无法迁移")
    return order, (u_width if sps_width(order) != u_width else None)


def _migrate_legacy_sp_width(arrays: dict, metadata: dict, order: int, legacy: Optional[int], cut) -> None:
    """把 2026-10-19 之前写的 checkpoint 的逐解点字段从 `(p+1)^3` 宽度截到 `sps_width(p)`（就地改 `arrays`）。

    旧布局里每单元前 `real_sps_per_cell` 个槽位是自由度、其余是填充，新宽度 `sps_width(p)` 不小于两类单元的
    真实自由度数，所以截掉的只有填充槽位，截取是精确的。SP 轴是第 1 轴、长度等于旧宽度的都是逐解点字段
    （U_sps、输运湍流场、nu_t、dual-time 上一时间层）；单元平均与 `tau_accum` 是一维，都不动。

    Args:
        arrays: 名字 -> 数组（`load_checkpoint`）或形状（`load_checkpoint_header`）
        cut: `(值, 新宽度) -> 截取后的值`
    """
    from autoflowcfd.fr.native_padding import sps_width

    if legacy is None:
        return
    width = sps_width(order)
    for name, value in arrays.items():
        shape = value if isinstance(value, tuple) else value.shape
        if len(shape) >= 2 and shape[1] == legacy:
            arrays[name] = cut(value, width)
    if 'n_sps_per_cell' in metadata:
        metadata['n_sps_per_cell'] = width
    logger.info(f"旧布局 checkpoint（P{order}，每单元 {legacy} 个解点槽位）已截为 {width} 个（只去掉填充槽位）")


def _check_tet_node_set(meta_group, order: int, checkpoint_path) -> None:
    """P4 及以上：checkpoint 的四面体节点集必须与当前一致，否则硬失败。

    2026-10-17 改正了 Warp & Blend 优化参数的下标（`fr/warp_blend_nodes.py`），P4+ 四面体解点位置随之改变；
    节点值在旧、新节点上含义不同，形状恰好相同、不会报错，续算会静默重解释成错解。P1~P3 的节点没有变，不检查。

    2026-10-19 修复：此前按 `conserved` 推阶数，而那是单元平均 `(n_cells, n_vars)`、恒为二维，这道检查从未生效。
    """
    from autoflowcfd.fr.warp_blend_nodes import TET_NODE_SET

    if order < 4:
        return
    saved = decode_attr(meta_group.attrs['tet_node_set']) if 'tet_node_set' in meta_group.attrs else None
    if saved != TET_NODE_SET:
        raise ValueError(
            f"Checkpoint {checkpoint_path} 的四面体节点集是 {saved!r}，当前是 {TET_NODE_SET!r}："
            f"P4 及以上的四面体解点位置不同，续算会把解点值静默重解释到另一组节点上（形状相同、不会报错）。"
            f"请从低阶 checkpoint 延拓（--init-from 低阶 checkpoint）或从头开始。")


#: 读取时不放进 metadata 的属性：2026-10-19 之前写的配置哈希（已删除，无意义）
_NOT_METADATA = ("config_hash",)


def _open(checkpoint_path) -> Path:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    return checkpoint_path


def _read_metadata(f, checkpoint_path) -> Tuple[int, dict]:
    meta_group = f["metadata"]
    _check_prism_basis(meta_group, checkpoint_path)
    metadata = {key: decode_attr(meta_group.attrs[key]) for key in meta_group.attrs.keys() if key not in _NOT_METADATA}
    return int(meta_group.attrs['iteration']), metadata


def _read_history(f) -> dict:
    conv_group = f["convergence/history"]
    history = {}
    if 'iterations' in conv_group:
        history['iterations'] = conv_group['iterations'][:].tolist()
    for group_name in ('residuals', 'coefficients'):
        if group_name in conv_group:
            history[group_name] = {name: conv_group[group_name][name][:].tolist()
                                   for name in conv_group[group_name].keys()}
    if 'cfl_history' in conv_group:
        history['cfl_history'] = conv_group['cfl_history'][:].tolist()
    if 'statistics' in f:
        stats_group = f['statistics']
        history['statistics'] = {name: stats_group[name][:] for name in stats_group.keys()}
        history['statistics'].update({name: stats_group.attrs[name] for name in stats_group.attrs.keys()})
    return history


def _apply_layout(f, metadata: dict, arrays: dict, cut, checkpoint_path) -> None:
    """布局迁移与 P4+ 节点集校验（`arrays` 为空，即 checkpoint 没有逐解点状态时不做）。"""
    if 'U_sps' not in arrays:
        return
    u = arrays['U_sps']
    order, legacy = _solution_layout(f["metadata"], int((u if isinstance(u, tuple) else u.shape)[1]))
    _migrate_legacy_sp_width(arrays, metadata, order, legacy, cut)
    _check_tet_node_set(f["metadata"], order, checkpoint_path)


def load_checkpoint(checkpoint_path: Union[str, Path]) -> Tuple[np.ndarray, dict, int, dict]:
    """读取 HDF5 checkpoint。

    Returns:
        `(solution, history, iteration, metadata)`：solution 为单元积分平均 `(n_cells, n_vars)`；history 为收敛历史
        字典；metadata 为写出端记录的全部属性（含 timestamp 与布局版本），逐解点 / 逐单元数据集在 `metadata['fields']`

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 布局不符（棱柱基、解点存储布局、P4+ 四面体节点集）
    """
    checkpoint_path = _open(checkpoint_path)
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    with h5py.File(checkpoint_path, 'r') as f:
        iteration, metadata = _read_metadata(f, checkpoint_path)
        sol_group = f["solution"]
        solution = sol_group["conserved"][:]
        fields = {name: sol_group[name][:] for name in sol_group.keys() if name != "conserved"}
        _apply_layout(f, metadata, fields, lambda a, w: np.ascontiguousarray(a[:, :w]), checkpoint_path)
        if fields:
            metadata['fields'] = fields
        history = _read_history(f)
    logger.info(f"Checkpoint loaded: {checkpoint_path} (iteration {iteration}, solution {solution.shape}, "
                f"{len(history.get('iterations', []))} history entries)")
    return solution, history, iteration, metadata


def load_checkpoint_header(checkpoint_path: Union[str, Path]) -> Tuple[dict, int, dict, dict]:
    """只读属性、收敛历史与各逐解点 / 逐单元数据集的形状，不读数组本身（布局校验与 `load_checkpoint` 相同，形状按
    迁移后的宽度给出）。分布式续算的每个 rank 构造求解器之前用它（数组由 root 读取后按 rank 分发）。

    Returns:
        `(history, iteration, metadata, field_shapes)`：metadata 不含 `fields`
    """
    checkpoint_path = _open(checkpoint_path)
    with h5py.File(checkpoint_path, 'r') as f:
        iteration, metadata = _read_metadata(f, checkpoint_path)
        sol_group = f["solution"]
        shapes = {name: tuple(sol_group[name].shape) for name in sol_group.keys() if name != "conserved"}
        _apply_layout(f, metadata, shapes, lambda shape, w: (shape[0], w) + shape[2:], checkpoint_path)
        history = _read_history(f)
    return history, iteration, metadata, shapes
