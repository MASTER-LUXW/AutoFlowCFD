"""AutoFlowCFD V2.0 - 求解器侧探针：每个时间步在探针位置求 FR 解多项式的值，写成时间序列。

气动声学的压力谱需要每个时间步的样本：此前 `post transient-psd` 的样本取自 checkpoint（间隔 = checkpoint 间隔，
通常 100 步一次，没有抗混叠，高于采样 Nyquist 频率的能量折叠进频带），取的是"离探针最近的单元的单元平均压力"，
时间轴是迭代数。

* 定位（集体操作，构造时一次）：root 在全局网格上找到包含探针的单元与参考坐标（`locate.py`），广播给各 rank；
  分布式时各 rank 只负责自己拥有的单元里的探针。
* 采样（集体操作，求解循环每步一次，`convergence_history.record_step` 调用）：各 rank 用插值行
  （`fr/point_interpolation.py`，与高阶 VTK 导出同一来源）乘本单元解点上的守恒量，root 汇总、换算成原始变量，
  追加一行 `time, iteration, 各探针 p u v w` 到 `probes.csv`（每行写完即关闭文件：中断时已写的样本完整）。
  阶数变化（Order Continuation）时按新阶数重建插值行，参考坐标不变。
* 续算：探针坐标随 checkpoint 持久化（`PROBE_POINTS_KEY`）；续算时截掉 `probes.csv` 里晚于 checkpoint 物理时间
  的行（中断前已写、续算会重算的那一段）再接着追加。
"""

import csv
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .locate import ProbeLocations, locate_probes
from .series_file import append_series_row, prepare_series_file

PROBE_FILE = "probes.csv"
PROBE_LAYOUT_FILE = "probes.json"
#: checkpoint 元数据里的探针坐标（续算时据此重新挂上探针）
PROBE_POINTS_KEY = "probe_points"
#: 每个探针记录的原始变量（列名 `probe{i}_{name}`）与它们在原始变量里的列号
RECORDED = (("p", 4), ("u", 1), ("v", 2), ("w", 3))


def _global_mesh(solver):
    """root 上的全局网格（完全分布式加载时在 `_root_context` 里，其余后端就是 `solver.mesh`）。"""
    ctx = getattr(solver, "_root_context", None)
    return ctx["mesh"] if ctx else solver.mesh


def _owned_global_cells(solver) -> Optional[np.ndarray]:
    """本 rank 拥有的单元的全局编号（与状态行顺序一致）；单机为 None（行号即全局编号）。"""
    partition = getattr(solver, "partition", None)
    if partition is None:
        return None
    return np.asarray(partition.local_cells, dtype=np.int64)[:int(partition.n_local_cells)]


def _state_rows(solver, rows: np.ndarray) -> np.ndarray:
    """本 rank 状态的若干行（守恒量前 5 个分量）拷到主机：`(len(rows), n_sps, 5)`。GPU 后端只拷这几行。"""
    from autoflowcfd.core.utils.array_module import array_module, to_host

    U = getattr(solver, "U_gpu", None)
    if U is None:
        return np.asarray(solver.state.U[rows, :, :5], dtype=np.float64)
    return np.asarray(to_host(U[array_module(U).asarray(rows)]), dtype=np.float64)[..., :5]


def _interpolation_weights(order: int, locations: ProbeLocations, n_sps: int) -> np.ndarray:
    """`(k, n_sps)`：第 i 行乘探针 i 所在单元的解点值得到该点的解多项式值（零填充槽位权重为 0）。"""
    from autoflowcfd.fr.point_interpolation import prism_interpolation_matrix, tet_interpolation_matrix

    W = np.zeros((len(locations.cells), n_sps))
    for i, (prism, rst) in enumerate(zip(locations.is_prism, locations.rst)):
        row = (prism_interpolation_matrix if prism else tet_interpolation_matrix)(order, rst)[0]
        W[i, :len(row)] = row
    return W


class ProbeRecorder:
    """见模块文档。构造与 `record` 都是集体操作（分布式时全部 rank 调用）。"""

    def __init__(self, solver, points, output_dir, resume_time: Optional[float] = None):
        """
        Args:
            solver: 已构造好的求解器（四个后端之一）
            points: 探针物理坐标 `(k, 3)`（米）
            output_dir: 输出目录（root 在这里写 `probes.csv` / `probes.json`）
            resume_time: 续算时 checkpoint 的物理时间：保留 `probes.csv` 里不晚于它的行、接着追加；None 时新建

        Raises:
            ValueError: 探针不在计算域内（`locate_probes`）
        """
        from autoflowcfd.core.mpi import get_comm, get_size, is_root

        self._root = is_root()
        self._comm = get_comm() if get_size() > 1 else None
        located, error = None, None
        if self._root:
            try:
                located = locate_probes(_global_mesh(solver), points)
            except ValueError as e:
                error = str(e)
        if self._comm is not None:
            located, error = self._comm.bcast((located, error), root=0)
        if error is not None:
            raise ValueError(error)
        if located is None:
            raise RuntimeError("探针定位结果缺失（root 未广播 locate_probes 的结果）")
        self.locations: ProbeLocations = located

        owned = _owned_global_cells(solver)
        if owned is None:
            self._mine = np.arange(len(located.cells))
            self._rows = located.cells.copy()
        else:
            order = np.argsort(owned)
            pos = np.searchsorted(owned, located.cells, sorter=order)
            pos = np.minimum(pos, len(owned) - 1)
            hit = owned[order[pos]] == located.cells
            self._mine = np.flatnonzero(hit)
            self._rows = order[pos[hit]]
        self._weights: Dict[int, np.ndarray] = {}   # 阶数 -> 本 rank 探针的插值权重

        self.path = Path(output_dir) / PROBE_FILE
        if self._root:
            self._prepare_file(resume_time)

    @property
    def points(self) -> np.ndarray:
        return self.locations.points

    def _header(self):
        return ["time", "iteration"] + [f"probe{i}_{name}" for i in range(len(self.points)) for name, _ in RECORDED]

    def _prepare_file(self, resume_time: Optional[float]) -> None:
        prepare_series_file(self.path, self._header(), resume_time, "探针坐标变了")
        layout = {"points": self.points.tolist(), "cells": self.locations.cells.tolist(),
                  "cell_is_prism": self.locations.is_prism.tolist(), "reference_coords": self.locations.rst.tolist(),
                  "columns": [name for name, _ in RECORDED]}
        (self.path.parent / PROBE_LAYOUT_FILE).write_text(json.dumps(layout, indent=2), encoding="utf-8")

    def sample(self, solver) -> Optional[np.ndarray]:
        """全部探针当前的原始变量 `(k, 5)`（rho, u, v, w, p），只在 root 返回，其余 rank 返回 None。"""
        from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive

        order = int(solver.current_order)          # Order Continuation 中途为当前所在阶数（solver.order 是目标阶数）
        local = np.zeros((len(self._mine), 5))
        if len(self._mine):
            U = _state_rows(solver, self._rows)                                  # (k_mine, n_sps, 5)
            if order not in self._weights:
                sub = ProbeLocations(points=self.points[self._mine], cells=self.locations.cells[self._mine],
                                     is_prism=self.locations.is_prism[self._mine], rst=self.locations.rst[self._mine])
                self._weights = {order: _interpolation_weights(order, sub, U.shape[1])}
            local = np.einsum("ks,ksv->kv", self._weights[order], U)
        parts = [(self._mine, local)] if self._comm is None else self._comm.gather((self._mine, local), root=0)
        if not self._root:
            return None
        conserved = np.empty((len(self.points), 5))
        for idx, values in parts:
            conserved[idx] = values
        return conserved_to_primitive(conserved)

    def record(self, solver) -> None:
        """求解循环每步调用一次（集体操作）：root 追加一行。"""
        Q = self.sample(solver)
        if Q is None:                                # 非 root
            return
        history = solver.convergence_history
        iteration = history.iterations[-1] if history.iterations else 0
        row = [repr(float(solver.time_integrator.current_time)), int(iteration)]
        row += [repr(float(Q[i, col])) for i in range(len(self.points)) for _, col in RECORDED]
        append_series_row(self.path, row)


def attach_probes(solver, points, output_dir, resume_time: Optional[float] = None) -> Optional[ProbeRecorder]:
    """给求解器挂上探针（集体操作）；`points` 为空时什么都不做、返回 None。"""
    if points is None or len(points) == 0:
        solver._probe_recorder = None
        return None
    solver._probe_recorder = ProbeRecorder(solver, points, output_dir, resume_time=resume_time)
    return solver._probe_recorder


def attach_probes_from_checkpoint(solver, metadata: dict, output_dir) -> Optional[ProbeRecorder]:
    """续算：checkpoint 记录了探针时按记录的坐标重新挂上，截掉晚于 checkpoint 物理时间的样本后接着追加。"""
    from autoflowcfd.core.utils.checkpoint_time import PHYSICAL_TIME_KEY

    points = metadata.get(PROBE_POINTS_KEY)
    if points is None or len(points) == 0:
        return None
    return attach_probes(solver, np.asarray(points, dtype=np.float64).reshape(-1, 3), output_dir,
                         resume_time=float(metadata[PHYSICAL_TIME_KEY]))


def read_probe_series(case_dir):
    """读算例目录的探针时间序列。

    Returns:
        `(points (k, 3), times (n,), values)`：values 为 `{变量名: (n, k)}`，变量名取 `RECORDED`（p、u、v、w）

    Raises:
        FileNotFoundError: 算例目录里没有 `probes.csv` / `probes.json`（求解时没有给探针）
        ValueError: 文件列与布局不一致
    """
    case_dir = Path(case_dir)
    csv_path, layout_path = case_dir / PROBE_FILE, case_dir / PROBE_LAYOUT_FILE
    if not csv_path.exists() or not layout_path.exists():
        raise FileNotFoundError(f"{case_dir} 里没有探针时间序列（{PROBE_FILE} / {PROBE_LAYOUT_FILE}）：求解时用 "
                                "`solve transient --probe X Y Z` 给出探针位置")
    points = np.asarray(json.loads(layout_path.read_text(encoding="utf-8"))["points"], dtype=np.float64)
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))
    header = ["time", "iteration"] + [f"probe{i}_{name}" for i in range(len(points)) for name, _ in RECORDED]
    if not rows or rows[0] != header:
        raise ValueError(f"{csv_path} 的列与 {layout_path} 的 {len(points)} 个探针不一致")
    data = np.asarray([[float(v) for v in r] for r in rows[1:]], dtype=np.float64).reshape(-1, len(header))
    values = {name: data[:, 2 + j::len(RECORDED)] for j, (name, _) in enumerate(RECORDED)}
    return points, data[:, 0], values
