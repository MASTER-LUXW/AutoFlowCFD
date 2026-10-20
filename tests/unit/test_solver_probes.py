"""求解器侧探针：定位、解多项式求值、每步写时间序列、续算接续、分布式汇总、PSD 读探针（I-170）。

此前 `post transient-psd` 的样本取自 checkpoint（间隔 = checkpoint 间隔、无抗混叠）、取离探针最近的单元的单元平均
压力、时间轴是迭代数。
"""

import csv
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from click.testing import CliRunner

from autoflowcfd.core.probes import locate_probes, read_probe_series
from autoflowcfd.core.probes.recorder import ProbeRecorder, attach_probes, attach_probes_from_checkpoint
from autoflowcfd.fr.point_interpolation import prism_interpolation_matrix, tet_interpolation_matrix
from autoflowcfd.grid.high_order import HighOrderMesh

_NODES = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1],          # 两个四面体
                   [2, 0, 0], [3, 0, 0], [2, 1, 0],                                  # 棱柱底面
                   [2.1, 0.1, 1.0], [3.3, -0.1, 1.2], [1.9, 1.2, 0.9]], dtype=float)  # 扭曲的顶面（映射为双线性）
_TETS = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int32)
_PRISMS = np.array([[5, 6, 7, 8, 9, 10]], dtype=np.int32)


def _mesh(order):
    vol = SimpleNamespace(cell_count=3, nodes=SimpleNamespace(get_coordinates=lambda: _NODES),
                          cells=SimpleNamespace(connectivity=_TETS), prism_cells=SimpleNamespace(connectivity=_PRISMS))
    mesh = HighOrderMesh(order=order)
    mesh.load_from_volume_mesh(vol)
    return mesh


def _linear(x):
    """一个线性场（每个分量不同系数），P>=1 的 FR 解精确表示它——棱柱双线性映射下同样在原生棱柱空间里。"""
    return np.stack([1.0 + 0.1 * x[..., 0], 2.0 - 0.3 * x[..., 1] + 0.2 * x[..., 2], 0.5 + 0.4 * x[..., 2],
                     -0.2 + 0.05 * x[..., 0] - 0.1 * x[..., 1], 3.0 + x.sum(axis=-1)], axis=-1)


@pytest.mark.parametrize("order", [0, 1, 2, 3])
def test_interpolation_rows_reproduce_the_solution_points(order):
    from autoflowcfd.fr.native_prism.basis import build_native_prism_nodes
    from autoflowcfd.fr.native_tet.basis import build_native_tet_operators

    for build, interp in ((build_native_prism_nodes, prism_interpolation_matrix),
                          (lambda o: build_native_tet_operators(o)[0], tet_interpolation_matrix)):
        nodes = build(order)
        np.testing.assert_allclose(interp(order, nodes), np.eye(len(nodes)), atol=1e-11)


def test_probes_are_located_with_their_reference_coordinates_in_both_cell_kinds():
    from autoflowcfd.fr.native_prism.basis import map_native_prism_to_physical
    from autoflowcfd.fr.native_tet.basis import map_native_tet_to_physical

    mesh = _mesh(2)
    rst_prism = np.array([[-0.6, -0.1, 0.3]])               # l = (0.35, 0.2, 0.45), t = 0.3：单元内部
    rst_tet = np.array([[-0.5, -0.6, -0.4]])                 # L = (0.25, 0.2, 0.3)
    p_prism = map_native_prism_to_physical(rst_prism, _NODES[mesh._fixed_prism_conn[0]])[0]
    p_tet = map_native_tet_to_physical(rst_tet, _NODES[mesh._fixed_tet_conn[1]])[0]
    loc = locate_probes(mesh, [p_prism, p_tet])
    assert loc.cells.tolist() == [0, 2] and loc.is_prism.tolist() == [True, False]
    np.testing.assert_allclose(loc.rst, np.vstack([rst_prism, rst_tet]), atol=1e-12)
    with pytest.raises(ValueError, match="不在计算域"):
        locate_probes(mesh, [[5.0, 5.0, 5.0]])


@pytest.mark.parametrize("order", [1, 2])
def test_sampled_values_are_the_solution_polynomial_at_the_probe(order):
    """解点上放一个线性场，探针取值等于该场在探针处的精确值（而非单元平均或最近解点的值）。"""
    from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive

    mesh = _mesh(order)
    U = _linear(mesh.sps_coords)
    points = np.array([[2.4, 0.3, 0.5], [0.2, 0.25, 0.3], [0.6, 0.55, 0.62]])
    solver = SimpleNamespace(mesh=mesh, current_order=order, state=SimpleNamespace(U=U))
    rec = ProbeRecorder.__new__(ProbeRecorder)
    with patch("autoflowcfd.core.probes.recorder.ProbeRecorder._prepare_file"):
        rec.__init__(solver, points, ".")
    np.testing.assert_allclose(rec.sample(solver), conserved_to_primitive(_linear(points)), rtol=1e-12)


_BC = {"x_min": "VELOCITY_INLET", "x_max": "PRESSURE_OUTLET", "wall_bottom": "WALL", "wall_top": "WALL",
       "z_min": "SYMMETRY", "z_max": "SYMMETRY"}
DT = 2e-4


def _channel_solver(cls_name="cpu"):
    from autoflowcfd.core.fr_solver import FRSolver
    from autoflowcfd.core.freestream_defaults import MU_MOLECULAR, P_INF, RHO_INF
    from autoflowcfd.core.time_integration import TimeIntegrationScheme
    from tests.validation._channel_mesh import build_channel_mesh_prism

    mesh = build_channel_mesh_prism(1, 3, 3, 2, 0.4, 0.1, 0.08)
    mesh.boundary_bc_types = dict(_BC)
    kw = dict(turb_model_name="NONE", time_scheme=TimeIntegrationScheme.DUAL_TIME, vel_inf=10.0, rho_inf=RHO_INF,
              p_inf=P_INF, mu_molecular=MU_MOLECULAR)
    if cls_name == "cpu":
        s = FRSolver(mesh=mesh, order=1, **kw)
    else:
        from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver
        from autoflowcfd.fr.operators import generate_fr_operators

        s = DistributedFRSolver(mesh=mesh, ops=generate_fr_operators(1), face_connectivity=mesh.face_connectivity,
                                n_ranks=1, order=1, **kw)
    s.order_continuation_enabled = False
    return s


_POINTS = [[0.13, 0.05, 0.03], [0.31, 0.02, 0.06]]


@pytest.mark.parametrize("backend", ["cpu", "distributed"])
def test_solve_writes_one_row_per_physical_step_and_resume_continues_it(tmp_path, backend):
    from autoflowcfd.cli.solve.checkpoint_io.write import write_checkpoint
    from autoflowcfd.core.utils.checkpoint import CheckpointManager

    solver = _channel_solver(backend)
    attach_probes(solver, _POINTS, tmp_path)
    solver.solve(max_iter=3, dt=DT, tol=0.0)
    points, times, values = read_probe_series(tmp_path)
    np.testing.assert_allclose(times, [DT, 2 * DT, 3 * DT], rtol=1e-12)
    np.testing.assert_allclose(points, _POINTS)
    np.testing.assert_allclose(values["p"][-1], solver._probe_recorder.sample(solver)[:, 4], rtol=1e-14)
    assert values["u"].shape == (3, 2)

    if backend == "cpu":
        meta = CheckpointManager().load(write_checkpoint(solver, str(tmp_path), 3, "mesh.nas", order=1,
                                                         turbulence_model="none", backend="cpu", quiet=True))[3]
        np.testing.assert_allclose(meta["probe_points"], _POINTS)
        np.testing.assert_allclose(meta["physical_time"], 3 * DT, rtol=1e-12)
        # 模拟第 2 步的 checkpoint、中断于第 3 步之后：续算时第 3 步那行被截掉，续算接着写
        meta["physical_time"] = 2 * DT
        resumed = _channel_solver()
        resumed.time_integrator.current_time = 2 * DT
        attach_probes_from_checkpoint(resumed, meta, tmp_path)
        assert read_probe_series(tmp_path)[1].tolist() == pytest.approx([DT, 2 * DT])
        resumed.solve(max_iter=2, dt=DT, tol=0.0)
        np.testing.assert_allclose(read_probe_series(tmp_path)[1], [DT, 2 * DT, 3 * DT, 4 * DT], rtol=1e-12)


def test_distributed_probe_values_are_gathered_from_the_owning_ranks():
    """两个 rank 各拥有一半单元：每个探针的值来自拥有它的 rank，root 按探针顺序拼回。"""
    from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive

    mesh = _mesh(2)
    U = _linear(mesh.sps_coords)
    owned = [np.array([2, 0]), np.array([1])]                  # rank 0：棱柱与第二个四面体；rank 1：第一个四面体
    points = np.array([[0.6, 0.55, 0.62], [0.2, 0.25, 0.3], [2.4, 0.3, 0.5]])   # 依次落在单元 2、1、0

    def rank_solver(r):
        return SimpleNamespace(mesh=mesh, current_order=2, state=SimpleNamespace(U=U[owned[r]]),
                               partition=SimpleNamespace(local_cells=owned[r], n_local_cells=len(owned[r])))

    located = locate_probes(mesh, points)

    class Comm:
        """模拟两个 rank 的集体通信：bcast 给出 root 的定位结果，gather 按调用顺序收集各 rank 的部分。"""

        def __init__(self):
            self.parts = []

        def bcast(self, obj, root=0):
            return located, None

        def gather(self, obj, root=0):
            self.parts.append(obj)
            return self.parts

    comm = Comm()
    recorders = []
    for r in (1, 0):                                           # rank 1 先交出自己的部分，rank 0（root）最后汇总
        with patch("autoflowcfd.core.mpi.get_size", return_value=2), \
             patch("autoflowcfd.core.mpi.get_comm", return_value=comm), \
             patch("autoflowcfd.core.mpi.is_root", return_value=(r == 0)), \
             patch("autoflowcfd.core.probes.recorder.ProbeRecorder._prepare_file"):
            recorders.append((r, ProbeRecorder(rank_solver(r), points, ".")))
    assert recorders[0][1]._mine.tolist() == [1] and sorted(recorders[1][1]._mine.tolist()) == [0, 2]
    assert recorders[0][1].sample(rank_solver(1)) is None
    gathered = recorders[1][1].sample(rank_solver(0))
    np.testing.assert_allclose(gathered, conserved_to_primitive(_linear(points)), rtol=1e-12)


def test_transient_psd_reads_the_probe_series(tmp_path):
    from autoflowcfd.cli.main import cli

    dt, n = 1e-4, 1024
    t = dt * np.arange(1, n + 1)
    (tmp_path / "probes.json").write_text('{"points": [[0, 0, 0]]}', encoding="utf-8")
    with open(tmp_path / "probes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "iteration", "probe0_p", "probe0_u", "probe0_v", "probe0_w"])
        for k, tk in enumerate(t):
            # 起动段（t < 0.02 s）一个大的 50 Hz 瞬态，之后是 700 Hz 的脉动
            p = 101325.0 + (50.0 * np.sin(2 * np.pi * 50 * tk) if tk < 0.02 else np.sin(2 * np.pi * 700 * tk))
            w.writerow([repr(tk), k + 1, repr(p), 10.0, 0.0, 0.0])
    out = tmp_path / "psd.csv"
    res = CliRunner().invoke(cli, ["post", "transient-psd", "--case", str(tmp_path), "-o", str(out),
                                   "--warmup-time", "0.02"])
    assert res.exit_code == 0, res.output
    assert "dominant f=7" in res.output and abs(float(res.output.split("dominant f=")[1].split()[0]) - 700) < 20
    assert out.exists()
