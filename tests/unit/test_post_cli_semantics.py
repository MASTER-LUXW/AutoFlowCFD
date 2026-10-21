"""post 命令的语义（I-170）：convergence 的变量是限定取值、请求的量没有记录时报错；transient-mean 的系数统计只重建
一次求解器、任何一个样本出错即报错（此前逐样本打警告跳过，且每个 checkpoint 都重新尝试重建）。"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from click.testing import CliRunner

from autoflowcfd.cli.main import cli


def test_convergence_variables_are_a_choice_and_missing_data_is_an_error(tmp_path):
    runner = CliRunner()
    res = runner.invoke(cli, ["post", "convergence", "--case", str(tmp_path), "--variables", "residuals"])
    assert res.exit_code == 2 and "residuals" in res.output                       # 此前静默忽略、画空图

    history = {"iterations": [1, 2], "residuals": {"monitor": [1.0, 0.5]}, "cfl_history": [1.0, 1.0]}
    with patch("autoflowcfd.cli.post.commands._load_history_only", return_value=(history, 2, {})):
        res = runner.invoke(cli, ["post", "convergence", "--case", str(tmp_path), "--variables", "coefficients",
                                  "-o", str(tmp_path / "c.png")])
    assert res.exit_code != 0 and "coefficients" in res.output and not (tmp_path / "c.png").exists()


class _Stats:
    """TransientStatistics 的替身：记录累积了哪些时刻与系数。"""

    def __init__(self, grid_data):
        self.times, self.coeffs = [], []

    def accumulate(self, solution, time):
        self.times.append(time)

    def accumulate_coefficients(self, Cd, Cl, Cs):
        self.coeffs.append(Cd)

    @property
    def n_coeff_samples(self):
        return len(self.coeffs)

    def compute_statistics(self):
        mean = {k: float(np.mean(self.coeffs)) if self.coeffs else 0.0 for k in ("Cd", "Cl", "Cs")}
        return SimpleNamespace(mean_fields={}, rms_fields={}, num_samples=len(self.times), sampling_time=0.1,
                               mean_coefficients=mean if self.coeffs else {}, std_coefficients=mean)


def _run_mean(tmp_path, coefficient_values, rebuild_error=None):
    ckpts = [Path(f"checkpoint_iter_{i:06d}.h5") for i in (100, 200, 300)]
    meta = {c: {"physical_time": t, "fields": {"U_sps": np.zeros((1, 1, 5))}} for c, t in zip(ckpts, (0.1, 0.2, 0.3))}
    built, restored, stats = [], [], []
    values = iter(coefficient_values)

    def rebuild(path, **kw):
        built.append(path)
        if rebuild_error is not None:
            raise rebuild_error
        return SimpleNamespace(_reference_area=2.0), 0, {}

    def coefficients(solver, reference_area):
        v = next(values)
        if isinstance(v, Exception):
            raise v
        return SimpleNamespace(Cd=v, Cl=0.0, Cs=0.0)

    def make_stats(grid_data):
        stats.append(_Stats(grid_data))
        return stats[-1]

    p = "autoflowcfd.cli.post.transient_commands"
    with patch(f"{p}._load_grid_data"), patch(f"{p}._locate_grid_file"), \
         patch(f"{p}._list_checkpoints", return_value=ckpts), \
         patch(f"{p}._to_solution_vector"), patch(f"{p}._export_point_fields_vtk"), \
         patch("autoflowcfd.core.utils.checkpoint.CheckpointManager.load",
               side_effect=lambda f: (None, {}, 0, meta[f])), \
         patch("autoflowcfd.postprocess.TransientStatistics", side_effect=make_stats), \
         patch("autoflowcfd.cli.solve.checkpoint_io.rebuild_solver_from_checkpoint", side_effect=rebuild), \
         patch("autoflowcfd.cli.solve.checkpoint_io.restore_solver_state_from_fields",
               side_effect=lambda s, f, m: restored.append(m["physical_time"])), \
         patch("autoflowcfd.postprocess.fr_coefficients.compute_aerodynamic_coefficients_fr",
               side_effect=coefficients):
        res = CliRunner().invoke(cli, ["post", "transient-mean", "--case", str(tmp_path), "--warmup-time", "0.15",
                                       "-o", str(tmp_path / "m.vtk")])
    return res, built, restored, stats[0]


def test_transient_mean_rebuilds_once_and_uses_physical_time(tmp_path):
    res, built, restored, stats = _run_mean(tmp_path, [1.0, 3.0])
    assert res.exit_code == 0, res.output
    assert "Cd=2.000000" in res.output
    assert stats.times == [0.2, 0.3]                       # 按物理时间排除起动段（此前按迭代数比较）
    assert len(built) == 1 and restored == [0.3]          # 只重建一次，之后只替换状态
    assert stats.coeffs == [1.0, 3.0]


def test_transient_mean_coefficient_failure_is_an_error_not_a_skipped_sample(tmp_path):
    res, _built, _restored, stats = _run_mean(tmp_path, [1.0, RuntimeError("壁面积分失败")])
    assert res.exit_code != 0 and "壁面积分失败" in res.output
    assert stats.coeffs == [1.0]


def test_transient_mean_rebuild_failure_is_an_error_not_retried_per_checkpoint(tmp_path):
    """此前重建失败只打警告，下一个 checkpoint 再重建一次（真实网格上每次 1~2 分钟）。"""
    res, built, _restored, _stats = _run_mean(tmp_path, [], rebuild_error=ValueError("没有物理无滑移固壁组"))
    assert res.exit_code != 0 and "没有物理无滑移固壁组" in res.output and len(built) == 1
