"""`utils doctor` 按包的依赖声明（pyproject，经包元数据）逐项核对（I-177）。

此前清单写死：scipy、tetgen、pyvista 没查；numba、h5py 被列为"可选"（求解核心依赖 numba 内核、续算依赖 h5py
checkpoint）；mpi4py / pymetis / psutil 不在任何清单里；Python 门槛写的 3.8。
"""

from importlib import metadata

from autoflowcfd.cli.utils_commands import _dependency_report


def test_every_declared_dependency_is_checked():
    issues, warnings, versions = _dependency_report()
    for name in ("numba", "h5py", "scipy", "tetgen", "pyvista", "numpy", "packaging", "mpi4py", "pymetis", "psutil"):
        assert name in versions


def test_missing_required_dependency_is_an_issue_and_missing_extra_a_warning(monkeypatch):
    real = metadata.version

    def fake(name):
        if name in ("numba", "mpi4py"):
            raise metadata.PackageNotFoundError(name)
        return real(name)

    monkeypatch.setattr(metadata, "version", fake)
    issues, warnings, versions = _dependency_report()
    assert any("必需依赖 numba" in i for i in issues)
    assert any("[mpi]" in w and "mpi4py" in w for w in warnings) and not any("mpi4py" in i for i in issues)
    assert versions["numba"] is None


def test_required_dependencies_cover_the_whole_supported_python_range():
    """必需依赖不带比包本身更窄的 Python 标记（I-194）：此前 python = "^3.10" 允许 3.12+，而 scipy 只对 <3.12 声明，
    3.12 上装出来的环境缺 scipy。"""
    from packaging.requirements import Requirement

    for spec in metadata.requires("autoflowcfd"):
        req = Requirement(spec)
        if req.marker is None or "extra" in str(req.marker):
            continue
        raise AssertionError(f"必需依赖 {req.name} 带 Python 标记 {req.marker}：应收紧包的 python 范围而不是给依赖加标记")


def test_lock_file_contains_every_declared_dependency():
    """poetry.lock 与依赖声明同步（I-194）：此前 I-177 新增的 mpi4py / pymetis / psutil 不在锁文件里。"""
    import re
    from pathlib import Path

    from packaging.requirements import Requirement

    lock = (Path(__file__).resolve().parents[2] / "poetry.lock").read_text(encoding="utf-8")
    locked = {m.lower().replace("_", "-") for m in re.findall(r'(?m)^name = "([^"]+)"', lock)}
    declared = {Requirement(s).name.lower().replace("_", "-") for s in metadata.requires("autoflowcfd")}
    assert declared <= locked, sorted(declared - locked)
