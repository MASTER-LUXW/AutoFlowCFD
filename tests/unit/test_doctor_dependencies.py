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
