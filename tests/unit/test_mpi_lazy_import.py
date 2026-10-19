"""MPI 惰性导入（I-94）：单机运行不导入 mpi4py（导入即 MPI_Init）。

此前 `autoflowcfd.core.mpi` 包导入时就 `from mpi4py import MPI`，而单机路径都导入它的 `is_root`：装了 mpi4py 的机器
上每次单机运行都初始化 MPI。本机没装 mpi4py，这里用一个记录导入的替身包验证。
"""

import sys

import pytest

import autoflowcfd.core.mpi as mpi_pkg
from tests.unit._patch_pkg import patch_pkg_attr

_FAKE_MPI = '''
import mpi4py
mpi4py.imported = True


class _Comm:
    def Get_rank(self):
        return 3

    def Get_size(self):
        return 4


COMM_WORLD = _Comm()
'''


@pytest.fixture
def fake_mpi4py(tmp_path, monkeypatch):
    pkg = tmp_path / "mpi4py"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("imported = False\n", encoding="utf-8")
    (pkg / "MPI.py").write_text(_FAKE_MPI, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in ("mpi4py", "mpi4py.MPI"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    for name in mpi_pkg._LAUNCH_ENV:
        monkeypatch.delenv(name, raising=False)
    patch_pkg_attr(monkeypatch, mpi_pkg, "_MPI", None)
    patch_pkg_attr(monkeypatch, mpi_pkg, "_import_attempted", False)
    yield
    for name in ("mpi4py", "mpi4py.MPI"):
        sys.modules.pop(name, None)


def test_single_process_run_does_not_import_mpi4py(fake_mpi4py):
    assert mpi_pkg.is_root() and mpi_pkg.get_size() == 1
    assert not mpi_pkg.mpi_available()
    assert "mpi4py.MPI" not in sys.modules


def test_mpi_launch_environment_imports_mpi4py(fake_mpi4py, monkeypatch):
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "4")
    assert mpi_pkg.get_rank() == 3 and mpi_pkg.get_size() == 4
    assert sys.modules["mpi4py"].imported


def test_distributed_path_can_require_mpi(fake_mpi4py):
    assert mpi_pkg.mpi_available(require=True)
    assert mpi_pkg.get_size() == 4
