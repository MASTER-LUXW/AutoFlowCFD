"""包入口：不覆盖模块自身的 __name__、版本号唯一来源、BLAS 线程默认值按本节点 MPI rank 数均分（I-178）。"""

import importlib.util
import multiprocessing
import os
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import autoflowcfd
from autoflowcfd import _version


def test_module_name_is_not_overridden():
    """此前 `__name__ = "AutoFlowCFD"` 覆盖了模块名（相对导入、pickle、日志按模块名定位都会错）。"""
    assert autoflowcfd.__name__ == "autoflowcfd"
    assert autoflowcfd.get_info()["name"] == "AutoFlowCFD"


def test_version_comes_from_the_package_metadata():
    assert autoflowcfd.__version__ == metadata.version("autoflowcfd") == autoflowcfd.get_version()


def test_version_falls_back_to_pyproject_when_not_installed(monkeypatch):
    def not_installed(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(_version.metadata, "version", not_installed)
    pyproject = (Path(autoflowcfd.__file__).resolve().parents[2] / "pyproject.toml").read_text(encoding="utf-8")
    expected = re.search(r'(?m)^version\s*=\s*"([^"]+)"', pyproject).group(1)
    assert _version.read_version() == expected


def test_empty_utils_package_is_gone():
    assert importlib.util.find_spec("autoflowcfd.utils") is None


_THREAD_VARS = ("MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "OMP_NUM_THREADS", "OMPI_COMM_WORLD_LOCAL_SIZE", "MPI_LOCALNRANKS")


def _threads_after_import(**env_vars):
    # 本进程 import autoflowcfd 时已把线程变量写进 os.environ，子进程会继承：全部去掉，只留本用例给的
    env = {k: v for k, v in os.environ.items() if k not in _THREAD_VARS}
    env.update(env_vars)
    code = "import os, autoflowcfd; print(os.environ['OMP_NUM_THREADS'], os.environ['OPENBLAS_NUM_THREADS'])"
    out = subprocess.run([sys.executable, "-c", code], env=dict(env, PYTHONIOENCODING="utf-8"), capture_output=True,
                         text=True, encoding="utf-8", errors="replace", check=True)
    return out.stdout.split()[-2:]


def test_blas_thread_default_is_divided_by_ranks_per_node():
    """MPI 每节点多个 rank 时各 rank 只取本节点核数的一份（此前每个 rank 都取全部核数，超订）。"""
    cpus = multiprocessing.cpu_count()
    assert _threads_after_import() == [str(cpus)] * 2
    assert _threads_after_import(OMPI_COMM_WORLD_LOCAL_SIZE="2") == [str(max(1, cpus // 2))] * 2
    assert _threads_after_import(MPI_LOCALNRANKS=str(4 * cpus)) == ["1"] * 2


def test_explicit_thread_setting_is_kept():
    assert _threads_after_import(OMPI_COMM_WORLD_LOCAL_SIZE="2", OMP_NUM_THREADS="3")[0] == "3"
