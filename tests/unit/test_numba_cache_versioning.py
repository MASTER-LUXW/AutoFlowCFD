# -*- coding: utf-8 -*-
"""numba 磁盘缓存按核源码版本隔离（`autoflowcfd/_numba_cache.py`）。

2026-09-25 真实事故：`compute_ausm_up_flux`（被全部无粘残差核内联）改了，
调用方源文件没变 -> numba 缓存不失效 -> CPU 残差核静默执行旧机器码，CPU/GPU
交叉验证当场不一致；换空缓存目录后全部通过。
"""

import os

from autoflowcfd._numba_cache import kernel_source_digest


def _write(p, text):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(text)


def test_digest_changes_when_an_inlined_kernel_source_changes(tmp_path):
    root = str(tmp_path / "pkg")
    _write(os.path.join(root, "a", "flux.py"), "from numba import njit\n@njit\ndef f(x):\n    return 0.25 * x\n")
    _write(os.path.join(root, "b", "residual.py"), "from numba import njit\nfrom ..a.flux import f\n")
    d0 = kernel_source_digest(root)
    _write(os.path.join(root, "a", "flux.py"), "from numba import njit\n@njit\ndef f(x):\n    return 1.0 * x\n")
    assert kernel_source_digest(root) != d0, "被内联函数的源码改了，缓存版本必须随之改变"


def test_digest_ignores_files_that_do_not_use_numba(tmp_path):
    root = str(tmp_path / "pkg")
    _write(os.path.join(root, "k.py"), "import numba\n")
    _write(os.path.join(root, "cli.py"), "print('a')\n")
    d0 = kernel_source_digest(root)
    _write(os.path.join(root, "cli.py"), "print('b')\n")
    assert kernel_source_digest(root) == d0, "与 numba 无关的文件改动不应触发全体重编译"


def test_package_import_points_numba_at_the_versioned_dir():
    import numba

    import autoflowcfd

    pkg = os.path.dirname(os.path.abspath(autoflowcfd.__file__))
    digest = kernel_source_digest(pkg)
    assert os.path.basename(os.path.normpath(numba.config.CACHE_DIR)) == digest
    assert os.path.normpath(os.environ["NUMBA_CACHE_DIR"]) == os.path.normpath(numba.config.CACHE_DIR)
