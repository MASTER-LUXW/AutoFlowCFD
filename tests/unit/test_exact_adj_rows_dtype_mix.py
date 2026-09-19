"""两张连接表整数 dtype 不一致时的 numba 段错误（真实生产缺陷回归）。

## 缺陷

`fr/face_flux_points_exact_normal_kernel.py::compute_exact_adj_rows_kernel`
的坍缩分支里，`p0 = node_coords[prism_conn[cell, 0]]` 与
`p0 = node_coords[tet_conn[tc, 0]]` 是同一个变量的 if/else 两支。两张表
的索引 dtype 不同时，`parallel=True` 下的 numba 生成**直接段错误**的代码
—— 不是异常、没有任何 Python 栈，只有一串
"Windows fatal exception: access violation"，进程 exit 139。

## 为什么它在生产里必然命中

`fr/face_flux_points_merge.py` 在 `mesh._fixed_prism_conn is None`
（**纯四面体网格**）时传的兜底是 `np.empty((0, 6), dtype=np.int64)`，
而真实的 `_fixed_tet_conn` 是 `int32`。于是任何纯四面体网格在
`build_face_flux_points` 阶段就段错误 —— 也就是 `solve steady/transient`
跑纯四面体体网格必现。实测 order 1/2/3 全部 exit 139。

它此前没被发现是因为唯一覆盖纯四面体网格的测试
（`test_vtk_export_highorder.py::TestExportHighorderVtkOrder3Tet`）本身
就是被这个段错误打断的 —— 段错误让 pytest 整个进程消失、连 summary 都
打不出来，所以在全量输出里表现成"跑到 98% 就没了"而不是一条 FAILED。

## 修复

在 `compute_exact_adj_rows_fast` 里把两张表统一归一化到 `int64`（该函数
本来就在对 `code_arr`/`sps_1d` 做同样的事）。修在 wrapper 而不是去改那
两处兜底的 dtype：调用方传什么 dtype 都不该让 kernel 崩，只改兜底的话
将来任何一个 int64 连接表的网格会从另一边再踩一次同一个坑。

## 为什么这里不直接断言"不段错误"

段错误杀掉整个进程，pytest 里**断言不了**。所以这两条测试用子进程跑，
断言子进程的返回码 —— 修复前 139、修复后 0。
"""

import subprocess
import sys
import textwrap

import numpy as np
import pytest

_REPO_ROOT_SNIPPET = textwrap.dedent(
    """
    import sys
    sys.path.insert(0, {src!r})
    import numpy as np
    from autoflowcfd.fr.face_flux_points_exact_normal_kernel import (
        compute_exact_adj_rows_fast,
    )
    from autoflowcfd.fr.quadrature_points import gauss_legendre

    nodes = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
    tet = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.{tet_dtype})
    prism = np.empty((0, 6), dtype=np.{prism_dtype})
    n1d = 2
    s1, _ = gauss_legendre(n1d)
    n_faces = 8
    out = compute_exact_adj_rows_fast(
        n_faces, n1d, s1, 0,
        cell_arr=np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64),
        axis_arr=np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64),
        side_arr=np.full(n_faces, -1.0),
        prism_conn=prism, tet_conn=tet, node_coords=nodes,
        code_arr=np.array([6, 7, 8, 9, 6, 7, 8, 9], dtype=np.int64),
    )
    assert np.all(np.isfinite(out)) and np.abs(out).max() > 0.0
    print("OK")
    """
)


def _run_in_subprocess(tet_dtype: str, prism_dtype: str, src: str):
    code = _REPO_ROOT_SNIPPET.format(
        src=src, tet_dtype=tet_dtype, prism_dtype=prism_dtype)
    return subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True, timeout=900)


@pytest.fixture(scope="module")
def src_dir():
    from pathlib import Path

    import autoflowcfd

    return str(Path(autoflowcfd.__file__).resolve().parent.parent)


@pytest.mark.parametrize("tet_dtype,prism_dtype", [
    ("int32", "int64"),   # 生产里真实出现的那一组（纯四面体网格的兜底）
    ("int64", "int32"),   # 反向：将来某个 int64 连接表的网格会命中这一侧
    ("int32", "int32"),
    ("int64", "int64"),
])
def test_mixed_connectivity_dtypes_do_not_crash(tet_dtype, prism_dtype,
                                                src_dir):
    """四种 dtype 组合都必须正常返回（修复前前两组 exit 139）。"""
    r = _run_in_subprocess(tet_dtype, prism_dtype, src_dir)
    assert r.returncode == 0, (
        f"tet={tet_dtype} prism={prism_dtype} 子进程返回码 {r.returncode}"
        f"（139 = 段错误，说明 numba 又对不同 dtype 的两支生成了坏代码）\n"
        f"stdout: {r.stdout[-2000:]}\nstderr: {r.stderr[-2000:]}")
    assert "OK" in r.stdout


def test_wrapper_normalizes_both_tables_to_int64():
    """白盒：wrapper 必须把两张表都归一化 —— 这是上面那条通过的机制。

    直接断言机制本身（而不是只断言"不崩"）：将来若有人把归一化删掉，
    在**恰好同 dtype** 的开发环境里上面那四条可能仍然全过，而生产的
    纯四面体网格照样崩。
    """
    import inspect

    from autoflowcfd.fr import face_flux_points_exact_normal_kernel as K

    src = inspect.getsource(K.compute_exact_adj_rows_fast)
    assert "prism_conn = np.ascontiguousarray(prism_conn, dtype=np.int64)" in src
    assert "tet_conn = np.ascontiguousarray(tet_conn, dtype=np.int64)" in src


def test_pure_tet_mesh_builds_face_geometry(tmp_path):
    """端到端：**纯四面体**网格（n_prism=0）必须能建完面几何。

    这是缺陷的真实触发路径 —— `build_face_flux_points` 里的兜底
    `np.empty((0, 6), dtype=np.int64)` 配 int32 的 `_fixed_tet_conn`。
    同样必须在子进程里跑，因为修复前它是段错误而不是异常。
    """
    from pathlib import Path

    import autoflowcfd

    src = str(Path(autoflowcfd.__file__).resolve().parent.parent)
    code = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {src!r})
        import numpy as np
        from types import SimpleNamespace
        from autoflowcfd.grid.high_order.high_order_mesh import HighOrderMesh

        class _N:
            def __init__(s, a): s._c = a
            def get_coordinates(s): return s._c

        class _C:
            def __init__(s, c): s.connectivity = c

        nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                          [1, 1, 1]], dtype=float)
        tet = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int32)
        mv = SimpleNamespace(cell_count=2, nodes=_N(nodes), cells=_C(tet),
                             prism_cells=_C(np.zeros((0, 6), dtype=np.int32)))
        for order in (1, 2, 3):
            m = HighOrderMesh(order=order)
            m.load_from_volume_mesh(mv)
            assert m.n_prism_cells == 0
            assert m.face_flux_points is not None
        print("OK")
    """)
    r = subprocess.run([sys.executable, "-c", code],
                       capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, (
        f"纯四面体网格建面几何子进程返回码 {r.returncode}"
        f"（139 = 段错误）\nstderr: {r.stderr[-3000:]}")
    assert "OK" in r.stdout
