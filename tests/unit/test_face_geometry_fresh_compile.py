"""面几何（面通量点 + 精确法向）在**全新 numba 编译**下能建完（2026-10-19）。

`cache=True` 的 numba 缓存按函数自身所在文件判断是否失效：被调函数或编译上下文变了，旧的编译结果仍可能被复用。
2026-10-19 删除面通量点坍缩分支（I-34）后，精确法向 kernel 在全新编译时段错误（其中不可达的坍缩面分支里按单元类型
条件定义的局部量，被 `parallel=True` 的 parfor 生成了越界访问），而全量测试复用旧缓存、照样通过。本测试在子进程里用
空的缓存目录建一次混合网格，专门覆盖"被编译缓存掩盖"的这一类问题（编译耗时约 1~2 分钟）。
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

_SNIPPET = textwrap.dedent(
    """
    import sys
    sys.path.insert(0, {root!r})
    from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh
    for order in (1, 2):
        mesh = _build_synthetic_mixed_mesh(order)
        assert mesh.face_flux_points is not None
    print("FRESH_COMPILE_OK")
    """
)


def test_mixed_mesh_face_geometry_builds_with_an_empty_numba_cache(tmp_path):
    root = str(Path(__file__).resolve().parents[2])
    env = dict(os.environ, AFCFD_NUMBA_CACHE_ROOT=str(tmp_path / "numba_cache"))
    env.pop("NUMBA_CACHE_DIR", None)
    res = subprocess.run([sys.executable, "-c", _SNIPPET.format(root=root)], capture_output=True, text=True,
                         timeout=1800, env=env, cwd=root)
    assert res.returncode == 0 and "FRESH_COMPILE_OK" in res.stdout, (
        f"全新编译建面几何失败（返回码 {res.returncode}）：\n{res.stderr[-3000:]}")
