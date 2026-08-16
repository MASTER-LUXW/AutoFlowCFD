"""`get_flat_face_geometry` 单槽缓存的身份安全性回归测试。

背景：缓存最初用 `id(mesh.face_flux_points)`（裸 int）当键，真实复现过
一次 CPython 的 id 复用竞争——旧 mesh 被 GC 后，新建的、完全不相关的
另一个 mesh 的 `face_flux_points` list 复用了同一个内存地址（同一个
`id()`），导致缓存把旧 mesh 的展平几何错误地返回给新 mesh（连续跑全量
测试套件时抓到过一次因此触发的 `ValueError: incompatible array sizes
for np.dot`；同样的全量套件另一次完全没触发——形状恰好兼容时不会崩溃，
是静默算错，比崩溃更危险）。修复为持有 `mesh.face_flux_points` 的强
引用本身当键、用 `is` 做身份比较，见 `fr_face_kernels_flat.py` 模块
文档"缓存键危险陷阱"一节。

这里不追求确定性复现真实的 CPython 内存复用（依赖分配器内部实现，
环境相关），而是直接验证修复后的不变量：无论创建/销毁多少个不同的
mesh 对象、是否显式触发 GC，`get_flat_face_geometry` 返回的几何永远
与"当前传入的 mesh"自身的面数/SPs 数一致，不会静默返回另一个 mesh 的
陈旧几何。
"""

import gc

from autoflowcfd.core.fr_face_kernels_flat import get_flat_face_geometry

from .test_fr_residual_inviscid import _build_synthetic_mixed_mesh


def test_same_mesh_repeated_calls_return_identical_cached_object():
    """同一个 mesh 对象重复调用应命中缓存，返回同一个对象（身份相同）。"""
    mesh = _build_synthetic_mixed_mesh(2)
    flat1 = get_flat_face_geometry(mesh, mesh.operators)
    flat2 = get_flat_face_geometry(mesh, mesh.operators)
    assert flat1 is flat2


def test_mesh_churn_never_returns_stale_geometry():
    """大量创建/销毁不同 mesh（穿插显式 GC，最大化 id 复用竞争窗口），
    每次都必须拿到与当前 mesh 自身一致的几何，不能是上一个 mesh 的。"""
    for i in range(30):
        order = 1 if i % 2 == 0 else 2
        mesh = _build_synthetic_mixed_mesh(order)
        flat = get_flat_face_geometry(mesh, mesh.operators)

        assert flat.n_faces == mesh.face_connectivity.n_faces
        assert flat.n_sps == mesh.n_sps_per_cell

        del mesh, flat
        gc.collect()
