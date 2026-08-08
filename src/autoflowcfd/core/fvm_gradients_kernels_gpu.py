"""CUDA GPU kernel：Green-Gauss 梯度重构 + Barth-Jespersen 限制器，未经真实 GPU 硬件验证。

⚠️ 本文件是 fvm_gradients_kernels.py（已验证，梯度误差 1.2e-14、限制器
误差 3.5e-14）的 CUDA 翻译，但两者的**并行化结构不同**，风险比
fvm_inviscid_kernels_gpu.py/fvm_viscous_kernels_gpu.py 更高，需要格外
留意：

- CPU 版本里跨面向单元 scatter-add（一个面同时写 owner 和 neighbour 两个
  单元）在 `@njit(cache=True)`（未加 `parallel=True`）的**串行**循环里
  完成，天然无竞争。GPU 版本改成一线程一面，多个面可能并发写同一个单元，
  所以这里必须用 `cuda.atomic.add`/`cuda.atomic.max`/`cuda.atomic.min`
  代替普通的 `+=`/比较赋值，否则会重现 CPU 侧那次真实出现过的并行数据
  竞争（见 fvm_viscous_kernels.py 的注释）。
- Barth-Jespersen 限制器额外把 CPU 版本里的一个 Python 闭包
  `_constrain(c, fidx, phi_arr)` 展开成独立 kernel（GPU 不支持在
  `@cuda.jit` 内定义闭包并跨 kernel 调用），逻辑保持一致。

在任何生产环境依赖这条路径之前，必须先在有 GPU 的机器上用和
fvm_gradients_kernels.py 同样的方法做数值对比验证，重点检查 atomic
scatter 部分的正确性（这是 CPU/GPU 结构差异最大的地方）。
"""

import numpy as np

try:
    from numba import cuda
    CUDA_AVAILABLE = cuda.is_available()
except Exception:
    CUDA_AVAILABLE = False


if CUDA_AVAILABLE:
    @cuda.jit(cache=True)
    def _gg_internal_scatter_kernel(cell_values, int_owner, int_neigh, int_areas, int_normals, grad):
        f = cuda.grid(1)
        if f >= int_owner.shape[0]:
            return
        o = int_owner[f]
        nb = int_neigh[f]
        a = int_areas[f]
        nx = int_normals[f, 0]; ny = int_normals[f, 1]; nz = int_normals[f, 2]
        n_vars = cell_values.shape[1]
        for v in range(n_vars):
            phi_f = 0.5 * (cell_values[o, v] + cell_values[nb, v])
            cx = phi_f * a * nx
            cy = phi_f * a * ny
            cz = phi_f * a * nz
            cuda.atomic.add(grad, (o, v, 0), cx)
            cuda.atomic.add(grad, (o, v, 1), cy)
            cuda.atomic.add(grad, (o, v, 2), cz)
            cuda.atomic.add(grad, (nb, v, 0), -cx)
            cuda.atomic.add(grad, (nb, v, 1), -cy)
            cuda.atomic.add(grad, (nb, v, 2), -cz)

    @cuda.jit(cache=True)
    def _gg_boundary_scatter_kernel(bnd_owner, bnd_areas, bnd_normals, bnd_values, grad):
        f = cuda.grid(1)
        if f >= bnd_owner.shape[0]:
            return
        o = bnd_owner[f]
        a = bnd_areas[f]
        nx = bnd_normals[f, 0]; ny = bnd_normals[f, 1]; nz = bnd_normals[f, 2]
        n_vars = bnd_values.shape[1]
        for v in range(n_vars):
            phi_b = bnd_values[f, v]
            cuda.atomic.add(grad, (o, v, 0), phi_b * a * nx)
            cuda.atomic.add(grad, (o, v, 1), phi_b * a * ny)
            cuda.atomic.add(grad, (o, v, 2), phi_b * a * nz)

    @cuda.jit(cache=True)
    def _gg_normalize_kernel(cell_volumes, grad):
        c = cuda.grid(1)
        if c >= cell_volumes.shape[0]:
            return
        vol = cell_volumes[c]
        if vol < 1e-30:
            vol = 1e-30
        n_vars = grad.shape[1]
        for v in range(n_vars):
            grad[c, v, 0] /= vol
            grad[c, v, 1] /= vol
            grad[c, v, 2] /= vol

    @cuda.jit(cache=True)
    def _bj_minmax_init_kernel(cell_values, u_max, u_min):
        c = cuda.grid(1)
        if c >= cell_values.shape[0]:
            return
        n_vars = cell_values.shape[1]
        for v in range(n_vars):
            u_max[c, v] = cell_values[c, v]
            u_min[c, v] = cell_values[c, v]

    @cuda.jit(cache=True)
    def _bj_minmax_scatter_kernel(cell_values, int_owner, int_neigh, u_max, u_min):
        f = cuda.grid(1)
        if f >= int_owner.shape[0]:
            return
        o = int_owner[f]
        nb = int_neigh[f]
        n_vars = cell_values.shape[1]
        for v in range(n_vars):
            vo = cell_values[o, v]
            vn = cell_values[nb, v]
            cuda.atomic.max(u_max, (o, v), vn)
            cuda.atomic.min(u_min, (o, v), vn)
            cuda.atomic.max(u_max, (nb, v), vo)
            cuda.atomic.min(u_min, (nb, v), vo)

    @cuda.jit(device=True, inline=True)
    def _bj_constrain_one(c, fidx, cell_values, grad, u_max, u_min, face_centers, cell_centroids, phi):
        n_vars = cell_values.shape[1]
        rx = face_centers[fidx, 0] - cell_centroids[c, 0]
        ry = face_centers[fidx, 1] - cell_centroids[c, 1]
        rz = face_centers[fidx, 2] - cell_centroids[c, 2]
        eps = 1e-12
        for v in range(n_vars):
            delta = grad[c, v, 0] * rx + grad[c, v, 1] * ry + grad[c, v, 2] * rz
            if delta > eps:
                d = delta if delta > eps else eps
                pf = (u_max[c, v] - cell_values[c, v]) / d
                if pf > 1.0:
                    pf = 1.0
            elif delta < -eps:
                d = delta if delta < -eps else -eps
                pf = (u_min[c, v] - cell_values[c, v]) / d
                if pf > 1.0:
                    pf = 1.0
            else:
                pf = 1.0
            if pf < 0.0:
                pf = 0.0
            cuda.atomic.min(phi, (c, v), pf)

    @cuda.jit(cache=True)
    def _bj_constrain_all_faces_kernel(owner, cell_values, grad, u_max, u_min, face_centers, cell_centroids, phi):
        f = cuda.grid(1)
        if f >= owner.shape[0]:
            return
        _bj_constrain_one(owner[f], f, cell_values, grad, u_max, u_min, face_centers, cell_centroids, phi)

    @cuda.jit(cache=True)
    def _bj_constrain_neighbour_kernel(int_neigh, int_face_idx, cell_values, grad, u_max, u_min,
                                        face_centers, cell_centroids, phi):
        f = cuda.grid(1)
        if f >= int_neigh.shape[0]:
            return
        _bj_constrain_one(int_neigh[f], int_face_idx[f], cell_values, grad, u_max, u_min,
                           face_centers, cell_centroids, phi)


def _launch_1d(kernel, n, *args):
    threads_per_block = 256
    blocks = (n + threads_per_block - 1) // threads_per_block
    kernel[blocks, threads_per_block](*args)


def green_gauss_gradient_gpu(
    cell_values: np.ndarray, int_owner: np.ndarray, int_neigh: np.ndarray,
    int_areas: np.ndarray, int_normals: np.ndarray,
    bnd_owner: np.ndarray, bnd_areas: np.ndarray, bnd_normals: np.ndarray,
    bnd_values: np.ndarray, cell_volumes: np.ndarray,
) -> np.ndarray:
    """签名与 fvm_gradients_kernels._green_gauss_gradient_kernel 一致。

    ⚠️ 未经真实 GPU 硬件验证，见本文件模块级文档字符串。
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA GPU not available in this environment")
    n_cells, n_vars = cell_values.shape
    d_grad = cuda.to_device(np.zeros((n_cells, n_vars, 3), dtype=np.float64))
    d_cell_values = cuda.to_device(np.ascontiguousarray(cell_values, dtype=np.float64))
    d_cell_volumes = cuda.to_device(np.ascontiguousarray(cell_volumes, dtype=np.float64))

    n_int = int_owner.shape[0]
    if n_int > 0:
        _launch_1d(
            _gg_internal_scatter_kernel, n_int, d_cell_values,
            cuda.to_device(np.ascontiguousarray(int_owner, dtype=np.int64)),
            cuda.to_device(np.ascontiguousarray(int_neigh, dtype=np.int64)),
            cuda.to_device(np.ascontiguousarray(int_areas, dtype=np.float64)),
            cuda.to_device(np.ascontiguousarray(int_normals, dtype=np.float64)),
            d_grad,
        )

    n_bnd = bnd_owner.shape[0]
    if n_bnd > 0:
        _launch_1d(
            _gg_boundary_scatter_kernel, n_bnd,
            cuda.to_device(np.ascontiguousarray(bnd_owner, dtype=np.int64)),
            cuda.to_device(np.ascontiguousarray(bnd_areas, dtype=np.float64)),
            cuda.to_device(np.ascontiguousarray(bnd_normals, dtype=np.float64)),
            cuda.to_device(np.ascontiguousarray(bnd_values, dtype=np.float64)),
            d_grad,
        )

    _launch_1d(_gg_normalize_kernel, n_cells, d_cell_volumes, d_grad)
    return d_grad.copy_to_host()


def barth_jespersen_limiter_gpu(
    cell_values: np.ndarray, grad: np.ndarray, owner: np.ndarray,
    face_centers: np.ndarray, cell_centroids: np.ndarray,
    int_owner: np.ndarray, int_neigh: np.ndarray, int_face_idx: np.ndarray,
) -> np.ndarray:
    """签名与 fvm_gradients_kernels._barth_jespersen_limiter_kernel 一致。

    ⚠️ 未经真实 GPU 硬件验证，见本文件模块级文档字符串。
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA GPU not available in this environment")
    n_cells, n_vars = cell_values.shape
    n_faces = face_centers.shape[0]

    d_cell_values = cuda.to_device(np.ascontiguousarray(cell_values, dtype=np.float64))
    d_grad = cuda.to_device(np.ascontiguousarray(grad, dtype=np.float64))
    d_face_centers = cuda.to_device(np.ascontiguousarray(face_centers, dtype=np.float64))
    d_cell_centroids = cuda.to_device(np.ascontiguousarray(cell_centroids, dtype=np.float64))
    d_owner = cuda.to_device(np.ascontiguousarray(owner, dtype=np.int64))
    d_int_owner = cuda.to_device(np.ascontiguousarray(int_owner, dtype=np.int64))
    d_int_neigh = cuda.to_device(np.ascontiguousarray(int_neigh, dtype=np.int64))
    d_int_face_idx = cuda.to_device(np.ascontiguousarray(int_face_idx, dtype=np.int64))

    d_u_max = cuda.device_array((n_cells, n_vars), dtype=np.float64)
    d_u_min = cuda.device_array((n_cells, n_vars), dtype=np.float64)
    _launch_1d(_bj_minmax_init_kernel, n_cells, d_cell_values, d_u_max, d_u_min)

    n_int = int_owner.shape[0]
    if n_int > 0:
        _launch_1d(_bj_minmax_scatter_kernel, n_int, d_cell_values, d_int_owner, d_int_neigh, d_u_max, d_u_min)

    d_phi = cuda.to_device(np.ones((n_cells, n_vars), dtype=np.float64))
    _launch_1d(
        _bj_constrain_all_faces_kernel, n_faces, d_owner, d_cell_values, d_grad,
        d_u_max, d_u_min, d_face_centers, d_cell_centroids, d_phi,
    )
    if n_int > 0:
        _launch_1d(
            _bj_constrain_neighbour_kernel, n_int, d_int_neigh, d_int_face_idx, d_cell_values, d_grad,
            d_u_max, d_u_min, d_face_centers, d_cell_centroids, d_phi,
        )

    return d_phi.copy_to_host()
