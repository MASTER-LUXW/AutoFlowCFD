"""AutoFlowCFD V2.0 - 逐面跨单元插值矩阵 -> 模板表 + 逐面编号。

## 为什么（2026-09-29）

`nb_src0_mat` / `ow_src0_mat`（邻居一侧多项式在本侧面通量点上的求值矩阵，
`(n_faces, n_fp, n_sps)` float64）是 P2/P3 几何内存的最大头：plate_demo（40.3 万面）
P3 两份合计 6.15 GiB，占 P3 几何的一半、P3 残差求值峰值的三分之一。

直边协调网格上，这个矩阵只由拓扑决定：两侧面参数化之间差一个顶点排列，邻居的
面坐标因此只能取有限个值；逐面差异只是 Newton 反求面坐标的舍入。实测（plate_demo
P2，40.3 万面）按量子 q 取整后的唯一矩阵数：

    q = 1e-13  20459     q = 1e-11  13829     q = 1e-9   254
    q = 1e-12  16600     q = 1e-10   4699     （逐位唯一 363598）

取 `q = FACE_MATRIX_QUANTUM = 1e-13`：同一模板内任意两个矩阵逐元素相差小于 q（舍入
误差量级），P3 这两份降到约 0.35 GiB。曲边或非协调面上矩阵真的逐面不同时各自成一个
模板，表示对任何网格都成立，只是协调直边网格上表很小。

消费方一律按 `tpl[tid[f]]` 取第 f 个面的矩阵（numba 核里是视图，不拷贝）。
"""

import numpy as np

#: 归并量子：同一模板内逐元素差异的上界。
FACE_MATRIX_QUANTUM = 1e-13

#: 取整键用 int64：|元素| / q 必须小于 2**63，即 |元素| < 9e5（插值权重远小于此）。
_MAX_ABS_ENTRY = 9.0e5

_CHUNK_FACES = 20000


def deduplicate_face_matrices(mats: np.ndarray, quantum: float = FACE_MATRIX_QUANTUM):
    """`(n_faces, n_fp, n_sps)` -> `(tpl (n_tpl, n_fp, n_sps), tid (n_faces,) int32)`。

    `mats[f]` 与 `tpl[tid[f]]` 逐元素相差小于 `quantum`；每个模板取首次出现的成员。
    """
    n = mats.shape[0]
    tid = np.empty(n, dtype=np.int32)
    if n == 0:
        return np.empty_like(mats), tid
    peak = float(np.abs(mats).max())
    if peak >= _MAX_ABS_ENTRY:
        raise ValueError(f"插值矩阵元素 {peak:.3e} 超出取整键的范围（< {_MAX_ABS_ENTRY:.0e}）")
    index = {}
    reps = []
    for a in range(0, n, _CHUNK_FACES):
        keys = np.round(mats[a:a + _CHUNK_FACES] / quantum).astype(np.int64)
        for i in range(keys.shape[0]):
            k = keys[i].tobytes()
            t = index.get(k)
            if t is None:
                t = len(reps)
                index[k] = t
                reps.append(a + i)
            tid[a + i] = t
    return np.ascontiguousarray(mats[np.asarray(reps, dtype=np.int64)]), tid
