"""AutoFlowCFD V2.0 - BJ 型越界判据：解点值超出顶点邻域极值区间即标记

从 `src/autoflowcfd/core/fr_operators/bounds_sensor.py`(原 506 行)拆出(2026-09-24, 项目"单文件不超 500 行"规范)。**纯搬家, 逻辑未改**。
"""


import numpy as np

from autoflowcfd.core.utils.array_module import array_module as _array_module

from ..vertex_stencil import accumulate_vertex_envelope

# 档位解析（`AFCFD_TROUBLED_SENSOR`）已拆到 `troubled_sensor_mode.py`
# （2026-09-19，项目"单文件不超 500 行"规范）。这里 re-export，全仓库
# `from ...bounds_sensor import resolve_troubled_sensor` 不用改。
from .constants import DEFAULT_BOUNDS_ABS_FRAC, DEFAULT_BOUNDS_REL_TOL
from .scatter import _scatter_minmax


def compute_bounds_violation_mask(
    field_nodal: np.ndarray,
    owner_cell: np.ndarray,
    neighbor_cell: np.ndarray,
    is_boundary: np.ndarray,
    *,
    rel_tol: float = DEFAULT_BOUNDS_REL_TOL,
    abs_frac: float = DEFAULT_BOUNDS_ABS_FRAC,
    ref_scales=None,
    bnd_dirichlet=None,
    bnd_mirror_normal=None,
    row_is_prism=None,
    n_real_prism=None,
    n_real_tet=None,
    vertex_stencil=None,
) -> np.ndarray:
    """逐单元判定"解点值越出了面邻居均值区间" —— **纯数组接口**。

    与 `artificial_viscosity.compute_troubled_cell_mask`（Persson-Peraire）
    平行的第二个判据，接口风格刻意保持一致（纯数组、不需要 solver / ops），
    这样两者可以在同一个门控入口里互换或取并集，且 CPU 单机 / CPU MPI
    local 排列 / GPU 三条路径都能用同一个内核。

    与 Persson-Peraire 的关键差异：本判据**不依赖模态分解**，因此没有
    "order=1 时顶模态就是全部非常数内容"那个退化（见模块文档）。

    Args:
        field_nodal: (n_cells, n_sps) 单标量场，或 (n_cells, n_sps, n_var)
            多变量场（多变量时逐变量判定后取**并集**——任一变量越界即
            标记该单元）。
        owner_cell: (n_faces,) 面的 owner 单元索引
        neighbor_cell: (n_faces,) 面的 neighbor 单元索引；边界面该项不被
            读取（可以是任意占位值，例如 -1）
        is_boundary: (n_faces,) 布尔，True=边界面。边界面不提供"邻居单元
            均值"——用**幽灵态**会把边界条件本身的物理跳跃（壁面镜像把
            法向速度取反）误判成越界。缺失那一侧要用 `bnd_dirichlet`
            给出的**物理边界值**补上，见该参数。
        vertex_stencil: 可选 `VertexStencil`（`vertex_stencil.py::
            build_vertex_stencil` 的产物）。给了它就**额外**把顶点邻域
            （共享任一顶点的全部单元）的均值计入包络。

            **这是 2026-09-19 对"欠解析光滑场上标记 100%"那条缺陷的修复**：
            面邻居在三维四面体上只有 4 个，其单元均值不能把本单元夹住，
            漏掉的 O(h|grad u|) 合法光滑变化被当成越界。实测（TGV 解析
            初场、三档加密）：

                模板       越界量观测阶      默认容差标记比例（n=16）
                面邻居     0.72 / 0.96          100.00%
                顶点邻居   1.58 / 1.28            6.18%

            中位越界从 8.04e-02 降到**恰好 0**。完整数据与那条被否掉的
            TVB 方案见 `vertex_stencil.py` 模块文档。

            `None` 时只用面邻居 —— 那是 2026-09-19 之前的行为，保留是
            为了让合成单元测试（没有 `_fixed_*_conn` 的网格）与历史对照
            仍然可跑；生产路径由 `fr_solver/filter.py` 负责传入。
        bnd_dirichlet: 可选 (n_faces, n_var)，边界面上该变量的物理边界
            值（守恒变量口径）；非有限值（NaN/inf）= 该面该变量没有
            Dirichlet 值，退回"排除"（对 `nb_max`/`nb_min` 的初值
            `cell_mean` 恰好是无操作，所以与不给它时逐位一致）。

            **为什么必须有它（2026-09-18 真实缺陷）**：单纯"排除"会让该
            单元在那个方向上的包络变成**单侧**的，从而破坏本判据的设计
            不变量"线性场恒不触发"——贴壁单元靠壁那个解点必然低于本单元
            均值、也低于上方邻居均值，而包络下界恰好就是本单元均值。

            定量（Blasius 平板，有精确解，nx=16，400 步，**固定 CFL 0.03
            同步数**的干净对照；壁面剪应力用胞内两点斜率提取，该提取已用
            "把解析解直接放到同一网格上"验证到中位 -0.49%、最大 1.82%）：

                off / sensor+persson（掩码恒空）  du/dy 1312.7  cf  -6.33%
                sensor+bounds（修复前）           du/dy   93.0  cf -93.37%

            全域标记率只有 0.400%，但**贴壁那一层是 14.393%**（集中 36
            倍，逐变量 rho_u 1.401% / rho_w 0.698%），而壁面剪应力恰好
            只由这一层决定：单元内扩散重建梯度约需 `h^2/nu/dt ~ 26` 步，
            而每 stage 14% 的投影率平均每 2 步就再清一次，梯度永远建不
            起来。这是 Barth-Jespersen 的已知性质——它是**单调性**判据，
            强梯度边界层本来就越出邻居均值包络（经典的"BJ 在光滑极值处
            损失精度"）。

            **两条被自己数据否掉的替代方案**（不要再试）：
              1. 边界侧用 owner 自身解点极值撑开包络 —— 边界单元恒不
                 可能被标记，而 `nz=1` 的准二维算例里每个单元都贴着两个
                 对称面，实测标记率变成全域 0.000%，判据整个失效；
              2. 把内部包络对本单元均值作**镜像**补上缺失一侧 —— 线性场
                 不变量恢复了（贴壁层 14.393% -> 2.099%），但贴边界单元
                 内部一个 50 倍的**真实**过冲也抓不到：镜像半宽由"本单元
                 均值与邻居均值之差"决定，而单元自身就是异常值时这个差
                 正好被它自己撑大。单邻居情形下仅凭单元均值在信息上
                 **无法**区分"陡峭单调"与"本单元是异常值"——必须引入
                 边界条件这一份外部信息。

            构造见 `core/fr_solver/boundary.py::build_boundary_dirichlet_table`。
        bnd_mirror_normal: 可选 (n_faces, 3)，**镜像型**边界面的单位外法向；
            非有限行 = 该面不适用镜像规则。

            **为什么静态的 `bnd_dirichlet` 不够**：对称面与滑移壁的外侧
            "邻居"是本单元的镜像，它的**单元均值**对动量分量是
            `m' = m - 2 (m . n) n` —— 依赖解，每个 stage 都不同，放不进
            静态表。我先前论证过"对称面上排除即精确"，**那是错的**：
            `m_n = 0` 只在**面上那一点**成立，镜像邻居的单元均值
            `-<m_n>` 一般不为零（`<m_n>` 约等于 `(h/2) d(m_n)/dn`，而
            对称面上 `dw/dz = -(du/dx + dv/dy)` 一般非零）。于是法向动量
            那一列留下的正是与无滑移壁同样结构的**单侧包络**。

            这不是假想：本项目唯一有精确解的粘性算例（Blasius 平板）
            展向两面都是 SYMMETRY、顶面是滑移壁，`nz=1` 时**每个**单元的
            `rho_w` 列包络都是单侧的 —— 而那个算例的开放问题恰好就是
            展向 `w` 的非物理增长。

            标量与切向分量的镜像均值**恰好等于**本单元均值，所以只有
            动量三列（索引 1..3）需要处理；其余列的镜像贡献对包络初值
            `cell_mean` 是无操作，直接跳过。
        row_is_prism / n_real_prism / n_real_tet: 可选，**真实自由度**的
            行掩码与两类单元各自的真实槽位数。三个要么全给、要么全不给。

            ## 为什么必须有（2026-09-18 实测的真实缺陷）

            原生基的零填充槽位**残差恒为零**（"零填充块对角"不变量刻意
            保证），于是那些槽位的解**冻结在初值**，而真实槽位在演化 ——
            实测 TGV（P2 四面体）30 步后填充值已经越出真实槽位区间达
            **14% 区间宽**。而本判据用 `field.max(axis=1)` /
            `.min(axis=1)` / `.mean(axis=1)`：不排除填充槽位就是在**冻结
            的馊值**上统计单元极值与均值。

            与 `fr/native_padding.py::real_sps_per_cell`（"哪些槽位是真的"
            的唯一判据来源）配套；本判据这个调用点此前漏掉了它，同一
            缺陷家族的另外两处是机制3（`troubled_cell.py`）与那两个
            `reduce_*_over_real_sps` 归约。

            **如实记录一条被自己数据否掉的假设**：我曾以为这就是 BJ 判据
            在 TGV 上 100% 误标的原因。**不是** —— step 0（填充与真实还
            完全一致时）就已经 100%，而且只用真实槽位重算仍然 100%。
            100% 误标是 BJ 判据在**欠分辨光滑场**上的固有行为（正弦场每
            波长只有 4 个单元、曲率强，单元内极值确实超出邻域均值包络），
            与填充无关。填充污染是另一条独立的、真实的缺陷。
        rel_tol: 相对容差系数，见模块文档
        abs_frac: 绝对地板系数（乘以下面的参考量级），见模块文档
        ref_scales: 可选的 (n_var,) **来流参考量级**（例如
            `[rho_inf, rho_inf*vel_inf, rho_inf*vel_inf, rho_inf*vel_inf,
            p_inf]`，与 `fr_solver/residual_diagnostics.py::
            _reference_scales` 同一套构造）。**强烈建议给出**：不给时
            退化为用该变量自身单元均值的 RMS 做尺度，而那对"大部分域内
            均值≈0、RMS 被局部区域主导"的变量（rho_v / rho_w）是没有
            意义的尺度——2026-09-16 实测证实了这一点，见模块文档
            "第一版用全场 RMS 做尺度为什么不行"一节。

    Returns:
        (n_cells,) 布尔掩码，True = 该单元越界。

    Raises:
        ValueError: 形状不自洽（不静默广播——静默广播会让一个形状 bug
            变成"判据恒不触发"，而恒不触发的门控在日志里看起来一切正常）。
    """
    xp = _array_module(field_nodal, owner_cell, neighbor_cell)
    field = xp.asarray(field_nodal)
    if field.ndim == 2:
        field = field[:, :, None]
    if field.ndim != 3:
        raise ValueError(
            f"field_nodal 必须是 (n_cells, n_sps) 或 (n_cells, n_sps, n_var)，"
            f"收到 {np.asarray(field_nodal).shape}"
        )
    n_cells = field.shape[0]

    owner = xp.asarray(owner_cell)
    neigh = xp.asarray(neighbor_cell)
    bnd = xp.asarray(is_boundary).astype(bool)
    if not (owner.shape == neigh.shape == bnd.shape) or owner.ndim != 1:
        raise ValueError(
            f"owner_cell/neighbor_cell/is_boundary 必须是同长度一维数组，"
            f"收到 {owner.shape} / {neigh.shape} / {bnd.shape}"
        )
    if owner.size and (int(owner.max()) >= n_cells or int(owner.min()) < 0):
        raise ValueError(
            f"owner_cell 越界：[{int(owner.min())}, {int(owner.max())}] "
            f"超出 [0, {n_cells})"
        )

    interior = ~bnd
    o_i = owner[interior]
    n_i = neigh[interior]
    if n_i.size and (int(n_i.max()) >= n_cells or int(n_i.min()) < 0):
        raise ValueError(
            f"内部面的 neighbor_cell 越界：[{int(n_i.min())}, {int(n_i.max())}] "
            f"超出 [0, {n_cells})"
        )

    n_var = field.shape[2]
    o_b = None
    bd_b = None
    if bnd_dirichlet is not None:
        bd = xp.asarray(bnd_dirichlet, dtype=xp.float64)
        if bd.shape != (owner.size, n_var):
            raise ValueError(
                f"bnd_dirichlet 形状 {bd.shape} 应为 "
                f"(n_faces={owner.size}, n_var={n_var})"
            )
        o_b = owner[bnd]
        bd_b = bd[bnd]

    o_mir = None
    if bnd_mirror_normal is not None:
        nrm_all = xp.asarray(bnd_mirror_normal, dtype=xp.float64)
        if nrm_all.shape != (owner.size, 3):
            raise ValueError(
                f"bnd_mirror_normal 形状 {nrm_all.shape} 应为 "
                f"(n_faces={owner.size}, 3)"
            )
        if n_var < 4:
            raise ValueError(
                f"bnd_mirror_normal 需要至少 4 个变量（动量占索引 1..3），"
                f"收到 n_var={n_var}"
            )
        sel_mir = bnd & xp.all(xp.isfinite(nrm_all), axis=1)
        if bool(xp.any(sel_mir)):
            o_mir = owner[sel_mir]
            nrm = nrm_all[sel_mir]
            # 单位化在这里一处完成（镜像公式 `m - 2(m.n)n` 只对单位法向
            # 成立）：构造方给的可能是逐通量点法向按面平均的结果，曲面上
            # 那不是单位向量。规模是边界面数、每 stage 一次，可忽略。
            nlen = xp.sqrt((nrm ** 2).sum(axis=1, keepdims=True))
            if not bool(xp.all(nlen > 0.0)):
                raise ValueError(
                    "bnd_mirror_normal 里存在零长度法向——镜像公式 "
                    "m - 2(m.n)n 对它没有定义，静默跳过会让那些面退回"
                    "单侧包络而在日志里看不出来")
            nrm = nrm / nlen

    if ref_scales is not None:
        ref = np.asarray(ref_scales, dtype=np.float64).ravel()
        if ref.size != n_var:
            raise ValueError(
                f"ref_scales 长度 {ref.size} 与变量数 {n_var} 不符"
            )
    else:
        ref = None

    # 三次全场归约一次算完**所有**变量（此前是逐变量 mean/max/min，
    # 5 个变量 15 次全场遍历）。这条判据是四条后端的**默认**路径、每个
    # RK stage 调一次，36 万单元实测那 15 次归约占 0.11 s/stage。
    #
    # **只统计真实槽位**（见 `row_is_prism` 参数文档）。实现方式是"两段
    # 切片 + 小尺寸合并"而不是掩码数组：真实槽位恒为**前缀**，所以
    #   ① 先在两类单元共有的前 `n_lo` 个槽位上归约（纯切片、零拷贝）；
    #   ② 再在 `[n_lo, n_hi)` 上归约，只对真实槽位更多的那类行合并。
    # 临时量只有 `(n_rows, n_var)` 量级 —— 用 `where(mask, field, ±inf)`
    # 那种写法会物化一个与 `field` 同样大的数组（79 万单元 P2 下 850 MB）。
    if row_is_prism is None:
        cell_means = field.mean(axis=1)      # (n_cells, n_var)
        cell_maxs = field.max(axis=1)
        cell_mins = field.min(axis=1)
    else:
        rip = xp.asarray(row_is_prism, dtype=bool)
        if rip.shape != (n_cells,):
            raise ValueError(
                f"row_is_prism 形状 {rip.shape} 应为 (n_rows={n_cells},)")
        n_rp = int(n_real_prism)
        n_rt = int(n_real_tet)
        n_sps_here = field.shape[1]
        for nm, v in (("n_real_prism", n_rp), ("n_real_tet", n_rt)):
            if not (1 <= v <= n_sps_here):
                raise ValueError(
                    f"{nm}={v} 超出 [1, n_sps={n_sps_here}] —— 真实槽位数"
                    f"必须落在 SP 轴长度内，越界说明上游的阶数/基与数组"
                    f"不自洽（按错的数统计会把冻结的填充值算进极值）")
        n_lo, n_hi = min(n_rp, n_rt), max(n_rp, n_rt)
        # 真实槽位更多的那一类：`row_hi` 为 True 的行多统计 [n_lo, n_hi)
        row_hi = rip if n_rp >= n_rt else ~rip
        sum_lo = field[:, :n_lo].sum(axis=1)
        cell_maxs = field[:, :n_lo].max(axis=1)
        cell_mins = field[:, :n_lo].min(axis=1)
        if n_hi > n_lo:
            ext = field[:, n_lo:n_hi]
            m2 = row_hi[:, None]
            sum_lo = xp.where(m2, sum_lo + ext.sum(axis=1), sum_lo)
            cell_maxs = xp.where(m2, xp.maximum(cell_maxs, ext.max(axis=1)),
                                 cell_maxs)
            cell_mins = xp.where(m2, xp.minimum(cell_mins, ext.min(axis=1)),
                                 cell_mins)
        cnt = xp.where(row_hi, float(n_hi), float(n_lo))[:, None]
        cell_means = sum_lo / cnt

    # 镜像型边界的动量均值：m' = m - 2 (m . n) n。标量与切向分量的镜像
    # 均值恰好等于本单元均值（对包络初值是无操作），所以只算动量三列，
    # 而且只算一次、不进逐变量循环。
    mir_mom = None
    if o_mir is not None:
        m = cell_means[o_mir, 1:4]
        dot = (m * nrm).sum(axis=1, keepdims=True)
        mir_mom = m - 2.0 * dot * nrm        # (n_mir, 3)

    mask = None
    for v in range(n_var):
        cell_mean = cell_means[:, v]
        # 邻域区间：自身均值 + 全部面邻居的均值。两个方向都要做——
        # 一条内部面同时是 owner 的邻居来源和 neighbor 的邻居来源。
        nb_max = cell_mean.copy()
        nb_min = cell_mean.copy()
        if o_i.size:
            _scatter_minmax(xp, nb_max, nb_min, o_i, cell_mean[n_i])
            _scatter_minmax(xp, nb_max, nb_min, n_i, cell_mean[o_i])
        if vertex_stencil is not None:
            # **顶点邻域**（共享任一顶点的全部单元）——面邻居在三维四面体
            # 上不能把本单元夹住，实测因此把 O(h|grad u|) 的合法光滑变化
            # 当成越界、在欠解析光滑场上标记 100% 的单元。完整实测数据
            # （含"经典 TVB 的 M h^2 修不了它"那条被否掉的方案）见
            # `vertex_stencil.py` 模块文档。
            accumulate_vertex_envelope(
                xp, _scatter_minmax, nb_max, nb_min, cell_mean,
                vertex_stencil)
        if bd_b is not None and o_b.size:
            # 边界面：有 Dirichlet 值的把它当"外侧均值"计入包络；NaN 的
            # 用 cell_mean 顶上 —— 而 nb_max/nb_min 的初值就是 cell_mean，
            # 所以那一支是无操作、与不给 bnd_dirichlet 时逐位一致。
            col = bd_b[:, v]
            val = xp.where(xp.isfinite(col), col, cell_mean[o_b])
            _scatter_minmax(xp, nb_max, nb_min, o_b, val)
        if mir_mom is not None and 1 <= v <= 3:
            _scatter_minmax(xp, nb_max, nb_min, o_mir, mir_mom[:, v - 1])
        if ref is not None:
            scale = float(ref[v])
        else:
            scale = float(xp.sqrt(xp.mean(cell_mean.astype(xp.float64) ** 2)))
        tol = rel_tol * (nb_max - nb_min) + abs_frac * max(scale, 1e-300)
        hit = (cell_maxs[:, v] > nb_max + tol) | (cell_mins[:, v] < nb_min - tol)
        mask = hit if mask is None else (mask | hit)

    return mask if mask is not None else xp.zeros(n_cells, dtype=bool)
