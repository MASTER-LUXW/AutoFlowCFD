"""AutoFlowCFD V2.0 - 按边界组预建的查表：Dirichlet 值、镜像法向、BJ 判据包络

从 `src/autoflowcfd/core/fr_solver/boundary.py`(原 506 行)拆出(2026-09-24, 项目"单文件不超 500 行"规范)。**纯搬家, 逻辑未改**。
"""


import numpy as np

from loguru import logger


from .constants import _NO_DIRICHLET


def build_boundary_dirichlet_table(provider, n_faces: int, n_var: int = 5):
    """给 BJ 越界判据构造逐边界面的**物理边界值**表，(n_faces, n_var)。

    为什么需要它（完整推导与实测见
    `fr_operators/bounds_sensor.py::compute_bounds_violation_mask` 的
    `bnd_dirichlet` 参数文档）：边界面不提供"邻居单元均值"，单纯排除会让
    该单元在那个方向上的包络变成**单侧**的，从而破坏判据的设计不变量
    "线性场恒不触发"。后果是贴壁单元被结构性误判——实测贴壁层命中率
    14.393%（全域只有 0.400%），壁面剪应力被压掉 14 倍
    （Blasius 平板 du/dy 93.0 vs `off` 的 1312.7，cf -93.37% vs -6.33%）。

    单邻居情形下仅凭单元均值**无法**区分"陡峭单调"与"本单元是异常值"
    （两条替代方案已被实测否掉，见那边文档），必须引入边界条件这一份
    外部信息。本函数就是那份信息。

    ## 各边界类型给什么

    * **无滑移壁面（WALL 且 `is_no_slip`）**：动量三列给 `0.0`。静止壁
      上 `rho*u_wall = 0`，与密度无关，所以这一项是精确的、不需要知道
      壁面密度。密度与总能没有 Dirichlet 值（本项目的壁面都是绝热壁，
      没有等温壁这种 BC 类型），留 NaN。
      这一档正是上面那个缺陷的全部来源：边界层的动量剖面单调终止于壁面。
    * **其余全部类型留 NaN**（退回排除，与修复前逐位一致）。逐项理由：
        - `SYMMETRY` / 滑移壁：**这一条我先前的论证是错的，已更正
          （2026-09-18）**。原话是"法向分量的真实镜像均值是 `-m_n`，而
          对称面上 `m_n` 物理上为零，两者一致"。`m_n = 0` 只在**面上
          那一点**成立；镜像邻居的**单元均值** `-<m_n>` 一般不为零
          （`<m_n>` 约等于 `(h/2) * d(m_n)/dn`，而对称面上
          `dw/dz = -(du/dx + dv/dy)` 一般非零）。所以"排除"在法向动量
          那一列**不是**无操作，它留下的正是与无滤移壁同样结构的单侧
          包络。
          正确的外侧贡献是把 owner 的动量均值对面法向作镜像：
          `m' = m - 2 (m . n) n`。它**依赖解**（每个 stage 都不同），
          所以放不进这张静态表 —— 由 `compute_bounds_violation_mask`
          的 `bnd_mirror_normal` 参数在判据内部就地算，见那边文档。
          本函数因此对 SYMMETRY / 滑移壁仍然留 NaN，但**不再声称那是
          精确的**：缺的那一半由镜像法向补。
        - `INLET` / `FARFIELD`：来流场在边界附近是均匀的，没有终止于
          边界的强梯度，单侧包络不构成约束。SEM 合成湍流入口逐面逐步
          变化，不存在可预先制表的定值。
        - `OUTLET`：只给定静压，动量无 Dirichlet；且出口处剖面的法向
          邻居都在内部，不存在单侧问题。
    * **移动壁面**（`wall_velocity` 存在且**有非零分量**；静止壁在配置里
      写作 `[0,0,0]`，按静止处理）：留 NaN 并打一次警告。
      `rho*u_wall` 需要壁面密度，而这里拿不到逐面密度；给个错的值比
      退回排除更糟。本项目目前没有移动壁算例。

    Args:
        provider: `build_boundary_ghost_provider` 的返回值。读它的
            `group_code`（(n_faces,) 每面的边界组编码，内部面为 -1）与
            `code_to_config`/`default_config`。**分布式路径无需特殊处理**：
            那几条路径已经把 `provider.group_code` 重切到
            `partition.local_faces`（见 `core/mpi/distributed_mesh_loader.py`
            与 `distributed_order_continuation.py`），与 `dist_flat_face`
            同一索引空间。
        n_faces: 面数，必须与判据里 `owner_cell` 的长度一致
        n_var: 变量数（守恒变量，通常 5）

    Returns:
        (n_faces, n_var) float64；`provider` 为 None 或没有 `group_code`
        时返回 None（调用方据此退回"排除"行为）。

    Raises:
        ValueError: `provider.group_code` 长度与 `n_faces` 不符 —— 索引
            空间对不上时静默继续会让整张表错位，那是一个看不出来的错误。
    """
    if provider is None:
        return None
    group_code = getattr(provider, "group_code", None)
    if group_code is None:
        return None
    gc = np.asarray(group_code)
    if gc.size != n_faces:
        raise ValueError(
            f"boundary_ghost_provider.group_code 长度 {gc.size} 与 n_faces "
            f"{n_faces} 不符——两者必须处在同一个面索引空间，否则整张"
            f"Dirichlet 表会错位"
        )
    if n_var < 4:
        raise ValueError(f"n_var={n_var} 至少要覆盖 3 个动量分量")

    table = np.full((n_faces, n_var), _NO_DIRICHLET, dtype=np.float64)
    code_to_config = getattr(provider, "code_to_config", {}) or {}
    default_config = getattr(provider, "default_config", None) or {}

    moving_wall_seen = False
    codes = np.unique(gc)
    for code in codes:
        cfg = code_to_config.get(int(code), default_config)
        if not cfg or cfg.get("type") != "WALL":
            continue
        # 按**物理事实**判断，不是按幽灵态构造开关（见
        # `build_boundary_ghost_provider` 的 type_map 里那段说明）：
        # WMLES 激活时 `is_no_slip` 被关掉以免与壁面模型双重计权，但那面
        # 墙物理上仍然是静止的不可穿透固壁、动量剖面照样单调终止于
        # `rho*u = 0`。用 `is_no_slip` 判会让 WMLES 算例整张表变成 NaN。
        #
        # 回退到 `is_no_slip` 只是为了兼容直接构造 `code_to_config` 的
        # 调用方（例如测试替身、`bc_overrides` 走底层名的算例）——它们
        # 不经过 type_map，没有 `physical_no_slip` 这个键。
        if not cfg.get("physical_no_slip", cfg.get("is_no_slip", True)):
            # 真正的滑移壁：切向速度不为零，动量没有 Dirichlet 值。
            continue
        # 静止壁在配置里写作 `wall_velocity=[0,0,0]` 而不是 None
        # （见 build_boundary_ghost_provider 的 type_map），所以判据是
        # "全零即静止"，不能写成 `is not None`——那会把所有真实算例的
        # 壁面都当成移动壁跳过（2026-09-18 实测踩到：表里 5 列全是 NaN，
        # 贴壁层命中率毫无变化）。
        wv = cfg.get("wall_velocity")
        if wv is not None and np.any(np.asarray(wv, dtype=np.float64) != 0.0):
            moving_wall_seen = True
            continue
        table[gc == code, 1:4] = 0.0

    if moving_wall_seen:
        logger.warning(
            "[BJ 判据] 检测到给定了 wall_velocity 的移动壁面。构造动量的"
            "Dirichlet 值需要逐面壁面密度（rho*u_wall），这里拿不到，"
            "因此这些面退回'排除'——那会让贴壁单元的邻域包络在壁面方向上"
            "单侧收窄、可能被误判（静止壁的定量后果是壁面剪应力被压掉 "
            "14 倍）。若本算例确实有移动壁且用到 sensor+bounds 门控，"
            "需要先把逐面壁面密度接进来。"
        )
    return table


def build_boundary_mirror_normals(provider, face_normal, n_faces: int):
    """给 BJ 越界判据构造逐边界面的**镜像法向**表，(n_faces, 3)。

    与 `build_boundary_dirichlet_table` 是同一个缺陷的两半：边界面不提供
    "邻居单元均值"，单纯排除会让包络单侧收窄、破坏"线性场恒不触发"这个
    设计不变量。无滑移壁那一半由静态 Dirichlet 表补；**对称面与滑移壁**
    这一半补不了 —— 它们的外侧"邻居"是本单元的镜像，动量均值
    `m' = m - 2 (m . n) n` 依赖解、每个 stage 都不同。所以这里只给出
    **法向**（纯几何、一次构造），镜像本身由
    `fr_operators/bounds_sensor.py::compute_bounds_violation_mask` 在判据
    内部就地算。

    这不是假想缺陷：本项目唯一有精确解的粘性算例（Blasius 平板）展向两面
    都是 SYMMETRY、顶面是滑移壁，而它的开放问题恰好就是展向 `w` 的非物理
    增长（见项目记忆 `blasius-spanwise-w-open`）。

    ## 哪些面算镜像面

    * `SYMMETRY`；
    * **滑移壁**（`type == "WALL"` 且 `physical_no_slip` 为假）—— 滑移壁的
      幽灵态构造与对称面**是同一个**（法向镜像反号 + 切向保持，完整论证见
      `build_boundary_ghost_provider` 的 type_map 注释），所以判据侧也必须
      用同一条规则，否则两个语义相同的边界会给出不同的门控结果。

    其余类型留 NaN（`INLET`/`FARFIELD`/`OUTLET` 的外侧不是镜像；无滑移壁
    由 Dirichlet 表负责，两张表互斥）。

    Args:
        provider: `build_boundary_ghost_provider` 的返回值
        face_normal: 面法向，接受两种形状：
              (n_faces, 3)        —— `FRFaceConnectivity.normal`（单机路径）
              (n_faces, n_fp, 3)  —— `flat.true_normal`（分布式路径，逐
                                     通量点），在这里按面求平均再单位化，
                                     归约成同一个 (n_faces, 3)。
            为什么归约放在这里：BJ 判据是**单元均值**级别的，一个面只需要
            一个代表法向；四条后端各自 reshape 就是四份要同步的实现。
            平面上各通量点法向本来就完全相同（本项目的对称面/滑移壁都是
            平面），归约是恒等变换；曲面上平均方向是这个粒度下唯一有意义的
            代表值。
            朝向无关紧要：镜像公式 `m - 2(m.n)n` 里 `n` 出现两次，对 `n`
            与 `-n` 同值。**不需要预先单位化**：单位化是镜像公式的前提，
            由消费方 `compute_bounds_violation_mask` 一处完成（按面平均
            之后在曲面上本来也不再是单位的）。
        n_faces: 面数，必须与判据里 `owner_cell` 的长度一致

    Returns:
        (n_faces, 3) float64；`provider` 为 None、没有 `group_code`、
        `face_normal` 为 None，或没有任何镜像面时返回 None（调用方据此
        退回"排除"行为，与修复前逐位一致）。

    Raises:
        ValueError: `group_code` 或 `face_normal` 的长度与 `n_faces` 不符。
    """
    if provider is None or face_normal is None:
        return None
    group_code = getattr(provider, "group_code", None)
    if group_code is None:
        return None
    gc = np.asarray(group_code)
    if gc.size != n_faces:
        raise ValueError(
            f"boundary_ghost_provider.group_code 长度 {gc.size} 与 n_faces "
            f"{n_faces} 不符——两者必须处在同一个面索引空间"
        )
    nrm = np.asarray(face_normal, dtype=np.float64)
    if nrm.ndim == 3 and nrm.shape[0] == n_faces and nrm.shape[2] == 3:
        # 逐通量点 -> 逐面（见 face_normal 文档）。平面上是恒等变换。
        nrm = nrm.mean(axis=1)
    if nrm.shape != (n_faces, 3):
        raise ValueError(
            f"face_normal 形状 {np.asarray(face_normal).shape} 应为 "
            f"(n_faces={n_faces}, 3) 或 (n_faces, n_fp, 3)")

    code_to_config = getattr(provider, "code_to_config", {}) or {}
    default_config = getattr(provider, "default_config", None) or {}

    out = np.full((n_faces, 3), _NO_DIRICHLET, dtype=np.float64)
    any_mirror = False
    for code in np.unique(gc):
        cfg = code_to_config.get(int(code), default_config)
        if not cfg:
            continue
        btype = cfg.get("type")
        if btype == "SYMMETRY":
            is_mirror = True
        elif btype == "WALL":
            # 与 Dirichlet 表用同一个判据（同一个事实来源）：物理滑移壁
            # 才是镜像面；物理无滑移壁由那张表负责，两者互斥。
            is_mirror = not cfg.get(
                "physical_no_slip", cfg.get("is_no_slip", True))
        else:
            is_mirror = False
        if not is_mirror:
            continue
        sel = gc == code
        out[sel] = nrm[sel]
        any_mirror = True

    return out if any_mirror else None


def make_bj_boundary_tables(get_provider, n_faces: int, face_normal=None,
                            to_device=None, n_var: int = 5):
    """构造 BJ 越界判据要的两张边界表的**惰性取值器**（零参可调用）。

    返回的可调用给出 `(bnd_dirichlet, bnd_mirror_normal)`，首次调用时求值
    并缓存。两张表一起给，因为它们是**同一个缺陷的两半**（边界面没有邻居
    单元均值 -> 包络单侧收窄 -> "线性场恒不触发"这个设计不变量被破坏）：
    无滑移壁那一半是静态值表，对称面/滑移壁那一半是镜像法向。

    ## 为什么是工厂而不是各后端各写一份

    四条后端（cpu-single / cpu-mpi / gpu-single / gpu-mpi）此前各有一份
    几乎相同的惰性闭包。本项目已经多次因为"两份实现只改了一份"出真实
    缺陷，而这里尤其危险：漏掉一条后端不会报错，只会让那条后端的贴壁
    单元被结构性误判（实测代价 cf -93%），在残差日志里完全看不出来。

    ## 为什么必须惰性

    多 GPU 的 `_init_modal_filter_distributed` 在 `boundary_ghost_provider`
    建好**之前**就被调用；CPU 分布式的 provider 又挂在惰性 `local_solver`
    属性上。两张表只依赖静态 BC 配置与几何，首次施加滤波时求值即可。

    Args:
        get_provider: 零参可调用，返回 `build_boundary_ghost_provider` 的
            结果（可能返回 None —— 此时两张表都是 None，判据退回"排除"）
        n_faces: 面数，必须与判据里 `owner_cell` 的长度一致
        face_normal: (n_faces, 3) 单位法向，或 None（None 时不构造镜像表）
        to_device: 可选，把 numpy 数组搬到计算设备的可调用（GPU 后端传
            `cp.asarray` 的设备上下文封装）。两张表都会经过它。
        n_var: 守恒变量数

    Returns:
        零参可调用 -> `(dirichlet, mirror_normal)`，各为数组或 None。
    """
    cache = []

    def _resolve():
        if not cache:
            prov = get_provider()
            d = build_boundary_dirichlet_table(prov, n_faces, n_var)
            m = build_boundary_mirror_normals(prov, face_normal, n_faces)
            if to_device is not None:
                d = None if d is None else to_device(d)
                m = None if m is None else to_device(m)
            cache.append((d, m))
        return cache[0]

    return _resolve
