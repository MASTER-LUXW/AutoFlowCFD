"""
AutoFlowCFD V2.0 - GPU 版时间积分

* `GPUTimeIntegrator`：**不是**独立实现。SSP-RK stage 推进、双时间步、
  IMEX 全部继承 CPU 的 `time_integration.TimeIntegrator`（它们与数组模块
  无关，CuPy 数组原样走同一份代码）。
* `compute_local_cfl_step_gpu`：GPU 版局部 CFL 时间步长。

**2026-09-25 删除的 GPU 独立拷贝**（`enforce_positivity_gpu`、
`_ssp_rk_stage_step_gpu`、`gpu_time_integration_dual.py`、
`gpu_time_integration_imex.py`）：

1. 正性用的是逐点硬钳（密度钳 1e-6、速度钳 1e4 m/s、压力钳 1 Pa），CPU 侧
   已证实它不守恒、且是 plate_demo 第 4134 步一步爆炸到 1e53 的放大器；
2. 双时间步里 2026-08-23 在 CPU 侧修掉的四处 CFL 缺陷一处都没修；
3. 顺序是"先钳制后滤波"，离开 stage 的状态不保证可容许。

现在 GPU 与 CPU 走同一份积分器、同一个守恒限制器
（`time_integration/positivity/`，GPU 上用它的数组模块无关实现）。
"""

from autoflowcfd.core.fr_operators.flux_kernels import GAMMA
from autoflowcfd.core.gpu import get_cupy
from autoflowcfd.core.time_integration.base import TimeIntegrator, scheme_from_name


def compute_local_cfl_step_gpu(
    U, cell_volumes, owner_cell, neighbor_cell, is_boundary,
    normals, areas, cell_owner, cell_areas,
    cfl: float = 1.0, mu_eff=None, poly_order: int = 0,
    det_jacs_sp=None, metric_flux_scale_sp=None,
    mach_ref=None, return_physical_too: bool = False,
):
    """GPU 版局部 CFL 时间步长计算。

    dt_i = CFL * order_factor * V_i / sum_f (|u.n|+a) A_f

    谱半径用物理声速 a（2026-08-25 代码审查修复）：此前这里用
    Weiss-Smith 预处理声速 c_precond 替代物理声速，低马赫数下（Mach 0.1
    时 c_precond≈36 m/s vs a≈340 m/s）dt 被高估 ~10 倍，有效 CFL 远超
    SSP-RK3 稳定极限——与 CPU 侧 cfl.py::compute_local_time_step 的
    2026-08-24 修复完全同因（该次修复只改了 CPU 路径，GPU 路径被遗漏，
    此处把 GPU 拉回与 CPU 一致的物理声速）。原理：显式积分的是物理通量，
    稳定性由物理通量谱半径（|u_n|+a）决定；预处理只改善条件数，不改变
    稳定极限。order_factor 与 CPU 侧同一公式 1/(2p+1)。

    Args:
        U: CuPy 数组 (n_cells, 1, n_vars)——**必须是单 SP 切片**，见下方
            函数体开头的校验与说明
        cell_volumes: CuPy 数组 (n_cells,)
        owner_cell, neighbor_cell, is_boundary: 面连接关系
        normals: CuPy 数组 (n_faces, 3)
        areas: CuPy 数组 (n_faces,)
        cell_owner: 边界面 → cell 映射
        cell_areas: 边界面面积
        cfl: CFL 数
        mu_eff: 有效粘度（可选）
        poly_order: 当前多项式阶数，用于 1/(2p+1) 的阶数相关收紧，
            与 CPU 侧 cfl.py::compute_local_time_step 的
            order_factor_advective/order_factor_viscous 同一公式。
        det_jacs_sp: 本次调用对应 SP 的 |det(J)|，CuPy 数组 (n_cells,)
            （可选）。与 metric_flux_scale_sp 一起提供时，额外施加第三个
            几何/度量 CFL 限制——与 CPU 侧 cfl.py::compute_local_time_step
            的 dt_geometric 同一机制：坍缩坐标下同一单元内不同 SP 的
            det(J) 天然可以相差几百倍（Duffy 坍缩变换在 P>=2 时的固有
            性质），此前 GPU 路径完全没有这一限制，可能重新触发 CPU 侧
            已经修复过的同一类发散（项目记忆 "Tet collapsed-coord
            anisotropy"）。
        metric_flux_scale_sp: 本次调用对应 SP 的度量"通量面积"标度
            sum_m||adj(J)[:,m,:]||，CuPy 数组 (n_cells,)（可选，与
            det_jacs_sp 一起提供时才生效）。
        mach_ref: 不为 None 时，额外算一份**预处理后**的平均流 dt——
            把对流项与几何项里的声速换成有效声速 sqrt(beta^2)*a
            （beta^2 按速度模取，见 gpu_preconditioning.py）。这一份
            **只有在调用方真的把 Gamma 作用到平均流残差上时才可以使用**
            （见上方"谱半径用物理声速"那段记录的 2026-08-25 事故：
            只改 CFL 不改方程必然失稳）。粘性限制与声速无关，原样复用。
        return_physical_too: True 时返回 (dt_mean_flow, dt_physical)。
            湍流标量（k/omega）必须用后者——它们是被动输运量、不含声学
            模态，但其显式更新刻意没有 point-implicit 阻尼，保守起见不跟着
            放大步长（与 CPU 侧 cfl.py 同一处理）。

    Returns:
        return_physical_too=False（默认）: dt_local，CuPy 数组 (n_cells,)
        return_physical_too=True: (dt_mean_flow, dt_physical)；未启用
            预处理（mach_ref is None）时两者是同一个数组对象。
    """
    cp = get_cupy()
    n_cells = U.shape[0]

    # **本函数按约定只处理单个 SP**（2026-09-14 更正）：
    # 此前这里写"使用 SP0 的值计算时间步长（简化：取每个 cell 第一个
    # SP）"——那个标注已不成立。两条生产调用方（单机
    # `gpu_solver.py::_compute_local_time_step_gpu` 与多 GPU
    # `gpu_distributed.py` 同名方法）都在**逐 SP 循环**里传
    # `U[:, sp:sp+1, :]` 这样的单 SP 切片、并对结果取单元内最小值，
    # 所以 `U[:, 0, ...]` 取到的就是当前那个 SP，不是"只用第一个 SP"
    # 的近似。（多 GPU 那条原先确实只传 SP0，那是真实简化，已在同一批
    # 改动里随该方法的重写一起修掉。）
    #
    # 下面这道校验把这条隐式契约变成显式失败：如果有人传完整的
    # (n_cells, n_sps, n_vars) 数组，旧代码会**静默**只用 SP0 算步长——
    # 逐 SP 的几何/度量 CFL 限制（dt_geometric，专门用来防坍缩坐标下
    # 同一单元内 det(J) 相差几百倍导致的局部刚性失稳）会因此大部分
    # 失效，而且不报任何错。宁可直接失败。
    if U.ndim != 3 or U.shape[1] != 1:
        raise ValueError(
            f"compute_local_cfl_step_gpu 期望单 SP 切片 (n_cells, 1, n_vars)，"
            f"实际收到 {tuple(U.shape)}——调用方必须按 SP 循环、逐 SP 调用"
            f"并对结果取单元内最小值（见 gpu_solver.py/gpu_distributed.py 的"
            f"_compute_local_time_step_gpu）。传完整数组会静默只用 SP0、"
            f"让逐 SP 的几何/度量 CFL 保护失效。")
    rho = cp.maximum(U[:, 0, 0], 1e-9)
    vel = U[:, 0, 1:4] / rho[:, None]
    ke = 0.5 * rho * cp.sum(vel**2, axis=1)
    p = cp.maximum((GAMMA - 1.0) * (U[:, 0, 4] - ke), 1.0)
    a = cp.sqrt(GAMMA * p / rho)

    # 阶数相关收紧（与 CPU 侧 cfl.py 同一公式）：显式 FR 格式的对流/
    # 粘性稳定极限随阶数衰减，对流 ~1/(2p+1)、粘性 ~1/(2p+1)^2。
    order_factor_advective = 1.0 / (2 * poly_order + 1)
    order_factor_viscous = order_factor_advective ** 2

    # 谱半径累加
    spectral = cp.zeros(n_cells, dtype=cp.float64)

    # 内部面贡献
    int_mask = ~is_boundary
    io = owner_cell[int_mask]
    ineigh = neighbor_cell[int_mask]
    n_int = normals[int_mask]
    a_int = areas[int_mask]

    un_o = cp.abs(cp.einsum('nd,nd->n', vel[io], n_int)) + a[io]
    un_n = cp.abs(cp.einsum('nd,nd->n', vel[ineigh], n_int)) + a[ineigh]

    cp.scatter_add(spectral, io, un_o * a_int)
    cp.scatter_add(spectral, ineigh, un_n * a_int)

    # 边界面贡献
    bnd_mask = is_boundary
    bo = owner_cell[bnd_mask]
    if bo.size > 0:
        n_b = normals[bnd_mask]
        a_b = areas[bnd_mask]
        un_b = cp.abs(cp.einsum('nd,nd->n', vel[bo], n_b)) + a[bo]
        cp.scatter_add(spectral, bo, un_b * a_b)

    spectral = cp.maximum(spectral, 1e-30)
    dt = cfl * order_factor_advective * cell_volumes / spectral

    # 粘性限制（阶数收紧与 CPU 侧 cfl.py 同一公式）
    if mu_eff is not None:
        Lc2 = cell_volumes ** (2.0 / 3.0)
        dt_visc = 0.25 * cfl * order_factor_viscous * rho * Lc2 / cp.maximum(mu_eff, 1e-30)
        dt = cp.minimum(dt, dt_visc)

    # 几何/度量 CFL 限制（与 CPU 侧 cfl.py::compute_local_time_step 的
    # dt_geometric 同一公式，见上方参数文档）：用该 SP 自己的 det(J) 当作
    # 局部"体积"，metric_flux_scale 当作局部"总通量面积"。
    if det_jacs_sp is not None and metric_flux_scale_sp is not None:
        wave_speed = cp.maximum(cp.sqrt(cp.sum(vel**2, axis=1)) + a, 1e-10)
        dt_geometric = cfl * cp.abs(det_jacs_sp) / cp.maximum(
            metric_flux_scale_sp * wave_speed, 1e-300
        )
        dt = cp.minimum(dt, dt_geometric)

    if mach_ref is None:
        return (dt, dt) if return_physical_too else dt

    # === 预处理后的平均流 dt（与 CPU 侧 cfl.py 的同名段落逐项对应）===
    from .gpu_preconditioning import preconditioned_sound_speed_gpu
    vel_mag = cp.sqrt(cp.sum(vel ** 2, axis=1))
    c_pre = preconditioned_sound_speed_gpu(vel_mag, a, float(mach_ref))

    spectral_p = cp.zeros(n_cells, dtype=cp.float64)
    un_o_p = cp.abs(cp.einsum('nd,nd->n', vel[io], n_int)) + c_pre[io]
    un_n_p = cp.abs(cp.einsum('nd,nd->n', vel[ineigh], n_int)) + c_pre[ineigh]
    cp.scatter_add(spectral_p, io, un_o_p * a_int)
    cp.scatter_add(spectral_p, ineigh, un_n_p * a_int)
    if bo.size > 0:
        un_b_p = cp.abs(cp.einsum('nd,nd->n', vel[bo], n_b)) + c_pre[bo]
        cp.scatter_add(spectral_p, bo, un_b_p * a_b)
    spectral_p = cp.maximum(spectral_p, 1e-30)
    dt_mean = cfl * order_factor_advective * cell_volumes / spectral_p

    if mu_eff is not None:
        Lc2 = cell_volumes ** (2.0 / 3.0)
        dt_mean = cp.minimum(
            dt_mean,
            0.25 * cfl * order_factor_viscous * rho * Lc2 / cp.maximum(mu_eff, 1e-30),
        )
    if det_jacs_sp is not None and metric_flux_scale_sp is not None:
        wave_speed_p = cp.maximum(vel_mag + c_pre, 1e-10)
        dt_mean = cp.minimum(
            dt_mean,
            cfl * cp.abs(det_jacs_sp) / cp.maximum(
                metric_flux_scale_sp * wave_speed_p, 1e-300),
        )

    return (dt_mean, dt) if return_physical_too else dt_mean


class GPUTimeIntegrator(TimeIntegrator):
    """GPU 后端的时间积分器 —— 数值本体全部继承 `TimeIntegrator`。

    与父类只差一点：`scheme` 接受用户侧字符串（`scheme_from_name` 解析，非法
    取值报错，不静默退回前向 Euler —— 此前 `_SCHEME_TABLE.get(scheme, _EULER)`
    正是那样）。CFL 不归积分器管：控制器 / 固定 CFL 由
    `adaptive_cfl/policy.py::build_cfl_policy` 在求解器上建立（此前这里的
    `cfl=1.0` 是 IMEX/双时间步下的实际 CFL，远超 P>=1 显式稳定极限）。

    `dual_time_steps` 默认值与 CPU 同一个（父类构造参数，理由见那里）；此前
    GPU 侧没有这个属性，调用方各自 `getattr(..., 5)` 兜底，内迭代预算比 CPU
    少 4 倍。
    """

    def __init__(self, scheme="ssp_rk3", dual_time_steps: int = 20):
        super().__init__(scheme=scheme_from_name(scheme), dual_time_steps=dual_time_steps)
