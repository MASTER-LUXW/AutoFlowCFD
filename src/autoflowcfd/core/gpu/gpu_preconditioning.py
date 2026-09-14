"""GPU 版 Weiss-Smith 伪时间预处理矩阵 Γ（2026-09-14）。

与 CPU 侧 `core/utils/preconditioning.py` 一一对应，完整的推导、
"为什么不改变收敛解"（det(Gamma)=beta^2>0）、以及"为什么必须与按预处理
波速取的 dt 成对出现"的论证都在那份模块文档里，这里不重复，只记录
GPU 侧的两处刻意差异：

1. **本函数从守恒变量 U 自己推导原始变量**，而 CPU 侧直接消费
   `solver.state.Q`。CPU 那样做是为了省掉 P2 规模下 1.2GiB 的额外分配，
   代价是依赖一条隐式契约（残差求值后 `state.Q` 仍是试探态——见
   `tests/unit/test_low_mach_preconditioner.py::
   TestTrialStatePrimitivesContract`）。GPU 侧 `compute_inviscid_residual_gpu`
   是**按参数**接收试探态 U_trial 的，并不保证 `self.Q_gpu` 与之同步，
   所以这里不去依赖任何副作用，直接算——GPU 上这几步逐点运算的代价
   远低于一次主机-设备往返或一次额外的全场同步。

2. 用 CuPy 的向量化表达式而不是 RawKernel：Γ 是纯逐点的秩一更新
   （约 15 次浮点运算/解点），完全是带宽受限，手写 kernel 拿不到额外
   收益，而向量化表达式与 CPU 侧 numba kernel 的运算顺序逐项对应、
   便于交叉验证（见 `tests/unit/test_gpu_low_mach_precond_crosscheck.py`）。

**本机无 CUDA/CuPy，无法实际运行验证**：与项目既有 GPU 移植同一做法，
正确性通过 (a) 与已验证正确的 CPU 实现逐项对照、(b) 自动跳过的 CPU-GPU
交叉一致性测试（有真实 GPU 时才会真正执行）两条路保证。
"""
from autoflowcfd.core.gpu import get_cupy

_GAMMA_GAS = 1.4


def precond_beta2_gpu(q2, a2, mach_ref: float, k: float = 1.1):
    """beta^2 = clip(max(|u|^2/a^2, k*mach_ref^2), 1e-10, 1)。

    与 CPU 侧 `_precond_beta2` 同一公式：用**速度模**而不是面法向速度
    （时间导数项的预处理是单元/解点局部的，与任何面法向无关；完整论证见
    CPU 侧 `preconditioned_sound_speed` 文档）。
    """
    cp = get_cupy()
    m2 = q2 / cp.maximum(a2, 1e-30)
    return cp.clip(cp.maximum(m2, k * mach_ref * mach_ref), 1e-10, 1.0)


def preconditioned_sound_speed_gpu(vel_mag, a, mach_ref: float, k: float = 1.1):
    """伪时间预处理下的有效声速 sqrt(beta^2)*a，beta^2 按速度模取。

    与 CPU 侧 `preconditioned_sound_speed` 对应——伪时间步长必须用它而
    不是按面法向速度取的 beta^2，否则 dt 会被系统性高估（流动与面法向
    越斜越严重），详见 CPU 侧同名函数文档。
    """
    cp = get_cupy()
    a_safe = cp.maximum(a, 1e-30)
    beta2 = precond_beta2_gpu(vel_mag ** 2, a_safe ** 2, mach_ref, k)
    return cp.sqrt(beta2) * a_safe


def apply_low_mach_preconditioner_gpu(residual, U, mach_ref: float, k: float = 1.1,
                                      out=None):
    """对平均流残差施加 `Gamma R`。

    Gamma = I - ((1-beta^2)/a^2) * phi * psi^T
        phi = (1, u, v, w, H)
        psi = (gamma-1) * (q^2/2, -u, -v, -w, 1)
    因此 `psi^T R = dp`（残差的"压力分量"），
        (Gamma R)_0 = R_0 - c,        c = (1-beta^2)/a^2 * dp
        (Gamma R)_i = R_i - c * u_i   (i = 1,2,3)
        (Gamma R)_4 = R_4 - c * H
    运算顺序与 CPU 侧 `_apply_low_mach_precond_kernel` 逐项对应。

    Args:
        residual: CuPy 数组 (n_cells, n_sps, n_vars)，n_vars 可以 > 5
            （SST 把 k/omega 挂在 5: 上）。**只有前 5 个变量被改动**，
            湍流标量是被动输运量、不含声学模态，必须原样保留。
        U: CuPy 数组 (n_cells, n_sps, >=5) 守恒变量，与 residual 同一状态。
            原始变量在本函数内部推导，不依赖调用方的任何副作用（见模块
            文档第 1 条）。
        mach_ref: 参考马赫数，给 beta^2 提供下限 k*mach_ref^2（避免驻点
            上 beta^2 退化到 0）。
        k: 下限的安全裕度倍数（Weiss-Smith 建议 1.1~1.2）。
        out: 输出数组，可以**就是** residual 本身（就地写）。默认 None 时
            返回新数组、不改动入参。
            为什么需要这个参数：RK 每个 stage 都要施加一次 Gamma，默认的
            "返回新数组"在 79 万单元 P2 规模下每 stage 要多分配约 1.2GiB
            **显存**——GPU 显存比主机内存紧张得多，而 stage 内的原始残差
            施加完就不再需要，就地写是安全的。只有 `residual0` 那一处必须
            保留原始残差（残差范数与自适应 CFL 都用它），那里才用默认的
            拷贝语义。与 CPU 侧 `apply_low_mach_preconditioner(..., out=)`
            同一设计。

    Returns:
        CuPy 数组，形状与 residual 相同（out 非 None 时就是 out）。
    """
    cp = get_cupy()
    gm1 = _GAMMA_GAS - 1.0

    rho = U[..., 0]
    inv_rho = 1.0 / cp.maximum(rho, 1e-30)
    u = U[..., 1] * inv_rho
    v = U[..., 2] * inv_rho
    w = U[..., 3] * inv_rho
    q2 = u * u + v * v + w * w
    p = gm1 * (U[..., 4] - 0.5 * rho * q2)

    # 非物理状态（正性限制器尚未介入的瞬态）原样透传，不在预处理里
    # 制造 NaN 去掩盖真正的问题——与 CPU 侧 kernel 同一处理。
    physical = (rho > 0.0) & (p > 0.0)

    a2 = _GAMMA_GAS * p * inv_rho
    a2_safe = cp.maximum(a2, 1e-30)
    beta2 = precond_beta2_gpu(q2, a2_safe, mach_ref, k)

    r0 = residual[..., 0]
    r1 = residual[..., 1]
    r2 = residual[..., 2]
    r3 = residual[..., 3]
    r4 = residual[..., 4]
    dp = gm1 * (0.5 * q2 * r0 - u * r1 - v * r2 - w * r3 + r4)
    coef = (1.0 - beta2) / a2_safe * dp
    H = a2_safe / gm1 + 0.5 * q2

    if out is None:
        out = residual.copy()
    elif out is not residual:
        out[...] = residual

    # 逐个写回是安全的，即使 `out is residual`（此时 r0..r4 是 out 的视图）：
    # `coef`/`u`/`v`/`w`/`H` 在上面已经全部物化成独立数组（它们只依赖
    # U 和 dp，而 dp 在任何写回之前就算完了），而第 i 个分量的新值只依赖
    # `r_i` 自己——写 out[...,0] 不会影响 new1..new4 的任何输入。所以
    # 不需要先攒 5 份临时数组再统一写（那样会多占 5 个全场数组的显存，
    # 正好抵消掉 out= 这个参数想省的那一份）。
    out[..., 0] = cp.where(physical, r0 - coef, r0)
    out[..., 1] = cp.where(physical, r1 - coef * u, r1)
    out[..., 2] = cp.where(physical, r2 - coef * v, r2)
    out[..., 3] = cp.where(physical, r3 - coef * w, r3)
    out[..., 4] = cp.where(physical, r4 - coef * H, r4)
    return out
