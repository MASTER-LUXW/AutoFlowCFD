"""验证 compute_scalar_diffusion_residual 修复后为耗散（正扩散）算子。

核心判据（数学上正确的稳定性判据）：离散能量增长率 Σ φ·R·V 对高频模态为负。
反扩散（修复前的 -div）会使该量为正、指数放大棋盘模态；正扩散为负。

附注（判据 1 的已知现象）：合成网格含小 detJ 单元，体积项 /detJ 会把
常数通量的离散散度残差放大——这是与平均流体积算子共享的既有度量一致性
性质（见 troubled_cell.py/viscous_flux.py 文档），生产路径由
suppress_residual_outliers 兜底，与本次符号修复正交。因此判据 1 用中位数
（对退化单元的尖峰稳健）而不是均值。

判据 5（量级回归，2026-08-25 代码审查新增）：光滑场残差的体积加权均值应接近
解析拉普拉斯值——若界面校正缺/多 |adj_row| 面元幅值因子（曾发现的量纲缺陷），
界面项会偏离 ~1/h² 倍，光滑场残差被界面伪影主导，偏离解析值。
"""
import sys
sys.path.insert(0, r"d:\myWorkspace\AutoFlowCFD\tests\validation")

import numpy as np
from _channel_mesh import build_channel_mesh
from autoflowcfd.core.turbulence.transport import compute_scalar_diffusion_residual

mesh = build_channel_mesh(order=1, nx=6, ny=6, nz=2, Lx=1.0, H=1.0, Lz=0.34)
ops = mesh.operators
n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
print(f"mesh: n_cells={n_cells}, n_sps={n_sps}")

sps = mesh.sps_coords.reshape(n_cells, n_sps, 3)
x = sps[:, :, 0]
det_jacs = mesh.jacobians["det_jacs"].reshape(n_cells, n_sps)
w_vol = np.abs(det_jacs)
Gamma = np.ones((n_cells, n_sps))
interior = (x > 0.25) & (x < 0.75)

# === 判据 1: 抛物线场内部残差符号为正（+div(grad phi) 方向）===
# 用中位数：合成网格小 detJ 单元的 /detJ 尖峰会污染均值（见模块文档附注）。
R = compute_scalar_diffusion_residual(x ** 2, Gamma, mesh, ops)
R_int = R[interior]
med = np.median(R_int)
print(f"[1] 抛物线场内部残差: median={med:.4f}, mean={R_int.mean():.4f} (扩散应为正, 反扩散为负)")
assert med > 0, f"扩散方向错误: median={med:.4f}"

# === 判据 2: SP 级棋盘模态离散能量率为负（耗散性，稳定性核心判据） ===
sp_parity = np.arange(n_sps) % 2
checker = np.where(sp_parity[None, :] == 0, 1.0, -1.0) * np.ones((n_cells, n_sps))
R_ch = compute_scalar_diffusion_residual(checker, Gamma, mesh, ops)
energy_rate = np.sum(checker * R_ch * w_vol)
print(f"[2] 棋盘模态离散能量率 Σφ·R·V = {energy_rate:.4f} (扩散应 < 0)")
assert energy_rate < 0, f"棋盘模态被放大（反扩散）: {energy_rate:.4f}"

# === 判据 3: 多个高频模态全部耗散（稳健性） ===
rng = np.random.default_rng(0)
worst = 0.0
best = 0.0
for trial in range(20):
    # 零均值高频扰动（逐 SP 随机 ±，减去单元均值以突出高频分量）
    pert = rng.standard_normal((n_cells, n_sps))
    pert -= pert.mean(axis=1, keepdims=True)
    Rp = compute_scalar_diffusion_residual(pert, Gamma, mesh, ops)
    rate = np.sum(pert * Rp * w_vol) / max(np.sum(pert**2 * w_vol), 1e-30)
    worst = max(worst, rate)
    best = min(best, rate)
print(f"[3] 20 个随机高频扰动: 最大归一化能量率 = {worst:.4f} (应 ≤ 0), "
      f"最小 = {best:.4f} (应 < 0)")
assert worst <= 0, f"存在被放大的高频模态（反扩散）: {worst:.4f}"
assert best < 0, "没有任何模态被耗散，算子疑似恒零"

# === 判据 4: 均匀场零残差（自由流保持性，不经过 /detJ 放大）===
R_const = compute_scalar_diffusion_residual(np.ones((n_cells, n_sps)), Gamma, mesh, ops)
print(f"[4] 均匀场残差最大幅值: {np.abs(R_const).max():.3e} (期望 ≈ 0)")
assert np.abs(R_const).max() < 1e-8, "均匀场残差非零"

# === 判据 5: 光滑场残差的体积加权均值与解析拉普拉斯同量级（量级回归）===
# φ = sin(πx) 的解析扩散值 = -π²·sin(πx)。体积加权平均把 /detJ 尖峰的权重压回
# 真实体积，度量的是算子的整体量级：界面校正若缺/多 |adj_row|（~O(h²)，
# 本网格 ~1/36），比值会偏离几十倍，仍能拦住量级灾难；但合成网格体积项
# 本身有度量一致性伪影（实测体积项均值 -12~-17 且随细化非单调波动，解析 -9.3），
# 总残差比值不能期望收敛到 1，只要求同量级（[0.3, 2]，对 36 倍量级错误仍有效）。
phi_s = np.sin(np.pi * x)
R_s = compute_scalar_diffusion_residual(phi_s, Gamma, mesh, ops)
core = (x > 0.3) & (x < 0.7)
w_core = w_vol[core]
discrete = np.sum(R_s[core] * w_core) / np.sum(w_core)
analytic_avg = np.sum(-np.pi**2 * phi_s[core] * w_core) / np.sum(w_core)
ratio = discrete / analytic_avg if abs(analytic_avg) > 1e-12 else float('nan')
print(f"[5] sin(πx) 场核心区体积加权残差: 离散={discrete:.4f}, "
      f"解析={analytic_avg:.4f}, 比值={ratio:.3f} (期望 ∈ [0.3, 2.0])")
assert np.sign(discrete) == np.sign(analytic_avg), "光滑场残差方向错误"
assert 0.3 <= ratio <= 2.0, f"界面校正量级异常（疑似缺/多面元幅值因子）: 比值={ratio:.3f}"

print("\n全部判据通过：扩散算子为耗散算子，符号与量级修复正确。")
