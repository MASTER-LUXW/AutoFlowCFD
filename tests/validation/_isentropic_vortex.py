"""等熵涡（isentropic vortex）精确解与配套网格/边界条件构造。

标准算例（Shu 1998 等经典等熵涡精度基准的物理内容不变，这里只是把
文献里常见的无量纲形式改写成与本项目其余验证算例一致的国际单位制
——直接套用文献的 rho_inf=1, p_inf=1 会导致 `time_integration.py::
enforce_positivity` 里 `p_floor=1.0` 这个物理正定性下限（单位 Pa，
按真实大气压标定）把涡核本该低于 1 的压力直接钳死在下限上，扭曲解本身，
不是把算例做"简化"了，是必须要做的单位换算）：

背景均匀来流沿 x 方向，涡核只随背景流平动（v 分量只是涡自身的旋转扰动，
不含背景分量），因此涡核中心的 y 坐标恒定，只需要 x 方向周期，y/z 方向
用远场条件即可（涡强度按 exp(-r^2) 衰减，标准 beta=5 参数下 r=5 处
exp(1-25)=4e-11，只要域宽 >= 5 倍涡核半径，远场壁面反射误差就在浮点
噪声量级，双向周期不是必需的）。

精确解（等熵关系 p/p_inf = (rho/rho_inf)^gamma，来自标准 Euler 涡量
扰动的封闭解，见 Shu 1998 / Spiegel et al. 2015）：
    r^2 = ((x-x0-u_inf*t)/Rc)^2 + ((y-y0)/Rc)^2
    du = -(beta*a_inf/(2*pi)) * exp(0.5*(1-r^2)) * (y-y0)/Rc
    dv =  (beta*a_inf/(2*pi)) * exp(0.5*(1-r^2)) * (x-x0-u_inf*t)/Rc
    theta = 1 - (gamma-1)*beta^2/(8*gamma*pi^2) * exp(1-r^2)   [= T/T_inf]
    rho = rho_inf * theta^(1/(gamma-1))
    p   = p_inf   * theta^(gamma/(gamma-1))
    u = u_inf + du,  v = dv,  w = 0
这是可压缩 Euler 方程的精确解析解（在随背景流平动的参考系下是定常解），
不依赖任何数值离散——用于定量验证 FR 离散 + AUSM+up + 周期边界条件
组合后，一个真正非定常、真正弯曲的二维流动结构能否在一个周期内被
准确平动而不产生虚假耗散/频散/周期面处的伪反射。
"""
from types import SimpleNamespace

import numpy as np

from ._periodic_mesh import build_periodic_channel_mesh_x
from ._channel_mesh import build_face_exact_ghost_provider

GAMMA = 1.4

# 标准算例参数（beta=5 是文献里几乎统一使用的经典取值，保证 theta_min
# = 1-(gamma-1)*beta^2/(8*gamma*pi^2) = 0.9095 > 0，全场密度/压力恒正，
# 不需要另外做数值保护）。
BETA = 5.0
RC = 1.0  # 涡核特征半径 (m)


def vortex_primitive_field(x, y, t, x0, y0, Lx, rho_inf, p_inf, u_inf):
    """返回 (rho, u, v, w, p)，输入 x,y,t 可以是任意形状的 numpy 数组。

    涡心沿 x 方向以 u_inf 平动；`Lx` 用于把涡心平动位置按周期域折叠
    回 [x0, x0+Lx) 区间内，避免 t 较大时 (x-x0-u_inf*t) 数值上变得
    很大（虽然 exp(-r^2) 会让远处贡献严格为零，折叠只是数值上更干净，
    物理结果不受影响，因为经过完整周期后 exp() 里的 r^2 依赖的是
    折叠后与折叠前完全等价的相对位移模 Lx）。
    """
    a_inf = np.sqrt(GAMMA * p_inf / rho_inf)
    x_center = x0 + np.mod(u_inf * t, Lx)
    dx = np.mod(x - x_center + Lx / 2.0, Lx) - Lx / 2.0  # 最近周期像的相对位移
    dy = y - y0
    r2 = (dx / RC) ** 2 + (dy / RC) ** 2

    swirl = (BETA * a_inf / (2.0 * np.pi)) * np.exp(0.5 * (1.0 - r2))
    du = -swirl * (dy / RC)
    dv = swirl * (dx / RC)

    theta = 1.0 - (GAMMA - 1.0) * BETA**2 / (8.0 * GAMMA * np.pi**2) * np.exp(1.0 - r2)
    rho = rho_inf * theta ** (1.0 / (GAMMA - 1.0))
    p = p_inf * theta ** (GAMMA / (GAMMA - 1.0))
    u = u_inf + du
    v = dv
    w = np.zeros_like(u)
    return rho, u, v, w, p


def primitive_to_conservative(rho, u, v, w, p):
    E = p / (GAMMA - 1.0) + 0.5 * rho * (u**2 + v**2 + w**2)
    return rho, rho * u, rho * v, rho * w, E


def build_vortex_mesh(order, nx, ny, nz, Lx, H, Lz):
    """涡沿 x 方向平动、x 方向周期，y/z 方向远场（见模块文档）。"""
    return build_periodic_channel_mesh_x(order, nx, ny, nz, Lx, H, Lz, side_bc_type="FARFIELD")


def build_vortex_farfield_ghost_provider(mesh, Lx, H, Lz, rho_inf, p_inf, u_inf):
    """wall_bottom/wall_top/z_min/z_max 四个非周期侧面统一按 FARFIELD
    处理，来流状态取背景均匀流（涡本身在这些侧面处的扰动幅值已按
    模块文档的域宽选择衰减到浮点噪声量级）。
    """
    Q_free = [rho_inf, u_inf, 0.0, 0.0, p_inf]
    bc_by_plane = {
        "wall_bottom": {"type": "FARFIELD", "Q_free": Q_free},
        "wall_top": {"type": "FARFIELD", "Q_free": Q_free},
        "z_min": {"type": "FARFIELD", "Q_free": Q_free},
        "z_max": {"type": "FARFIELD", "Q_free": Q_free},
    }
    return build_face_exact_ghost_provider(mesh, Lx, H, Lz, bc_by_plane)
