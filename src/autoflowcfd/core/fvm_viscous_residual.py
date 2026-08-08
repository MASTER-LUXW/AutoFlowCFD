"""稳态求解器用的二阶粘性 RANS 残差。

本模块取代了早期一阶、纯无粘的残差实现，提供物理上完整的单元中心有限
体积残差，包含：

* **MUSCL 重构**：Green-Gauss 梯度 + Barth-Jespersen 限制器，给出真正
  二阶精度的面左/右状态，而不是之前用的单元中心值。
* **无粘通量**（HLLC Riemann 求解器）在重构状态上求值。
* **粘性通量**：分子粘性 + 湍流（涡）剪切应力，以及热/湍流扩散，用
  Stokes 假设下的可压缩牛顿应力张量。
* **SST k-omega 源项**：production、dissipation 和 cross-diffusion
  真正耦合进 k、omega 方程，涡粘性反馈进动量方程的粘性通量。

整条路径都是向量化 NumPy 实现，因此结果确定、便于单元测试。守恒变量
顺序为 ``[rho, rho u, rho v, rho w, E, rho k, rho w_sst]``，湍流量以
守恒（密度加权）形式携带，与无粘通量里的输运形式一致。

`ViscousRANSResidual` 本身只有构造函数、`to_primitive`/`compute`（编排整个
残差计算的入口）和几个跨物理项共享的辅助方法（湍流量的壁面 ghost 值、
k/omega 梯度、速度梯度）。三块具体物理各自的实现拆到了同目录下三个
mixin 文件里（纯粹是控制单文件行数，不是独立的概念层）：

* `fvm_residual_inviscid.InviscidFluxMixin` —— MUSCL 重构 + AUSM+up（HLLC
  备用参考实现）。
* `fvm_residual_viscous.ViscousFluxMixin` —— 应力/热传导/湍流扩散通量 +
  Menter 壁面函数。
* `fvm_residual_sst.SSTSourceMixin` —— 应变率、涡粘性、SST 源项。
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from loguru import logger

from .fvm_gradients import FaceGeometry, green_gauss_gradient
from .fvm_residual_inviscid import InviscidFluxMixin
from .fvm_residual_viscous import ViscousFluxMixin
from .fvm_residual_sst import SSTSourceMixin

# GPU（CUDA）kernel 可用性标记：三个 mixin 文件各自从对应的 *_kernels_gpu.py
# 导入自己需要的 dispatch 函数，这里只需要 CUDA_AVAILABLE 本身（__init__
# 用它判断 use_gpu 请求是否可行）。见各 *_gpu.py 模块文档：**从未在真实
# GPU 硬件上运行验证过**。
from .fvm_inviscid_kernels_gpu import CUDA_AVAILABLE

GAMMA = 1.4
R_GAS = 287.058          # J/(kg K)，干空气气体常数


class ViscousRANSResidual(InviscidFluxMixin, ViscousFluxMixin, SSTSourceMixin):
    """计算完整的二阶粘性 RANS 残差。

    Parameters
    ----------
    geom:
        已定向的 :class:`FaceGeometry`。
    mu_lam:
        分子动力粘度（Pa s）。
    wall_distance:
        每个单元到最近粘性壁面的距离（m）。SST 混合函数需要此量；若为
        ``None``，则假设全场为一个很大的值（等效自由来流行为）。
    turbulent:
        若为 False，则关闭 k/omega 源项和涡粘性（层流 Navier-Stokes）。
    mach_ref:
        参考（自由来流）马赫数，仅供 AUSM+up 无粘通量的低马赫数缩放函数
        f_a（见 _ausm_up）用来规整其驻点行为——**不是**对伪时间步长或
        波速结构做预处理（那种做法试过又撤销了，见 solver_steady.py 里
        的说明）。默认值 0.1 是给不会算出具体算例马赫数的调用方（例如
        瞬态求解器）用的通用安全兜底值；若能拿到本次求解实际的自由来流
        马赫数，应传入以保证物理一致的缩放。
    wall_face_mask:
        布尔数组，形状 (n_faces,)，与 `geom` 的完整面排列对齐（约定同
        `wall_distance` 的 wall_face_mask 参数传给 estimate_wall_distance
        时一致）——分类为 WALL/GROUND（粘性无滑移壁面）的边界面为 True，
        其余（SYMMETRY/INLET/OUTLET/FARFIELD 边界，以及所有内部面）为
        False。默认 None 表示完全不启用壁面函数处理，退化为一直解到壁面
        （与之前的行为完全一致）——传入此参数即可在这些面上启用 Menter
        的 scalable/automatic 壁面处理（见 _wall_function_targets），使
        较粗的近壁网格（y+ 可到 100+，不必是 y+~1）也能给出物理上合理的
        壁面摩擦和 k/omega。
    use_gpu:
        为 True 且真的存在可用 CUDA 设备时，把热点循环（AUSM+up 通量、
        粘性通量、SST 涡粘性/源项、Green-Gauss 梯度）分发到
        `fvm_*_kernels_gpu.py` 里的 CUDA kernel；否则静默回退到 CPU
        （Numba）路径并给出警告。这些 GPU kernel 在本项目里**从未在真实
        GPU 硬件上运行过**——见各自模块的文档字符串。
    """

    def __init__(self, geom: FaceGeometry, mu_lam: float = 1.7894e-5,
                 wall_distance: np.ndarray | None = None,
                 turbulent: bool = True,
                 mach_ref: float = 0.1,
                 wall_face_mask: np.ndarray | None = None,
                 use_gpu: bool = False):
        self.geom = geom
        self.mu_lam = float(mu_lam)
        self.turbulent = turbulent
        self.mach_ref = float(mach_ref)
        # GPU dispatch 是可选项，若实际没有可用的 CUDA 设备会静默降级到
        # CPU/Numba 路径——该路径未经真实硬件验证的原因见本模块顶部的
        # GPU kernel 导入处注释（本开发环境没有 GPU 可供测试）。
        if use_gpu and not CUDA_AVAILABLE:
            logger.warning(
                "use_gpu=True was requested but no CUDA device is available "
                "in this environment - falling back to the CPU (Numba) "
                "residual path."
            )
        self._use_gpu = bool(use_gpu) and CUDA_AVAILABLE
        n = geom.n_cells
        if wall_distance is None:
            self.wall_distance = np.full(n, 1.0e9, dtype=np.float64)
        else:
            self.wall_distance = np.maximum(np.asarray(wall_distance, np.float64), 1e-9)

        # 把 wall_face_mask 收窄到只含边界面，顺序与下面马上建立的
        # self._bo/geom.bnd_owner 一致——这正是 wall_shear_stress()/
        # _viscous_flux 的边界部分实际遍历时用的顺序。
        if wall_face_mask is not None:
            self._wall_face_mask_b = np.asarray(wall_face_mask, dtype=bool)[geom.boundary_mask]
        else:
            self._wall_face_mask_b = None

        # 预计算内部面的 owner->neighbour 几何量。
        self._im = geom.internal_mask
        self._io = geom.int_owner
        self._in = geom.int_neigh
        d = geom.cell_centroids[self._in] - geom.cell_centroids[self._io]
        self._dist = np.maximum(np.linalg.norm(d, axis=1), 1e-12)
        self._e_ON = d / self._dist[:, None]           # owner->neighbour 单位向量

        # 预计算边界面的 owner->ghost 几何量（作用与上面的
        # _dist/_e_ON 相同，只是这里的"neighbour"是镜像 ghost 状态）。
        # ghost 状态的构造方式（见 BoundaryConditionHandler）使得*面*上
        # 的值正好是 owner 与 ghost 的中点平均——也就是说 ghost 被假定位于
        # 跨面镜像点，距离是 owner 到面距离的两倍，而不是就在面上。如果
        # 这里直接用面距离，会把真实的 owner->ghost 间距减半，从而把推算
        # 出来的近壁梯度放大一倍。
        self._bo = geom.bnd_owner
        if self._bo.size:
            db = geom.centers[geom.boundary_mask] - geom.cell_centroids[self._bo]
            self._bdist = np.maximum(2.0 * np.linalg.norm(db, axis=1), 1e-12)
            self._e_OB = db / (0.5 * self._bdist[:, None])
        else:
            self._bdist = np.zeros(0)
            self._e_OB = np.zeros((0, 3))

    # ------------------------------------------------------------------
    # 原始变量 <-> 守恒变量
    # ------------------------------------------------------------------
    @staticmethod
    def to_primitive(U: np.ndarray):
        """从守恒量 U 返回 (rho, vel(n,3), p, T, k, omega)。"""
        rho = np.maximum(U[:, 0], 1e-9)
        vel = U[:, 1:4] / rho[:, None]
        ke = 0.5 * rho * np.sum(vel**2, axis=1)
        p = np.maximum((GAMMA - 1.0) * (U[:, 4] - ke), 1.0)
        T = p / (rho * R_GAS)
        k = np.maximum(U[:, 5] / rho, 0.0)
        omega = np.maximum(U[:, 6] / rho, 1e-6)
        return rho, vel, p, T, k, omega

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def compute(self, U: np.ndarray, boundary_states: np.ndarray) -> np.ndarray:
        """返回残差 R，满足 dU/dt 使得 V_i dU_i/dt = -R_i（R 已经除以体积）。

        Args:
            U: 守恒解，形状 (n_cells, 7)。
            boundary_states: 边界面的 ghost 守恒状态，形状 (n_faces, 7)；
                只读取属于边界面的那些行。

        Returns:
            残差数组，形状 (n_cells, 7)，已经除以单元体积，因此更新式为
            ``U -= dt * R``。
        """
        geom = self.geom
        n_cells = geom.n_cells
        rho, vel, p, T, k, omega = self.to_primitive(U)

        # 涡粘性（需要应变率 -> 速度梯度）。
        grad_vel = self._velocity_gradient(vel, U, boundary_states)
        mu_t = self._eddy_viscosity(rho, k, omega, grad_vel) if self.turbulent \
            else np.zeros(n_cells)

        # 支持壁面函数的 k/omega 边界 ghost 值，_turbulence_gradient 和
        # _viscous_flux 共用（见 _turbulence_wall_ghost 自己的文档字符串，
        # 说明这是为了替换掉之前两处各自独立推导、在启用壁面函数时会互相
        # 不一致的边界 k/omega）。
        k_ghost_b, omega_ghost_b = self._turbulence_wall_ghost(rho, vel, k, omega, boundary_states)

        # k/omega 梯度，这里只算一次，供 _viscous_flux（湍流扩散项，无论
        # self.turbulent 是否为真都需要——见下）和 _sst_sources
        # （production/cross-diffusion，仅在湍流开启时用）共用。以前这两处
        # 各自独立计算一份同样的 Green-Gauss 梯度（且都没有边界贡献），
        # 而 _inviscid_flux 里单独的 7 变量 MUSCL 重构梯度也带着 k/omega
        # 这两列——同一个物理量每次残差计算要算 3 遍。这里共用的版本额外
        # 包含了边界（ghost 状态）贡献，是那两份重复计算都缺失的部分。
        grad_turb = self._turbulence_gradient(k, omega, k_ghost_b, omega_ghost_b)

        flux_accum = np.zeros((n_cells, 7), dtype=np.float64)

        # --- 无粘通量：MUSCL + HLLC ---
        self._inviscid_flux(U, boundary_states, flux_accum)

        # --- 粘性通量（分子 + 湍流）---
        self._viscous_flux(rho, vel, T, k, omega, mu_t, grad_vel, grad_turb,
                           boundary_states, flux_accum, k_ghost_b, omega_ghost_b)

        # 把通量的面积分转换成残差（除以体积 V）。
        residual = flux_accum / geom.cell_volumes[:, None]

        # --- SST 源项（体积项，直接加进残差）---
        if self.turbulent:
            self._sst_sources(rho, k, omega, mu_t, grad_vel, grad_turb, residual)

        return residual

    def _turbulence_wall_ghost(self, rho: np.ndarray, vel: np.ndarray,
                               k: np.ndarray, omega: np.ndarray,
                               boundary_states: np.ndarray):
        """边界 k/omega 的 ghost 值，各自形状 (n_bf,)——支持壁面函数，是
        _turbulence_gradient 和 _viscous_flux 的湍流扩散项共同依赖的唯一
        数据源。

        对每个边界面，先取 `boundary_states` 里已经编码好的 ghost 值
        （固壁上 k=0 的 Dirichlet / omega 零梯度——见
        BoundaryConditionHandler._wall_bc；INLET/FARFIELD/OUTLET 上则是
        自由来流值）。若构造时提供了壁面函数掩码，WALL/GROUND 面上的值
        会被 log-law 模型给出的近壁目标值覆盖（用与 boundary_states 自身
        ghost 相同的镜像方式构造，使面平均值恰好等于目标值）——这与
        _viscous_flux 以前完全独立地自行应用的机制相同，但那时只在那一个
        函数内部局部生效，不影响其他地方。

        这曾是一个真实的 bug：_turbulence_gradient（每次残差求值单独
        调用一次）一直在自己从 `boundary_states` 计算 k/omega 的 ghost 值，
        也就是说即使启用了壁面函数、扩散通量已经在用 log-law 目标值，它
        用的却始终是解析到壁面（k=0，omega 零梯度）那一套值。而 SST 的
        F1 混合、cross-diffusion，以及（间接地）production 在每个近壁
        单元都依赖的 k/omega 梯度，因此和实际建模的壁面处理方式在暗中
        不一致——它一直假设第一层网格解析到 y+~1，即便是在专门为壁面函数
        设计的更粗网格上，这会在（现在更大的）近壁单元高度上强行造出一个
        过陡的 k/omega 梯度。让两处消费者都从这一份共用、已正确调整的
        ghost 值取数，从结构上保证了两者的一致性。
        """
        bo = self.geom.bnd_owner
        if not bo.size:
            return np.zeros(0), np.zeros(0)
        rho_b = np.maximum(boundary_states[self.geom.boundary_mask, 0], 1e-9)
        k_b = np.maximum(boundary_states[self.geom.boundary_mask, 5] / rho_b, 0.0)
        omega_b = np.maximum(boundary_states[self.geom.boundary_mask, 6] / rho_b, 1e-6)

        wm = self._wall_face_mask_b
        if wm is not None and np.any(wm):
            tang_dir, tang_mag = self._wall_tangential_velocity(rho, vel, boundary_states)
            y_p = self.wall_distance[bo][wm]
            _, k_wall, omega_wall = self._wall_function_targets(rho[bo][wm], tang_mag, y_p)
            k_b = k_b.copy()
            omega_b = omega_b.copy()
            k_b[wm] = np.maximum(2.0 * k_wall - k[bo][wm], 0.0)
            omega_b[wm] = np.maximum(2.0 * omega_wall - omega[bo][wm], 1e-6)
        return k_b, omega_b

    def _turbulence_gradient(self, k: np.ndarray, omega: np.ndarray,
                             k_ghost_b: np.ndarray, omega_ghost_b: np.ndarray) -> np.ndarray:
        """[k, omega] 的 Green-Gauss 梯度，形状 (n_cells, 2, 3)。

        每次残差求值只算一次，_viscous_flux 和 _sst_sources 共用（为什么
        以前是重复计算见 compute() 文档字符串里的说明）。边界 ghost 值
        来自 _turbulence_wall_ghost（支持壁面函数，见其自身文档字符串），
        这里不重新计算。
        """
        bo = self.geom.bnd_owner
        turb_b = None
        if bo.size:
            turb_b = np.column_stack([k_ghost_b, omega_ghost_b])
        turb_vars = np.column_stack([k, omega])
        return green_gauss_gradient(turb_vars, self.geom, turb_b, use_gpu=self._use_gpu)

    # ------------------------------------------------------------------
    # 速度梯度（供应变率与粘性应力使用）
    # ------------------------------------------------------------------
    def _velocity_gradient(self, vel, U, boundary_states):
        # 边界面速度取自 ghost 状态
        bo = self.geom.bnd_owner
        if bo.size:
            rho_b = np.maximum(boundary_states[self.geom.boundary_mask, 0], 1e-9)
            vel_b = boundary_states[self.geom.boundary_mask, 1:4] / rho_b[:, None]
        else:
            vel_b = None
        # 每个速度分量的梯度 -> (n_cells, 3, 3)：[单元, 分量, 方向]
        return green_gauss_gradient(vel, self.geom, vel_b, use_gpu=self._use_gpu)


def estimate_wall_distance(geom: FaceGeometry, wall_face_mask: np.ndarray) -> np.ndarray:
    """估算每个单元中心到最近壁面的距离。

    用 KD-Tree 空间索引把复杂度降到 O(N log M)，而不是暴力 O(N*M)。对
    280 万单元、13 万壁面来说，计算时间从小时级降到秒级。

    Args:
        geom: 带单元中心和面中心的面几何
        wall_face_mask: 标识壁面边界面的布尔掩码

    Returns:
        每个单元到最近壁面的最小距离数组
    """
    n_cells = geom.n_cells
    wall_faces = np.where(wall_face_mask)[0]

    if wall_faces.size == 0:
        logger.warning("No wall faces found, returning large default distance")
        return np.full(n_cells, 1.0e9)

    # 取出壁面面心坐标
    wall_pts = geom.centers[wall_faces]
    cc = geom.cell_centroids

    logger.info(f"Building KD-Tree for {len(wall_pts)} wall points...")

    # 用壁面点构建 KD-Tree（O(M log M)）
    tree = cKDTree(wall_pts)

    # 为所有单元中心查询最近邻（O(N log M)）
    logger.info(f"Querying nearest wall distance for {n_cells} cells...")
    distances, _ = tree.query(cc, k=1, workers=-1)  # workers=-1 使用全部 CPU 核心

    logger.success(f"Wall distance computed: min={distances.min():.4e}, max={distances.max():.4e}, mean={distances.mean():.4e}")

    return np.maximum(distances, 1e-9)
