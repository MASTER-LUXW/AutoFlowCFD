"""壁面 y+ 统计（求解收尾报告；单机求解器、GPU 主机视图、分布式汇总视图共用）。

逐个物理无滑移壁面通量点（壁面力积分同一组点，`postprocess/wall_forces.py`）：

    y+ = u_tau h / nu，   u_tau = sqrt(|tau_w| / rho)

* `tau_w`：壁面力积分里同一个粘性力（含内罚项的粘性公共通量；壁面模型激活时为壁面模型牵引）的切向分量除以
  该点物理面积——就是求解器实际施加的壁面剪应力；
* `rho`、`nu = mu / rho`：贴壁单元的解外插到该通量点；
* `h`：贴壁单元积分形心到壁面的法向距离（与平衡壁面模型的匹配高度同一个定义，
  `core/turbulence/wall_model/matching.py::owner_centroid_heights`）。

适用性判据按近壁处理方式（`core/turbulence/wall_model/treatment.py`）：

* ``resolved``（低雷诺数壁面处理）要求贴壁单元落在粘性底层（y+ < 5，壁面律的粘性底层上限）：Menter 的 omega
  壁面值是粘性底层的渐近解，贴壁单元越出粘性底层时壁面条件不再成立；
* ``auto``（平衡壁面模型）在粘性底层到对数律层都成立；匹配点越出对数律层（y+ 超过约 300 进入尾流区）时
  壁面律不再描述速度剖面。

判据取 95% 分位数：驻点、分离点附近 tau_w -> 0，个别点的 y+ 不代表网格的近壁分辨率。
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from loguru import logger

#: 粘性底层上限（壁面律 u+ = y+ 的适用范围）
VISCOUS_SUBLAYER_Y_PLUS = 5.0
#: 对数律层的典型上限（之上为尾流区）
LOG_LAYER_Y_PLUS = 300.0
#: 判据所用的分位数
_QUANTILE = 0.95

#: 用到近壁处理方式的 RANS 模型（与 `treatment.needs_wall_model` 同一组）
_RANS_MODELS = ("SA", "SST", "DDES", "IDDES")


@dataclass
class WallYPlus:
    """壁面通量点上的 y+ 与统计量。"""
    y_plus: np.ndarray          # (m,) 贴壁单元形心高度处的 y+
    position: np.ndarray        # (m, 3) 通量点坐标

    def quantile(self, q: float) -> float:
        return float(np.quantile(self.y_plus, q))


def wall_y_plus(solver) -> Optional[WallYPlus]:
    """求解器当前状态的壁面 y+；没有物理无滑移壁面时 None。"""
    from autoflowcfd.boundary.wall_kind import is_physical_no_slip_wall
    from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
    from autoflowcfd.core.fr_residual.wall_points import wall_flux_points
    from autoflowcfd.core.turbulence.wall_model.matching import cell_integration_weights, owner_centroid_heights
    from autoflowcfd.postprocess.wall_forces import sample_wall

    host = solver.host_view() if hasattr(solver, "host_view") else solver
    provider = host.boundary_ghost_provider
    codes = [code for code, cfg in provider.code_to_config.items() if is_physical_no_slip_wall(cfg)]
    if not codes:
        return None
    mesh = host.mesh
    flat = get_flat_face_geometry(mesh, host.ops)
    rec, _, fp, _ = wall_flux_points(flat, np.asarray(provider.group_code), codes)
    if rec.size == 0:
        return None
    samples = sample_wall(host, codes, viscous=True)
    normal = np.asarray(flat.true_normal)[rec, fp]
    area = np.asarray(flat.true_area_weight)[rec, fp]
    f = samples.force_viscous
    f_t = f - np.einsum('kd,kd->k', f, normal)[:, None] * normal
    tau_w = np.linalg.norm(f_t, axis=1) / area

    owner = np.asarray(flat.owner_cell)[rec].astype(np.int64)
    E = np.asarray(flat.boundary_extrap_native)[np.asarray(flat.owner_face_op)[rec], fp]      # (m, n_sps)
    rho = np.einsum('ks,ks->k', E, np.asarray(host.state.Q)[owner, :, 0])
    nu = float(host.mu_molecular) / rho
    W, _, _, _ = cell_integration_weights(mesh)
    h = owner_centroid_heights(np.asarray(mesh.sps_coords, dtype=np.float64), W, owner, samples.position, normal)
    return WallYPlus(y_plus=np.sqrt(tau_w / rho) * h / nu, position=samples.position)


def report_wall_y_plus(solver, turb_model_name: str, wall_treatment: str) -> Optional[WallYPlus]:
    """打印壁面 y+ 统计，并按近壁处理方式检查适用范围（模块文档）；没有物理无滑移壁面时不打印。"""
    yp = wall_y_plus(solver)
    if yp is None:
        return None
    q = yp.quantile(_QUANTILE)
    print(f"\n=== 壁面 y+（贴壁单元形心高度，{yp.y_plus.size} 个壁面通量点，近壁处理 {wall_treatment}）===")
    print(f"   min = {yp.y_plus.min():.3g}   中位数 = {yp.quantile(0.5):.3g}   "
          f"95% 分位 = {q:.3g}   max = {yp.y_plus.max():.3g}")
    if str(turb_model_name).upper() not in _RANS_MODELS:
        return yp
    if wall_treatment == "resolved" and q > VISCOUS_SUBLAYER_Y_PLUS:
        logger.warning(
            f"壁面 y+ 的 95% 分位为 {q:.3g}，贴壁单元越出粘性底层（y+ < {VISCOUS_SUBLAYER_Y_PLUS:g}）：低雷诺数壁面处理"
            f"（--wall-treatment resolved）的壁面条件在这里不成立。加密壁面网格（grid generate-volume --target-yplus 1）"
            f"或改用 --wall-treatment auto")
    elif wall_treatment == "auto" and q > LOG_LAYER_Y_PLUS:
        logger.warning(
            f"壁面 y+ 的 95% 分位为 {q:.3g}，匹配点越出对数律层（y+ 约 {LOG_LAYER_Y_PLUS:g} 以上为尾流区），"
            f"平衡壁面模型的壁面律在这里不再描述速度剖面；加密壁面网格")
    return yp
