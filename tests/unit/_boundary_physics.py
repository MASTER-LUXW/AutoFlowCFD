"""测试用的边界幽灵态提供者构造：显式物理设置（`BoundaryPhysics`）的一组常用取值，个别测试按需覆盖。"""

from autoflowcfd.core.fr_solver.boundary import BoundaryPhysics, build_boundary_ghost_provider


def ghost_provider(mesh, freestream, turb_model_name="NONE", *, ops=None, wall_modelled=False, mu_molecular=1.8e-5,
                   turbulence_intensity=0.01, viscosity_ratio=5.0, sem_num_eddies=None, bc_overrides=None):
    """`ops` 缺省取网格自己的同阶算子（只有合成湍流入口的通量点定位用到）。"""
    physics = BoundaryPhysics(freestream, mu_molecular, turb_model_name, wall_modelled, turbulence_intensity,
                              viscosity_ratio, sem_num_eddies)
    return build_boundary_ghost_provider(mesh, mesh.operators if ops is None else ops, physics, bc_overrides or {})
