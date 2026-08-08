"""FRSolver.solve() 每次求解开始前的一次性准备工作。

从 solver_steady.py 拆出来的 mixin：组装面几何（FaceGeometry）、估算
壁面距离、构造本次求解要用的 ViscousRANSResidual、预热参考面积缓存——
这些都是 solve() 迭代循环开始前只执行一次的线性准备步骤，和循环本体
（CFL 自适应、收敛判断、checkpoint）逻辑上是两件事，拆成单独文件只是
为了控制单文件行数，不改变任何计算逻辑或调用顺序。
"""

import time

import numpy as np
from loguru import logger

from ..config.solver_config import TurbulenceModel
from .fvm_gradients import FaceGeometry
from .fvm_viscous_residual import ViscousRANSResidual, estimate_wall_distance


class SteadySetupMixin:
    """提供 `_prepare_geometry_and_residual` 给 `FRSolver`。

    依赖宿主类（`FRSolver`）已有的 `self.grid_data`/`self.config`/
    `self.face_extractor`/`self.bc_handler`/`self.aero_calculator`/
    `self._get_cell_volumes`/`self._use_gpu_residual` 属性，不独立维护状态。
    """

    def _prepare_geometry_and_residual(self):
        """组装 geom/wall_distance/residual 等 solve() 循环需要的一次性对象。

        Returns:
            (geom, wall_distance, wall_face_mask, mu_lam, turbulent,
             mach_ref, residual) 元组，字段含义见各自在 solve() 里原先的
             同名局部变量。
        """
        # CRITICAL FIX: Use optimized face extraction from VolumeMeshData instead of slow FVMFaceExtractor
        logger.info("Using pre-computed face data from VolumeMeshData (optimized radix-sort)...")
        t_face_start = time.perf_counter()

        # Ensure faces exist (uses optimized FaceExtractor with argsort)
        face_data_obj = self.grid_data.ensure_faces_exist()

        # Compute cell centroids FIRST (needed for gradient reconstruction)
        nodes_array = np.column_stack([
            self.grid_data.nodes.x,
            self.grid_data.nodes.y,
            self.grid_data.nodes.z,
        ])
        # Prism cells (if any - see VolumeMeshData.prism_cells) occupy the
        # front of the global cell-index space, tets the rest (same
        # convention grid_data.get_cell_volumes() below already follows) -
        # centroids must be built the same way, or they'd misalign against
        # cell_volumes/geom.cell_centroids for every BL cell once a prism
        # mesh is in play (a plain tets-only average silently produced only
        # n_tet rows, not n_prism+n_tet, for a mixed mesh here previously).
        tet_connectivity_int64 = self.grid_data.cells.connectivity.astype(np.int64)
        tet_centroids = nodes_array[tet_connectivity_int64].mean(axis=1)
        prism_cells_obj = getattr(self.grid_data, 'prism_cells', None)
        if prism_cells_obj is not None:
            prism_connectivity_int64 = prism_cells_obj.connectivity.astype(np.int64)
            prism_centroids = nodes_array[prism_connectivity_int64].mean(axis=1)
            cell_centroids = np.vstack([prism_centroids, tet_centroids])
        else:
            cell_centroids = tet_centroids

        # Store in face_extractor for later use
        self.face_extractor.cell_centroids = cell_centroids

        # Convert FaceData to the plain-dict format the rest of solve() uses
        face_data = {
            'connectivity': face_data_obj.connectivity,
            'normals': face_data_obj.normal,
            'areas': face_data_obj.area,
            'centers': face_data_obj.center,
            'boundary_flags': (face_data_obj.connectivity[:, 1] < 0).astype(np.int32),
            'cell_centroids': cell_centroids,
        }

        t_face_end = time.perf_counter()
        logger.success(f"Face data prepared in {t_face_end - t_face_start:.2f}s (optimized)")

        # Expose face data on face_extractor - bc_handler/aero_calculator
        # both read face arrays from this shared holder.
        self.face_extractor.face_connectivity = face_data['connectivity']
        self.face_extractor.face_normals = face_data['normals']
        self.face_extractor.face_areas = face_data['areas']
        self.face_extractor.boundary_flags = face_data['boundary_flags']

        # Assemble shared geometry bundle for the residual.
        cell_volumes = self._get_cell_volumes()
        geom = FaceGeometry(
            connectivity=face_data['connectivity'],
            normals=face_data['normals'],
            areas=face_data['areas'],
            centers=face_data['centers'],
            boundary_flags=face_data['boundary_flags'],
            cell_centroids=face_data['cell_centroids'],
            cell_volumes=cell_volumes,
        )

        # Wall distance from viscous-wall boundary faces (WALL/GROUND).
        try:
            self.bc_handler._precompute_face_types()
            wall_face_mask = np.zeros(geom.n_faces, dtype=bool)
            for f, t in self.bc_handler._face_types.items():
                if t in ("WALL", "GROUND"):
                    wall_face_mask[f] = True
            logger.info(f"Wall face mask computed: {np.sum(wall_face_mask)} wall faces")
            wall_distance = estimate_wall_distance(geom, wall_face_mask)
            logger.info(f"Wall distance estimated: min={wall_distance.min():.4e}, max={wall_distance.max():.4e}")
        except Exception as e:
            logger.error(f"Failed to compute wall distance: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

        # Molecular viscosity (Sutherland at 288 K ~ 1.79e-5 Pa s).
        mu_lam = 1.7894e-5
        turbulent = self.config.turbulence != TurbulenceModel.NONE

        # Reference (freestream) Mach number, used two ways:
        #  1. ViscousRANSResidual's inviscid flux: AUSM+up (see
        #     fvm_viscous_residual.py's _ausm_up), which replaced HLLC as
        #     the live flux specifically for its low-Mach robustness. It
        #     has a built-in low-Mach scaling function f_a that this
        #     reference Mach regularizes so f_a stays bounded away from
        #     zero at genuine stagnation points, instead of a wave-speed-
        #     bracket-based preconditioner (tried first, reverted - it
        #     narrowed HLLC's SL/SR margin around the star-state wave
        #     speed Sstar by ~10x at this case's M~0.09 everywhere in the
        #     domain at once, causing a much faster, more widespread
        #     numerical blow-up than the unpreconditioned scheme; see
        #     fvm_viscous_residual.py's _hllc docstring for the history).
        #  2. TimeIntegrator.local_time_step below: relaxes the pseudo-
        #     time CFL restriction that a density-based scheme otherwise
        #     inherits from the acoustic speed (~340 m/s) even though the
        #     physical flow here is far slower - this part never touches
        #     the flux itself, only how big a step is stable to take.
        gamma_air = 1.4
        a_inf = np.sqrt(gamma_air * self.config.p_inf / self.config.rho_inf)
        mach_ref = self.config.vel_inf / max(a_inf, 1e-30)

        residual = ViscousRANSResidual(
            geom, mu_lam=mu_lam, wall_distance=wall_distance, turbulent=turbulent,
            mach_ref=mach_ref,
            wall_face_mask=wall_face_mask if self.config.use_wall_functions else None,
            use_gpu=self._use_gpu_residual,
        )
        if self.config.use_wall_functions:
            logger.info(
                "Wall functions enabled (Menter scalable/automatic wall treatment) "
                f"on {np.sum(wall_face_mask)} WALL/GROUND faces - near-wall mesh no "
                "longer needs y+~1 to be accurate."
            )

        # Aerodynamic reference area - called here only to warm
        # AeroCoefficientCalculator's internal cache (_cached_ref_area)
        # before the iteration loop starts; the per-iteration
        # compute_coefficients() call re-fetches it from that cache, so
        # the return value itself isn't needed here.
        body_face_indices = self.aero_calculator._identify_body_faces()
        self.aero_calculator._compute_reference_area(body_face_indices)

        return geom, wall_distance, wall_face_mask, mu_lam, turbulent, mach_ref, residual
