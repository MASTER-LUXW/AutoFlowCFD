"""tetgen 核心域填充：Steiner 点预算估算与远场背景点播种。

从 mesh_tetgen_core.py 拆分出来，是 fill_core_volume 调用前的两个准备
步骤：估算给 tetgen 的 Steiner 点预算（steinerleft），以及在空旷远场
预先播撒一批背景点，避免 tetgen 初始 Delaunay 阶段连出跨越整个域的巨大
四面体。
"""

from typing import List, Optional, Tuple

import numpy as np
from loguru import logger


def estimate_steinerleft(
    points: np.ndarray,
    regions: Optional[List[Tuple[np.ndarray, int, float]]],
) -> int:
    """Estimate a Steiner-point budget (tetgen's `steinerleft`) generous
    enough for the requested region(s), scaled to the actual problem size
    rather than a fixed constant.

    tetgen's default steinerleft=100000 is a global cap on how many Steiner
    points it will ever insert, shared across the WHOLE mesh - with a
    region's own maxvolume target well below the PLC's natural
    (unconstrained) tet size, it can run out long before that target is
    reached everywhere, silently leaving a long tail of oversized cells in
    whatever pockets happened to refine last (measured directly: a 5.5x3x3
    m domain capped at 0.05 m with a fixed 300,000 budget left 6-10% of
    cells over 1.5x the target and a worst-case cell ~5-6x over).

    The domain-wide grading region (present whenever max_cell_size is set -
    see mesh_background._build_merged_mesh) always has the LARGEST maxvol
    of any region passed here, so bbox_volume / coarsest_maxvol estimates
    how many cells it alone needs to fill the core - that's the number this
    behaves identically to when there is exactly one region (unchanged from
    the original single-region formula, and - as of Stage B's core-side
    local repair regions being removed, see mesh_repair.py's module
    docstring - the only case `regions` now ever actually contains in
    practice: at most 1 entry).

    The `n_extra_regions` handling below is dead in current usage but kept
    rather than special-cased away, in case a future caller legitimately
    passes more than one region again: dividing the FULL bbox by the
    smallest maxvol among several regions (an earlier version of this
    function, using `min(maxvol for ...)`) badly overestimates whenever one
    of them is a small local patch rather than a domain-wide target -
    observed directly on a real case with Stage B's now-removed core
    regions, an estimate of ~17.8 BILLION target-sized tets for a domain
    whose single-region core fill converged around 1.2M tets. Note this
    estimate is advisory only, not a hard constraint: tetgen was confirmed
    to converge to the *identical* actual tet count regardless of whether
    steinerleft was the (buggy) inflated value or this function's corrected
    one - the real 5x core-fill blowup that estimate coincided with
    (1.2M -> 6.1M tets) turned out to be a separate, still-unresolved
    tetgen multi-region-refinement behavior (see mesh_repair.py), not
    something this budget number was ever actually causing.

    Args:
        points: PLC boundary points, shape=(n, 3) - only used for its
            bounding-box volume
        regions: (seed_point, region_id, maxvol) tuples, or None/empty for
            an unconstrained (nobisect=True) fill

    Returns:
        steinerleft, clamped to [300_000, 20_000_000] - or 100_000
        (tetgen's own default) when no regions are active at all.
    """
    if not regions:
        return 100_000

    bbox_volume = float(np.prod(np.max(points, axis=0) - np.min(points, axis=0)))
    coarsest_maxvol = max(maxvol for _, _, maxvol in regions)
    estimated_tets = bbox_volume / max(coarsest_maxvol, 1e-30)

    n_extra_regions = len(regions) - 1
    extra_tets = n_extra_regions * 200_000

    logger.info(
        f"Steiner-point budget estimate: ~{estimated_tets:,.0f} domain-wide target-sized tets"
        + (f" + {n_extra_regions} local repair region(s) x 200,000" if n_extra_regions else "")
    )
    return int(np.clip((estimated_tets + extra_tets) * 3.0, 300_000, 20_000_000))


def generate_core_background_points(
    plc_points: np.ndarray,
    plc_faces: np.ndarray,
    target_edge_length: float,
    grid_spacing_factor: float = 2.5,
    clearance_factor: float = 3.0,
) -> np.ndarray:
    """Pre-seed the sparse far field with a coarse background point grid, to
    be passed to `fill_core_volume` as `background_points` so tetgen's
    INITIAL Delaunay tetrahedralization already has points spread through
    empty far-field space, instead of only the PLC's own boundary points.

    Root cause this targets: with only boundary points as input, tetgen's
    first-pass Delaunay step can connect distant boundary points (e.g.
    inlet-to-outlet, across genuinely empty space) into one huge initial
    tet; its own SECOND-pass quality/volume refinement is then relied on to
    split it back down toward the region's max_cell_size target - but was
    found, on a real case, to leave at least one such tet (14.15 m^3, see
    mesh_background_merge.py's own history for this finding) completely
    unrefined, identically whether volume_cap_fraction was loosened or
    tightened, or whether the region had one seed or ~27 scattered ones -
    neither changed that cell at all. A point already present at the FIRST
    pass can't be "missed" by a refinement pass that runs later - this
    sidesteps reliance on that second pass ever reaching the far field, at
    least at this function's own (coarse) spacing.

    Two filters keep the candidate grid from doing more harm than good:
      (a) clearance from the existing PLC surface (`clearance_factor *
          target_edge_length`, checked against the nearest PLC point via
          KDTree) - close to the BL outer surface or a fine core-only wall,
          the existing mesh is already fine enough, and a background point
          crowding in there risks a degenerate sliver instead of helping;
      (b) genuinely inside the closed PLC volume (ray-casting parity test,
          reusing mesh_domain_classify's own vectorized ray/triangle
          intersection routine) - a point outside the PLC would violate
          tetgen's assumption that every input point lies within the
          region its facets enclose, which for a NON-convex domain (a real
          possibility here - a car body's own hole carves a concavity out
          of an otherwise box-like tunnel) a plain bounding-box grid alone
          cannot guarantee.

    Args:
        plc_points: (n, 3) full PLC boundary point set (BL outer surface +
            core-only faces) - the SAME array `fill_core_volume` receives
            as its own `points`
        plc_faces: (m, 3) full PLC boundary triangles, closed and
            watertight - the SAME array `fill_core_volume` receives as its
            own `faces`
        target_edge_length: the far-field grading target (max_cell_size)
            this grid should not need to be finer than
        grid_spacing_factor: background grid spacing, as a multiple of
            target_edge_length. Deliberately coarser than the target
            itself - this is a seed grid to break up otherwise-huge
            initial tets, not a substitute for the region's own volume-
            based refinement, which still runs on top of it
        clearance_factor: minimum allowed distance to the nearest PLC
            point, as a multiple of target_edge_length

    Returns:
        (k, 3) float64 background points, k possibly 0 if the domain is
        too small (relative to target_edge_length) for any grid cell to
        clear both filters
    """
    from scipy.spatial import cKDTree
    from .mesh_domain_classify import _ray_triangle_intersect_count

    if target_edge_length <= 0.0 or len(plc_points) == 0:
        return np.empty((0, 3), dtype=np.float64)

    bbox_min = plc_points.min(axis=0)
    bbox_max = plc_points.max(axis=0)
    spacing = target_edge_length * grid_spacing_factor

    axes = [
        np.arange(bbox_min[i] + spacing * 0.5, bbox_max[i], spacing)
        for i in range(3)
    ]
    if any(len(a) == 0 for a in axes):
        return np.empty((0, 3), dtype=np.float64)

    gx, gy, gz = np.meshgrid(*axes, indexing='ij')
    candidates = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    tree = cKDTree(plc_points)
    clearance = target_edge_length * clearance_factor
    dist, _ = tree.query(candidates, k=1, workers=-1)
    candidates = candidates[dist >= clearance]
    if len(candidates) == 0:
        logger.info("Core background-point seeding: 0 candidates cleared the PLC-clearance filter")
        return np.empty((0, 3), dtype=np.float64)

    v0 = plc_points[plc_faces[:, 0]]
    v1 = plc_points[plc_faces[:, 1]]
    v2 = plc_points[plc_faces[:, 2]]
    direction = np.array([1.0, 0.0, 0.0])
    inside_mask = np.zeros(len(candidates), dtype=bool)
    for i in range(len(candidates)):
        hits = _ray_triangle_intersect_count(candidates[i], direction, v0, v1, v2)
        inside_mask[i] = (hits % 2) == 1

    result = candidates[inside_mask].astype(np.float64)
    logger.info(
        f"Core background-point seeding: {len(result)}/{len(candidates)} inside-domain "
        f"candidates kept (grid spacing={spacing:.3f}m, clearance={clearance:.3f}m)"
    )
    return result
