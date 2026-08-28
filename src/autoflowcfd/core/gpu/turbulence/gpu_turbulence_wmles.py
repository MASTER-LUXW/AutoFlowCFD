"""
AutoFlowCFD V2.0 - GPU 版 WMLES 壁面剪应力修正接入 (#7 第四次评审第四轮)

`core/utils/solver_helpers.py::compute_wmles_wall_stress_correction`
本身只在 WALL 边界面（占总面数很小一部分）上运行：对每个 WALL 面用
owner 单元的坍缩坐标模态基外插（`ops.boundary_extrap_prism/tet`）+
`_distribute_from_face` 投影，这套外插/分配机制是纯 CPU 端专用对象，
本项目里没有、也没有必要为它单独建一套 GPU 版本——这不是一个每步都要
扫过全部单元/SPs 的热点路径（如残差体积项/界面项那样），而是与边界
幽灵态计算完全同一类"只在边界面小子集上跑、CPU↔GPU 往返可忽略"的
场景（见 gpu_inviscid.py/gpu_viscous.py 的边界幽灵态处理文档，同一套
既定模式）。

本文件因此不重新实现摩擦速度迭代求解/坍缩坐标外插，而是直接复用已
验证的 CPU 函数本身，用一个轻量 facade 对象把 GPU 求解器的状态
（Q_gpu/wall_distance_gpu 下载到 CPU）适配成该函数期望的 `solver` 接口。
"""

import types

from autoflowcfd.core.gpu import get_cupy


def compute_wmles_wall_stress_correction_gpu(solver, U=None, Q=None):
    """GPU 版 WMLES 壁面剪应力修正入口，与 CPU 版
    `FRSolver.compute_viscous_residual` 里的同名调用点语义完全一致：
    必须在残差组装阶段（`compute_viscous_residual_gpu` 内部）叠加到
    粘性残差上，不能放到 step() 完成状态更新之后才生效。

    Args:
        solver: GPUFRSolver 实例，需要 `solver.wmles_model` 已设置为真实
            的 CPU 版 `WMLESModel` 实例（见 gpu_solver.py 构造处），
            `solver.wall_distance_gpu` 已初始化。
        U, Q: 可选，调用方（`compute_viscous_residual_gpu`）当前正在
            求值的试验解（CuPy 数组）——SSP-RK2/RK3 每个 stage 用的 U
            与 `solver.U_gpu`（已接受的上一步解）不同，必须用这一 stage
            自己的状态算壁面剪应力，不能恒用 `solver.U_gpu`/`solver.
            Q_gpu`（那是与 CPU 版 `compute_viscous_residual` 读
            `solver.state.Q`——RK 子步临时替换过的 trial 状态——同一个
            语义要求，否则多 stage 格式下壁面剪应力会用错 stage 的状态）。
            两者都为 None 时退回 `solver.U_gpu`/`solver.Q_gpu`（对应
            forward-Euler 单 stage 或调用方明确要用当前接受态的场景）。

    Returns:
        (n_cells, n_sps, 5) CuPy 数组，无 WMLES 模型/无 WALL 边界/尚未
        计算壁面距离时返回 None（与 CPU 版语义一致）。
    """
    if solver.wmles_model is None:
        return None
    cp = get_cupy()

    from autoflowcfd.core.utils.solver_helpers import compute_wmles_wall_stress_correction

    U_gpu = U if U is not None else solver.U_gpu
    if Q is not None:
        Q_gpu = Q
    else:
        from autoflowcfd.core.gpu.residual.gpu_flux import conserved_to_primitive_gpu
        Q_gpu = conserved_to_primitive_gpu(U_gpu[..., :5]) if U is not None else solver.Q_gpu

    U_cpu = cp.asnumpy(U_gpu)
    Q_cpu = cp.asnumpy(Q_gpu)
    wall_distance_cpu = (
        cp.asnumpy(solver.wall_distance_gpu) if solver.wall_distance_gpu is not None else None
    )

    state = types.SimpleNamespace(U=U_cpu, Q=Q_cpu)
    facade = types.SimpleNamespace(
        wmles_model=solver.wmles_model,
        mesh=solver.mesh,
        ops=solver.ops,
        wall_distance=wall_distance_cpu,
        state=state,
    )

    correction_cpu = compute_wmles_wall_stress_correction(facade)
    if correction_cpu is None:
        return None
    return cp.asarray(correction_cpu)
