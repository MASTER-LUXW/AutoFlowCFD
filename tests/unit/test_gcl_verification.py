"""HighOrderMesh.verify_gcl 回归测试。

2026-08-23 真实 bug 修复：P1 阶的 GCL 诊断此前给出 0.105（远超
tolerance=1e-8），被此前的调查记录为"已知、搁置"的诊断局限——真实
求解器残差路径完全正常（P1 均匀自由流场相对残差 1.25e-10，是 P1/P2/P3
三者里最好的，见 order_continuation.py 模块文档），问题出在诊断函数
自己用跟当前求解阶数绑死的坍缩坐标微分矩阵去微分一个有理函数
（adj(J)），P1 的 degree-1 矩阵次数不够。修复：P1 专门改用真实求解器
体积项本来就在用的过积分（over-integration）基础设施做这个诊断检验，
见 HighOrderMesh.verify_gcl/_verify_gcl_overintegrated 文档。
"""

from .test_fr_residual_inviscid import _build_synthetic_mixed_mesh


def test_p1_gcl_passes_after_overintegration_fix():
    """真实 bug 回归测试：P1 GCL 此前在这份合成网格上给出 0.105，
    远超 tolerance=1e-8，verify_gcl() 返回 False。修复后应稳定通过。"""
    mesh = _build_synthetic_mixed_mesh(1)
    assert mesh.verify_gcl(tolerance=1e-8) is True


def test_p2_gcl_still_passes_native_path_unaffected():
    """P2 保留原生（非过积分）检验路径不变——确认这次只为 P1 新增的
    过积分分支没有连带影响到 P2 已经良好的行为。"""
    mesh = _build_synthetic_mixed_mesh(2)
    assert mesh.verify_gcl(tolerance=1e-8) is True
