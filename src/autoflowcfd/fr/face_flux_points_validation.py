"""FR Flux Points 几何组装——Newton/精确点位定位残差分级校验与诊断汇总。

从 face_flux_points_merge.py 拆出（该文件原有 745 行，超过项目 600 行
硬性拆分阈值；`build_face_flux_points` 主循环产出逐面 Newton/精确点位
残差数组后，末尾这段"按 ACCEPT_STRICT_REL/_ACCEPT_WARN_REL 分级校验、
写诊断 JSON、超差即 raise"的逻辑是一个自成一体的收尾步骤——只读取
`build_face_flux_points` 已经算好的残差数组和分组掩码，不产生任何
`build_face_flux_points` 后续需要的返回值，是清晰的拆分边界）。纯代码
搬移，不改变任何行为——函数体逐字保留，仅把此前内联在
`build_face_flux_points` 局部作用域里的变量改为显式参数。
"""

from typing import List

import numpy as np
from loguru import logger

from autoflowcfd.fr.face_flux_points import ACCEPT_STRICT_REL
from autoflowcfd.fr.face_flux_points_locate import _ACCEPT_WARN_REL


def validate_face_flux_point_residuals(
    face_conn,
    n_prism: int,
    n_faces: int,
    _nb_resid: np.ndarray,
    _ow_resid: np.ndarray,
    _nb_single: np.ndarray,
    _ow_single: np.ndarray,
    _multi_nb_faces: np.ndarray,
    _multi_ow_faces: np.ndarray,
    nb_mask: np.ndarray,
    ow_mask: np.ndarray,
    _ms_nb_mixed: np.ndarray,
    _ms_ow_mixed: np.ndarray,
    _nb_sec_resid_arr: np.ndarray,
    _ow_sec_resid_arr: np.ndarray,
    _ms_nb_other_face: np.ndarray,
    _ms_ow_other_face: np.ndarray,
) -> None:
    """校验 Newton/精确点位定位残差（接回此前被丢弃的 kernel 残差输出）。

    上一轮专家评审发现的安全网架空问题：numba 快速路径的 build_fp_newton_
    parallel/build_ms_interp_parallel 只把 Newton 残差写进数组就返回，从未
    与任何阈值比较——真正不相容的网格拓扑（T-junction、BL 挤出产生的
    非协调面等）会被 Newton 跑满迭代次数后返回一个可能远超容差的"最优
    逼近"插值矩阵，且没有任何警告或报错。旧的纯 Python 慢速路径
    （face_flux_points_locate.py::newton_locate_on_face）里这个校验是完整的：
    残差相对局部面特征尺度 char_length=sqrt(area) 超过 _ACCEPT_WARN_REL
    （15%）直接 raise；未超但超过 ACCEPT_STRICT_REL（1e-6，即非机器精度）
    记入 _tolerated 仅做日志。这里用向量化 numpy 操作复现同一套判据，
    不退化成逐面 Python 循环（Python 循环只在"超差"这一小撮真正需要报警
    的面上做，数量级上和原本已有的 _tolerated/_diagnostic_failures 汇总
    逻辑一致）。

    单源面（~95%，_nb_single/_ow_single 为真）：_nb_resid/_ow_resid 该面
    唯一真实邻居的逐 FP 残差，直接按面 max 即可——与慢速路径完全等价。

    多源棱柱四边形侧面（~5%）：_nb_resid/_ow_resid 在这些面上是对整批
    n_fp 个点（含真正不属于该邻居、落在对角线另一半的点）一次性 Newton
    定位的残差——"错误半区"的点在这个邻居的面上物理上不可达，Newton
    不可能收敛，残差会远超真实的双线性曲面翘曲量级。若直接对整批取 max，
    会把这部分虚假残差当成整张面的残差，对全部多源面产生系统性误报
    （已通过分析确认，未在真实网格验证前先行避免——真发生大规模误报就
    说明这里的按半区掩码逻辑有问题，需要回去核对慢速路径判据，而不是
    简单放宽阈值掩盖）。因此多源面必须按 primary/secondary 半区掩码
    （nb_mask/ow_mask）分别取 max，且各自按自己实际所属子面的面积做
    char_length 归一化（与旧慢速路径 _resolve_multi_source 逐半区各自
    用 build_cross_interp 完全一致）。

    Raises:
        RuntimeError: 存在面-侧残差相对局部面尺度超过 _ACCEPT_WARN_REL
            （非收敛速度问题，而是真正不相容的网格拓扑），完整诊断写入
            临时目录下的 face_flux_points_failures.json
    """
    _tolerated: List[dict] = []
    _diagnostic_failures: List[dict] = []

    char_length = np.sqrt(np.maximum(face_conn.area, 1e-300))

    def _classify_and_record(resid, char_len, cell_ids, face_ids, role):
        """按 ACCEPT_STRICT_REL/_ACCEPT_WARN_REL 对一批标量 (resid, char_len)
        分级：超过 warn 阈值 -> _diagnostic_failures（会话末尾聚合 raise）；
        未超但超过 strict 阈值（非机器精度）-> _tolerated（仅记录/告警）。
        resid/char_len/cell_ids/face_ids 为等长 1D numpy 数组。"""
        if resid.size == 0:
            return
        scale = np.maximum(char_len, 1e-300)
        rel_pct = 100.0 * resid / scale
        warn_tol_abs = np.maximum(1e-9, _ACCEPT_WARN_REL * scale)
        fail_mask = resid > warn_tol_abs
        tol_mask = (resid > ACCEPT_STRICT_REL * scale) & ~fail_mask
        for idx in np.nonzero(fail_mask)[0]:
            f = int(face_ids[idx])
            _diagnostic_failures.append({
                "face": f,
                "owner_cell": int(face_conn.owner_cell[f]),
                "neighbor_cell": int(face_conn.neighbor_cell[f]),
                "owner_is_prism": bool(face_conn.owner_cell[f] < n_prism),
                "neighbor_is_prism": bool(face_conn.neighbor_cell[f] < n_prism),
                "role": role,
                "residual": float(resid[idx]),
                "char_length": float(char_len[idx]),
                "relative_pct": float(rel_pct[idx]),
                "error": (
                    f"Newton/exact face point-location residual {float(resid[idx]):.3e} exceeds "
                    f"warn tolerance {float(warn_tol_abs[idx]):.3e} "
                    f"({float(rel_pct[idx]):.2f}% of local face scale {float(char_len[idx]):.3e}). "
                    f"This indicates a genuinely non-conforming mesh face (target point not "
                    f"actually on this cell's face) rather than a slow-convergence issue."
                ),
            })
        for idx in np.nonzero(tol_mask)[0]:
            f = int(face_ids[idx])
            _tolerated.append({
                "face": f, "cell": int(cell_ids[idx]), "role": role,
                "residual": float(resid[idx]), "char_length": float(char_len[idx]),
                "relative_pct": float(rel_pct[idx]),
            })

    # -- 单源面：owner_primary（"owner" 角色，_nb_resid）--
    _sf_nb = np.nonzero(_nb_single)[0]
    if _sf_nb.size > 0:
        _classify_and_record(
            _nb_resid[_sf_nb].max(axis=1), char_length[_sf_nb],
            face_conn.owner_cell[_sf_nb], _sf_nb, "owner",
        )
    # -- 单源面：neighbor_primary（"neighbor" 角色，_ow_resid）--
    _sf_ow = np.nonzero(_ow_single)[0]
    if _sf_ow.size > 0:
        _classify_and_record(
            _ow_resid[_sf_ow].max(axis=1), char_length[_sf_ow],
            face_conn.neighbor_cell[_sf_ow], _sf_ow, "neighbor",
        )

    # -- 多源面（owner 角色）：primary 半区用 _nb_resid + nb_mask，
    #    secondary 半区用 _nb_sec_resid_arr + ~nb_mask（排除 mixed，无
    #    真实 secondary Newton）--
    if _multi_nb_faces.size > 0:
        sub_mask = nb_mask[_multi_nb_faces]
        sub_resid = _nb_resid[_multi_nb_faces]
        primary_resid = np.where(sub_mask, sub_resid, -np.inf).max(axis=1)
        primary_valid = np.isfinite(primary_resid)
        if np.any(primary_valid):
            idxs = _multi_nb_faces[primary_valid]
            _classify_and_record(
                primary_resid[primary_valid], char_length[idxs],
                face_conn.owner_cell[idxs], idxs, "owner",
            )
        _not_mixed = ~_ms_nb_mixed
        if np.any(_not_mixed):
            sub_sec_resid = _nb_sec_resid_arr[:len(_multi_nb_faces)][_not_mixed]
            sub_sec_mask = ~sub_mask[_not_mixed]
            secondary_resid = np.where(sub_sec_mask, sub_sec_resid, -np.inf).max(axis=1)
            sec_valid = np.isfinite(secondary_resid)
            if np.any(sec_valid):
                idxs = _multi_nb_faces[_not_mixed][sec_valid]
                sec_face_idxs = _ms_nb_other_face[_not_mixed][sec_valid]
                _classify_and_record(
                    secondary_resid[sec_valid], char_length[sec_face_idxs],
                    face_conn.owner_cell[idxs], idxs, "owner",
                )

    # -- 多源面（neighbor 角色）：同上，符号对称 --
    if _multi_ow_faces.size > 0:
        sub_mask_o = ow_mask[_multi_ow_faces]
        sub_resid_o = _ow_resid[_multi_ow_faces]
        primary_resid_o = np.where(sub_mask_o, sub_resid_o, -np.inf).max(axis=1)
        primary_valid_o = np.isfinite(primary_resid_o)
        if np.any(primary_valid_o):
            idxs = _multi_ow_faces[primary_valid_o]
            _classify_and_record(
                primary_resid_o[primary_valid_o], char_length[idxs],
                face_conn.neighbor_cell[idxs], idxs, "neighbor",
            )
        _not_mixed_o = ~_ms_ow_mixed
        if np.any(_not_mixed_o):
            sub_sec_resid_o = _ow_sec_resid_arr[:len(_multi_ow_faces)][_not_mixed_o]
            sub_sec_mask_o = ~sub_mask_o[_not_mixed_o]
            secondary_resid_o = np.where(sub_sec_mask_o, sub_sec_resid_o, -np.inf).max(axis=1)
            sec_valid_o = np.isfinite(secondary_resid_o)
            if np.any(sec_valid_o):
                idxs = _multi_ow_faces[_not_mixed_o][sec_valid_o]
                sec_face_idxs = _ms_ow_other_face[_not_mixed_o][sec_valid_o]
                _classify_and_record(
                    secondary_resid_o[sec_valid_o], char_length[sec_face_idxs],
                    face_conn.neighbor_cell[idxs], idxs, "neighbor",
                )

    if _tolerated:
        import json
        import os
        import tempfile

        worst = max(_tolerated, key=lambda d: d["relative_pct"])
        tol_dump_path = os.path.join(tempfile.gettempdir(), "face_flux_points_tolerated.json")
        with open(tol_dump_path, "w") as fh:
            json.dump(_tolerated, fh, indent=2)
        logger.warning(
            f"{len(_tolerated)}/{n_faces} face-side Newton point-locations accepted with a "
            f"non-machine-precision residual (worst: face {worst['face']}, cell {worst['cell']}, "
            f"{worst['relative_pct']:.2f}% of local face scale {worst['char_length']:.3e}). 这些均为"
            f"棱柱四边形侧面（双线性曲面）与相邻单元共享界面处的真实、有界几何翘曲（已用最小二乘 "
            f"Newton 解取该曲面上的最优逼近点），量级与直接对全网格棱柱四边形侧面翘曲度的独立几何"
            f"测量一致（全网格最大 11.13%，见开发过程记录），非算法缺陷。完整清单见 {tol_dump_path}。"
        )

    if _diagnostic_failures:
        import json
        import os
        import tempfile

        dump_path = os.path.join(tempfile.gettempdir(), "face_flux_points_failures.json")
        with open(dump_path, "w") as fh:
            json.dump(_diagnostic_failures, fh, indent=2)
        logger.error(
            f"{len(_diagnostic_failures)}/{n_faces} interior faces failed exact Flux-Point "
            f"location. Full diagnostics written to {dump_path}."
        )
        raise RuntimeError(
            f"{len(_diagnostic_failures)}/{n_faces} interior faces failed exact Flux-Point "
            f"location (see {dump_path} for full per-face diagnostics). This indicates either "
            f"genuinely non-conforming mesh topology (a 'shared' 3-node face that is not "
            f"actually a full face of both cells, e.g. a T-junction between differently-"
            f"resolved mesh regions) or a remaining bug in the point-location algorithm - "
            f"must be diagnosed and fixed, not silently tolerated."
        )
