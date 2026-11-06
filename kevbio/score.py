"""FU/GU Biometry 2026 metric: per-view MRE (px) and measurement MAE (px; degrees for the AoP angle), macro-averaged
over the nine views. Ported from US-Foundation (gubiometry/metrics.py). The FA/HC projection is from
https://github.com/harshitAgr/fubiometry2026 (docker/geometry_project.py), Copyright 2026 Harshit Agrawal /
AIATELLA Oy, Apache-2.0."""
import math

import numpy as np

TASK_ORDER = ["A4C", "AOP", "FA", "FUGC", "HC", "IVC", "PLAX", "PSAX", "fetal_femur"]


def _dist(p, i, j):
    return float(np.hypot(p[i, 0] - p[j, 0], p[i, 1] - p[j, 1]))


def _angle_deg(p, a, b, c):
    v1, v2 = p[b] - p[a], p[c] - p[a]
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9: return 0.0
    return math.degrees(math.acos(float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))))


def _pi_half_sum(p, i, j, k, l):
    return math.pi * (_dist(p, i, j) + _dist(p, k, l)) / 2.0


def _min_dist(p, i, j, k, l):
    return min(_dist(p, i, j), _dist(p, k, l))


# (name, kind, landmark indices) per view, from the organisers' measurement table
MEASUREMENT_SPECS = {
    "A4C": [("LV_horizontal", "dist", (0, 1)), ("LV_vertical", "dist", (2, 3)),
            ("RV_horizontal", "dist", (4, 5)), ("RV_vertical", "dist", (6, 7)),
            ("LA_horizontal", "dist", (8, 9)), ("LA_vertical", "dist", (10, 11)),
            ("RA_horizontal", "dist", (12, 13)), ("RA_vertical", "dist", (14, 15))],
    "AOP": [("HSD", "dist", (0, 2)), ("AOP", "angle", (0, 1, 3))],
    "FA": [("FA", "pi_half_sum", (0, 1, 2, 3))],
    "FUGC": [("CL", "dist", (0, 1))],
    "HC": [("BPD", "min_dist", (0, 1, 2, 3)), ("HC", "pi_half_sum", (0, 1, 2, 3))],
    "IVC": [("IVC", "dist", (0, 1))],
    "PLAX": [("LV", "dist", (0, 1)), ("RV", "dist", (2, 3)), ("IVS", "dist", (4, 5)),
             ("LVPW", "dist", (6, 7)), ("VAO", "dist", (8, 9)), ("STJ", "dist", (10, 11)),
             ("AAO", "dist", (12, 13)), ("AV", "dist", (14, 15)), ("LVOT", "dist", (16, 17)),
             ("LA", "dist", (18, 19)), ("RVOT", "dist", (20, 21))],
    "PSAX": [("RVOT", "dist", (0, 1)), ("PA", "dist", (2, 3))],
    "fetal_femur": [("FL", "dist", (0, 1))],
}
_KIND = {"dist": _dist, "angle": _angle_deg, "pi_half_sum": _pi_half_sum, "min_dist": _min_dist}


def measurements_from_points(task, pts):
    """{name: value}; None when a required landmark is missing (< 0)."""
    pts = np.asarray(pts, dtype=np.float64)
    return {name: None if any(pts[i, 0] < 0 or pts[i, 1] < 0 for i in idx) else _KIND[kind](pts, *idx)
            for name, kind, idx in MEASUREMENT_SPECS.get(task, [])}


def project_fa(p):
    """Axis-aligned, concentric FA diameters with their lengths kept."""
    p = np.asarray(p, dtype=np.float64)
    cx, cy = p.mean(0)
    t = math.copysign(np.hypot(*(p[0] - p[1])) / 2.0, p[0, 1] - p[1, 1])
    w = math.copysign(np.hypot(*(p[2] - p[3])) / 2.0, p[2, 0] - p[3, 0])
    return np.array([[cx, cy + t], [cx, cy - t], [cx + w, cy], [cx - w, cy]])


def project_hc(p):
    """Perpendicular, concentric HC diameters with their lengths kept."""
    p = np.asarray(p, dtype=np.float64)
    cx, cy = p.mean(0)
    ax, ay = (p[0] - p[1]) / 2.0
    bx, by = (p[2] - p[3]) / 2.0
    na, nb = math.hypot(ax, ay), math.hypot(bx, by)
    if na < 1e-9 or nb < 1e-9:
        return np.array([[cx + ax, cy + ay], [cx - ax, cy - ay], [cx + bx, cy + by], [cx - bx, cy - by]])
    alpha, beta = math.atan2(ay, ax), math.atan2(by, bx)
    phi = 0.5 * math.atan2(na * na * math.sin(2 * alpha) - nb * nb * math.sin(2 * beta),
                           na * na * math.cos(2 * alpha) - nb * nb * math.cos(2 * beta))
    u, v = np.array([math.cos(phi), math.sin(phi)]), np.array([-math.sin(phi), math.cos(phi)])
    sa = math.copysign(na, ax * u[0] + ay * u[1])
    sb = math.copysign(nb, bx * v[0] + by * v[1])
    c = np.array([cx, cy])
    return np.stack([c + sa * u, c - sa * u, c + sb * v, c - sb * v])


def project(points, task):
    if task == "FA" and len(points) == 4: return project_fa(points)
    if task == "HC" and len(points) == 4: return project_hc(points)
    return points


def _sample_mre(pred, gt):
    pred, gt = np.asarray(pred, dtype=np.float64), np.asarray(gt, dtype=np.float64)
    n = min(len(pred), len(gt))
    pred, gt = pred[:n], gt[:n]
    visible = (gt[:, 0] >= 0) & (gt[:, 1] >= 0)
    if visible.sum() == 0: return None
    return float(np.hypot(pred[visible, 0] - gt[visible, 0], pred[visible, 1] - gt[visible, 1]).mean())


def challenge_score(entries):
    """entries: [{"task_id", "pred_px": (K, 2), "gt_px": (K, 2)}] in original pixels, missing landmarks at (-1, -1)."""
    by_task = {}
    for e in entries: by_task.setdefault(e["task_id"], []).append(e)
    mre, mae = {}, {}
    for task, es in by_task.items():
        errs = [m for m in (_sample_mre(e["pred_px"], e["gt_px"]) for e in es) if m is not None]
        mre[task] = float(np.mean(errs)) if errs else float("nan")
        names = [n for n, _, _ in MEASUREMENT_SPECS.get(task, [])]
        abs_err = {n: [] for n in names}
        for e in es:
            gm, pm = measurements_from_points(task, e["gt_px"]), measurements_from_points(task, e["pred_px"])
            for n in names:
                if gm[n] is not None and pm[n] is not None: abs_err[n].append(abs(pm[n] - gm[n]))
        valid = [float(np.mean(v)) for v in abs_err.values() if v]
        mae[task] = float(np.mean(valid)) if valid else float("nan")
    tasks = [t for t in TASK_ORDER if t in by_task]
    return {"mre": float(np.nanmean([mre[t] for t in tasks])), "mae": float(np.nanmean([mae[t] for t in tasks])),
            "per_task_mre": mre, "per_task_mae": mae}
