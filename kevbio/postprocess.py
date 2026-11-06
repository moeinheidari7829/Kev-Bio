"""Shape prior and length check fitted on the training labels of each fold."""
import numpy as np

from .score import MEASUREMENT_SPECS

LENGTH_TASKS = ("A4C", "PLAX", "PSAX", "IVC")
PCT, SHRINK, MAHA_MAX, MAX_ITERS = (5.0, 95.0), 0.2, 4.0, 3


def fit_lengths(train):
    vals = {}
    for r in train:
        if r["task"] not in LENGTH_TASKS: continue
        p = np.asarray(r["landmarks"], float)
        for mi, (_, kind, idx) in enumerate(MEASUREMENT_SPECS[r["task"]]):
            if kind != "dist" or (p[list(idx)] < 0).any(): continue
            vals.setdefault((r["task"], mi), []).append(float(np.linalg.norm(p[idx[0]] - p[idx[1]])))
    return {k: tuple(np.percentile(v, PCT)) for k, v in vals.items()}


def fit_shapes(train):
    X = {}
    for r in train:
        p = np.asarray(r["landmarks"], float)
        if (p < 0).any(): continue
        X.setdefault(r["task"], []).append((p / [r["meta"]["width"], r["meta"]["height"]]).ravel())
    out = {}
    for t, xs in X.items():
        xs = np.asarray(xs)
        S = np.cov(xs, rowvar=False)
        out[t] = (xs.mean(0), (1 - SHRINK) * S + SHRINK * np.diag(np.diag(S)))
    return out


def length_check(task, pts, bands):
    if task not in LENGTH_TASKS: return pts
    pts = pts.copy()
    for mi, (_, kind, (i, j)) in enumerate(MEASUREMENT_SPECS[task]):
        if kind != "dist" or (task, mi) not in bands: continue
        lo, hi = bands[(task, mi)]
        d = pts[i] - pts[j]
        L = float(np.linalg.norm(d))
        if lo <= L <= hi: continue
        target = lo if L < lo else hi
        u = d / L if L > 1e-9 else np.array([0.0, 1.0])
        c = (pts[i] + pts[j]) / 2
        pts[i], pts[j] = c + u * target / 2, c - u * target / 2
    return pts


def _conditional(mu, S, x, k):
    a = [2 * k, 2 * k + 1]
    b = [i for i in range(len(x)) if i not in a]
    Sab, Sbb = S[np.ix_(a, b)], S[np.ix_(b, b)]
    K = np.linalg.solve(Sbb, Sab.T).T
    return mu[a] + K @ (x[b] - mu[b]), S[np.ix_(a, a)] - K @ Sab.T


def shape_prior(task, pts, size, shapes):
    if task not in shapes: return pts
    mu, S = shapes[task]
    x = (pts / size).ravel().copy()
    for _ in range(MAX_ITERS):
        best, arg, mean = -1.0, None, None
        for k in range(len(pts)):
            m, C = _conditional(mu, S, x, k)
            r = x[2 * k:2 * k + 2] - m
            d = float(np.sqrt(r @ np.linalg.solve(C, r)))
            if d > best: best, arg, mean = d, k, m
        if best <= MAHA_MAX: break
        x[2 * arg:2 * arg + 2] = mean
    return x.reshape(len(pts), 2) * size


class PostProcess:
    def __init__(self, train):
        self.bands, self.shapes = fit_lengths(train), fit_shapes(train)

    def __call__(self, task, pts, size):
        return length_check(task, shape_prior(task, pts, size, self.shapes), self.bands)
