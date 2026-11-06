"""Kev-Bio inference.

    python -m kevbio.predict --run runs/fold0/kevbio --data data/fold0/val.jsonl --layout_train data/fold0/train.jsonl \
        --out runs/fold0/kevbio/predictions.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .backbone import DTYPES
from .model import KMAX, READOUT_TEMP, KevBio, load_records
from .score import challenge_score

CARDIAC = ("A4C", "PLAX", "PSAX", "IVC")


def zoom(image, s):
    """Zoom about the image centre on the same canvas: x -> c + s (x - c)."""
    import cv2
    a = np.asarray(image)
    h, w = a.shape[:2]
    c = np.array([(w - 1) / 2, (h - 1) / 2])
    m = np.array([[s, 0, c[0] * (1 - s)], [0, s, c[1] * (1 - s)]])
    return Image.fromarray(cv2.warpAffine(a, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0))


def unzoom(points, s, size):
    c = np.array([(size[0] - 1) / 2, (size[1] - 1) / 2])
    return c + (np.asarray(points) - c) / s


def find_split(image, gap=0.02, min_share=0.3):
    """x of the dark vertical gap between the two panes of a split-screen frame, else None."""
    g = np.asarray(image.convert("L"), np.float32)
    h, w = g.shape
    occ = np.convolve((g[int(0.12 * h):int(0.85 * h)] > 25).mean(0), np.ones(5) / 5, "same")
    lo, hi = int(0.3 * w), int(0.7 * w)
    x = lo + int(np.argmin(occ[lo:hi]))
    if occ[x] > gap: return None
    share = occ[:x].sum() / max(occ.sum(), 1e-9)
    return x if min_share <= share <= 1 - min_share else None


def keep_pane(image, x, left):
    a = np.asarray(image).copy()
    if left: a[:, x:] = 0
    else: a[:, :x] = 0
    return Image.fromarray(a)


def pane_mass(logits, x, width):
    """[Q, KMAX, H, W] logits -> [Q, KMAX] probability mass left of pixel column x."""
    Wm = logits.shape[-1]
    left = ((torch.arange(Wm, device=logits.device) + 0.5) / Wm * width) < x
    p = logits.float().flatten(-2).softmax(-1).view(logits.shape)
    return p.sum(-2)[..., left].sum(-1)


def fit_layout(records, tasks):
    """Per view, a Gaussian over the normalised landmark layout with Ledoit-Wolf shrinkage."""
    out = {}
    for t in tasks:
        X = np.array([(r["landmarks"] / [r["meta"]["width"], r["meta"]["height"]]).ravel() for r in records
                      if r["task"] == t and (r["landmarks"] >= 0).all()])
        if len(X) < 3: continue
        n, p = X.shape
        Xc = X - X.mean(0)
        S = Xc.T @ Xc / n
        mu = np.trace(S) / p
        delta = ((S - mu * np.eye(p)) ** 2).sum()
        beta = min(((Xc ** 2).sum(1) ** 2).sum() / n ** 2 - (S ** 2).sum() / n, delta)
        out[t] = (X.mean(0), (1 - beta / delta) * S + beta / delta * mu * np.eye(p))
    return out


def layout_bias(model, item, points, shape, beta, s):
    """Log-prior of each landmark given all other landmarks at `points`, with covariance / beta, on the map grid of
    the zoom-s view."""
    W, H = item["size"]
    mean, cov = shape
    x = (points / [W, H]).ravel()
    c = np.array([(W - 1) / 2, (H - 1) / 2])
    Hm, Wm = item["map_hw"]
    gx, gy = np.meshgrid((np.arange(Wm) + 0.5) / Wm * W, (np.arange(Hm) + 0.5) / Hm * H)
    bias = np.zeros((len(item["qids"]), KMAX, Hm, Wm), np.float32)
    for qi, qid in enumerate(item["qids"]):
        for k, li in enumerate(model.specs["by_id"][qid]["points"]):
            i, o = [2 * li, 2 * li + 1], [j for j in range(len(x)) if j // 2 != li]
            A = cov[np.ix_(i, o)] @ np.linalg.inv(cov[np.ix_(o, o)])
            m = (mean[i] + A @ (x[o] - mean[o])) * [W, H]
            V = (cov[np.ix_(i, i)] - A @ cov[np.ix_(o, i)]) * np.outer([W, H], [W, H]) * s * s / beta
            m = c + s * (m - c)
            Vi = np.linalg.inv(V)
            dx, dy = gx - m[0], gy - m[1]
            bias[qi, k] = -0.5 * (Vi[0, 0] * dx * dx + 2 * Vi[0, 1] * dx * dy + Vi[1, 1] * dy * dy) / model.readout_temp
    return torch.from_numpy(bias)


@torch.no_grad()
def predict(model, recs, autocast, tta=(1.0,), pane_tasks=(), layout=None, layout_beta=3.0):
    """-> per record (K, 2) landmarks in original pixels."""
    model.eval()
    amp = lambda: torch.autocast("cuda", dtype=autocast, enabled=autocast != torch.float32)
    out = []
    for rec in recs:
        image = Image.open(rec["image"]).convert("RGB") if tta != (1.0,) or rec["task"] in pane_tasks else None
        split = find_split(image) if rec["task"] in pane_tasks else None
        if split is not None:
            item = model.encode(rec, image=image)
            with amp():
                o = model(item)
            K = torch.tensor([len(model.specs["by_id"][q]["points"]) for q in item["qids"]])
            used = torch.arange(KMAX)[None] < K[:, None]
            left = (pane_mass(o["logits"], split, item["size"][0]).cpu() * used).sum(1) / K >= 0.5
            panes = {side: keep_pane(image, split, side) for side in (True, False) if (left == side).any()}
        views = []
        for s in tta:
            if split is None:
                item = model.encode(rec, image=None if image is None or s == 1.0 else zoom(image, s))
                with amp():
                    o = model(item)
            else:
                outs = {}
                for side, img in panes.items():
                    item = model.encode(rec, image=img if s == 1.0 else zoom(img, s))
                    with amp():
                        outs[side] = model(item)
                o = {"logits": torch.stack([outs[bool(l)]["logits"][qi] for qi, l in enumerate(left.tolist())])}
            views.append((s, item, o["logits"]))
        size = (rec["meta"]["width"], rec["meta"]["height"])
        pred = np.mean([unzoom(model.decode(item, lg), s, size) for s, item, lg in views], 0)
        if layout and rec["task"] in layout:
            reread = []
            for s, item, lg in views:
                item["map_hw"] = tuple(lg.shape[-2:])
                bias = layout_bias(model, item, pred, layout[rec["task"]], layout_beta, s).to(lg.device)
                reread.append(unzoom(model.decode(item, lg.float() + bias), s, size))
            pred = np.mean(reread, 0)
        out.append(pred)
    return out


def score(recs, preds):
    return challenge_score([{"task_id": r["task"], "gt_px": r["landmarks"], "pred_px": p} for r, p in zip(recs, preds)])


def write_predictions(recs, preds, path):
    """Predictions in the challenge submission format."""
    rows = [{"image_path": f"{r['task']}/{r['meta']['image_id']}", "task_id": r["task"],
             "predicted_points_pixels": p.reshape(-1).tolist(),
             "predicted_points_normalized": (p / [r["meta"]["width"], r["meta"]["height"]]).reshape(-1).tolist()}
            for r, p in zip(recs, preds)]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(rows) + "\n")


def main():
    p = argparse.ArgumentParser(description="Kev-Bio inference")
    p.add_argument("--run", required=True, help="trained run directory")
    p.add_argument("--data", required=True, help="records to predict, e.g. data/fold0/val.jsonl")
    p.add_argument("--out", required=True, help="output JSON")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bf16", choices=DTYPES)
    p.add_argument("--tta", default="0.92,1.0,1.08", help="zoom scales averaged at test time (1.0 = off)")
    p.add_argument("--panes", default=",".join(CARDIAC), help="views read one pane at a time on split-screen frames")
    p.add_argument("--layout_train", help="training records for the layout prior (unset = off)")
    p.add_argument("--layout_tasks", default=",".join(CARDIAC))
    p.add_argument("--layout_beta", type=float, default=3.0)
    p.add_argument("--readout_temp", type=float, default=READOUT_TEMP)
    args = p.parse_args()

    model = KevBio.from_run(args.run, args.device)
    model.readout_temp = args.readout_temp
    recs = load_records(args.data)
    layout = fit_layout(load_records(args.layout_train), args.layout_tasks.split(",")) if args.layout_train else None
    t0 = time.time()
    preds = predict(model, recs, DTYPES[args.dtype], tta=tuple(float(x) for x in args.tta.split(",")),
                    pane_tasks=tuple(t for t in args.panes.split(",") if t), layout=layout, layout_beta=args.layout_beta)
    write_predictions(recs, preds, args.out)
    s = score(recs, preds)
    print(f"MRE {s['mre']:.2f} px  MAE {s['mae']:.2f}  ({len(recs)} images, {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
