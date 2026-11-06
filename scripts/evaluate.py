"""Five-fold evaluation with the challenge metric.

    python scripts/evaluate.py --pred "runs/fold{k}/kevbio/predictions.json"
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kevbio.postprocess import PostProcess  # noqa: E402
from kevbio.score import TASK_ORDER, challenge_score, project  # noqa: E402

VIEW = {"AOP": "AoP", "fetal_femur": "Femur"}


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def score_fold(pred_path, train, val):
    preds = {(p["task_id"], Path(p["image_path"]).name): p for p in json.loads(Path(pred_path).read_text())}
    post = PostProcess(train)
    out = {}
    for variant in ("raw", "post"):
        entries = []
        for r in val:
            pts = np.array(preds[(r["task"], r["meta"]["image_id"])]["predicted_points_pixels"], float).reshape(-1, 2)
            if variant == "post": pts = post(r["task"], pts, np.array([r["meta"]["width"], r["meta"]["height"]], float))
            entries.append({"task_id": r["task"], "gt_px": np.array(r["landmarks"], float), "pred_px": project(pts, r["task"])})
        out[variant] = challenge_score(entries)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="prediction file pattern with {k} for the fold")
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "data"))
    ap.add_argument("--folds", default="0,1,2,3,4")
    args = ap.parse_args()
    folds = [int(f) for f in args.folds.split(",")]
    res = {k: score_fold(args.pred.format(k=k), load(f"{args.data}/fold{k}/train.jsonl"), load(f"{args.data}/fold{k}/val.jsonl"))
           for k in folds}
    names = {"raw": "raw", "post": "post-processed"}
    for v, name in names.items():
        print(f"{name:15s} " + "  ".join(f"fold {k}: {res[k][v]['mre']:.2f} / {res[k][v]['mae']:.2f}" for k in folds))
    print(f"\n{'':15s} {'MRE (px)':>16s} {'MAE':>16s}")
    for v, name in names.items():
        mre, mae = [res[k][v]["mre"] for k in folds], [res[k][v]["mae"] for k in folds]
        print(f"{name:15s} {np.mean(mre):9.2f} ± {np.std(mre):4.2f} {np.mean(mae):9.2f} ± {np.std(mae):4.2f}")
    print("\nper view (post-processed, mean over folds), MRE / MAE")
    for t in TASK_ORDER:
        m = np.mean([res[k]["post"]["per_task_mre"][t] for k in folds])
        a = np.mean([res[k]["post"]["per_task_mae"][t] for k in folds])
        print(f"  {VIEW.get(t, t):6s} {m:7.2f} / {a:6.2f}")


if __name__ == "__main__":
    main()
