"""Prepare the FU/GU Biometry 2026 labelled data for Kev-Bio.

    python scripts/prepare_data.py --raw /path/to/challenge_download --out data

"""
import argparse
import ast
import csv
import io
import json
import os
import sys
import zipfile
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from kevbio.score import measurements_from_points  # noqa: E402

# (labelled zip, landmark CSV) per view, as in the organisers' Google Drive folder
LABELED = {
    "A4C": ("A4C/labeled/A4C_labeled.zip", "A4C/labeled/A4C_train.csv"),
    "AOP": ("AOP/labeled/AOP.zip", "AOP/labeled/AOP_train.csv"),
    "FA": ("FA/labeled/FA.zip", "FA/labeled/FA_train_new.csv"),
    "FUGC": ("FUGC/labeled/FUGC.zip", "FUGC/labeled/FUGC.csv"),
    "HC": ("HC/labeled/HC_train.zip", "HC/labeled/HC_train.csv"),
    "IVC": ("IVC/labeled/IVC.zip", "IVC/labeled/IVC_train.csv"),
    "PLAX": ("PLAX/labeled/PLAX.zip", "PLAX/labeled/PLAX_train.csv"),
    "PSAX": ("PSAX/labeled/PSAX.zip", "PSAX/labeled/PSAX_train.csv"),
    "fetal_femur": ("fetal_femur/labeled/fetal_femur.zip", "fetal_femur/labeled/fetal_femur_train.csv"),
}
FLAGGED = {f"Patient{p}_Plane5_{i}.png" for p, i in [
    ("00757", "1_of_1"), ("00863", "1_of_2"), ("00863", "2_of_2"), ("01025", "1_of_1"), ("01035", "2_of_4"),
    ("01035", "4_of_4"), ("01130", "2_of_4"), ("01221", "2_of_2"), ("01246", "1_of_2"), ("01248", "1_of_1"),
    ("01249", "1_of_1"), ("01301", "1_of_2"), ("01301", "2_of_2"), ("01304", "2_of_2"), ("01475", "1_of_1"),
    ("01476", "1_of_1"), ("01477", "1_of_2"), ("01478", "1_of_1"), ("01480", "1_of_1"), ("01481", "1_of_1"),
    ("01605", "2_of_2"), ("01606", "2_of_2"), ("01607", "1_of_2"), ("01608", "1_of_1"), ("01609", "1_of_1")]}


def find(raw, rel):
    """The file at raw/rel, or the only file with the same view and name anywhere below raw."""
    if (raw / rel).exists(): return raw / rel
    task, name = Path(rel).parts[0], Path(rel).name
    hits = [p for p in raw.rglob(name) if task in p.parts and p.parent.name == "labeled"]
    if len(hits) != 1: raise FileNotFoundError(f"{rel} not found under {raw}")
    return hits[0]


def extract(zip_path, dest):
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = os.path.basename(info.filename)
            if info.is_dir() or not name.lower().endswith((".png", ".jpg", ".jpeg")): continue
            if "__MACOSX" in info.filename or name.startswith("._"): continue
            if not (dest / name).exists(): (dest / name).write_bytes(zf.read(info))


def read_landmarks(csv_path, n):
    """image file name -> n (x, y) landmarks, from the CSV's coordinate columns in order."""
    raw = csv_path.read_bytes()
    try: text = raw.decode("utf-8")
    except UnicodeDecodeError: text = raw.decode("gbk")
    rows = list(csv.DictReader(io.StringIO(text)))
    cols = [c for c in rows[0] if c.endswith("_xy")]
    if len(cols) != n: raise ValueError(f"{csv_path}: {len(cols)} landmark columns, expected {n}")
    return {Path(r["image_path"]).name: [list(map(float, ast.literal_eval(r[c]))) for c in cols] for r in rows}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--raw", required=True, help="the downloaded challenge data")
    p.add_argument("--out", default=str(REPO / "data"))
    p.add_argument("--splits", default=str(REPO / "splits" / "kfold_group_v1"))
    args = p.parse_args()
    raw, out, splits = Path(args.raw), Path(args.out), Path(args.splits)
    specs = json.loads((REPO / "kevbio" / "specs.json").read_text())

    records = {}
    for task, (zip_rel, csv_rel) in LABELED.items():
        image_dir = out / "images" / task
        extract(find(raw, zip_rel), image_dir)
        questions = [m for m in specs["measurements"] if m["task"] == task]
        view = specs["views"][task]
        for fname, pts in read_landmarks(find(raw, csv_rel), specs["keypoints"][task]).items():
            if fname in FLAGGED: continue
            with Image.open(image_dir / fname) as im: width, height = im.size
            values = list(measurements_from_points(task, pts).values())
            records[(task, fname)] = {
                "image": f"../images/{task}/{fname}", "task": task,
                "state": f"{view[0].upper() + view[1:]} ({specs['domains'][task]} ultrasound), original size {width} x {height} pixels.",
                "questions": [q["id"] for q in questions], "landmarks": pts,
                "values": {q["id"]: v for q, v in zip(questions, values) if v is not None},
                "meta": {"image_id": fname, "width": width, "height": height}}
        print(f"{task:12s} {sum(t == task for t, _ in records)} labelled images")

    for k in range(5):
        split = json.loads((splits / f"fold_{k}.json").read_text())
        (out / f"fold{k}").mkdir(parents=True, exist_ok=True)
        for name in ("train", "val"):
            recs = [records[tuple(key)] for key in split[name] if tuple(key) in records]
            with open(out / f"fold{k}" / f"{name}.jsonl", "w", encoding="utf-8") as f:
                for r in recs: f.write(json.dumps(r) + "\n")
            print(f"fold {k} {name}: {len(recs)} images")


if __name__ == "__main__":
    main()
