"""Train Kev-Bio on one fold.

    python -m kevbio.train --train data/fold0/train.jsonl --val data/fold0/val.jsonl --out runs/fold0/kevbio
"""
import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .aug import Augment
from .backbone import DTYPES, Scaler, VLMBackbone
from .model import KevBio, load_records, load_specs
from .predict import predict, score


def distributed():
    """(rank, world size, device) under torchrun, else (0, 1, None)."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world == 1: return 0, 1, None
    import torch.distributed as dist
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl")
    return dist.get_rank(), world, f"cuda:{local}"


def balanced_order(records, n, rng):
    """n indices sampled with replacement, every view equally likely."""
    tasks = [r["task"] for r in records]
    count = {t: tasks.count(t) for t in set(tasks)}
    w = np.array([1.0 / count[t] for t in tasks])
    return rng.choice(len(records), size=n, replace=True, p=w / w.sum()).tolist()


def training_item(model, rec, aug):
    image = Image.open(rec["image"]).convert("RGB")
    points = rec["landmarks"]
    if aug is not None:
        arr, points = aug(np.asarray(image), points, rec["task"])
        image = Image.fromarray(arr)
    return model.encode(rec, image=image, points=points)


class EncodedRecords(torch.utils.data.Dataset):
    """Loads, augments and encodes one epoch's records in DataLoader workers."""

    def __init__(self, model, records, order, seed, aug=True):
        self.model, self.records, self.order, self.seed, self.use_aug = model, records, order, seed, aug
        self.aug = None

    def __len__(self):
        return len(self.order)

    def __getitem__(self, i):
        if self.aug is None and self.use_aug:
            info = torch.utils.data.get_worker_info()
            self.aug = Augment(seed=self.seed * 1000 + (info.id if info else 0))
        return training_item(self.model, self.records[self.order[i]], self.aug)


def _as_is(x):
    return x


@torch.no_grad()
def swap(params, ema):
    for p, e in zip(params, ema):
        tmp = p.detach().clone(); p.copy_(e.to(p.dtype)); e.copy_(tmp.float())


def train(args):
    rank, world, ddp_device = distributed()
    if world > 1: import torch.distributed as dist
    torch.manual_seed(args.seed); random.seed(args.seed)
    device, autocast = ddp_device or args.device, DTYPES[args.dtype]
    say = (lambda *a: print(*a, flush=True)) if rank == 0 else (lambda *a: None)
    specs = load_specs()
    backbone = VLMBackbone.from_run(args.init_from, device, trainable=True, max_pixels=args.max_pixels)[0]
    backbone.vlm.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    train_recs = load_records(args.train)
    val_recs = load_records(args.val) if args.val else []
    model = KevBio(backbone, specs, Scaler.fit(train_recs, specs), vision={"usfm_weights": args.usfm_weights})
    for p in model.parameters():                # trainable LoRA layers in fp32
        if p.requires_grad and p.dtype != torch.float32: p.data = p.data.float()

    vision = model.qwen_vision_params + model.usfm_params
    heads = [p for m in model.head_modules().values() if m is not None for p in m.parameters()]
    skip = {id(p) for p in vision + heads}
    lm = [p for p in backbone.parameters() if p.requires_grad and id(p) not in skip]
    dino = model.dino.params()
    params = lm + heads + vision + dino
    if world > 1:
        for p in params: dist.broadcast(p.data, 0)
    opt = torch.optim.AdamW([{"params": heads, "lr": args.head_lr}, {"params": lm, "lr": args.lr},
                             {"params": vision, "lr": args.vision_lr}, {"params": dino, "lr": args.lr}],
                            weight_decay=args.weight_decay)
    say(f"trainable parameters: {sum(p.numel() for p in params):,}; train {len(train_recs)}, val {len(val_recs)}, GPUs {world}")

    per_rank = len(train_recs) // world
    steps = math.ceil(per_rank * args.epochs / args.accum)
    cosine = lambda f: 0.5 * (1 + math.cos(math.pi * min(1.0, f)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup) * cosine(s / steps))
    ema = [p.detach().clone().float() for p in params]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    log, step, start, t0 = [], 0, 0, time.time()
    resume = out / "resume.pt"
    if resume.exists():
        state = torch.load(resume, map_location="cpu", weights_only=False)
        for p, v in zip(params, state["params"]): p.data.copy_(v.to(p.device))
        for e, v in zip(ema, state["ema"]): e.copy_(v.to(e.device))
        opt.load_state_dict(state["opt"]); sched.load_state_dict(state["sched"])
        start, step, log = state["epoch"] + 1, state["step"], state["log"]
        say(f"resumed from {resume} at epoch {start}")

    for epoch in range(start, args.epochs):
        model.train()
        rng = np.random.default_rng(args.seed + epoch)
        order = balanced_order(train_recs, len(train_recs), rng)[rank::world][:per_rank]
        items = torch.utils.data.DataLoader(EncodedRecords(model, train_recs, order, (args.seed * 100 + epoch) * 100 + rank),
                                            batch_size=None, shuffle=False, num_workers=args.workers, collate_fn=_as_is,
                                            prefetch_factor=4 if args.workers else None)
        running, parts_log, chunk = [], {}, []
        for i, item in enumerate(items):
            chunk.append(item)
            last = (i + 1) % args.accum == 0 or i + 1 == len(order)
            if len(chunk) < args.micro and not last: continue
            with torch.autocast("cuda", dtype=autocast, enabled=autocast != torch.float32):
                outs = model.forward_batch(chunk)
            losses = [model.loss(it, o) for it, o in zip(chunk, outs)]
            (sum(loss for loss, _ in losses) / args.accum).backward()
            for loss, parts in losses:
                running.append(loss.item())
                for k, v in parts.items(): parts_log.setdefault(k, []).append(v)
            chunk = []
            if not last: continue
            if world > 1:
                for p in params:
                    if p.grad is None: p.grad = torch.zeros_like(p)
                    dist.all_reduce(p.grad); p.grad /= world
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            with torch.no_grad():
                torch._foreach_lerp_(ema, [p.detach().float() for p in params], 1 - min(args.ema, (1 + step) / (10 + step)))
            if step % 10 == 0:
                n = args.accum * 10
                say(f"epoch {epoch} step {step}/{steps} loss {np.mean(running[-n:]):.4f} "
                    + " ".join(f"{k} {np.mean(v[-n:]):.4f}" for k, v in parts_log.items()) + f" {time.time() - t0:.0f}s")
        entry = {"epoch": epoch, "loss": float(np.mean(running)), **{k: float(np.mean(v)) for k, v in parts_log.items()}}
        swap(params, ema)                       # evaluate and save the EMA weights
        if rank == 0:
            if val_recs:                        # monitoring only; the last epoch is always the result
                s = score(val_recs, predict(model, val_recs, autocast))
                entry["val_mre"], entry["val_mae"] = s["mre"], s["mae"]
            log.append(entry)
            say(json.dumps(entry))
            model.save(out, {"init_from": args.init_from, "training_args": vars(args), "epoch": epoch})
            (out / "training_log.json").write_text(json.dumps(log, indent=2) + "\n")
        swap(params, ema)
        if rank == 0 and epoch + 1 < args.epochs:
            torch.save({"params": [p.detach().cpu() for p in params], "ema": [e.cpu() for e in ema], "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "epoch": epoch, "step": step, "log": log}, resume)
        if world > 1: dist.barrier()
    if rank == 0:
        resume.unlink(missing_ok=True)
        say(f"saved {out}")
    if world > 1: dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser(description="Train Kev-Bio")
    p.add_argument("--train", required=True)
    p.add_argument("--val", help="validation records, for monitoring only")
    p.add_argument("--out", required=True)
    p.add_argument("--init_from", default="jaredpalmer/kev-0.8b@v1.0", help="Kev checkpoint")
    p.add_argument("--usfm_weights", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bf16", choices=DTYPES)
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--accum", type=int, default=8, help="images per optimiser step")
    p.add_argument("--micro", type=int, default=8, help="images per forward pass (memory only; same gradient)")
    p.add_argument("--lr", type=float, default=1e-4, help="language model and DINOv2 LoRA")
    p.add_argument("--vision_lr", type=float, default=5e-5, help="Qwen vision and USFM LoRA")
    p.add_argument("--head_lr", type=float, default=1e-3, help="newly initialised modules")
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--max_pixels", type=int, default=640 * 640)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.usfm_weights is None:
        from .backbone import DEFAULT_USFM
        args.usfm_weights = str(DEFAULT_USFM)
    train(args)


if __name__ == "__main__":
    main()
