"""Kev-Bio model."""
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .backbone import (DEFAULT_USFM, QWEN_VISION_LORA, SPECIAL, VIT_LORA, Scaler, USFMEncoder, VLMBackbone, inject_lora,
                       lora_state, user_tokens)

KMAX = 4                       # most landmarks of any measurement (FA / HC ellipses)
COORD_SCALE = 4.0              # auxiliary regression target: (x / W - 0.5) * COORD_SCALE
READOUT_TEMP = 2.0             # inference read-out: soft-argmax(softmax(READOUT_TEMP * logits))
LAMBDA_HM, LAMBDA_GEOM, LAMBDA_AUX, SIGMA = 0.1, 0.2, 1.0, 2.0       # loss weights; heatmap target sigma in map cells
SPECS = Path(__file__).resolve().parent / "specs.json"


def load_specs(path=SPECS):
    table = json.loads(Path(path).read_text())
    table["by_id"] = {m["id"]: m for m in table["measurements"]}
    return table


def load_records(path):
    """Records written by scripts/prepare_data.py; image paths are relative to the JSONL file."""
    root, out = Path(path).parent, []
    for line in open(path, encoding="utf-8"):
        if not line.strip(): continue
        r = json.loads(line)
        r["image"] = (root / r["image"]).resolve()
        r["landmarks"] = np.asarray(r["landmarks"], dtype=np.float64)
        out.append(r)
    return out


def measure(kind, p):
    """Differentiable clinical measurement from [K, 2] landmarks in pixels."""
    if kind == "dist": return (p[0] - p[1]).norm()
    if kind == "pi_half_sum": return math.pi * ((p[0] - p[1]).norm() + (p[2] - p[3]).norm()) / 2
    if kind == "min_dist": return torch.minimum((p[0] - p[1]).norm(), (p[2] - p[3]).norm())
    if kind == "angle":
        v1, v2 = p[1] - p[0], p[2] - p[0]
        cos = (v1 * v2).sum() / (v1.norm() * v2.norm() + 1e-9)
        return torch.rad2deg(torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6)))
    raise ValueError(kind)


def soft_argmax(logits):
    """[..., H, W] -> (x, y) in [0, 1], cell centres at (j + 0.5) / W."""
    H, W = logits.shape[-2:]
    p = logits.flatten(-2).float().softmax(-1).view(logits.shape)
    xs = (torch.arange(W, device=logits.device, dtype=torch.float32) + 0.5) / W
    ys = (torch.arange(H, device=logits.device, dtype=torch.float32) + 0.5) / H
    return torch.stack([(p.sum(-2) * xs).sum(-1), (p.sum(-1) * ys).sum(-1)], -1)


# ---------------------------------------------------------------------- modules

def _gn(ch, groups=8):
    return nn.GroupNorm(min(groups, ch), ch)


class ResBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv1, self.n1 = nn.Conv2d(cin, cout, 3, padding=1), _gn(cout)
        self.conv2, self.n2 = nn.Conv2d(cout, cout, 3, padding=1), _gn(cout)
        self.skip = nn.Identity() if cin == cout else nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), _gn(cout))
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.n2(self.conv2(self.act(self.n1(self.conv1(x))))) + self.skip(x))


class ChannelNorm(nn.LayerNorm):
    def forward(self, x):
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class PixelNeck(nn.Module):
    """Patch grids + a conv stem on the image -> L2-normalised pixel map F at 4x the grid resolution."""

    def __init__(self, in_dims, mid=256, emb=128):
        super().__init__()
        self.norms = nn.ModuleList(ChannelNorm(d) for d in in_dims)
        self.inp = nn.Conv2d(sum(in_dims), mid, 1)
        up = mid // 2
        self.up = nn.Sequential(nn.ConvTranspose2d(mid, up, 4, 2, 1), _gn(up), nn.SiLU(inplace=True),
                                nn.ConvTranspose2d(up, up, 4, 2, 1), _gn(up), nn.SiLU(inplace=True))
        self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1), _gn(32), nn.SiLU(inplace=True),
                                  nn.Conv2d(32, 64, 3, 2, 1), _gn(64), nn.SiLU(inplace=True))
        self.body = nn.Sequential(ResBlock(up + 64, up), nn.Conv2d(up, up, 3, padding=1), _gn(up),
                                  nn.SiLU(inplace=True), nn.Conv2d(up, emb, 1))

    def forward(self, grids, image):            # [C_i, h, w] each, [3, 16h, 16w] -> [emb, 4h, 4w]
        x = torch.cat([n(g[None]) for n, g in zip(self.norms, grids)], 1)
        x = torch.cat([self.up(self.inp(x)), self.stem(image[None])], 1)
        return F.normalize(self.body(x)[0], dim=0)


class PixelPointer(nn.Module):
    """Decision state h_m -> landmark queries q_{m,k} -> logits q^T F / tau over the pixel map."""

    def __init__(self, d, n_meas, emb=128, tau=0.05):
        super().__init__()
        self.emb = emb
        self.mlp = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, KMAX * emb))
        self.endpoint = nn.Parameter(torch.randn(n_meas, KMAX, emb) * 0.02)
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau)))

    def forward(self, h, columns, fmap):        # [Q, d], [Q], [emb, H, W] -> [Q, KMAX, H, W]
        q = F.normalize(self.mlp(h).view(len(h), KMAX, self.emb) + self.endpoint[columns], dim=-1)
        return torch.einsum("qkc,chw->qkhw", q, fmap) / self.log_tau.exp().clamp(min=0.01)


class RegressionHead(nn.Module):
    """Auxiliary head: decision state -> KMAX landmark coordinates and the standardised measurement value."""

    def __init__(self, d, n_meas, hidden=256):
        super().__init__()
        self.out = 2 * KMAX + 1
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, n_meas * self.out))

    def forward(self, h, columns):
        return self.net(h).view(len(h), -1, self.out)[torch.arange(len(h), device=h.device), columns]


class DinoDense(nn.Module):
    """DINOv2 with LoRA -> patch grids of three blocks, on the image resized to about `pixels` pixels."""

    MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self, name, layers, pixels, lora, checkpointing=True):
        super().__init__()
        import timm
        self.vit = timm.create_model(name, pretrained=True, num_classes=0, img_size=518, dynamic_img_size=True)
        self.patch = self.vit.patch_embed.patch_size[0]
        depth = len(self.vit.blocks)
        self.layers = sorted([depth // 2 - 1, 3 * depth // 4 - 1, depth - 1] if layers == "auto" else layers)
        self.pixels, self.checkpointing = pixels, checkpointing
        for p in self.vit.parameters(): p.requires_grad_(False)
        inject_lora(self.vit, lora, VIT_LORA, "dino")

    def params(self):
        return [p for p in self.vit.parameters() if p.requires_grad]

    def preprocess(self, image):
        W, H = image.size
        n = self.pixels / self.patch ** 2
        gw = max(1, round(math.sqrt(n * W / H))); gh = max(1, round(n / gw))
        x = np.asarray(image.convert("RGB").resize((self.patch * gw, self.patch * gh), Image.BICUBIC), np.float32) / 255.0
        return torch.from_numpy((x - self.MEAN) / self.STD).permute(2, 0, 1)[None]

    def forward(self, pixels):                  # [B, 3, P h, P w] -> [[B, C, h, w] per layer]
        v = self.vit
        h, w = pixels.shape[-2] // self.patch, pixels.shape[-1] // self.patch
        t = v.norm_pre(v.patch_drop(v._pos_embed(v.patch_embed(pixels))))
        out = []
        for i, blk in enumerate(v.blocks[: self.layers[-1] + 1]):
            ckpt = self.checkpointing and self.training and torch.is_grad_enabled()
            t = torch.utils.checkpoint.checkpoint(blk, t, use_reentrant=False) if ckpt else blk(t)
            if i in self.layers: out.append(v.norm(t)[:, v.num_prefix_tokens:].transpose(1, 2).reshape(len(t), -1, h, w))
        return out


def unmerge(tokens, h, w, merge=2):
    """Qwen's patch order (2 x 2 merge windows) -> [C, h, w]."""
    return tokens.view(h // merge, w // merge, merge, merge, -1).permute(4, 0, 2, 1, 3).reshape(-1, h, w)


def unpatchify(pixel_values, h, w, patch=16, temporal=2, merge=2):
    """Qwen's flattened pixel patches -> the resized image [3, h * patch, w * patch]."""
    x = pixel_values.view(-1, 3, temporal, patch, patch)[:, :, 0]
    x = x.view(h // merge, w // merge, merge, merge, 3, patch, patch)
    return x.permute(4, 0, 2, 5, 1, 3, 6).reshape(3, h * patch, w * patch)


# ---------------------------------------------------------------------- model

DEFAULT_CFG = {"vis_layers": "auto", "neck_mid": 256, "emb": 128, "tau": 0.05,
               "dino_model": "vit_large_patch14_dinov2.lvd142m", "dino_layers": "auto", "dino_pixels": 644 * 644,
               "dino_lora": 32}
DEFAULT_VISION = {"qwen_vision_lora": 16, "usfm_lora": 16, "usfm_weights": str(DEFAULT_USFM)}


class KevBio(nn.Module):
    def __init__(self, backbone, specs, scaler, cfg=None, vision=None):
        super().__init__()
        self.backbone, self.specs, self.scaler = backbone, specs, scaler
        self.cfg = {k: v for k, v in {**DEFAULT_CFG, **(cfg or {})}.items() if k in DEFAULT_CFG}
        self.vision = {k: v for k, v in {**DEFAULT_VISION, **(vision or {})}.items() if k in DEFAULT_VISION}
        self.readout_temp = READOUT_TEMP
        c, v, dev = self.cfg, self.vision, backbone.device
        self.measure_ids = [m["id"] for m in specs["measurements"]]
        self.column = {qid: i for i, qid in enumerate(self.measure_ids)}
        d = backbone.vlm.config.text_config.hidden_size
        self.state_id, self.q_id, self.decide_id = (backbone.tok.convert_tokens_to_ids(SPECIAL[i]) for i in (0, 1, 4))
        visual = backbone.vlm.visual
        self.qwen_vision_params = inject_lora(visual, v["qwen_vision_lora"], QWEN_VISION_LORA, "vision")
        self.usfm = USFMEncoder(v["usfm_weights"]).to(dev)
        self.usfm_proj = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, d), nn.GELU(), nn.Linear(d, d)).to(dev)
        self.usfm_params = inject_lora(self.usfm.vit, v["usfm_lora"], VIT_LORA, "usfm")
        self.reg_head = RegressionHead(d, len(self.measure_ids)).to(dev)
        if c["vis_layers"] == "auto":           # blocks 3, 7, 11 of 12
            depth = len(visual.blocks)
            c["vis_layers"] = [depth // 3 - 1, 2 * depth // 3 - 1, depth - 1]
        self._vis = {}
        for i in c["vis_layers"]:
            visual.blocks[i].register_forward_hook(lambda mod, inp, out, i=i: self._vis.__setitem__(i, out))
        self.dino = DinoDense(c["dino_model"], c["dino_layers"], c["dino_pixels"], c["dino_lora"]).to(dev)
        c["dino_layers"] = self.dino.layers
        dims = [visual.config.hidden_size] * len(c["vis_layers"]) + [768] + [self.dino.vit.embed_dim] * len(self.dino.layers)
        self.neck = PixelNeck(dims, c["neck_mid"], c["emb"]).to(dev)
        self.pointer = PixelPointer(d, len(self.measure_ids), c["emb"], c["tau"]).to(dev)

    @property
    def device(self):
        return self.backbone.device

    def head_modules(self):
        return {"reg_head": self.reg_head, "pointer": self.pointer, "neck": self.neck, "usfm_proj": self.usfm_proj}

    def encode(self, rec, image=None, points=None):
        """Token ids and pixel tensors of one record; `image` and `points` override the record's (augmentation)."""
        b = self.backbone
        if image is None: image = Image.open(rec["image"]).convert("RGB")
        points = rec["landmarks"] if points is None else points
        vis = b.image_processor(images=[image], return_tensors="pt")
        pixels, grid = vis["pixel_values"], vis["image_grid_thw"]
        n = USFMEncoder.GRID
        vision = ([b.vision_start] + [b.image_token_id] * (int(grid.prod()) // b.merge ** 2) + [b.vision_end]
                  + [b.vision_start] + [b.image_token_id] * (n * n) + [b.vision_end])
        grids = torch.cat([grid, torch.tensor([[1, n * b.merge, n * b.merge]])])
        ids = [self.state_id] + vision + user_tokens(b.tok, rec["state"])
        decides = []
        for qid in rec["questions"]:
            ids += [self.q_id] + user_tokens(b.tok, self.specs["by_id"][qid]["instructions"]) + [self.decide_id]
            decides.append(len(ids) - 1)
        return {"ids": ids, "pixels": pixels, "grid": grid, "grids": grids, "qids": list(rec["questions"]),
                "decides": decides, "usfm_pixels": self.usfm.preprocess(image),
                "dino_pixels": self.dino.preprocess(image),
                "size": (float(rec["meta"]["width"]), float(rec["meta"]["height"])),
                "points": np.asarray(points, dtype=np.float32), "rec": rec}

    def forward(self, item):
        return self.forward_batch([item])[0]

    def forward_batch(self, items):
        """-> per item {"reg": [Q, 2 KMAX + 1], "logits": [Q, KMAX, Hp, Wp]}. Sequences are right-padded."""
        b, c, dev = self.backbone, self.cfg, self.device
        L = max(len(it["ids"]) for it in items)
        ids = torch.tensor([it["ids"] + [0] * (L - len(it["ids"])) for it in items], device=dev)
        att = torch.tensor([[1] * len(it["ids"]) + [0] * (L - len(it["ids"])) for it in items], device=dev)
        columns = [torch.tensor([self.column[q] for q in it["qids"]], device=dev) for it in items]
        self._vis.clear()
        grid = torch.cat([it["grid"] for it in items])
        qwen = b.vlm.get_image_features(torch.cat([it["pixels"] for it in items]).to(dev), grid.to(dev))
        vis = {i: self._vis[i].split(grid.prod(-1).tolist()) for i in c["vis_layers"]}
        usfm_tokens = self.usfm(torch.cat([it["usfm_pixels"] for it in items]).to(dev))
        embeds = b.vlm.get_input_embeddings()(ids)
        image = torch.cat([g.to(embeds.dtype) for f, t in zip(qwen.pooler_output, usfm_tokens) for g in (f, self.usfm_proj(t))])
        embeds = embeds.masked_scatter((ids == b.image_token_id)[..., None], image)
        position_ids, _ = b.vlm.get_rope_index(ids, mm_token_type_ids=(ids == b.image_token_id).int(),
                                               image_grid_thw=torch.cat([it["grids"] for it in items]).to(dev),
                                               attention_mask=att)
        hs = b.vlm.language_model(inputs_embeds=embeds, position_ids=position_ids, attention_mask=att).last_hidden_state
        h = [hs[i, torch.tensor(it["decides"], device=dev)] for i, it in enumerate(items)]
        dense = [None] * len(items)
        shapes = {}
        for i, it in enumerate(items): shapes.setdefault(tuple(it["dino_pixels"].shape), []).append(i)
        for idx in shapes.values():
            out = self.dino(torch.cat([items[i]["dino_pixels"] for i in idx]).to(dev))
            for j, i in enumerate(idx): dense[i] = [g[j] for g in out]
        outs = []
        for i, it in enumerate(items):
            _, gh, gw = it["grid"][0].tolist()
            th, tw = dense[i][0].shape[-2:]                         # the map is built on DINOv2's patch grid
            size = lambda g: g if g.shape[-2:] == (th, tw) else F.interpolate(g[None].float(), size=(th, tw), mode="bilinear", align_corners=False)[0]
            grids = [size(unmerge(vis[l][i], gh, gw, b.merge)) for l in c["vis_layers"]]
            grids.append(size(usfm_tokens[i].t().reshape(-1, USFMEncoder.GRID, USFMEncoder.GRID)))
            grids += [size(g) for g in dense[i]]
            image = unpatchify(it["pixels"].to(dev), gh, gw).float()
            if (th, tw) != (gh, gw):
                image = F.interpolate(image[None], size=(16 * th, 16 * tw), mode="bilinear", align_corners=False)[0]
            fmap = self.neck([g.to(torch.float32) for g in grids], image)
            hi = h[i].float()
            outs.append({"reg": self.reg_head(hi, columns[i]), "logits": self.pointer(hi, columns[i], fmap)})
        return outs

    def targets(self, item):
        """Per question: normalised landmarks [Q, KMAX, 2], visibility mask, measurement value (nan if unknown)."""
        W, H = item["size"]
        pts = torch.as_tensor(item["points"])
        Q = len(item["qids"])
        xy, mask, value = torch.zeros(Q, KMAX, 2), torch.zeros(Q, KMAX, dtype=torch.bool), torch.full((Q,), float("nan"))
        for qi, qid in enumerate(item["qids"]):
            m = self.specs["by_id"][qid]
            p = pts[m["points"]]
            vis = (p >= 0).all(-1)
            K = len(m["points"])
            xy[qi, :K] = p / torch.tensor([W, H])
            mask[qi, :K] = vis
            if vis.all(): value[qi] = measure(m["kind"], p.double()).float()
        return xy, mask, value

    def loss(self, item, out):
        """L = L_ptr + lambda_hm L_hm + lambda_geom L_geom + lambda_aux L_aux."""
        dev = self.device
        W, H = item["size"]
        diag = math.hypot(W, H)
        xy, mask, value = (t.to(dev) for t in self.targets(item))
        wh = torch.tensor([W, H], device=dev)
        parts = {}
        reg = out["reg"].float()
        geo = reg[:, :2 * KMAX].view(-1, KMAX, 2)
        parts["aux_xy"] = F.huber_loss(geo[mask], ((xy - 0.5) * COORD_SCALE)[mask], delta=1.0)
        has = ~value.isnan()
        if has.any():
            z = torch.tensor([self.scaler.forward(q, float(v), item["rec"]) if not math.isnan(v) else 0.0
                              for q, v in zip(item["qids"], value.tolist())], device=dev)
            parts["aux_value"] = F.huber_loss(reg[has, -1], z[has], delta=1.0)
        logits = out["logits"]
        ptr = soft_argmax(logits)
        parts["ptr"] = (((ptr - xy).abs() * wh).sum(-1) / diag)[mask].mean()
        Hm, Wm = logits.shape[-2:]
        gx, gy = xy[..., 0] * Wm - 0.5, xy[..., 1] * Hm - 0.5     # cross-entropy to a Gaussian at the true landmark
        xs = torch.arange(Wm, device=dev, dtype=torch.float32)
        ys = torch.arange(Hm, device=dev, dtype=torch.float32)
        s2 = 2 * SIGMA ** 2
        tgt = torch.exp(-(xs.view(1, 1, 1, -1) - gx[..., None, None]) ** 2 / s2
                        - (ys.view(1, 1, -1, 1) - gy[..., None, None]) ** 2 / s2).flatten(-2)
        tgt = tgt / tgt.sum(-1, keepdim=True).clamp(min=1e-12)
        parts["hm"] = (-(tgt * logits.flatten(-2).float().log_softmax(-1)).sum(-1))[mask].mean()
        if has.any():                           # measurement from the pointer landmarks
            errs = []
            for qi, qid in enumerate(item["qids"]):
                if not has[qi]: continue
                m = self.specs["by_id"][qid]
                errs.append((measure(m["kind"], ptr[qi, :len(m["points"])] * wh) - value[qi]).abs()
                            / self.scaler.stats[qid]["train_median"])
            parts["geom"] = torch.stack(errs).mean()
        total = (LAMBDA_AUX * (parts["aux_xy"] + parts.get("aux_value", 0.0)) + parts["ptr"]
                 + LAMBDA_HM * parts["hm"] + LAMBDA_GEOM * parts.get("geom", 0.0))
        return total, {k: float(v.detach()) for k, v in parts.items()}

    def decode(self, item, logits):
        """-> (K, 2) landmarks in pixels; a landmark shared by two questions is averaged."""
        W, H = item["size"]
        K = len(item["rec"]["landmarks"])
        xy = soft_argmax(logits.float() * self.readout_temp).cpu()
        acc, cnt = np.zeros((K, 2)), np.zeros(K)
        for qi, qid in enumerate(item["qids"]):
            for k, idx in enumerate(self.specs["by_id"][qid]["points"]):
                acc[idx] += xy[qi, k].numpy() * [W, H]; cnt[idx] += 1
        return acc / np.maximum(cnt, 1)[:, None]

    def save(self, out, extra=None):
        out = Path(out); out.mkdir(parents=True, exist_ok=True)
        b = self.backbone
        b.peft.save_pretrained(out)
        specs = {k: v for k, v in self.specs.items() if k != "by_id"}
        torch.save({"base": b.base, "base_revision": b.revision, "lora": b.peft.peft_config["default"].r, "task": "biometry",
                    "multimodal": {"max_pixels": b.max_pixels}, "vision": self.vision, "biometry": self.cfg,
                    "specs": specs, "scaler": self.scaler.stats, **(extra or {})}, out / "head.pt")
        torch.save({"qwen_vision_lora": lora_state(b.vlm.visual), "usfm_lora": lora_state(self.usfm.vit),
                    "usfm_proj": self.usfm_proj.state_dict()}, out / "vision_adapters.pt")
        torch.save({k: m.state_dict() for k, m in self.head_modules().items() if k != "usfm_proj"},
                   out / "biometry_heads.pt")
        torch.save({"mode": "lora", "state": lora_state(self.dino.vit)}, out / "dino.pt")

    @classmethod
    def from_run(cls, run, device, weights_dtype="fp32"):
        backbone, meta = VLMBackbone.from_run(run, device, weights_dtype=weights_dtype)
        specs = dict(meta["specs"]); specs["by_id"] = {m["id"]: m for m in specs["measurements"]}
        vision = dict(meta["vision"])
        if not Path(vision.get("usfm_weights", "")).exists(): vision["usfm_weights"] = str(DEFAULT_USFM)
        m = cls(backbone, specs, Scaler(meta["scaler"]), meta["biometry"], vision)
        saved = torch.load(Path(run) / "vision_adapters.pt", map_location="cpu")
        for module, key in ((backbone.vlm.visual, "qwen_vision_lora"), (m.usfm.vit, "usfm_lora")):
            missing = set(saved[key]) - set(module.state_dict())
            if missing: raise ValueError(f"{key}: {len(missing)} saved LoRA tensors have no layer")
            module.load_state_dict(saved[key], strict=False)
        m.usfm_proj.load_state_dict(saved["usfm_proj"])
        m.dino.vit.load_state_dict(torch.load(Path(run) / "dino.pt", map_location="cpu")["state"], strict=False)
        heads = torch.load(Path(run) / "biometry_heads.pt", map_location="cpu")
        for k, module in m.head_modules().items():
            if k in heads: module.load_state_dict(heads[k])
        return m
