# Photometric and affine augmentation adapted from https://github.com/harshitAgr/fubiometry2026
# (experiments/augment.py, experiments/kp_aug_dataset.py), Copyright 2026 Harshit Agrawal / AIATELLA Oy, Apache-2.0.
import numpy as np

CARDIAC = ("A4C", "PLAX", "PSAX", "IVC")


def _photometric():
    import albumentations as A
    return [
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=False, elementwise=True, p=0.3),
        A.GaussNoise(std_range=(0.02, 0.08), per_channel=False, p=0.2),
        A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(0.3, 0.9), p=0.15),
    ]


def _affine():
    import albumentations as A
    import cv2
    return A.Affine(scale=(0.88, 1.12), rotate=(-12, 12), translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                    shear=0.0, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, fill=0,
                    keep_ratio=True, fit_output=False, rotate_method="largest_box", p=0.5)


def _compose(ops):
    import albumentations as A
    return A.Compose(ops, keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))


def colour_to_grey(image, min_chroma=40):
    """Doppler colour pixels -> their luminance."""
    a = image.astype(np.int16)
    colour = (a.max(-1) - a.min(-1)) > min_chroma
    if not colour.any(): return image
    out = image.copy()
    out[colour] = np.round(a[colour] @ np.array([0.299, 0.587, 0.114])).clip(0, 255).astype(np.uint8)[:, None]
    return out


def landmark_zoom(image, points, rng, margin=0.08):
    """Random aspect-preserving crop that keeps all visible landmarks inside, resized back to the image size."""
    import cv2
    h, w = image.shape[:2]
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    vis = (pts >= 0).all(1)
    if not vis.any(): return image, pts
    size = np.array([w, h], dtype=np.float64)
    lo = np.clip(pts[vis].min(0) - margin * size, 0, size)
    hi = np.clip(pts[vis].max(0) + margin * size, 0, size)
    fmin = max((hi[0] - lo[0]) / w, (hi[1] - lo[1]) / h, 0.5)
    if fmin >= 1: return image, pts
    f = rng.uniform(fmin, 1.0)
    cw, ch = f * w, f * h
    x0 = rng.uniform(max(0.0, hi[0] - cw), min(lo[0], w - cw))
    y0 = rng.uniform(max(0.0, hi[1] - ch), min(lo[1], h - ch))
    m = np.array([[1 / f, 0, -x0 / f], [0, 1 / f, -y0 / f]], dtype=np.float32)
    out = cv2.warpAffine(image, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    new = pts.copy()
    new[vis] = (pts[vis] - [x0, y0]) / f
    return out, new


class Augment:
    """(uint8 H x W x 3 image, landmarks in pixels, task) -> augmented image and landmarks. Missing landmarks (< 0)
    are left unchanged; an affine draw that moves a visible landmark out of the frame is retried up to 10 times."""

    MAX_TRIES = 10

    def __init__(self, seed=None, p_grey=0.5, p_zoom=0.5):
        self.geo = _compose([*_photometric(), _affine()])
        self.fallback = _compose(_photometric())
        if seed is not None:
            self.geo.set_random_seed(seed); self.fallback.set_random_seed(seed + 1)
        self.rng = np.random.default_rng(seed)
        self.p_grey, self.p_zoom = p_grey, p_zoom

    def __call__(self, image, points, task=None):
        if task in CARDIAC:
            if self.rng.random() < self.p_grey: image = colour_to_grey(image)
            if self.rng.random() < self.p_zoom: image, points = landmark_zoom(image, points, self.rng)
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        visible = (pts >= 0).all(1)
        h, w = image.shape[:2]
        for _ in range(self.MAX_TRIES):
            out = self.geo(image=image, keypoints=pts)
            k = np.asarray(out["keypoints"], dtype=np.float32).reshape(-1, 2)
            inside = (k[:, 0] >= 0) & (k[:, 1] >= 0) & (k[:, 0] < w) & (k[:, 1] < h)
            if len(k) == len(pts) and inside[visible].all():
                k[~visible] = pts[~visible]
                return out["image"], k
        out = self.fallback(image=image, keypoints=pts)
        return out["image"], pts
