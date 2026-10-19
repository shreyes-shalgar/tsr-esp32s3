"""
Build a stress set that simulates the one scenario that could really cost us.

Seven of the seventeen classes (69, 70, 72, 83-86) exist in the supplied data
ONLY as 32x32 pre-augmented crops. If the hidden test set presents those classes
as ordinary photographs instead, accuracy on them falls and nothing in the
validation score would have warned us.

This builds a proxy for that: the pictogram classes are composited onto real
photographic backgrounds taken from the other classes' images, then put through
the things a camera does - perspective, blur, JPEG, lighting, noise. The photo
classes get the same camera degradations without the compositing.

It is a proxy, not the hidden set. It estimates exposure; it does not prove
anything about the real test data.
"""
import argparse
import io
import json
import os
import random
import sys

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_data import CLASS_IDS, ID2IDX, load_image, IMG

# Paths are resolved relative to the repository root so the scripts run
# from a fresh checkout on any machine.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*parts):
    return os.path.join(_REPO, *parts)


PICTOGRAM = {69, 70, 72, 83, 84, 85, 86}


def open_rgb(path):
    im = Image.open(path)
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        im = Image.alpha_composite(Image.new("RGBA", im.size, (255, 255, 255, 255)), im)
    return im.convert("RGB")


def perspective(im, rng, amount=0.16):
    """Random mild perspective warp - a sign photographed off-axis."""
    w, h = im.size
    d = amount
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    dst = [(rng.uniform(0, d) * w, rng.uniform(0, d) * h),
           (w - rng.uniform(0, d) * w, rng.uniform(0, d) * h),
           (w - rng.uniform(0, d) * w, h - rng.uniform(0, d) * h),
           (rng.uniform(0, d) * w, h - rng.uniform(0, d) * h)]
    # solve the 8 coefficients of the projective transform
    A, B = [], []
    for (x, y), (u, v) in zip(dst, src):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); B.append(u)
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y]); B.append(v)
    coeffs = np.linalg.solve(np.array(A, dtype=np.float64), np.array(B, dtype=np.float64))
    return im.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BILINEAR)


def camera(im, rng):
    """What a camera does to a scene: focus, exposure, compression, sensor noise."""
    if rng.random() < 0.7:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.4)))
    im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.62, 1.42))
    im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.65, 1.35))
    im = ImageEnhance.Color(im).enhance(rng.uniform(0.6, 1.4))
    if rng.random() < 0.7:
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=int(rng.integers(28, 73)))
        buf.seek(0)
        im = Image.open(buf).convert("RGB")
    a = np.asarray(im).astype(np.int16)
    a += rng.integers(-9, 10, a.shape, dtype=np.int16)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def build(ds_root, out_npz, seed=1234):
    rng = np.random.default_rng(seed)
    pyrng = random.Random(seed)
    paths = json.load(open(_p("out", "data_paths.json")))["val"]

    # backgrounds: patches from the photographic classes, so the context is real
    bg_pool = []
    for p in paths:
        cid = int(os.path.basename(os.path.dirname(p)))
        if cid not in PICTOGRAM:
            bg_pool.append(p)

    X, y = [], []
    for p in paths:
        cid = int(os.path.basename(os.path.dirname(p)))
        im = open_rgb(p)

        if cid in PICTOGRAM:
            # place the pictogram into a photographic scene with margin, the way
            # a real photo of that sign would look
            bg = open_rgb(bg_pool[pyrng.randrange(len(bg_pool))])
            side = rng.integers(96, 168)
            bg = bg.resize((int(side * 1.6), int(side * 1.6)), Image.BILINEAR)
            bg = bg.filter(ImageFilter.GaussianBlur(rng.uniform(1.5, 4.0)))
            sign = im.resize((side, side), Image.BICUBIC)
            sign = perspective(sign, rng, 0.13)
            ox = int((bg.size[0] - side) * rng.uniform(0.15, 0.85))
            oy = int((bg.size[1] - side) * rng.uniform(0.15, 0.85))
            bg.paste(sign, (ox, oy))
            # crop back around the sign, loosely, as a detector would
            m = int(side * rng.uniform(0.04, 0.20))
            box = (max(0, ox - m), max(0, oy - m),
                   min(bg.size[0], ox + side + m), min(bg.size[1], oy + side + m))
            im = bg.crop(box)
        else:
            im = perspective(im, rng, 0.13)

        im = camera(im, rng)
        X.append(np.asarray(im.resize((IMG, IMG), Image.BILINEAR), dtype=np.uint8))
        y.append(ID2IDX[cid])

    X = np.stack(X); y = np.array(y, dtype=np.int32)
    np.savez_compressed(out_npz, Xva=X, yva=y)
    print(f"stress set: {X.shape}, {len(set(y.tolist()))} classes -> {out_npz}")
    return X, y


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default=_p("ds"))
    ap.add_argument("--out", default=_p("out", "stress.npz"))
    a = ap.parse_args()
    build(a.ds, a.out)
