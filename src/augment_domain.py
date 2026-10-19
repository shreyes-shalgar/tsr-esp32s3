"""
Expand the training set so the seven pictogram classes are also seen as
photographs.

Classes 69, 70, 72 and 83-86 are supplied only as 32x32 pre-augmented crops with
no background. Measured on a held-out stress set, the model trained on that data
alone scores 59 % on those classes once they appear as ordinary photographs,
against 97 % on the classes that were photographs to begin with. This closes
that gap by synthesising the missing view.

Deliberately NOT a copy of the stress-set generator: the parameter ranges
differ, and backgrounds are drawn only from TRAINING images, so the evaluation
is not simply measuring whether the model memorised one transformation.
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
from prepare_data import CLASS_IDS, ID2IDX, IMG

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


def perspective(im, rng, amount):
    w, h = im.size
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    dst = [(rng.uniform(0, amount) * w, rng.uniform(0, amount) * h),
           (w - rng.uniform(0, amount) * w, rng.uniform(0, amount) * h),
           (w - rng.uniform(0, amount) * w, h - rng.uniform(0, amount) * h),
           (rng.uniform(0, amount) * w, h - rng.uniform(0, amount) * h)]
    A, B = [], []
    for (x, y), (u, v) in zip(dst, src):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); B.append(u)
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y]); B.append(v)
    c = np.linalg.solve(np.array(A, dtype=np.float64), np.array(B, dtype=np.float64))
    return im.transform((w, h), Image.PERSPECTIVE, c, Image.BILINEAR)


def camera(im, rng):
    """Wider, coarser ranges than the stress generator uses, on purpose."""
    if rng.random() < 0.75:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 1.9)))
    im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.55, 1.55))
    im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.55, 1.5))
    im = ImageEnhance.Color(im).enhance(rng.uniform(0.45, 1.55))
    if rng.random() < 0.6:
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=int(rng.integers(22, 85)))
        buf.seek(0)
        im = Image.open(buf).convert("RGB")
    a = np.asarray(im).astype(np.int16)
    a += rng.integers(-12, 13, a.shape, dtype=np.int16)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def synthesise(path, bg_paths, rng, pyrng):
    sign = open_rgb(path)
    bg = open_rgb(bg_paths[pyrng.randrange(len(bg_paths))])
    side = int(rng.integers(80, 190))
    bg = bg.resize((int(side * rng.uniform(1.35, 2.0)),
                    int(side * rng.uniform(1.35, 2.0))), Image.BILINEAR)
    bg = bg.filter(ImageFilter.GaussianBlur(rng.uniform(1.0, 5.0)))
    s = sign.resize((side, side), Image.BICUBIC)
    s = perspective(s, rng, rng.uniform(0.04, 0.20))
    ox = int(max(0, bg.size[0] - side) * rng.uniform(0.1, 0.9))
    oy = int(max(0, bg.size[1] - side) * rng.uniform(0.1, 0.9))
    bg.paste(s, (ox, oy))
    m = int(side * rng.uniform(0.0, 0.26))
    box = (max(0, ox - m), max(0, oy - m),
           min(bg.size[0], ox + side + m), min(bg.size[1], oy + side + m))
    return camera(bg.crop(box), rng)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", type=int, default=5,
                    help="synthetic photographic views per pictogram training image")
    ap.add_argument("--data", default=_p("out", "data.npz"))
    ap.add_argument("--paths", default=_p("out", "data_paths.json"))
    ap.add_argument("--out", default=_p("out", "data_domain.npz"))
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    pyrng = random.Random(a.seed)
    d = np.load(a.data)
    Xtr, ytr, Xva, yva = d["Xtr"], d["ytr"], d["Xva"], d["yva"]
    train_paths = json.load(open(a.paths))["train"]

    # backgrounds come from TRAINING photographs only - never validation
    bg = [p for p in train_paths
          if int(os.path.basename(os.path.dirname(p))) not in PICTOGRAM]
    pict = [p for p in train_paths
            if int(os.path.basename(os.path.dirname(p))) in PICTOGRAM]
    print(f"{len(pict)} pictogram training images, {len(bg)} backgrounds available")

    newX, newy = [], []
    for p in pict:
        cid = int(os.path.basename(os.path.dirname(p)))
        for _ in range(a.variants):
            im = synthesise(p, bg, rng, pyrng)
            newX.append(np.asarray(im.resize((IMG, IMG), Image.BILINEAR), dtype=np.uint8))
            newy.append(ID2IDX[cid])

    Xtr2 = np.concatenate([Xtr, np.stack(newX)], axis=0)
    ytr2 = np.concatenate([ytr, np.array(newy, dtype=np.int32)], axis=0)
    np.savez_compressed(a.out, Xtr=Xtr2, ytr=ytr2, Xva=Xva, yva=yva)
    print(f"training set {len(Xtr)} -> {len(Xtr2)} images  ({len(newX)} synthesised)")
    print("class counts:", np.bincount(ytr2, minlength=17).tolist())
    print("written to", a.out)


if __name__ == "__main__":
    main()
