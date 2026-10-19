"""
AI Arena 2026 - Track 1 : dataset preparation.

Canonical preprocessing. This EXACT function is reused by the host-side UART
sender, so training pixels and inference pixels are bit-identical.

Steps per image:
  1. Open, composite any alpha over WHITE (54 images have real transparency;
     naively dropping the alpha channel leaves black holes).
  2. Convert to RGB (handles 1-bit bilevel images present in classes 69-86).
  3. Resize to 48x48, bilinear.
  4. Emit uint8 HWC.
"""
import os, sys, json
import numpy as np
from PIL import Image

# Paths are resolved relative to the repository root so the scripts run
# from a fresh checkout on any machine.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*parts):
    return os.path.join(_REPO, *parts)


IMG = 48
CLASS_IDS = [0, 2, 3, 4, 5, 6, 7, 24, 43, 69, 70, 72, 83, 84, 85, 86, 87]
IDX2ID = {i: c for i, c in enumerate(CLASS_IDS)}
ID2IDX = {c: i for i, c in enumerate(CLASS_IDS)}


def load_image(path):
    """Canonical preprocessing -> uint8 array (48,48,3)."""
    im = Image.open(path)
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    im = im.convert("RGB")
    im = im.resize((IMG, IMG), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def load_split(root):
    X, y, paths = [], [], []
    for cid in CLASS_IDS:
        d = os.path.join(root, str(cid))
        for f in sorted(os.listdir(d)):
            X.append(load_image(os.path.join(d, f)))
            y.append(ID2IDX[cid])
            paths.append(os.path.join(d, f))
    return np.stack(X), np.array(y, dtype=np.int32), paths


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else _p("ds")
    out = sys.argv[2] if len(sys.argv) > 2 else _p("out", "data.npz")
    Xtr, ytr, ptr = load_split(os.path.join(ds, "Train"))
    Xva, yva, pva = load_split(os.path.join(ds, "Validation"))
    np.savez_compressed(out, Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva)
    with open(out.replace(".npz", "_paths.json"), "w") as fh:
        json.dump({"train": ptr, "val": pva}, fh)
    print(f"train {Xtr.shape} {ytr.shape}  val {Xva.shape} {yva.shape}")
    print("class counts (train):", np.bincount(ytr, minlength=17).tolist())
    print("class counts (val)  :", np.bincount(yva, minlength=17).tolist())
    print("saved ->", out)
