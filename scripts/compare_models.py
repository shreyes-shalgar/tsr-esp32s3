"""
Score one or more float Keras models side by side on the honest holdout and on
the two constructed proxies, so a candidate can be rejected on evidence rather
than on a single validation number.

    python src/stress_set.py && python src/lowlight_set.py   # build proxies
    python scripts/compare_models.py model/trained_model.keras candidate.keras

Validation is the only genuinely held-out set. The stress and night sets are
our own constructions; they exist to catch a model that wins on validation by
memorising it, not to select a model on their own.
"""
import sys, os, glob
import numpy as np, tensorflow as tf
S = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SETS = [("validation", f"{S}/out/data.npz", "Xva", "yva"),
        ("stress(photo)", f"{S}/out/stress.npz", "Xva", "yva")]
for lv in ("0.70", "0.85", "0.95"):
    p = f"{S}/out/lowlight/camera_{lv}.npz"
    if os.path.exists(p):
        SETS.append((f"night {lv}", p, "Xva", "yva"))

def score(m, X, y):
    p = []
    for i in range(0, len(X), 256):
        p.append(np.argmax(m(X[i:i+256].astype("float32"), training=False).numpy(), 1))
    return float((np.concatenate(p) == y).mean())

def main(paths):
    ms = [(os.path.basename(p), tf.keras.models.load_model(p, compile=False)) for p in paths]
    hdr = f"{'set':16s}" + "".join(f"{n[:22]:>24s}" for n, _ in ms)
    print(hdr); print("-" * len(hdr))
    for name, f, kx, ky in SETS:
        d = np.load(f)
        X, y = d[kx], d[ky] if ky in d else d["y"]
        row = f"{name:16s}"
        for _, m in ms:
            row += f"{score(m, X, y)*100:>23.2f}%"
        print(row)

if __name__ == "__main__":
    main(sys.argv[1:])
