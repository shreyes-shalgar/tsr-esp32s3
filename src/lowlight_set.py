"""
Low-light robustness: can the model still read a sign at night?

Two ways of making a dark image, because they measure different things.

  blend   I' = I * (1 - a)
          A straight alpha blend toward a black image, which is what "blend the
          dataset with a black image" means literally. Every pixel scales by the
          same factor, nothing else changes. This is optimistic: real darkness
          is not a clean multiply.

  camera  gamma, then scale, then shot + read noise, then quantise
          What a sensor actually produces at low exposure. Fewer photons means
          the signal-to-noise ratio collapses, not just the mean level, and the
          noise is what kills a classifier rather than the darkness itself.

Both are generated at the same set of levels so the gap between them is
visible, and both preserve the exact class balance of the source set.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_data import CLASS_IDS

# Paths are resolved relative to the repository root so the scripts run
# from a fresh checkout on any machine.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*parts):
    return os.path.join(_REPO, *parts)



def blend_black(X, a):
    """I * (1 - a). a = 0 is the original, a = 1 is pure black."""
    return np.clip(X.astype(np.float32) * (1.0 - a), 0, 255).astype(np.uint8)


def low_light_camera(X, a, rng, read_noise=2.0, full_well=9000.0):
    """
    A sensor model at the SAME displayed brightness as blend_black, so the only
    difference between the two is the noise and the nonlinearity.

    linearise (gamma 2.2) -> cut the exposure -> Poisson shot noise + Gaussian
    read noise -> re-encode. The exposure scale is (1-a)**2.2 so that the mean
    display level lands on I*(1-a), matching the blend. At a = 0.9 the sensor
    collects about 0.6 % of the photons, and shot noise scales with the square
    root of that - which is what actually destroys the image, not the dimness.
    """
    lin = (X.astype(np.float32) / 255.0) ** 2.2
    scale = max(1e-6, (1.0 - a) ** 2.2)
    photons = np.clip(lin * full_well * scale, 0.0, None)
    noisy = rng.poisson(photons).astype(np.float32)
    noisy += rng.normal(0.0, read_noise, noisy.shape)
    out_lin = np.clip(noisy / full_well, 0.0, 1.0)
    return np.clip((out_lin ** (1.0 / 2.2)) * 255.0, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=_p("out", "data.npz"))
    ap.add_argument("--outdir", default=_p("out", "lowlight"))
    ap.add_argument("--levels", default="0.3,0.5,0.7,0.85,0.95")
    ap.add_argument("--seed", type=int, default=99)
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    d = np.load(a.src)
    X, y = d["Xva"], d["yva"]

    for lv in [float(v) for v in a.levels.split(",")]:
        b = blend_black(X, lv)
        c = low_light_camera(X, lv, rng)
        np.savez_compressed(f"{a.outdir}/blend_{lv:.2f}.npz", Xva=b, yva=y)
        np.savez_compressed(f"{a.outdir}/camera_{lv:.2f}.npz", Xva=c, yva=y)
        print(f"level {lv:.2f}: blend mean {b.mean():6.1f}   camera mean {c.mean():6.1f}"
              f"   ({len(b)} images, class balance preserved)")


if __name__ == "__main__":
    main()
