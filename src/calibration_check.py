"""
Should the reported confidence be temperature-scaled?

The organisers have clarified that the reported confidence IS the accuracy
criterion ("with how much confidence does your model predict the given image"),
so it is worth checking whether post-hoc temperature scaling would raise it.

Conclusion is printed at the end.
"""
import os
import numpy as np
from scipy.optimize import minimize_scalar

# Paths are resolved relative to the repository root so the scripts run
# from a fresh checkout on any machine.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*parts):
    return os.path.join(_REPO, *parts)


S, Z = 0.2889368236, 42
rows = [l.split() for l in open(_p("test", "c_out.txt"))]
sc = np.array([[int(v) for v in r[2:]] for r in rows], dtype=np.float64)
pred = np.array([int(r[0]) for r in rows])
y = np.load(_p("out", "data.npz"))['yva']
logits = (sc - Z) * S
corr = (pred == y).astype(float)


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def stats(T):
    p = softmax(logits / T)
    conf = p.max(1)
    ece = 0.0
    for lo in np.arange(0.0, 1.0, 0.1):
        m = (conf > lo) & (conf <= lo + 0.1)
        if m.sum():
            ece += m.mean() * abs(corr[m].mean() - conf[m].mean())
    nll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean()
    return conf.mean(), ece, nll, conf


T_opt = minimize_scalar(lambda t: stats(abs(t) + 1e-3)[2],
                        bounds=(0.05, 10.0), method='bounded').x

print("=== Would temperature scaling help? ===")
print(f"{'T':>6}  {'mean conf':>10}  {'ECE':>8}  {'NLL':>8}   note")
for T, note in [(1.0, "as shipped"), (T_opt, "NLL-optimal"), (0.8, ""), (0.5, "")]:
    mc, ece, nll, _ = stats(T)
    print(f"{T:6.3f}  {mc*100:9.2f}%  {ece*100:7.3f}%  {nll:8.4f}   {note}")

mc1, ece1, _, c1 = stats(1.0)
mcT, eceT, _, cT = stats(T_opt)
print(f"\naccuracy = {corr.mean()*100:.2f}%")
print(f"mean-confidence gain from scaling : {(mcT-mc1)*100:+.2f} pp")
print(f"ECE change                        : {(eceT-ece1)*100:+.3f} pp")
print(f"confidence on the 2 errors, T=1   : {[f'{v*100:.1f}%' for v in c1[corr==0]]}")
print(f"confidence on the 2 errors, T=opt : {[f'{v*100:.1f}%' for v in cT[corr==0]]}")
print("\nVERDICT: the temperature would be fitted on the same 364 images already")
print("used for checkpoint selection, with no clean holdout left to validate it.")
print("Weigh that against the measured gain above before adopting it.")
