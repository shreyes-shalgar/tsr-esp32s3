"""
Bit-exact NumPy re-implementation of the TFLite int8 reference kernels.

This mirrors gemmlowp / tflite::reference_ops arithmetic exactly so that the C
firmware can be validated against it, and it against the TFLite interpreter.
"""
import numpy as np

INT32_MIN, INT32_MAX = -(2**31), 2**31 - 1


def quantize_multiplier(m):
    """TFLite QuantizeMultiplier: real multiplier -> (int32 M0, shift)."""
    if m == 0.0:
        return 0, 0
    sig, exp = np.frexp(m)                 # m = sig * 2**exp, sig in [0.5,1)
    q = int(round(sig * (1 << 31)))
    if q == (1 << 31):
        q //= 2
        exp += 1
    assert q <= INT32_MAX
    return q, int(exp)


def srdhm(a, b):
    """SaturatingRoundingDoublingHighMul (int32 x int32 -> int32)."""
    a = np.int64(a); b = np.int64(b)
    overflow = (a == b) and (a == INT32_MIN)
    ab = a * b
    nudge = (1 << 30) if ab >= 0 else (1 - (1 << 30))
    res = int((ab + nudge) // (1 << 31)) if (ab + nudge) >= 0 else -int((-(ab + nudge)) // (1 << 31)) - (0 if (-(ab + nudge)) % (1 << 31) == 0 else 1)
    # exact C truncation-toward-zero semantics of (int64)/(1<<31)
    v = ab + nudge
    res = int(v // (1 << 31)) if v >= 0 else -int((-v) // (1 << 31))
    return INT32_MAX if overflow else int(res)


def rdpot(x, exp):
    """RoundingDivideByPOT."""
    if exp == 0:
        return int(x)
    mask = (1 << exp) - 1
    rem = x & mask
    thr = (mask >> 1) + (1 if x < 0 else 0)
    return (x >> exp) + (1 if rem > thr else 0)


def mbqm(x, qm, shift):
    """MultiplyByQuantizedMultiplier."""
    left = shift if shift > 0 else 0
    right = 0 if shift > 0 else -shift
    return rdpot(srdhm(int(x) * (1 << left), qm), right)


# ------------------------------------------------------------------ kernels
def conv2d(inp, w, bias, mult, shift, in_zp, out_zp, stride, pad_same,
           amin=-128, amax=127):
    """inp (H,W,Cin) int8 ; w (Cout,kh,kw,Cin) int8 ; per-channel mult/shift."""
    H, W, Cin = inp.shape
    Co, kh, kw, _ = w.shape
    OH = (H + stride - 1) // stride if pad_same else (H - kh) // stride + 1
    OW = (W + stride - 1) // stride if pad_same else (W - kw) // stride + 1
    if pad_same:
        pt = max(0, ((OH - 1) * stride + kh - H)) // 2
        pl = max(0, ((OW - 1) * stride + kw - W)) // 2
    else:
        pt = pl = 0
    x = inp.astype(np.int32) - in_zp
    out = np.empty((OH, OW, Co), dtype=np.int8)
    for oy in range(OH):
        for ox in range(OW):
            iy0, ix0 = oy * stride - pt, ox * stride - pl
            ys0, ys1 = max(0, -iy0), min(kh, H - iy0)
            xs0, xs1 = max(0, -ix0), min(kw, W - ix0)
            patch = x[iy0 + ys0: iy0 + ys1, ix0 + xs0: ix0 + xs1, :]
            for co in range(Co):
                ww = w[co, ys0:ys1, xs0:xs1, :].astype(np.int32)
                acc = int(np.sum(patch * ww)) + int(bias[co])
                v = mbqm(acc, int(mult[co]), int(shift[co])) + out_zp
                out[oy, ox, co] = max(amin, min(amax, v))
    return out


def maxpool(inp, k, stride, amin=-128, amax=127):
    H, W, C = inp.shape
    OH, OW = (H - k) // stride + 1, (W - k) // stride + 1
    out = np.empty((OH, OW, C), dtype=np.int8)
    for oy in range(OH):
        for ox in range(OW):
            blk = inp[oy * stride: oy * stride + k, ox * stride: ox * stride + k, :]
            out[oy, ox, :] = np.clip(blk.reshape(-1, C).max(axis=0), amin, amax)
    return out


def avgpool(inp, k, stride, amin=-128, amax=127):
    H, W, C = inp.shape
    OH, OW = (H - k) // stride + 1, (W - k) // stride + 1
    out = np.empty((OH, OW, C), dtype=np.int8)
    n = k * k
    for oy in range(OH):
        for ox in range(OW):
            blk = inp[oy * stride: oy * stride + k,
                      ox * stride: ox * stride + k, :].astype(np.int32)
            s = blk.reshape(-1, C).sum(axis=0)
            acc = np.where(s > 0, (s + n // 2) // n, -((-s + n // 2) // n))
            out[oy, ox, :] = np.clip(acc, amin, amax)
    return out


def fully_connected(inp, w, bias, mult, shift, in_zp, out_zp,
                    amin=-128, amax=127):
    """inp (Cin,) int8 ; w (Cout,Cin) int8."""
    x = inp.astype(np.int32) - in_zp
    Co = w.shape[0]
    out = np.empty(Co, dtype=np.int8)
    for co in range(Co):
        acc = int(np.dot(x, w[co].astype(np.int32))) + int(bias[co])
        v = mbqm(acc, int(mult[co]), int(shift[co])) + out_zp
        out[co] = max(amin, min(amax, v))
    return out
