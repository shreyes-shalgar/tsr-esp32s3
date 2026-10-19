"""Vectorised, bit-exact NumPy execution of the extracted int8 graph."""
import numpy as np

I32MIN, I32MAX = -(2**31), 2**31 - 1


def srdhm_v(a, b):
    a = a.astype(np.int64); b = np.asarray(b, dtype=np.int64)
    ab = a * b
    nudge = np.where(ab >= 0, np.int64(1 << 30), np.int64(1 - (1 << 30)))
    v = ab + nudge
    return np.where(v >= 0, v >> 31, -((-v) >> 31))


def rdpot_v(x, exp):
    exp = np.asarray(exp, dtype=np.int64)
    mask = (np.int64(1) << exp) - 1
    rem = x & mask
    thr = (mask >> 1) + (x < 0).astype(np.int64)
    return (x >> exp) + (rem > thr).astype(np.int64)


def mbqm_v(x, qm, shift):
    """MultiplyByQuantizedMultiplier, vectorised over the last axis."""
    x = np.asarray(x, dtype=np.int64)
    shift = np.asarray(shift, dtype=np.int64)
    left = np.maximum(shift, 0)
    right = np.maximum(-shift, 0)
    return rdpot_v(srdhm_v(x << left, qm), right)


def _im2col(x, kh, kw, stride, pt, pl, OH, OW):
    H, W, C = x.shape
    xp = np.zeros((H + 2 * pt + 1, W + 2 * pl + 1, C), dtype=np.int32)
    xp[pt:pt + H, pl:pl + W, :] = x
    cols = np.empty((OH * OW, kh * kw * C), dtype=np.int32)
    n = 0
    for oy in range(OH):
        for ox in range(OW):
            cols[n] = xp[oy * stride: oy * stride + kh,
                         ox * stride: ox * stride + kw, :].reshape(-1)
            n += 1
    return cols


def conv(x, L):
    W_ = L["W"]; Co, kh, kw, Ci = W_.shape
    H, Wd, _ = x.shape
    s = L["stride"]
    OH, OW = L["out_shape"][0], L["out_shape"][1]
    pt = max(0, (OH - 1) * s + kh - H) // 2
    pl = max(0, (OW - 1) * s + kw - Wd) // 2
    xs = x.astype(np.int32) - L["in_zp"]
    cols = _im2col(xs, kh, kw, s, pt, pl, OH, OW)          # (OH*OW, kh*kw*Ci)
    wm = W_.reshape(Co, -1).astype(np.int32)               # (Co, kh*kw*Ci)
    acc = cols.astype(np.int64) @ wm.T.astype(np.int64)    # (OH*OW, Co)
    acc += L["B"].astype(np.int64)
    v = mbqm_v(acc, L["mult"], L["shift"]) + L["out_zp"]
    return np.clip(v, -128, 127).astype(np.int8).reshape(OH, OW, Co)


def maxpool(x, L):
    k, s = L["k"], L["stride"]
    H, W, C = x.shape
    OH, OW = (H - k) // s + 1, (W - k) // s + 1
    v = x.reshape(OH, s, OW, s, C) if (s == k and H == OH * k) else None
    return v.max(axis=(1, 3)).astype(np.int8)


def avgpool(x, L):
    k, s = L["k"], L["stride"]
    H, W, C = x.shape
    OH, OW = (H - k) // s + 1, (W - k) // s + 1
    blk = x.reshape(OH, k, OW, k, C).astype(np.int32)
    ssum = blk.sum(axis=(1, 3))
    n = k * k
    acc = np.where(ssum > 0, (ssum + n // 2) // n, -((-ssum + n // 2) // n))
    return np.clip(acc, -128, 127).astype(np.int8)


def fc(x, L):
    xs = x.reshape(-1).astype(np.int64) - L["in_zp"]
    acc = L["W"].astype(np.int64) @ xs + L["B"].astype(np.int64)
    v = mbqm_v(acc, L["mult"], L["shift"]) + L["out_zp"]
    return np.clip(v, -128, 127).astype(np.int8)


def run(P, img_u8):
    """img_u8: (48,48,3) uint8 -> int8 output vector (17,)"""
    x = (img_u8.astype(np.int32) - 128).astype(np.int8)
    for L in P["layers"]:
        o = L["op"]
        x = conv(x, L) if o == "conv" else \
            maxpool(x, L) if o == "maxpool" else \
            avgpool(x, L) if o == "avgpool" else \
            fc(x, L)
    return x
