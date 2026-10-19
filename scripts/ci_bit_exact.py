#!/usr/bin/env python3
"""
Compare the C runtime's int8 logits against TFLite's reference kernels.

Reads host_out.txt, which test/host_test.c writes as

    <model_index> <official_class_id> <17 int8 scores>

one line per image, and re-runs the same images through the reference
interpreter. Exits non-zero if the argmax ever disagrees.

The reference resolver matters: the XNNPACK delegate does not reproduce
TFLite's reference int8 arithmetic, so comparing against the default
interpreter proves nothing about bit-exactness.
"""
import os
import sys

import numpy as np

try:
    from ai_edge_litert.interpreter import Interpreter, OpResolverType
except ImportError:  # pragma: no cover - fallback for full TensorFlow
    from tensorflow.lite.python.interpreter import Interpreter, OpResolverType

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*parts):
    return os.path.join(_REPO, *parts)


CLASS_IDS = [0, 2, 3, 4, 5, 6, 7, 24, 43, 69, 70, 72, 83, 84, 85, 86, 87]


def main():
    host = sys.argv[1] if len(sys.argv) > 1 else _p("host_out.txt")
    if not os.path.exists(host):
        print(f"ERROR: {host} not found. Build and run test/host_test.c first.",
              file=sys.stderr)
        return 2

    d = np.load(_p("out", "data.npz"))
    X, y = d["Xva"], d["yva"]

    it = Interpreter(model_path=_p("model", "quantized_model_int8.tflite"),
                     experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]

    ref = np.empty((len(X), len(CLASS_IDS)), dtype=np.int16)
    for i in range(len(X)):
        # The graph's input quantisation is scale 1.0, zero point -128, so the
        # int8 tensor is simply the raw pixel minus 128.
        q = (X[i].astype(np.int16) - 128).astype(np.int8)
        it.set_tensor(inp["index"], q.reshape(inp["shape"]))
        it.invoke()
        ref[i] = it.get_tensor(out["index"])[0]

    rows = [l.split() for l in open(host) if l.strip()]
    if len(rows) != len(X):
        print(f"ERROR: {host} has {len(rows)} rows, expected {len(X)}.",
              file=sys.stderr)
        return 2
    c = np.array([[int(v) for v in r[2:]] for r in rows], dtype=np.int16)
    ids = np.array([int(r[1]) for r in rows])

    same_vec = (c == ref).all(1)
    same_arg = c.argmax(1) == ref.argmax(1)
    truth = np.array([CLASS_IDS[t] for t in y])
    acc = (ids == truth).mean()

    print(f"identical argmax        : {same_arg.sum()}/{len(ref)}")
    print(f"identical score vectors : {same_vec.sum()}/{len(ref)}")
    for i in np.where(~same_vec)[0]:
        delta = np.abs(c[i].astype(int) - ref[i].astype(int))
        print(f"  image {i}: max |delta| {delta.max()} LSB on "
              f"{(delta > 0).sum()} of {len(CLASS_IDS)} logits, "
              f"argmax {'same' if same_arg[i] else 'DIFFERENT'}")
    print(f"C runtime accuracy      : {acc * 100:.2f}% "
          f"({(ids == truth).sum()}/{len(y)})")

    if not same_arg.all():
        print("FAIL: the C runtime disagrees with the reference kernels.",
              file=sys.stderr)
        return 1
    print("PASS: argmax agreement is complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
