"""
AI Arena 2026 - Track 1 : post-training full-integer int8 quantization.

Produces:
  model/model_float32.tflite   - unquantized reference
  model/model_int8.tflite      - full-integer int8, int8 in / int8 out

The Keras graph carries its own Rescaling(1/255) layer, so the network input is
float [0,255]. With a representative set spanning that range the converter
derives input scale = 1.0, zero_point = -128, i.e. on-device the conversion from
a received uint8 pixel to the model's int8 input is exactly  (int8)(pixel - 128).
That keeps the firmware free of any floating point on the input path.
"""
import os, json, argparse
import numpy as np
import tensorflow as tf

# Paths are resolved relative to the repository root so the scripts run
# from a fresh checkout on any machine.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*parts):
    return os.path.join(_REPO, *parts)


CLASS_IDS = [0, 2, 3, 4, 5, 6, 7, 24, 43, 69, 70, 72, 83, 84, 85, 86, 87]


def representative_gen(X, n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))[:n]
    def gen():
        for i in idx:
            yield [X[i:i + 1].astype("float32")]     # already 0..255
    return gen


def tflite_predict(path, X, batch_report=None):
    it = tf.lite.Interpreter(model_path=path)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    preds = np.zeros(len(X), dtype=np.int32)
    for i in range(len(X)):
        x = X[i:i + 1].astype("float32")
        if inp["dtype"] == np.int8:
            s, z = inp["quantization"]
            x = np.clip(np.round(x / s + z), -128, 127).astype(np.int8)
        it.set_tensor(inp["index"], x)
        it.invoke()
        preds[i] = int(np.argmax(it.get_tensor(out["index"])[0]))
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keras", required=True)
    ap.add_argument("--data", default=_p("out", "data.npz"))
    ap.add_argument("--outdir", default=_p("model"))
    a = ap.parse_args()

    d = np.load(a.data)
    Xtr, Xva, yva = d["Xtr"].astype("float32"), d["Xva"].astype("float32"), d["yva"]

    m = tf.keras.models.load_model(a.keras)

    # ---- float32 reference ----------------------------------------------
    c = tf.lite.TFLiteConverter.from_keras_model(m)
    f32 = c.convert()
    p32 = os.path.join(a.outdir, "model_float32.tflite")
    open(p32, "wb").write(f32)

    # ---- full-integer int8 ----------------------------------------------
    c = tf.lite.TFLiteConverter.from_keras_model(m)
    c.optimizations = [tf.lite.Optimize.DEFAULT]
    c.representative_dataset = representative_gen(Xtr)
    c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    c.inference_input_type = tf.int8
    c.inference_output_type = tf.int8
    i8 = c.convert()
    p8 = os.path.join(a.outdir, "model_int8.tflite")
    open(p8, "wb").write(i8)

    # ---- accuracy comparison --------------------------------------------
    acc32 = float((tflite_predict(p32, Xva) == yva).mean())
    acc8 = float((tflite_predict(p8, Xva) == yva).mean())

    it = tf.lite.Interpreter(model_path=p8); it.allocate_tensors()
    inp = it.get_input_details()[0]
    iscale, izp = inp["quantization"]
    ops = sorted({d_["op_name"] for d_ in []}) if False else None

    res = {
        "float32_bytes": len(f32), "int8_bytes": len(i8),
        "compression": round(len(f32) / len(i8), 2),
        "val_acc_float32": acc32, "val_acc_int8": acc8,
        "acc_drop_pp": round((acc32 - acc8) * 100, 2),
        "input_scale": float(iscale), "input_zero_point": int(izp),
    }
    print(json.dumps(res, indent=2))
    json.dump(res, open(os.path.join(a.outdir, "quantization_report.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
