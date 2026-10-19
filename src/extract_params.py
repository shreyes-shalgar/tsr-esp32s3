"""Pull weights + quantization params out of the int8 .tflite into a dict."""
import numpy as np, tensorflow as tf
from tflite_int8_ref import quantize_multiplier

SKIP = ("DELEGATE", "SHAPE", "STRIDED_SLICE", "PACK")


def extract(path):
    it = tf.lite.Interpreter(model_path=path)
    it.allocate_tensors()
    td = {d["index"]: d for d in it.get_tensor_details()}

    def q(i):
        p = td[i]["quantization_parameters"]
        return p["scales"], p["zero_points"]

    layers = []
    for op in it._get_ops_details():
        name = op["op_name"]
        if name in SKIP:
            continue
        oi = op["outputs"][0]
        osc, ozp = q(oi)
        if name == "CONV_2D":
            ii, wi, bi = op["inputs"][0], op["inputs"][1], op["inputs"][2]
            isc, izp = q(ii)
            wsc, _ = q(wi)
            W = it.get_tensor(wi); B = it.get_tensor(bi)
            reals = (isc[0] * wsc) / osc[0]
            ms = [quantize_multiplier(float(r)) for r in reals]
            stride = 2 if (td[ii]["shape"][1] // td[oi]["shape"][1]) == 2 else 1
            layers.append(dict(op="conv", W=W, B=B,
                               mult=np.array([m for m, _ in ms], dtype=np.int32),
                               shift=np.array([s for _, s in ms], dtype=np.int32),
                               in_zp=int(izp[0]), out_zp=int(ozp[0]),
                               stride=stride,
                               in_shape=tuple(int(v) for v in td[ii]["shape"][1:]),
                               out_shape=tuple(int(v) for v in td[oi]["shape"][1:])))
        elif name in ("MAX_POOL_2D", "AVERAGE_POOL_2D"):
            ii = op["inputs"][0]
            ish, osh = td[ii]["shape"], td[oi]["shape"]
            k = int(ish[1] // osh[1])
            layers.append(dict(op="maxpool" if name == "MAX_POOL_2D" else "avgpool",
                               k=k, stride=k,
                               in_shape=tuple(int(v) for v in ish[1:]),
                               out_shape=tuple(int(v) for v in osh[1:])))
        elif name == "FULLY_CONNECTED":
            ii, wi, bi = op["inputs"][0], op["inputs"][1], op["inputs"][2]
            isc, izp = q(ii); wsc, _ = q(wi)
            W = it.get_tensor(wi); B = it.get_tensor(bi)
            reals = (isc[0] * wsc) / osc[0]
            ms = [quantize_multiplier(float(r)) for r in reals]
            layers.append(dict(op="fc", W=W, B=B,
                               mult=np.array([m for m, _ in ms], dtype=np.int32),
                               shift=np.array([s for _, s in ms], dtype=np.int32),
                               in_zp=int(izp[0]), out_zp=int(ozp[0]),
                               in_shape=(int(W.shape[1]),),
                               out_shape=(int(W.shape[0]),)))
        elif name == "RESHAPE":
            continue
    inp = it.get_input_details()[0]
    out = it.get_output_details()[0]
    return dict(layers=layers,
                in_scale=float(inp["quantization"][0]),
                in_zp=int(inp["quantization"][1]),
                out_scale=float(out["quantization"][0]),
                out_zp=int(out["quantization"][1]))
