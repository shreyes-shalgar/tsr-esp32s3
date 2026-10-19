"""
AI Arena 2026 - Track 1 : training.

Design constraints (ESP32-S3 WROOM, 512 KB SRAM):
  * ops restricted to CONV_2D / MAX_POOL_2D / AVERAGE_POOL_2D / RESHAPE /
    FULLY_CONNECTED so the int8 runtime stays tiny and portable
  * no softmax in the graph - the device only needs argmax
  * peak activation memory bounded by architecture choice (see build_model)

Augmentation notes: the seven pictogram classes (69,70,72,83-86) are supplied
as 32x32 upscaled crops while the other ten are full-resolution photographs.
`random_resolution` deliberately destroys that resolution cue so the network
cannot take the domain shortcut.
"""
import os, json, argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Paths are resolved relative to the repository root so the scripts run
# from a fresh checkout on any machine.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*parts):
    return os.path.join(_REPO, *parts)


IMG = 48
NCLS = 17
CLASS_IDS = [0, 2, 3, 4, 5, 6, 7, 24, 43, 69, 70, 72, 83, 84, 85, 86, 87]
AUTOTUNE = tf.data.AUTOTUNE


# ---------------------------------------------------------------- augmentation
def random_resolution(x):
    """Randomly degrade then restore resolution (breaks the 32x32 domain cue)."""
    s = tf.random.uniform([], 16, 49, dtype=tf.int32)
    y = tf.image.resize(x, (s, s), method="bilinear")
    return tf.image.resize(y, (IMG, IMG), method="bilinear")


def random_grayscale(x):
    """Collapse to luma and replicate. The evaluation host can send a 48x48
    grayscale payload (-g flag in send_image.py); without this the model loses
    ~24 points of accuracy on that path."""
    g = tf.image.rgb_to_grayscale(x)
    return tf.image.grayscale_to_rgb(g)


def random_lowlight(x):
    """
    Night-time capture, modelled rather than just dimmed.

    A straight brightness scale is not what darkness does to a camera: fewer
    photons means the signal-to-noise ratio collapses, and it is the noise that
    destroys a classifier, not the dimness. So this linearises out of gamma,
    cuts the exposure, applies Poisson shot noise plus Gaussian read noise, and
    re-encodes. Measured on the validation set, an unhardened model holds to
    about 70 % darkness and then falls off a cliff.
    """
    a = tf.random.uniform([], 0.25, 0.93)
    full_well = 9000.0
    lin = tf.pow(tf.maximum(x, 1e-6), 2.2)
    scale = tf.pow(1.0 - a, 2.2)
    photons = lin * full_well * scale
    noisy = tf.random.poisson([], photons) + tf.random.normal(tf.shape(x), 0.0, 2.0)
    return tf.pow(tf.clip_by_value(noisy / full_well, 1e-6, 1.0), 1.0 / 2.2)


def augment(x, y):
    x = tf.image.random_brightness(x, 0.25)
    x = tf.image.random_contrast(x, 0.7, 1.4)
    x = tf.image.random_saturation(x, 0.6, 1.5)
    x = tf.cond(tf.random.uniform([]) < 0.35, lambda: random_resolution(x), lambda: x)
    x = tf.cond(tf.random.uniform([]) < 0.25, lambda: random_grayscale(x), lambda: x)
    x = tf.cond(tf.random.uniform([]) < 0.22, lambda: random_lowlight(x), lambda: x)
    x = tf.clip_by_value(x, 0.0, 1.0)
    return x, y


def build_aug_head():
    # geometric aug; NO horizontal flip - signs are directional (24 "go right",
    # 43 "go right or straight", 83/84 ascent vs descent) so mirroring
    # would create wrong labels.
    return keras.Sequential([
        layers.RandomRotation(0.045, fill_mode="nearest"),
        layers.RandomZoom(0.18, 0.18, fill_mode="nearest"),
        layers.RandomTranslation(0.12, 0.12, fill_mode="nearest"),
    ], name="geo_aug")


# ---------------------------------------------------------------------- model
def conv_block(x, f, k=3, stride=1, name=""):
    x = layers.Conv2D(f, k, strides=stride, padding="same", use_bias=False,
                      name=f"{name}_conv")(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    return layers.ReLU(name=f"{name}_relu")(x)


def build_model(variant="A", width=1.0):
    """
    variant A - RAM optimised : first conv is stride 2, peak activation ~9 KB
    variant B - accuracy first: first conv is stride 1, peak activation ~36 KB
    """
    w = lambda n: max(8, int(round(n * width)))
    inp = keras.Input((IMG, IMG, 3), name="image")
    x = layers.Rescaling(1.0 / 255.0, name="rescale")(inp)

    if variant == "A":
        x = conv_block(x, w(16), stride=2, name="b1")          # 24x24
        x = conv_block(x, w(32), name="b2a")
        x = layers.MaxPooling2D(2, name="p2")(x)               # 12x12
        x = conv_block(x, w(64), name="b3a")
        x = layers.MaxPooling2D(2, name="p3")(x)               # 6x6
        x = conv_block(x, w(64), name="b4a")
    else:
        x = conv_block(x, w(16), name="b1")                    # 48x48
        x = layers.MaxPooling2D(2, name="p1")(x)               # 24x24
        x = conv_block(x, w(32), name="b2a")
        x = layers.MaxPooling2D(2, name="p2")(x)               # 12x12
        x = conv_block(x, w(64), name="b3a")
        x = layers.MaxPooling2D(2, name="p3")(x)               # 6x6
        x = conv_block(x, w(64), name="b4a")

    x = layers.AveragePooling2D(6, name="gap")(x)              # 1x1xC
    x = layers.Flatten(name="flat")(x)
    x = layers.Dropout(0.3, name="drop")(x)
    out = layers.Dense(NCLS, name="logits")(x)                 # logits, no softmax
    return keras.Model(inp, out, name=f"tsr_{variant}")


# ------------------------------------------------------------------- training
def make_ds(X, y, training, batch, aug_head=None):
    ds = tf.data.Dataset.from_tensor_slices((X.astype("float32") / 255.0, y))
    if training:
        ds = ds.shuffle(len(X), reshuffle_each_iteration=True)
        ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch)
    if training and aug_head is not None:
        ds = ds.map(lambda a, b: (aug_head(a, training=True), b),
                    num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(lambda a, b: (a * 255.0, b), num_parallel_calls=AUTOTUNE)
    else:
        ds = ds.map(lambda a, b: (a * 255.0, b), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="A")
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--data", default=_p("out", "data.npz"))
    ap.add_argument("--outdir", default=_p("model"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use-all", action="store_true",
                    help="train on Train+Validation combined (no honest holdout; "
                         "validation metrics become training-set metrics)")
    a = ap.parse_args()

    tf.keras.utils.set_random_seed(a.seed)
    d = np.load(a.data)
    Xtr, ytr, Xva, yva = d["Xtr"], d["ytr"], d["Xva"], d["yva"]
    if a.use_all:
        Xtr = np.concatenate([Xtr, Xva], axis=0)
        ytr = np.concatenate([ytr, yva], axis=0)
        print(f"training on Train+Validation: {len(Xtr)} images "
              f"(validation metrics below are NOT a holdout)")

    cnt = np.bincount(ytr, minlength=NCLS).astype("float32")
    cw = {i: float((cnt.sum() / (NCLS * cnt[i]))) for i in range(NCLS)}

    aug_head = build_aug_head()
    tr = make_ds(Xtr, ytr, True, a.batch, aug_head)
    va = make_ds(Xva, yva, False, a.batch)

    m = build_model(a.variant, a.width)
    steps = int(np.ceil(len(Xtr) / a.batch))
    sched = keras.optimizers.schedules.CosineDecay(
        a.lr, decay_steps=a.epochs * steps, warmup_target=a.lr,
        warmup_steps=5 * steps)
    m.compile(optimizer=keras.optimizers.AdamW(sched, weight_decay=1e-4),
              loss=keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=["accuracy"])

    tag = f"{a.variant}_w{a.width}" + ("_all" if a.use_all else "")
    ckpt = os.path.join(a.outdir, f"best_{tag}.keras")
    hist = m.fit(tr, validation_data=va, epochs=a.epochs, class_weight=cw,
                 verbose=2,
                 callbacks=[keras.callbacks.ModelCheckpoint(
                     ckpt, monitor="val_accuracy", save_best_only=True,
                     mode="max", verbose=0)])

    m.load_weights(ckpt)
    loss, acc = m.evaluate(va, verbose=0)
    nparam = m.count_params()
    print(f"\n=== {tag} : best val_acc={acc:.4f}  params={nparam}")
    with open(os.path.join(a.outdir, f"result_{tag}.json"), "w") as fh:
        json.dump({"variant": a.variant, "width": a.width, "val_acc": float(acc),
                   "params": int(nparam)}, fh)


if __name__ == "__main__":
    main()
