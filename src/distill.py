#!/usr/bin/env python3
"""
Ensemble knowledge distillation into the shipped student.

Why this and not a bigger model or an ensemble on the device: the challenge
scores plain accuracy on a hidden test set, and breaks ties on RAM, inference
time and model size (guide sections 9 and 11). So every accuracy gain has to
be free at inference. Distillation is: the student architecture, arena and
weight count are byte-for-byte unchanged - only the values in the weights
differ.

Two ImageNet-pretrained teachers are fine-tuned on Train only (Validation is
never shown to them), their softened predictions are averaged, and the student
is trained to match that average as well as the hard label. With 1,460
training images the pretrained features carry information no 40k-parameter
network can learn from the data alone; the soft targets are how that
information gets transferred.

Stages are checkpointed so the run can be resumed.
"""
import os, sys, json, argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_training import (build_model, build_aug_head, augment,
                            CLASS_IDS, NCLS, IMG, AUTOTUNE, _p)

WORK = os.environ.get("KD_WORK", _p("out", "kd"))
os.makedirs(WORK, exist_ok=True)


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------- teacher
TEACHERS = [
    ("mnv2_100_128", 128, 1.00),
    ("mnv2_075_96",   96, 0.75),
]


def build_teacher(res, alpha):
    inp = keras.Input((IMG, IMG, 3))                 # 0..255
    x = layers.Resizing(res, res, interpolation="bilinear")(inp)
    x = layers.Rescaling(1 / 127.5, offset=-1.0)(x)  # MobileNetV2 domain
    bb = keras.applications.MobileNetV2(
        weights="imagenet", include_top=False, alpha=alpha,
        input_shape=(res, res, 3))
    x = bb(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(NCLS)(x)
    return keras.Model(inp, out), bb


def aug_ds(X, y, batch, head, seed):
    ds = tf.data.Dataset.from_tensor_slices((X.astype("float32") / 255.0, y))
    ds = ds.shuffle(len(X), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(augment, num_parallel_calls=AUTOTUNE).batch(batch)
    ds = ds.map(lambda a, b: (head(a, training=True) * 255.0, b),
                num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


def plain_ds(X, y, batch):
    return tf.data.Dataset.from_tensor_slices(
        (X.astype("float32"), y)).batch(batch).prefetch(AUTOTUNE)


def fit_teacher(name, res, alpha, Xtr, ytr, Xva, yva, cw, epochs, batch):
    ck = os.path.join(WORK, f"teacher_{name}.keras")
    if os.path.exists(ck):
        log(f"[teacher {name}] cached")
        return keras.models.load_model(ck)
    m, bb = build_teacher(res, alpha)
    head = build_aug_head()
    tr = aug_ds(Xtr, ytr, batch, head, seed=7)
    va = plain_ds(Xva, yva, batch)
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                      label_smoothing=0.05) \
        if "label_smoothing" in keras.losses.SparseCategoricalCrossentropy.__init__.__code__.co_varnames \
        else keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # stage 1: head only
    bb.trainable = False
    m.compile(optimizer=keras.optimizers.AdamW(1e-3, weight_decay=1e-4),
              loss=loss, metrics=["accuracy"])
    m.fit(tr, validation_data=va, epochs=max(3, epochs // 6),
          class_weight=cw, verbose=2)

    # stage 2: full fine-tune, low LR
    bb.trainable = True
    steps = int(np.ceil(len(Xtr) / batch))
    sched = keras.optimizers.schedules.CosineDecay(
        1e-4, decay_steps=epochs * steps, warmup_target=3e-4,
        warmup_steps=2 * steps)
    m.compile(optimizer=keras.optimizers.AdamW(sched, weight_decay=1e-5),
              loss=loss, metrics=["accuracy"])
    m.fit(tr, validation_data=va, epochs=epochs, class_weight=cw, verbose=2)
    _, acc = m.evaluate(va, verbose=0)
    log(f"[teacher {name}] val_acc={acc:.4f}")
    m.save(ck)
    json.dump({"val_acc": float(acc)},
              open(os.path.join(WORK, f"teacher_{name}.json"), "w"))
    return m


# ------------------------------------------------------------------ view pool
def make_views(Xtr, ytr, k, batch=128):
    """K augmented copies of the training set, materialised as uint8."""
    f = os.path.join(WORK, f"views_k{k}.npz")
    if os.path.exists(f):
        d = np.load(f)
        log(f"[views] cached {d['X'].shape}")
        return d["X"], d["y"]
    head = build_aug_head()
    outX, outY = [], []
    for j in range(k):
        tf.keras.utils.set_random_seed(1000 + j)
        ds = tf.data.Dataset.from_tensor_slices((Xtr.astype("float32") / 255.0, ytr))
        ds = ds.map(augment, num_parallel_calls=AUTOTUNE).batch(batch)
        ds = ds.map(lambda a, b: (head(a, training=True) * 255.0, b))
        for xb, yb in ds:
            outX.append(np.clip(np.round(xb.numpy()), 0, 255).astype(np.uint8))
            outY.append(yb.numpy())
        log(f"[views] {j + 1}/{k}")
    X = np.concatenate(outX); y = np.concatenate(outY)
    np.savez_compressed(f, X=X, y=y)
    log(f"[views] {X.shape}")
    return X, y


def teacher_logits(models, X, batch=128):
    f = os.path.join(WORK, f"tlogits_{len(X)}.npy")
    if os.path.exists(f):
        log("[tlogits] cached")
        return np.load(f)
    acc = np.zeros((len(X), NCLS), dtype=np.float64)
    for mi, m in enumerate(models):
        out = []
        for i in range(0, len(X), batch):
            out.append(m(X[i:i + batch].astype("float32"), training=False).numpy())
            if (i // batch) % 40 == 0:
                log(f"[tlogits {mi}] {i}/{len(X)}")
        acc += np.concatenate(out).astype(np.float64)
    acc /= len(models)
    np.save(f, acc.astype(np.float32))
    return acc.astype(np.float32)


# -------------------------------------------------------------------- student
def kd_train(Xv, yv, tlog, Xva, yva, cw, epochs, batch, alpha, T, lr, seed, tag):
    tf.keras.utils.set_random_seed(seed)
    st = build_model("A", 1.0)

    ds = tf.data.Dataset.from_tensor_slices(
        (Xv.astype("float32"), yv.astype("int32"), tlog))
    ds = ds.shuffle(min(len(Xv), 60000), reshuffle_each_iteration=True)
    ds = ds.batch(batch).prefetch(AUTOTUNE)
    va = plain_ds(Xva, yva, batch)

    steps = int(np.ceil(len(Xv) / batch))
    sched = keras.optimizers.schedules.CosineDecay(
        lr, decay_steps=epochs * steps, warmup_target=lr, warmup_steps=2 * steps)
    opt = keras.optimizers.AdamW(sched, weight_decay=1e-4)
    cwv = tf.constant([cw[i] for i in range(NCLS)], dtype=tf.float32)

    @tf.function
    def step(xb, yb, tb):
        with tf.GradientTape() as tape:
            s = st(xb, training=True)
            hard = tf.nn.sparse_softmax_cross_entropy_with_logits(yb, s)
            soft = tf.reduce_sum(
                tf.nn.softmax(tb / T) *
                (tf.nn.log_softmax(tb / T) - tf.nn.log_softmax(s / T)), axis=-1)
            w = tf.gather(cwv, yb)
            l = tf.reduce_mean(w * ((1 - alpha) * hard + alpha * (T * T) * soft))
        g = tape.gradient(l, st.trainable_variables)
        opt.apply_gradients(zip(g, st.trainable_variables))
        return l

    best, ck = 0.0, os.path.join(WORK, f"student_{tag}.keras")
    for e in range(epochs):
        tot, nb = 0.0, 0
        for xb, yb, tb in ds:
            tot += float(step(xb, yb, tb)); nb += 1
        corr = tot_n = 0
        for xb, yb in va:
            p = np.argmax(st(xb, training=False).numpy(), axis=-1)
            corr += int((p == yb.numpy()).sum()); tot_n += len(p)
        acc = corr / tot_n
        if acc >= best:
            best = acc; st.save(ck)
        log(f"[student {tag}] epoch {e+1}/{epochs} loss={tot/nb:.4f} "
            f"val={acc:.4f} best={best:.4f}")
    log(f"[student {tag}] BEST {best:.4f} -> {ck}")
    return ck, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=_p("out", "data.npz"))
    ap.add_argument("--views", type=int, default=24)
    ap.add_argument("--teacher-epochs", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=0.75)
    ap.add_argument("--T", type=float, default=4.0)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="kd")
    a = ap.parse_args()

    d = np.load(a.data)
    Xtr, ytr, Xva, yva = d["Xtr"], d["ytr"], d["Xva"], d["yva"]
    cnt = np.bincount(ytr, minlength=NCLS).astype("float32")
    cw = {i: float(cnt.sum() / (NCLS * cnt[i])) for i in range(NCLS)}
    log(f"train={Xtr.shape} val={Xva.shape}")

    ms = [fit_teacher(n, r, al, Xtr, ytr, Xva, yva, cw,
                      a.teacher_epochs, a.batch) for n, r, al in TEACHERS]

    Xv, yv = make_views(Xtr, ytr, a.views)
    tl = teacher_logits(ms, Xv)
    log(f"[tlogits] {tl.shape}  teacher-on-views agreement with label: "
        f"{(np.argmax(tl,1)==yv).mean():.4f}")

    ck, best = kd_train(Xv, yv, tl, Xva, yva, cw, a.epochs, a.batch,
                        a.alpha, a.T, a.lr, a.seed, a.tag)
    json.dump({"student": ck, "val_acc": best},
              open(os.path.join(WORK, f"result_{a.tag}.json"), "w"))


if __name__ == "__main__":
    main()
