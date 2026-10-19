#!/usr/bin/env python3
"""
Stage 2 of distillation: train the student against an ensemble soft target.

Split out from distill.py so the expensive parts - fine-tuning the teachers and
running them over the augmented view pool - are done once and cached, and the
cheap part (which ensemble, which temperature, which alpha) can be swept.

Ensembling is done on *probabilities at temperature T*, not on raw logits.
Different heads put logits on different scales; averaging them directly lets
whichever model happens to be most confident dominate the target. Averaging
distributions is scale-free and is what "ensemble prediction" actually means.
"""
import os, sys, json, argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_training import build_model, NCLS, _p

WORK = os.environ.get("KD_WORK", _p("out", "kd"))


def log(*a): print(*a, flush=True)


def probs_of(model, X, T, batch=128, tag=""):
    f = os.path.join(WORK, f"p_{tag}_{len(X)}_T{T}.npy")
    if os.path.exists(f):
        return np.load(f)
    out = []
    for i in range(0, len(X), batch):
        z = model(X[i:i + batch].astype("float32"), training=False).numpy()
        out.append(tf.nn.softmax(z / T).numpy().astype(np.float32))
        if (i // batch) % 50 == 0:
            log(f"  [{tag}] {i}/{len(X)}")
    p = np.concatenate(out)
    np.save(f, p)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", required=True,
                    help="keras files to ensemble as the teacher")
    ap.add_argument("--views", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=0.75)
    ap.add_argument("--T", type=float, default=4.0)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data", default=_p("out", "data.npz"))
    a = ap.parse_args()

    d = np.load(a.data); Xva, yva = d["Xva"], d["yva"]
    ytr = d["ytr"]
    v = np.load(os.path.join(WORK, f"views_k{a.views}.npz"))
    Xv, yv = v["X"], v["y"]
    log(f"views {Xv.shape}")

    P = None
    for m in a.members:
        mdl = keras.models.load_model(m, compile=False)
        p = probs_of(mdl, Xv, a.T, tag=os.path.basename(m).replace(".keras", ""))
        log(f"  {os.path.basename(m)}: agrees with label on views "
            f"{(np.argmax(p,1)==yv).mean():.4f}")
        P = p if P is None else P + p
        del mdl
    P /= len(a.members)
    log(f"ensemble agrees with label on views {(np.argmax(P,1)==yv).mean():.4f}")

    cnt = np.bincount(ytr, minlength=NCLS).astype("float32")
    cwv = tf.constant([cnt.sum() / (NCLS * cnt[i]) for i in range(NCLS)],
                      dtype=tf.float32)

    tf.keras.utils.set_random_seed(a.seed)
    st = build_model("A", 1.0)
    # Keep the pool as uint8 and cast per batch: 35k x 48x48x3 as float32 is
    # ~1 GB, and tf.data would hold a second copy of it.
    ds = tf.data.Dataset.from_tensor_slices((Xv, yv.astype("int32"), P))
    ds = ds.shuffle(20000, reshuffle_each_iteration=True)
    ds = ds.batch(a.batch)
    ds = ds.map(lambda x, y, p: (tf.cast(x, tf.float32), y, p),
                num_parallel_calls=tf.data.AUTOTUNE).prefetch(2)

    steps = int(np.ceil(len(Xv) / a.batch))
    sched = keras.optimizers.schedules.CosineDecay(
        a.lr, decay_steps=a.epochs * steps, warmup_target=a.lr, warmup_steps=2 * steps)
    opt = keras.optimizers.AdamW(sched, weight_decay=1e-4)
    T, alpha = a.T, a.alpha

    @tf.function
    def step(xb, yb, pb):
        with tf.GradientTape() as tape:
            s = st(xb, training=True)
            hard = tf.nn.sparse_softmax_cross_entropy_with_logits(yb, s)
            soft = -tf.reduce_sum(pb * tf.nn.log_softmax(s / T), axis=-1)
            w = tf.gather(cwv, yb)
            l = tf.reduce_mean(w * ((1 - alpha) * hard + alpha * (T * T) * soft))
        g = tape.gradient(l, st.trainable_variables)
        opt.apply_gradients(zip(g, st.trainable_variables))
        return l

    Xvaf = Xva.astype("float32")   # 364 images, negligible
    best, ck = 0.0, os.path.join(WORK, f"student_{a.tag}.keras")
    for e in range(a.epochs):
        tot = nb = 0
        for xb, yb, pb in ds:
            tot += float(step(xb, yb, pb)); nb += 1
        pr = np.concatenate([np.argmax(st(Xvaf[i:i+256], training=False).numpy(), 1)
                             for i in range(0, len(Xvaf), 256)])
        acc = float((pr == yva).mean())
        if acc >= best:
            best = acc; st.save(ck)
        log(f"[{a.tag}] epoch {e+1}/{a.epochs} loss={tot/nb:.4f} val={acc:.4f} best={best:.4f}")
    log(f"[{a.tag}] BEST {best:.4f} -> {ck}")
    json.dump({"tag": a.tag, "members": a.members, "T": T, "alpha": alpha,
               "val_acc": best}, open(os.path.join(WORK, f"res_{a.tag}.json"), "w"))


if __name__ == "__main__":
    main()
