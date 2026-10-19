---
name: Model and data rules
description: Rules for training, quantisation and export
applyTo: 'src/**/*.py'
---

- Preprocessing lives in exactly one place: `load_image()` in
  `src/prepare_data.py`. Training, evaluation and the host sender all call it.
  Duplicating resize or colour handling anywhere else is how training and
  inference silently diverge.
- Pillow's `Image.resize()` defaults to **BICUBIC**, not BILINEAR. State the
  filter explicitly every time.
- Composite alpha over **white**. 54 images have real transparency; dropping the
  channel leaves black holes.
- Fix the seed (`tf.keras.utils.set_random_seed`) in every experiment. With 364
  validation images one image is 0.27 % — an unreproducible delta is noise.
- Report **int8** accuracy. Float32 is not what ships.
- Any TFLite interpreter used as a correctness reference must be constructed
  with `experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF`.
  The default delegates to XNNPACK, which rounds differently and will show
  hundreds of false mismatches.
- Never add horizontal-flip augmentation: classes 24, 43, 83 and 84 are
  directional and mirroring manufactures wrong labels.
- Keep the graph to `CONV_2D`, `MAX_POOL_2D`, `AVERAGE_POOL_2D`, `RESHAPE`,
  `FULLY_CONNECTED`. A sixth operator means writing and bit-verifying a new C
  kernel.
- No softmax in the exported graph. The device computes argmax.
- After any model change, run `src/export_c_model.py` and confirm its assert on
  `input scale = 1.0, zero_point = -128` still passes — the firmware hard-codes
  `(int8)(pixel - 128)`.
