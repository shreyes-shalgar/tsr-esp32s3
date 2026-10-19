# AI Arena 2026 — Track 1: Traffic Sign Recognition on ESP32-S3

You are working on a hackathon submission. Read this file completely before your
first action. It is the contract for the whole project.

---

## 1. The goal, stated precisely

Classify a traffic sign image into one of **17 classes**, running as **int8
inference on Zephyr RTOS**, driven over **UART at 115200 8N1**, on **QEMU
(qemu_x86)** and buildable for **ESP32-S3 WROOM**.

Scoring (from the official guide):

| Category | Weight |
|---|---|
| Model Accuracy | 30 % |
| Optimization & Quantization | 20 % |
| Embedded Deployment (QEMU + ESP32) | 20 % |
| Inference Performance | 15 % |
| Code Quality & Documentation | 10 % |
| Innovation | 5 % |

Tie-breaker, in this order: **lowest RAM**, then **fastest inference**, then
**smallest model**.

The organisers have separately clarified that the **confidence percentage the
board reports is treated as the accuracy criterion** — "with how much confidence
does your model predict the given image", reported as `90%` not `0.90`.

Final scoring uses a **hidden test set** you will never see.

---

## 2. Non-negotiable constraints

Violating any of these invalidates the submission. Do not "improve" your way
past them.

1. **UART is 115200 baud, 8N1, no flow control.** The organisers have stated
   that a submission with the wrong baud rate will not be evaluated.
2. **Model input is 48×48×3 RGB** (6,912 bytes). Confirmed final.
3. **The prediction must report the OFFICIAL class ID**, not the 0–16 array
   index. The official IDs are non-consecutive:
   `0, 2, 3, 4, 5, 6, 7, 24, 43, 69, 70, 72, 83, 84, 85, 86, 87`.
4. **Exact line protocol** — the host script parses on these keywords:
   ```
   board → host   READY
   host  → board  ASCII hex, 32 bytes (64 chars) per line
   host  → board  STOP
   board → host   RECEIVED <n> bytes
   board → host   PREDICTION class=<id> label=<name> confidence=<p>%
   board → host   READY            ← required; multiple images run back-to-back
   ```
   On any failure emit a line starting with `ERROR` instead of `PREDICTION`,
   or the host waits out a 30-second timeout.
5. **Nothing may be printed between `RECEIVED` and `PREDICTION`.** The host
   measures inference time as the gap between those two lines. Any extra line
   there is scored as your inference time.
6. **No heap in the inference path.** No `malloc`, no dynamic allocation.
7. **The firmware must build for both `qemu_x86` and
   `esp32s3_devkitc/esp32s3/procpu`.**

---

## 3. Current state: a verified baseline already exists

**Do not rebuild this from scratch.** A working, measured solution is in the
repo. Your job is to improve it without breaking it.

| Metric | Value |
|---|---|
| Validation accuracy (int8, measured on QEMU over real UART) | **99.45 %** (362/364) |
| Parameters | 62,145 |
| int8 model | 72,408 B |
| Weights in firmware flash | 63,740 B |
| **Peak inference RAM** | **27,648 B** |
| Inference on QEMU | ~9 ms |
| ESP32-S3 | FLASH 199,684 B (2.4 %), DRAM 87,472 B (21.9 %) |
| Bit-exactness vs TFLite reference | argmax **364/364** |
| Confidence calibration | ECE 0.62 %; 99.06 % when correct, 57.17 % when wrong |

Architecture: 4 conv blocks, stride-2 first conv, avgpool, dense.
`48×48×3 → conv3×3/s2 ×16 → conv3×3 ×32 → pool → conv3×3 ×64 → pool →
conv3×3 ×64 → avgpool6 → dense 17 (logits, no softmax in the graph)`.

Only five operator types are used: `CONV_2D`, `MAX_POOL_2D`,
`AVERAGE_POOL_2D`, `RESHAPE`, `FULLY_CONNECTED`.

**Any change you make must clear the gate in section 6 before you keep it.**

---

## 4. Traps already discovered — do not rediscover these

Each of these cost real debugging time. They are facts, not guesses.

1. **TensorFlow Lite does not use one rounding rule.** `CONV_2D` uses gemmlowp's
   `SaturatingRoundingDoublingHighMul` + `RoundingDivideByPOT`. Per-channel
   `FULLY_CONNECTED` uses a higher-precision path. Applying either rule
   uniformly produces 1-LSB errors on ~4 % of images. Both are implemented in
   `src/qemu_app/src/tsr_model.c`. **Do not "simplify" them into one.**

2. **The Python TFLite interpreter silently delegates to XNNPACK**, whose int8
   rounding differs from the reference kernels. For any bit-exactness comparison
   you MUST construct it as:
   ```python
   tf.lite.Interpreter(model_path=...,
       experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
   ```
   Comparing against the default interpreter will show ~343/364 mismatches that
   are not real.

3. **UART RX must be polled, not interrupt-driven.** Leaving bytes in the FIFO
   preserves flow control back to the sender. An ISR draining into a ring buffer
   defeats that backpressure and silently drops bytes once the buffer fills.
   This was observed failing. Do not "optimise" it into an ISR.

4. **`uart_irq_update()` returns `void` in Zephyr 4.4**, not `int`. Older
   examples that write `if (!uart_irq_update(dev))` will not compile.

5. **QEMU's socket chardev discards output when no client is connected.** Launch
   with `server=on,wait=on` so QEMU blocks until the test harness attaches,
   otherwise the boot banner and first `READY` are lost and the harness hangs.

6. **The dataset is two different datasets.** Classes 0, 2–7, 24, 43, 87 are
   photographs (80–280 px). Classes 69, 70, 72, 83–86 are 32×32 tight crops with
   an alpha channel, pre-augmented with horizontal flips, some 1-bit black and
   white. A network can separate the two groups on image statistics alone
   without looking at the sign. `random_resolution` augmentation exists to
   destroy that shortcut. **Keep it.**

7. **Colour was a crutch.** Without grayscale augmentation the model scored
   74.18 % on grayscale input vs 98.35 % on colour. Adding random grayscale
   augmentation (p = 0.25) fixed that *and* raised colour accuracy to 99.45 %.
   **Keep it.**

8. **Never use horizontal flip augmentation.** Classes 24 (Go Right), 43 (Go
   right or straight), 83 (Steep descent) and 84 (Steep ascent) are directional.
   Mirroring manufactures wrong labels.

9. **Pillow's `Image.resize()` defaults to BICUBIC, not BILINEAR.** Training and
   host-side preprocessing must agree. The canonical function is
   `load_image()` in `src/prepare_data.py` — use it everywhere.

10. **54 images have real alpha transparency.** Dropping the alpha channel
    leaves black holes. Composite over **white**.

11. **Input quantisation is exactly `(int8)(pixel - 128)`.** The converter
    derives `scale = 1.0, zero_point = -128` because the Keras graph carries its
    own `Rescaling(1/255)`. If you change the model, re-verify this holds — the
    firmware depends on it and has an assert in `export_c_model.py`.

12. **`west build` requires Python 3.12+.** If it fails on
    `relative_to(..., walk_up=True)`, invoke cmake directly:
    `cmake -B build -GNinja -DBOARD=qemu_x86 src/qemu_app && ninja -C build`.

---

## 5. The two residual errors

Both misclassifications in the entire validation set are **class 83 ↔ 84**
(Steep descent / Steep ascent), in both directions. These signs are mirror
images, and the organisers supplied those classes pre-augmented *with*
horizontal flips — so a flipped descent is pixel-identical to an ascent. Some
label assignments in that pair are intrinsically ambiguous.

Both errors reported **57.17 % confidence**, the first percentile. Every image
above 60 % confidence is classified correctly.

Treat 362/364 as close to the practical ceiling on this labelling. Chasing it
with a bigger model is a bad trade against the RAM tie-breaker.

---

## 6. The gate — run this before claiming any result

```
python scripts/verify_all.py
```

It builds for both boards, checks bit-exactness against the TFLite reference,
runs all 364 validation images over the real UART on QEMU, and prints a
pass/fail table with RAM, flash and timing.

**Rules:**
- **Never report a number you did not measure with this script.** No estimates,
  no "should be roughly". If you did not run it, say so.
- A change that lowers accuracy, raises peak RAM, or breaks bit-exactness is
  **reverted**, not explained away.
- If the gate fails, fix it before starting anything else.

---

## 7. Working style

- **Measure, then decide.** Every architectural claim in this project came from
  a control experiment. Follow that: build the variant, run the gate, compare.
- **State uncertainty plainly.** If something is untested — for example anything
  about real ESP32-S3 hardware, which nobody here can flash — say so rather
  than implying it was verified.
- **Prefer reverting to patching.** The baseline is verified; speculative fixes
  layered on a broken change are how projects lose a working state.
- Comment *why*, not *what*. The interesting comments in this codebase explain
  non-obvious decisions (the two rounding rules, polled RX). Match that.
- Do not add dependencies to the firmware. It is deliberately
  dependency-free — no TensorFlow Lite Micro, no CMSIS-NN, no vendor blobs.
