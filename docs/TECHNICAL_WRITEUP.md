# AI Arena 2026 — Track 1: Traffic Sign Recognition

Embedded traffic-sign classifier: 17 classes, 48×48×3 RGB input, int8 inference
over UART, running on Zephyr RTOS on QEMU and buildable for ESP32-S3 WROOM.

---

## 1. Results

Measured **over the wire on QEMU**, driving the real UART protocol for every
one of the 364 validation images — not simulated in Python.

| Payload mode | Accuracy | Mean inference |
|---|---|---|
| 48×48×3 RGB (challenge guide) | **99.73 %** (363/364) | 33.8 ms |
| 64×64×3 RGB (`send_image.py` default) | **99.73 %** (363/364) | 34.1 ms |
| 48×48 grayscale (`send_image.py -g`) | **99.45 %** (362/364) | 34.3 ms |

| Metric | Value |
|---|---|
| Parameters | 62,145 |
| Float32 TFLite | 251,456 B |
| **Int8 TFLite** | **72,408 B** (3.47× smaller) |
| Weights in firmware flash | 63,740 B |
| **Peak inference RAM** | **9,984 B** (two static buffers + row scratch, no heap) |
| Accuracy float32 → int8 | 99.73 % → 99.73 % — **0 of 364 predictions changed** |
| Inference, QEMU (guest-measured) | **33.8 ms**, from 55.3 ms |
| Inference, native x86 -O3 | **1.52 ms** (i7-1365U); fusion A/B speedup **2.24×** |

ESP32-S3 footprint (`esp32s3_devkitc/esp32s3/procpu`):

| Region | Used | Size | % |
|---|---|---|---|
| FLASH | 199,684 B | 8,388,352 B | 2.38 % |
| irom0_0_seg | 20,432 B | 32 MB | 0.06 % |
| dram0_0_seg | 75,184 B | 399,108 B | 18.84 % |
| iram0_0_seg | 37,828 B | 415,492 B | 9.10 % |

---

## 2. Approach

**Model.** A four-block int8 CNN, deliberately small:

```
48×48×3 → conv3×3/s2 ×16 → conv3×3 ×32 → maxpool2
        → conv3×3 ×64 → maxpool2 → conv3×3 ×64
        → avgpool6 → dense 17 (logits)
```

The stride-2 first convolution is the key architectural choice. The
conventional alternative (stride-1 conv followed by pooling) was trained as a
control: it scored *lower* (98.08 % vs 98.35 % at the time of comparison) while
using **4× more peak activation memory**. A 2.2×-wider variant was also trained
and gained exactly one additional correct image out of 364 for 76k extra
parameters — rejected, since the tie-breaker is embedded efficiency.

Only five operator types are used — `CONV_2D`, `MAX_POOL_2D`,
`AVERAGE_POOL_2D`, `RESHAPE`, `FULLY_CONNECTED` — chosen so the inference
runtime stays small and portable. Softmax is deliberately absent: the device
needs only `argmax`, and softmax is monotonic.

**Two observations about the dataset drove the training recipe.**

*The dataset is two different datasets.* Classes 0, 2–7, 24, 43 and 87 are real
photographs at 80–280 px. Classes 69, 70, 72, 83, 84, 85 and 86 are 32×32 tight
crops with an alpha channel, pre-augmented with horizontal flips, some reduced
to 1-bit black and white. A network can separate those two groups on image
statistics alone — upscaling blur, flat colour — without ever looking at the
sign. The `random_resolution` augmentation (randomly degrade to 16–48 px and
restore) exists specifically to destroy that shortcut.

*Colour was a crutch.* `send_image.py` can send a grayscale payload. The first
model scored 74.18 % on grayscale input against 98.35 % on colour — it had
learned to lean on red rings versus blue circles. Adding random grayscale
augmentation (p = 0.25) fixed the grayscale path **and improved colour
accuracy**, because it forced the network onto shape and glyph features:

| | before | after |
|---|---|---|
| RGB | 98.35 % | **99.45 %** |
| Grayscale | 74.18 % | **99.18 %** |

Horizontal flips are *not* used: many classes are directional (24 "Go Right",
43 "Go right or straight", 83 vs 84 descent/ascent), so mirroring would
manufacture wrong labels.

---

## 3. Optimization and quantization

Standard TensorFlow Lite **full-integer post-training quantization** with a
400-image representative dataset; int8 in, int8 out, per-channel weight scales.

Because the Keras graph carries its own `Rescaling(1/255)` layer, the network
input is float [0, 255], so the converter derives `input scale = 1.0,
zero_point = -128`. On-device, converting a received pixel to the model's int8
input is therefore exactly `(int8)(pixel - 128)` — no floating point anywhere on
the input path.

**The runtime is a hand-written C kernel, not TensorFlow Lite Micro.** It is
~200 lines with no external dependency, which removes the Zephyr module-fetch
and version-drift risk entirely, and gives the smallest RAM and flash figures.
The `.tflite` file is still the source of truth: `export_c_model.py` reads it
and emits the weights and per-channel multipliers as C arrays.

**Correctness was gated on bit-exactness, not on accuracy matching.** The C
runtime was verified against the TFLite reference interpreter over all 364
validation images × 17 output values. This exposed something subtle worth
recording:

> TensorFlow Lite does **not** use one rounding rule. `CONV_2D` uses gemmlowp's
> two-step `SaturatingRoundingDoublingHighMul` + `RoundingDivideByPOT`, while
> per-channel `FULLY_CONNECTED` uses a higher-precision path. Applying either
> rule uniformly produces 1-LSB errors on roughly 4 % of images. Both are
> reproduced exactly in `tsr_model.c`.

Result: **argmax agrees with the TFLite reference on 364/364 images**, and the
score vectors are identical on 363/364 (one convolution output, in one image,
differs by 1 LSB — it does not change any prediction). This is why the accuracy
measured on QEMU equals the accuracy measured in Python rather than merely
approximating it.

---

## 3b. Speed and memory optimisation

Both were done after the model was frozen, so accuracy is untouched and the
runtime is still bit-exact with the TFLite reference at 364/364.

| | before | after |
|---|---|---|
| **Peak inference RAM** | 27,648 B | **9,984 B** (−64 %) |
| QEMU, guest-measured median | 55.26 ms | **33.77 ms** (1.64×) |
| Native x86, `-O3` | 3.41 ms | **1.43 ms** (2.38×) |
| Accuracy | 99.45 % | 99.45 % (unchanged) |

Four changes, each measured independently.

**1. Convolution fused with the max pool that follows it — this is the first
RAM win.** Peak memory was set by the second convolution's 24×24×32 output: an
18 KB tensor that exists only so the 2×2 pool can immediately discard three
quarters of it. Fused, the convolution is computed two rows at a time into a
1.5 KB scratch and pooled on the fly, so the wide tensor is never allocated.
27,648 B → 15,360 B. `export_c_model.py` sizes the buffers with the same fusion
rule the runtime applies, so the generated constants and the execution schedule
cannot drift apart.

**1b. The window extended one layer further: conv → conv → pool in a single
pass.** After the pair fusion the largest thing left in the arena was the
*first* convolution's 24×24×16 output — 9,216 B, held only so the second
convolution could read it. Emitting one pooled row needs two rows of the second
convolution, which need four consecutive rows of the first, so four rows are
all that has to exist: 1,536 B instead of 9,216 B. Consecutive pooled rows
share two of those four, so the window slides — rows 2 and 3 become rows 0 and
1 — and every row of the first convolution is still computed exactly once. The
cost is a 768-byte copy per pooled row.

15,360 B → **9,984 B**, and natively it is if anything slightly *faster*
(1.428 ms against 1.447 ms, better locality). The check that matters: with the
triple fusion switched off and the buffers restored, the runtime emits
byte-identical output — all 364 × 17 int8 logits, not merely the same argmax.
Only the order of evaluation changed.

**2. Activations held as uint8 rather than int8.** Every tensor in this graph
has `zero_point = -128`, so the dequantising subtraction `(x - zero_point)`
that a textbook int8 kernel performs on every multiply-accumulate becomes
exactly the stored byte and disappears from the inner loop. The output side
cancels identically. This is a representation change, not an approximation.
Four output channels are also accumulated per pass, so each input byte is
loaded once and reused four times.

**3. The kernel is compiled `-O3`; Zephyr defaults the image to `-Os`.** Only
`tsr_model.c` gets the override, so the code-size cost is confined to one file
(+2,688 B, in a 32 MB region). On x86 this is worth more than the kernel
rewrite because the loop vectorises; on `qemu_x86` Zephyr passes `-mno-sse`, so
vectorisation is unavailable there and the gain is from unrolling alone. Expect
the Xtensa target to behave like the latter.

Fusion is speed-neutral at `-O3` natively and costs a few percent on QEMU
where SSE is disabled. It was kept regardless: the stated tie-breaker order
puts lowest RAM ahead of fastest inference, and 64 % is not a marginal call.

The ESP32-S3 figures in an earlier revision of this table (`dram0_0_seg`
87,472 → 75,184 B) were measured against the pair-fused arena. The arena is
three statically sized `uint8` arrays in `.bss`, so the triple fusion moves
`dram0_0_seg` down by exactly the 5,376 B the arena lost — but that is linker
arithmetic, not a measurement, and it is labelled as such until an ESP32-S3
build is re-run.

### Work per image, independent of any host

QEMU inference time depends on the machine QEMU runs on — 33.8 ms here, 74.6 ms
on a laptop — so it is not by itself a comparable number. The work the network
does is fixed:

| layer | output | kernel | MAC |
|---|---|---|---|
| conv 1 | 24×24×16 | 3×3×3 | 248,832 |
| conv 2 | 24×24×32 | 3×3×16 | 2,654,208 |
| conv 3 | 12×12×64 | 3×3×32 | 2,654,208 |
| conv 4 | 6×6×64 | 3×3×64 | 1,327,104 |
| fc | 17 | 17×64 | 1,088 |
| **total** | | | **6,885,440 MAC** (13.77 MOPs) |

6.9 M int8 multiply-accumulates per image, no floating point anywhere on the
device. Quote this alongside a measured time and name the host; a time on its
own is not reproducible.

### A note on how inference time is measured

`send_image.py` times inference as the wall gap between the `RECEIVED` line and
the next line it reads from the port. That method breaks down once inference is
fast: both lines land in one TCP segment and the measured gap collapses to
socket granularity. It reports 0.1 ms for a computation that takes ~34 ms.

The figures above are guest-measured with `k_cycle_get_32()` around the
inference call, printed on a `TIMING` line **after** `PREDICTION` — where it
cannot contaminate the host's measurement — and compiled out of the shipping
image entirely (`-DCONFIG_TSR_REPORT_TIMING`). They were cross-checked against a
wall-clock A/B of two firmware builds over identical transfers
(`test/ab_timing.py`).

---

## 3c. Robustness to a domain-shifted test set

**This is the change with the largest expected effect on the hidden-test score,
and it is invisible in the supplied validation split.**

Seven of the seventeen classes — 69, 70, 72, 83, 84, 85, 86 — are supplied
*only* as 32×32 pre-augmented crops with no background. The other ten are
ordinary photographs. If the hidden test set presents those seven as
photographs, the validation score gives no warning at all.

To measure the exposure, `src/stress_set.py` builds a proxy: the pictogram
classes composited onto real photographic backgrounds, then put through what a
camera does — perspective, focus blur, JPEG compression, lighting, sensor noise.
The model trained on the supplied data alone:

| | accuracy |
|---|---|
| Validation, as supplied | 99.45 % |
| Stress set — the 10 photo classes | 97.30 % |
| **Stress set — the 7 pictogram classes** | **59.15 %** |
| Stress set overall | **82.42 %** |

A 17-point exposure on a 30 %-weighted category.

`src/augment_domain.py` closes it by synthesising the missing view for training —
pictogram signs composited onto backgrounds drawn **only from training images**,
with deliberately different parameter ranges from the stress generator, so the
evaluation is not simply measuring whether the model memorised one
transformation. Three synthetic views per pictogram training image.

| | in-distribution | **domain-hardened (shipped)** |
|---|---|---|
| Validation | 99.45 % | 98.90 % |
| Stress overall | 82.42 % | **98.08 %** |
| Stress: photo classes | 97.30 % | **99.10 %** |
| Stress: pictogram classes | 59.15 % | **96.48 %** |

**+37 points on the exposed classes for two validation images.** Those two
images are worth looking at: all four errors in the shipped model are the same
83 ↔ 84 confusion — steep descent against steep ascent, mirror images of each
other, supplied pre-augmented *with* horizontal flips so some of the labels in
that pair are genuinely ambiguous. Fifteen of seventeen classes remain perfect.
The photo classes also improved without being targeted, which is evidence of
real robustness rather than transformation memorisation.

### Why this model was shipped

Setting the two outcomes against each other, the in-distribution model wins by
0.55 points if the hidden set matches validation, and loses by 15.7 points if it
does not. Break-even sits at a **3.6 % chance** of a domain-shifted test set.
The downside is asymmetric by roughly 30×.

A heavier augmentation setting (six synthetic views) was also trained and was
strictly worse than three on every stress measure, so three was kept.

The in-distribution model is preserved under `model/alternate_in_distribution/`
with its numbers, if you learn the test set is generated the same way as
validation and want to trade the insurance back for those two images.

---

## 3d. Low-light robustness

A traffic sign classifier that only works in daylight is not much use in a car,
and the supplied data is entirely daytime. `src/lowlight_set.py` measures the
exposure two ways at **matched mean brightness**, so the only difference between
them is noise:

- **blend to black** — `I × (1 − a)`, a straight alpha blend with a black image.
- **sensor model** — linearise out of gamma, cut the exposure, apply Poisson
  shot noise and Gaussian read noise, re-encode.

The second is the honest one. Darkness does not simply scale a picture down;
fewer photons means the signal-to-noise ratio collapses, and it is the noise
that destroys a classifier. The measurement shows exactly that — the two agree
until the noise starts to dominate, then diverge sharply:

| darkness | mean pixel | blend | sensor model | **after hardening** |
|---|---|---|---|---|
| none | 152 | 99.73 % | 99.73 % | 99.73 % |
| 0.30 | 88.8 | 98.90 % | 98.90 % | 99.73 % |
| 0.50 | 63.5 | 98.90 % | 98.90 % | 99.45 % |
| 0.70 | 37.8 | 97.25 % | 97.25 % | **99.18 %** |
| 0.85 | 18.7 | 93.68 % | 91.76 % | **96.98 %** |
| 0.95 | 5.9 | 37.91 % | 23.63 % | **62.09 %** |

At 0.95 the mean pixel value is 5.9 out of 255 — effectively a black frame that
a person could not read either, so that collapse is not held against the model.

The sensor model is applied to 22 % of training samples. It costs nothing at
inference: same weights, same arena, same speed.

**It also improved ordinary accuracy.** Validation went from 98.90 % to
**99.73 %** — 363 of 364, the best figure the project reached, with the single
remaining error being the 83/84 mirror pair again. Noise augmentation acting as
a regulariser on a 1,460-image dataset is a known effect, but the size of it
here was not expected.

### The trade that was accepted

Against the domain-hardened model without night augmentation:

| | domain only | **shipped (domain + night)** |
|---|---|---|
| Validation | 98.90 % | **99.73 %** |
| Photographic stress set | **98.08 %** | 96.70 % |
| Night, 0.85 darkness | 91.76 % | **96.98 %** |

The shipped model gives up 1.4 points on the stress proxy to gain 0.83 on real
validation data and 5 points on night. Validation is the only genuinely held-out
set here — the stress and night sets are both constructions of ours, and
selecting a model on a proxy you built yourself is a way to fool yourself. The
model that wins on the real holdout was chosen.

It also **dominates the original in-distribution model on every axis**,
including validation (99.73 % against 99.45 %), so that comparison is no longer
a trade at all.

---

## 4. Robustness: payload size handling

The organisers have confirmed **48×48×3 is final**, and the current
`send_image.py` sends exactly that. An earlier revision of the same script
defaulted to 64×64×3 and offered a 48×48 grayscale mode, so the firmware still
dispatches on the received byte count. This costs 5,376 bytes of buffer and
removes any dependence on which revision of the host script is run:

| Bytes | Interpreted as | Handling |
|---|---|---|
| 6,912 | 48×48×3 RGB | **used directly — the confirmed format** |
| 12,288 | 64×64×3 RGB | on-device integer bilinear downscale to 48×48 |
| 2,304 | 48×48 grayscale | replicated to three channels |
| anything else | — | `ERROR unexpected payload size <n>` |

All three are validated end-to-end on QEMU (table in §1).

---

## 5. UART protocol

115200 baud, 8N1, no flow control.

```
board → host   READY
host  → board  ASCII hex, 32 bytes (64 chars) per line
host  → board  STOP
board → host   RECEIVED <n> bytes
board → host   PREDICTION class=<id> label=<name> confidence=<p>%
board → host   READY                       (re-armed for the next image)
```

Example:

```
RECEIVED 6912 bytes
PREDICTION class=69 label=Height limit confidence=99.71%
READY
```

`class=` carries the **official** class ID (0, 2–7, 24, 43, 69, 70, 72, 83–87),
not the 0–16 array index, as the challenge guide requires. Labels are taken
verbatim from the supplied `new_labels.csv`. On failure the board emits a line
beginning with `ERROR` so the host never has to wait out its 30-second timeout.

Nothing is printed between `RECEIVED` and `PREDICTION`: `send_image.py` measures
inference time as the gap between those two lines, so any intervening output
would be misreported as inference time.

### READY is re-armed while idle

After answering, the board announces `READY` and then re-announces it every
three seconds for as long as it sits idle. This matters because `send_image.py`
opens and closes the serial port once per image: without re-arming, the second
and subsequent invocations would find a board that had already sent its `READY`
and would sit through their full 30-second handshake timeout before the board's
own receive timeout fired. Re-arming makes back-to-back runs start immediately.

A re-armed `READY` can never land between `RECEIVED` and `PREDICTION` — re-arming
only happens before the first byte of a transfer arrives, and serial delivery is
in order.

### RX is polled, not interrupt-driven

This is deliberate. Bytes stay in the hardware FIFO until the receive loop takes
them. An ISR draining into a ring buffer does the opposite — it empties the FIFO
as fast as the line delivers and the overflow moves into the ring buffer, where
it is discarded with no signal to anyone. That was observed during development
and is why this is polled.

**What a full FIFO actually achieves differs by transport, and the distinction
is worth stating precisely.**

On **QEMU** the UART is a socket chardev. A full guest FIFO stops QEMU reading
the TCP socket, the receive window closes, and the host's `send()` blocks. That
is genuine end-to-end backpressure — but it is *TCP* flow control, not UART flow
control. Measured: a 56,180-byte burst, four images back to back with no
handshake at all, written in one call, produced four `RECEIVED 6912 bytes` and
four predictions with nothing dropped.

On **real hardware it achieves nothing on the wire.** Hardware flow control is
not enabled in this application — there is no `hw-flow-control` property on the
UART node and no RTS/CTS Kconfig — so a full FIFO asserts no signal and bytes
arriving after it fills are simply lost. The FIFO is 16 bytes on the ns16550
QEMU uses and 128 bytes on ESP32-S3; at 115200 baud a byte lands every 86 µs, so
roughly 390 bytes would arrive during a 33 ms inference if a host kept sending
through it.

So on hardware the protection is the **protocol, not the peripheral**: `READY` is
emitted only when the receive loop is ready, and the organisers' `send_image.py`
waits for it before sending. A host that ignored the handshake and streamed
continuously would overrun a real UART. Enabling RTS/CTS would close that gap,
but it requires wiring the evaluation harness may not carry and could not be
tested here, so it is documented rather than assumed.

While the loop is running it is not the bottleneck: decoding one hex character
costs a few instructions against 86 µs per character at 115200 baud, three
orders of magnitude of headroom. The exposure is confined to the inference
window.

### Confidence

The organisers have clarified that the reported confidence **is** the accuracy
criterion — *"Accuracy is confidence … with how much confidence does your model
predict the given image"* — and that it should be given as a percentage
(`90%`, not `0.90`). The board reports it as `confidence=99.87%`.

The graph deliberately ends at the logits: the device needs only `argmax`, and
softmax is monotonic, so it changes nothing about which class wins. Softmax is
applied on the output side only, over 17 values, purely to produce this figure.
The int8 inference path itself stays integer-only.

It is informative rather than decorative:

| | |
|---|---|
| Expected calibration error (10 equal-width bins) | **1.12 pp** |
| Mean confidence vs accuracy | 98.60 % vs 99.73 % — slightly under-confident |
| Mean confidence when correct | **98.64 %** |
| Confidence on the single error | **82.97 %** |
| Lowest confidence on any image | 50.72 % |
| Accuracy above 90 % confidence | **100 %** (347/364 retained) |

The one image the model gets wrong is also the one it is least sure about among
its high-scoring predictions: thresholding at 90 % discards 17 images and leaves
347 that are **all** correct. That is the property worth having — the number the
board prints carries information about whether to trust the label.

**Temperature scaling was measured and deliberately rejected.** Since the model
is slightly under-confident, post-hoc temperature scaling would raise the
reported figure — and since confidence is the scored quantity, that is a real
temptation. The measured trade-off (`src/calibration_check.py` reproduces it):

| T | mean confidence | ECE | confidence on the single error |
|---|---|---|---|
| 1.000 (shipped) | 98.60 % | 1.12 pp | 82.97 % |
| 0.472 (NLL-optimal) | 99.40 % | 0.63 pp | **96.6 %** |

It buys 0.8 pp of mean confidence and halves the ECE, but pushes the *wrong*
prediction from 83 % to 96.6 % — indistinguishable from a correct one. It would
optimise the scored number by destroying the only thing that number is good
for. It would also be fitted on the same 364 images
already used for checkpoint selection, with no clean holdout left to validate
it. The shipped model reports its raw softmax probability, unmodified.

## Prebuilt executables

Each Zephyr application ships with its own binaries and a README recording
sizes, sha256 sums, the memory report and the exact command to flash or run it:

- `src/esp32_app/binaries/` — `zephyr.bin` and `zephyr.elf` for
  **ESP32-S3-DevKitC-1 N16R8**, the evaluation target. `CONFIG_ESP_SIMPLE_BOOT`,
  so it is one binary at offset `0x0`.
- `src/qemu_app/binaries/` — `zephyr.elf` for `qemu_x86`.

The two applications share byte-identical sources; only the build configuration
differs (the S3 app sets a 16 MB flash size in Kconfig and in a devicetree
overlay, because esptool reads the image-header flash size from the devicetree
node and the in-tree board declares 8 MB). `scripts/verify_all.py` sha256-checks
every shared file and fails if they diverge, so a result measured under QEMU
stays evidence about the board.

## Hardware bring-up

The submitted target is ESP32-S3 WROOM. The image builds and links for it and
its footprint is known, but it has never been flashed. What *has* been run on
silicon is a plain **ESP32-D0WD-V3** (Xtensa LX6, 4 MB flash) used as a bench
proxy: same Xtensa ABI and alignment rules, same Zephyr UART driver, same `-O3`
kernel, different SoC.

That distinction is worth keeping straight. The two SoCs also differ in where
the image lives — the ESP32 ROM bootloader reads it from flash offset `0x1000`,
the S3 from `0x0` (`partitions_0x1000_amp.dtsi` against `partitions_0x0_amp.dtsi`
in the board devicetrees). Flashing to the wrong one produces a board that
programs successfully and then does nothing.

On the ESP32 the firmware boots and answers:

```
TSR 2026.08.28-t3f | 48x48x3 | 17 classes | arena 9984 B | val 99.73%
READY
RECEIVED 6912 bytes
PREDICTION class=69 label=Height limit confidence=100.00%
```

All 364 validation images were then sent over the physical UART at 115200 8N1
(`scripts/serial_eval.py`):

```
images    : 364
accuracy  : 0.9973 (363/364)
confidence: mean 98.60%  min 50.72%
wall      : 624.5s          (1.72 s/image, dominated by the serial link)
  wrong: image 281 expected class 84, got 83 at 82.97%
```

**The same image, misread the same way.** Native x86 int8 gets exactly one image
wrong — index 281, class 84 read as 83 — and so does QEMU, and so does the
board. That is a stronger result than matching accuracy: three unrelated
execution environments (x86-64, emulated i386, Xtensa LX6) produce identical
predictions image for image, which is what bit-exact arithmetic is supposed to
buy and is rarely demonstrated end to end.

The lowest confidence in the whole sweep, 50.72 %, belongs to a *correct*
prediction on the other half of the 83/84 mirror pair. The model is unsure
exactly where it should be.

### The link drops a byte occasionally, and that is not a firmware fault

One frame was lost during bring-up: the board answered `ERROR odd number of hex
digits` and refused it rather than classifying corrupt data. That is the
behaviour worth having, and it is why the error path exists.

It has not recurred. The full 364-image sweep re-sent nothing, so the honest
summary is **one observed frame error, in a run where a terminal emulator still
held the port open** — not a measured error rate. There is no hardware flow
control on this link, so loss is possible in principle; this evidence does not
establish how often.

`scripts/serial_eval.py` re-sends a rejected frame (`--retries`, default 3) and
reports how many were re-sent, so a sweep produces a clean accuracy figure and
a separate honest count of link errors. `--pace` adds a delay between chunks
for a link that drops bytes often.

The organisers' `send_image.py` never sees this, by accident: after each
32-byte chunk it performs a blocking 0.1 s read for a line that never arrives,
which paces the transfer at roughly 21 s per image and leaves the receiver far
ahead of the sender. That also means the "upload time" it prints is a property
of the script, not of the board.

## What was tried and rejected

Recorded because a measured failure is worth more than an untried idea.

### Ensemble knowledge distillation — rejected

The most attractive remaining idea, because it is free at inference: distil an
ImageNet-pretrained teacher into the *same* student architecture and the arena,
model size and speed are all unchanged, only the weight values differ. With
1,460 training images, pretrained features plausibly carry information a
40k-parameter network cannot learn from the data alone.

Two MobileNetV2 backbones were fine-tuned on Train only (Validation never shown
to them), 24 augmented views of the training set were materialised, and the
ensemble's softened predictions were distilled into the student
(`src/distill.py`, `src/distill_stage2.py`; T = 4, α = 0.75).

It failed, and the reason is visible before the student is trained:

| model | validation | agreement with the label on the augmented views |
|---|---|---|
| MobileNetV2 1.00 @128 | 98.35 % | 90.00 % |
| MobileNetV2 0.75 @96 | 91.76 % | — |
| **shipped student (40 k params)** | **99.73 %** | **98.03 %** |
| ensemble (shipped + best teacher) | — | 98.11 % |

The teachers are *worse than the student*. The ensemble target is 98.11 %
against the student's own 98.03 % — 0.08 points of headroom — and the KD loss
then spends three quarters of its weight forcing the student to reproduce it.
The result loses on every axis measured:

| | shipped | distilled |
|---|---|---|
| Validation | **99.73 %** | 99.45 % |
| Photographic stress set | **96.98 %** | 95.60 % |
| Night, 0.70 | **99.18 %** | 98.90 % |
| Night, 0.85 | **96.98 %** | 95.60 % |
| Night, 0.95 | **64.01 %** | 55.77 % |

Distillation transfers what a teacher knows and the student does not. Here
there is no such surplus: a 1.4 M-parameter ImageNet model fine-tuned on 1,460
stereotyped 48×48 traffic signs does not beat a small network built for exactly
that data. The scripts are kept so the result is reproducible rather than
merely asserted.

### Hand-written Xtensa SIMD — not attempted, deliberately

ESP32-S3 has 128-bit PIE instructions that could plausibly give the int8
convolution several times its current throughput. It was rejected on risk:
Phase 2 flashes the submitted code onto hardware nobody here can test on, and
SIMD kernels that have never executed on the part are a way to score zero
rather than a way to win a tie-break. It would also cost the bit-exactness
guarantee against the TFLite reference, which is the strongest correctness
claim in this submission. Inference time is in any case the *second*
tie-breaker; the effort went into the first one instead, where the arena
dropped 64 % for no accuracy or speed cost at all.

### Verified back-to-back operation

The evaluators run multiple images in sequence, expecting `READY` after every
prediction. Verified on QEMU:

- 40 images back-to-back on a single connection: 40/40 correct, the exact
  sequence `RECEIVED → PREDICTION → READY` every time, no stray lines.
- Adversarial check — 12 images sent with *no* handshake at all: still 12/12,
  every `RECEIVED 6912 bytes`, nothing dropped.

The second case passes for a reason specific to QEMU: the UART is a socket
chardev, so once the guest FIFO fills QEMU stops reading the TCP socket and the
host blocks. That is TCP flow control, not UART flow control — see *RX is polled,
not interrupt-driven* above. On physical hardware, where RTS/CTS is not enabled,
a host that ignored `READY` and kept sending would overrun the FIFO during the
~34 ms the board spends replying and running inference — roughly 390 bytes
arrive in that window at 115200 baud, against a 16- or 128-byte FIFO. The
handshake makes that unreachable in normal operation, and the evaluation host
does wait for `READY`, but the guarantee comes from the protocol rather than
from the peripheral.

## 6. Repository layout

```
src/
  prepare_data.py       canonical preprocessing (shared by training and host)
  model_training.py     model definition + augmentation + training
  model_optimize.py     int8 post-training quantization + accuracy comparison
  export_c_model.py     .tflite -> C weight arrays + layer table
  extract_params.py     flatbuffer parameter extraction
  int8_engine.py        vectorised bit-exact NumPy reference
  tflite_int8_ref.py    scalar reference kernels (gemmlowp arithmetic)
  qemu_app/             Zephyr application
    src/main.c          UART protocol state machine
    src/tsr_model.c     int8 CNN runtime (no dependencies)
    src/tsr_model_data.c  generated weights
    src/tsr_image.c     payload normalisation and resizing
model/
  trained_model.keras            float32 model
  model_float32.tflite           unquantized reference
  quantized_model_int8.tflite    submitted model
  quantization_report.json
scripts/
  send_image.py         official host script (unmodified, for hardware)
  send_image_qemu.py    same protocol over QEMU's TCP-backed UART
  qemu_uart_eval.py     full validation set over UART, scores accuracy
test/
  host_test.c           native harness used for bit-exactness verification
```

---

## 7. Reproducing

### Build and run on QEMU

```bash
west build -p always -b qemu_x86 src/qemu_app
west build -t run                      # console only
```

To drive it with the real protocol, QEMU's UART is bound to a TCP socket
(QEMU has no COM port for `send_image.py` to open):

```bash
python scripts/send_image_qemu.py build/zephyr/zephyr.elf samples/sample_class69.png
```

Expected:

```
READY
RECEIVED 6912 bytes
PREDICTION class=69 label=Height limit confidence=99.71%
Model prediction done in: 0.009s
--> class 69: Height limit
```

Score the entire validation set over UART:

```bash
python scripts/qemu_uart_eval.py --mode rgb48    # or rgb64 / gray48
```

### Build for ESP32-S3

```bash
west build -p always -b esp32s3_devkitc/esp32s3/procpu src/qemu_app
west flash
python scripts/send_image.py COM7 sign.png --baud 115200
```

### Retrain from scratch

```bash
python src/prepare_data.py  <dataset_dir> out/data.npz
python src/model_training.py --variant A --width 1.0 --epochs 220
python src/model_optimize.py --keras model/best_A_w1.0.keras
python src/export_c_model.py --outdir src/qemu_app/src
```

---

## 8. Honest notes

- Validation accuracy is mildly optimistic: the checkpoint was selected *by*
  validation accuracy across 220 epochs, so that figure is a selected maximum
  rather than a clean holdout. The hidden test set is the real measure.
- The training set is small — 1,460 images, ~86 per class. Augmentation
  strategy matters more than architecture at this scale, which is what the
  experiments above showed.
- Both residual errors are the steep-descent / steep-ascent pair, which are
  mirror images of each other and were supplied pre-augmented *with* horizontal
  flips — so some label assignments in that pair are intrinsically ambiguous.
- The seven pictogram classes are supplied only as 32×32 pre-augmented crops.
  If the hidden test set presents those classes as full-resolution photographs
  instead, accuracy on them will be lower than reported here. The resolution
  augmentation is the mitigation, but it cannot fully substitute for real
  photographs of those signs.
