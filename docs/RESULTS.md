# Results, and the command that reproduces each one

Every figure below was measured on the tree it appears in. Nothing here is an
estimate; where a number could not be re-measured it is marked as such.

Re-verified 2026-09-02 with `model/quantized_model_int8.tflite` and the C
runtime in `src/qemu_app/src/`.

## Accuracy

| | value | how |
|---|---|---|
| Validation accuracy, int8 TFLite | **99.73 %** (363/364) | `scripts/verify_all.py` |
| Validation accuracy, C runtime | **99.73 %** (363/364) | `test/host_test.c` |
| Validation accuracy, over UART on QEMU | **99.73 %** (363/364) | `scripts/qemu_uart_eval.py --mode rgb48` |
| Validation accuracy, over UART on hardware | **99.73 %** (363/364) | `scripts/serial_eval.py` |
| Validation accuracy, float32 TFLite | **99.73 %** (363/364) | `scripts/compare_models.py` |
| 64×64 RGB payload | 99.73 % | `qemu_uart_eval.py --mode rgb64` |
| 48×48 grayscale payload | 99.45 % (362/364) | `qemu_uart_eval.py --mode gray48` |

**The single error is the same image everywhere.** Index 281, true class 84
(*steep ascent*), predicted 83 (*steep descent*) — the mirror pair — at 82.97 %
confidence, identically on native x86, on QEMU, and on hardware. Sixteen of the
seventeen classes have zero errors.

## Quantisation cost

| | |
|---|---|
| float32 TFLite | 99.73 % (363/364) |
| int8 TFLite | 99.73 % (363/364) |
| Predictions changed | **0 of 364** |

Full-integer post-training quantisation with per-channel weight scales and a
400-image representative dataset. The cost was not "small"; it was nil on this
set.

## Bit-exactness of the C runtime

Compared against TFLite's **reference** int8 kernels
(`OpResolverType.BUILTIN_REF` — the XNNPACK delegate does not reproduce
reference int8 arithmetic, so comparing against it proves nothing):

| | |
|---|---|
| Identical argmax | **364 / 364** |
| Identical complete 17-value score vectors | **362 / 364** |
| Maximum deviation where they differ | **1 LSB**, images 262 and 285, argmax unchanged |

`scripts/verify_all.py` gates on argmax agreement and fails the build if it ever
drops below 364/364.

## Memory and size

| | |
|---|---|
| Peak inference RAM (arena) | **9,984 B** — `tsr_buf_a` 4,608 + `tsr_buf_b` 2,304 + `tsr_scratch` 3,072 |
| Heap used | none |
| Parameters | 62,145 |
| int8 TFLite file | 72,408 B |
| float32 TFLite file | 251,456 B |
| Weights in firmware flash | 63,740 B |
| MACs per inference | 6,885,440 |
| Runtime source | 493 lines, **337 lines of code** |

ESP32-S3 (`esp32s3_devkitc/esp32s3/procpu`, DevKitC-1 N16R8):

| segment | bytes | capacity | used |
|---|---:|---:|---:|
| FLASH | 199,700 | 16,776,960 | 1.19 % |
| `iram0_0_seg` | 37,828 | 415,492 | 9.10 % |
| `dram0_0_seg` | 69,808 | 399,108 | 17.49 % |

## Fusion

| schedule | arena | note |
|---|---:|---|
| unfused | 27,648 B | every intermediate materialised |
| conv + pool | 15,360 B | pre-pool tensor never written |
| conv + conv + pool | **9,984 B** | shipped |

- Code cost: **+2,688 B** of IROM.
- Accuracy change: **0** — byte-identical on all 364 × 17 logits.
- Speedup: **2.24×** native `-O3`, **1.64×** under QEMU.

## Confidence calibration

The organisers confirmed the reported confidence *is* the accuracy criterion, so
it is worth stating precisely what it is: softmax over the dequantised int8
logits, output side only, uncapped and unscaled. The integer inference path is
untouched by it.

| | |
|---|---|
| Expected calibration error, 10 equal-width bins | **1.12 pp** |
| Mean confidence | 98.60 % (accuracy 99.73 % — slightly under-confident) |
| Mean confidence when correct | 98.64 % |
| Confidence on the single error | 82.97 % |
| Lowest confidence on any image | 50.72 % |
| Accuracy above 90 % confidence | **100 %**, 347/364 retained |
| Accuracy above 60 % confidence | 99.72 %, 359/364 retained |

**Temperature scaling was measured and rejected.**

| T | mean confidence | ECE | confidence on the error |
|---|---|---|---|
| 1.000 (shipped) | 98.60 % | 1.12 pp | 82.97 % |
| 0.472 (NLL-optimal) | 99.40 % | 0.63 pp | **96.6 %** |

It improves both scored quantities and makes the wrong prediction
indistinguishable from a right one. It would also be fitted on the same 364
images used for checkpoint selection. Reproduce with `src/calibration_check.py`.

## Robustness

Stress set — pictogram classes composited onto photographic backgrounds with
perspective, blur, JPEG and lighting variation (`src/stress_set.py`):

| model | validation | stress overall | stress, pictogram classes |
|---|---|---|---|
| in-distribution | 99.45 % | 82.42 % | **59.15 %** |
| domain-hardened | 98.90 % | 98.08 % | 96.48 % |
| shipped (domain + night) | **99.73 %** | 96.7 % | — |

Grayscale (`gray48` payload mode):

| | RGB | grayscale |
|---|---|---|
| without grayscale augmentation | 98.35 % | **74.18 %** |
| with grayscale augmentation | 99.45 % | **99.18 %** |

Low light (`src/lowlight_set.py`, physical sensor model: de-gamma → Poisson shot
noise → Gaussian read noise, at matched brightness):

| darkness | before | after |
|---|---|---|
| 0.70 | 97.25 % | **99.18 %** |
| 0.85 | 91.76 % | **96.98 %** |
| 0.95 | 23.63 % | **62.09 %** |

Alpha-blending toward black overstates robustness at high darkness because it
omits shot noise; the two models agree until noise dominates, then diverge.

## Timing

Wall-clock, machine specific, and not transferable:

| | |
|---|---|
| Native x86 `-O3`, i7-1365U | 1.52 ms |
| QEMU guest-measured median (`k_cycle_get_32`) | 33.8 ms, from 55.3 ms |
| Hardware, whole request including UART transfer | 1.72 s/image, dominated by 6,912 bytes of hex at 115200 baud |

The QEMU figure is guest-measured around the inference call and cross-checked
against a wall-clock A/B of both builds. The host script's line-gap method is
unreliable once inference is this fast.

## Link reliability

Zero re-sends across the 364-image hardware sweep. One frame was rejected during
bring-up (`ERROR odd number of hex digits`) while a terminal emulator still held
the port — one event, not a measured error rate.

## What is not proven

- 364 images is a small validation set. One error is one error.
- The stress and low-light sets are our own constructions, not held-out data.
- The evaluation target is the ESP32-S3; the 364-image hardware sweep ran on an
  ESP32-D0WD-V3 bench proxy from byte-identical source. The S3 image was built
  and linked, and its memory report is above, but the sweep was not repeated on
  S3 silicon.
