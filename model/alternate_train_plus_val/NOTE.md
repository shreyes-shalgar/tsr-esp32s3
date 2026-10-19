# Alternate model — trained on Train + Validation

Same architecture and **the same training recipe as the shipped model**,
including the domain and low-light hardening, but fitted on all 1,824 labelled
images (Train 1,460 + Validation 364) instead of 1,460.

Regenerated 28 Aug 2026. The previous file here predated the low-light
hardening and was not in fact the same recipe; its night-0.85 accuracy was
68.13 % against the shipped model's 96.98 %. If you have an older copy of this
package, discard that one.

**Its reported "validation accuracy" of 100 % is meaningless** — the validation
set is inside its training data. It is quoted here only so the number is not
mistaken for a holdout result.

## Why it is not the primary submission

The argument for it is that ~25 % more data usually helps on an unseen test
set, and the hidden test set is what is scored. The argument against turned out
to be measurable. Scored on the two constructed proxies (`src/stress_set.py`,
`src/lowlight_set.py`):

| | shipped (Train only) | this model (Train + Val) |
|---|---|---|
| Validation | **99.73 %** (honest holdout) | 100 % (meaningless) |
| Photographic stress set | **96.98 %** | 84.07 % |
| Night, 0.70 darkness | 99.18 % | 99.18 % |
| Night, 0.85 darkness | 96.98 % | **98.35 %** |
| Night, 0.95 darkness | **64.01 %** | 61.81 % |

Read the stress row carefully before drawing a conclusion from it. The stress
set is built by applying photographic transforms to the *validation* images, so
for the shipped model those images are unseen, while for this model they are
transformed copies of its own training data. That asymmetry flatters neither
model in an obvious direction — but a network that had memorised the originals
and still dropped 13 points once they were rescaled and recoloured is showing
that the extra data was absorbed as memorisation rather than generalisation.
That is the specific failure the shipped model's augmentation exists to
prevent.

Combined with the fact that this model cannot be verified end to end — there is
no holdout left to verify it against — it stays the alternate.

To swap it in:

```bash
python src/export_c_model.py \
    --tflite model/alternate_train_plus_val/quantized_model_int8.tflite \
    --outdir src/qemu_app/src
cmake -B build -GNinja -DBOARD=qemu_x86 src/qemu_app && ninja -C build
python scripts/qemu_uart_eval.py --mode rgb48
```

The reported accuracy from that last command will be 100 % for the same reason.
Rebuild from `model/quantized_model_int8.tflite` to restore a verifiable
configuration.
