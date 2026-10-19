# Alternate model — trained without domain hardening

Same architecture and training recipe, but fitted only on the supplied data with
no synthesised photographic views of the pictogram classes.

| | this model | shipped model |
|---|---|---|
| Validation | **99.45 %** | 98.90 % |
| Stress set, overall | 82.42 % | **98.08 %** |
| Stress set, pictogram classes | 59.15 % | **96.48 %** |

It is two validation images better and 15.7 points worse under domain shift.

Ship it **only** if you learn that the hidden test set is generated the same way
as the supplied train and validation splits — that is, the seven pictogram
classes appear as 32×32 crops there too. Otherwise the shipped model is the
better bet; break-even sits at a 3.6 % chance of shift.

To swap it in:

```
python src/export_c_model.py \
    --tflite model/alternate_in_distribution/quantized_model_int8.tflite \
    --outdir src/qemu_app/src
cmake -B build -GNinja -DBOARD=qemu_x86 src/qemu_app && ninja -C build
python scripts/verify_all.py
```
