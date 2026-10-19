# Dataset provenance

`out/data.npz` and `samples/` contain images **supplied by the AI Arena 2026
organisers** for Track 1. They are included here so the accuracy claims in
`docs/RESULTS.md` can be reproduced without them.

| | |
|---|---|
| `out/data.npz` | `Xtr` 1460 × 48 × 48 × 3 uint8, `ytr`; `Xva` 364 × 48 × 48 × 3 uint8, `yva` |
| `samples/` | one representative 48 × 48 PNG per class, 17 files |

The class ids (0, 2, 3, 4, 5, 6, 7, 24, 43, 69, 70, 72, 83, 84, 85, 86, 87)
follow a Chinese traffic-sign dataset numbering. The raw `Train/` and
`Validation/` folders the organisers distributed are **not** in this repository.

## If you need to remove it

The MIT license in `LICENSE` covers the code in this repository. It does **not**
cover these images, and this repository is not the right place to settle their
redistribution terms. If in doubt, drop them:

```bash
git rm -r --cached out/data.npz samples/
printf 'out/data.npz\nsamples/\n' >> .gitignore
git commit -m "Remove organiser-supplied dataset"
```

Everything except the accuracy-reproduction steps still works without it: the
firmware, the C runtime, the models, and single-image inference on any image you
supply yourself. `src/prepare_data.py` regenerates `out/data.npz` from a
`Train/` + `Validation/` tree if you have one.

## Regenerating

```bash
python src/prepare_data.py <path to the Train/Validation tree> out/data.npz
python src/augment_domain.py                       # synthesised photographic views
python src/lowlight_set.py                         # physical low-light sensor model
python src/model_training.py --data out/data_domain3.npz
python src/model_optimize.py --keras model/trained_model.keras
python src/export_c_model.py --tflite model/quantized_model_int8.tflite \
                             --outdir src/qemu_app/src
```

`src/prepare_data.py` writes `out/data_paths.json` alongside the `.npz`;
`src/augment_domain.py` needs it, because it reloads the original files at full
resolution rather than upscaling the 48 × 48 crops. That file records absolute
paths from the machine that built it and is not included here.
