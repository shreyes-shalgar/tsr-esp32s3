---
name: ship
description: Final pre-submission check. Run before packaging.
agent: tsr-lead
---

Produce the submission and prove it works from a clean checkout.

1. Run the **full** gate: `python scripts/verify_all.py`. Not `--quick`.
2. Confirm the submission tree matches the required structure:
   ```
   src/model_training.py, src/model_optimize.py, src/qemu_app/
   model/trained_model, model/quantized_model
   documentation/Solution_Documentation.pptx
   README
   ```
3. Confirm the PPT covers all five required sections: approach and why;
   training method and results; optimisation technique with size before/after
   and the effect on accuracy; how the model was integrated with Zephyr and run
   on QEMU; and performance on the validation set on QEMU — accuracy, model
   size, RAM usage, inference time.
4. Zip it, then **verify from a clean room**: extract the zip to a fresh
   directory, build from that copy only, and run one image end-to-end through
   the UART protocol. A submission that only builds in the working tree is a
   submission that fails for the evaluators.
5. Re-read the README for numbers that have drifted since they were written.
   Every figure must match the gate output from step 1. Stale numbers in
   documentation are the most common way a good project loses marks for
   "Code Quality & Documentation".

Report the final table of measured metrics and explicitly flag anything that is
**not** verified — in particular, that the ESP32-S3 image builds and links but
has not been run on physical hardware.
