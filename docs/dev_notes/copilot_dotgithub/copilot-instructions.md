# Project instructions

Read `AGENTS.md` at the repository root before your first action. It is the
project contract: hard constraints, the exact UART protocol, the measured
baseline, and a list of traps that already cost real debugging time.

Three rules that override any local judgement:

1. **Never report a number you did not measure.** `python scripts/verify_all.py`
   is the only source of truth for accuracy, RAM, model size and timing. If you
   did not run it in this session, say the figure is unverified.

2. **The baseline is verified; a regression is reverted, not explained.** If a
   change lowers accuracy, raises peak RAM, or breaks bit-exactness with the
   TFLite reference, revert it. Do not layer a fix on top of a broken change.

3. **Say what is untested.** Nothing here can flash physical ESP32-S3 hardware.
   Report that the image builds and links; do not imply it was run.

House style: comment *why*, not *what*. Prefer a control experiment over an
argument. When two options differ by less than one validation image (0.27 %),
choose the one with the smaller memory footprint — the tie-breaker is lowest RAM
first, then fastest inference, then smallest model.
