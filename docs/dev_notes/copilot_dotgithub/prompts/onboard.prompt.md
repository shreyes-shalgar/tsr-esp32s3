---
name: onboard
description: First run. Establish ground truth about the repo before changing anything.
agent: tsr-lead
---

Do not edit any file in this session.

1. Read `AGENTS.md` in full.
2. Run `python scripts/verify_all.py` and wait for it to finish.
3. Report a table of the measured baseline: accuracy (native and QEMU),
   bit-exactness agreement, peak RAM, int8 model size, mean inference time, and
   the ESP32-S3 FLASH/DRAM/IRAM figures.
4. If any check fails, diagnose the cause and report it. Do not fix it yet.
5. List the three changes you judge most likely to raise the final score, with
   the reason and the risk for each, ordered by expected value. Say which one
   you would do first and why.

End by asking for confirmation before you start work.
