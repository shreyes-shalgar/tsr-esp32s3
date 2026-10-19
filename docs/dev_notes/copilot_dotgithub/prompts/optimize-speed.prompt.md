---
name: optimize-speed
description: Speed up the int8 convolution kernel without losing bit-exactness. Highest-value remaining work.
agent: tsr-lead
argument-hint: optional — a specific optimisation to try
---

Goal: reduce mean inference time. Inference performance is 15 % of the score and
tie-breaker #2, and the kernel is currently plain unoptimised C — this is the
largest untapped category in the project.

Hard constraint: **argmax agreement with the TFLite reference must stay at
364/364.** A faster kernel that changes one prediction is a failed experiment,
not a trade-off.

Method — follow it exactly:

1. Have `tsr-bench` record the current mean inference time as the baseline.
2. Delegate **one** optimisation to `tsr-kernel`. Candidates, in order:
   - accumulate 2–4 output channels per input load
   - split each output row into padded prologue / unpadded body / padded
     epilogue so the padding test leaves the inner loop
   - specialise the `IC == 3` first layer
   ${input:idea:optional specific idea to try first}
3. Run `python scripts/verify_all.py`.
4. If bit-exactness holds and time improved, keep it. If bit-exactness broke,
   revert immediately — do not attempt to patch the rounding back into
   agreement. If time did not improve, revert and say so; a null result reported
   honestly is worth more than a change kept out of sunk cost.
5. Repeat from step 2 with the next candidate.

Report a before/after table with the exact commands that produced each number.
Do not report an improvement you have not reproduced with the full gate.
