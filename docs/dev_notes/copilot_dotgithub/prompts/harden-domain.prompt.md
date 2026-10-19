---
name: harden-domain
description: Reduce the risk that the hidden test set differs from the training distribution. Insurance, not accuracy chasing.
agent: tsr-lead
---

The single largest risk to the final score is not the two known errors — it is a
hidden test set drawn differently from the training data.

Specifically: classes 69, 70, 72 and 83–86 are supplied **only** as 32×32
pre-augmented crops with an alpha channel. If the hidden set presents those
classes as full-resolution photographs, accuracy on seven of seventeen classes
falls, and nothing in the validation score would have warned us.

Task:

1. With `tsr-ml`, build a **held-out stress set** from the existing validation
   images that simulates the shift: for the pictogram classes, composite them
   onto photographic backgrounds, apply perspective warp, motion blur, JPEG
   compression and lighting variation. Do not train on this — it is a probe.
2. Measure current accuracy on that stress set. That number is your estimate of
   exposure. Report it plainly even if it is bad.
3. If exposure is material, add matching augmentation to training, retrain, and
   re-measure **both** the stress set and the real validation set.
4. Accept the change only if stress-set accuracy improves and validation
   accuracy does not regress by more than one image. Run the full gate.

Constraints: no horizontal flips (classes 24, 43, 83, 84 are directional); keep
grayscale and resolution augmentation; keep the five-operator limit.

Be honest in the report about what this does and does not prove. A stress set
you built yourself is a proxy for the hidden set, not the hidden set. Say that.
