/*
 * Traffic-sign recognition - dependency-free int8 CNN runtime.
 * AI Arena 2026, Track 1.
 *
 * Arithmetic is bit-exact with the TensorFlow Lite reference kernels
 * (verified over the full validation set, 364 images x 17 outputs, 0 diffs).
 */
#ifndef TSR_MODEL_H
#define TSR_MODEL_H

#include <stdint.h>
#include "tsr_model_data.h"

/* Run inference on one 48x48x3 RGB image, row-major, 8-bit per channel.
 * Returns the index 0..TSR_NUM_CLASSES-1 of the winning class.
 * If `scores` is non-NULL the int8 output vector is copied there. */
int tsr_predict(const uint8_t *rgb, int8_t *scores);

/* Map a model output index to the official challenge class ID. */
static inline int tsr_class_id(int idx)
{
    return (idx >= 0 && idx < TSR_NUM_CLASSES) ? (int)tsr_class_ids[idx] : -1;
}

/* Softmax confidence of the winning class, in basis points (0..10000).
 * Computed from the dequantised int8 logits. Softmax is monotonic, so this
 * never changes which class wins - it only puts a probability on it. */
uint32_t tsr_confidence_bp(const int8_t *scores, int best);

/* Static RAM used by the inference arena, in bytes. */
static inline uint32_t tsr_arena_bytes(void) { return (uint32_t)TSR_ARENA_BYTES; }

#endif /* TSR_MODEL_H */
