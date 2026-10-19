#include "tsr_model.h"
#include <math.h>

/* ------------------------------------------------------------------ */
/* Fixed-point requantisation.                                        */
/*                                                                    */
/* Convolution and fully-connected use DIFFERENT rounding in TFLite:  */
/*  - conv: gemmlowp SaturatingRoundingDoublingHighMul + RoundingDiv  */
/*  - fc  : higher-precision int64 round-half-away-from-zero          */
/* Both are reproduced exactly; mixing them up costs 1 LSB on ~4% of  */
/* images, which was verified empirically against the TFLite ref.     */
/* ------------------------------------------------------------------ */

static inline int32_t tsr_srdhm(int32_t a, int32_t b)
{
    int64_t ab = (int64_t)a * (int64_t)b;
    int64_t nudge = (ab >= 0) ? (int64_t)(1 << 30) : (int64_t)(1 - (1 << 30));
    int64_t v = ab + nudge;
    return (int32_t)(v / ((int64_t)1 << 31));   /* C division truncates to 0 */
}

static inline int32_t tsr_rdpot(int32_t x, int exp)
{
    int32_t mask, rem, thr;
    if (exp <= 0) return x;
    mask = ((int32_t)1 << exp) - 1;
    rem  = x & mask;
    thr  = (mask >> 1) + ((x < 0) ? 1 : 0);
    return (x >> exp) + ((rem > thr) ? 1 : 0);
}

/* gemmlowp path - used by CONV_2D */
static inline int32_t tsr_mbqm(int32_t x, int32_t m, int shift)
{
    int left  = (shift > 0) ? shift : 0;
    int right = (shift > 0) ? 0 : -shift;
    return tsr_rdpot(tsr_srdhm(x * ((int32_t)1 << left), m), right);
}

/* exact path - used by FULLY_CONNECTED */
static inline int32_t tsr_mbqm_exact(int32_t acc, int32_t m, int shift)
{
    int ts = 31 - shift;                       /* shift < 0 => ts in 32..62 */
    int64_t prod = (int64_t)acc * (int64_t)m;
    int64_t half = (int64_t)1 << (ts - 1);
    return (int32_t)((prod >= 0) ? ((prod + half) >> ts)
                                 : -(((-prod) + half) >> ts));
}

static inline int8_t tsr_sat8(int32_t v)
{
    return (int8_t)((v < -128) ? -128 : ((v > 127) ? 127 : v));
}

static inline uint8_t tsr_satu8(int32_t v)
{
    return (uint8_t)((v < 0) ? 0 : ((v > 255) ? 255 : v));
}

/* ------------------------------------------------------------------ */
/* Activation representation                                          */
/*                                                                    */
/* Activations are held as uint8 (= the int8 value + 128) rather than  */
/* int8. Every tensor in this graph has zero_point -128, so the        */
/* dequantising subtraction (x - zero_point) that a textbook int8      */
/* kernel performs on every multiply-accumulate becomes exactly the    */
/* stored uint8 byte, and disappears from the inner loop. The output   */
/* side cancels the same way: storing (requantised + zero_point + 128) */
/* clamped to 0..255 is bit-identical to storing the int8 result,      */
/* because adding 128 is monotonic and maps the clamp range exactly.   */
/*                                                                     */
/* This is a representation change, not an approximation - the graph   */
/* still produces byte-identical results to the TFLite reference.      */
/* The general (zero_point != -128) path is kept for correctness if a  */
/* retrained model ever breaks that assumption.                        */
/* ------------------------------------------------------------------ */

/* ------------------------------------------------------------------ */
/* Kernels                                                            */
/* ------------------------------------------------------------------ */

/* One output row of a convolution, written to `orow` (out_w * out_c bytes).
 * Factoring the row out is what makes conv+maxpool fusion possible without
 * duplicating the kernel.
 *
 * `in` need not be the whole input tensor: it may be a window whose first row
 * is logical input row `in_row0`. Padding clamps against the layer's real
 * in_h, so a window produces byte-identical results to the full tensor as
 * long as it contains every row the clamped kernel actually reads. */
static void tsr_conv2d_row(const uint8_t *in, uint8_t *orow,
                           const tsr_layer_t *L, int oy, int in_row0)
{
    const int IH = L->in_h,  IW = L->in_w,  IC = L->in_c;
    const int OW = L->out_w, OC = L->out_c;
    const int K = L->ksize, S = L->stride, P = L->pad;
    const int out_bias = L->out_zp + 128;
    const int patch = K * K * IC;
    const int fast = (L->in_zp == -128);
    const int in_off = -L->in_zp - 128;          /* 0 on the fast path */
    const int iy0 = oy * S - P;
    const int ky0 = (iy0 < 0) ? -iy0 : 0;
    const int ky1 = (iy0 + K > IH) ? (IH - iy0) : K;
    int ox, co, ky, kx, ci;

    for (ox = 0; ox < OW; ++ox) {
        const int ix0 = ox * S - P;
        const int kx0 = (ix0 < 0) ? -ix0 : 0;
        const int kx1 = (ix0 + K > IW) ? (IW - ix0) : K;
        uint8_t *o = orow + ox * OC;

        /* Four output channels per pass. Each input byte is loaded once and
         * reused across the four weight streams, so the input side of the loop
         * does a quarter of the memory traffic it did when the channel loop
         * was outermost. */
        for (co = 0; co + 4 <= OC; co += 4) {
            const int8_t *w0 = L->w + (co + 0) * patch;
            const int8_t *w1 = w0 + patch;
            const int8_t *w2 = w1 + patch;
            const int8_t *w3 = w2 + patch;
            int32_t a0 = L->bias[co + 0], a1 = L->bias[co + 1];
            int32_t a2 = L->bias[co + 2], a3 = L->bias[co + 3];

            for (ky = ky0; ky < ky1; ++ky) {
                const int base = ((ky * K) + kx0) * IC;
                const uint8_t *ip = in + (((iy0 + ky - in_row0) * IW) + ix0 + kx0) * IC;
                const int8_t *p0 = w0 + base, *p1 = w1 + base;
                const int8_t *p2 = w2 + base, *p3 = w3 + base;

                for (kx = kx0; kx < kx1; ++kx) {
                    if (fast) {
                        for (ci = 0; ci < IC; ++ci) {
                            const int32_t x = ip[ci];
                            a0 += x * p0[ci];
                            a1 += x * p1[ci];
                            a2 += x * p2[ci];
                            a3 += x * p3[ci];
                        }
                    } else {
                        for (ci = 0; ci < IC; ++ci) {
                            const int32_t x = (int32_t)ip[ci] + in_off;
                            a0 += x * p0[ci];
                            a1 += x * p1[ci];
                            a2 += x * p2[ci];
                            a3 += x * p3[ci];
                        }
                    }
                    ip += IC; p0 += IC; p1 += IC; p2 += IC; p3 += IC;
                }
            }
            o[co + 0] = tsr_satu8(tsr_mbqm(a0, L->mult[co + 0], L->shift[co + 0]) + out_bias);
            o[co + 1] = tsr_satu8(tsr_mbqm(a1, L->mult[co + 1], L->shift[co + 1]) + out_bias);
            o[co + 2] = tsr_satu8(tsr_mbqm(a2, L->mult[co + 2], L->shift[co + 2]) + out_bias);
            o[co + 3] = tsr_satu8(tsr_mbqm(a3, L->mult[co + 3], L->shift[co + 3]) + out_bias);
        }

        /* tail, for an output-channel count that is not a multiple of four */
        for (; co < OC; ++co) {
            const int8_t *wbase = L->w + co * patch;
            int32_t acc = L->bias[co];
            for (ky = ky0; ky < ky1; ++ky) {
                const uint8_t *ip = in + (((iy0 + ky - in_row0) * IW) + ix0 + kx0) * IC;
                const int8_t *wp = wbase + ((ky * K) + kx0) * IC;
                for (kx = kx0; kx < kx1; ++kx) {
                    for (ci = 0; ci < IC; ++ci)
                        acc += ((int32_t)ip[ci] + in_off) * (int32_t)wp[ci];
                    ip += IC;
                    wp += IC;
                }
            }
            o[co] = tsr_satu8(tsr_mbqm(acc, L->mult[co], L->shift[co]) + out_bias);
        }
    }
}

static void tsr_conv2d(const uint8_t *in, uint8_t *out, const tsr_layer_t *L)
{
    const int stride = L->out_w * L->out_c;
    int oy;
    for (oy = 0; oy < L->out_h; ++oy)
        tsr_conv2d_row(in, out + oy * stride, L, oy, 0);
}

/*
 * Convolution fused with the 2x2 stride-2 max pool that follows it.
 *
 * This is what sets peak RAM. Run separately, the convolution has to
 * materialise its whole output - 24x24x32, 18 KB - purely so the pool can
 * immediately throw three quarters of it away. Fused, only the two rows the
 * pool is currently consuming exist at once: 1.5 KB of scratch instead of an
 * 18 KB buffer, and the arena drops 44%.
 *
 * The arithmetic is untouched, so this stays bit-exact; only the schedule
 * changes. `export_c_model.py` sizes the buffers with the same fusion rule, so
 * the generated constants and this schedule cannot drift apart.
 */
static void tsr_conv_maxpool2(const uint8_t *in, uint8_t *out,
                              const tsr_layer_t *C, const tsr_layer_t *P,
                              uint8_t *scratch)
{
    const int CW = C->out_w, CC = C->out_c;
    const int row = CW * CC;
    const int PH = P->out_h, PW = P->out_w;
    uint8_t *r0 = scratch, *r1 = scratch + row;
    int opy, opx, c;

    for (opy = 0; opy < PH; ++opy) {
        tsr_conv2d_row(in, r0, C, opy * 2, 0);
        tsr_conv2d_row(in, r1, C, opy * 2 + 1, 0);

        for (opx = 0; opx < PW; ++opx) {
            const uint8_t *a = r0 + (opx * 2) * CC;
            const uint8_t *b = a + CC;
            const uint8_t *d = r1 + (opx * 2) * CC;
            const uint8_t *e = d + CC;
            uint8_t *o = out + (opy * PW + opx) * CC;
            for (c = 0; c < CC; ++c) {
                uint8_t m = a[c];
                if (b[c] > m) m = b[c];
                if (d[c] > m) m = d[c];
                if (e[c] > m) m = e[c];
                o[c] = m;
            }
        }
    }
}

/*
 * conv -> conv -> 2x2/s2 max pool, all three in one pass.
 *
 * The pair fusion above still had to materialise the first convolution's
 * whole output: 24x24x16, 9 KB, and the single largest thing in the arena.
 * Extending the window one layer further removes it. To emit one pooled row
 * the second convolution needs two of its own output rows, which need four
 * consecutive rows of the first convolution - 4 x 24 x 16 = 1.5 KB - so that
 * is all that is ever held.
 *
 * Consecutive pooled rows share two of those four rows, so the window slides:
 * rows 2 and 3 become rows 0 and 1, and only two rows are computed fresh.
 * Each row of the first convolution is therefore evaluated exactly once, as
 * it was before fusing - the arena shrinks for the price of a 768-byte copy
 * per pooled row, not for the price of redoing the convolution.
 *
 * Correctness rests on two things, both asserted by the caller's guard:
 * the second convolution is 3x3 stride 1 with padding 1, so output rows
 * 2*opy and 2*opy+1 read exactly logical rows 2*opy-1 .. 2*opy+2; and rows
 * outside the tensor are never read, because tsr_conv2d_row clamps the kernel
 * against the layer's real in_h. The strip is zeroed anyway so the pass is
 * deterministic regardless.
 *
 * Bit-exact with the unfused schedule - same kernel, same order of
 * accumulation, only a different order of evaluation.
 */
static void tsr_conv_conv_maxpool2(const uint8_t *in, uint8_t *out,
                                   const tsr_layer_t *C1, const tsr_layer_t *C2,
                                   const tsr_layer_t *P, uint8_t *scratch)
{
    const int S1W = C1->out_w, S1C = C1->out_c;
    const int s1row = S1W * S1C;                 /* one row of conv-1 output */
    const int C2W = C2->out_w, C2C = C2->out_c;
    const int c2row = C2W * C2C;
    const int PH = P->out_h, PW = P->out_w;
    uint8_t *strip = scratch;                    /* 4 rows of conv-1 output   */
    uint8_t *r0 = scratch + 4 * s1row;           /* 2 rows of conv-2 output   */
    uint8_t *r1 = r0 + c2row;
    int opy, opx, c, sidx, q, i;

    for (opy = 0; opy < PH; ++opy) {
        const int row0 = opy * 2 - 1;            /* logical conv-1 row of strip[0] */
        int sfirst = 0;

        if (opy > 0) {
            /* slide: rows 2,3 of the previous window are rows 0,1 of this one */
            const uint8_t *from = strip + 2 * s1row;
            for (i = 0; i < 2 * s1row; ++i) strip[i] = from[i];
            sfirst = 2;
        }

        for (sidx = sfirst; sidx < 4; ++sidx) {
            uint8_t *dstrow = strip + sidx * s1row;
            q = row0 + sidx;
            if (q < 0 || q >= C1->out_h) {
                for (i = 0; i < s1row; ++i) dstrow[i] = 0;
            } else {
                tsr_conv2d_row(in, dstrow, C1, q, 0);
            }
        }

        tsr_conv2d_row(strip, r0, C2, opy * 2,     row0);
        tsr_conv2d_row(strip, r1, C2, opy * 2 + 1, row0);

        for (opx = 0; opx < PW; ++opx) {
            const uint8_t *a = r0 + (opx * 2) * C2C;
            const uint8_t *b = a + C2C;
            const uint8_t *d = r1 + (opx * 2) * C2C;
            const uint8_t *e = d + C2C;
            uint8_t *o = out + (opy * PW + opx) * C2C;
            for (c = 0; c < C2C; ++c) {
                uint8_t m = a[c];
                if (b[c] > m) m = b[c];
                if (d[c] > m) m = d[c];
                if (e[c] > m) m = e[c];
                o[c] = m;
            }
        }
    }
}

static void tsr_maxpool(const uint8_t *in, uint8_t *out, const tsr_layer_t *L)
{
    const int IW = L->in_w, IC = L->in_c;
    const int OH = L->out_h, OW = L->out_w;
    const int K = L->ksize, S = L->stride;
    int oy, ox, c, ky, kx;

    /* Adding 128 is monotonic, so a max over the uint8 representation picks
     * the same element as a max over the int8 values. */
    for (oy = 0; oy < OH; ++oy)
        for (ox = 0; ox < OW; ++ox) {
            uint8_t *o = out + (oy * OW + ox) * IC;
            for (c = 0; c < IC; ++c) {
                uint32_t m = 0;
                for (ky = 0; ky < K; ++ky) {
                    const uint8_t *ip = in + (((oy * S + ky) * IW) + ox * S) * IC + c;
                    for (kx = 0; kx < K; ++kx) {
                        if ((uint32_t)*ip > m) m = *ip;
                        ip += IC;
                    }
                }
                o[c] = (uint8_t)m;
            }
        }
}

static void tsr_avgpool(const uint8_t *in, uint8_t *out, const tsr_layer_t *L)
{
    const int IW = L->in_w, IC = L->in_c;
    const int OH = L->out_h, OW = L->out_w;
    const int K = L->ksize, S = L->stride;
    const int n = K * K;
    int oy, ox, c, ky, kx;

    for (oy = 0; oy < OH; ++oy)
        for (ox = 0; ox < OW; ++ox) {
            uint8_t *o = out + (oy * OW + ox) * IC;
            for (c = 0; c < IC; ++c) {
                int32_t s = 0, acc;
                for (ky = 0; ky < K; ++ky) {
                    const uint8_t *ip = in + (((oy * S + ky) * IW) + ox * S) * IC + c;
                    for (kx = 0; kx < K; ++kx) { s += *ip; ip += IC; }
                }
                /* TFLite rounds the int8 sum away from zero, and the sign of
                 * that sum decides the direction - so the uint8 sum has to be
                 * shifted back into int8 space before rounding, not after. */
                s -= 128 * n;
                acc = (s > 0) ? ((s + n / 2) / n) : ((s - n / 2) / n);
                o[c] = tsr_satu8(acc + 128);
            }
        }
}

static void tsr_fc(const uint8_t *in, int8_t *out, const tsr_layer_t *L)
{
    const int CI = L->in_c, CO = L->out_c;
    const int in_off = -L->in_zp - 128;
    const int out_zp = L->out_zp;
    int co, ci;

    for (co = 0; co < CO; ++co) {
        const int8_t *w = L->w + co * CI;
        int32_t acc = L->bias[co];
        for (ci = 0; ci < CI; ++ci)
            acc += ((int32_t)in[ci] + in_off) * (int32_t)w[ci];
        out[co] = tsr_sat8(tsr_mbqm_exact(acc, L->mult[co], L->shift[co]) + out_zp);
    }
}

/* ------------------------------------------------------------------ */
/* Graph execution - two static ping-pong buffers plus a row scratch  */
/* ------------------------------------------------------------------ */

static uint8_t tsr_buf_a[TSR_BUF_A];
static uint8_t tsr_buf_b[TSR_BUF_B];
static uint8_t tsr_scratch[TSR_SCRATCH];

/* Must match the fusion rules in export_c_model.py, which sizes the buffers. */
static inline int tsr_fuses_with_next(int i)
{
    const tsr_layer_t *a, *b;
    if (i + 1 >= TSR_NUM_LAYERS) return 0;
    a = &tsr_layers[i];
    b = &tsr_layers[i + 1];
    return (a->op == TSR_OP_CONV && b->op == TSR_OP_MAXPOOL &&
            b->ksize == 2 && b->stride == 2);
}

/* conv -> conv -> 2x2/s2 pool. The second convolution must be 3x3 stride 1
 * padding 1 for the four-row window to cover exactly what it reads. */
static inline int tsr_fuses_triple(int i)
{
    const tsr_layer_t *a, *b, *c;
    if (i + 2 >= TSR_NUM_LAYERS) return 0;
    a = &tsr_layers[i]; b = &tsr_layers[i + 1]; c = &tsr_layers[i + 2];
    return (a->op == TSR_OP_CONV && b->op == TSR_OP_CONV &&
            c->op == TSR_OP_MAXPOOL && c->ksize == 2 && c->stride == 2 &&
            b->ksize == 3 && b->stride == 1 && b->pad == 1);
}

int tsr_predict(const uint8_t *rgb, int8_t *scores)
{
    /* The model's input quantisation is scale 1.0, zero_point -128, so the
     * uint8 pixel IS the activation byte. The first convolution reads the
     * received image in place - no conversion pass, and no arena buffer spent
     * holding a copy of the input. */
    const uint8_t *src = rgb;
    uint8_t *dst = tsr_buf_a;
    int8_t logits[TSR_NUM_CLASSES];
    int i = 0, best;
    int32_t bestv;

    while (i < TSR_NUM_LAYERS) {
        const tsr_layer_t *L = &tsr_layers[i];

        if (L->op == TSR_OP_FC) {
            tsr_fc(src, logits, L);
            break;
        }

        if (tsr_fuses_triple(i)) {
            tsr_conv_conv_maxpool2(src, dst, L, &tsr_layers[i + 1],
                                   &tsr_layers[i + 2], tsr_scratch);
            i += 3;
        } else if (tsr_fuses_with_next(i)) {
            tsr_conv_maxpool2(src, dst, L, &tsr_layers[i + 1], tsr_scratch);
            i += 2;
        } else {
            switch (L->op) {
            case TSR_OP_CONV:    tsr_conv2d(src, dst, L);  break;
            case TSR_OP_MAXPOOL: tsr_maxpool(src, dst, L); break;
            default:             tsr_avgpool(src, dst, L); break;
            }
            i += 1;
        }

        src = dst;
        dst = (dst == tsr_buf_a) ? tsr_buf_b : tsr_buf_a;
    }

    best = 0;
    bestv = logits[0];
    for (i = 1; i < TSR_NUM_CLASSES; ++i)
        if ((int32_t)logits[i] > bestv) { bestv = logits[i]; best = i; }

    if (scores)
        for (i = 0; i < TSR_NUM_CLASSES; ++i) scores[i] = logits[i];

    return best;
}

/* ------------------------------------------------------------------ */
/* Confidence                                                          */
/*                                                                     */
/* The graph deliberately ends at the logits - the device only needs   */
/* argmax, and softmax is monotonic. The host protocol reports a       */
/* confidence figure, so softmax is applied here, over 17 values, on   */
/* the output side only. The int8 inference path stays integer-only.   */
/* ------------------------------------------------------------------ */
uint32_t tsr_confidence_bp(const int8_t *scores, int best)
{
    float z[TSR_NUM_CLASSES];
    float zmax, sum = 0.0f, p;
    int i;

    for (i = 0; i < TSR_NUM_CLASSES; ++i) {
        z[i] = ((float)scores[i] - (float)TSR_OUT_ZP) * TSR_OUT_SCALE;
    }
    zmax = z[0];
    for (i = 1; i < TSR_NUM_CLASSES; ++i) {
        if (z[i] > zmax) zmax = z[i];
    }
    for (i = 0; i < TSR_NUM_CLASSES; ++i) {
        z[i] = expf(z[i] - zmax);
        sum += z[i];
    }
    if (sum <= 0.0f) return 0U;
    p = z[best] / sum;
    if (p < 0.0f) p = 0.0f;
    if (p > 1.0f) p = 1.0f;
    /* Reported verbatim, not capped. Measured over the validation set this
     * figure is well calibrated - ECE 0.62 %, and slightly UNDER-confident
     * (mean confidence 98.83 % against 99.45 % accuracy), so clipping it
     * would understate an already-conservative number. */
    return (uint32_t)(p * 10000.0f + 0.5f);
}
