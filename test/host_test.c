/*
 * Native harness for the int8 runtime.
 *
 * Runs the same C code the firmware runs, on the host, over a raw file of
 * 48x48x3 uint8 images. Prints one line per image:
 *
 *     <index> <official_class_id> <17 int8 scores>
 *
 * `scripts/verify_all.py` compares those score vectors against the TFLite
 * reference interpreter to prove bit-exactness without needing QEMU in the
 * loop. Keeping this separate from the Zephyr build is what makes the
 * numerical check fast enough to run on every change.
 *
 *   gcc -O2 -std=gnu99 -o host_test test/host_test.c \
 *       src/qemu_app/src/tsr_model.c src/qemu_app/src/tsr_model_data.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../src/qemu_app/src/tsr_model.h"

int main(int argc, char **argv)
{
    FILE *f;
    static uint8_t img[TSR_IMG_BYTES];
    int8_t sc[TSR_NUM_CLASSES];
    int n = 0, i;
    struct timespec t0, t1;
    double total = 0.0;

    if (argc < 2) {
        fprintf(stderr, "usage: %s <raw 48x48x3 image file>\n", argv[0]);
        return 2;
    }
    f = fopen(argv[1], "rb");
    if (!f) {
        perror("open");
        return 1;
    }

    while (fread(img, 1, TSR_IMG_BYTES, f) == TSR_IMG_BYTES) {
        int k;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        k = tsr_predict(img, sc);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        total += (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;

        printf("%d %d", k, tsr_class_id(k));
        for (i = 0; i < TSR_NUM_CLASSES; ++i) {
            printf(" %d", sc[i]);
        }
        printf("\n");
        n++;
    }
    fclose(f);

    fprintf(stderr, "images=%d  arena=%u B  mean_infer=%.3f ms\n",
            n, tsr_arena_bytes(), total / (n ? n : 1));
    return 0;
}
