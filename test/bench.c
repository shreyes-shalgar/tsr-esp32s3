/*
 * Timing harness. Runs the same images repeatedly so the measurement is not
 * dominated by file I/O, and reports ns per inference.
 *
 *   gcc -O2 -std=gnu99 -o bench test/bench.c qemu_app/src/tsr_model.c \
 *       qemu_app/src/tsr_model_data.c -lm
 *   ./bench test/val_images.bin 200
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../qemu_app/src/tsr_model.h"

#define NIMG 32

int main(int argc, char **argv)
{
    FILE *f;
    static uint8_t imgs[NIMG][TSR_IMG_BYTES];
    int8_t sc[TSR_NUM_CLASSES];
    int reps = (argc > 2) ? atoi(argv[2]) : 100;
    int n = 0, i, r;
    long checksum = 0;
    struct timespec t0, t1;

    if (argc < 2) { fprintf(stderr, "usage: %s <raw> [reps]\n", argv[0]); return 2; }
    f = fopen(argv[1], "rb");
    if (!f) { perror("open"); return 1; }
    while (n < NIMG && fread(imgs[n], 1, TSR_IMG_BYTES, f) == TSR_IMG_BYTES) n++;
    fclose(f);
    if (!n) { fprintf(stderr, "no images\n"); return 1; }

    /* warm up so the first run's cold cache is not in the measurement */
    for (i = 0; i < n; ++i) checksum += tsr_predict(imgs[i], sc);

    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (r = 0; r < reps; ++r)
        for (i = 0; i < n; ++i)
            checksum += tsr_predict(imgs[i], sc);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    {
        double ns = ((t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec))
                    / (double)(reps * n);
        printf("%.0f ns/inference  (%.3f ms)  images=%d reps=%d checksum=%ld\n",
               ns, ns / 1e6, n, reps, checksum);
    }
    return 0;
}
