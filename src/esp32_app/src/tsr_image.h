/* Payload -> 48x48x3 RGB normalisation.
 *
 * send_image.py can present three different payloads:
 *   64*64*3 = 12288  RGB          (script default)
 *   48*48*3 =  6912  RGB          (challenge guide)
 *   48*48   =  2304  grayscale    (script -g flag)
 * The firmware accepts all three and normalises to the model input.
 */
#ifndef TSR_IMAGE_H
#define TSR_IMAGE_H

#include <stdint.h>
#include <stddef.h>

#define TSR_PAYLOAD_RGB64   (64 * 64 * 3)
#define TSR_PAYLOAD_RGB48   (48 * 48 * 3)
#define TSR_PAYLOAD_GRAY48  (48 * 48)
#define TSR_PAYLOAD_MAX     TSR_PAYLOAD_RGB64

/* Returns 0 on success, -1 if len is not a recognised payload size.
 * `out` must hold 48*48*3 bytes. */
int tsr_prepare_input(const uint8_t *payload, size_t len, uint8_t *out);

/* Human-readable description of the last accepted payload format. */
const char *tsr_last_format(void);

#endif /* TSR_IMAGE_H */
