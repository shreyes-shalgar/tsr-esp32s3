/*
 * AI Arena 2026 - Track 1 : Traffic Sign Recognition
 * Zephyr UART inference application.
 *
 * Protocol (matches the supplied send_image.py exactly):
 *   board -> host : "READY"
 *   host  -> board: ASCII hex bytes, 32 bytes (64 chars) per line
 *   host  -> board: "STOP"
 *   board -> host : "RECEIVED <n> bytes"
 *   board -> host : "PREDICTION class=<id> label=<name> confidence=<p>%"
 *   board -> host : "READY"                               (ready for the next image)
 *
 * On any failure the board emits a line starting with "ERROR" instead of
 * PREDICTION, so the host never has to wait out its 30 s timeout.
 *
 * The host measures inference time as the gap between the RECEIVED line and
 * the next line it receives, so nothing may be printed between the two.
 *
 * UART: 115200 8N1, no flow control.
 */
#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/drivers/uart.h>
#include <zephyr/sys/util.h>

#include "tsr_model.h"
#include "tsr_image.h"

/* Bump both when the shipped model changes. */
#define TSR_BUILD_TAG  "2026.08.28-t3f"     /* + conv-conv-pool triple fusion */
#define TSR_VAL_ACC    "99.73%"

#define RX_IDLE_TIMEOUT_MS 20000   /* between bytes, once a transfer has begun */
#define RX_ARM_TIMEOUT_MS  3000    /* while idle: re-announce READY this often */

static const struct device *const uart_dev =
	DEVICE_DT_GET(DT_CHOSEN(zephyr_console));

static uint8_t  payload[TSR_PAYLOAD_MAX];
static uint8_t  image[TSR_IMG_BYTES];
static int8_t   scores[TSR_NUM_CLASSES];
#ifdef CONFIG_TSR_REPORT_TIMING
static uint32_t last_infer_us;
#endif

/* ------------------------------------------------------------------ */
/* UART plumbing                                                       */
/* ------------------------------------------------------------------ */

static void uart_write_str(const char *s)
{
	while (*s) {
		uart_poll_out(uart_dev, (unsigned char)*s++);
	}
}

/* Minimal unsigned decimal printer - avoids pulling in printf. */
static void uart_write_u32(uint32_t v)
{
	char b[12];
	int i = 0;

	if (v == 0) {
		uart_poll_out(uart_dev, '0');
		return;
	}
	while (v && i < (int)sizeof(b)) {
		b[i++] = (char)('0' + (v % 10U));
		v /= 10U;
	}
	while (i--) {
		uart_poll_out(uart_dev, (unsigned char)b[i]);
	}
}

/* Polled receive.
 *
 * Polling rather than interrupt-driven RX is deliberate. Bytes stay in the
 * hardware FIFO until this loop takes them, so the FIFO fills when the
 * application is busy. An ISR draining into a ring buffer does the opposite:
 * it empties the FIFO as fast as the line delivers, and the overflow moves
 * into the ring buffer where it is discarded with no signal to anyone. That
 * was observed during development and is why this is polled.
 *
 * Be precise about what a full FIFO actually accomplishes, because it differs
 * by transport:
 *
 *   QEMU  - the UART is a socket chardev. When the guest FIFO is full QEMU
 *           stops reading the TCP socket, the receive window closes and the
 *           host's own send() blocks. That is TCP flow control, and it is
 *           genuine end-to-end backpressure.
 *
 *   Board - hardware flow control is NOT enabled (no hw-flow-control in the
 *           devicetree, no RTS/CTS Kconfig), so a full FIFO asserts nothing on
 *           the wire. Bytes arriving after it fills are lost. The FIFO is 16
 *           bytes on ns16550 and 128 on ESP32-S3, and at 115200 baud a byte
 *           lands every 86 us - so roughly 390 bytes would arrive during a
 *           33 ms inference if a host kept sending through it.
 *
 * On hardware the protection is therefore the protocol, not the peripheral:
 * READY is emitted only when this loop is ready to receive, and the
 * organisers' send_image.py waits for it. A host that ignored the handshake
 * and streamed continuously would overrun a real UART. Enabling RTS/CTS would
 * close that gap, but it needs wiring the evaluation harness may not carry and
 * could not be tested here, so it is documented rather than assumed.
 *
 * Decoding one hex character is a few instructions against 86 us per character
 * at 115200 baud, so while this loop is running it has three orders of
 * magnitude of headroom; the exposure is confined to the inference window.
 */
/* Print basis points (0..10000) as "dd.dd". */
static void uart_write_pct(uint32_t bp)
{
	uart_write_u32(bp / 100U);
	uart_poll_out(uart_dev, '.');
	uart_poll_out(uart_dev, (unsigned char)('0' + ((bp / 10U) % 10U)));
	uart_poll_out(uart_dev, (unsigned char)('0' + (bp % 10U)));
}

static int rx_getc(uint8_t *c, int timeout_ms)
{
	int64_t deadline = k_uptime_get() + timeout_ms;
	uint32_t spins = 0;

	for (;;) {
		if (uart_poll_in(uart_dev, c) == 0) {
			return 0;
		}
		if (++spins >= 2000U) {
			spins = 0;
			if (k_uptime_get() > deadline) {
				return -1;
			}
			k_yield();
		}
	}
}

/* ------------------------------------------------------------------ */
/* Hex stream reception                                                */
/* ------------------------------------------------------------------ */

static inline int hex_val(uint8_t c)
{
	if (c >= '0' && c <= '9') return c - '0';
	if (c >= 'a' && c <= 'f') return c - 'a' + 10;
	if (c >= 'A' && c <= 'F') return c - 'A' + 10;
	return -1;
}

static inline uint8_t up(uint8_t c)
{
	return (c >= 'a' && c <= 'z') ? (uint8_t)(c - 'a' + 'A') : c;
}

/* Returns bytes received, or negative on error.
 *  -1 timeout, -2 overflow, -3 odd number of hex digits */
static int receive_payload(void)
{
	static const char stop[] = "STOP";
	size_t nbytes = 0;
	int hi = -1;
	int match = 0;
	int overflow = 0;
	int started = 0;
	uint8_t c;

	for (;;) {
		if (rx_getc(&c, started ? RX_IDLE_TIMEOUT_MS : RX_ARM_TIMEOUT_MS) != 0) {
			if (!started) {
				/* Still idle. Re-announce READY so a host that attaches
				 * after the first announcement - a fresh run of
				 * send_image.py on an already-running board - sees the
				 * handshake immediately instead of waiting out its
				 * 30 s timeout. */
				uart_write_str("READY\r\n");
				continue;
			}
			return -1;
		}
		started = 1;

		/* End-of-stream token, matched case-insensitively. */
		if (up(c) == (uint8_t)stop[match]) {
			match++;
			if (match == 4) {
				break;
			}
			continue;
		}
		match = (up(c) == (uint8_t)stop[0]) ? 1 : 0;

		int v = hex_val(c);
		if (v < 0) {
			continue;               /* whitespace / CR / LF */
		}
		if (hi < 0) {
			hi = v;
		} else {
			if (nbytes < TSR_PAYLOAD_MAX) {
				payload[nbytes++] = (uint8_t)((hi << 4) | v);
			} else {
				overflow = 1;
			}
			hi = -1;
		}
	}

	if (overflow) {
		return -2;
	}
	if (hi >= 0) {
		return -3;
	}
	return (int)nbytes;
}

/* ------------------------------------------------------------------ */

int main(void)
{
	int n, idx, rc;

	if (!device_is_ready(uart_dev)) {
		return -1;
	}

	/* Version line. Printed once at boot, before the protocol starts, so it
	 * cannot interfere with the host's parsing - and it makes "which build am
	 * I actually running?" answerable at a glance. */
	uart_write_str("\r\nTSR " TSR_BUILD_TAG " | 48x48x3 | 17 classes | arena ");
	uart_write_u32(tsr_arena_bytes());
	uart_write_str(" B | val " TSR_VAL_ACC "\r\n");

	for (;;) {
		uart_write_str("READY\r\n");

		n = receive_payload();
		if (n < 0) {
			uart_write_str(n == -1 ? "ERROR timeout waiting for data\r\n" :
				       n == -2 ? "ERROR payload too large\r\n" :
						 "ERROR odd number of hex digits\r\n");
			continue;
		}

		/* Host starts its inference timer on this line and stops it on the
		 * next one, so nothing may be emitted in between. */
		uart_write_str("RECEIVED ");
		uart_write_u32((uint32_t)n);
		uart_write_str(" bytes\r\n");

		rc = tsr_prepare_input(payload, (size_t)n, image);
		if (rc != 0) {
			uart_write_str("ERROR unexpected payload size ");
			uart_write_u32((uint32_t)n);
			uart_write_str("\r\n");
			continue;
		}

#ifdef CONFIG_TSR_REPORT_TIMING
		{
			uint32_t c0 = k_cycle_get_32();
			idx = tsr_predict(image, scores);
			last_infer_us = k_cyc_to_us_floor32(k_cycle_get_32() - c0);
		}
#else
		idx = tsr_predict(image, scores);
#endif

		/* class= carries the OFFICIAL challenge class ID (0, 2-7, 24, 43,
		 * 69, 70, 72, 83-87), not the 0-16 array index, as the challenge
		 * guide requires. */
		uart_write_str("PREDICTION class=");
		uart_write_u32((uint32_t)tsr_class_id(idx));
		uart_write_str(" label=");
		uart_write_str(tsr_class_names[idx]);
		uart_write_str(" confidence=");
		uart_write_pct(tsr_confidence_bp(scores, idx));
		uart_write_str("%\r\n");

#ifdef CONFIG_TSR_REPORT_TIMING
		/* Benchmark builds only. The host times inference as the gap between
		 * RECEIVED and the next line, so this is emitted AFTER PREDICTION,
		 * where it cannot contaminate that measurement - and it is compiled
		 * out of the shipping image entirely. */
		uart_write_str("TIMING infer_us=");
		uart_write_u32(last_infer_us);
		uart_write_str("\r\n");
#endif
	}

	return 0;
}
