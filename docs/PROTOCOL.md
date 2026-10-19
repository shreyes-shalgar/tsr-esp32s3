# UART protocol

**115200 baud, 8N1, no flow control.** Hardware RTS/CTS is *not* enabled — see
"Backpressure" below for what that does and does not buy you.

## Sequence

```
board -> host : READY
host  -> board: <hex payload, any number of lines>
host  -> board: STOP
board -> host : RECEIVED <n> bytes
board -> host : PREDICTION class=<official_id> label=<name> confidence=<p>%
board -> host : READY
```

On any failure the board emits a line beginning with `ERROR` instead of
`PREDICTION`, so a host never has to wait out its own timeout:

```
ERROR timeout waiting for data
ERROR payload too large
ERROR odd number of hex digits
ERROR unexpected payload size <n>
```

While idle the board re-announces `READY` every 3 seconds, so a host that
attaches to the port late still finds the handshake.

## Payload

Pixel bytes, hex-encoded, uppercase or lowercase, newlines anywhere. Row-major,
interleaved RGB.

| bytes | meaning |
|---|---|
| **6,912** | 48 × 48 × 3 RGB — the challenge guide's format, and the one measured |
| 12,288 | 64 × 64 × 3 RGB — downscaled on the board |
| 2,304 | 48 × 48 grayscale — replicated to three channels |

Any other length is rejected with `ERROR unexpected payload size`.

## Confidence

`confidence=<p>%` is a percentage with two decimals, e.g. `confidence=99.87%`.
It is the softmax probability of the winning class, computed over the 17
dequantised int8 logits **on the output side only** — the inference path itself
is integer-only, and softmax is monotonic, so it changes nothing about which
class wins. It is uncapped and untempered; see `docs/RESULTS.md` for its
measured calibration.

## `class=` is the official dataset id, not the model index

The model has 17 outputs indexed 0–16. The board prints the challenge's class
id, which is sparse:

```
0, 2, 3, 4, 5, 6, 7, 24, 43, 69, 70, 72, 83, 84, 85, 86, 87
```

`tsr_class_id()` in `src/qemu_app/src/tsr_model.c` does the mapping. A host that
compares against the model index rather than the official id will see about 6 %
accuracy and conclude something is very wrong.

## Backpressure

Under QEMU the UART is a TCP socket chardev, so the transport itself applies
flow control: if the guest stops reading, the host's `sendall` blocks. That is a
property of the socket, **not** of the protocol, and it does not exist on real
hardware, where the UART has no RTS/CTS enabled and a host that transmits faster
than the board consumes will lose bytes.

What protects the hardware link instead is arithmetic: at 115200 baud a
character takes 86 µs, while decoding one hex character costs a few
instructions. Three orders of magnitude of headroom. The only window where the
board is not reading is the inference itself, which is why `STOP` precedes it
and `READY` follows it.

Across a 364-image hardware sweep there were zero re-sends.

## Host tools

| | |
|---|---|
| `scripts/send_image.py` | one image to a real serial port |
| `scripts/send_image_qemu.py` | one image, launching QEMU for you |
| `scripts/serial_eval.py` | the full 364-image sweep over a COM port |
| `scripts/qemu_uart_eval.py` | the full sweep under QEMU |
| `scripts/make_hex_file.py` | a hex file for Tera Term's *Send file (Binary)* |
