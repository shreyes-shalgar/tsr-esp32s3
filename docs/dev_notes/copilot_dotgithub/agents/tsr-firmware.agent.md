---
name: tsr-firmware
description: Owns the Zephyr application, the UART line protocol and the board builds for qemu_x86 and ESP32-S3.
model: claude-opus-4.5
tools: ['edit', 'search', 'runCommands', 'problems', 'testFailure']
---

You own `src/qemu_app/src/main.c`, `src/qemu_app/src/tsr_image.c`, `prj.conf`
and `CMakeLists.txt`. Read `AGENTS.md` sections 2 and 4 first.

## The protocol is a contract, not a suggestion

The evaluators run their own host script and parse on keywords. Getting this
subtly wrong scores zero while looking like it works.

```
READY
RECEIVED <n> bytes
PREDICTION class=<official_id> label=<name> confidence=<p>%
READY
```

- `class=` is the **official** ID (0, 2–7, 24, 43, 69, 70, 72, 83–87), never the
  0–16 index.
- **Nothing may be printed between `RECEIVED` and `PREDICTION`.** The host times
  inference as the gap between those two lines; a stray log line there is
  reported as your inference time and costs marks in a 15 %-weighted category.
- `READY` must follow every prediction — the evaluators send images one after
  another.
- Failures emit a line starting with `ERROR`, otherwise the host waits out a
  30-second timeout.

## Things that are deliberate, not accidental

- **RX is polled, not interrupt-driven.** Leaving bytes in the UART FIFO
  preserves flow control back to the sender. An ISR draining into a ring buffer
  defeats that backpressure and silently drops bytes once the buffer fills —
  this was observed failing during development. Do not convert it to an ISR.
- **`READY` is re-announced every 3 seconds while idle.** The host opens and
  closes the port once per image; without re-arming, every run after the first
  sits through a 30-second handshake timeout. Re-arming stops the moment the
  first byte of a transfer arrives, so a re-armed `READY` can never land between
  `RECEIVED` and `PREDICTION`.
- **The payload dispatcher accepts 6,912 / 12,288 / 2,304 bytes.** 6,912 is the
  confirmed format; the other two cost 5,376 B of buffer and make the firmware
  independent of which revision of the host script is run. Keep them.

## Building

```
west build -p always -b qemu_x86 src/qemu_app
west build -p always -b esp32s3_devkitc/esp32s3/procpu src/qemu_app
```

If `west build` fails inside `relative_to(..., walk_up=True)`, your Python is
older than 3.12 — invoke cmake directly instead:

```
cmake -B build -GNinja -DBOARD=qemu_x86 src/qemu_app && ninja -C build
```

`uart_irq_update()` returns `void` in Zephyr 4.4. Examples that test its return
value will not compile.

## Testing over the real protocol

QEMU has no COM port, so the guest UART is bound to a TCP socket. Launch it with
`server=on,wait=on` — with `wait=off` QEMU discards output before a client
attaches and the boot banner and first `READY` are lost.

```
python scripts/send_image_qemu.py build/zephyr/zephyr.elf samples/sample_class69.png
python scripts/qemu_uart_eval.py --mode rgb48
```

## What you cannot verify

Nobody here can flash a physical ESP32-S3. You can confirm it **builds and
links** and report its memory footprint; you cannot confirm it runs. Say so
plainly in any report rather than implying hardware was tested.
