# esp32_app — the evaluation target

Zephyr application for **ESP32-S3-DevKitC-1 N16R8** (ESP32-S3-WROOM-1, 16 MB
flash, 8 MB octal PSRAM), board target `esp32s3_devkitc/esp32s3/procpu`.

The source here is **byte-identical** to `../qemu_app/src/` — one runtime, two
targets. `scripts/verify_all.py` checks that with sha256 on every file and
fails the gate if they ever diverge, so a result measured under QEMU remains
evidence about the board.

The only differences are in the build configuration:

- `prj.conf` adds `CONFIG_ESPTOOLPY_FLASHSIZE_16MB=y`.
- `boards/esp32s3_devkitc_esp32s3_procpu.overlay` sets `flash0` to 16 MB.
  esptool takes the flash size for the image header from the devicetree node,
  not from Kconfig, and the in-tree board declares 8 MB. The header value
  configures the SPI flash size register at boot.

PSRAM is deliberately not enabled. The inference arena is 9,984 bytes of
internal SRAM; nothing here needs external RAM.

## Prebuilt binaries

`binaries/`, built from this source on 2026-08-31:

| file | bytes | sha256 |
|---|---:|---|
| `zephyr.bin` | 199,700 | `224f80b09168f0c1d52082b248c6cb6017e8527af97f0d45ea31fd9831afe339` |
| `zephyr.elf` | 2,011,068 | `7f96f28460920beab4e7ce0df692fb0aec67ab20e76f5a1ab5deaad5c89f743b` |

Zephyr memory report for this image:

```
FLASH        199,700 B / 16,776,960 B    1.19 %
iram0_0_seg   37,828 B /    415,492 B    9.10 %
dram0_0_seg   69,808 B /    399,108 B   17.49 %
irom0_0_seg   21,472 B
drom0_0_seg  134,164 B
```

The inference arena is present in this image at the expected size, read back
from the symbol table rather than assumed: `tsr_buf_a` 4,608 B + `tsr_buf_b`
2,304 B + `tsr_scratch` 3,072 B = **9,984 B**.

## Flash it

The scripted way, on a PC with only Python installed:

```
python -m pip install esptool pyserial pillow numpy
python scripts/flash_and_test.py            # add --full for all 364 images
```

By hand:

`CONFIG_ESP_SIMPLE_BOOT=y`, so there is no separate bootloader image: one
binary at offset **0x0** is the whole operation.

```
python -m esptool --chip esp32s3 --port COM<n> --baud 460800 write-flash \
    --flash-mode dio --flash-freq 80m --flash-size 16MB \
    0x0 src/esp32_app/binaries/zephyr.bin
```

Use the USB socket marked **UART**, not the one marked USB: the board
devicetree disables `usb_serial` and puts the console on `uart0`, GPIO43/44, at
115200 8N1. If esptool cannot enter download mode, hold **BOOT**, tap
**RESET**, release **BOOT**.

## Run it

```
python scripts/send_image.py COM<n> samples/class69_height_limit.png
python scripts/serial_eval.py COM<n>                 # all 364 images
```

Expect the banner, then `99.73 % (363/364)`.

## Rebuild it

```
cmake -B build_esp -GNinja -DBOARD=esp32s3_devkitc/esp32s3/procpu src/esp32_app
ninja -C build_esp
```

`ESP_IDF_PATH` must point at `<west workspace>/modules/hal/espressif` or cmake
stops with `Assertion failed: ESP_IDF_PATH is not set`. `setup_environment.ps1`
sets it.

## Status

This image has been **built and linked, never flashed**. The hardware run
reported in `docs/TECHNICAL_WRITEUP.md` was on an ESP32-D0WD-V3 bench proxy — a
different SoC, which also takes its image at 0x1000 rather than 0x0.
