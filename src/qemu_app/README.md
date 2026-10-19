# qemu_app — the reproducible demo

Zephyr application for `qemu_x86`. Same source as `../esp32_app/src/`,
byte-identical and checked by `scripts/verify_all.py`.

## Prebuilt binary

`binaries/`, built from this source on 2026-08-31:

| file | bytes | sha256 |
|---|---:|---|
| `zephyr.elf` | 551,904 | `448d92744b1b885b8e6a2a953f4df605a83b7118dc402065a0e20febc9995652` |

## Run it

```
python scripts/send_image_qemu.py src/qemu_app/binaries/zephyr.elf samples/class69_height_limit.png
python scripts/qemu_uart_eval.py --elf src/qemu_app/binaries/zephyr.elf --mode rgb48
```

The first sends one image and prints the prediction; the second scores all 364
validation images over the UART protocol and reports **99.73 % (363/364)**.

## Rebuild it

```
cmake -B build -GNinja -DBOARD=qemu_x86 src/qemu_app
ninja -C build
```

Both images print a build banner on boot, so which binary is running is
answerable at a glance:

```
TSR 2026.08.28-t3f | 48x48x3 | 17 classes | arena 9984 B | val 99.73%
```
