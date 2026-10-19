# Building the firmware

## Why Zephyr is not in this repository

This repository holds a **Zephyr application**, not a Zephyr distribution.
`src/qemu_app/` and `src/esp32_app/` each contain what an application is
supposed to contain — `CMakeLists.txt`, `prj.conf`, `src/`, and for the S3 a
board overlay — and nothing else. The RTOS is fetched, not vendored, because:

- Zephyr plus the modules these two boards need is around **1 GB** of source,
  and the full module set is closer to **7 GB**. Neither belongs in a repository
  whose own content is 15 MB.
- It is separately licensed (Apache-2.0) and separately released. A vendored
  snapshot goes stale silently; a pinned revision does not.
- `west.yml` in the repository root pins the exact revision, which is a stronger
  reproducibility claim than a copied tree — you get the same source, and you
  can see at a glance which source that is.

This is the standard Zephyr *freestanding application* layout. The application
finds the RTOS through `ZEPHYR_BASE`, and `find_package(Zephyr REQUIRED HINTS
$ENV{ZEPHYR_BASE})` in each `CMakeLists.txt` is the whole of the coupling.

If you already have a Zephyr v4.4 workspace, you do not need `west.yml` at all —
skip to [Build](#build).

## Get the toolchain and the RTOS

You need Python 3.10+, CMake 3.20+, Ninja, `west`, and the Zephyr SDK.

```bash
pip install west

# from the repository root
west init -l .
west update          # ~1 GB, a few minutes
west zephyr-export
pip install -r ../zephyr/scripts/requirements.txt
```

`west init -l .` makes this repository the manifest repository and creates the
workspace **one level up**, so the layout ends as:

```
<workspace>/
├── tsr-esp32s3/      <- this repository, and the manifest
├── zephyr/           <- fetched, v4.4.0
├── modules/          <- fetched, six of them
└── .west/
```

`west.yml` pins Zephyr at tag **v4.4.0** and narrows `west update` to the six
modules these boards actually need: `hal_espressif`, `hal_xtensa`, `cmsis`,
`picolibc`, `segger`, `mbedtls`. That allowlist was verified by building the
ESP32-S3 image with those modules and no others. Delete the `name-allowlist`
block to fetch everything.

Then the **Zephyr SDK**, which carries the two compilers this project needs:
`x86_64-zephyr-elf` for QEMU and `xtensa-espressif_esp32s3_zephyr-elf` for the
S3. The firmware here was built with **SDK 1.0.1**; install it from the
[sdk-ng releases](https://github.com/zephyrproject-rtos/sdk-ng/releases),
following the instructions there for your platform, and run its `setup.sh`
(or `setup.cmd`) so CMake can find it.

A minimal install is enough — only those two toolchains are used.

On Windows, `scripts/setup_environment.ps1` finds an existing SDK and west
workspace and exports the environment for the current shell:

```powershell
. .\scripts\setup_environment.ps1
```

## Build

```bash
# QEMU
cmake -B build_qemu -GNinja -DBOARD=qemu_x86 src/qemu_app
ninja -C build_qemu

# ESP32-S3
cmake -B build_esp -GNinja -DBOARD=esp32s3_devkitc/esp32s3/procpu src/esp32_app
ninja -C build_esp
```

Expect this memory report from the S3 build:

```
Memory region         Used Size  Region Size  %age Used
           FLASH:      199700 B   16776960 B      1.19%
     iram0_0_seg:       37828 B     415492 B      9.10%
     dram0_0_seg:       69808 B     399108 B     17.49%
     irom0_0_seg:       21472 B        32 MB      0.06%
     drom0_0_seg:      134164 B        32 MB      0.40%
```

Those five figures reproduce exactly. The **binary does not** hash-match the
prebuilt one in `src/esp32_app/binaries/` — Zephyr builds are not byte
reproducible across build paths, host toolchains and revisions. Same size, same
map, different bytes. The `sha256` recorded in each app's README identifies the
shipped artifact; it is not a target for your own build to hit.

## Run

```bash
# one image through QEMU
python scripts/send_image_qemu.py build_qemu/zephyr/zephyr.elf samples/class69_height_limit.png

# all 364 over the UART protocol
python scripts/qemu_uart_eval.py --elf build_qemu/zephyr/zephyr.elf --mode rgb48

# the full gate, including both builds
python scripts/verify_all.py
```

Flashing an ESP32-S3 is [`docs/FLASHING.md`](FLASHING.md).

## Building nothing at all

The numerical claims do not need Zephyr, QEMU or a board. The runtime is plain
C, so a host compiler is enough:

```bash
gcc -O2 -std=gnu99 -o host_test test/host_test.c \
    src/qemu_app/src/tsr_model.c src/qemu_app/src/tsr_model_data.c -lm
python -c "import numpy as np; open('va.raw','wb').write(np.load('out/data.npz')['Xva'].tobytes())"
./host_test va.raw > host_out.txt
python scripts/ci_bit_exact.py
```

That is exactly what `.github/workflows/verify.yml` does on every push.
