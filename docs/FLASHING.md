# Running on the ESP32-S3

Written against `esp32s3_devkitc/esp32s3/procpu`, Zephyr 4.4. Everything here is
derived from the board files and this project's build output; the one thing
nobody has done yet is put it on a physical board, so treat the first run as the
test it is.

Paths below assume the office machine:

```
west workspace   C:\_DevTools\zephyr
Zephyr source    C:\_DevTools\zephyr\zephyr
project          C:\src\tsr-esp32s3
```

---

## The short way: one script

On a PC with nothing but Python, from the submission folder:

```powershell
python -m pip install esptool pyserial pillow numpy
python scripts\flash_and_test.py
```

No Zephyr, no SDK, no toolchain — the firmware in
`src\esp32_app\binaries\zephyr.bin` is already built. The script finds the
USB-serial adapter, confirms the chip really is an ESP32-S3 (and refuses to
flash if it is not, because the image offset differs by chip), flashes at
`0x0`, waits for the boot banner and classifies one known sample:

```
[1] checking this PC
  python packages: ok
  firmware      : 199,700 B  sha256 224f80b09168f0c1...
  image header  : flash size 16MB
[2] identifying the board
  chip          : ESP32-S3
  flash         : 16MB
[3] flashing
[4] verifying over UART
  TSR 2026.08.28-t3f | 48x48x3 | 17 classes | arena 9984 B | val 99.73%
  single image  : class=69 confidence=100.00%  (correct)
Done.
```

Add `--full` to score all 364 validation images on the board afterwards
(about ten minutes; expect `0.9973 (363/364)`). `--port COM11` if auto-detect
picks the wrong adapter. `--no-flash` tests whatever is already on the board.

Everything below is the same thing done by hand, plus the ESP32-vs-S3
differences that matter if you use a different board.

---

## Which chip do you actually have?

Run this first. It takes five seconds and it decides three of the commands
below:

```powershell
python -m esptool --port COM12 flash-id
```

| it says | board target | flash offset | `--chip` |
|---|---|---|---|
| `ESP32-S3` | `esp32s3_devkitc/esp32s3/procpu` | **0x0** | `esp32s3` |
| `ESP32-D0WD` / `ESP32-D0WDQ6` (plain ESP32) | `esp32_devkitc/esp32/procpu` | **0x1000** | `esp32` |

**The flash offset is not cosmetic.** The plain ESP32's ROM bootloader looks for
the image at 0x1000; the S3's looks at 0x0. Zephyr encodes this in the board's
partition table — `partitions_0x1000_amp.dtsi` for ESP32,
`partitions_0x0_amp.dtsi` for S3 — and writing to the wrong one gives you a
board that flashes successfully and then does nothing, with no error to explain
why.

Also note the flash size esptool reports (4MB or 8MB) and use it in
`--flash_size` below.

**The challenge specifies ESP32-S3 WROOM, and Phase 2 will be evaluated on an
S3.** A plain ESP32 is a useful proxy — same Xtensa ABI and alignment rules,
same UART driver, so a numerical or portability bug found there would almost
certainly apply to the S3 as well — but it does not validate the S3 image. If
you test on a plain ESP32, report it as what it is.

---

## 0. Which USB socket

The DevKitC has **two** USB-C sockets and it matters which one you use.

| socket | silkscreen | what it is |
|---|---|---|
| the one to use | **UART** | USB-to-serial bridge - GPIO43/44 on S3, GPIO1/3 on plain ESP32 |
| not this one | **USB** | native USB-serial-JTAG |

This application's console is `uart0` on GPIO43 (TX) / GPIO44 (RX) at 115200,
and the board's devicetree has `&usb_serial { status = "disabled"; }`. Plug into
the **UART** socket and you get a COM port; plug into **USB** and the firmware
will appear dead because nothing is listening on the pins it is talking to.

Find the port number in Device Manager under *Ports (COM & LPT)* — it will be a
Silicon Labs CP210x or a WCH CH34x. Note the number; `COM12` is used as the
example below.

---

## 1. Set up the shell

```powershell
cd C:\src\tsr-esp32s3
. .\setup_environment.ps1
```

Dot-sourced, so the environment survives into your shell. It prints what it
found. The line that matters here is:

```
ESP_IDF_PATH   C:\_DevTools\zephyr\modules\hal\espressif
```

If that says `<-- MISSING`, the build will stop with
`Assertion failed: ESP_IDF_PATH is not set`, and the fix is to point
`-ZephyrWorkspace` at the folder that really contains `modules\hal\espressif`.

---

## 2. Build

```powershell
# ESP32-S3
cmake -B build_esp -GNinja -DBOARD=esp32s3_devkitc/esp32s3/procpu src\esp32_app
ninja -C build_esp

# plain ESP32 (ESP32-D0WD)
cmake -B build_esp32 -GNinja -DBOARD=esp32_devkitc/esp32/procpu src\qemu_app   # LX6 proxy board
ninja -C build_esp32
```

Each SoC needs its own Xtensa toolchain in the Zephyr SDK. If cmake cannot find
a compiler, add it:

```powershell
cd $env:ZEPHYR_SDK_INSTALL_DIR
.\setup.cmd -t xtensa-espressif_esp32_zephyr-elf        # plain ESP32
.\setup.cmd -t xtensa-espressif_esp32s3_zephyr-elf      # ESP32-S3
```

Expect a memory report at the end. From the reference build:

```
dram0_0_seg    66,356 B / 399,108 B   16.6%
iram0_0_seg    34,760 B / 415,492 B    8.4%
```

The output to flash is `build_esp\zephyr\zephyr.bin`.

**If cmake complains about missing binary blobs**, fetch them once:

```powershell
cd C:\_DevTools\zephyr
west blobs fetch hal_espressif
cd C:\src\tsr-esp32s3
```

This project uses no WiFi or Bluetooth, so it should not need them — but the
Espressif HAL sometimes asks anyway.

---

## 3. Flash

The build sets `CONFIG_ESP_SIMPLE_BOOT=y`, so there is no separate bootloader
image: one binary at offset `0x0` is the whole flash operation.

**Option A — esptool directly.** Simplest, and it does not care where you run it
from:

```powershell
# ESP32-S3, image at 0x0
python -m esptool --chip esp32s3 --port COM12 --baud 460800 write-flash `
    --flash-mode dio --flash-freq 80m --flash-size 16MB `
    0x0 src\esp32_app\binaries\zephyr.bin

# plain ESP32, image at 0x1000 - the offset is the part that matters
python -m esptool --chip esp32 --port COM12 --baud 460800 write-flash `
    --flash-mode dio --flash-freq 40m --flash-size 4MB `
    0x1000 build_esp32\zephyr\zephyr.bin
```

**Option B — `west flash`.** `flash` is a west *extension* command, so it only
exists inside the workspace — same trap as `west build`:

```powershell
cd C:\_DevTools\zephyr
west flash -d C:\src\tsr-esp32s3\build_esp
cd C:\src\tsr-esp32s3
```

Either way, if esptool cannot get the chip into download mode
(`Failed to connect to ESP32-S3`), do it by hand: **hold BOOT, tap RESET,
release BOOT**, then re-run. Close Tera Term first — Windows will not share a
COM port.

---

## If it sits at "waiting for the boot banner"

The script pulses the board's reset line before listening, so this should not
normally happen. When it does, it waits 40 s and then prints the checklist
below. In rough order of likelihood:

1. **Wrong USB socket.** The DevKitC has two. The firmware's console is `uart0`
   on GPIO43/44 and the board devicetree disables the native USB serial, so on
   the socket marked **USB** nothing will ever appear. Move to **UART**.
2. **Still in download mode after flashing.** esptool normally resets out of it;
   if it did not, tap **RESET** and re-run with `--no-flash`.
3. **Nothing was flashed, or it went to the wrong offset.** The S3 image loads
   at `0x0`. An image written at `0x1000` (the original ESP32 offset) programs
   successfully and then does nothing.
4. **The adapter cannot drive the reset lines.** The script says so if the pulse
   fails. Use `--no-reset` and press RESET by hand as it starts listening.

To see the port raw, open Tera Term on it at **115200 8N1** and press RESET. A
working board prints the ESP-ROM line, the Zephyr boot log, then:

```
TSR 2026.08.28-t3f | 48x48x3 | 17 classes | arena 9984 B | val 99.73%
READY
```

Nothing at all means socket or driver. Garbage means the wrong baud. ROM
messages but no `TSR` line means the flash did not take.

Close Tera Term before re-running anything — Windows gives a COM port to one
process at a time.

---

## 4. First contact

Reset the board (RESET button), then open Tera Term on **COM12 at 115200, 8-N-1**.
You should see:

```
*** Booting Zephyr OS build v4.4.0 ***
TSR 2026.08.28-t3f | 48x48x3 | 17 classes | arena 9984 B | val 99.73%
READY
```

That banner is the whole system alive on hardware: firmware booted, model
linked, arena allocated, UART talking. If you get ROM garbage instead of the
banner, you are on the wrong socket or the wrong baud.

Close Tera Term before running any script — one process per COM port.

---

## 5. One image, using the organisers' own script

```powershell
python scripts\send_image.py COM12 samples\class69_height_limit.png
```

`scripts\send_image.py` is the organisers' script, unmodified — the same one the
evaluation team will run. Expected:

```
READY
RECEIVED 6912 bytes
PREDICTION class=69 label=Height limit confidence=100.00%
```

Any image works; it resizes to 48×48 for you:

```powershell
python scripts\send_image.py COM12 C:\path\to\my_photo.jpg
```

---

## 6. Score all 364 images on the board

This is the run worth doing, because it turns "the image links and the footprint
is known" into a measured hardware number:

```powershell
python scripts\serial_eval.py COM12 --limit 20      # smoke test first
python scripts\serial_eval.py COM12                 # the full sweep
```

Expect **99.73 % (363/364)**, the same as QEMU — the runtime is bit-exact, so
the board should agree image for image. The single expected error is class 83
against 84, steep descent versus steep ascent, which are mirror images.

It takes about ten minutes: one image is ~14 KB of hex, roughly 1.3 s on the
wire at 115200, plus inference. Progress prints every 25 images and any
mismatches are listed at the end.

**If the hardware number differs from 99.73 %, stop and tell me.** The C runtime
is verified bit-exact against TensorFlow's reference kernels on x86; a different
answer on Xtensa would mean something real — an endianness or alignment
assumption, or a compiler difference in the `-O3` kernel — and it would be worth
knowing before submission rather than after.

---

## 7. Tera Term by hand, on hardware

Same as the QEMU walkthrough in `RUN_ON_YOUR_LAPTOP.md` §4b, except you connect
to the serial port instead of a TCP socket:

1. Tera Term → File → New connection → **Serial**, port COM12.
2. Setup → Serial port → **115200**, 8 bit, none, 1 stop, **flow control: none**.
3. `python scripts\make_hex_file.py samples\class69_height_limit.png sign69.txt`
4. File → Send file…, tick **Binary**, choose `sign69.txt`.

---

## Two things to know before you trust a result

**No hardware flow control.** RTS/CTS is not enabled (no `hw-flow-control` in
the devicetree), so the FIFO cannot signal the sender. The protection is the
`READY` handshake, and both `send_image.py` and `serial_eval.py` wait for it. A
terminal that pastes a huge block without waiting can overrun the 128-byte FIFO
during the inference window — that is why step 7 says send a *file* rather than
pasting.

**Timing measured over the wire is not inference time.** The gap between
`RECEIVED` and `PREDICTION` as seen by the host includes serial latency. For a
figure to quote, use the MAC count — **6,885,440 int8 MACs per image** — and say
which host any wall-clock number came from.
