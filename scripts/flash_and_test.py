#!/usr/bin/env python3
"""
Flash the prebuilt ESP32-S3 firmware and prove it works. One command, on a PC
with nothing but Python installed.

    python scripts/flash_and_test.py                 # find the board, flash, test
    python scripts/flash_and_test.py --port COM11    # if auto-detect picks wrong
    python scripts/flash_and_test.py --full          # ...and score all 364 images
    python scripts/flash_and_test.py --no-flash      # just test what is on the board

No Zephyr, no SDK, no toolchain: the binary in src/esp32_app/binaries/ is
already built. Only Python packages are needed and the script tells you the one
pip line if they are missing.

Target: ESP32-S3-DevKitC-1 N16R8 (ESP32-S3-WROOM-1, 16 MB flash). The script
refuses to flash anything that does not identify as an ESP32-S3, because the
image offset differs by chip - S3 takes it at 0x0, the original ESP32 at
0x1000 - and flashing to the wrong offset gives a board that programs fine and
then does nothing.

Use the USB socket marked UART, not the one marked USB.
"""
import argparse
import hashlib
import os
import re
import subprocess
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "scripts"))

FIRMWARE = os.path.join(_REPO, "src", "esp32_app", "binaries", "zephyr.bin")
SAMPLE = os.path.join(_REPO, "samples", "class69_height_limit.png")
EXPECT_CLASS = 69
BUILD_TAG = "2026.08.28-t3f"
PIP = "python -m pip install esptool pyserial pillow numpy"


def die(msg, code=1):
    print("\nERROR: " + msg, file=sys.stderr)
    sys.exit(code)


def step(n, text):
    print(f"\n[{n}] {text}", flush=True)


# --------------------------------------------------------------- prechecks --
def check_deps():
    missing = []
    for mod, why in (("serial", "pyserial"), ("PIL", "pillow"),
                     ("numpy", "numpy"), ("esptool", "esptool")):
        try:
            __import__(mod)
        except ImportError:
            missing.append(why)
    if missing:
        die(f"missing Python packages: {', '.join(missing)}\n       {PIP}")
    print("  python packages: ok")


def check_firmware():
    if not os.path.exists(FIRMWARE):
        die(f"firmware not found: {FIRMWARE}\n"
            f"       Run this from the submission folder, or rebuild with\n"
            f"       cmake -B build_esp -GNinja "
            f"-DBOARD=esp32s3_devkitc/esp32s3/procpu src/esp32_app")
    blob = open(FIRMWARE, "rb").read()
    if blob[0] != 0xE9:
        die("firmware does not start with the 0xE9 ESP image magic byte - "
            "wrong file?")
    sizes = {0: "1MB", 1: "2MB", 2: "4MB", 3: "8MB", 4: "16MB", 5: "32MB"}
    hdr = sizes.get(blob[3] >> 4, "?")
    print(f"  firmware      : {len(blob):,} B  sha256 "
          f"{hashlib.sha256(blob).hexdigest()[:16]}...")
    print(f"  image header  : flash size {hdr}")
    if hdr != "16MB":
        print(f"  NOTE: header says {hdr}, this board is a 16 MB N16R8. It will "
              f"still boot, but the SPI flash size register will be short.")
    return hdr


# ------------------------------------------------------------------ esptool --
def esptool_run(args, capture=True):
    """esptool 5 renamed the commands; fall back to the v4 spelling."""
    base = [sys.executable, "-m", "esptool"]
    r = subprocess.run(base + args, capture_output=capture, text=True)
    if r.returncode != 0 and capture and "Unknown command" in (r.stdout + r.stderr):
        legacy = [a.replace("-", "_") if a in ("flash-id", "write-flash") else a
                  for a in args]
        r = subprocess.run(base + legacy, capture_output=capture, text=True)
    return r


def find_port(explicit):
    if explicit:
        return explicit
    from serial.tools import list_ports
    cands = []
    for p in list_ports.comports():
        blob = f"{p.description} {p.manufacturer or ''} {p.hwid}".lower()
        if any(k in blob for k in ("ch340", "ch34", "cp210", "usb-serial",
                                   "uart", "silicon labs", "wch")):
            cands.append(p)
    if not cands:
        allp = list(list_ports.comports())
        if not allp:
            die("no serial ports found. Plug the board into the socket marked "
                "UART (not USB) and check the driver in Device Manager.")
        die("no USB-serial adapter recognised. Ports seen:\n       " +
            "\n       ".join(f"{p.device}  {p.description}" for p in allp) +
            "\n       Pass one explicitly with --port")
    if len(cands) > 1:
        die("more than one USB-serial adapter:\n       " +
            "\n       ".join(f"{p.device}  {p.description}" for p in cands) +
            "\n       Pass the right one with --port")
    print(f"  port          : {cands[0].device}  ({cands[0].description})")
    return cands[0].device


def identify(port):
    r = esptool_run(["--port", port, "flash-id"])
    out = r.stdout + r.stderr
    if r.returncode != 0:
        die("esptool could not talk to the board:\n" + out.strip()[-600:] +
            "\n       Close Tera Term / PuTTY first - Windows gives a COM port "
            "to one process at a time.\n"
            "       If it still fails: hold BOOT, tap RESET, release BOOT, retry.")
    chip = re.search(r"Chip type:\s*(\S+)", out) or re.search(r"Chip is (\S+)", out)
    size = re.search(r"Detected flash size:\s*(\S+)", out)
    chip = chip.group(1) if chip else "?"
    size = size.group(1) if size else "?"
    print(f"  chip          : {chip}")
    print(f"  flash         : {size}")
    if "ESP32-S3" not in chip.upper():
        die(f"this is not an ESP32-S3 (reads as {chip}).\n"
            f"       The S3 image loads at 0x0; an original ESP32 loads at "
            f"0x1000, and flashing the wrong offset\n"
            f"       gives a board that programs successfully and then does "
            f"nothing. Refusing to flash.")
    return size


def flash(port, size):
    fs = size if size in ("4MB", "8MB", "16MB", "32MB") else "16MB"
    r = esptool_run(["--chip", "esp32s3", "--port", port, "--baud", "460800",
                     "write-flash", "--flash-mode", "dio", "--flash-freq", "80m",
                     "--flash-size", fs, "0x0", FIRMWARE], capture=False)
    if r.returncode != 0:
        die("flashing failed. Close any terminal holding the port, then hold "
            "BOOT, tap RESET, release BOOT and retry.")


# ------------------------------------------------------------------- verify --
def open_link(port, baud, tcp, do_reset=True):
    from serial_eval import SerialLink, TcpLink
    if tcp:
        return TcpLink(tcp)
    link = SerialLink(port, baud)
    if do_reset:
        # Pulse the auto-reset circuit so the board reboots and prints its
        # banner, instead of hoping it happens to be at a point where it will
        # say something. On a DevKitC the USB-serial chip drives EN through
        # RTS and GPIO0 through DTR:
        #     DTR de-asserted -> GPIO0 high  -> normal boot, not download mode
        #     RTS asserted    -> EN low      -> held in reset
        # Releasing RTS starts the chip running the application.
        try:
            link.s.dtr = False       # GPIO0 high: run the app, not the ROM loader
            link.s.rts = True        # EN low: in reset
            time.sleep(0.15)
            link.s.reset_input_buffer()
            link.s.rts = False       # EN high: run
            time.sleep(0.30)
        except Exception as e:
            # Some adapters do not expose the modem lines. Not fatal - the
            # firmware re-announces READY every three seconds anyway.
            print(f"  (could not pulse reset: {e}; relying on the READY re-arm)")
    return link


def verify(port, baud, tcp, full, do_reset=True):
    from serial_eval import Board, CLASS_IDS
    import numpy as np

    link = open_link(port, baud, tcp, do_reset)
    b = Board(link)
    print("  reset pulsed, waiting for the boot banner ..."
          if do_reset and not tcp else "  waiting for the board ...",
          flush=True)
    banner, deadline = None, time.time() + 40
    while time.time() < deadline:
        line = b.readline(2.0)
        if line is None:
            continue
        if line.startswith("TSR "):
            banner = line
            print("  " + line)
        if line.endswith("READY"):
            break
    else:
        link.close()
        die("no READY from the board after 40 s. In rough order of likelihood:\n"
            "       1. Wrong USB socket. Use the one marked UART, not USB - the\n"
            "          firmware's console is uart0 on GPIO43/44 and the board\n"
            "          devicetree disables the native USB serial.\n"
            "       2. Still in download mode after flashing. Tap RESET on the\n"
            "          board and re-run with --no-flash.\n"
            "       3. Nothing was flashed, or it went to the wrong offset. The\n"
            "          S3 image loads at 0x0. Re-run without --no-flash.\n"
            "       4. This adapter cannot drive the reset lines; try\n"
            "          --no-reset and press RESET by hand as it starts.\n"
            "       To see the raw port, open Tera Term on it at 115200 8N1 and\n"
            "       press RESET: you should get the Zephyr boot log.")
    if banner and BUILD_TAG not in banner:
        print(f"  WARNING: expected build tag {BUILD_TAG}, banner says otherwise. "
              f"Old firmware?")
    elif not banner:
        print("  (banner missed - it is printed once at power-on; harmless)")

    from PIL import Image
    im = Image.open(SAMPLE)
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        im = Image.alpha_composite(Image.new("RGBA", im.size, (255,)*4), im)
    payload = im.convert("RGB").resize((48, 48)).tobytes()
    pred, nb, conf = b.infer(payload)
    if pred is None:
        link.close(); die("the board did not answer with a prediction.")
    ok = (pred == EXPECT_CLASS)
    print(f"  single image  : class={pred} confidence={conf}%  "
          f"({'correct' if ok else 'EXPECTED ' + str(EXPECT_CLASS)})")
    if not ok:
        link.close(); die("wrong class on a known-good sample.")
    b.wait_ready(timeout=20)

    if not full:
        link.close()
        return

    data = os.path.join(_REPO, "out", "data.npz")
    if not os.path.exists(data):
        link.close(); print("  (out/data.npz missing - skipping the full sweep)")
        return
    d = np.load(data); X, y = d["Xva"], d["yva"]
    print(f"  full sweep    : {len(X)} images, about 10 minutes ...", flush=True)
    good = n = retried = 0
    t0 = time.time()
    for i in range(len(X)):
        pred = None
        for attempt in range(4):
            if attempt:
                retried += 1
                if not b.wait_ready(timeout=15):
                    break
            pred, _, _ = b.infer(X[i].tobytes())
            if pred is not None:
                break
        if pred is None:
            print(f"  image {i}: no prediction - stopping"); break
        n += 1
        good += (pred == CLASS_IDS[y[i]])
        if not b.wait_ready(timeout=30):
            print("  lost READY - stopping"); break
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(X)}  acc={good/(i+1):.4f}", flush=True)
    link.close()
    print(f"\n  accuracy      : {good/n:.4f} ({good}/{n})"
          f"   expected 0.9973 (363/364)")
    if retried:
        print(f"  link          : {retried} frame(s) re-sent (no hardware flow "
              f"control on this link)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None, help="COM11, /dev/ttyUSB0, ...")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--full", action="store_true",
                    help="after flashing, score all 364 validation images")
    ap.add_argument("--no-flash", action="store_true",
                    help="skip flashing, just test what is already on the board")
    ap.add_argument("--no-reset", action="store_true",
                    help="do not pulse DTR/RTS to reboot the board before testing")
    ap.add_argument("--tcp", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()

    print("ESP32-S3 flash and test  —  AI Arena 2026 Track 1")
    step(1, "checking this PC")
    check_deps()
    check_firmware()

    port = a.port
    if not a.tcp:
        port = find_port(a.port)
        if not a.no_flash:
            step(2, "identifying the board")
            size = identify(port)
            step(3, "flashing")
            flash(port, size)
            print("  waiting for reset ...")
            time.sleep(2.0)
        else:
            print("  --no-flash: leaving the firmware as it is")

    step(4 if not a.no_flash else 2, "verifying over UART")
    verify(port, a.baud, a.tcp, a.full, do_reset=not a.no_reset)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
