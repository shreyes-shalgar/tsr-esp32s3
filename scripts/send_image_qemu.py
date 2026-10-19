#!/usr/bin/env python3
"""
QEMU counterpart of the official send_image.py.

send_image.py talks to a physical COM port. QEMU has no COM port, so this
script launches QEMU with the guest UART bound to a TCP socket and then speaks
the identical protocol over that socket. Output format matches send_image.py so
the two are directly comparable.

    python scripts/send_image_qemu.py build/zephyr/zephyr.elf sign.png
    python scripts/send_image_qemu.py build/zephyr/zephyr.elf sign.png --size 48
"""
import argparse, os, re, socket, subprocess, sys, time

CLASS_NAMES = {
    0: "Speed limit (5km/h)",   2: "Speed limit (30km/h)", 3: "Speed limit (40km/h)",
    4: "Speed limit (50km/h)",  5: "Speed limit (60km/h)", 6: "Speed limit (70km/h)",
    7: "Speed limit (80km/h)",  24: "Go Right",            43: "Go right or straight",
    69: "Height limit",         70: "Weight limit",        72: "Length limit",
    83: "Steep descent",        84: "Steep ascent",        85: "Narrow road",
    86: "Narrow bridge",        87: "Unknown",
}


def load_payload(path, size, grayscale, raw):
    if raw:
        return open(path, "rb").read()
    from PIL import Image
    im = Image.open(path)
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        im = Image.alpha_composite(Image.new("RGBA", im.size, (255, 255, 255, 255)), im)
    if grayscale:
        return im.convert("L").resize((48, 48)).tobytes()
    return im.convert("RGB").resize((size, size)).tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("elf")
    ap.add_argument("image")
    ap.add_argument("--size", type=int, default=48, choices=[48, 64],
                    help="RGB payload edge length (default 48, matching send_image.py)")
    ap.add_argument("-g", "--grayscale", action="store_true")
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--port", type=int, default=5678)
    ap.add_argument("--qemu", default=os.environ.get("QEMU_BIN", "qemu-system-i386"))
    a = ap.parse_args()

    payload = load_payload(a.image, a.size, a.grayscale, a.raw)
    print(f"payload: {len(payload)} bytes")

    cmd = [a.qemu, "-m", "32", "-cpu", "qemu32,+nx,+pae", "-machine", "q35",
           "-device", "isa-debug-exit,iobase=0xf4,iosize=0x04", "-machine", "acpi=off",
           "-accel", "tcg,tb-size=64",
           "-chardev", f"socket,id=con,host=127.0.0.1,port={a.port},server=on,wait=on",
           "-serial", "chardev:con", "-monitor", "none", "-display", "none",
           "-no-reboot", "-net", "none", "-kernel", a.elf]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.0)
    buf = b""

    def readline(sock, timeout=30.0):
        nonlocal buf
        sock.settimeout(timeout)
        while b"\n" not in buf:
            try:
                d = sock.recv(4096)
            except socket.timeout:
                return None
            if not d:
                return None
            buf += d
        line, buf = buf.split(b"\n", 1)
        return line.decode(errors="ignore").strip()

    try:
        s = socket.create_connection(("127.0.0.1", a.port), timeout=15)
        end = time.time() + 30
        while time.time() < end:
            l = readline(s)
            if l:
                print(l)
            if l and l.endswith("READY"):
                break
        else:
            raise SystemExit("timed out waiting for READY")

        t0 = time.time()
        for i in range(0, len(payload), 32):
            s.sendall(payload[i:i + 32].hex().encode() + b"\n")
        s.sendall(b"STOP\n")
        print(f"Upload finished in {time.time() - t0:.2f}s")

        t_recv = None
        while True:
            l = readline(s, 60)
            if l is None:
                raise SystemExit("timed out waiting for PREDICTION/ERROR")
            print(l)
            if t_recv is not None:
                print(f"Model prediction done in: {time.time() - t_recv:.3f}s")
                t_recv = None
            if l.startswith("RECEIVED"):
                t_recv = time.time()
            if l.startswith("PREDICTION"):
                m = re.search(r"class=(-?\d+)", l)
                cid = int(m.group(1)) if m else int(l.split()[1])
                print(f"--> class {cid}: {CLASS_NAMES.get(cid, '?')}")
                break
            if l.startswith("ERROR"):
                break
    finally:
        proc.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
