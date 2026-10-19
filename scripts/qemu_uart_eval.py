#!/usr/bin/env python3
"""
End-to-end evaluation of the Zephyr app running on QEMU.

Launches qemu-system-i386 with the UART bound to a TCP socket, then speaks the
exact protocol send_image.py uses (READY -> hex -> STOP -> RECEIVED ->
PREDICTION) for every image, and scores the result.
"""
import os
import argparse, re, socket, subprocess, sys, tempfile, time
import numpy as np

# Paths are resolved relative to the repository root so the scripts run
# from a fresh checkout on any machine.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*parts):
    return os.path.join(_REPO, *parts)


# The build directory is chosen by whoever runs cmake/west, so look in the
# places the instructions suggest rather than assuming one of them.
_ELF_CANDIDATES = [
    ("build", "zephyr", "zephyr.elf"),            # cmake -B build   (documented)
    ("qemu_app", "build", "zephyr", "zephyr.elf"),
    ("src", "qemu_app", "build", "zephyr", "zephyr.elf"),
    ("build_qemu", "zephyr", "zephyr.elf"),
]


def _find_elf():
    for parts in _ELF_CANDIDATES:
        c = _p(*parts)
        if os.path.exists(c):
            return c
    return _p(*_ELF_CANDIDATES[0])


QEMU = os.environ.get("QEMU_BIN", "qemu-system-i386")


def qemu_cmd(elf, port, icount, qemu=None):
    c = [qemu or QEMU, "-m", "32", "-cpu", "qemu32,+nx,+pae", "-machine", "q35",
         "-device", "isa-debug-exit,iobase=0xf4,iosize=0x04", "-machine", "acpi=off",
         "-accel", "tcg,tb-size=64",
         "-chardev", f"socket,id=con,host=127.0.0.1,port={port},server=on,wait=on",
         "-serial", "chardev:con", "-monitor", "none", "-display", "none",
         "-no-reboot", "-net", "none", "-kernel", elf]
    if icount:
        c += ["-icount", "shift=5,align=off,sleep=off", "-rtc", "clock=vm"]
    return c


class Board:
    def __init__(self, sock):
        self.s = sock
        self.buf = b""

    def readline(self, timeout=30.0):
        self.s.settimeout(timeout)
        while b"\n" not in self.buf:
            try:
                d = self.s.recv(4096)
            except socket.timeout:
                return None
            if not d:
                return None
            self.buf += d
        line, self.buf = self.buf.split(b"\n", 1)
        return line.decode(errors="ignore").strip()

    def wait_ready(self, timeout=30.0):
        end = time.time() + timeout
        while time.time() < end:
            l = self.readline(timeout=max(0.1, end - time.time()))
            if l is None:
                continue
            if l.endswith("READY"):
                return True
        return False

    def infer(self, payload: bytes):
        for i in range(0, len(payload), 32):
            self.s.sendall(payload[i:i + 32].hex().encode() + b"\n")
        self.s.sendall(b"STOP\n")
        t_recv = None
        pred, nbytes, ms = None, None, None
        while True:
            l = self.readline(timeout=60.0)
            if l is None:
                return None, None, None
            if t_recv is not None and ms is None:
                ms = (time.time() - t_recv) * 1000.0
            if l.startswith("RECEIVED"):
                nbytes = int(l.split()[1]); t_recv = time.time()
            elif l.startswith("PREDICTION"):
                m = re.search(r"class=(-?\d+)", l)
                pred = int(m.group(1)) if m else int(l.split()[1])
                return pred, nbytes, ms
            elif l.startswith("ERROR"):
                print("  board error:", l); return None, nbytes, ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elf", default=None)
    ap.add_argument("--data", default=_p("out", "data.npz"))
    ap.add_argument("--port", type=int, default=5678)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--mode", choices=["rgb48", "rgb64", "gray48"], default="rgb48")
    ap.add_argument("--icount", action="store_true")
    ap.add_argument("--qemu", default=None, help="path to qemu-system-i386")
    a = ap.parse_args()

    if a.elf is None:
        a.elf = _find_elf()
    if not os.path.exists(a.elf):
        print(f"ERROR: firmware not found: {a.elf}\n"
              f"Build it first, from the folder containing src\\qemu_app:\n"
              f"  cmake -B build -GNinja -DBOARD=qemu_x86 "
              f"-DZEPHYR_BASE=$env:ZEPHYR_BASE src/qemu_app\n"
              f"  ninja -C build\n"
              f"or point at an existing build with  --elf <path to zephyr.elf>",
              file=sys.stderr)
        return 2
    if not os.path.exists(a.data):
        print(f"ERROR: dataset not found: {a.data}\n"
              f"Pass one with  --data <path to .npz>  (out/data.npz ships with "
              f"the submission).", file=sys.stderr)
        return 2

    d = np.load(a.data); X, y = d["Xva"], d["yva"]
    if a.limit:
        X, y = X[:a.limit], y[:a.limit]

    CLASS_IDS = [0,2,3,4,5,6,7,24,43,69,70,72,83,84,85,86,87]
    # wait=on: QEMU blocks until we connect, so the boot banner and the first
    # READY are never lost to a not-yet-connected socket.
    errf = tempfile.TemporaryFile()

    def _qemu_died(prefix):
        """QEMU exiting looks like a reset socket on Windows; say what it said."""
        errf.seek(0)
        msg = errf.read().decode(errors="ignore").strip()
        print(f"{prefix}\nQEMU exited with code {proc.poll()}.", file=sys.stderr)
        if msg:
            print("QEMU said:\n  " + msg.replace("\n", "\n  "), file=sys.stderr)
        else:
            print(f"No output. Check that {QEMU if a.qemu is None else a.qemu} is "
                  f"on PATH and that port {a.port} is free.", file=sys.stderr)

    try:
        proc = subprocess.Popen(qemu_cmd(a.elf, a.port, a.icount, a.qemu),
                                stdout=subprocess.DEVNULL, stderr=errf)
    except FileNotFoundError:
        print(f"ERROR: {a.qemu or QEMU} not found on PATH.\n"
              f"Install QEMU, or pass  --qemu <path to qemu-system-i386.exe>",
              file=sys.stderr)
        return 2
    try:
        # QEMU binds the listening socket a little after exec, and how long
        # that takes depends on what else the machine is doing - during a full
        # gate run TensorFlow has just finished hammering the CPU. A fixed
        # sleep here is a race that shows up as "connection refused" on a
        # loaded machine, so poll for the port instead.
        s, deadline, last = None, time.time() + 30.0, None
        while time.time() < deadline:
            if proc.poll() is not None:
                _qemu_died("ERROR: QEMU exited before accepting a connection.")
                return 3
            try:
                s = socket.create_connection(("127.0.0.1", a.port), timeout=5)
                break
            except OSError as e:
                last = e
                time.sleep(0.25)
        if s is None:
            _qemu_died(f"ERROR: QEMU never accepted a connection on port "
                       f"{a.port} within 30 s (last error: {last}).")
            return 3
        b = Board(s)
        if not b.wait_ready():
            if proc.poll() is not None:
                _qemu_died("ERROR: no READY - QEMU stopped.")
                return 3
            print("no READY from board"); return 1

        ok, times, n = 0, [], 0
        t0 = time.time()
        for i in range(len(X)):
            img = X[i]
            if a.mode == "rgb48":
                payload = img.tobytes()
            elif a.mode == "gray48":
                g = (0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2])
                payload = np.clip(np.round(g),0,255).astype(np.uint8).tobytes()
            else:
                from PIL import Image
                payload = np.asarray(Image.fromarray(img).resize((64,64)),
                                     dtype=np.uint8).tobytes()
            pred, nb, ms = b.infer(payload)
            if pred is None:
                print(f"  image {i}: no prediction"); break
            n += 1
            if ms: times.append(ms)
            if pred == CLASS_IDS[y[i]]: ok += 1
            if not b.wait_ready(timeout=30):
                print("  lost READY"); break
            if (i+1) % 50 == 0:
                print(f"  {i+1}/{len(X)}  acc={ok/(i+1):.4f}", flush=True)
        el = time.time() - t0
        print(f"\nmode={a.mode} icount={a.icount}")
        print(f"images   : {n}")
        print(f"accuracy : {ok/n:.4f} ({ok}/{n})")
        if times:
            print(f"inference: mean {np.mean(times):.1f} ms  median {np.median(times):.1f} ms")
        print(f"wall     : {el:.1f}s")
    except ConnectionError as e:
        _qemu_died(f"ERROR: connection to QEMU lost ({e}).")
        return 3
    finally:
        proc.kill()
        errf.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
