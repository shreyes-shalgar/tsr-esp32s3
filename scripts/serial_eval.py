#!/usr/bin/env python3
"""
Score the board over a real serial port, the same way qemu_uart_eval.py scores
it over QEMU.

This is what turns "the ESP32-S3 image links and the footprint is known" into a
measured hardware accuracy figure. It speaks the identical protocol the
organisers' send_image.py speaks - READY, hex lines, STOP, RECEIVED,
PREDICTION - just for all 364 validation images in one run.

    python scripts/serial_eval.py COM7
    python scripts/serial_eval.py COM7 --limit 20          # quick smoke test
    python scripts/serial_eval.py /dev/ttyUSB0 --baud 115200

At 115200 baud one image is ~14 KB of hex, about 1.3 s on the wire, so a full
364-image sweep takes roughly 10 minutes. --limit 20 first is a good idea.

--tcp exists so the transport-independent half of this script can be exercised
against QEMU without a board attached; it is not how you would use it in
anger.
"""
import argparse
import os
import re
import socket
import sys
import time

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*parts):
    return os.path.join(_REPO, *parts)


CLASS_IDS = [0, 2, 3, 4, 5, 6, 7, 24, 43, 69, 70, 72, 83, 84, 85, 86, 87]


# --------------------------------------------------------------- transports
class SerialLink:
    """pyserial, byte-oriented, no line discipline."""

    def __init__(self, port, baud, pace=0.0):
        import serial                      # imported late: only this path needs it
        self.s = serial.Serial(port, baud, timeout=0.2)
        self.pace = pace

    def write(self, b):
        self.s.write(b)
        self.s.flush()
        if self.pace:
            time.sleep(self.pace)

    def read(self, n):
        return self.s.read(n)

    def close(self):
        self.s.close()


class TcpLink:
    """QEMU's socket chardev, so the protocol half can be tested board-less."""

    def __init__(self, hostport):
        h, _, p = hostport.rpartition(":")
        self.s = socket.create_connection((h or "127.0.0.1", int(p)), timeout=10)
        self.s.settimeout(0.2)

    def write(self, b):
        self.s.sendall(b)

    def read(self, n):
        try:
            return self.s.recv(n)
        except socket.timeout:
            return b""

    def close(self):
        self.s.close()


# ----------------------------------------------------------------- protocol
class Board:
    def __init__(self, link):
        self.link = link
        self.buf = b""

    def readline(self, timeout):
        end = time.time() + timeout
        while b"\n" not in self.buf:
            if time.time() > end:
                return None
            d = self.link.read(4096)
            if d:
                self.buf += d
        line, self.buf = self.buf.split(b"\n", 1)
        return line.decode(errors="ignore").strip()

    def wait_ready(self, timeout=40.0):
        end = time.time() + timeout
        while time.time() < end:
            l = self.readline(max(0.2, end - time.time()))
            if l is not None and l.endswith("READY"):
                return True
        return False

    def infer(self, payload, timeout=90.0):
        self.buf = b""                    # drop any stale READY re-arms
        for i in range(0, len(payload), 32):
            self.link.write(payload[i:i + 32].hex().encode() + b"\n")
        self.link.write(b"STOP\n")
        nbytes = None
        end = time.time() + timeout
        while time.time() < end:
            l = self.readline(max(0.2, end - time.time()))
            if l is None:
                continue
            if l.startswith("RECEIVED"):
                try:
                    nbytes = int(l.split()[1])
                except (IndexError, ValueError):
                    pass
            elif l.startswith("PREDICTION"):
                m = re.search(r"class=(-?\d+)", l)
                c = re.search(r"confidence=([\d.]+)", l)
                return (int(m.group(1)) if m else None), nbytes, \
                       (float(c.group(1)) if c else None)
            elif l.startswith("ERROR"):
                print("  board error:", l)
                return None, nbytes, None
        return None, nbytes, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", nargs="?", help="COM7, /dev/ttyUSB0, ...")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--data", default=_p("out", "data.npz"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--retries", type=int, default=3,
                    help="re-send an image this many times if the board "
                         "rejects the frame (default 3)")
    ap.add_argument("--pace", type=float, default=0.0,
                    help="seconds to pause after each 32-byte chunk. Only "
                         "needed on a link that drops bytes often; costs "
                         "216x this per image")
    ap.add_argument("--tcp", default=None,
                    help="host:port instead of a serial port (QEMU testing)")
    a = ap.parse_args()

    if not a.tcp and not a.port:
        print("ERROR: give a serial port (e.g. COM7) or --tcp host:port",
              file=sys.stderr)
        return 2
    if not os.path.exists(a.data):
        print(f"ERROR: dataset not found: {a.data}", file=sys.stderr)
        return 2

    d = np.load(a.data)
    X, y = d["Xva"], d["yva"]
    if a.limit:
        X, y = X[:a.limit], y[:a.limit]

    if a.tcp:
        link = TcpLink(a.tcp)
    else:
        try:
            link = SerialLink(a.port, a.baud, a.pace)
        except ImportError:
            print("ERROR: pyserial not installed.  pip install pyserial",
                  file=sys.stderr)
            return 2
        except Exception as e:
            print(f"ERROR: could not open {a.port}: {e}\n"
                  f"Check the port in Device Manager, and that no terminal "
                  f"(Tera Term, PuTTY) still has it open.", file=sys.stderr)
            return 2

    b = Board(link)
    print(f"waiting for READY on {a.tcp or a.port} ...", flush=True)
    if not b.wait_ready():
        print("no READY. Reset the board (EN button) and try again.",
              file=sys.stderr)
        link.close()
        return 1

    ok = n = retried = failed = 0
    confs, wrong = [], []
    t0 = time.time()
    try:
        for i in range(len(X)):
            # A dropped byte on the wire makes the board reject the frame
            # ("ERROR odd number of hex digits") rather than predict on
            # corrupt data - which is the behaviour you want, but it means the
            # host has to re-send. There is no hardware flow control on this
            # link, so loss is possible; a 364-image sweep here re-sent
            # nothing, so this path is insurance rather than a routine event.
            pred = None
            for attempt in range(a.retries + 1):
                if attempt:
                    retried += 1
                    if not b.wait_ready(timeout=15):
                        break
                pred, nb, conf = b.infer(X[i].tobytes())
                if pred is not None:
                    break
                print(f"  image {i}: frame rejected, re-sending "
                      f"({attempt + 1}/{a.retries})", file=sys.stderr)
            if pred is None:
                print(f"  image {i}: no prediction after {a.retries} retries "
                      f"- stopping", file=sys.stderr)
                failed += 1
                break
            n += 1
            if conf is not None:
                confs.append(conf)
            if pred == CLASS_IDS[y[i]]:
                ok += 1
            else:
                wrong.append((i, CLASS_IDS[y[i]], pred, conf))
            if not b.wait_ready(timeout=40):
                print("  lost READY - stopping", file=sys.stderr)
                break
            if (i + 1) % 25 == 0:
                el = time.time() - t0
                print(f"  {i+1}/{len(X)}  acc={ok/(i+1):.4f}  "
                      f"{el/(i+1):.2f} s/image", flush=True)
    finally:
        link.close()

    if not n:
        return 1
    print(f"\ntransport : {'tcp ' + a.tcp if a.tcp else a.port + ' @ ' + str(a.baud)}")
    print(f"images    : {n}")
    print(f"accuracy  : {ok/n:.4f} ({ok}/{n})")
    if retried or failed:
        print(f"link      : {retried} frame(s) re-sent after a dropped byte"
              + (f", {failed} unrecoverable" if failed else "")
              + "  (no hardware flow control on this link)")
    if confs:
        print(f"confidence: mean {np.mean(confs):.2f}%  min {np.min(confs):.2f}%")
    print(f"wall      : {time.time()-t0:.1f}s")
    for i, exp, got, c in wrong:
        print(f"  wrong: image {i} expected class {exp}, got {got} "
              f"at {c if c is not None else '?'}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
