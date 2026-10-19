"""
A/B the two firmware builds over the identical UART path.

The per-line timing the host script reports is unreliable once inference gets
fast: RECEIVED and PREDICTION coalesce into one TCP segment and the measured
"gap" collapses to socket granularity rather than compute time. Differencing
total wall time between two builds over the same transfer avoids that - the
transfer cost is identical, so the whole difference is inference.
"""
import os
import socket
import subprocess
import sys
import time

import numpy as np

# Paths are resolved relative to the repository root so the scripts run
# from a fresh checkout on any machine.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*parts):
    return os.path.join(_REPO, *parts)


N = 60
QEMU = "/usr/bin/qemu-system-i386"


def run(elf, port):
    cmd = [QEMU, "-m", "32", "-cpu", "qemu32,+nx,+pae", "-machine", "q35",
           "-device", "isa-debug-exit,iobase=0xf4,iosize=0x04", "-machine", "acpi=off",
           "-chardev", f"socket,id=con,host=127.0.0.1,port={port},server=on,wait=on",
           "-serial", "chardev:con", "-monitor", "none", "-display", "none",
           "-no-reboot", "-net", "none", "-kernel", elf]
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2.0)
    s = socket.create_connection(("127.0.0.1", port), timeout=15)
    buf = b""

    def rl(t=40):
        nonlocal buf
        s.settimeout(t)
        while b"\n" not in buf:
            try:
                d = s.recv(8192)
            except socket.timeout:
                return None
            if not d:
                return None
            buf += d
        l, buf = buf.split(b"\n", 1)
        return l.decode(errors="ignore").strip()

    X = np.load(_p("out", "data.npz"))["Xva"]
    payloads = [b"".join(X[i].tobytes()[k:k + 32].hex().encode() + b"\n"
                         for k in range(0, 6912, 32)) + b"STOP\n" for i in range(N)]

    while True:
        l = rl()
        if l and l.endswith("READY"):
            break

    t0 = time.perf_counter()
    for pl in payloads:
        s.sendall(pl)
        while True:
            l = rl(40)
            if l is None or l.startswith(("PREDICTION", "ERROR")):
                break
        while True:
            l = rl(10)
            if l is None or l.endswith("READY"):
                break
    el = time.perf_counter() - t0
    p.kill()
    return el


if __name__ == "__main__":
    a_elf, b_elf = sys.argv[1], sys.argv[2]
    ta = run(a_elf, 6301)
    tb = run(b_elf, 6303)
    print(f"build A  {ta:7.2f} s total   {ta / N * 1000:7.1f} ms/image")
    print(f"build B  {tb:7.2f} s total   {tb / N * 1000:7.1f} ms/image")
    print(f"delta    {(ta - tb) / N * 1000:+7.1f} ms/image attributable to inference")
    if tb > 0:
        print(f"speedup on the changed portion: {ta / tb:.2f}x on total round-trip")
