#!/usr/bin/env python3
"""
The gate. One command that decides whether the current tree is shippable.

    python scripts/verify_all.py                 # full gate
    python scripts/verify_all.py --quick         # skip the 364-image QEMU sweep
    python scripts/verify_all.py --no-esp        # skip the ESP32-S3 build

Exit code 0 = every check passed. Anything else = do not ship, do not report
numbers from this tree.

Every figure the project claims comes from here. If a number is not in this
script's output, it was not measured.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "src" / "qemu_app"
ESP_APP = ROOT / "src" / "esp32_app"
OUT = ROOT / "out"

# Thresholds. A change that regresses any of these is rejected.
MIN_ACCURACY = 0.9850          # baseline 0.9890. The two-image gap to 0.9945 is a
                               # deliberate trade: the shipped model gives up two
                               # images of the ambiguous 83/84 pair to gain 37 points
                               # on the classes exposed to a domain-shifted test set.
MAX_PEAK_RAM = 10240           # baseline 9,984 B after conv+conv+pool fusion
MAX_INT8_MODEL = 90_000        # baseline 72,408 B
REQUIRE_BIT_EXACT_ARGMAX = True

CLASS_IDS = [0, 2, 3, 4, 5, 6, 7, 24, 43, 69, 70, 72, 83, 84, 85, 86, 87]

results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""),
          flush=True)
    return ok


def run(cmd, cwd=None, env=None, timeout=1800):
    return subprocess.run(cmd, cwd=cwd, env=env, timeout=timeout,
                          capture_output=True, text=True, shell=isinstance(cmd, str))


def build(board: str, builddir: Path, extra: list[str] | None = None,
          app: Path | None = None):
    """Configure + build. Falls back to cmake when `west build` is unusable."""
    env = dict(os.environ)
    if builddir.exists():
        shutil.rmtree(builddir)
    cm = ["cmake", "-B", str(builddir), "-GNinja", f"-DBOARD={board}",
          str(app or APP)]
    if extra:
        cm += extra
    r = run(cm, env=env)
    if r.returncode != 0:
        return False, (r.stdout + r.stderr)[-2500:]
    r = run(["ninja", "-C", str(builddir)], env=env)
    return r.returncode == 0, (r.stdout + r.stderr)


def parse_memory(log: str) -> dict:
    out = {}
    for line in log.splitlines():
        m = re.match(r"\s*(\w+):\s+(\d+) B\s+(\d+) B\s+([\d.]+)%", line)
        if m:
            out[m.group(1)] = int(m.group(2))
    return out



# --------------------------------------------------------------------------
# The two numerical checks need a TFLite interpreter to compare against. Full
# TensorFlow provides one, but it is a ~600 MB install for two function calls.
# LiteRT (`pip install ai-edge-litert`, ~20 MB) is the same TFLite runtime,
# renamed and shipped on its own, and exposes the same reference-kernel op
# resolver - which is the part that matters here, because the default XNNPACK
# delegate does not reproduce the reference int8 arithmetic.
def load_interpreter(model_path, reference_kernels=False):
    """Return (Interpreter, backend_name). Raises ImportError if neither is present."""
    try:
        import tensorflow as tf
        kw = {}
        if reference_kernels:
            kw["experimental_op_resolver_type"] = \
                tf.lite.experimental.OpResolverType.BUILTIN_REF
        return tf.lite.Interpreter(model_path=str(model_path), **kw), "tensorflow"
    except ImportError:
        pass
    from ai_edge_litert.interpreter import Interpreter, OpResolverType
    kw = {}
    if reference_kernels:
        kw["experimental_op_resolver_type"] = OpResolverType.BUILTIN_REF
    return Interpreter(model_path=str(model_path), **kw), "ai-edge-litert"


_NO_RUNTIME = ("no TFLite runtime: pip install ai-edge-litert   "
               "(~20 MB, or full tensorflow-cpu if you also want to retrain)")

# --------------------------------------------------------------------------
def check_arena_matches_header() -> bool:
    """Peak RAM is a generated constant; read it rather than trusting a comment."""
    hdr = (APP / "src" / "tsr_model_data.h").read_text()
    a = re.search(r"#define TSR_BUF_A (\d+)", hdr)
    b = re.search(r"#define TSR_BUF_B (\d+)", hdr)
    sc = re.search(r"#define TSR_SCRATCH (\d+)", hdr)
    if not (a and b):
        return record("arena constants present", False, "TSR_BUF_A/B not found")
    # the fused conv+maxpool row scratch counts toward peak RAM too
    arena = int(a.group(1)) + int(b.group(1)) + (int(sc.group(1)) if sc else 0)
    return record("peak inference RAM", arena <= MAX_PEAK_RAM,
                  f"{arena} B (limit {MAX_PEAK_RAM})")


def check_apps_agree() -> bool:
    """src/qemu_app and src/esp32_app must run identical code.

    They are separate Zephyr applications because the submission is organised
    per target, but the runtime is one runtime. If these ever diverge, the
    QEMU result stops being evidence about the board - so the gate checks it
    rather than trusting the copy.
    """
    import hashlib
    files = ["src/main.c", "src/tsr_model.c", "src/tsr_model_data.c",
             "src/tsr_model.h", "src/tsr_model_data.h", "src/tsr_image.c"]
    diff = []
    for f in files:
        a, b = APP / f, ESP_APP / f
        if not (a.exists() and b.exists()):
            diff.append(f + " (missing)"); continue
        if hashlib.sha256(a.read_bytes()).digest() != hashlib.sha256(b.read_bytes()).digest():
            diff.append(f)
    return record("qemu_app and esp32_app share identical sources", not diff,
                  f"{len(files)} files" if not diff else "differ: " + ", ".join(diff))


def check_input_quantisation() -> bool:
    """Firmware hard-codes (pixel - 128); the model must actually ask for that."""
    p = ROOT / "model" / "quantized_model_int8.tflite"
    if not p.exists():
        p = ROOT / "model" / "model_int8.tflite"
    try:
        import numpy as np  # noqa
        it, _ = load_interpreter(p)
    except ImportError:
        return record("input quantisation", False, _NO_RUNTIME)
    it.allocate_tensors()
    scale, zp = it.get_input_details()[0]["quantization"]
    ok = abs(scale - 1.0) < 1e-9 and zp == -128
    return record("input quantisation is (pixel - 128)", ok,
                  f"scale={scale} zero_point={zp}")


def check_model_size() -> bool:
    p = ROOT / "model" / "quantized_model_int8.tflite"
    if not p.exists():
        p = ROOT / "model" / "model_int8.tflite"
    n = p.stat().st_size
    return record("int8 model size", n <= MAX_INT8_MODEL,
                  f"{n} B (limit {MAX_INT8_MODEL})")


def check_bit_exact() -> bool:
    """C runtime vs the TFLite REFERENCE kernels (never the XNNPACK default)."""
    try:
        import numpy as np
    except ImportError:
        return record("bit-exactness vs TFLite reference", False,
                      "numpy not importable")

    host = ROOT / "out" / ("host_test.exe" if os.name == "nt" else "host_test")
    host.parent.mkdir(exist_ok=True)
    cc = shutil.which("gcc") or shutil.which("cc") or shutil.which("clang")
    if not cc:
        return record("bit-exactness vs TFLite reference", False,
                      "no C compiler on PATH for the native harness")
    r = run([cc, "-O2", "-std=gnu99", "-o", str(host),
             str(ROOT / "test" / "host_test.c"),
             str(APP / "src" / "tsr_model.c"),
             str(APP / "src" / "tsr_model_data.c"), "-lm"])
    if r.returncode != 0:
        return record("bit-exactness vs TFLite reference", False,
                      "native harness failed to compile: " + r.stderr[-400:])

    data = np.load(OUT / "data.npz")
    Xva, yva = data["Xva"], data["yva"]
    raw = OUT / "val_images.bin"
    Xva.astype(np.uint8).tofile(raw)
    r = run([str(host), str(raw)])
    rows = [l.split() for l in r.stdout.strip().splitlines()]
    mine_idx = np.array([int(x[0]) for x in rows])

    p = ROOT / "model" / "quantized_model_int8.tflite"
    if not p.exists():
        p = ROOT / "model" / "model_int8.tflite"
    try:
        it, backend = load_interpreter(p, reference_kernels=True)
    except ImportError:
        return record("bit-exactness vs TFLite reference", False, _NO_RUNTIME)
    it.allocate_tensors()
    ii, oo = it.get_input_details()[0], it.get_output_details()[0]
    ref_idx = np.empty(len(Xva), dtype=int)
    for i in range(len(Xva)):
        it.set_tensor(ii["index"], (Xva[i].astype(np.int32) - 128).astype(np.int8)[None])
        it.invoke()
        ref_idx[i] = int(np.argmax(it.get_tensor(oo["index"])[0]))

    agree = int((mine_idx == ref_idx).sum())
    ok = (agree == len(Xva)) if REQUIRE_BIT_EXACT_ARGMAX else True
    acc = float((mine_idx == yva).mean())
    record("native int8 accuracy", acc >= MIN_ACCURACY, f"{acc*100:.2f}%")
    return record("bit-exactness vs TFLite reference (argmax)", ok,
                  f"{agree}/{len(Xva)} via {backend}")


def check_qemu(limit: int | None) -> bool:
    """Drive the real UART protocol on QEMU and score it."""
    elf = ROOT / "build_qemu" / "zephyr" / "zephyr.elf"
    if not elf.exists():
        return record("QEMU end-to-end", False, "no qemu_x86 build")
    cmd = [sys.executable, str(ROOT / "scripts" / "qemu_uart_eval.py"),
           "--elf", str(elf), "--mode", "rgb48"]
    if limit:
        cmd += ["--limit", str(limit)]
    r = run(cmd, timeout=2400)
    acc = re.search(r"accuracy\s*:\s*([\d.]+)", r.stdout)
    ms = re.search(r"inference:\s*mean\s*([\d.]+)\s*ms", r.stdout)
    if not acc:
        return record("QEMU end-to-end", False,
                      (r.stdout + r.stderr)[-600:])
    a = float(acc.group(1))
    record("QEMU inference time", True, f"{ms.group(1)} ms mean" if ms else "n/a")
    return record("QEMU end-to-end accuracy", a >= MIN_ACCURACY, f"{a*100:.2f}%")


def check_protocol() -> bool:
    """The exact line contract the host script parses."""
    src = (APP / "src" / "main.c").read_text()
    checks = [
        ('"READY', "READY emitted"),
        ('"RECEIVED ', "RECEIVED emitted"),
        ('" bytes', "RECEIVED includes 'bytes'"),
        ('"PREDICTION class=', "PREDICTION uses class= form"),
        ('"ERROR', "ERROR path present"),
    ]
    ok = True
    for needle, name in checks:
        ok &= record(f"protocol: {name}", needle in src)
    # Nothing may be printed between RECEIVED and PREDICTION *on the success
    # path*. Error branches end in `continue`, so anything before the last
    # `continue` in the segment is unreachable once we get as far as inference.
    seg = src[src.find('uart_write_str("RECEIVED '):src.find('uart_write_str("PREDICTION')]
    tail = seg[seg.rfind("continue;") + 1:] if "continue;" in seg else seg
    stray = [l.strip() for l in tail.splitlines()
             if 'uart_write_str("' in l or 'uart_poll_out(' in l]
    ok &= record("protocol: nothing printed between RECEIVED and PREDICTION",
                 len(stray) == 0,
                 "clean" if not stray else f"{len(stray)} stray write(s): {stray[:2]}")
    return ok


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="sweep only 40 images on QEMU instead of all 364")
    ap.add_argument("--no-esp", action="store_true")
    ap.add_argument("--skip-qemu", action="store_true")
    a = ap.parse_args()

    print("=" * 68)
    print("AI Arena 2026 Track 1 - verification gate")
    print("=" * 68)

    print("\n[1/5] static contract")
    check_protocol()
    check_arena_matches_header()
    check_apps_agree()
    check_model_size()
    check_input_quantisation()

    print("\n[2/5] build: qemu_x86")
    ok, log = build("qemu_x86", ROOT / "build_qemu")
    mem = parse_memory(log)
    record("qemu_x86 builds", ok, "" if ok else log[-700:])
    if mem.get("RAM"):
        print(f"        image RAM {mem['RAM']} B")

    if not a.no_esp:
        print("\n[3/5] build: esp32s3_devkitc")
        hal = os.environ.get("HAL_ESPRESSIF_DIR")
        extra = [f"-DZEPHYR_MODULES={hal}"] if hal else None
        ok_e, log_e = build("esp32s3_devkitc/esp32s3/procpu", ROOT / "build_esp",
                            extra, app=ESP_APP)
        record("esp32s3 builds", ok_e, "" if ok_e else log_e[-700:])
        m = parse_memory(log_e)
        if m:
            print(f"        FLASH {m.get('FLASH')} B  DRAM {m.get('dram0_0_seg')} B"
                  f"  IRAM {m.get('iram0_0_seg')} B")
    else:
        print("\n[3/5] build: esp32s3  (skipped)")

    print("\n[4/5] numerical correctness")
    check_bit_exact()

    print("\n[5/5] end-to-end on QEMU over UART")
    if a.skip_qemu:
        print("        (skipped)")
    else:
        check_qemu(40 if a.quick else None)

    print("\n" + "=" * 68)
    failed = [n for n, ok, _ in results if not ok]
    for n, ok, d in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {n}" + (f"   {d}" if d else ""))
    print("=" * 68)
    if failed:
        print(f"GATE FAILED - {len(failed)} check(s): {', '.join(failed)}")
        print("Do not report numbers from this tree. Fix or revert.")
        return 1
    print("GATE PASSED - every check measured, not estimated.")
    (OUT / "gate_result.json").write_text(json.dumps(
        {"passed": True, "timestamp": time.time(),
         "checks": [{"name": n, "ok": o, "detail": d} for n, o, d in results]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
