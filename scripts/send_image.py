#!/usr/bin/env python3
"""Send an image to the board as ASCII hex and print the prediction.

The board prints "READY", then expects hex bytes followed by "STOP".

Usage:
    python scripts/send_image.py COM7 sign.png
    python scripts/send_image.py /dev/ttyACM0 sign.raw --raw
"""

import argparse
import sys
import time

import serial  # pyserial

WIDTH = 48
HEIGHT = 48
CHANNELS = 3
NUM_BYTES = WIDTH * HEIGHT * CHANNELS

READY_TIMEOUT_S = 30.0
RESULT_TIMEOUT_S = 30.0
POST_RESULT_DRAIN_S = 1.0


def load_payload(path: str, raw: bool) -> bytes:
    if raw:
        with open(path, "rb") as f:
            data = f.read()
    else:
        from PIL import Image  # pillow

        img = Image.open(path).convert("RGB").resize((WIDTH, HEIGHT))
        data = img.tobytes()

    if len(data) != NUM_BYTES:
        raise SystemExit(f"expected {NUM_BYTES} bytes, got {len(data)}")

    return data


def read_and_echo_line(port: serial.Serial) -> str | None:
    line = port.readline().decode(errors="ignore").strip()
    if line:
        print(line)
        return line
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("port")
    parser.add_argument("image")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--raw", action="store_true",
                        help="image file is already 48x48x3 raw RGB")
    args = parser.parse_args()

    payload = load_payload(args.image, args.raw)

    with serial.Serial(args.port, args.baud, timeout=0.1) as port:
        # Wait for the board handshake.
        ready_deadline = time.monotonic() + READY_TIMEOUT_S
        while time.monotonic() < ready_deadline:
            line = read_and_echo_line(port)
            if line and line.endswith("READY"):
                break
        else:
            raise SystemExit("timed out waiting for READY")

        send_start = time.monotonic()
        total = len(payload)
        for i in range(0, total, 32):
            port.write(payload[i:i + 32].hex().encode() + b"\n")
            sent = min(i + 32, total)
            pct = sent * 100.0 / total
            print(f"\rSending image: {pct:5.1f}% ({sent}/{total} bytes)", end="", flush=True)
            # Pass through any board logs that arrive during upload.
            while True:
                line = read_and_echo_line(port)
                if line is None:
                    break
        print()  # newline after the progress bar
        port.write(b"STOP\n")
        send_elapsed = time.monotonic() - send_start
        print(f"Upload finished in {send_elapsed:.2f}s")

        result_deadline = time.monotonic() + RESULT_TIMEOUT_S
        result_seen = False
        received_time: float | None = None
        while time.monotonic() < result_deadline:
            line = read_and_echo_line(port)
            if line is None:
                continue
            if received_time is not None:
                elapsed = time.monotonic() - received_time
                print(f"Model prediction done in: {elapsed:.3f}s")
                received_time = None
            if line.startswith("RECEIVED"):
                received_time = time.monotonic()
            if line.startswith(("PREDICTION", "ERROR")):
                result_seen = True
                break

        if not result_seen:
            raise SystemExit("timed out waiting for PREDICTION/ERROR")

        # Drain briefly so trailing UART lines are also passed through.
        drain_deadline = time.monotonic() + POST_RESULT_DRAIN_S
        while time.monotonic() < drain_deadline:
            if read_and_echo_line(port) is None:
                continue

    return 0


if __name__ == "__main__":
    sys.exit(main())
