#!/usr/bin/env python3
"""
Turn an image into a plain text file you can paste or send from a terminal
program (Tera Term, PuTTY, RealTerm, minicom).

The UART protocol is just lines of hex followed by STOP, so the whole payload
can live in a text file and be sent with Tera Term's File -> Send file. That
makes it possible to drive the board by hand, without any Python on the loop -
useful for demonstrating the protocol, and the only practical route if you are
testing over a real COM port with no host script.

    python scripts/make_hex_file.py samples/class69_height_limit.png out.txt
    python scripts/make_hex_file.py my_photo.jpg out.txt --size 64
    python scripts/make_hex_file.py my_photo.jpg out.txt -g

Then in Tera Term:  File -> Send file...  tick **Binary**, choose out.txt.

Lines are CRLF-terminated and end with STOP. The firmware skips whitespace,
CR and LF, so the exact line ending does not matter - but Binary mode does,
because text mode may re-translate the ending and stall on an EOF byte.
"""
import argparse, os, sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "scripts"))


def load_payload(path, size, grayscale):
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
    ap.add_argument("image")
    ap.add_argument("out")
    ap.add_argument("--size", type=int, default=48, choices=[48, 64])
    ap.add_argument("-g", "--grayscale", action="store_true")
    ap.add_argument("--bytes-per-line", type=int, default=32)
    a = ap.parse_args()

    p = load_payload(a.image, a.size, a.grayscale)
    n = a.bytes_per_line
    with open(a.out, "wb") as fh:
        for i in range(0, len(p), n):
            fh.write(p[i:i + n].hex().encode() + b"\r\n")
        fh.write(b"STOP\r\n")
    print(f"{len(p)} bytes -> {a.out} "
          f"({(len(p) + n - 1)//n} hex lines + STOP)")


if __name__ == "__main__":
    main()
