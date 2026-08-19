# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: Apache-2.0
"""Render docs/demo.gif from the REAL output of examples/demo_timeline.py.

Runs the demo, captures stdout, and animates it as a terminal typing session.
Nothing is staged or edited: what the GIF shows is what the script printed.

Run: python tools/render_demo_gif.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "demo.gif"

BG = (11, 15, 20)
FG = (222, 228, 236)
DIM = (126, 138, 156)
ACCENT = (10, 132, 255)
GREEN = (86, 214, 148)
FONT_SIZE = 15
LINE_H = 22
PAD = 18
COLS = 95


def _font() -> ImageFont.FreeTypeFont:
    for name in ("consola.ttf", "CascadiaMono.ttf", "DejaVuSansMono.ttf"):
        try:
            return ImageFont.truetype(name, FONT_SIZE)
        except OSError:
            continue
    return ImageFont.load_default()


def _color(line: str) -> tuple[int, int, int]:
    s = line.strip()
    if s.startswith("[") and "]" in s[:4]:
        return ACCENT
    if s.startswith("A:") or "ms per write" in s or "succeeded" in s:
        return GREEN
    if s.startswith("stored") or s.startswith("Q:") or s.startswith("0."):
        return DIM
    return FG


def main() -> int:
    print("running the demo...", flush=True)
    proc = subprocess.run([sys.executable, str(ROOT / "examples" / "demo_timeline.py")],
                          capture_output=True, text=True, cwd=ROOT)
    if proc.returncode != 0:
        print(proc.stdout[-2000:], proc.stderr[-2000:])
        return proc.returncode
    lines = [ln.rstrip() for ln in proc.stdout.splitlines()]
    print(f"captured {len(lines)} lines, rendering...", flush=True)

    font = _font()
    width = PAD * 2 + int(font.getlength("M") * COLS)
    height = PAD * 2 + LINE_H * (len(lines) + 1)

    frames, durations = [], []
    for i in range(len(lines) + 1):
        img = Image.new("RGB", (width, height), BG)
        d = ImageDraw.Draw(img)
        for j, line in enumerate(lines[:i]):
            d.text((PAD, PAD + j * LINE_H), line[:COLS], font=font, fill=_color(line))
        if i < len(lines):  # cursor on the line being typed
            d.rectangle([PAD, PAD + i * LINE_H + 3,
                         PAD + 8, PAD + i * LINE_H + FONT_SIZE + 3], fill=ACCENT)
        frames.append(img)
        blank = not lines[i - 1].strip() if i else False
        durations.append(700 if blank else 260)
    durations[-1] = 5000  # hold the final frame

    OUT.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(OUT, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True)
    kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT} ({width}x{height}, {len(frames)} frames, {kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
