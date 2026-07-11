#!/usr/bin/env python3
"""Generate a deterministic RGB test image for the headless passthrough test.

Not a substitute for the real Kodak Marcie / ASWF footage (see test-assets/),
just a hermetic fixture so CI/local render tests don't depend on external assets.
"""
import sys

import numpy as np
from PIL import Image


def make(path: str, w: int = 336, h: int = 224) -> None:
    ys = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    xs = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    r = xs * np.ones_like(ys)
    g = ys * np.ones_like(xs)
    b = np.sqrt((xs - 0.5) ** 2 + (ys - 0.5) ** 2)
    # a few hard edges so structure is easy to correlate
    b[(h // 3):(h // 3 + 8), :] = 1.0
    b[:, (w // 2):(w // 2 + 8)] = 0.0
    rgb = np.clip(np.stack([r, g, b], axis=-1), 0, 1)
    Image.fromarray((rgb * 255).astype(np.uint8), "RGB").save(path)
    print(f"wrote {path} ({w}x{h})")


if __name__ == "__main__":
    make(sys.argv[1] if len(sys.argv) > 1 else "test-assets/synthetic_test.png")
