"""Turn a density panel from the paper into a (ny, nx) array of 0/1 values.

    python paper_reference/digitize.py image.png x0 y0 x1 y1 nx ny out.npy

(x0, y0, x1, y1) = pixel crop that contains the panel *and its frame* but not
the colour bar.  The frame (outer non-white bounding box) is detected
automatically and each element is sampled at its centre.  Row 0 of the
output is the BOTTOM of the domain (same convention as mesh.to_grid).
"""
import sys

import numpy as np
from PIL import Image


def digitize(path, crop, nx, ny):
    img = np.asarray(Image.open(path).convert('L').crop(crop), float) / 255.0
    dark = img < 0.92
    rows = np.flatnonzero(dark.mean(axis=1) > 0.05)
    cols = np.flatnonzero(dark.mean(axis=0) > 0.05)
    r0, r1, c0, c1 = rows[0], rows[-1], cols[0], cols[-1]
    inset = 1.5                                  # frame line thickness (px)
    ys = r0 + inset + (np.arange(ny) + 0.5) * (r1 - r0 - 2 * inset) / ny
    xs = c0 + inset + (np.arange(nx) + 0.5) * (c1 - c0 - 2 * inset) / nx
    val = img[np.round(ys).astype(int)][:, np.round(xs).astype(int)]
    return (1.0 - val)[::-1]                     # dark = 1 = solid, flip so row 0 = bottom


if __name__ == '__main__':
    p, x0, y0, x1, y1, nx, ny, out = sys.argv[1:]
    a = digitize(p, tuple(map(int, (x0, y0, x1, y1))), int(nx), int(ny))
    np.save(out, (a > 0.5).astype(np.uint8))
    print(out, a.shape, 'solid fraction', (a > 0.5).mean())
