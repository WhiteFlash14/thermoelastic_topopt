"""2D shape primitives used to describe the design domain.

A domain is ``union(add shapes) - union(subtract shapes)``.  Every shape only
needs to answer "is this point inside?" (vectorized), plus give a bounding box.
Supported shape types (as written in the problem file):

    {type: rectangle, x: [x0, x1], y: [y0, y1]}
    {type: circle,    center: [cx, cy], radius: r}
    {type: polygon,   points: [[x, y], [x, y], ...]}      # any simple polygon
    {type: image,     file: shape.png, width: 120}         # dark pixels = material
                      (optional: height, origin: [x0, y0], threshold: 0.5, invert: false)

All coordinates are in the length unit declared at the top of the problem file.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class Shape:
    def contains(self, pts: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def bbox(self):  # pragma: no cover - interface
        raise NotImplementedError


class Rectangle(Shape):
    def __init__(self, x, y):
        self.x0, self.x1 = sorted(map(float, x))
        self.y0, self.y1 = sorted(map(float, y))

    def contains(self, p):
        return (p[:, 0] >= self.x0) & (p[:, 0] <= self.x1) & (p[:, 1] >= self.y0) & (p[:, 1] <= self.y1)

    def bbox(self):
        return self.x0, self.x1, self.y0, self.y1


class Circle(Shape):
    def __init__(self, center, radius):
        self.c = np.asarray(center, float)
        self.r = float(radius)

    def contains(self, p):
        return np.sum((p - self.c) ** 2, axis=1) <= self.r ** 2

    def bbox(self):
        return self.c[0] - self.r, self.c[0] + self.r, self.c[1] - self.r, self.c[1] + self.r


class Polygon(Shape):
    def __init__(self, points):
        self.p = np.asarray(points, float)
        if self.p.ndim != 2 or self.p.shape[1] != 2 or len(self.p) < 3:
            raise ValueError('polygon needs at least 3 [x, y] points')

    def contains(self, pts):
        # even-odd ray casting, vectorized over query points
        x, y = pts[:, 0], pts[:, 1]
        inside = np.zeros(len(pts), bool)
        px, py = self.p[:, 0], self.p[:, 1]
        j = len(self.p) - 1
        for i in range(len(self.p)):
            xi, yi, xj, yj = px[i], py[i], px[j], py[j]
            cond = (yi > y) != (yj > y)
            with np.errstate(divide='ignore', invalid='ignore'):
                xint = (xj - xi) * (y - yi) / (yj - yi) + xi
            inside ^= cond & (x < xint)
            j = i
        return inside

    def bbox(self):
        return self.p[:, 0].min(), self.p[:, 0].max(), self.p[:, 1].min(), self.p[:, 1].max()


class ImageMask(Shape):
    """Black-and-white (or grayscale) picture of the part; dark = material."""

    def __init__(self, file, width, height=None, origin=(0.0, 0.0), threshold=0.5,
                 invert=False, base_dir: Path | None = None):
        from PIL import Image  # Pillow ships with matplotlib/Anaconda
        path = Path(file)
        if not path.is_absolute() and base_dir is not None:
            path = Path(base_dir) / path
        img = np.asarray(Image.open(path).convert('L'), float) / 255.0
        self.mask = img < threshold
        if invert:
            self.mask = ~self.mask
        nrow, ncol = self.mask.shape
        self.w = float(width)
        self.h = float(height) if height is not None else self.w * nrow / ncol
        self.x0, self.y0 = map(float, origin)

    def contains(self, p):
        nrow, ncol = self.mask.shape
        u = (p[:, 0] - self.x0) / self.w
        v = (p[:, 1] - self.y0) / self.h
        ok = (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)
        col = np.clip((u * ncol).astype(int), 0, ncol - 1)
        row = np.clip(((1 - v) * nrow).astype(int), 0, nrow - 1)  # image row 0 = top
        out = np.zeros(len(p), bool)
        out[ok] = self.mask[row[ok], col[ok]]
        return out

    def bbox(self):
        return self.x0, self.x0 + self.w, self.y0, self.y0 + self.h


def make_shape(spec: dict, base_dir: Path | None = None) -> Shape:
    if not isinstance(spec, dict) or 'type' not in spec:
        raise ValueError(f'shape must be a mapping with a "type" key, got {spec!r}')
    t = spec['type'].lower()
    if t in ('rectangle', 'rect', 'box'):
        return Rectangle(spec['x'], spec['y'])
    if t == 'circle':
        return Circle(spec['center'], spec['radius'])
    if t == 'polygon':
        return Polygon(spec['points'])
    if t == 'image':
        return ImageMask(spec['file'], spec['width'], spec.get('height'), spec.get('origin', (0, 0)),
                         spec.get('threshold', 0.5), spec.get('invert', False), base_dir)
    raise ValueError(f'unknown shape type {t!r} (use rectangle, circle, polygon or image)')


class Domain:
    """union(add) minus union(subtract)."""

    def __init__(self, add, subtract=()):
        if not add:
            raise ValueError('geometry.add must contain at least one shape')
        self.add = list(add)
        self.subtract = list(subtract)

    def contains(self, p):
        inside = np.zeros(len(p), bool)
        for s in self.add:
            inside |= s.contains(p)
        for s in self.subtract:
            inside &= ~s.contains(p)
        return inside

    def bbox(self):
        b = np.array([s.bbox() for s in self.add])
        return b[:, 0].min(), b[:, 1].max(), b[:, 2].min(), b[:, 3].max()
