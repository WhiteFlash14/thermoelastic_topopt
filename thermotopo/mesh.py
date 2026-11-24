"""Structured square-element (Q4) mesh of an arbitrary 2D domain + node selectors.

The bounding box of the domain is covered by a regular grid of square elements
of size ``h``; an element is kept when its centre lies inside the domain
(union/subtraction of shapes, or an image).  This is the standard "voxel"
approach of density-based topology optimization: any 2D shape can be
described, curved edges are approximated by a staircase of elements.

Internally everything is SI (metres).  Coordinates given in the problem file
(e.g. millimetres) are converted with ``mesh.scale``.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

from .geometry import Domain


class GridMesh:
    def __init__(self, domain: Domain, h: float, scale: float = 1.0,
                 passive_solid: Domain | None = None, passive_void: Domain | None = None):
        if h <= 0:
            raise ValueError('element_size must be > 0')
        self.domain = domain
        self.h_file = float(h)              # element size in file units
        self.scale = float(scale)           # file unit -> metre
        self.h = self.h_file * self.scale   # element size in metres
        x0, x1, y0, y1 = domain.bbox()
        self.x0, self.y0 = x0, y0
        self.nx = max(1, int(math.ceil((x1 - x0) / h - 1e-9)))
        self.ny = max(1, int(math.ceil((y1 - y0) / h - 1e-9)))
        ii, jj = np.meshgrid(np.arange(self.nx), np.arange(self.ny))   # jj = row (y), ii = col (x)
        ii, jj = ii.ravel(), jj.ravel()
        cen = np.column_stack([x0 + (ii + 0.5) * h, y0 + (jj + 0.5) * h])
        keep = domain.contains(cen)
        if not keep.any():
            raise ValueError('no element centre lies inside the geometry - check shapes / element_size')
        self.elem_i, self.elem_j = ii[keep], jj[keep]
        self.centers_file = cen[keep]
        # node numbering on the full grid, then compress to used nodes
        nxp = self.nx + 1
        n00 = self.elem_j * nxp + self.elem_i
        cells_full = np.column_stack([n00, n00 + 1, n00 + nxp + 1, n00 + nxp])   # counter-clockwise
        used = np.unique(cells_full)
        remap = -np.ones((self.nx + 1) * (self.ny + 1), int)
        remap[used] = np.arange(len(used))
        self.cells = remap[cells_full]
        gi, gj = used % nxp, used // nxp
        self.nodes_file = np.column_stack([x0 + gi * h, y0 + gj * h])
        self.nodes = self.nodes_file * self.scale
        self.centers = self.centers_file * self.scale
        self.ne = len(self.cells)
        self.nn = len(self.nodes)
        self.passive_solid = passive_solid.contains(self.centers_file) if passive_solid else np.zeros(self.ne, bool)
        self.passive_void = passive_void.contains(self.centers_file) if passive_void else np.zeros(self.ne, bool)
        self.passive_void &= ~self.passive_solid
        self._node_tree = cKDTree(self.nodes_file)
        self._grid_elem = -np.ones((self.ny, self.nx), int)
        self._grid_elem[self.elem_j, self.elem_i] = np.arange(self.ne)

    # ------------------------------------------------------------------ helpers
    @property
    def bbox_file(self):
        n = self.nodes_file
        return n[:, 0].min(), n[:, 0].max(), n[:, 1].min(), n[:, 1].max()

    def to_grid(self, values, fill=np.nan):
        """Element values -> (ny, nx) image (row 0 = bottom) for plotting."""
        a = np.full((self.ny, self.nx), fill, float)
        a[self.elem_j, self.elem_i] = values
        return a

    def element_at(self, p_file):
        """Index of the element containing point p (file units) or -1."""
        i = int(math.floor((p_file[0] - self.x0) / self.h_file))
        j = int(math.floor((p_file[1] - self.y0) / self.h_file))
        if 0 <= i < self.nx and 0 <= j < self.ny:
            return int(self._grid_elem[j, i])
        return -1

    # ------------------------------------------------------------- selectors
    def select(self, where: dict, interpolate_points: bool = False):
        """Return (node_ids, weights).

        ``where`` is one of (coordinates in file units):
            {point: [x, y]}                      nearest node (or exact point if interpolate_points)
            {points: [[x, y], ...]}              nearest node of each point
            {line: [[x1, y1], [x2, y2]], tol: t} nodes within t of the segment (default t = h/4)
            {box: [[xmin, ymin], [xmax, ymax]]}  nodes inside the rectangle
            {circle: {center: [x, y], radius: r}}
            {edge: top|bottom|left|right}        nodes on that side of the bounding box
        Weights sum to 1 (equal split, or bilinear shape functions for an exact point).
        """
        if not isinstance(where, dict) or len(set(where) - {'tol'}) != 1:
            raise ValueError(f'"where" must have exactly one selector key, got {where!r}')
        h = self.h_file
        tol = float(where.get('tol', 0.25 * h))
        P = self.nodes_file
        if 'point' in where:
            p = np.asarray(where['point'], float)
            if interpolate_points:
                e = self.element_at(p)
                if e >= 0:
                    xy = self.nodes_file[self.cells[e]]
                    rx = (p[0] - xy[0, 0]) / h
                    ry = (p[1] - xy[0, 1]) / h
                    w = np.array([(1 - rx) * (1 - ry), rx * (1 - ry), rx * ry, (1 - rx) * ry])
                    ok = w > 1e-10
                    w = w[ok] / w[ok].sum()
                    return self.cells[e][ok], w
            ids = np.array([self._node_tree.query(p)[1]])
        elif 'points' in where:
            ids = np.unique(self._node_tree.query(np.asarray(where['points'], float))[1])
        elif 'line' in where:
            a, b = (np.asarray(v, float) for v in where['line'])
            ab = b - a
            L2 = float(ab @ ab)
            t = np.clip(((P - a) @ ab) / L2, 0, 1) if L2 > 0 else np.zeros(len(P))
            d = np.linalg.norm(P - (a + t[:, None] * ab), axis=1)
            ids = np.flatnonzero(d <= tol)
        elif 'box' in where:
            (xa, ya), (xb, yb) = where['box']
            xa, xb = sorted([xa, xb]); ya, yb = sorted([ya, yb])
            ids = np.flatnonzero((P[:, 0] >= xa - 1e-9 * h) & (P[:, 0] <= xb + 1e-9 * h) &
                                 (P[:, 1] >= ya - 1e-9 * h) & (P[:, 1] <= yb + 1e-9 * h))
        elif 'circle' in where:
            c = np.asarray(where['circle']['center'], float)
            ids = np.flatnonzero(np.linalg.norm(P - c, axis=1) <= float(where['circle']['radius']) + 1e-9 * h)
        elif 'edge' in where:
            x0, x1, y0, y1 = self.bbox_file
            side = where['edge'].lower()
            col, val = {'left': (0, x0), 'right': (0, x1), 'bottom': (1, y0), 'top': (1, y1)}[side]
            ids = np.flatnonzero(np.abs(P[:, col] - val) <= tol)
        else:
            raise ValueError(f'unknown selector {where!r}')
        if len(ids) == 0:
            raise ValueError(f'selector {where!r} did not match any mesh node')
        return ids, np.full(len(ids), 1.0 / len(ids))

    # ------------------------------------------------------------- symmetry
    def mirror_map(self, axis: str):
        """Element index of the mirror image (about the bbox centre line)."""
        x0, x1, y0, y1 = self.bbox_file
        c = self.centers_file.copy()
        if axis == 'vertical':      # mirror x -> (x0 + x1) - x
            c[:, 0] = x0 + x1 - c[:, 0]
        elif axis == 'horizontal':
            c[:, 1] = y0 + y1 - c[:, 1]
        else:
            raise ValueError("symmetry axis must be 'vertical' or 'horizontal'")
        d, idx = cKDTree(self.centers_file).query(c)
        if np.max(d) > 1e-6 * self.h_file:
            raise ValueError(f'mesh is not symmetric about the {axis} centre line; remove the symmetry option')
        return idx
