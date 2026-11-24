"""Finite-element building blocks: bilinear Q4 elements, plane stress, 2x2 Gauss.

Because the mesh is made of identical square elements, every element matrix is
the *solid* matrix times an interpolation factor:

    K_m,e   = E(rho_e)/E0 * ke0            (Meng Eq. 8)
    K_T,e   = lambda(rho_e)/lambda0 * kt0  (Meng Eq. 3)
    F_th,e  = beta(rho_e)/(E0 alpha0) * g0 @ theta_e     (Meng Eq. 14)

with theta = T - T_ref interpolated bilinearly inside the element.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

GAUSS = (-1 / np.sqrt(3), 1 / np.sqrt(3))
PHI = np.array([1.0, 1.0, 0.0])          # thermal strain direction (Meng Eq. 11)


def plane_stress_D(E, nu):
    return E / (1 - nu ** 2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])


def _q4(xi, eta, h):
    N = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta), (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
    dN = 0.25 * np.array([[-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],
                          [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]])
    grad = dN * (2.0 / h)              # square element: J = h/2 * I
    B = np.zeros((3, 8))
    B[0, 0::2] = grad[0]
    B[1, 1::2] = grad[1]
    B[2, 0::2] = grad[1]
    B[2, 1::2] = grad[0]
    detJ = (h / 2) ** 2
    return N, grad, B, detJ


def element_matrices(h, t, E0, nu, k0, alpha0):
    """Solid element matrices for a square element of side h (m), thickness t (m)."""
    D = plane_stress_D(E0, nu)
    ke = np.zeros((8, 8))
    kt = np.zeros((4, 4))
    g = np.zeros((8, 4))
    for xi in GAUSS:
        for eta in GAUSS:
            N, grad, B, dA = _q4(xi, eta, h)
            w = dA * t
            ke += w * B.T @ D @ B
            kt += w * k0 * grad.T @ grad
            g += w * np.outer(B.T @ D @ (alpha0 * PHI), N)
    _, _, Bc, _ = _q4(0.0, 0.0, h)
    return dict(ke0=ke, kt0=kt, g0=g, D0=D, Bc=Bc)


class ReducedSystem:
    """Assemble element-scaled matrices directly into the free-DOF system."""

    def __init__(self, cells_dof: np.ndarray, ndof: int, fixed: np.ndarray):
        self.ndof = ndof
        nloc = cells_dof.shape[1]
        self.edof = cells_dof
        free = np.ones(ndof, bool)
        free[fixed] = False
        self.free = np.flatnonzero(free)
        self.fixed = np.flatnonzero(~free)
        red = -np.ones(ndof, int)
        red[self.free] = np.arange(len(self.free))
        r = np.repeat(cells_dof, nloc, axis=1).ravel()      # row index of each (e, a, b)
        c = np.tile(cells_dof, (1, nloc)).ravel()           # col index
        self.keep = (red[r] >= 0) & (red[c] >= 0)
        self.rows = red[r[self.keep]]
        self.cols = red[c[self.keep]]
        self.nfree = len(self.free)
        self.rows_full = r
        self.cols_full = c

    def matrix(self, scale: np.ndarray, kloc: np.ndarray):
        vals = (scale[:, None] * kloc.ravel()[None, :]).ravel()
        A = sparse.csc_matrix((vals[self.keep], (self.rows, self.cols)), shape=(self.nfree, self.nfree))
        return A

    def full_matrix(self, scale, kloc):
        vals = (scale[:, None] * kloc.ravel()[None, :]).ravel()
        return sparse.csr_matrix((vals, (self.rows_full, self.cols_full)), shape=(self.ndof, self.ndof))

    @staticmethod
    def factor(A):
        return splu(A, permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.0,
                    options=dict(SymmetricMode=True))


def scatter_add(n, idx, vals):
    out = np.zeros(n)
    np.add.at(out, idx.ravel(), vals.ravel())
    return out
