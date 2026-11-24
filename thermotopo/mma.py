"""Method of Moving Asymptotes (Svanberg 1987 / 2007), self-contained NumPy version.

This is a direct port of Krister Svanberg's classic ``mmasub``/``subsolv``
routines (the version used in most topology-optimization codes).  It solves

    min  f0(x) + a0*z + sum_i (c_i*y_i + 0.5*d_i*y_i^2)
    s.t. f_i(x) - a_i*z - y_i <= 0,   xmin <= x <= xmax,  y >= 0, z >= 0

With a0=1, a_i=0, large c_i and d_i=1 this is the standard nonlinear program
"min f0 s.t. f_i <= 0" (the y_i are artificial slack variables that make an
infeasible starting point, e.g. x = 1 everywhere, admissible).

One call of :meth:`MMA.update` = one optimization *iteration* in the sense of
the paper (Meng et al. 2021 count MMA iterations, max 200).
"""
from __future__ import annotations

import numpy as np


class MMA:
    def __init__(self, n: int, m: int, xmin, xmax, move: float = 0.2,
                 asyinit: float = 0.5, asyincr: float = 1.2, asydecr: float = 0.7,
                 c: float = 1000.0, d: float = 1.0):
        self.n, self.m = n, m
        self.xmin = np.broadcast_to(np.asarray(xmin, float), (n,)).copy()
        self.xmax = np.broadcast_to(np.asarray(xmax, float), (n,)).copy()
        self.move = move
        self.asyinit, self.asyincr, self.asydecr = asyinit, asyincr, asydecr
        self.a0 = 1.0
        self.a = np.zeros(m)
        self.c = np.full(m, c)
        self.d = np.full(m, d)
        self.iter = 0
        self.xold1 = None
        self.xold2 = None
        self.low = None
        self.upp = None

    def update(self, x, f0val, df0dx, fval, dfdx):
        """Return the next design. ``fval`` (m,), ``dfdx`` (m, n)."""
        x = np.asarray(x, float)
        self.iter += 1
        if self.xold1 is None:
            self.xold1 = x.copy()
            self.xold2 = x.copy()
        xnew, self.low, self.upp = _mmasub(
            self.m, self.n, self.iter, x, self.xmin, self.xmax, self.xold1, self.xold2,
            np.asarray(df0dx, float), np.asarray(fval, float).reshape(-1),
            np.asarray(dfdx, float).reshape(self.m, self.n),
            self.low, self.upp, self.a0, self.a, self.c, self.d, self.move,
            self.asyinit, self.asyincr, self.asydecr)
        self.xold2 = self.xold1
        self.xold1 = x.copy()
        return xnew


def _mmasub(m, n, it, xval, xmin, xmax, xold1, xold2, df0dx, fval, dfdx,
            low, upp, a0, a, c, d, move, asyinit, asyincr, asydecr):
    epsimin = 1e-7
    raa0 = 1e-5
    albefa = 0.1
    xrange = xmax - xmin
    # --- moving asymptotes ---------------------------------------------------
    if it <= 2 or low is None:
        low = xval - asyinit * xrange
        upp = xval + asyinit * xrange
    else:
        zzz = (xval - xold1) * (xold1 - xold2)
        factor = np.ones(n)
        factor[zzz > 0] = asyincr
        factor[zzz < 0] = asydecr
        low = xval - factor * (xold1 - low)
        upp = xval + factor * (upp - xold1)
        low = np.clip(low, xval - 10 * xrange, xval - 0.01 * xrange)
        upp = np.clip(upp, xval + 0.01 * xrange, xval + 10 * xrange)
    # --- move limits -----------------------------------------------------------
    alfa = np.maximum.reduce([low + albefa * (xval - low), xval - move * xrange, xmin])
    beta = np.minimum.reduce([upp - albefa * (upp - xval), xval + move * xrange, xmax])
    # --- approximating functions ----------------------------------------------
    xmamiinv = 1.0 / np.maximum(xrange, 1e-5)
    ux1 = upp - xval
    xl1 = xval - low
    ux2 = ux1 * ux1
    xl2 = xl1 * xl1
    p0 = np.maximum(df0dx, 0.0)
    q0 = np.maximum(-df0dx, 0.0)
    pq0 = 0.001 * (p0 + q0) + raa0 * xmamiinv
    p0 = (p0 + pq0) * ux2
    q0 = (q0 + pq0) * xl2
    P = np.maximum(dfdx, 0.0)
    Q = np.maximum(-dfdx, 0.0)
    PQ = 0.001 * (P + Q) + raa0 * xmamiinv[None, :]
    P = (P + PQ) * ux2[None, :]
    Q = (Q + PQ) * xl2[None, :]
    b = P @ (1.0 / ux1) + Q @ (1.0 / xl1) - fval
    x = _subsolv(m, n, epsimin, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d)
    return x, low, upp


def _subsolv(m, n, epsimin, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d):
    """Primal-dual Newton interior-point solver of the MMA subproblem."""
    een = np.ones(n)
    eem = np.ones(m)
    epsi = 1.0
    x = 0.5 * (alfa + beta)
    y = eem.copy()
    z = 1.0
    lam = eem.copy()
    xsi = np.maximum(een / (x - alfa), een)
    eta = np.maximum(een / (beta - x), een)
    mu = np.maximum(eem, 0.5 * c)
    zet = 1.0
    s = eem.copy()

    def residual(x, y, z, lam, xsi, eta, mu, zet, s, epsi):
        ux1 = upp - x
        xl1 = x - low
        plam = p0 + P.T @ lam
        qlam = q0 + Q.T @ lam
        gvec = P @ (1.0 / ux1) + Q @ (1.0 / xl1)
        dpsidx = plam / ux1**2 - qlam / xl1**2
        rex = dpsidx - xsi + eta
        rey = c + d * y - mu - lam
        rez = a0 - zet - a @ lam
        relam = gvec - a * z - y + s - b
        rexsi = xsi * (x - alfa) - epsi
        reeta = eta * (beta - x) - epsi
        remu = mu * y - epsi
        rezet = zet * z - epsi
        res = lam * s - epsi
        r = np.concatenate([rex, rey, [rez], relam, rexsi, reeta, remu, [rezet], res])
        return r

    while epsi > epsimin:
        r = residual(x, y, z, lam, xsi, eta, mu, zet, s, epsi)
        residunorm = np.linalg.norm(r)
        residumax = np.max(np.abs(r))
        ittt = 0
        while residumax > 0.9 * epsi and ittt < 200:
            ittt += 1
            ux1 = upp - x
            xl1 = x - low
            ux2 = ux1 * ux1
            xl2 = xl1 * xl1
            ux3 = ux1 * ux2
            xl3 = xl1 * xl2
            uxinv1 = 1.0 / ux1
            xlinv1 = 1.0 / xl1
            uxinv2 = 1.0 / ux2
            xlinv2 = 1.0 / xl2
            plam = p0 + P.T @ lam
            qlam = q0 + Q.T @ lam
            gvec = P @ uxinv1 + Q @ xlinv1
            GG = P * uxinv2[None, :] - Q * xlinv2[None, :]
            dpsidx = plam / ux2 - qlam / xl2
            delx = dpsidx - epsi / (x - alfa) + epsi / (beta - x)
            dely = c + d * y - lam - epsi / y
            delz = a0 - a @ lam - epsi / z
            dellam = gvec - a * z - y - b + epsi / lam
            diagx = 2.0 * (plam / ux3 + qlam / xl3) + xsi / (x - alfa) + eta / (beta - x)
            diagxinv = 1.0 / diagx
            diagy = d + mu / y
            diagyinv = 1.0 / diagy
            diaglam = s / lam
            diaglamyi = diaglam + diagyinv
            if m < n:
                blam = dellam + dely / diagy - GG @ (delx * diagxinv)
                bb = np.concatenate([blam, [delz]])
                Alam = np.diag(diaglamyi) + (GG * diagxinv[None, :]) @ GG.T
                AA = np.zeros((m + 1, m + 1))
                AA[:m, :m] = Alam
                AA[:m, m] = a
                AA[m, :m] = a
                AA[m, m] = -zet / z
                sol = np.linalg.solve(AA, bb)
                dlam = sol[:m]
                dz = sol[m]
                dx = -delx * diagxinv - (GG.T @ dlam) * diagxinv
            else:
                diaglamyiinv = 1.0 / diaglamyi
                dellamyi = dellam + dely / diagy
                Axx = np.diag(diagx) + (GG.T * diaglamyiinv[None, :]) @ GG
                azz = zet / z + a @ (a * diaglamyiinv)
                axz = -GG.T @ (a * diaglamyiinv)
                bx = delx + GG.T @ (dellamyi * diaglamyiinv)
                bz = delz - a @ (dellamyi * diaglamyiinv)
                AA = np.zeros((n + 1, n + 1))
                AA[:n, :n] = Axx
                AA[:n, n] = axz
                AA[n, :n] = axz
                AA[n, n] = azz
                sol = np.linalg.solve(AA, -np.concatenate([bx, [bz]]))
                dx = sol[:n]
                dz = sol[n]
                dlam = (GG @ dx) * diaglamyiinv - dz * (a * diaglamyiinv) + dellamyi * diaglamyiinv
            dy = -dely / diagy + dlam / diagy
            dxsi = -xsi + epsi / (x - alfa) - (xsi * dx) / (x - alfa)
            deta = -eta + epsi / (beta - x) + (eta * dx) / (beta - x)
            dmu = -mu + epsi / y - (mu * dy) / y
            dzet = -zet + epsi / z - zet * dz / z
            ds = -s + epsi / lam - (s * dlam) / lam
            xx = np.concatenate([y, [z], lam, xsi, eta, mu, [zet], s])
            dxx = np.concatenate([dy, [dz], dlam, dxsi, deta, dmu, [dzet], ds])
            stmxx = np.max(-1.01 * dxx / xx)
            stmalfa = np.max(-1.01 * dx / (x - alfa))
            stmbeta = np.max(1.01 * dx / (beta - x))
            steg = 1.0 / max(stmalfa, stmbeta, stmxx, 1.0)
            old = (x, y, z, lam, xsi, eta, mu, zet, s)
            itto = 0
            resinew = 2.0 * residunorm
            while resinew > residunorm and itto < 50:
                itto += 1
                x = old[0] + steg * dx
                y = old[1] + steg * dy
                z = old[2] + steg * dz
                lam = old[3] + steg * dlam
                xsi = old[4] + steg * dxsi
                eta = old[5] + steg * deta
                mu = old[6] + steg * dmu
                zet = old[7] + steg * dzet
                s = old[8] + steg * ds
                r = residual(x, y, z, lam, xsi, eta, mu, zet, s, epsi)
                resinew = np.linalg.norm(r)
                steg = steg / 2.0
            residunorm = resinew
            residumax = np.max(np.abs(r))
        epsi = 0.1 * epsi
    return x
